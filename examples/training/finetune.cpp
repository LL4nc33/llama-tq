#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "lora-training.h"

#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <csignal>
#include <fstream>

#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#  define _exit ::_exit
#else
#  include <unistd.h>
#endif

// Global hook so SIGINT/SIGTERM (e.g. timeout(1) sending SIGTERM at the
// hard limit) can still flush the trained LoRA weights to disk before the
// process dies. Without this, multi-hour runs lose all progress when the
// shell-level timeout fires before the final llama_adapter_lora_save call.
static llama_adapter_lora * g_lora_adapter_for_signal = nullptr;
static std::string          g_adapter_out_for_signal;

static void finetune_write_current_state();

// SIGINT/SIGTERM ask the training loop to stop: the callbacks save the adapter and the position after the current
// optimizer step completes, so the saved state is consistent (a save in the middle of a step would mix updated and
// not yet updated tensors). A second signal exits at once.
static volatile sig_atomic_t g_stop_requested = 0;

static void finetune_on_signal(int signum) {
    if (g_stop_requested || !g_lora_adapter_for_signal || g_adapter_out_for_signal.empty()) {
        _exit(128 + signum);
    }
    g_stop_requested = signum;
}

// Regex-based parameter filter for selective fine-tuning (e.g. skip Mamba/SSM layers).
// When the tensor name matches the regex, it is EXCLUDED from training.
static bool finetune_param_filter_skip_regex(const struct ggml_tensor * t, void * ud) {
    auto * re = static_cast<std::regex *>(ud);
    return !std::regex_search(t->name, *re);
}

// LoRA training: every model tensor stays frozen, only the adapter's A and B (marked trainable when the adapter is
// created) are trained. Otherwise F32 model tensors such as the norms would change too, but they are not part of the
// saved adapter, so inference with --lora (and --resume) would not see those changes.
static bool finetune_param_filter_none(const struct ggml_tensor * t, void * ud) {
    GGML_UNUSED(t);
    GGML_UNUSED(ud);
    return false;
}

// C.4: Periodic mid-training checkpoint state. Tracks how many training
// batches have been seen since process start and the target path / adapter
// to flush. Hooked into ggml_opt_epoch via a wrapper callback that calls
// the progress bar AND triggers a save every N batches.
struct finetune_checkpoint_state {
    llama_adapter_lora * adapter         = nullptr;
    llama_context      * ctx             = nullptr;
    const lr_opt       * lr              = nullptr;
    std::string          path;
    int                  every_n         = 0;   // 0 = disabled
    int64_t              batches_seen    = 0;   // monotonically increasing
    int64_t              last_save_batch = 0;   // batches_seen at most recent save

    // training position for --resume, written next to the adapter as <adapter>.state
    int                  epoch                = 0;
    int64_t              datapoint_offset     = 0; // datapoints skipped at the start of this epoch (resume)
    int64_t              ubatches_done        = 0; // training ubatches done in this epoch
    int64_t              ubatches_per_datapoint = 1;
    int64_t              stop_after           = 0; // stop (and save) after this many training windows in this run
    int64_t              windows_done         = 0;
};
static finetune_checkpoint_state g_checkpoint_state;

static void finetune_write_state(int epoch, int64_t datapoints_done) {
    const auto & cs = g_checkpoint_state;
    if (cs.path.empty()) {
        return;
    }
    // optimizer state (AdamW moments) as <adapter>.opt, position and learning-rate step as <adapter>.state
    if (llama_opt_save_state(cs.ctx, (cs.path + ".opt").c_str()) != 0) {
        fprintf(stderr, "\n%s: failed to write the optimizer state '%s.opt'\n", __func__, cs.path.c_str());
    }
    if (FILE * f = fopen((cs.path + ".state").c_str(), "w")) {
        fprintf(f, "{\"epoch\": %d, \"datapoints_done\": %" PRId64 ", \"lr_step\": %" PRId64 "}\n",
                epoch, datapoints_done, cs.lr->step);
        fclose(f);
    }
}

static void finetune_write_current_state() {
    const auto & cs = g_checkpoint_state;
    finetune_write_state(cs.epoch, cs.datapoint_offset + cs.ubatches_done / cs.ubatches_per_datapoint);
}

static void finetune_epoch_callback_with_checkpoint(
        bool                train,
        ggml_opt_context_t  opt_ctx,
        ggml_opt_dataset_t  dataset,
        ggml_opt_result_t   result,
        int64_t             ibatch,
        int64_t             ibatch_max,
        int64_t             t_start_us) {
    // Always run the progress-bar visualisation first so user feedback stays
    // identical to the upstream default.
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result,
                                         ibatch, ibatch_max, t_start_us);

    auto & cs = g_checkpoint_state;
    // a window (one datapoint of n_ctx tokens) is complete after every ubatches_per_datapoint ubatches;
    // stopping is only exact there, as the next ubatches of a started window were already applied
    const bool window_complete = !train || ibatch % cs.ubatches_per_datapoint == 0;
    if (train && window_complete && cs.stop_after > 0 && ++cs.windows_done >= cs.stop_after) {
        g_stop_requested = -1; // --stop-after reached: stop like on a signal
    }
    if (g_stop_requested && window_complete) {
        // stopped in the training pass: resume at the next datapoint; in the evaluation pass: at the next epoch
        if (train) {
            cs.ubatches_done = ibatch;
        }
        if (llama_adapter_lora_save_to_file(cs.adapter, cs.path.c_str()) == 0) {
            if (train) {
                finetune_write_current_state();
            } else {
                finetune_write_state(cs.epoch + 1, 0);
            }
            if (g_stop_requested > 0) {
                fprintf(stderr, "\nstopped by signal %d, saved '%s' (continue with --resume)\n", (int) g_stop_requested, cs.path.c_str());
            } else {
                fprintf(stderr, "\nstopped after %" PRId64 " windows (--stop-after), saved '%s' (continue with --resume)\n", cs.windows_done, cs.path.c_str());
            }
        }
        _exit(g_stop_requested > 0 ? 128 + g_stop_requested : 0);
    }

    // Only count training batches; eval-pass callbacks don't modify weights.
    if (!train) {
        return;
    }
    cs.ubatches_done = ibatch;
    if (cs.every_n <= 0 || !cs.adapter || cs.path.empty()) {
        return;
    }
    cs.batches_seen++;
    if (!window_complete || cs.batches_seen - cs.last_save_batch < cs.every_n) {
        return;
    }
    cs.last_save_batch = cs.batches_seen;
    if (llama_adapter_lora_save_to_file(cs.adapter, cs.path.c_str()) != 0) {
        fprintf(stderr, "\n%s: checkpoint flush at batch %" PRId64 " FAILED to '%s'\n",
                __func__, cs.batches_seen, cs.path.c_str());
    } else {
        finetune_write_current_state();
        fprintf(stderr, "\n%s: checkpoint at batch %" PRId64 " → '%s'\n",
                __func__, cs.batches_seen, cs.path.c_str());
    }
}

// Training data as JSONL, one example per line:
//   {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}  chat; only the assistant turns are trained
//   {"text": "..."}                                                               plain text; every token is trained
// A chat is rendered with the model's chat template. Each assistant turn is the difference between the conversation
// up to it with the generation prompt and the conversation including it (compared as tokens of whole strings), so the
// trained tokens are what the model generates at inference, end-of-turn token included. Some templates render the last
// assistant turn differently (Qwen3: with an empty <think> block); an earlier turn then ends at its first
// end-of-generation token in the whole conversation. Examples where neither works are skipped.
static bool finetune_load_jsonl(llama_context * ctx, const std::string & path, const std::string & chat_template,
        std::vector<llama_token> & tokens, std::vector<uint8_t> & train) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, chat_template);

    std::ifstream file(path);
    if (!file) {
        LOG_ERR("%s: cannot open '%s'\n", __func__, path.c_str());
        return false;
    }

    // whole strings are tokenized, as at inference, so the boundaries match what the model sees and generates
    auto tokenize = [&](const std::string & text) {
        std::vector<llama_token> t = common_tokenize(ctx, text, false, true);
        if (llama_vocab_get_add_bos(vocab) && (t.empty() || t[0] != llama_vocab_bos(vocab))) {
            t.insert(t.begin(), llama_vocab_bos(vocab));
        }
        return t;
    };

    int64_t n_chat = 0, n_text = 0, n_skipped = 0, n_lines = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++n_lines;
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        nlohmann::ordered_json j;
        try {
            j = nlohmann::ordered_json::parse(line);
        } catch (const std::exception & e) {
            LOG_ERR("%s: line %" PRId64 ": invalid JSON: %s\n", __func__, n_lines, e.what());
            return false;
        }
        if (j.contains("text")) {
            const std::vector<llama_token> t = tokenize(j.at("text").get<std::string>());
            tokens.insert(tokens.end(), t.begin(), t.end());
            train.insert(train.end(), t.size(), 1);
            const llama_token eos = llama_vocab_eos(vocab);
            if (eos != LLAMA_TOKEN_NULL) {
                tokens.push_back(eos);
                train.push_back(1);
            }
            ++n_text;
            continue;
        }
        if (!j.contains("messages")) {
            LOG_ERR("%s: line %" PRId64 ": expected \"messages\" or \"text\"\n", __func__, n_lines);
            return false;
        }

        std::vector<common_chat_msg> msgs;
        for (const auto & m : j.at("messages")) {
            common_chat_msg msg;
            msg.role    = m.at("role").get<std::string>();
            msg.content = m.at("content").get<std::string>();
            msgs.push_back(std::move(msg));
        }
        auto render = [&](size_t n, bool add_generation_prompt) {
            common_chat_templates_inputs inputs;
            inputs.messages.assign(msgs.begin(), msgs.begin() + n);
            inputs.add_generation_prompt = add_generation_prompt;
            inputs.use_jinja             = true;
            return common_chat_templates_apply(tmpls.get(), inputs).prompt;
        };

        // each assistant turn i is placed in the tokenization of the whole conversation: it starts where the prompt
        // render(i, generation prompt) stops agreeing with it and ends at its first end-of-generation token, or else
        // where render(i + 1) ends, if that is a prefix of the conversation
        const std::vector<llama_token> conv = tokenize(render(msgs.size(), false));
        std::vector<uint8_t> conv_train(conv.size(), 0);
        bool ok      = true;
        bool trained = false;
        for (size_t i = 0; i < msgs.size() && ok; ++i) {
            if (msgs[i].role != "assistant") {
                continue;
            }
            const std::vector<llama_token> prompt = tokenize(render(i, true));
            const std::vector<llama_token> full   = tokenize(render(i + 1, false));
            size_t start = 0;
            while (start < prompt.size() && start < conv.size() && prompt[start] == conv[start]) {
                ++start;
            }
            // what follows the end-of-generation token (e.g. a newline) is never generated, so it is not trained
            size_t end = full.size() <= conv.size() && std::equal(full.begin(), full.end(), conv.begin()) ? full.size() : 0;
            for (size_t k = start; k < conv.size() && (end == 0 || k < end); ++k) {
                if (llama_vocab_is_eog(vocab, conv[k])) {
                    end = k + 1;
                    break;
                }
            }
            if (end <= start) {
                ok = false;
                break;
            }
            std::fill(conv_train.begin() + start, conv_train.begin() + end, 1);
            trained = true;
        }
        if (!ok || !trained) {
            ++n_skipped;
            continue;
        }
        tokens.insert(tokens.end(), conv.begin(), conv.end());
        train.insert(train.end(), conv_train.begin(), conv_train.end());
        ++n_chat;
    }

    if (getenv("LLAMA_FINETUNE_SHOW_MASK")) {
        // the start of the data with the trained spans in [[ ]], to check the mask against the chat template
        std::string shown;
        bool in_train = false;
        for (size_t i = 0; i < tokens.size() && i < 400; ++i) {
            if (train[i] != in_train) {
                shown += train[i] ? "[[" : "]]";
                in_train = train[i];
            }
            shown += common_token_to_piece(ctx, tokens[i]);
        }
        LOG_INF("%s: data start:\n%s%s\n", __func__, shown.c_str(), in_train ? "]]" : "");
    }

    int64_t n_train = 0;
    for (uint8_t t : train) {
        n_train += t;
    }
    LOG_INF("%s: %" PRId64 " chats, %" PRId64 " texts, %" PRId64 " skipped (assistant turns not found in the chat); "
            "%zu tokens, %" PRId64 " trained (%.1f %%)\n", __func__, n_chat, n_text, n_skipped,
            tokens.size(), n_train, tokens.empty() ? 0.0 : 100.0*n_train/tokens.size());
    if (n_skipped > 0 && n_chat == 0) {
        LOG_ERR("%s: no chat example could be rendered with this chat template\n", __func__);
        return false;
    }
    return !tokens.empty();
}

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n",
                __func__);
        params.use_mmap = false;
    }
    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        // flash attention has no backward pass: with it, attention (and everything below it) would get no gradient
        LOG_INF("%s: force disabling flash attention because it has no backward pass\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    // load the model and apply lora adapter, if any
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::vector<llama_token> tokens;
    std::vector<uint8_t>     train; // empty: every token is trained
    const std::string & data_path = params.prompt_file;
    if (data_path.size() >= 6 && data_path.compare(data_path.size() - 6, 6, ".jsonl") == 0) {
        if (!finetune_load_jsonl(ctx, data_path, params.chat_template, tokens, train)) {
            return 1;
        }
    } else {
        tokens = common_tokenize(ctx, params.prompt, true);
    }
    if ((int64_t) tokens.size() <= (int64_t) llama_n_ctx(ctx) + 1) {
        LOG_ERR("%s: %zu tokens of training data, need more than the context size %u\n", __func__, tokens.size(), llama_n_ctx(ctx));
        return 1;
    }
    const int64_t stride = llama_n_ctx(ctx) / 2;
    // the dataset starting skip datapoints later (to resume within an epoch)
    auto make_dataset = [&](int64_t skip) {
        const std::vector<llama_token> t(tokens.begin() + skip*stride, tokens.end());
        if (train.empty()) {
            return common_opt_dataset_init(ctx, t, stride);
        }
        const std::vector<uint8_t> m(train.begin() + skip*stride, train.end());
        return common_opt_dataset_init_masked(ctx, t, m, stride);
    };
    ggml_opt_dataset_t dataset = make_dataset(0);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    // Compile optional skip-regex (Mamba/SSM tensors etc.) ONCE, hold it stable
    // for the lifetime of the optimizer setup.
    std::regex skip_re;
    bool skip_re_active = !params.train_skip_regex.empty();
    if (skip_re_active) {
        try {
            skip_re = std::regex(params.train_skip_regex);
        } catch (const std::regex_error & e) {
            LOG_ERR("%s: invalid --train-skip-regex '%s': %s\n",
                    __func__, params.train_skip_regex.c_str(), e.what());
            return 1;
        }
        LOG_INF("%s: parameter filter active — tensors matching /%s/ will be FROZEN (no gradients)\n",
                __func__, params.train_skip_regex.c_str());
    }

    // Stage-3: optionally bootstrap a fresh LoRA training adapter. The base
    // tensors matched by --lora-train-regex are then frozen via the same
    // skip path; only A and B receive gradients.
    llama_adapter_lora * lora_adapter = nullptr;
    if (!params.lora_train_regex.empty()) {
        lora_training_config lcfg;
        lcfg.target_regex = params.lora_train_regex;
        lcfg.rank         = params.lora_train_rank;
        lcfg.alpha        = params.lora_train_alpha;
        lora_adapter      = lora_training_init(model, lcfg);
        if (!lora_adapter) {
            LOG_ERR("%s: --lora-train-regex /%s/ matched zero tensors\n",
                    __func__, params.lora_train_regex.c_str());
            return 1;
        }
        LOG_INF("%s: LoRA training adapter initialised — rank=%d alpha=%.1f\n",
                __func__, params.lora_train_rank, (double) params.lora_train_alpha);

        // Stage-3.5: attach the adapter to the context so build_lora_mm / build_lora_mm_id
        // pick up the A/B tensors during forward-graph construction. Without this, the
        // adapter lives only in model->loras (lifetime tracking) but never enters the
        // computational graph — backward then sees zero trainable params.
        float lora_scale = 1.0f;
        llama_set_adapters_lora(ctx, &lora_adapter, 1, &lora_scale);

        // Compute the same output path as the post-training block below so
        // the signal handler can flush the current adapter state on SIGTERM.
        g_lora_adapter_for_signal = lora_adapter;
        g_adapter_out_for_signal  = params.out_file;
        {
            const std::string ext = ".gguf";
            if (g_adapter_out_for_signal.size() >= ext.size() &&
                g_adapter_out_for_signal.compare(g_adapter_out_for_signal.size() - ext.size(),
                                                 ext.size(), ext) == 0) {
                g_adapter_out_for_signal.insert(g_adapter_out_for_signal.size() - ext.size(), ".lora");
            } else {
                g_adapter_out_for_signal += ".lora.gguf";
            }
        }
        std::signal(SIGTERM, finetune_on_signal);
        std::signal(SIGINT,  finetune_on_signal);

        // C.4: arm periodic mid-training checkpoint, if requested. Saves to
        // the same path as the signal handler / final save so a crashed run
        // can be resumed via --lora <path>.
        g_checkpoint_state.adapter = lora_adapter;
        g_checkpoint_state.ctx     = ctx;
        g_checkpoint_state.lr      = &params.lr;
        g_checkpoint_state.path    = g_adapter_out_for_signal;
        g_checkpoint_state.ubatches_per_datapoint = std::max<int64_t>(1, llama_n_ctx(ctx) / llama_n_ubatch(ctx));
        g_checkpoint_state.stop_after = params.train_stop_after;
        if (params.checkpoint_every_n_batches > 0) {
            g_checkpoint_state.every_n = params.checkpoint_every_n_batches;
            LOG_INF("%s: periodic checkpoint enabled — flushing every %d training batches to '%s'\n",
                    __func__, params.checkpoint_every_n_batches, g_adapter_out_for_signal.c_str());
        }
    }

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/lora_adapter ? finetune_param_filter_none :
                             skip_re_active ? finetune_param_filter_skip_regex : llama_opt_param_filter_all,
        /*param_filter_ud =*/skip_re_active ? (void *) &skip_re : nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
        /*grad_clip       =*/params.grad_clip,
    };
    llama_opt_init(ctx, model, lopt_params);

    // --resume: continue from the adapter, optimizer state and position saved at -o (by a checkpoint, a signal or the
    // end of an epoch)
    int     start_epoch     = 0;
    int64_t start_datapoint = 0;
    if (params.train_resume && lora_adapter) {
        const std::string & path = g_adapter_out_for_signal;
        std::ifstream state_file(path + ".state");
        if (!state_file) {
            LOG_WRN("%s: --resume: no '%s.state', starting from scratch\n", __func__, path.c_str());
        } else {
            const nlohmann::json state = nlohmann::json::parse(state_file);
            start_epoch     = state.at("epoch").get<int>();
            start_datapoint = state.at("datapoints_done").get<int64_t>();
            params.lr.step  = state.value("lr_step", (int64_t) 0);
            const int32_t n_pairs = llama_adapter_lora_load_weights(lora_adapter, path.c_str());
            if (n_pairs < 0) {
                LOG_ERR("%s: --resume: cannot load the adapter '%s' (different targets or rank?)\n", __func__, path.c_str());
                return 1;
            }
            if (llama_opt_load_state(ctx, (path + ".opt").c_str()) != 0) {
                LOG_ERR("%s: --resume: cannot load the optimizer state '%s.opt' (same optimizer and targets?)\n", __func__, path.c_str());
                return 1;
            }
            LOG_INF("%s: --resume: %d LoRA pairs from '%s', continuing at epoch %d, datapoint %" PRId64 "\n",
                    __func__, n_pairs, path.c_str(), start_epoch + 1, start_datapoint);
        }
    }

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    // The callback shows the progress bar, tracks the position (for --resume), flushes periodic checkpoints and handles
    // a stop request; it only counts training batches.
    ggml_opt_epoch_callback train_cb = lora_adapter ? finetune_epoch_callback_with_checkpoint : ggml_opt_epoch_callback_progress_bar;

    for (lr.epoch = start_epoch; lr.epoch < lr.epochs; ++lr.epoch) {
        const int64_t skip = lr.epoch == (unsigned) start_epoch ? std::min(start_datapoint, idata_split) : 0;
        g_checkpoint_state.epoch            = lr.epoch;
        g_checkpoint_state.datapoint_offset = skip;
        g_checkpoint_state.ubatches_done    = 0;
        if (skip > 0) {
            ggml_opt_dataset_t dataset_rest = make_dataset(skip);
            llama_opt_epoch(ctx, dataset_rest, result_train, result_eval, idata_split - skip, train_cb, train_cb);
            ggml_opt_dataset_free(dataset_rest);
        } else {
            llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, train_cb, train_cb);
        }
        fprintf(stderr, "\n");

        if (lora_adapter) {
            // epoch boundary: flush adapter and position, so a later --resume starts at the next epoch
            if (llama_adapter_lora_save_to_file(lora_adapter, g_adapter_out_for_signal.c_str()) == 0) {
                finetune_write_state(lr.epoch + 1, 0);
            }
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    if (lora_adapter != nullptr) {
        // Saving the merged base+adapter as a GGUF is not yet supported for
        // quantised tensors, so write the adapter alone (round-trippable via
        // --lora). Defaults to <out_file>.lora.gguf if --output is the model.
        std::string adapter_out = params.out_file;
        const std::string ext = ".gguf";
        if (adapter_out.size() >= ext.size() &&
            adapter_out.compare(adapter_out.size() - ext.size(), ext.size(), ext) == 0) {
            adapter_out.insert(adapter_out.size() - ext.size(), ".lora");
        } else {
            adapter_out += ".lora.gguf";
        }
        if (llama_adapter_lora_save_to_file(lora_adapter, adapter_out.c_str()) != 0) {
            LOG_ERR("%s: failed to write LoRA adapter to '%s'\n", __func__, adapter_out.c_str());
        } else {
            LOG_INF("%s: wrote trained LoRA adapter to '%s' — load with --lora %s\n",
                    __func__, adapter_out.c_str(), adapter_out.c_str());
        }
    } else {
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    llama_backend_free();

    return 0;
}
