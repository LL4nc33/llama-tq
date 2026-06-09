#include "arg.h"
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
static volatile sig_atomic_t g_signal_save_done = 0;

[[noreturn]] static void finetune_save_adapter_on_signal(int signum) {
    if (g_signal_save_done || !g_lora_adapter_for_signal || g_adapter_out_for_signal.empty()) {
        _exit(128 + signum);
    }
    g_signal_save_done = 1;
    // Best-effort save — the rest of the process may already be in an
    // inconsistent state, but the adapter buffers are independent.
    llama_adapter_lora_save_to_file(g_lora_adapter_for_signal,
                                    g_adapter_out_for_signal.c_str());
    _exit(128 + signum);
}

// Regex-based parameter filter for selective fine-tuning (e.g. skip Mamba/SSM layers).
// When the tensor name matches the regex, it is EXCLUDED from training.
static bool finetune_param_filter_skip_regex(const struct ggml_tensor * t, void * ud) {
    auto * re = static_cast<std::regex *>(ud);
    return !std::regex_search(t->name, *re);
}

// C.4: Periodic mid-training checkpoint state. Tracks how many training
// batches have been seen since process start and the target path / adapter
// to flush. Hooked into ggml_opt_epoch via a wrapper callback that calls
// the progress bar AND triggers a save every N batches.
struct finetune_checkpoint_state {
    llama_adapter_lora * adapter         = nullptr;
    std::string          path;
    int                  every_n         = 0;   // 0 = disabled
    int64_t              batches_seen    = 0;   // monotonically increasing
    int64_t              last_save_batch = 0;   // batches_seen at most recent save
};
static finetune_checkpoint_state g_checkpoint_state;

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

    // Only count training batches; eval-pass callbacks don't modify weights.
    if (!train) {
        return;
    }
    auto & cs = g_checkpoint_state;
    if (cs.every_n <= 0 || !cs.adapter || cs.path.empty()) {
        return;
    }
    cs.batches_seen++;
    if (cs.batches_seen - cs.last_save_batch < cs.every_n) {
        return;
    }
    cs.last_save_batch = cs.batches_seen;
    if (llama_adapter_lora_save_to_file(cs.adapter, cs.path.c_str()) != 0) {
        fprintf(stderr, "\n%s: checkpoint flush at batch %" PRId64 " FAILED to '%s'\n",
                __func__, cs.batches_seen, cs.path.c_str());
    } else {
        fprintf(stderr, "\n%s: checkpoint at batch %" PRId64 " → '%s'\n",
                __func__, cs.batches_seen, cs.path.c_str());
    }
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

    std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, true);
    ggml_opt_dataset_t       dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

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
        std::signal(SIGTERM, finetune_save_adapter_on_signal);
        std::signal(SIGINT,  finetune_save_adapter_on_signal);

        // C.4: arm periodic mid-training checkpoint, if requested. Saves to
        // the same path as the signal handler / final save so a crashed run
        // can be resumed via --lora <path>.
        if (params.checkpoint_every_n_batches > 0) {
            g_checkpoint_state.adapter = lora_adapter;
            g_checkpoint_state.path    = g_adapter_out_for_signal;
            g_checkpoint_state.every_n = params.checkpoint_every_n_batches;
            LOG_INF("%s: periodic checkpoint enabled — flushing every %d training batches to '%s'\n",
                    __func__, params.checkpoint_every_n_batches, g_adapter_out_for_signal.c_str());
        }
    }

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/skip_re_active ? finetune_param_filter_skip_regex : llama_opt_param_filter_all,
        /*param_filter_ud =*/skip_re_active ? (void *) &skip_re : nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    // Use the checkpoint-aware callback only when the periodic checkpoint
    // feature is active; otherwise fall through to the upstream progress
    // bar with zero overhead. Eval pass keeps the plain progress bar — we
    // don't want a stray flush triggered from validation batches.
    ggml_opt_epoch_callback train_cb =
        (params.checkpoint_every_n_batches > 0 && lora_adapter)
            ? finetune_epoch_callback_with_checkpoint
            : ggml_opt_epoch_callback_progress_bar;

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        train_cb, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

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
