// Block-diffusion generation for diffusion_gemma.
//
// Implements the reference block-diffusion loop (EntropyBoundSampler + StableAndConfident
// stopping + linear temperature schedule) with KV-cache reuse:
//
//   * ENCODER phase (causal, no self-conditioning): the prompt is prefilled once into the
//     unified sliding-window KV cache. Its per-layer K/V become the read-only prefix.
//   * DECODER phase (bidirectional, self-conditioned): each denoising step decodes only the
//     canvas tokens at positions [n_past, n_past+canvas). They read the cached prefix and
//     attend the canvas bidirectionally. After reading the logits the canvas K/V is rolled
//     back (llama_memory_seq_rm) so the cache keeps only the committed prefix.
//
// This avoids re-encoding the prompt on every denoising step. Multi-block autoregressive
// generation (commit the finalized canvas via an encoder pass, then advance n_past) is
// layered on top of this single-block loop.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

// imatrix activation collector (Stats / IMatrixCollector / g_collector / ik_collect_imatrix).
// Used only when DG_DUMP_IMATRIX is set, to gather an importance matrix from the real denoise
// (decoder) forward passes instead of the causal encoder prefill.
#include "imatrix-collector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// reference defaults from generation_config.json / DiffusionGemmaGenerationConfig
static constexpr int   DEF_CANVAS_LENGTH      = 256;
static constexpr int   DEF_MAX_DENOISE_STEPS  = 48;
static constexpr float ENTROPY_BOUND          = 0.1f;   // EntropyBoundSamplerConfig.entropy_bound
static constexpr float TEMP_MIN               = 0.4f;   // LinearTemperatureScheduleConfig.t_min
static constexpr float TEMP_MAX               = 0.8f;   // LinearTemperatureScheduleConfig.t_max
static constexpr float CONFIDENCE_THRESHOLD   = 0.005f; // StableAndConfident.confidence_threshold
static constexpr int   STABILITY_THRESHOLD    = 1;      // StableAndConfident.stability_threshold
static constexpr int   GPU_SAMPLING_MAX_TOP_K = 1024;   // small top-k sort kernel limit

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static float env_float(const char * name, float def) {
    const char * v = getenv(name);
    return v ? strtof(v, nullptr) : def;
}

// -----------------------------------------------------------------------------------------------
// self_cond_mlp activation dumper (DG_DUMP_SELFCOND) — QAT distillation data collector.
//
// Collects aligned (x_in, y_target) pairs for isolated self_cond_mlp distillation training. The
// graph (src/models/diffusion-gemma.cpp) tags two tensors via cb(), present ONLY in the decoder
// (denoise) phase, both shape {n_embd, n_tokens}:
//   * "self_cond_mlp_in"  — rms-normed soft-embed fed into the self_cond gated FFN  (= x_in)
//   * "self_cond_mlp_out" — output of that FFN (the f16/Q4-teacher target)          (= y_target)
//
// This is a small dedicated eval callback (NOT the imatrix collector). It is registered via
// ctx_params.cb_eval and gated by enable()/disable() exactly around the denoise llama_decode, so
// the encoder prefill / canvas-commit decodes contribute nothing (the tensors don't exist there).
//
// OUTPUT FORMAT (consumed by the Python self_cond_mlp training script):
//   <path>.in.bin   raw little-endian float32, row-major append of [n_tokens][n_embd] per step.
//   <path>.out.bin  raw little-endian float32, same layout, ALIGNED 1:1 with .in.bin per step.
//   <path>.idx      raw little-endian int32 pairs (block_idx, cur_step) per collected step, ALIGNED
//                   1:1 with the .in/.out records. Lets the trainer form consecutive-step pairs
//                   (pre_{t-1}, pre_t) for the contraction-hinge loss WITHOUT crossing a block
//                   boundary (each block resets self-cond to zero — pairing across it is invalid).
//                   cur_step counts DOWN n_steps..1 within a block, so adjacency is |Δcur_step|==1
//                   AND same block_idx.
//   <path>.meta     one text line: "n_embd=<E> n_tokens=<T> n_steps=<S> dtype=float32 layout=row-major(token,embd)"
//                   written at the end (once the final step count is known).
//
// Both files grow by exactly T*E floats per collected step, and in[step] aligns with out[step]
// (same forward, same shape, same write order). Reader: open both files, read S = filesize/(T*E*4)
// steps, reshape each to [S, T, E]; x = in[s,t,:], y = out[s,t,:] is one training pair.
//
// Subsampling + cap (keeps the file from exploding; ~2.9 MB per tensor per step at E=2816,T=256):
//   DG_DUMP_SELFCOND_STRIDE   collect only every Nth denoise step (default 1 — consecutive steps,
//                             required for the contraction-hinge pairing; raise only for the old
//                             single-step distill where adjacency does not matter).
//   DG_DUMP_SELFCOND_MAX_STEPS hard cap on collected steps total across all prompts (default 2000).
// Default (DG_DUMP_SELFCOND unset): callback never registered, zero overhead, behaviour unchanged.
struct selfcond_dumper {
    bool        enabled_path = false; // DG_DUMP_SELFCOND set (callback registered)
    bool        gate         = false; // true only around the denoise decode (enable/disable bracket)
    std::string path;
    int         stride       = 1;     // collect every Nth eligible step (1 = consecutive, for hinge)
    int         max_steps    = 2000;  // hard cap on collected steps (across prompts)

    std::ofstream in_f;
    std::ofstream out_f;
    std::ofstream idx_f;              // (block_idx, cur_step) int32 pairs, 1:1 with in/out records

    int     eligible_seen = 0;        // denoise steps seen (gate flips per step)
    int     steps_written = 0;        // steps actually dumped (capped by max_steps)
    bool    want_this_step= false;    // this step passes the stride+cap filter
    bool    wrote_in      = false;    // exactly one "in" written this step
    bool    wrote_out     = false;    // exactly one "out" written this step
    int32_t cur_block     = 0;        // set by the loop before each step (for the .idx sidecar)
    int32_t cur_step      = 0;        // set by the loop before each step (counts down n_steps..1)
    int64_t n_embd        = 0;        // captured from the first tensor (for the .meta file)
    int64_t n_tokens      = 0;
    std::vector<float> scratch;       // host staging for GPU tensors

    void open(const std::string & p, int s, int m) {
        path = p; stride = std::max(1, s); max_steps = std::max(0, m);
        in_f.open(path + ".in.bin",  std::ios::binary | std::ios::trunc);
        out_f.open(path + ".out.bin", std::ios::binary | std::ios::trunc);
        idx_f.open(path + ".idx",     std::ios::binary | std::ios::trunc);
        enabled_path = in_f.good() && out_f.good() && idx_f.good();
    }
    // the loop calls this right before enable() each step, so the .idx record (and any pairing
    // logic) knows which (block, step) this collected record belongs to.
    void set_step(int32_t block, int32_t step) { cur_block = block; cur_step = step; }
    // call enable() right before a denoise llama_decode, disable() right after. enable() decides
    // (once per step) whether this step is collected; that bracket guarantees one in + one out.
    void enable() {
        gate = true;
        want_this_step = enabled_path && (steps_written < max_steps) && (eligible_seen % stride == 0);
        wrote_in = false;
        wrote_out = false;
    }
    void disable() {
        // count one collected step iff both halves landed (defensive: only advance when aligned).
        // Write the (block, step) index in the SAME order as the in/out records so the trainer can
        // reconstruct adjacency without guessing.
        if (gate && want_this_step && wrote_in && wrote_out) {
            const int32_t rec[2] = { cur_block, cur_step };
            idx_f.write((const char *) rec, sizeof(rec));
            ++steps_written;
        }
        ++eligible_seen;
        gate = false;
        want_this_step = false;
    }
    void finalize() {
        if (!enabled_path) return;
        in_f.flush(); out_f.flush(); idx_f.flush();
        std::ofstream meta(path + ".meta", std::ios::trunc);
        meta << "n_embd=" << n_embd << " n_tokens=" << n_tokens
             << " n_steps=" << steps_written
             << " dtype=float32 layout=row-major(token,embd) idx=int32(block,cur_step)\n";
    }
};

static selfcond_dumper g_selfcond;

// dedicated eval callback: interested ONLY in the two self_cond_mlp taps, ONLY while gated to a
// denoise decode that passed the stride/cap filter. Writes raw float32 to the .in / .out files.
static bool dg_collect_selfcond(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    if (!g_selfcond.gate || !g_selfcond.want_this_step) {
        return false; // not in a collected denoise step: report no interest, materialize nothing
    }
    const bool is_in  = strcmp(t->name, "self_cond_mlp_in")  == 0;
    const bool is_out = strcmp(t->name, "self_cond_mlp_out") == 0;
    if (!is_in && !is_out) {
        return false;
    }
    if (ask) {
        // already-have-this-half guard: never write a tensor twice in one step
        return is_in ? !g_selfcond.wrote_in : !g_selfcond.wrote_out;
    }

    // ask == false: the data is ready. Shape is {n_embd, n_tokens}; ggml is column-major so the
    // contiguous layout is exactly [n_tokens][n_embd] floats (token-major, embd-minor) — the row-
    // major (token,embd) layout documented above. Copy from GPU if needed, else read t->data.
    const int64_t E = t->ne[0]; // n_embd
    const int64_t T = t->ne[1]; // n_tokens
    if (g_selfcond.n_embd == 0) { g_selfcond.n_embd = E; g_selfcond.n_tokens = T; }

    const size_t nbytes = ggml_nbytes(t);
    const float * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const float *) t->data;
    } else {
        g_selfcond.scratch.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(t, g_selfcond.scratch.data(), 0, nbytes);
        data = g_selfcond.scratch.data();
    }

    std::ofstream & f = is_in ? g_selfcond.in_f : g_selfcond.out_f;
    f.write((const char *) data, (std::streamsize) ((size_t) E * (size_t) T * sizeof(float)));
    if (is_in)  g_selfcond.wrote_in  = true;
    else        g_selfcond.wrote_out = true;
    return true;
}

// apply the model's chat template to the user prompt (this is a chat-trained model)
static std::string format_chat(llama_model * model, const std::string & prompt) {
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg user;
    user.role = "user";
    user.content = prompt;
    inputs.messages.push_back(user);
    inputs.add_generation_prompt = true;
    return common_chat_templates_apply(tmpls.get(), inputs).prompt;
}

// per-run diffusion config (parsed once in main(), shared by every prompt of an imatrix run).
struct dg_config {
    int   canvas_length;
    int   n_steps;
    int   max_canvases;
    float entropy_bound;
    int   topk_fixed;
    int   topk_start;
    int   topk_end;
    int   topk_tail;
    bool  collect_imatrix;  // DG_DUMP_IMATRIX set: bracket the denoise decode with g_collector
    bool  collect_selfcond; // DG_DUMP_SELFCOND set: bracket the denoise decode with g_selfcond
};

// Run one prompt end-to-end: prefill the prefix, denoise the canvas block(s), print the answer.
// The caller owns `model` (shared across prompts) and frees it; this frees only the per-prompt
// context + batch. Returns 0 on success, 1 on error. When cfg.collect_imatrix is set, the denoise
// (decoder) forward passes feed g_collector; the encoder prefill and canvas-commit decodes do not.
static int run_one_prompt(llama_model * model, const common_params & params, const dg_config & cfg,
                          const std::string & prompt_text) {
    const int   canvas_length = cfg.canvas_length;
    const int   n_steps       = cfg.n_steps;
    const int   max_canvases  = cfg.max_canvases;
    const float entropy_bound = cfg.entropy_bound;
    const int   topk_fixed    = cfg.topk_fixed;
    const int   topk_start    = cfg.topk_start;
    const int   topk_end      = cfg.topk_end;
    const int   topk_tail     = cfg.topk_tail;

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // Build the prompt prefix. Text-only path tokenizes the chat-formatted prompt. Multimodal
    // path (--mmproj + --image) tokenizes via libmtmd: the image marker expands to the gemma
    // image tokens and the vision embeddings are produced by the GEMMA4V mmproj.
    const bool use_mm = !params.mmproj.path.empty() && !params.image.empty();

    std::vector<llama_token> prompt_tokens;               // text-only prefill
    mtmd::context_ptr        mctx_vision;                 // multimodal context
    mtmd::input_chunks       mm_chunks(mtmd_input_chunks_init());
    int prefix_len = 0;                                   // total positions in the prompt prefix

    if (use_mm) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu       = params.mmproj_use_gpu;
        mparams.print_timings = false;
        mparams.n_threads     = params.cpuparams.n_threads;
        mctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!mctx_vision) {
            LOG_ERR("error: failed to load mmproj '%s'\n", params.mmproj.path.c_str());
            return 1; // caller owns + frees the shared model
        }

        // load image(s) and build one media marker per image
        mtmd::bitmaps bitmaps;
        std::string markers;
        for (const auto & img : params.image) {
            // fork mtmd API: 2-arg helper returning mtmd_bitmap* directly (no result struct)
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(mctx_vision.get(), img.c_str()));
            if (!bmp.ptr) {
                LOG_ERR("error: failed to load image '%s'\n", img.c_str());
                return 1; // caller owns + frees the shared model
            }
            bitmaps.entries.push_back(std::move(bmp));
            markers += mtmd_default_marker();
            markers += "\n";
        }

        // chat-format with the image marker(s) prepended to the user content
        const std::string formatted = format_chat(model, markers + prompt_text);
        LOG_INF("formatted prompt: %s\n", formatted.c_str());

        mtmd_input_text text;
        text.text          = formatted.c_str();
        text.add_special   = false;
        text.parse_special = true;
        auto bmp_c = bitmaps.c_ptr();
        if (mtmd_tokenize(mctx_vision.get(), mm_chunks.ptr.get(), &text, bmp_c.data(), bmp_c.size()) != 0) {
            LOG_ERR("error: mtmd_tokenize failed\n");
            return 1; // caller owns + frees the shared model
        }
        prefix_len = (int) mtmd_helper_get_n_pos(mm_chunks.ptr.get());
    } else {
        // text-only: chat-format and tokenize (turn/channel special tokens)
        if (!prompt_text.empty()) {
            const std::string formatted = format_chat(model, prompt_text);
            LOG_INF("formatted prompt: %s\n", formatted.c_str());
            prompt_tokens = common_tokenize(vocab, formatted, /*add_special*/ false, /*parse_special*/ true);
        }
        prefix_len = (int) prompt_tokens.size();
    }

    // Context holds the committed prefix (prompt + finalized canvases) plus the canvas being
    // denoised, plus one extra canvas of headroom (the in-flight canvas's K/V is written then
    // rolled back each denoising step, so the ring buffer needs room before cells are reused).
    const int n_ctx_min = prefix_len + (max_canvases + 1) * canvas_length;
    const int n_ctx     = std::max<int>(n_ctx_min, (int) params.n_ctx);
    // Largest single decode is the prompt prefill (prefix_len) or a canvas pass (canvas_length).
    const int n_ub      = std::max(std::max(prefix_len, canvas_length), 1);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_ctx;
    ctx_params.n_batch  = n_ub;
    ctx_params.n_ubatch = n_ub;
    ctx_params.no_perf  = params.no_perf;
    // Honor the KV-cache quantization + flash-attention flags (ktq2_1/vtq2_1 etc.); the default
    // ctx params would otherwise pin K/V to f16 regardless of --cache-type-k/-v.
    ctx_params.type_k      = params.cache_type_k;
    ctx_params.type_v      = params.cache_type_v;
    ctx_params.flash_attn_type = params.flash_attn_type;
    // Register a decoder-path eval callback. ctx_params.cb_eval holds exactly ONE callback, so the
    // self_cond dumper and the imatrix collector are mutually exclusive; DG_DUMP_SELFCOND wins (the
    // warning is emitted once in main()). Both stay gated and only fire around the denoise decode
    // below, so the encoder prefill / canvas-commit decodes don't contribute. No-op when neither set.
    if (cfg.collect_selfcond) {
        ctx_params.cb_eval           = dg_collect_selfcond;
        ctx_params.cb_eval_user_data = nullptr;
    } else if (cfg.collect_imatrix) {
        ctx_params.cb_eval           = ik_collect_imatrix;
        ctx_params.cb_eval_user_data = nullptr;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        return 1; // caller owns + frees the shared model
    }
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    llama_set_diffusion_prompt_len(ctx, prefix_len);
    llama_memory_t mem = llama_get_memory(ctx);

    const int topk_max_requested =
        (topk_start > 0 && topk_end > 0) ? std::max(topk_start, topk_end) : topk_fixed;
    // GPU/device sampling defaults OFF in this fork: the device self-cond and device-loop paths
    // require the canvas/self-cond inputs to live on a CUDA buffer (set_diffusion_input_backend),
    // a GPU-input-placement perf pass not yet wired here. The host (CPU) sampling path is correct
    // and coherent. Re-enable per env once that pass lands: DG_GPU_SAMPLING / DG_DEVICE_* = 1.
    const bool gpu_sampling_requested = env_int("DG_GPU_SAMPLING", 0) != 0;
    const bool gpu_sampling_topk_ok = topk_max_requested <= 0 || topk_max_requested <= GPU_SAMPLING_MAX_TOP_K;
    const bool use_gpu_sampling = gpu_sampling_requested &&
                                  gpu_sampling_topk_ok &&
                                  llama_diffusion_sample_topk_supported(ctx);
    const bool use_device_self_cond = use_gpu_sampling && env_int("DG_DEVICE_SELFCOND", 1) != 0;
    const bool use_device_loop      = use_device_self_cond && env_int("DG_DEVICE_LOOP", 1) != 0;
    const int device_early_stop_interval = use_device_loop ? 1 : 0;
    llama_set_diffusion_gpu_sampling(ctx, use_gpu_sampling);

    LOG_INF("diffusion-gemma: prefix=%d canvas=%d max_canvases=%d steps=%d entropy_bound=%.3f temp=[%.2f,%.2f] n_ctx=%d mm=%d\n",
            prefix_len, canvas_length, max_canvases, n_steps, entropy_bound, TEMP_MIN, TEMP_MAX, n_ctx, (int) use_mm);
    LOG_INF("diffusion-gemma: gpu sampling: %s%s%s%s\n",
            use_gpu_sampling ? "on" : "off",
            (!gpu_sampling_topk_ok ? " (top-k exceeds CUDA fast-path limit)" :
             (!gpu_sampling_requested ? " (disabled by DG_GPU_SAMPLING=0)" : "")),
            use_device_self_cond ? " | device self-cond: on" : "",
            use_device_loop ? " | device loop: on" : "");
    if (device_early_stop_interval > 0) {
        LOG_INF("diffusion-gemma: device early-stop interval=%d\n", device_early_stop_interval);
    }
    if (topk_fixed > 0 || (topk_start > 0 && topk_end > 0)) {
        LOG_INF("diffusion-gemma: top-k sampling: fixed=%d anneal=[%d->%d] tail_correction=%d (vocab=%d)\n",
                topk_fixed, topk_start, topk_end, topk_tail, n_vocab);
    }

    std::mt19937 rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed);
    std::uniform_int_distribution<int> rand_tok(0, n_vocab - 1);
    std::uniform_real_distribution<float> rand_unif(0.0f, 1.0f);

    llama_batch batch = llama_batch_init(n_ub, 0, 1);

    // ---- ENCODER phase: prefill the prompt prefix into the KV cache (no self-conditioning) ----
    int n_past = 0;
    llama_set_diffusion_decoder_phase(ctx, false);
    llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);
    const auto t_prefill_start = std::chrono::steady_clock::now();
    if (use_mm) {
        // mtmd_helper_eval_chunks decodes text chunks (tokens, causal) and the image chunk
        // (vision embeddings, bidirectional for gemma) into the cache, managing causal_attn.
        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(mctx_vision.get(), ctx, mm_chunks.ptr.get(),
                /*n_past*/ 0, /*seq_id*/ 0, /*n_batch*/ n_ub, /*logits_last*/ true, &new_n_past)) {
            LOG_ERR("error: multimodal prefill failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            return 1; // caller owns + frees the shared model
        }
        n_past = (int) new_n_past; // prompt + image K/V is now the committed read-only prefix
    } else if (prefix_len > 0) {
        llama_set_causal_attn(ctx, true);
        batch.n_tokens = prefix_len;
        for (int i = 0; i < prefix_len; ++i) {
            batch.token[i]     = prompt_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == prefix_len - 1) ? 1 : 0; // logits unused; keep n_outputs >= 1
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: prompt prefill (encoder) decode failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            return 1; // caller owns + frees the shared model
        }
        n_past = prefix_len; // prompt K/V is now the committed read-only prefix
    }
    const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill_start).count();
    if (prefix_len > 0) {
        // encoder-phase prefill: causal, NO self-conditioning -> same forward as the base
        // gemma4 model (the self-cond matmul only runs in the decoder/denoise phase).
        LOG_INF("prefill (encoder, no self-cond): %d tokens in %.3f s (%.1f tok/s)\n",
                prefix_len, prefill_s, prefill_s > 0.0 ? prefix_len / prefill_s : 0.0);
    }

    std::vector<llama_token> canvas(canvas_length);
    std::vector<llama_token> argmax_canvas(canvas_length, -1);
    std::vector<llama_token> prev_argmax(canvas_length, -1);
    std::vector<llama_token> accepted(canvas_length);
    // sparse self-conditioning over the canvas: the previous step's top-SC_K token ids + their
    // (renormalized) softmax probabilities per position. Fed to the next decode (Option-2 graph
    // gather: the decoder gathers just these SC_K embedding rows and blends them, instead of a
    // dense full-vocab probs @ token_embd matmul). Zero probs => no self-conditioning (step 1).
    // SC_K must match llama_model_diffusion_gemma::N_SC_TOPK (the graph's fixed gather width).
    const int SC_K = 256;
    std::vector<int32_t> sc_ids ((size_t) SC_K * canvas_length, 0);
    std::vector<float>   sc_probs((size_t) SC_K * canvas_length, 0.0f);

    // all generated tokens across the autoregressive canvas blocks
    std::vector<llama_token> generated;

    // ---- autoregressive block loop: each block denoises a canvas against the cached prefix,
    //      then (if continuing) commits its finalized tokens to the cache as the next prefix ----
    int n_blocks_run  = 0;
    int n_steps_total = 0;
    const auto t_gen_start = std::chrono::steady_clock::now();
    const bool log_step_timing = env_int("DG_TIMING", 0) != 0;
    // DG_SC_TEMP: separate (typically lower) temperature for the self-conditioning probabilities
    // only. Sharpening the self-cond distribution feeds fewer noise tokens back into the next
    // denoise step, which stabilizes the loop on aggressively quantized (e.g. 2-bit) logits.
    // Sampling/entropy/argmax are untouched. <= 0 or unset => use the sampling temperature (no-op).
    const float sc_temp_env = env_float("DG_SC_TEMP", 0.0f);
    const bool  use_sc_temp = sc_temp_env > 0.0f;
    // DG_DITHER: stochastic-resonance dithering on the self-conditioning probs only. Adds decaying
    // Gaussian noise (sigma_t = DG_DITHER * cur_step/n_steps) to the stored spr values, then clamps
    // to >= 0 and renormalizes per position. Early steps inject more noise (search/escape from a
    // chaotic attractor where 2-bit quant repeats neighbouring tokens), late steps ~0 (converge).
    // Composes after sc_temp normalization. <= 0 or unset => exact previous behaviour (no overhead).
    const float dither_base = env_float("DG_DITHER", 0.0f);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    double decode_enqueue_s = 0.0;
    double sample_sync_s    = 0.0;
    double host_loop_s      = 0.0;

    bool done = false;
    bool failed = false;
    for (int block = 0; block < max_canvases && !done; ++block) {
    ++n_blocks_run;
    // 1. initialize canvas with random tokens
    for (auto & t : canvas) t = rand_tok(rng);
    std::fill(prev_argmax.begin(), prev_argmax.end(), -1);
    llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0); // first step: zero self-conditioning

    // 2. denoising loop (DECODER phase): cur_step = n_steps .. 1
    int step_k = 0; // top-k used for the current step (0 = full softmax), for logging
    for (int cur_step = n_steps; cur_step >= 1; --cur_step) {
        ++n_steps_total;
        // 2a. decode the canvas only, at positions [n_past, n_past+canvas). Bidirectional, with
        // self-conditioning; it reads the cached prefix read-only. All canvas tokens are outputs.
        llama_set_causal_attn(ctx, false);
        llama_set_diffusion_decoder_phase(ctx, true);
        batch.n_tokens = canvas_length;
        for (int j = 0; j < canvas_length; ++j) {
            batch.token[j]     = canvas[j];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        const auto t_decode_start = std::chrono::steady_clock::now();
        // collect imatrix activations from THIS (decoder/denoise) forward pass only. Bracketing the
        // decode keeps the collector off for the encoder prefill and the canvas-commit decodes, so
        // the imatrix reflects exactly the non-causal, self-conditioned denoise path.
        if (cfg.collect_imatrix) {
            g_collector.enable();
        }
        if (cfg.collect_selfcond) {
            g_selfcond.set_step(block, cur_step); // (block, cur_step) for the .idx sidecar pairing
            g_selfcond.enable();
        }
        const int decode_rc = llama_decode(ctx, batch);
        if (cfg.collect_imatrix) {
            g_collector.disable();
        }
        if (cfg.collect_selfcond) {
            g_selfcond.disable();
        }
        if (decode_rc != 0) {
            LOG_ERR("error: llama_decode failed at step %d\n", cur_step);
            break;
        }
        decode_enqueue_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_decode_start).count();

        // 2b. linear temperature schedule: t = t_min + (t_max - t_min) * (cur_step / n_steps)
        const float temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * ((float) cur_step / (float) n_steps);

        std::vector<float> entropy(canvas_length);
        std::vector<llama_token> sampled(canvas_length);

        // k for this step: 0 = full softmax. With annealing, k is high at the first (high-entropy)
        // step and low at the last, since early canvases are flat (need many tokens) and late ones
        // are peaked (a few suffice).
        int k_step = 0;
        if (topk_start > 0 && topk_end > 0) {
            const float frac = (n_steps > 1) ? (float) (cur_step - 1) / (float) (n_steps - 1) : 0.0f; // 1 at first step (cur_step=n_steps), 0 at last (cur_step=1)
            k_step = (int) lroundf(topk_end + (topk_start - topk_end) * frac);
        } else if (topk_fixed > 0) {
            k_step = topk_fixed;
        }
        if (k_step <= 0 || k_step >= n_vocab) k_step = 0;
        step_k = k_step;

        if (use_device_loop) {
            bool sampled_ok = true;
            const int block_step = n_steps - cur_step + 1;
            const bool use_device_early_stop = device_early_stop_interval > 0;
            const bool reset_stop_state = use_device_early_stop && block_step == 1;
            const bool check_stop = use_device_early_stop;
            int32_t stop_flag = 0;
            const auto t_sample_start = std::chrono::steady_clock::now();
            llama_diffusion_sample_params sample_params = {
                /* .n_tokens              = */ canvas_length,
                /* .top_k                 = */ k_step,
                /* .self_cond_top_k       = */ SC_K,
                /* .temperature           = */ temp,
                /* .seed                  = */ params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed,
                /* .step                  = */ (uint32_t) n_steps_total,
                /* .top_k_tail_correction = */ topk_tail != 0,
            };
            llama_diffusion_sample_result sample_result = {
                /* .sampled         = */ nullptr,
                /* .argmax          = */ nullptr,
                /* .entropy         = */ nullptr,
                /* .self_cond_ids   = */ nullptr,
                /* .self_cond_probs = */ nullptr,
                /* .final_tokens    = */ (cur_step == 1 || check_stop) ? argmax_canvas.data() : nullptr,
                /* .stop            = */ check_stop ? &stop_flag : nullptr,
                /* .entropy_bound   = */ entropy_bound,
                /* .confidence_threshold = */ CONFIDENCE_THRESHOLD,
                /* .stability_threshold  = */ STABILITY_THRESHOLD,
                /* .update_canvas_on_device = */ cur_step > 1,
                /* .update_stop_state_on_device = */ use_device_early_stop,
                /* .check_stop_on_device = */ check_stop,
                /* .reset_stop_state = */ reset_stop_state,
            };
            sampled_ok = llama_diffusion_sample_topk(ctx, &sample_params, &sample_result);
            sample_sync_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_sample_start).count();
            llama_memory_seq_rm(mem, 0, n_past, -1);
            if (!sampled_ok) {
                LOG_ERR("error: CUDA diffusion device loop failed at step %d (k=%d)\n", cur_step, k_step);
                failed = true;
                done = true;
                break;
            }
            if (check_stop && stop_flag != 0) {
                break;
            }
            continue;
        }

        // sparse self-cond for the NEXT step: top-SC_K (id, prob) per position. Cleared each step;
        // unused slots stay (id 0, prob 0) so the graph gather contributes nothing for them.
        if (!use_device_self_cond) {
            std::fill(sc_probs.begin(), sc_probs.end(), 0.0f);
            std::fill(sc_ids.begin(),   sc_ids.end(),   0);
        }

        bool sampled_ok = true;
        const auto t_sample_start = std::chrono::steady_clock::now();
        // decaying stochastic-resonance sigma for this step (computed once, applied per position
        // to the self-cond probs in both host paths below). cur_step runs n_steps..1, so sigma
        // starts at dither_base and decays to ~0. <= 0 => disabled, no per-position overhead.
        const float dither_sigma = dither_base > 0.0f ? dither_base * ((float) cur_step / (float) n_steps) : 0.0f;
        if (use_gpu_sampling) {
            llama_diffusion_sample_params sample_params = {
                /* .n_tokens              = */ canvas_length,
                /* .top_k                 = */ k_step,
                /* .self_cond_top_k       = */ SC_K,
                /* .temperature           = */ temp,
                /* .seed                  = */ params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed,
                /* .step                  = */ (uint32_t) n_steps_total,
                /* .top_k_tail_correction = */ topk_tail != 0,
            };
            llama_diffusion_sample_result sample_result = {
                /* .sampled         = */ sampled.data(),
                /* .argmax          = */ argmax_canvas.data(),
                /* .entropy         = */ entropy.data(),
                /* .self_cond_ids   = */ use_device_self_cond ? nullptr : sc_ids.data(),
                /* .self_cond_probs = */ use_device_self_cond ? nullptr : sc_probs.data(),
                /* .final_tokens    = */ nullptr,
                /* .stop            = */ nullptr,
                /* .entropy_bound   = */ 0.0f,
                /* .confidence_threshold = */ 0.0f,
                /* .stability_threshold  = */ 0,
                /* .update_canvas_on_device = */ false,
                /* .update_stop_state_on_device = */ false,
                /* .check_stop_on_device = */ false,
                /* .reset_stop_state = */ false,
            };
            sampled_ok = llama_diffusion_sample_topk(ctx, &sample_params, &sample_result);
            if (!sampled_ok) {
                LOG_ERR("error: CUDA diffusion sampling failed at step %d (k=%d)\n", cur_step, k_step);
            }
        } else if (k_step == 0) {
            // canvas logits occupy rows [0, canvas_length) (canvas-only ubatch)
            const float * logits = llama_get_logits(ctx);
            // ---- full softmax over the whole vocabulary (reference behaviour) ----
            // self-cond still feeds only the top-SC_K tokens (full-normalized probs); the dropped
            // tail carries negligible embedding weight and the post RMS norm absorbs the scale.
            std::vector<float> probs(n_vocab);
            std::vector<std::pair<float,int>> scheap; scheap.reserve(SC_K); // min-heap of (x, idx), size SC_K
            const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
            for (int j = 0; j < canvas_length; ++j) {
                const float * lg = logits + (size_t) j * n_vocab;
                float maxl = -INFINITY;
                int   amax = 0;
                scheap.clear();
                for (int v = 0; v < n_vocab; ++v) {
                    const float x = lg[v] / temp;
                    if (x > maxl) { maxl = x; amax = v; }
                    if ((int) scheap.size() < SC_K) {
                        scheap.push_back({x, v});
                        std::push_heap(scheap.begin(), scheap.end(), cmp);
                    } else if (x > scheap.front().first) {
                        std::pop_heap(scheap.begin(), scheap.end(), cmp);
                        scheap.back() = {x, v};
                        std::push_heap(scheap.begin(), scheap.end(), cmp);
                    }
                }
                float sum = 0.0f;
                for (int v = 0; v < n_vocab; ++v) {
                    const float p = expf(lg[v] / temp - maxl);
                    probs[v] = p;
                    sum += p;
                }
                float ent = 0.0f;
                const float r = rand_unif(rng) * sum;
                float cum = 0.0f;
                int   tok = amax;
                bool  picked = false;
                for (int v = 0; v < n_vocab; ++v) {
                    const float p = probs[v] / sum;
                    if (p > 0.0f) ent -= p * logf(p);
                    cum += probs[v];
                    if (!picked && cum >= r) { tok = v; picked = true; }
                }
                // store top-SC_K self-cond (full-normalized probability per selected token)
                int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                float   * spr = sc_probs.data() + (size_t) j * SC_K;
                int slot = 0;
                if (!use_sc_temp) {
                    for (auto & h : scheap) { sid[slot] = h.second; spr[slot] = expf(h.first - maxl) / sum; ++slot; }
                } else {
                    // separate sc_temp for the self-cond probs: recompute the softmax over the
                    // scheap (top-SC_K) entries from the RAW logits at sc_temp. scheap stores
                    // {lg[v]/temp, v}, so the raw logit is h.first * temp. We normalize over the
                    // scheap set only (not the full vocab): the self-cond gather uses just these
                    // top-SC_K tokens and the dropped tail carries negligible embedding weight
                    // (absorbed by the post RMS norm), as noted above for the full-softmax path.
                    const float inv_sc = 1.0f / sc_temp_env;
                    float max_sc = -INFINITY;
                    for (auto & h : scheap) { const float xs = (h.first * temp) * inv_sc; if (xs > max_sc) max_sc = xs; }
                    float sum_sc = 0.0f;
                    for (auto & h : scheap) { sum_sc += expf((h.first * temp) * inv_sc - max_sc); }
                    for (auto & h : scheap) {
                        sid[slot] = h.second;
                        spr[slot] = expf((h.first * temp) * inv_sc - max_sc) / sum_sc;
                        ++slot;
                    }
                }
                // stochastic-resonance dithering: perturb the finished spr distribution, clamp to
                // >= 0, then renormalize over the stored slots so it stays a valid distribution.
                if (dither_sigma > 0.0f) {
                    float dsum = 0.0f;
                    for (int i = 0; i < slot; ++i) {
                        float p = spr[i] + dither_sigma * ndist(rng);
                        if (p < 0.0f) p = 0.0f;
                        spr[i] = p;
                        dsum += p;
                    }
                    if (dsum > 0.0f) { const float inv = 1.0f / dsum; for (int i = 0; i < slot; ++i) spr[i] *= inv; }
                }
                entropy[j]       = ent;
                sampled[j]       = tok;
                argmax_canvas[j] = amax;
            }
        } else {
            // canvas logits occupy rows [0, canvas_length) (canvas-only ubatch)
            const float * logits = llama_get_logits(ctx);
            // ---- top-k host sampling: softmax / entropy / sample / self-cond over the top-k
            // logits only. Self-cond feeds the top min(k,SC_K) tokens (renormalized over the
            // sampled top-k), gathered in-graph; the dropped tail carries negligible weight. ----
            const int heap_k = std::max(k_step, SC_K); // collect enough for both sampling and self-cond
            std::vector<std::pair<float,int>> heap; // min-heap of (logit/temp, idx), size heap_k
            heap.reserve(heap_k);
            const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
            for (int j = 0; j < canvas_length; ++j) {
                const float * lg = logits + (size_t) j * n_vocab;
                float maxl = -INFINITY;
                int   amax = 0;
                heap.clear();
                for (int v = 0; v < n_vocab; ++v) {
                    const float x = lg[v] / temp;
                    if (x > maxl) { maxl = x; amax = v; }
                    if ((int) heap.size() < heap_k) {
                        heap.push_back({x, v});
                        std::push_heap(heap.begin(), heap.end(), cmp);
                    } else if (x > heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end(), cmp);
                        heap.back() = {x, v};
                        std::push_heap(heap.begin(), heap.end(), cmp);
                    }
                }
                // sort the collected entries by logit descending (exp is monotonic with x): the
                // first k_step drive sampling/entropy, the first SC_K drive self-cond.
                std::sort(heap.begin(), heap.end(), [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });

                // softmax over the sampled top-k (renormalized); reuse .first to hold exp value
                float Zk = 0.0f;
                for (int i = 0; i < k_step; ++i) { const float e = expf(heap[i].first - maxl); heap[i].first = e; Zk += e; }

                float ent;
                if (topk_tail) {
                    // exact full entropy via logsumexp over all logits (one expf pass, no per-token logf):
                    //   H = ln(Z) - (sum_i (z_i-max) e_i)/Z
                    double Zf = 0.0, T = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        const double d = (double) (lg[v] / temp) - (double) maxl;
                        const double e = exp(d);
                        Zf += e; T += d * e;
                    }
                    ent = (float) (log(Zf) - T / Zf);
                } else {
                    ent = 0.0f;
                    for (int i = 0; i < k_step; ++i) { const float q = heap[i].first / Zk; if (q > 0.0f) ent -= q * logf(q); }
                }

                // multinomial sample over the sampled top-k
                const float r = rand_unif(rng) * Zk;
                float cum = 0.0f;
                int   tok = amax;
                bool  picked = false;
                for (int i = 0; i < k_step; ++i) {
                    cum += heap[i].first;
                    if (!picked && cum >= r) { tok = heap[i].second; picked = true; }
                }

                // store top-SC_K self-cond (renormalized over the sampled top-k)
                int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                float   * spr = sc_probs.data() + (size_t) j * SC_K;
                const int n_sc = std::min(k_step, SC_K);
                if (!use_sc_temp) {
                    for (int i = 0; i < n_sc; ++i) { sid[i] = heap[i].second; spr[i] = heap[i].first / Zk; }
                } else {
                    // separate sc_temp for the self-cond probs: heap[i].first now holds the
                    // sampling-temp exp value, so recompute from the RAW logits lg[idx] at sc_temp,
                    // renormalized over the same top-n_sc tokens (matching the top-k path semantics).
                    const float inv_sc = 1.0f / sc_temp_env;
                    float max_sc = -INFINITY;
                    for (int i = 0; i < n_sc; ++i) { const float xs = lg[heap[i].second] * inv_sc; if (xs > max_sc) max_sc = xs; }
                    float sum_sc = 0.0f;
                    for (int i = 0; i < n_sc; ++i) { sum_sc += expf(lg[heap[i].second] * inv_sc - max_sc); }
                    for (int i = 0; i < n_sc; ++i) {
                        sid[i] = heap[i].second;
                        spr[i] = expf(lg[heap[i].second] * inv_sc - max_sc) / sum_sc;
                    }
                }
                // stochastic-resonance dithering: perturb the finished spr distribution, clamp to
                // >= 0, then renormalize over the stored slots so it stays a valid distribution.
                if (dither_sigma > 0.0f) {
                    float dsum = 0.0f;
                    for (int i = 0; i < n_sc; ++i) {
                        float p = spr[i] + dither_sigma * ndist(rng);
                        if (p < 0.0f) p = 0.0f;
                        spr[i] = p;
                        dsum += p;
                    }
                    if (dsum > 0.0f) { const float inv = 1.0f / dsum; for (int i = 0; i < n_sc; ++i) spr[i] *= inv; }
                }
                entropy[j]       = ent;
                sampled[j]       = tok;
                argmax_canvas[j] = amax;
            }
        }
        sample_sync_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_sample_start).count();
        if (!sampled_ok) {
            llama_memory_seq_rm(mem, 0, n_past, -1);
            failed = true;
            done = true;
            break;
        }

        const auto t_host_start = std::chrono::steady_clock::now();

        // roll back the canvas K/V written by this decode so the cache keeps only the committed
        // prefix [0, n_past); the next step re-decodes the canvas fresh against that prefix.
        llama_memory_seq_rm(mem, 0, n_past, -1);

        // 2c. entropy-bound accept: sort positions by entropy ascending, accept the prefix
        // where sum(entropy of all-but-last) <= entropy_bound (monotonic -> prefix selection)
        std::vector<int> order(canvas_length);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return entropy[a] < entropy[b]; });

        std::vector<char> accept_mask(canvas_length, 0);
        float prefix = 0.0f;
        for (int k = 0; k < canvas_length; ++k) {
            if (prefix <= entropy_bound) {
                accept_mask[order[k]] = 1;
                prefix += entropy[order[k]];
            } else {
                break;
            }
        }

        // accepted canvas: accepted positions take the sampled token, others keep current
        int n_accept = 0;
        for (int i = 0; i < canvas_length; ++i) {
            if (accept_mask[i]) { accepted[i] = sampled[i]; ++n_accept; }
        }

        // mean entropy (confidence)
        const float mean_ent = std::accumulate(entropy.begin(), entropy.end(), 0.0f) / canvas_length;

        // 2d. stopping: stable (argmax canvas unchanged for STABILITY_THRESHOLD steps) AND confident
        bool stable = (STABILITY_THRESHOLD == 0) || (argmax_canvas == prev_argmax);
        bool confident = mean_ent < CONFIDENCE_THRESHOLD;
        LOG_INF("step %3d  temp=%.3f  k=%d  accepted=%4d/%d  mean_entropy=%.4f%s\n",
                cur_step, temp, step_k, n_accept, canvas_length, mean_ent,
                (stable && confident) ? "  [STOP]" : "");
        if (stable && confident) {
            break;
        }
        prev_argmax = argmax_canvas;

        // self-conditioning for the NEXT denoising step. With device self-cond this was already
        // copied D2D into the reused graph input tensors by llama_diffusion_sample_topk().
        if (!use_device_self_cond) {
            llama_set_diffusion_self_cond_topk(ctx, sc_ids.data(), sc_probs.data(), SC_K, canvas_length);
        }

        // 2e. renoise non-accepted positions with fresh random tokens -> next canvas
        for (int i = 0; i < canvas_length; ++i) {
            canvas[i] = accept_mask[i] ? accepted[i] : rand_tok(rng);
        }
        host_loop_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_host_start).count();
    }
    if (failed) {
        break;
    }

    // 3. block output = the inline argmax of the last (stable) denoising step's logits.
    // This matches the reference (DiffusionGemma _denoising_step uses argmax(processed_logits)
    // taken during the denoising forward, read once the canvas is stable + confident). There is
    // no separate read-out: the never-accepted tail is the model's own prediction given the
    // settled context, rather than a stale-random scratch buffer.
    const std::vector<llama_token> & block_out = argmax_canvas;

    // accumulate this block's finalized tokens; stop after a block that contains an EOG token
    generated.insert(generated.end(), block_out.begin(), block_out.end());
    for (int j = 0; j < canvas_length; ++j) {
        if (llama_vocab_is_eog(vocab, block_out[j])) { done = true; break; }
    }

    // 4. COMMIT (ENCODER phase): if another block follows, write the finalized canvas's plain
    // (non-self-conditioned, causal) K/V into the cache and advance the prefix pointer, so the
    // next block's canvas cross-attends to it. Skipped on the last block / on EOG.
    if (!done && block + 1 < max_canvases) {
        llama_set_causal_attn(ctx, true);
        llama_set_diffusion_decoder_phase(ctx, false);
        llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);
        batch.n_tokens = canvas_length;
        for (int j = 0; j < canvas_length; ++j) {
            batch.token[j]     = block_out[j];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = (j == canvas_length - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: canvas commit (encoder) decode failed at block %d\n", block);
            break;
        }
        n_past += canvas_length; // finalized canvas is now part of the read-only prefix
        LOG_INF("committed block %d -> n_past=%d\n", block, n_past);
    }
    } // end autoregressive block loop

    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen_start).count();

    llama_batch_free(batch);

    if (failed) {
        llama_free(ctx);
        return 1; // caller owns + frees the shared model and the backend
    }

    // the full denoised canvas (thought channel + response), for reference
    LOG_INF("\n=== generated canvas ===\n%s\n", common_detokenize(vocab, generated, false).c_str());

    // the model answers in a "<|channel>thought ... <channel|>" block followed by the response;
    // extract the final response (after the last channel-close), truncated at the first
    // end-of-generation token, and drop trailing duplicate sentences.
    llama_token chan_close = LLAMA_TOKEN_NULL;
    {
        auto t = common_tokenize(vocab, "<channel|>", false, true);
        if (t.size() == 1) chan_close = t[0];
    }
    const int n_gen = (int) generated.size();
    int start = 0;
    if (chan_close != LLAMA_TOKEN_NULL) {
        for (int j = 0; j < n_gen; ++j) if (generated[j] == chan_close) start = j + 1;
    }
    std::vector<llama_token> answer;
    for (int j = start; j < n_gen; ++j) {
        if (llama_vocab_is_eog(vocab, generated[j])) break;
        answer.push_back(generated[j]);
    }
    std::string ans = common_detokenize(vocab, answer, false);
    // drop a trailing exact-duplicate of the answer if the canvas repeated it
    {
        std::string s = ans;
        size_t h = s.find_first_not_of(" \n\t"); if (h != std::string::npos) s = s.substr(h);
        const size_t half = s.size() / 2;
        if (half > 0 && s.compare(0, half, s, s.size() - half, half) == 0) {
            ans = s.substr(0, half); // "X X" -> "X"
        }
    }
    LOG_INF("=== answer ===\n%s\n", ans.c_str());

    // generation timing (excludes model load + prompt prefill): wall-clock of the denoising
    // block loop, the canvas tokens produced, and effective throughput.
    const int n_canvas_tok = n_blocks_run * canvas_length;
    LOG_INF("=== perf ===\n");
    LOG_INF("generation: %d block(s), %d denoising steps, %d canvas tokens in %.2f s "
            "(%.1f canvas tok/s, %.3f s/step); answer tokens=%d\n",
            n_blocks_run, n_steps_total, n_canvas_tok, gen_s,
            gen_s > 0.0 ? n_canvas_tok / gen_s : 0.0,
            n_steps_total > 0 ? gen_s / n_steps_total : 0.0,
            (int) answer.size());
    if (log_step_timing) {
        LOG_INF("timing: decode enqueue %.3f s, sample/sync %.3f s, host loop %.3f s\n",
                decode_enqueue_s, sample_sync_s, host_loop_s);
    }

    llama_free(ctx);
    return 0; // caller owns + frees the shared model and the backend
}

// split a prompt file into one prompt per non-empty line (trailing CR/whitespace trimmed). Used in
// imatrix-collection mode so a single run denoises many diverse prompts, accumulating the imatrix.
static std::vector<std::string> split_prompts_by_line(const std::string & text) {
    std::vector<std::string> prompts;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t a = line.find_first_not_of(" \t");
        if (a == std::string::npos) {
            continue; // blank line
        }
        const size_t b = line.find_last_not_of(" \t");
        prompts.push_back(line.substr(a, b - a + 1));
    }
    return prompts;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();

    // diffusion config (env-overridable for quick CPU testing)
    // canvas_length is fixed at the trained block size (256); overriding it is for experiments only.
    const int   canvas_length = env_int("DG_CANVAS", DEF_CANVAS_LENGTH);
    const int   n_steps       = env_int("DG_STEPS", DEF_MAX_DENOISE_STEPS);
    // number of autoregressive canvas blocks = ceil(-n / canvas_length), i.e. -n is the total
    // number of tokens to generate (e.g. -n 256 -> 1 canvas, -n 512 -> 2, -n 1024 -> 4). The
    // model may stop earlier on an EOG token. DG_MAX_CANVASES overrides; default (no -n) is 1.
    const int   blocks_from_n = params.n_predict > 0 ? (params.n_predict + canvas_length - 1) / canvas_length : 1;
    const int   max_canvases  = std::max(env_int("DG_MAX_CANVASES", blocks_from_n), 1); // autoregressive blocks
    const float entropy_bound = ENTROPY_BOUND;

    // top-k host sampling (CLI flags; default = full softmax over the whole vocab):
    //   --top-k k                 : top-k logits per position for softmax/sample/self-cond (0 = full).
    //   --top-k-start/--top-k-end : anneal k from START (first/high-entropy step) to END (last step).
    //   --top-k-tail-correction   : exact full-vocab entropy (logsumexp) for the accept/stop signal,
    //                               instead of the under-estimating top-k entropy.
    // --top-k uses its own "0 = disabled" convention and is applied only when explicitly passed.
    const int topk_fixed = (params.sampling.user_sampling_config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K)
                         ? params.sampling.top_k
                         : 0;
    const int topk_start = params.diffusion.top_k_start;
    const int topk_end   = params.diffusion.top_k_end;
    const int topk_tail  = params.diffusion.top_k_tail_correction ? 1 : 0;

    // DG_DUMP_IMATRIX=<path>: collect an importance matrix from the real denoise (decoder) forward
    // passes and write it to <path> at the end. Unset (default) => no eval callback, no overhead,
    // behaviour identical to before. Prefer .gguf (else pass --output-format dat for legacy .dat).
    const char * imatrix_path = getenv("DG_DUMP_IMATRIX");
    bool         collect_imatrix = imatrix_path != nullptr && imatrix_path[0] != '\0';

    // DG_DUMP_SELFCOND=<path>: collect aligned self_cond_mlp (in, out) pairs from the denoise
    // (decoder) forward passes for isolated self_cond_mlp distillation. Writes <path>.in.bin,
    // <path>.out.bin, <path>.meta (see selfcond_dumper above). Unset (default) => no callback,
    // no overhead. Mutually exclusive with DG_DUMP_IMATRIX (one cb_eval slot); selfcond wins.
    const char * selfcond_path = getenv("DG_DUMP_SELFCOND");
    const bool   collect_selfcond = selfcond_path != nullptr && selfcond_path[0] != '\0';
    if (collect_selfcond && collect_imatrix) {
        LOG_WRN("diffusion-gemma: DG_DUMP_SELFCOND and DG_DUMP_IMATRIX both set; only one eval "
                "callback is supported — DG_DUMP_SELFCOND takes precedence, imatrix disabled\n");
        collect_imatrix = false;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    // Offload all layers to the GPU by default (when built with a GPU backend, e.g.
    // -DGGML_CUDA=ON). Pass -ngl N to limit offload, or -ngl 0 to force CPU. With a
    // CPU-only build this has no effect. (params.n_gpu_layers defaults to -1 = auto.)
    model_params.n_gpu_layers = params.n_gpu_layers >= 0 ? params.n_gpu_layers : 999;
    model_params.devices      = params.devices.data();
    model_params.use_mmap     = params.use_mmap;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }
    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: not a diffusion model\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const dg_config cfg = {
        /* .canvas_length   = */ canvas_length,
        /* .n_steps         = */ n_steps,
        /* .max_canvases    = */ max_canvases,
        /* .entropy_bound   = */ entropy_bound,
        /* .topk_fixed      = */ topk_fixed,
        /* .topk_start      = */ topk_start,
        /* .topk_end        = */ topk_end,
        /* .topk_tail        = */ topk_tail,
        /* .collect_imatrix  = */ collect_imatrix,
        /* .collect_selfcond = */ collect_selfcond,
    };

    // Build the list of prompts to run. Normal use = the single --prompt / -p (or -f file content).
    // In imatrix mode with a -f prompt file, each non-empty line is a separate prompt (a separate
    // denoise loop); the imatrix accumulates across all of them for a more representative matrix.
    const bool multi_prompt_mode = collect_imatrix || collect_selfcond;
    std::vector<std::string> prompts;
    if (multi_prompt_mode && !params.prompt_file.empty()) {
        prompts = split_prompts_by_line(params.prompt);
        if (prompts.empty()) {
            LOG_ERR("error: collection mode set but prompt file '%s' has no prompts\n", params.prompt_file.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        if (collect_selfcond) {
            LOG_INF("diffusion-gemma: self_cond collection over %zu prompt(s) from '%s' -> %s.{in,out}.bin\n",
                    prompts.size(), params.prompt_file.c_str(), selfcond_path);
        } else {
            LOG_INF("diffusion-gemma: imatrix collection over %zu prompt(s) from '%s' -> %s\n",
                    prompts.size(), params.prompt_file.c_str(), imatrix_path);
        }
    } else {
        prompts.push_back(params.prompt);
        if (collect_imatrix) {
            LOG_INF("diffusion-gemma: imatrix collection over 1 prompt -> %s\n", imatrix_path);
            LOG_INF("diffusion-gemma: for more coverage pass a -f prompt file (one prompt per line) "
                    "or re-run with --in-file %s to merge across runs\n", imatrix_path);
        } else if (collect_selfcond) {
            LOG_INF("diffusion-gemma: self_cond collection over 1 prompt -> %s.{in,out}.bin\n", selfcond_path);
            LOG_INF("diffusion-gemma: for more coverage pass a -f prompt file (one prompt per line)\n");
        }
    }

    // set up the self_cond dumper before any context is created: open the .in/.out append files
    // and read the stride/cap knobs. The dumper starts gated off; run_one_prompt enables it only
    // around the denoise decode. Mutually exclusive with the imatrix collector (DG_DUMP_SELFCOND won above).
    if (collect_selfcond) {
        const int sc_stride = env_int("DG_DUMP_SELFCOND_STRIDE", 1);
        const int sc_max    = env_int("DG_DUMP_SELFCOND_MAX_STEPS", 2000);
        g_selfcond.open(selfcond_path, sc_stride, sc_max);
        if (!g_selfcond.enabled_path) {
            LOG_ERR("error: DG_DUMP_SELFCOND set but could not open '%s.in.bin' / '%s.out.bin' for writing\n",
                    selfcond_path, selfcond_path);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        LOG_INF("diffusion-gemma: self_cond dumper: stride=%d (every Nth step) max_steps=%d\n",
                sc_stride, sc_max);
    }

    // set up the imatrix collector before any context is created: register the backend eval
    // callback (so it fires on the denoise decodes) and point save_imatrix() at the output path.
    // The collector starts DISABLED; run_one_prompt enables it only around the denoise decode.
    if (collect_imatrix) {
        common_params imat_params = params;
        imat_params.out_file   = imatrix_path;
        // chunk accounting uses n_ctx / n_parallel; the canvas denoise decodes a single
        // canvas_length ubatch per step, so model these as one "sequence" of canvas_length tokens.
        imat_params.n_parallel = 1;
        imat_params.n_ctx      = canvas_length;
        g_collector.set_params(imat_params);
        params.cb_eval           = ik_collect_imatrix;
        params.cb_eval_user_data = nullptr;
    }

    int rc = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts.size() > 1) {
            LOG_INF("\n=== prompt %zu/%zu ===\n", i + 1, prompts.size());
        }
        if (run_one_prompt(model, params, cfg, prompts[i]) != 0) {
            rc = 1;
            break; // still write whatever imatrix data was collected so far (below)
        }
    }

    // write the accumulated decoder-path imatrix (gguf or legacy dat per --output-format).
    if (collect_imatrix) {
        LOG_INF("diffusion-gemma: saving denoise-path imatrix to '%s'\n", imatrix_path);
        g_collector.save_imatrix();
    }

    // flush the self_cond .in/.out files and write the .meta sidecar (now that the step count is known).
    if (collect_selfcond) {
        g_selfcond.finalize();
        LOG_INF("diffusion-gemma: self_cond dumper wrote %d step(s) (n_embd=%lld n_tokens=%lld) to "
                "'%s.{in,out}.bin' (+ .meta)\n",
                g_selfcond.steps_written, (long long) g_selfcond.n_embd, (long long) g_selfcond.n_tokens,
                selfcond_path);
    }

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
