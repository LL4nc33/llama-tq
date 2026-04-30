#include "models.h"

// LLM_ARCH_TALKIE — Talkie 1930 13B (talkie-lm/talkie-1930-13b-it).
//
// Folded layout from distillery's HF repackage:
//   - HeadGain folded into wq
//   - ActGain.attn folded into wo
//   - ActGain.mlp folded into ffn_down
//   - WeightGain folded into lm_head (model.output)
//   - RMSNorm tensors are synthesized "ones" (multiplicative no-op)
//
// Structurally retained:
//   - per-layer embed_skip_scale [1] applied at the END of each block:
//       cur = cur + embed_skip_scale[il] * RMSNorm(embd)
//
// Block order (matches dtestnyrr/talkie-1930-13b-it modeling_talkie.py):
//   x = x + attn(rms_norm(x))
//   x = x + mlp(rms_norm(x))
//   x = x + embed_skip_scale[il] * e_x      (e_x = rms_norm(embd), computed once)
//
// Note: qk_norm = RMSNorm on Q/K after RoPE (activation-side, not foldable); no
// runtime Q/K RMSNorm is applied here.

llm_build_talkie::llm_build_talkie(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // Talkie uses inverse rotation (sin sign-flipped vs NEOX). Equivalent to
    // standard forward rope with negated positions: cos(-θ)=cos θ, sin(-θ)=-sin θ.
    // Routes through the well-tested forward CUDA kernel (incl. fused
    // ROPE+VIEW+SET_ROWS for KV-cache writes), which the rope_back kernel
    // cannot do. Built once outside the layer loop.
    // I32 → F32 → neg → I32 (ggml_neg requires F32/F16).
    ggml_tensor * inp_pos_f32     = ggml_cast(ctx0, inp_pos, GGML_TYPE_F32);
    ggml_tensor * inp_pos_neg_f32 = ggml_neg(ctx0, inp_pos_f32);
    ggml_tensor * inp_pos_neg     = ggml_cast(ctx0, inp_pos_neg_f32, GGML_TYPE_I32);
    cb(inp_pos_neg, "inp_pos_neg", -1);

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Talkie's model.py uses F.rms_norm(x, (D,)) without explicit eps. The
    // EARLIER assumption that PyTorch resolves eps=None to torch.finfo(bf16).eps
    // (7.8125e-3) was WRONG — empirically verified by distillery via interactive
    // PyTorch: F.rms_norm(x, (D,), eps=None) and eps=1e-6 produce identical
    // first-value -0.906, while eps=7.8125e-3 produces -0.152 (8× too small).
    // PyTorch's F.rms_norm uses a much smaller internal default (effectively
    // ~1e-6 or smaller, NOT dtype-finfo-eps).
    //
    // The 8× under-scale at L0 (where embd raw_std ≈ 0.013, so eps=7.8e-3
    // dominates the rms² ≈ 0.00017 term) cascaded through all 40 layers and
    // produced byte-token gibberish. With eps=1e-6 (= llama.cpp default and
    // PyTorch's actual behavior) the trajectory matches reference.
    //
    // Fix verified end-to-end: Q4_K_M GGUF + this build now produces vintage
    // 1930s English on the canonical "If scientists discover life on other
    // planets," prompt.
    const float talkie_eps = 1e-6f;

    // Talkie: precompute RMSNorm(embedding) once, used as additive skip in every layer.
    ggml_tensor * e_x = ggml_rms_norm(ctx0, inpL, talkie_eps);
    cb(e_x, "talkie_e_x", -1);

    // Talkie's forward (model.py): x = self.embed(input_ids); x = F.rms_norm(x, ...);
    //                               for block in blocks: x = block(e_x, x, cos_sin)
    // The state passed into the FIRST block is the RMS-normed embedding, not raw.
    // Without this assignment the first attention sees raw-embed-magnitude inputs
    // through the attn-residual (inpSA = inpL = raw), which corrupts Q/K/V and
    // propagates through all subsequent layers. Bug discovered by distillery
    // by walking talkie.cpp side-by-side with talkie-lm/talkie src/talkie/model.py.
    inpL = e_x;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // pre-attn RMSNorm — inlined with talkie_eps (bf16 default 7.8125e-3).
        // attn_norm weight is "ones" (multiplicative no-op) but kept for generality.
        cur = ggml_rms_norm(ctx0, inpL, talkie_eps);
        if (model.layers[il].attn_norm) {
            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);
        }
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Talkie's apply_rotary_emb (model.py) uses an inverse rotation:
            //   y1 = x1*cos + x2*sin
            //   y2 = -x1*sin + x2*cos
            // vs standard NEOX (y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos).
            // Only difference is sign of sin. Since cos(-θ)=cos θ and sin(-θ)=-sin θ,
            // we can use the standard forward rope with negated positions to get
            // identical math. This routes through the well-tested forward kernel
            // (incl. CUDA fused ROPE+VIEW+SET_ROWS for KV-cache writes), which the
            // rope_back kernel cannot do (asserts forward=true in fused path).
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos_neg, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos_neg, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // qk_norm: RMSNorm on Q and K after RoPE (no learnable scale).
            // Activation-side normalization, cannot be folded into weights.
            Qcur = ggml_rms_norm(ctx0, Qcur, talkie_eps);
            Kcur = ggml_rms_norm(ctx0, Kcur, talkie_eps);

            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            // Slice e_x to match the truncated token set as well.
            e_x = ggml_get_rows(ctx0, e_x, inp_out_ids);
        }

        // attention residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // pre-ffn RMSNorm — inlined with talkie_eps.
        cur = ggml_rms_norm(ctx0, ffn_inp, talkie_eps);
        if (model.layers[il].ffn_norm) {
            cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
        }
        cb(cur, "ffn_norm", il);

        // SwiGLU FFN (Mistral-style)
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // ffn residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_residual", il);

        // Talkie embed_skip: cur = cur + embed_skip_scale[il] * e_x
        // embed_skip_scale is a [1] tensor that broadcasts over [D, n_tokens].
        //
        // Debug toggle: GGML_TALKIE_NO_SKIP=1 zeroes the skip term to test
        // whether embed_skip is dominating the residual stream and producing
        // a constant-bias logits collapse. Co-tracked with distillery via
        // LEGION 0205 mail.
        static const bool talkie_no_skip = []() {
            const char * env = std::getenv("GGML_TALKIE_NO_SKIP");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }();
        if (!talkie_no_skip) {
            ggml_tensor * skip = ggml_mul(ctx0, e_x, model.layers[il].embed_skip_scale);
            cb(skip, "embed_skip", il);
            cur = ggml_add(ctx0, cur, skip);
        }
        cb(cur, "l_out", il);

        cur = build_cvec(cur, il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final RMSNorm — inlined with talkie_eps.
    cur = ggml_rms_norm(ctx0, cur, talkie_eps);
    if (model.output_norm) {
        cur = ggml_mul(ctx0, cur, model.output_norm);
    }

    cb(cur, "result_norm", -1);
    cb(cur, "talkie_final_hidden", -1);  // pre-lm_head, layer-39-final-hidden compare
    res->t_embd = cur;

    // lm_head (WeightGain folded in)
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    cb(cur, "talkie_final_logits", -1);  // post-lm_head, byte-vs-BPE distribution check
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
