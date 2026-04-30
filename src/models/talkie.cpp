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

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Talkie: precompute RMSNorm(embedding) once, used as additive skip in every layer.
    ggml_tensor * e_x = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
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

        // pre-attn RMSNorm (weight is "ones" but kept multiplicative for generality)
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
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
            // The only difference is the sign of sin. ggml_rope_ext_back is the
            // backward kernel which exactly applies sin_sign = -1.0 on top of the
            // NEOX layout (pairs at (i, i+n_dims/2)), so it computes Talkie's
            // forward rotation. Layout/freq math is otherwise identical.
            Qcur = ggml_rope_ext_back(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext_back(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // qk_norm: RMSNorm on Q and K after RoPE (no learnable scale).
            // Activation-side normalization, cannot be folded into weights.
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);

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

        // pre-ffn RMSNorm
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
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

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head (WeightGain folded in)
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
