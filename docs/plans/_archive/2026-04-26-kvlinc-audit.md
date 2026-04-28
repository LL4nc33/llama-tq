# KVLinC vs llama-tq PolarQuant Audit (2026-04-26)

## Verdict: PARTIAL overlap

We share the high-level recipe (Hadamard rotation on V, asymmetric K vs V quantization, per-head treatment, Q untouched), but every concrete primitive differs: their quantizer is uniform-affine INT2, ours is Lloyd-Max codebook; their group size is 128, ours is 32; they keep the last 128 tokens FP16, we don't; they have a trainable linear-correction adapter, we have a closed-form norm-correction scalar. Their headline novelty (the LinC adapter) is genuinely orthogonal to our stack and is the only port worth doing.

## Techniques we already have ✓

| KVLinC technique | Where it lives in llama-tq |
|---|---|
| Hadamard rotation of V before quantization | `ggml/src/ggml-cuda/turboquant.cuh` (VTQ pipeline), graph-level `self_v_rot` in build_graph |
| Per-head application of the rotation | VTQ block_size=32 quantizes per-head head_dim slices; `self_v_rot` is per-head |
| Asymmetric K vs V treatment | KTQ (K) full RHT + sb[4] sign bits; VTQ (V) fixed rotation, no sign bits |
| Q stays FP16 (no quantization on queries) | KTQ moves FWHT to Q-side at dot time; Q tensor never written to cache |
| Custom FA decode kernel that streams quantized blocks and dequantizes on-chip | `fattn-common.cuh` `vec_dot_KQ_ktq*`, `dequantize_V_*` |
| Bit-packing into 32-bit words | `block_ktq*_1` / `block_vtq*_1` packed layouts in `ggml-common.h` |
| 2-bit cache target | KTQ2_1 / VTQ2_1 (3.5 / 2.5 bpw) ship today |
| Sparse-V optimization | `Sparse V Dequant` already in v4+ (skips ~90% of positions at long ctx) |

## Techniques KVLinC has that we don't, ranked by win/effort

### 1. Linear Correction Adapter (LinC) — HIGH win, HIGH effort (the only real novelty)

The defining contribution. A trainable additive correction term computed in parallel with quantized softmax attention. Per Table 3 in the paper, the adapter alone contributes **~6.4–11% PPL improvement** at 2-bit (independent of, and stackable with, Hadamard).

Mechanism (their Eq. 5–6):
```
phi(X) = [softmax(X·W1), softmax(X·W2)] in R^D    (D=256)
S_n  = S_{n-1} + phi_k(K_n^FP16)^T · V_n          (recurrent state)
P_n  = P_{n-1} + phi_k(K_n^FP16)
Y_n  = (Sum exp(QK^T/sqrt(d))·V_q + phi_q(Q_n)·S_n) / (Sum exp(QK^T/sqrt(d)) + phi_q(Q_n)·P_n)
```
- W1, W2 in R^(d*128) learned per layer per head (~1% extra params)
- Constant memory in seq-len (state is 2 small matrices per head)
- O(L) compute (linear-attention add-on, no quadratic blowup)
- Trains in 2 hr (8B) on 4xH200, Alpaca calibration; LLM weights frozen

**Why win is real for us:** v7 has no error-correction layer. Norm-correction fixes Lloyd-Max bias inside one block but cannot recover inter-block softmax distortion. KVLinC adapter compensates exactly that, additive with Hadamard.

**Effort:** large but bounded.
- C++/CUDA: new linear-attention recurrent kernel — `ggml/src/ggml-cuda/fattn-common.cuh`, new `ggml/src/ggml-cuda/linc.cu`. Estimated ~600–1000 LOC.
- Adapter weights: GGUF tensor sidecar `blk.{layer}.attn_linc_{w1,w2}.weight`, loaded in `llama-model.cpp`.
- Calibration tooling: Python harness using transformers + llama.cpp Python binding. ~300 LOC.
- Risk: per (model, dataset) calibration. Doesn't fit zero-calibration philosophy. Ship as opt-in (`--cache-linc-adapter path.gguf`).

### 2. Recent-token FP16 window — MEDIUM win, LOW effort

KVLinC keeps the most recent 128 tokens in FP16 and only quantizes older positions. Recent tokens dominate softmax weight. Effective bpw moves from pure 2-bit to ~2.71-bit but PPL gain substantial.

**Effort:** small, ~400 LOC. Touch:
- `src/llama-kv-cache.cpp` — split cache into 128-slot f16 head + quantized tail; rotate on each new token
- `ggml/src/ggml-cuda/fattn-common.cuh` — FA needs to attend to two K/V types in one kernel
- `common/arg.cpp` — `--cache-fp16-window N`

### 3. Channel-wise scaling for K — LOW-MEDIUM win, MEDIUM effort

KVLinC's optimal axis split is `Q_C(K)` (channel-wise scale) and `Q_T(V·H)` (token-wise). Our FWHT design forces per-block scaling. Cannot meaningfully adopt without abandoning RHT. **Skip.**

### 4. Uniform-affine quantizer — UNKNOWN win, HIGH effort

KVLinC uses uniform INT2 because they don't have RHT (no Beta marginal guarantee). Our Lloyd-Max is provably optimal for our pipeline. **Skip.**

### 5. Group size 128 instead of 32 — LOW win, HIGH effort

Already on roadmap as low priority. **Skip.**

### 6. No random sign flips — NEGATIVE win, N/A

Would break our Beta(15.5, 15.5) codebook optimality. **Skip.**

## Recommendation

**Port: #1 (LinC adapter) and #2 (FP16 recent-token window).**
**Skip: #3, #4, #5, #6.**

If only one: do **#2 first** (low risk, clear win, no calibration). Tackle #1 if PPL becomes the limiting factor and you accept a calibration step.

## Concrete file:line landing zones

| Port | Files |
|---|---|
| LinC adapter weights | `gguf-py/gguf/constants.py`, `src/llama-model-loader.cpp`, `src/llama-model.h` |
| LinC kernel | `ggml/src/ggml-cuda/linc.cu` (new, ~800 LOC) |
| LinC FA hook | `ggml/src/ggml-cuda/fattn-common.cuh` after VKQ accumulation |
| LinC calibration tool | `tools/linc-calibrate/` (new) |
| LinC CLI flag | `common/arg.cpp` — `--cache-linc-adapter <path.gguf>` |
| FP16 recent-token cache | `src/llama-kv-cache.cpp` storage + `--cache-fp16-window N` |
| FP16 window FA dispatch | `ggml/src/ggml-cuda/fattn-common.cuh` — `(f16_head, ktq_tail)` mixed pairs |
| Docs | `docs/turboquant.md` Roadmap + new `docs/turboquant-linc.md` |

## Sources

- [KVLinC arxiv abstract](https://arxiv.org/abs/2510.05373)
- [KVLinC HTML](https://arxiv.org/html/2510.05373v1)
- [KVLinC OpenReview](https://openreview.net/forum?id=FkaDML963W)
- [TurboQuant base (Zandieh et al.)](https://arxiv.org/abs/2504.19874)
