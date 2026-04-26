# llama-tq vs upstream llama.cpp — A/B Benchmark

Date: 2026-04-26
Hardware: gpu00 — 2× RTX 2060 12GB (CC 7.5), CUDA, FA on
Methodology: `llama-bench -p 512 -n 128 -ngl 99 -fa 1 -r 3` (sequential, GPU verified clean between runs)

---

## 1. Build Info

| Item | Value |
|------|-------|
| llama-tq SHA | `6e50fc701` (turboquant branch) |
| upstream SHA | `0c6ee1cad` (master, 2026-04-26) |
| CUDA arch | 75 (Turing, RTX 2060) |
| Build flags | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release` |
| FA | enabled (`-fa 1`) |
| Repetitions | r=3 (main), r=2 (smoke) |

---

## 2. Smoke Sanity — Qwen3.5-0.8B-Q8_0

Sanity check: f16/f16 cache. Both engines should produce comparable numbers.

| Engine | KV cache | pp512 (t/s) | tg128 (t/s) |
|--------|----------|------------:|------------:|
| upstream | f16/f16 | 7375.67 ± 25.46 | 185.03 ± 0.18 |
| llama-tq | f16/f16 | 6976.33 ± 2.32  | 182.41 ± 0.04 |
| Δ        |          | **−5.4%**       | **−1.4%**    |

Smoke result: llama-tq is within 5–6% of upstream on a tiny model where the TQ codepath has no model-specific optimizations (no MoE prefetch, no FA-tuned kernels for this size). No catastrophic regression — proceed.

---

## 3. Main Benchmark — Qwen3.6-35B-A3B-UD-IQ2_XXS (10.01 GiB, 34.66 B params)

| ID | Engine    | KV cache       | KV bpw¹ | pp512 (t/s)        | tg128 (t/s)      | PPL (4 chunks)       |
|----|-----------|----------------|--------:|-------------------:|-----------------:|---------------------:|
| A  | upstream  | f16/f16        |   16.0  | 1182.32 ± 4.33     | 74.15 ± 0.18     | 5.8806 ± 0.4598      |
| B  | upstream  | q8_0/q8_0      |    8.5  | 1173.28 ± 3.22     | 72.42 ± 0.12     | 5.9015 ± 0.4626      |
| C  | llama-tq  | f16/f16        |   16.0  | 1012.87 ± 1.71     | 73.24 ± 0.22     | 5.8794 ± 0.4603      |
| D  | llama-tq  | q8_0/q8_0      |    8.5  | 1007.42 ± 0.84     | 70.71 ± 0.17     | —                    |
| E  | llama-tq  | ktq2_1/vtq2_2  |    ~3   | 1008.15 ± 1.96     | 72.21 ± 0.11     | 5.9764 ± 0.4693      |
| F  | llama-tq  | ktq3_1/vtq3_2  |    ~4   | 1008.70 ± 0.67     | 72.01 ± 0.22     | —                    |

¹ Approx KV bits-per-weight per element. KTQ/VTQ values reflect TurboQuant v5 packed structs.

### Headline observations (35B):

1. **PPL @ ktq2_1+vtq2_2 vs upstream f16: +1.6%** (5.9764 vs 5.8806) — well below "user-perceptible" threshold for a ~5× KV compression.
2. **TG @ ktq2_1+vtq2_2 vs upstream f16: −2.6%** (72.21 vs 74.15 t/s) — small cost, big VRAM win at long context.
3. **PP @ llama-tq vs upstream: −14.3%** (1008 vs 1182 t/s) — TQ branch has carried regression on prompt processing across all configs; pp is bottlenecked by something other than the KV-cache codepath (likely MoE-routing or FA dispatch). Worth investigating but not blocking for KV-cache value prop.
4. **TG @ llama-tq f16/f16 vs upstream f16/f16: −1.2%** (73.24 vs 74.15) — TG is flat across engines.
5. **TG @ ktq3_1+vtq3_2 ≈ ktq2_1+vtq2_2** (72.01 vs 72.21) — TQ overhead is dominated by FWHT/dequant not bpw, so going wider in KV bits buys you ~nothing in latency. Use the smallest bpw your PPL tolerates.

---

## 4. Comparison — Qwen3.6-27B-UD-IQ2_XXS (8.73 GiB, 26.90 B params, dense)

| ID | Engine    | KV cache       | pp512 (t/s)      | tg128 (t/s)    |
|----|-----------|----------------|-----------------:|---------------:|
| A  | upstream  | f16/f16        | 417.69 ± 0.05    | 15.64 ± 0.02   |
| B  | upstream  | q8_0/q8_0      | 413.00 ± 0.31    | 15.54 ± 0.02   |
| C  | llama-tq  | f16/f16        | 408.12 ± 0.40    | 15.58 ± 0.02   |
| D  | llama-tq  | q8_0/q8_0      | 404.23 ± 0.88    | 15.32 ± 0.01   |
| E  | llama-tq  | ktq2_1/vtq2_2  | 404.36 ± 0.42    | 15.43 ± 0.02   |

### 27B observations:

1. Dense 27B is heavily TG-bound at ~15.6 t/s on this hardware (memory-bandwidth limited, not compute).
2. **TG @ ktq2_1+vtq2_2 vs upstream f16: −1.3%** (15.43 vs 15.64) — even smaller cost than 35B.
3. **PP @ llama-tq vs upstream f16: −2.3%** (408 vs 417) — much smaller PP gap than the 35B MoE case, suggesting the 35B PP regression is MoE-specific.

---

## 5. Honest Caveats

- **PPL is 4 chunks × 512 tokens (= ~2K tokens of wikitext-2)**. This is enough to surface gross numerical bugs, not enough for a publishable PPL number. The ±0.46 confidence interval is much wider than the 0.10 PPL gap between configs — meaning the PPL ranking between f16, q8, and ktq2 is **not statistically distinguishable at this sample size**. Run `--chunks 100+` for publication-grade numbers.
- **No KV-memory column in the table.** llama-bench doesn't print the KV-cache footprint per-config. Theoretical KV bytes/token at 35B (n_kv_heads × head_dim × n_layers × bytes): f16 ~204 KB/tok, q8 ~108 KB/tok, ktq2_1+vtq2_2 ~38 KB/tok (~5.3× vs f16). At 200K context as deployed, that's roughly 41 GB → 7.6 GB — the actual reason for KTQ existing.
- **PP regression on MoE (35B) is real and reproducible** — −14% across all KV configs (including f16/f16, which has nothing to do with TurboQuant). This is independent of TQ types and likely lives in the MoE codepath that diverged on the turboquant branch. Track separately.
- **Smoke result (Qwen3.5-0.8B) shows −5% pp** even on a tiny dense model. Some constant overhead exists in the TQ build beyond MoE. Worth profiling.
- **r=3** repetitions per config; standard deviations on tg128 are <0.5% so single-run variance isn't muddying the comparison.

---

## 6. Headline Summary

> **TurboQuant v5 (ktq2_1+vtq2_2) on Qwen3.6-35B-A3B costs −2.6% TG and +1.6% PPL versus upstream f16/f16 — buying ~5× KV-cache compression that makes 200K context viable on 24GB. The −14% PP regression on MoE is unrelated to TQ and lives elsewhere on the turboquant branch.**

---

## 7. Re-bench post-fattn-tq fix (2026-04-26)

**Hypothesis:** Commit `3c2042ea4` ("perf(fa): extract TQ helpers to fattn-tq.cuh — fix register pressure on pure-f16 kernels") would recover the −14% PP regression on MoE 35B by removing TQ-specific helpers from the f16 kernel TUs (reducing register pressure / spills).

**Setup:** Same hardware, same flags as Section 3. Sequential, GPUs verified idle between runs.

| ID | Engine    | KV cache       | pp512 (t/s) BEFORE fix | pp512 (t/s) AFTER fix | Δ pp512 | tg128 BEFORE | tg128 AFTER | Δ tg128 |
|----|-----------|----------------|-----------------------:|----------------------:|--------:|-------------:|------------:|--------:|
| A  | upstream  | f16/f16        | 1182.32 ± 4.33         | 1181.07 ± 3.80        | −0.1%   | 74.15        | 73.94       | −0.3%   |
| C  | llama-tq  | f16/f16        | 1012.87 ± 1.71         | 1010.30 ± 21.33       | −0.3%   | 73.24        | 73.28       | +0.1%   |
| E  | llama-tq  | ktq2_1/vtq2_2  | 1008.15 ± 1.96         | 1014.13 ± 2.07        | +0.6%   | 72.21        | 72.27       | +0.1%   |

### Verdict: fix did NOT recover the regression

- llama-tq f16/f16 pp512 = **1010.3 t/s** post-fix vs upstream **1181.1 t/s** = **−14.5%** (was −14.3% before fix)
- ktq2_1/vtq2_2 pp512 = 1014.1 t/s — within noise of pre-fix 1008.2
- tg128 unchanged across the board

The fattn-tq.cuh extraction was a clean refactor (fewer includes per TU, no functional change) but did not reduce register pressure enough to move the needle on this MoE workload. The −14% PP gap on Qwen3.6-35B-A3B remains attributable to the MoE codepath divergence (or another non-FA bottleneck) on the turboquant branch.

### XQuant Phase 1 sanity (test-xktq-roundtrip)

```
FAIL scenario 1: identical inputs should reconstruct identically
[scenario 1] identical inputs — KTQ vs XKTQ-paired MSE: 7.018e-03 (expect ~0, fp16 scale noise)
```

XKTQ-paired path diverges from solo KTQ even with identical K/V inputs (MSE 7e-03). This is a Phase 1 stub — paired packing is not yet a true superset of solo. Not a runtime crash, but the foundation is not numerically clean yet. Track in XQuant Phase 1 follow-up.

### Next investigation directions for the −14% PP gap

1. Isolate non-FA contribution: bench with `-fa 0` on llama-tq vs upstream — if gap persists, FA is exonerated.
2. MoE routing path: diff `ggml_cuda_op_mul_mat_id` and surrounding TUs between turboquant and master.
3. nsys trace one pp512 step on each engine; look for kernel time deltas >1ms in non-attention ops.
