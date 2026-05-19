# Layer 3 Q1 Spike — CUDA TQ FA-vec Symmetric K==V Support

**Date:** 2026-05-09
**Branch:** turboquant (test build run from `/workspace/llama-tq` @ `f477fc7a8`, dispatcher matches `1910c4180`)
**Model:** `qwen3.5-0.8b-q8_0.gguf` on the test rig, RTX 2060 dual-GPU (`-ngl 99 -mg 0 -fa 1 -c 512`)
**Question:** Does the CUDA TQ FA-vec dispatcher support symmetric K==V (both layers same TQ type), so we have a CUDA reference to validate Vulkan V1 against?

---

## Task 1 — Dispatcher Matrix (`ggml/src/ggml-cuda/fattn-tq.cuh:952-1032`)

The dispatcher is split into two **independent** constexpr functions: K-side (`get_vec_dot_KQ`) selects on `type_K` only, V-side (`get_dequantize_V`) selects on `type_V` only. There is no `(type_K, type_V)` joint table — any combination compiles as long as both halves are individually populated. Joint admissibility is enforced earlier, in the FA selector at `ggml/src/ggml-cuda/fattn.cu:324-360`.

### K-side (`get_vec_dot_KQ`, lines 952-980)

| K type | K-dispatcher entry | Notes |
|--------|--------------------|-------|
| F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16 | yes | upstream |
| KTQ1_1, KTQ2_1, KTQ3_1, KTQ4_1 | yes | TurboQuant K |
| **VTQ\*** | **NO** | static_assert fires — VTQ has no K-side vec-dot |

### V-side (`get_dequantize_V`, lines 982-1032)

| V type | V-dispatcher entry |
|--------|--------------------|
| F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16 | yes |
| KTQ1_1, KTQ2_1, KTQ3_1, KTQ4_1 | yes |
| VTQ1_1, VTQ2_1, VTQ3_1, VTQ4_1 | yes |
| VTQ2_2, VTQ3_2, VTQ4_2 | yes |
| VTQ2_3, VTQ3_3, VTQ4_3, VTQ3_V8 | yes |

### Joint admissibility (`fattn.cu:324-360`)

The vec-kernel selector explicitly comments **"VTQ types are V-cache only — always asymmetric K!=V"** (line 324). Concretely:

1. Line 334 rejects `K->type != V->type` unless `is_vtq_v` OR `(KTQ K + f16 V)`.
2. Lines 339-360 hard-whitelist K to: F32/F16/Q4_*/Q5_*/Q8_0/BF16/KTQ1-4_1. **Any VTQ type as K → falls into `default:` → returns `BEST_FATTN_KERNEL_NONE`.**
3. Line 380 (`!GGML_CUDA_FA_ALL_QUANTS`): with default build, only symmetric TQ, KTQ-K + standard-V, and KTQ-K + VTQ-V are compiled.

### Question answers

| Tuple | Defined? | Path |
|-------|----------|------|
| `(KTQ2_1, KTQ2_1)` | **YES** | symmetric KTQ; K=line 970, V=line 1000; passes line 334 K==V check; K whitelist OK |
| `(VTQ2_2, VTQ2_2)` | **NO** | VTQ2_2 absent from K-dispatcher; FA selector returns NONE before instantiation; runtime falls back to non-FA attention |
| `(KTQ2_1, VTQ2_2)` | **YES** | prod path; K=line 970, V=line 1015; passes line 334 because `is_vtq_v=true` |

---

## Task 2 — Empirical Tests on the test rig

### Test A — symmetric KTQ2_1 K==V

```
--cache-type-k ktq2_1 --cache-type-v ktq2_1
```

Tail of output:
```
> The capital of Austria is

| [Start thinking]
Thinking Process:

1.  **Analyze the Request:**
    *   Question: "llama_memory_breakdown_print: | memory breakdown [MiB] | total    free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - CUDA0 (RTX 2060)   | 11832 = 10340 + ( 365 =   296 +      15 +      53) +        1125 |
llama_memory_breakdown_print: |   - CUDA1 (RTX 2060)   | 11832 =  8413 + ( 975 =   466 +      10 +     498) +        2442 |
llama_memory_breakdown_print: |   - Host               |                   266 =   257 +       0 +       9                |


[ Prompt: 541.8 t/s | Generation: 173.6 t/s ]

Exiting...
===EXIT=0===
```

**Result:** clean exit 0, coherent generation, **FA-vec ON** (PP 541.8, TG 173.6 t/s — same magnitude as Test C prod baseline). Symmetric KTQ2_1 instantiates the vec kernel correctly.

### Test B — symmetric VTQ2_2 K==V

```
--cache-type-k vtq2_2 --cache-type-v vtq2_2
```

Tail of output:
```
> The capital of Austria is

|-\| [Start thinking]
Thinking Process:

1.  **Analyze the Request:**
    *   Question: "llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - CUDA0 (RTX 2060)   | 11832 = 9758 + ( 411 =   296 +      13 +     101) +        1661 |
llama_memory_breakdown_print: |   - CUDA1 (RTX 2060)   | 11832 = 7833 + (1020 =   466 +       9 +     544) +        2977 |
llama_memory_breakdown_print: |   - Host               |                  380 =   257 +       0 +     122                |


[ Prompt: 45.9 t/s | Generation: 21.7 t/s ]

Exiting...
===EXIT=0===
```

**Result:** exit 0, output coherent, but **FA-vec OFF — non-FA fallback active** (PP 45.9 = 12× slower than Test A; TG 21.7 = 8× slower than Test A). No GGML_ABORT was triggered because `fattn.cu:339-360` returns `BEST_FATTN_KERNEL_NONE` cleanly before any kernel template is instantiated, and llama.cpp transparently falls through to dequant-then-vanilla-attention. Functionally usable, but **not on the FA-vec path the V1 plan targets**, so it provides no CUDA reference for Vulkan FA validation.

### Test C — asymmetric prod (KTQ2_1 K + VTQ2_2 V)

```
--cache-type-k ktq2_1 --cache-type-v vtq2_2
```

Tail of output:
```
> The capital of Austria is

| [Start thinking]
Thinking Process:

1.  **Analyze the Request:**
    *   Question: "llama_memory_breakdown_print: | memory breakdown [MiB] | total    free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - CUDA0 (RTX 2060)   | 11832 = 10340 + ( 365 =   296 +      15 +      53) +        1125 |
llama_memory_breakdown_print: |   - CUDA1 (RTX 2060)   | 11832 =  8413 + ( 976 =   466 +      11 +     498) +        2442 |
llama_memory_breakdown_print: |   - Host               |                   266 =   257 +       0 +       9                |


[ Prompt: 533.5 t/s | Generation: 173.0 t/s ]

Exiting...
===EXIT=0===
```

**Result:** exit 0, coherent, **FA-vec ON** (PP 533.5, TG 173.0 — prod baseline confirmed).

### Summary

| Test | K | V | Exit | PP t/s | TG t/s | FA-vec? | Output coherent? |
|------|---|---|------|--------|--------|---------|------------------|
| A | KTQ2_1 | KTQ2_1 | 0 | 541.8 | 173.6 | **yes** | yes |
| B | VTQ2_2 | VTQ2_2 | 0 |  45.9 |  21.7 | **no (fallback)** | yes |
| C | KTQ2_1 | VTQ2_2 | 0 | 533.5 | 173.0 | **yes** | yes |

---

## Task 3 — Decision

**Outcome: variant of (C) — Symmetric works, but only for KTQ types.**

- Symmetric **KTQ K==V** is fully supported on the CUDA FA-vec path (Test A) — produces a valid reference for Vulkan V1.
- Symmetric **VTQ K==V** is *not* supported on FA-vec by design: VTQ blocks have no K-side vec-dot kernel and the FA selector excludes VTQ from the K whitelist. The runtime degrades to non-FA without aborting.

This nuance means the original A/B/C/D options collapse as follows:

- (A) "Both work symmetric on FA-vec" — **rejected** (VTQ symmetric falls back).
- (B) "Symmetric runtime-aborts, must extend dispatcher" — **rejected** (no abort; fallback exists).
- (C) "Works for one direction only" — **selected.** KTQ symmetric works on FA-vec; VTQ symmetric does not.

### Implication for Vulkan V1

The Vulkan V1 plan (Layer 3 POC, commit `d959f50bd`) ships **symmetric only** because the Vulkan pipeline-key design hard-rejects K!=V at `ggml-vulkan.cpp:15466`. Combined with this spike:

- V1 should ship **symmetric KTQ only** (e.g., KTQ2_1+KTQ2_1) for the first cut. CUDA FA-vec gives us a byte-exact reference for unit/integration testing (compare per-token logits Vulkan-vs-CUDA at the same KV-cache contents).
- V1 must **not** advertise symmetric VTQ as a supported configuration. The CUDA "reference" for symmetric VTQ is the non-FA fallback path, which is a different algorithm — not useful as a numerical oracle for the Vulkan FA-vec port.
- Asymmetric KTQ-K + VTQ-V (the prod path) remains out of scope for V1 *on Vulkan* — it requires lifting the V-side pipeline-key restriction. This is a Layer 4+ task and matches the spec-master plan as written.

### Concrete next-action recommendation for Layer 4

No CUDA dispatcher extension is required to unblock V1. Recommended sequence:

1. **V1 (Layer 4, ship now):** Vulkan FA-vec for symmetric `(KTQ2_1, KTQ2_1)`. Validate against CUDA Test-A path (FA-vec ON) using a fixed-prompt logit-diff harness.
2. **V1.5 (optional, before V2):** add symmetric `(KTQ3_1, KTQ3_1)` and `(KTQ4_1, KTQ4_1)` to Vulkan — same pipeline-key shape, only the dequant block-size differs. CUDA reference is free (Test A pattern).
3. **V2 (Layer 5):** lift the K!=V restriction in `ggml-vulkan.cpp:15466` to enable asymmetric KTQ-K + VTQ-V (prod parity). This is the larger pipeline-key refactor that the POC report flagged. CUDA reference: Test C path.
4. **Symmetric VTQ on Vulkan: do not pursue.** It would require porting a K-side vec-dot for VTQ blocks that does not exist on CUDA either; we would be inventing a kernel with no reference. If symmetric VTQ ever becomes interesting (it currently is not — VTQ exists *because* it is V-only-optimal), the right place to add the K-side dispatcher entry is `fattn-tq.cuh` first, not Vulkan.

### What this means for the Q1 risk

The POC's worry — "we may sink time into FA wiring with no CUDA reference" — is resolved: **for V1 (symmetric KTQ), the CUDA reference exists and works today**. The Vulkan port can proceed without prerequisite CUDA dispatcher work. The spec-master plan stands; only the wording around "symmetric supported types" should be tightened to "symmetric KTQ" rather than "symmetric TQ".
