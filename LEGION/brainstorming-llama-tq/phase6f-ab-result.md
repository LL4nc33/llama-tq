# Phase 6f A/B Result (2026-04-27)

**Verdict:** Code-complete, **bench shows no measurable win.** Phase 6f
"hot-expert L3 prefetch" did not deliver the projected +25-40% TG.

## Test Setup

- Hardware: gpu00 (Ryzen 7 3700X, DDR4-3200, 2× RTX 2060)
- Model: Qwen3-Next-80B-A3B-Instruct UD-IQ2_XXS (24.41 GiB)
- Offload: 46/48 expert layers on CPU (`-ot blk\.([2-9]|1[0-9]|2[0-9]|3[0-9]|4[0-7]).ffn_.*_exps\.weight=CPU`)
- Bench: `llama-bench -p 0 -n 64 -r 3 -t 12`
- Hotness: top-20 per layer, 48 layers, mean dispatch share 0.55-0.61
- OMP: WAIT_POLICY=active, PROC_BIND=close, PLACES=cores

## Numbers

| Run | Baseline (t/s) | With hotness (t/s) | Δ |
|---|---|---|---|
| 1 | 24.59 ± 2.19 | 24.17 ± 1.82 | -1.7% |
| 2 | 25.03 ± 0.54 | 24.74 ± 1.19 | -1.2% |

**Within ±2% noise band, no detectable speedup.** Mostly small regressions.

Both confirm hotness IS loaded:
```
expert_hotness_load: loaded expert hotness profile '/tmp/expert-hotness-80b.json' (n_expert=512, n_layers=48, top_k=20)
expert_hotness_install_cpu: installed expert hotness into CPU backend (48 layers, top_k=20)
```

## Why Lever A Underperformed

Several plausible causes; not yet diagnosed:

### 1. Prefetch timing too tight
The hook fires AFTER `ggml_barrier(threadpool)` and immediately before the
`for cur_a` loop. By the time `__builtin_prefetch(t0)` is queued, the mat-vec
is already racing through the same cache lines. Software prefetch needs 50-200ns
lead time to be useful.

**Fix idea:** issue prefetch in the previous tensor's forward (one-step lookahead).
Requires graph-level scheduling.

### 2. HW prefetcher already covers this case
The existing line-by-line prefetch loop at lines 1675-1684 inside the per-`cur_a`
iteration already covers next-active-expert. If HW prefetcher detects the
sequential pattern, our top-20 SW prefetch is redundant.

### 3. L3 thrashing
10 MB hot working set + FA scratch + KV reads + norm + residual on the same CCX
may evict prefetched lines faster than mat-vec consumes them.

**Diagnostic:** `perf stat -e LLC-loads,LLC-load-misses` would reveal this.

### 4. Bottleneck not bandwidth
Decoding at 25 t/s on 80B-IQ2 = ~150 ms per token → ~16 ms per offloaded layer.
Offloaded expert weights per token: 10 active experts × 0.5 MB × 20 layers ≈ 100 MB,
which at 40 GB/s DDR4 = 2.5 ms. The other 13.5 ms is FA, residual, dequant, PCIe
back to GPU. Bandwidth wasn't the dominant cost in this regime.

### 5. Calibration set bias
The 80B hotness was profiled on wikitext-2 only (256 tokens × 48 layers).
A ~256-token sample of news prose may not predict expert dispatch on
mixed agentic/code workloads where bench operates.

## What Ships Anyway

The Phase 6f infrastructure stays in the fork:

- `common/router-profile.{h,cpp}` — extended with topk-IDs (`tag='K'`)
- `common/expert-hotness.{h,cpp}` — JSON loader + CPU backend installer
- `ggml/include/ggml-cpu.h` — `ggml_cpu_set_expert_hotness()` public API
- `ggml/src/ggml-cpu/ggml-cpu.c` — process-global table + name-parse + prefetch loop
- `tools/profile-router.py` — `--mode hotness` analyzer producing `expert-hotness.json`
- `tools/llama-bench/llama-bench.cpp` — env-var pickup
- `--expert-hotness <path>` CLI flag in common/arg.cpp

This is **opt-in** (zero overhead when flag not passed). Future work that
reuses this infrastructure:

- Phase 6f-v2: graph-scheduler-aware prefetch with one-step lookahead
- Phase 6g: profile-guided per-layer regex quant override (Path C, see phase6g spec)
- Phase 6h: profile-guided gating bias (static prune via gate-weight `-inf`)

## Recommendation

Don't enable Phase 6f in production deploy scripts. Keep code as research
infrastructure. Diagnose root cause via `perf stat` + nsys before the next
attempt.

The profiler infrastructure (Phase 6a) IS valuable on its own — it killed
the adaptive-k thesis with 30 min of measurement, and fed Phase 6g's design.
