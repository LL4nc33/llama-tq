# 80B-A3B Low-Hanging Performance Wins

**Date:** 2026-04-24
**Target:** Qwen3-Next-80B-A3B, currently 25-28 tok/s TG @ 200k ctx, running on gpu00:8791
**Hardware:** 2× RTX 2060 12 GB (PCIe x16/x4 asymmetric), Ryzen 7 5700G (Zen 3, 8 Cores, DDR4-3200)
**Goal:** Push TG from ~28 to ~32-35 tok/s (+15-25%) within 1 day of work
**Constraint:** Running llama-server must not be broken — test on a second port, only swap when confirmed.

## Baseline

Current deploy (from `ps aux` snapshot):

```
llama-server -m Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf \
  --host 0.0.0.0 --port 8791 -c 200000 -ngl 99 -ts 12,12 -fa on \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --parallel 1 --fit-target 128 \
  -ot "blk\.(0|..|13)\.ffn_(up|down|gate)_exps\.=CUDA0,blk\.(14|..|27)\.ffn_(up|down|gate)_exps\.=CUDA1,blk\.(28|..|47)\.ffn_(up|down|gate)_exps\.=CPU" \
  --jinja --reasoning off
```

Split: 14 GPU0 + 14 GPU1 + 20 CPU layers. 25.73 tok/s TG (distillery baseline).

Memory-bandwidth ceiling: DDR4-3200 Dual = 40 GB/s real. Per-token CPU-traffic 0.75 GB → 53 tok/s hard-ceiling. Current 48% efficiency.

## Three Targets

Ordered by ease + independence. Each can be tested standalone without blocking the others.

### A) Huge Pages for mmap (GENERIC — any CPU-offloaded MoE benefits)

**What:** Add `MADV_HUGEPAGE` hint after `mmap()` in `src/llama-mmap.cpp`. Reduces TLB-pressure when random-access hits expert weights.

**Expected gain:** +3-8% TG on CPU-offloaded portion. Zero quality loss.

**Implementation:**

1. File: `src/llama-mmap.cpp`, function `llama_mmap::impl::init()` at ~line 460 (just after `mmap()` returns).
2. Add env-gated block:

```cpp
#ifdef __linux__
if (getenv("LLAMA_MMAP_HUGEPAGES")) {
    if (madvise(addr, file->size(), MADV_HUGEPAGE)) {
        LLAMA_LOG_WARN("madvise(MADV_HUGEPAGE) failed: %s\n", strerror(errno));
    } else {
        LLAMA_LOG_INFO("mmap: MADV_HUGEPAGE hint applied\n");
    }
}
#endif
```

3. Verify THP availability on gpu00: `cat /sys/kernel/mm/transparent_hugepage/enabled` must show `[madvise]` or `[always]`.

**Optional Phase A2 — Explicit HugeTLB (only if A1 shows <3% gain):**

Switch mmap flags to `MAP_HUGETLB | MAP_HUGE_2MB` with pre-allocated hugepages pool. Requires root or `CAP_IPC_LOCK`. Skip unless A1 underperforms.

**Test plan:**

1. Build branch `turboquant-hugepages` on gpu00 (fresh build, ~15 min).
2. Start on port **8795** (not the live 8791):

```bash
LLAMA_MMAP_HUGEPAGES=1 ./build/bin/llama-server \
  -m ~/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf \
  --host 0.0.0.0 --port 8795 -c 32768 -ngl 99 -ts 12,12 -fa on \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 --parallel 1 --fit-target 128 \
  -ot "..." --jinja --reasoning off
```

3. Smoke-test: 5 runs of tg128 via `curl POST /completion`, compare to baseline on 8791.
4. Verify hugepages actually used: `grep AnonHugePages /proc/PID/smaps | awk '{s+=$2} END {print s/1024 " MB"}'` — should show multi-GB after warmup.

**Decision gate:** if delta < 1%, skip to B. If ≥ 3%, keep on `turboquant-hugepages` and merge after B/C validation.

### B) Thread-pinning + NUMA-distribute

**What:** Zen 3 5700G has 2 CCXs (cores 0-3 and 4-7) with separate L3 caches. Cross-CCX traffic is slower than intra-CCX. Current `--threads` default spawns threads wherever the kernel schedules — often crossing CCX boundaries.

**Expected gain:** +3-5% TG. Zero-code-change — only CLI/env config.

**Implementation:**

No code changes. Test these configs on port 8795 one at a time:

**B1. `--numa distribute`:**

```bash
./build/bin/llama-server ... --numa distribute
```

Round-robin tensor-assignment across NUMA nodes. On 5700G this maps to CCX boundaries since it's single-socket.

**B2. Explicit threadsetaffinity via `taskset` + `--threads`:**

```bash
taskset -c 0-7 ./build/bin/llama-server ... --threads 8 --threads-batch 8
```

Pin all 8 cores explicitly. Default might be using SMT siblings which hurts memory-bound code.

**B3. Reduce decode threads, keep prefill high:**

```bash
./build/bin/llama-server ... --threads 4 --threads-batch 8
```

Memory-bound decode often doesn't scale past 4 threads. Might reduce L3 contention.

**Test plan:** Same 5-run tg128 for each config, A/B vs baseline.

**Prerequisites:** Check `numactl --hardware` output on gpu00. If single node (expected for 5700G), NUMA flags are no-ops but taskset still works.

**Decision gate:** Best config wins, merge into systemd service file.

### C) ngram Speculative Decoding

**What:** llama-server supports `--spec-type ngram-mod` — reuses n-grams from the existing context as "draft tokens", no separate draft model needed. Works especially well on repetitive output patterns (code, markdown lists, structured JSON).

**Expected gain:** +10-20% TG on repetitive outputs. Minimal-to-zero gain on purely creative free-form text. Zero VRAM overhead (no draft model).

**Implementation:**

No code changes. All plumbing exists (verified in speculative decoding spec #149). Just add flags:

```bash
./build/bin/llama-server ... \
  --spec-type ngram-mod \
  --draft-max 8 --draft-min 2 --draft-p-min 0.6
```

`--draft-max 8` = generate up to 8 draft tokens per verify cycle.
`--draft-min 2` = minimum before we skip drafting.
`--draft-p-min 0.6` = reject draft tokens below this probability.

**Test plan:**

On port 8795, run 3 benchmark scenarios and compare against baseline:

1. **Code generation:** "Write a Python quicksort with test cases" — expects high acceptance (>40%)
2. **Chat:** "Explain quantization in 3 paragraphs" — expects moderate acceptance (~15-25%)
3. **Listing:** "List 20 programming languages with a one-line description each" — expects very high acceptance (>60%)

Capture from `/v1/chat/completions` response timings:

- `prompt_tokens_per_second`
- `predicted_tokens_per_second`
- `draft_n` / `draft_n_accepted` (acceptance ratio)

**Decision gate:**

- If average accept-rate < 10% across all 3 scenarios: skip, doesn't pay.
- If ≥ 15% on at least one scenario: keep, merge into service as optional flag.
- If ≥ 30% on code/structured: make default for that endpoint.

## Test Sequence (single day)

Morning — A (Huge Pages):

1. Patch `llama-mmap.cpp`, commit to `turboquant-hugepages`, push, pull on gpu00, build. (1h)
2. Start test-server on 8795 with `LLAMA_MMAP_HUGEPAGES=1`. (5 min)
3. 5-run tg128 bench, compare to 8791 baseline. Log to `docs/bench-80b-hugepages-2026-04-24.md`. (30 min)
4. Decision: merge to `turboquant` if ≥ 3%, otherwise document and park.

Afternoon — B (Thread config), C (ngram-spec):

5. B1, B2, B3 sequential on 8795, same 5-run tg128. (1.5h total)
6. Pick winner, log.
7. C (ngram-spec) with 3 scenarios on 8795. (45 min)
8. Decision for C, log.

Evening — Merge + service update:

9. Winning combination → update `/etc/systemd/system/on-llm-80b.service` (assuming it exists; otherwise manual relaunch script).
10. Graceful restart on 8791 with new config.
11. 10-run validation against pre-change baseline.
12. Document final numbers in `docs/bench-qwen3-next-80b.md` and update README if gain > 5%.

## Risks

| Risk | Mitigation |
|---|---|
| Huge Pages silent no-op (THP disabled) | Verify `AnonHugePages` via smaps after start |
| Thread-pinning hurts prefill (PP down) | Measure PP and TG separately, keep `--threads-batch` high |
| ngram-spec regresses on free text | Only enable for specific endpoint/model if needed, per-request via `--spec-type` param |
| Live server interruption during test | Use port 8795 throughout; only touch 8791 at step 10 with verified config |
| GPU VRAM spike during parallel-load | Test server uses `-c 32768` not 200k during benches; full ctx only for final validation |

## Files Touched

- `src/llama-mmap.cpp` — Huge Pages patch
- New: `docs/bench-80b-hugepages-2026-04-24.md`
- New: `docs/bench-80b-threads-2026-04-24.md`
- New: `docs/bench-80b-ngram-spec-2026-04-24.md`
- Update: `docs/bench-qwen3-next-80b.md` (final summary)
- Update: `README.md` if gain > 5%
- Update: systemd unit or start-script (local on gpu00)

## Expected Cumulative Result

If all three hit their expected range:

- A: +3-8%
- B: +3-5%
- C: +10-20% on applicable workloads, +0% on others

**Stacked (independent):** +15-30% on repetitive/structured workloads, +6-13% on free-form text.

**TG target after this sprint:** ~32-36 tok/s TG on structured workloads, ~30-32 on free-form.
