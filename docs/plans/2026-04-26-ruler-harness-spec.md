# RULER + LongBench Evaluation Harness Spec

**Status:** SPEC ONLY
**Date:** 2026-04-26

## 1. Executive Summary

`llama-perplexity` measures token-level NLL on wikitext + 200-sample HellaSwag — neither covers long-context degradation nor task quality. Every Phase 5 spike currently ships on PPL + vibes.

**Decision:** RULER = primary harness (synthetic, deterministic, length-parametric). LongBench = secondary (real-task validation). Both run as **out-of-tree Python wrappers** driving llama-server `/completion` endpoint. NOT integrated into `llama-perplexity` (both score generated text, not logits).

**Total wrapper LOC:** ~500 LOC bash+python. Smallest viable run on 35B-A3B: ~3.2h wall-clock.

## 2. Tool Selection

| Option | Verdict |
|---|---|
| Extend llama-perplexity for RULER | Rejected — RULER scores generated strings against gold, not logits |
| lm-evaluation-harness LongBench | Considered, deferred — heavy deps |
| Native NVIDIA RULER repo | **Adopted (subset)** — vendored, replace pred_*.py with our pred_llama.py |
| Native THUDM LongBench repo | **Adopted (subset)** — vendored metrics.py |
| LongBench v2 | Defer to Phase 6+ |

## 3. Architecture / Data Flow

```
bench/harness/  (NEW)
├── prepare_ruler.py        → vendored NVIDIA/RULER prepare.py wrapper
├── prepare_longbench.py    → THUDM/LongBench dataset → JSONL
├── pred_llama.py           → POST /completion, write preds.jsonl
├── score_ruler.py          → wraps RULER eval/evaluate.py → summary.csv
├── score_longbench.py      → wraps LongBench metrics.py → results.json
├── run_gate.sh             → orchestrator + threshold checker
├── profiles/
│   ├── smoke.yaml          → 4 tasks × {4k,16k,64k} × n=100/100/50
│   ├── full.yaml           → 13 tasks × 6 lengths × n=200
│   └── longbench_en.yaml
├── baselines/
│   └── qwen36_35b_a3b_ktq21_vtq22.json
└── vendor/
    ├── ruler/              → snapshot of NVIDIA/RULER scripts/
    └── longbench/          → metrics.py snapshot
```

Wrapper external; failure-isolated from upstream merges.

## 4. Smallest Viable Test Set

### RULER smoke (per length)
4 tasks × 100 samples = 400 inferences/length:
- `niah_single_3` (NIAH single needle)
- `niah_multikey_2` (multi-key retrieval — kills naive sliding-window KV)
- `vt` (variable tracking 4 chains × 4 hops)
- `cwe` (common-word extraction — TQ-sensitive)

Lengths: 4k, 16k, 64k. Skip 8k/32k/128k for smoke.

### LongBench smoke
6 tasks × 50 samples = 300 inferences:
- narrativeqa, qasper, hotpotqa, gov_report, trec, lcc

Avg input ~6.7k tokens. Short-ctx generation quality signal.

### Full profiles (release gate)
- RULER: 13 tasks × 200 samples × {4k,8k,16k,32k,64k,128k} = 15600 inferences
- LongBench: 14 EN tasks full = ~3750 inferences

## 5. Time Budget on 35B-A3B (test-box)

Reference: ~30 t/s decode at 4k, ~12 t/s at 64k. Prefill: ~1500 t/s at 4k, ~800 at 64k.

### RULER smoke wall-clock

| Length | Per sample | × samples |
|---|---|---|
| 4k | 7.0 s | 12 min (n=100) |
| 16k | 19 s | 32 min (n=100) |
| 64k | 91 s | 76 min (n=50) |

**Total RULER smoke ≈ 2 h.** Acceptable nightly.

### LongBench smoke
~12 s/sample × 300 ≈ 1 h.

### Combined nightly gate ≈ 3.2 h. Schedule 02:00–05:00 Vienna.

## 6. Bench-Gate Thresholds

Calibrated against KTQ2_1 + VTQ2_2 baseline on Qwen3.6-35B-A3B (capture once, lock as `bench/harness/baselines/qwen36_35b_a3b_ktq21_vtq22.json`).

| Metric | Regression budget | Hard floor |
|---|---|---|
| RULER 4k avg | -1.0 pp | ≥ 92.0 |
| RULER 16k avg | -2.0 pp | ≥ 80.0 |
| RULER 64k avg | -4.0 pp | ≥ 55.0 |
| RULER cwe 64k (TQ-sensitive) | -5.0 pp | ≥ 40.0 |
| LongBench EN avg | -1.5 pp | ≥ 35.0 |
| Wikitext PPL | +1.5% | — |
| HellaSwag 200 | -0.5 pp | — |

```bash
gate "ruler_4k_avg"  ge 92.0  delta_ge -1.0
gate "ruler_16k_avg" ge 80.0  delta_ge -2.0
gate "ruler_64k_avg" ge 55.0  delta_ge -4.0
gate "ruler_cwe_64k" ge 40.0  delta_ge -5.0
gate "longbench_en_avg" ge 35.0 delta_ge -1.5
gate "wikitext_ppl" delta_le_pct 1.5
gate "hellaswag_200" delta_ge -0.5
```

Each spike PR pastes `run_gate.sh --spike <name>` output. Red gate = no merge.

## 7. LOC Budget

| Component | Lang | LOC |
|---|---|---|
| prepare_ruler.py | py | ~30 |
| prepare_longbench.py | py | ~40 |
| pred_llama.py | py | ~150 |
| score_ruler.py | py | ~40 |
| score_longbench.py | py | ~60 |
| run_gate.sh | bash | ~80 |
| baselines/*.json | json | ~50 |
| README.md | md | ~50 |
| **Total** | | **~500 LOC** |

Plus vendored snapshot of `RULER/scripts/data/synthetic/` (~800 LOC) and `LongBench/metrics.py` (~150 LOC) under `bench/harness/vendor/` with VENDOR.md SHA pin.

## 8. Risks

1. **Tokenizer mismatch.** RULER uses HF tokenizer for length budgeting; llama.cpp uses its own. Mitigation: `--retokenize-with` flag calling `/tokenize` endpoint. ~30 LOC.
2. **Chat template drift.** RULER's `template_type` lacks Qwen3 entry. Maintain `templates/qwen3.yaml`, re-baseline on upstream change.
3. **MoE non-determinism.** Qwen3.6-35B-A3B expert routing order-sensitive at high concurrency. Mitigation: `parallel=1`, `--cont-batching false` for evals.
4. **Format-specific gold answers.** RULER cwe expects whitespace-separated word list. Mitigation: normalize prediction (lowercase, strip punct) per task.
5. **LongBench language mix.** v1 has 5 ZH tasks. Smoke uses `longbench_en.yaml` only.
6. **`cache_prompt=false` cost.** Forces full prefill per sample. **Do not optimize this away** — KV reuse masks TQ regressions.
7. **Baseline staleness.** Each model swap → new baseline. Convention: `bench/harness/baselines/<model_tag>.json`.
8. **Upstream drift in vendored RULER/LongBench.** Pin commit SHA in VENDOR.md, manual refresh quarterly.

## 9. Implementation Order

1. Vendor RULER + LongBench metrics.py, pin SHAs
2. pred_llama.py + smoke test against existing live server
3. prepare_ruler.py + score_ruler.py for niah_single_3 only at 4k. End-to-end on Qwen3.5-0.8b first
4. Expand to 4 tasks × 3 lengths
5. Add LongBench (en subset)
6. Capture baseline on KTQ2_1+VTQ2_2 35B-A3B
7. Wire run_gate.sh, document in `docs/bench/harness.md`
8. Add to phase-5 PR template

**Estimated:** 1.5 dev-days for phases 1–5, 0.5 day for baseline + gating wiring. Total ~2 dev-days.

## Sources

- [RULER paper (arXiv:2404.06654)](https://arxiv.org/abs/2404.06654)
- [NVIDIA/RULER GitHub](https://github.com/NVIDIA/RULER)
- [LongBench paper (arXiv:2308.14508)](https://arxiv.org/abs/2308.14508)
- [THUDM/LongBench GitHub](https://github.com/THUDM/LongBench)
