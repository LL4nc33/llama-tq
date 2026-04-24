# llama-tq Autoresearch Loop

A Karpathy-style autonomous research loop for KV-cache quantization
experiments. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The loop is agent-ready: an LLM (or a human) proposes a single-parameter
change, the loop builds, measures, and scores. Keeps experiments that
beat the current best, reverts others.

## Scoring metric

```
score = ppl_delta_pct  +  0.5 * tg_slowdown_pct
```

- `ppl_delta_pct` — wikitext-2 PPL delta vs f16/f16 baseline on
  `Qwen3.6-35B-A3B-IQ2_XXS` at ctx=2048 / 5 chunks
- `tg_slowdown_pct` — `llama-bench -gen 256` throughput slowdown vs
  baseline (f16/f16)

Lower is better. Weight on TG is 0.5 because we already have KTQ_1 at
~1.4% PPL; any new design must not cost more than 2× that in throughput.

## Layout

```
autoresearch/
├── README.md              (this file)
├── baseline.json          (reference numbers: f16/f16, ktq2_1, ktq3_1, etc.)
├── experiments/
│   └── YYYY-MM-DD-HHMM/   (one dir per experiment)
│       ├── params.json    (what was changed)
│       ├── patch.diff     (code change applied)
│       ├── build.log      (build output)
│       ├── metrics.json   (ppl, tg, score)
│       └── decision.txt   (keep / revert + reason)
├── scripts/
│   ├── measure.sh         (run PPL + TG, emit metrics.json)
│   ├── run_experiment.sh  (apply patch → build → measure → score)
│   └── baseline.sh        (generate baseline.json)
└── agents/
    └── program.md         (instructions for the LLM agent)
```

## First-pass axes (from docs/plans/2026-04-24-ktq2-trellis-design.md)

- **Axis C** (recommended start): per-head codebook for K — smallest code
  change, leaves FA kernel untouched
- **Axis A**: warp-parallel Viterbi on single K vectors
- **Axis B**: streaming incremental trellis
- **Axis D**: outlier-channel split (shares v6 VTQ_OUT work)

## Target budget

15 min per experiment × 10-15 experiments per session = ~3 h for a
full overnight run. Each experiment must leave the tree in a buildable
state or auto-revert.

## Status

Infrastructure-only skeleton. First experiment axis TBD — seeding with a
trivial known-hyperparameter-scan to validate the loop before investing
in KTQ_2 kernel code.
