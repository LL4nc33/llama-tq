# Autoresearch Loop

A Karpathy-style autonomous research loop for KV-cache quantization experiments. An agent (LLM or human) proposes a single-parameter change, the loop builds, measures, scores, and either keeps or reverts.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Scoring

```
score = ppl_delta_pct + 0.5 × tg_slowdown_pct
```

Lower is better. PPL is measured against the f16/f16 baseline on `Qwen3.6-35B-A3B-IQ2_XXS` at ctx=2048, 5 chunks. TG slowdown is `llama-bench -gen 256` throughput vs the same baseline. The 0.5× weight on TG reflects that the current KTQ_1 family already costs ~1.4 % PPL — any new design must not exceed roughly 2× that in throughput cost.

## Layout (suggested)

```
autoresearch/
├── baseline.json       reference numbers for f16/f16, ktq2_1, ktq3_1, …
├── experiments/
│   └── YYYY-MM-DD-HHMM/
│       ├── params.json   what was changed
│       ├── patch.diff    applied code change
│       ├── build.log
│       ├── metrics.json  ppl, tg, score
│       └── decision.txt  keep / revert + reason
└── scripts/
    ├── measure.sh         run PPL + TG, emit metrics.json
    ├── run_experiment.sh  apply patch → build → measure → score → keep/revert
    └── baseline.sh        regenerate baseline.json
```

The `autoresearch/` directory itself is git-ignored; experiment outputs are local-only by design (sample sizes are small, results are noisy without multi-seed averaging).

## Single-step protocol

1. Read current best score and the last few experiment entries.
2. Pick **one** axis and **one** knob value to change.
3. Write a minimal patch — change only what is necessary.
4. Run `run_experiment.sh` — handles build, measure, score, auto-revert.
5. If kept: log the insight. If reverted: log why the hypothesis failed.

## Hard constraints

- **Hardware:** Turing CC 7.5 only — no WGMMA, no FP8 tensor cores.
- **Memory:** Max 64 KB shared memory per SM.
- **Time:** A single experiment must fit in 15 minutes (build + measure).
- **Read-path:** Never modify the FA-kernel read path — encoder-only changes.
- **Build failures:** Auto-revert.
- **One knob at a time.** No "while I'm here" fixes mid-experiment.
- **No parallel runs** — they share the same GPU.

## What this loop is good for

- Hyperparameter sweeps (codebook sizes, block sizes, group sizes)
- Single-axis ablations (e.g. "does norm correction help on this V-type?")
- Tightly-scoped kernel-level changes with measurable PPL/TG impact

## What this loop is NOT for

- Multi-knob designs (use a hand-written design doc + targeted bench)
- Quality evaluations beyond wikitext PPL (use a separate downstream-eval matrix)
- Validating cross-model generalization (loop is single-model by construction)

## Lessons learned

The signal-to-noise ratio is the main bottleneck. PPL deltas under ~0.3 % at chunks=5 are within run-to-run noise, so the loop reliably distinguishes only changes that move PPL by ≥ 0.5 % or TG by ≥ 3 %. Smaller effects need either larger chunk counts (slower per experiment) or multi-seed averaging (also slower).

A 30-minute calibration measurement once killed a multi-week implementation effort by showing the underlying assumption did not hold for the target model family. The lesson: **measure first, build second.**
