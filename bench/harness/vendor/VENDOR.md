# Vendored Dependencies

Pinned snapshots of upstream evaluation code. Failure-isolated from upstream merges.

## NVIDIA/RULER

- Repo: https://github.com/NVIDIA/RULER
- Pinned SHA: `ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13`
- Vendored at: 2026-04-26
- Vendored paths (subset):
  - `scripts/data/synthetic/` → `bench/harness/vendor/ruler/data/synthetic/`
  - `scripts/eval/synthetic/` → `bench/harness/vendor/ruler/eval/synthetic/`
- License: Apache-2.0 (preserved in source headers)

We import only the per-task generators (`niah.py`, `variable_tracking.py`,
`common_words_extraction.py`, `freq_words_extraction.py`, `qa.py`) plus the
metric dispatch table in `eval/synthetic/constants.py`. Our `pred_llama.py`
replaces RULER's `pred/pred_*.py`.

### Adapter notes

Upstream RULER scripts assume:
- A `transformers` tokenizer for length budgeting (we add a deferred-import
  shim in `prepare_ruler.py`; only loaded when actually preparing data).
- A pre-built dataset under `scripts/data/synthetic/json/<task>.jsonl`
  (we generate this on demand via `prepare_ruler.py`).
- An `args` namespace with `tokenizer_path`, `max_seq_length`, `num_samples`,
  `save_dir`. Our wrapper builds that namespace explicitly.

When upstream task generators reference NLTK/wonderwords for filler text,
they will be installed from `requirements.txt` when the first real run lands.
For the dry-run path no deps are needed.

## THUDM/LongBench

- Repo: https://github.com/THUDM/LongBench
- Pinned SHA: `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
- Vendored at: 2026-04-26
- Vendored paths:
  - `LongBench/metrics.py` → `bench/harness/vendor/longbench/metrics.py`
  - `LongBench/config/dataset2prompt.json` → `bench/harness/vendor/longbench/config/`
  - `LongBench/config/dataset2maxlen.json` → `bench/harness/vendor/longbench/config/`
- License: MIT (preserved)

### Adapter notes

`metrics.py` imports `jieba`, `fuzzywuzzy`, `rouge`:
- `jieba` only used by ZH scorers (`rouge_zh_score`, `qa_f1_zh_score`) —
  not needed for `longbench_en` profile.
- `fuzzywuzzy` used by `code_sim_score` (lcc, repobench-p) — required for EN.
- `rouge` used by `rouge_score` (gov_report, qmsum, multi_news) — required.

Our `score_longbench.py` only dispatches EN scorers, but the metrics file is
vendored unmodified to preserve attribution and ease ZH activation later.

## Refresh Policy

Manual refresh quarterly or on relevant upstream fix. On refresh:
1. Clone upstream at new SHA, diff vendored subset.
2. Update SHA above.
3. Re-baseline against `bench/harness/baselines/<model_tag>.json`.
4. Note rationale in commit message.

## Why Vendor Instead of Pip

- RULER and LongBench are not packaged on PyPI as runnable evaluators.
- Both repos couple tasks + scoring + their own `pred_*.py`. We replace the
  prediction layer; vendoring lets us strip what we don't need.
- Pinning the SHA keeps gate thresholds reproducible.
