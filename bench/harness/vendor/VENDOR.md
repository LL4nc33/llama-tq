# Vendored Dependencies — TODO

This directory will hold pinned snapshots of upstream evaluation code so the
harness is failure-isolated from upstream merges. None of this is checked
in yet; the skeleton ships with placeholder scorers.

## Targets

### NVIDIA/RULER
- Repo: https://github.com/NVIDIA/RULER
- Target SHA: **TODO — pin latest tagged release at vendor time**
- Path subset to vendor: `scripts/data/synthetic/`, `scripts/eval/synthetic/`
- Drop into: `bench/harness/vendor/ruler/`
- Replaces: nothing — we ship our own `pred_llama.py`. We import only the
  prepare/scoring modules.

### THUDM/LongBench
- Repo: https://github.com/THUDM/LongBench
- Target SHA: **TODO — pin latest tagged release at vendor time**
- Path subset to vendor: `metrics.py`, `config/dataset2prompt.json`,
  `config/dataset2maxlen.json`
- Drop into: `bench/harness/vendor/longbench/`

## Refresh Policy

Manual refresh quarterly or whenever a relevant upstream fix lands.
On refresh: bump SHA in this file, re-baseline against
`bench/harness/baselines/<model_tag>.json`, note in commit message.

## Why Vendor Instead of Pip

- RULER and LongBench are not packaged on PyPI as evaluation runners.
- Both repos couple tasks + scoring + their own `pred_*.py`. We replace
  the prediction layer; vendoring lets us strip what we don't need.
- Pinning the SHA means our gate thresholds stay reproducible.

## When Vendor Lands

Update `score_ruler.py` to import the real metric and remove the
placeholder substring scorer. Same for the LongBench equivalent
(not yet skeletoned).
