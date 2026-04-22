# Trellis v2 Phase-1 Harness

Standalone parametric sweep for the vtq2_2 / vtq3_2 trellis-quantizer
design. Produces MSE + encode-time data across configs. Results inform
the Phase-2 GPU port.

**Does not build with llama-tq main tree.** Self-contained for fast
iteration.

See [../../docs/plans/2026-04-17-trellis-v2-phase1-report.md](../../docs/plans/2026-04-17-trellis-v2-phase1-report.md)
for the full Phase-1 report, experiments, and Phase-2 candidates.

## Build

```bash
cd tests/trellis-phase1
# Option A: direct gcc
gcc -O3 -Wall -Wextra trellis_main.c trellis_common.c trellis_code.c trellis_encdec.c -o trellis_phase1 -lm

# Option B: cmake
cmake -B build
cmake --build build
```

## Run

```bash
# Synthetic N(0,1)
./trellis_phase1 --mode gauss --n 32768 --out results/gauss.csv

# Real V-weights (extract first via extract_v_samples.py)
./trellis_phase1 --mode real --data vcache.bin --n 32768 --out results/real.csv

# Non-Gaussian stress tests
./trellis_phase1 --mode laplace   --n 32768 --out results/laplace.csv
./trellis_phase1 --mode student5  --n 32768 --out results/student5.csv
./trellis_phase1 --mode vcache_real --n 32768 --out results/vcache_real.csv
```

## Extract real V-weights from a GGUF

```bash
PYTHONPATH=../../gguf-py python3 extract_v_samples.py \
    --model ~/models/your-model.gguf \
    --out /tmp/vcache.bin \
    --max-samples 65536
```

Writes float32 samples post-RHT that the harness consumes via `--mode real`.

## Config Matrix (current: Phase-1 best candidates)

Edit `trellis_main.c` to change. Format: `L, K, QK, beam, norm, group,
shared_d, code, label`.

Phase-1 best configs (see RUN5_NOTES.md for full table):

| Label                  | L  | K | QK  | G | shared_d | bpw   | MSE ratio |
|------------------------|----|---|-----|---|----------|-------|-----------|
| Q256_G4_sharedD        | 16 | 2 | 256 | 4 | yes      | 2.031 | 0.590     |
| K3_Q128_G4_sharedD     | 16 | 3 | 128 | 4 | yes      | 3.063 | 0.174     |

## Code Functions

| Name    | Source                     | Memory   | Best for        |
|---------|----------------------------|----------|-----------------|
| 3GAUSS  | Weyl hash + 3-byte CLT     | 0        | GPU-minimal     |
| TABLE   | Precomputed inv-Gaussian CDF| 2^L·4B  | post-RHT data (primary) |
| T5      | Precomputed inv-Student-t(5)| 2^L·4B  | heavy-tail data |

## Gates

- **MSE gate (synthetic Gaussian)**: trellis MSE ≤ 0.7 · Lloyd-Max 2-bit MSE (0.1175 → 0.0822)
- **Real-data gate**: MSE ratio on post-RHT V-weights within 10% of Gaussian ratio
- **PPL gate** (Run 2, separate, not yet done): vtq2_2 PPL ≤ vtq2_1 PPL on Qwen3.5-27B Dense

## Status

- [x] Scaffold
- [x] Code functions (3GAUSS, TABLE, T5)
- [x] Viterbi encoder (full + beam-pruned)
- [x] Bitshift decoder (iterative; GPU-parallel variant deferred to Phase 2)
- [x] Group chaining + shared_d
- [x] Rolling-buffer DP (low-memory)
- [x] QK=256 support
- [x] Real V-weight extractor (GGUF)
- [x] Synthetic stress tests (5 distributions)
- [x] Real-data sweeps (Qwen 0.8B + 27B)
- [x] bpw floor established: 2.031 bpw (2-bit) / 3.063 bpw (3-bit)

## Results at a glance

All MSE ratios are against Lloyd-Max 2-bit MSE baseline (0.1175 for
Gaussian). Real data is post-RHT V-weights from Qwen3.5-27B.

| Config               | bpw   | gauss ratio | real ratio | Phase-2 port? |
|----------------------|-------|-------------|------------|---------------|
| vtq2_1 (baseline)    | 2.500 | 1.00        | 1.00       | N/A (current) |
| Q256_G4_sharedD      | 2.031 | 0.59        | **0.59**   | **yes (2-bit)** |
| Q256_G1              | 2.125 | 0.55        | **0.55**   | alt 2-bit     |
| baseline_Q128_G2     | 2.188 | 0.57        | 0.57       | alt 2-bit     |
| K3_Q128_G4_sharedD   | 3.063 | 0.17        | **0.17**   | **yes (3-bit)** |
| K3_Q32_G4 (old)      | 3.625 | 0.21        | 0.20       | superseded    |
