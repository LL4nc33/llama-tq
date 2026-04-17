# Trellis v2 Phase-1 Harness

Standalone parametric sweep for the vtq2_2 trellis-quantizer design.
Produces MSE + encode-time data across configs.

**Does not build with llama-tq main tree.** Self-contained for fast iteration.

## Build

```bash
cd tests/trellis-phase1
cmake -B build
cmake --build build
```

## Run

```bash
# Synthetic N(0,1) sweep
./build/trellis_phase1 --mode gauss --n 32768 --out results/gauss.csv

# Real V-cache sweep (needs vcache.bin from Task #93 dumper)
./build/trellis_phase1 --mode real --n 32768 --data vcache.bin --out results/real.csv
```

## Config Matrix (see trellis_main.c)

| Label          | L  | K | QK  | bpw  |
|----------------|----|---|-----|------|
| L8_K2_Q32      | 8  | 2 | 32  | 2.5  |
| L16_K2_Q32     | 16 | 2 | 32  | 3.0  |
| L16_K2_Q64     | 16 | 2 | 64  | 2.5  |
| L16_K2_Q128    | 16 | 2 | 128 | 2.25 |
| L8_K2_Q128     | 8  | 2 | 128 | 2.125|
| L12_K2_Q64     | 12 | 2 | 64  | 2.5  |
| L16_K3_Q32     | 16 | 3 | 32  | 3.5  |

## Gates

- **MSE gate** (Gaussian): trellis MSE ≤ 0.7 × Lloyd-Max 2-bit MSE (0.1175 → 0.0822)
- **Real-data gate**: real-V-cache MSE ratio close to Gaussian ratio
- **PPL gate** (Run 2, separate): vtq2_2 PPL ≤ vtq2_1 PPL on Qwen3.5-27B Dense

## Status

- [x] Scaffold (this file)
- [ ] Code functions (Task #90)
- [ ] Viterbi encoder (Task #91)
- [ ] Bitshift decoder (Task #92)
- [ ] Run 1 sweep (Task #94)
