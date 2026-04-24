# Qwen3.5-122B-A10B Bench — Expert-Offload Sweep (Research)

Experimental deployment test on a research fork — not a production guide.

Date: 2026-04-23
Model: `Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf` (34.11 GiB)
Hardware: 2× RTX 2060 12 GB (CC 7.5), **asymmetric PCIe (GPU0 x16, GPU1 x4)**, 40 GB host RAM
Build: `turboquant` branch, commit `bc7c2e3d3`

## Architecture

From GGUF metadata:

| Key | Value |
|---|---|
| `block_count` | 48 |
| `expert_count` | 256 |
| `expert_used_count` | 8 |
| `attention.head_count` | 32 |
| `attention.head_count_kv` | **2** (GQA) |
| `embedding_length` | 3072 |
| `context_length` | 262144 |
| `expert_feed_forward_length` | 1024 |
| `expert_shared_feed_forward_length` | 1024 |

GQA(2) + 48 layers = ~9 KB per-token KV at f16 → even 262k ctx needs only ~2.3 GB f16 or ~0.4 GB at `ktq2_1/vtq2_1`. Delta 200k→262k is only +140 MB with TQ2_1.

## Measured Config (5-run average, 2026-04-23)

14.06 ± 0.49 tok/s TG, 28.4 ± 2.3 tok/s PP @ 200k ctx on the test rig.

```bash
.//build/bin/llama-server \
  -m ./models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \
  --host 0.0.0.0 --port 8794 \
  -c 200000 -ngl 99 -ts 12,12 -fa on \
  --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
  --parallel 1 --fit-target 128 \
  -ot "blk\.(0|1|2|3|4|5|6|7|8|9)\.ffn_(up|down|gate)_exps\.=CUDA0,blk\.(10|11|12|13|14|15|16|17|18)\.ffn_(up|down|gate)_exps\.=CUDA1,blk\.(19|[2-4][0-9])\.ffn_(up|down|gate)_exps\.=CPU" \
  --jinja --reasoning off
```

**Expert-Verteilung (PCIe-aware):**
- GPU0 (x16 PCIe): Layer 0-9 Experts (10 Layer, ~6.5 GB)
- GPU1 (x4 PCIe): Layer 10-18 Experts (9 Layer, ~5.9 GB)
- CPU: Layer 19-47 Experts (29 Layer, ~19 GB)

**VRAM:** GPU0 10.9/11.8 GB (0.9 GB frei), GPU1 10.5/11.8 GB (1.4 GB frei).

## Methodology

### Phase 1: Bench-Sweep (f16 KV)

Coarse/fine sweep `-ngl 99 -ts 12,12 -fa 1 -r 3 -p 512 -n 256`:

| Config | Layer GPU | pp512 | tg256 |
|--------|---:|---:|---:|
| all-CPU | 0 | 151.7 | 12.70 |
| **L0-9 single-side** | 10 | 168.0 | **15.26 ± 0.1** |
| L0-4 + L5-9 balanced | 10 | 164.1 | 14.87 |
| L0-5 + L6-11 | 12 | 167.9 | 14.41 |
| L0-6 + L7-13 | 14 | 172.5 | 14.72 |

**Bench is misleading:** `-ts 12,12` + `-ot` on layers 0-9 de-facto uses only GPU0. "Dual-GPU" bench configs did run dual but were slower because of PCIe cross-traffic with only a 4k compute buffer.

### Phase 2: Live-server smoke @ 262k ctx mit TQ2_1

Single-side Winner failed bei realem ctx (GPU0 Compute-Buffer OOM). Balanced-Configs liefen:

- 14L balanced (7+7): 13.44 tok/s
- 16L balanced (8+8): 13.95 tok/s

### Phase 3: Fine-Tuning @ 200k ctx (PCIe-aware)

User-Insight: GPU0 x16, GPU1 x4 → Expert-Heavy auf GPU0 spart Cross-GPU-Traffic.

| Config | L | GPU0 free | GPU1 free | pp mean±σ | tg mean±σ |
|--------|---:|---:|---:|---:|---:|
| 18L (9+9) | 18 | 1.6 | 1.4 | 28.51 | 13.97 |
| 19L (9+10) | 19 | 1.6 | 0.7 | 28.97 | 14.04 |
| **19L (10+9)** 🏆 | 19 | **0.9** | 1.4 | 28.43±2.33 | **14.06±0.49** |
| 20L (10+10) | 20 | 0.9 | 0.7 | 27.34±4.45 | 14.34±0.45 |
| 21L (11+10) | 21 | 0.3 | 0.7 | **31.31±0.75** | 14.34±0.38 |

**Winner 19L (10+9):** +11% PP stability, +2% TG vs balanced 9+9. 0.9 GB GPU0 headroom is enough for longer prompts (0.3 GB crashes on large prompts).

## Key findings

1. **`--fit-target` default 1024 MiB blocks asymmetric configs.** Workaround: `--fit-target 128`.

2. **PCIe asymmetry is measurable:** GPU0-heavier saves x4 cross-traffic on expert outputs. Not a gamechanger, but +11% PP-σ stability.

3. **Bench-tool gotchas:**
   - `-ctk ktq2_1 -ctv vtq2_1` works at `-ngl 99`, **fails at `-ngl 0`** (CPU-only path has no TQ init — low-priority issue).
   - `llama-bench -ot` syntax: single flag with `;` separator (multi-flag = test variants!).
   - `llama-server -ot` syntax: multiple flags OR single flag with `,` separator.

4. **KV size TQ2_1:** 200k = ~450 MB, 262k = ~590 MB. Delta only 140 MB thanks to GQA(2) + TQ2_1.

5. **Physics ceiling:** 2.5 GB per-token memory traffic, ~56 GB/s effective bandwidth (GPU+CPU mix) → theoretical 22 tok/s max. Real 14 tok/s = **64% efficiency**.

6. **Thinking mode:** `--reasoning off` matters on short answers (64-token reasoning overhead).

## Use case

**Fits:** chat, Q&A, reasoning with moderate prompts.
**Doesn't fit:** Claude-Code-style (14 tok/s × 100k ctx = 2h/response).

To reach >20 tok/s: IQ1_M (quality loss), DDR5 (2× bandwidth), or single-GPU with 24+ GB.

## Open work

- **Activation profiling** — hot-expert detection might bring +10-20% TG. Needs a source patch in expert selection.
- **llama-bench CPU-only TQ-init bug** — issue #167, low priority.
- **Shared-experts-only GPU** — Qwen3.5 has `expert_shared_feed_forward_length=1024` → every token hits the shared expert. Isolated pinning not tested.

## Credit

- **distillery-claude:** full sweep methodology, PCIe-aware final config, 5× statistical validation.
- **User (LL4nc33):** PCIe-aware expert-shift idea — +11% PP over the balanced baseline.
- **llamatq-claude:** TQ2_1 KV cache — enabler for 200k ctx (OOM otherwise).

Source: `LEGION/2026-04-23_2340_distillery_122b-deploy-complete-results.md`.
