# Qwen3.5-122B-A10B Bench — Expert-Offload Sweep + Prod Deploy

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

## Production Deploy (on-llm-122b.service)

**Validated 5× (2026-04-23):** 14.06 ± 0.49 tok/s TG, 28.4 ± 2.3 tok/s PP @ 200k ctx.

```bash
/home/claude/llama-tq/build/bin/llama-server \
  -m /home/claude/models/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf \
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

**Bench täuscht:** `-ts 12,12` + `-ot` Layer 0-9 nutzt de-facto nur GPU0. "Dual-GPU" Bench-Configs liefen echt dual, waren aber langsamer wegen PCIe-Cross-Traffic bei nur 4k Compute-Buffer.

### Phase 2: Prod-Smoke @ 262k ctx mit TQ2_1

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

**Gewinner 19L (10+9):** +11% PP-Stabilität, +2% TG vs balanced 9+9. 0.9 GB GPU0-Headroom ist Prod-safe (0.3 GB crasht bei großen Prompts).

## Key Findings

1. **`--fit-target` Default 1024 MiB blockt asymmetrische Configs.** Workaround: `--fit-target 128`.

2. **PCIe-Asymmetrie messbar:** GPU0-Heavier spart x4-Cross-Traffic bei Expert-Outputs. Kein Gamechanger, aber +11% PP-σ-Stabilität.

3. **Bench-Tool-Gotchas:**
   - `-ctk ktq2_1 -ctv vtq2_1` funktioniert bei `-ngl 99`, **failed bei `-ngl 0`** (CPU-only path hat keinen TQ-Init — Issue low-prio).
   - `llama-bench -ot` Syntax: Single flag mit `;` Separator (Multi-Flag = Test-Varianten!)
   - `llama-server -ot` Syntax: Multiple Flags ODER single flag mit `,` Separator.

4. **KV-Size TQ2_1:** 200k = ~450 MB, 262k = ~590 MB. Delta nur 140 MB dank GQA(2) + TQ2_1.

5. **Physik-Ceiling:** 2.5 GB/Token Memory-Traffic, ~56 GB/s effektive Bandwidth (GPU+CPU-Mix) → theoretisch 22 tok/s max. Real 14 tok/s = **64% Effizienz**.

6. **Thinking-Mode:** `--reasoning off` kritisch für Prod (64 Token Reasoning-Overhead pro kurzer Antwort).

## Use-Case

**Passt:** Chat, Q&A, Reasoning mit moderaten Prompts.
**Passt nicht:** Claude-Code-Style (14 tok/s × 100k ctx = 2h/Response).

Für >20 tok/s: IQ1_M (Quality-Loss), DDR5 (2× Bandwidth), oder Single-GPU mit 24+ GB.

## Open Work

- **Activation-Profiling** — Hot-Expert-Detection könnte +10-20% TG bringen. Braucht Source-Patch in Expert-Selection.
- **llama-bench CPU-only TQ-Init Bug** — Issue #167, low-prio.
- **Shared-Experts-only GPU** — Qwen3.5 hat `expert_shared_feed_forward_length=1024` → jedes Token hits Shared-Expert. Isoliertes Pinning nicht getestet.

## Credit

- **distillery-claude:** Full sweep methodology, PCIe-aware final config, 5× statistical validation.
- **User (LL4nc33):** PCIe-aware Expert-Shift Idee — +11% PP vom balanced Baseline.
- **llamatq-claude:** TQ2_1 KV-Cache — Enabler für 200k ctx (sonst OOM).

Source: `LEGION/2026-04-23_2340_distillery_122b-deploy-complete-results.md`.
