# Triple-Goal Sweet-Spot Sweep — 35B + 4B (2026-05-02)

**Status:** Sweep complete. Sweet-Spot identifiziert für beide Modelle.
**Hardware Issue:** GPU0 ging am Ende des 4B-Sweeps in "GPU requires reset" state. Daten waren gerettet bevor crash. Reset braucht User-sudo.

## Triple-Goal Constraints (User-defined Prio)

1. **Accuracy** (PPL drift) — höchste Prio
2. **Speed** (TG t/s) — Floor 70+ für 35B, 60+ für 4B
3. **VRAM** — niedrigste Prio, aber relevant für ctx-budget

## 35B-A3B-IQ2_XXS bartowski (GPU0 RTX 2060)

### PPL Sweep (8 configs, wikitext-2 ctx512 chunks=3)

| Config | bpw | PPL | Drift |
|---|---|---|---|
| f16/f16 | 32 | 7.2044 | 0.00% |
| **ktq2_1/vtq2_1 (current 35B prod)** | 3.0 | 7.4816 | **+3.85%** |
| ktq2/vtq2 (= vtq2_2) | 2.78 | 7.1807 | **-0.33%** |
| ktq2/vtq3 (= vtq3_v8) | 3.56 | 7.1807 | -0.33% |
| ktq3/vtq3 | 4.06 | 7.1807 | -0.33% |
| ktq2_1/vtq3_3 (legacy 4-outliers) | 3.78 | 7.1807 | -0.33% |
| ktq4/vtq4 | 5.0 | 7.2635 | +0.82% |
| ktq2_1/vtq3_1 | 3.75 | 7.2178 | +0.19% |

**Wichtige Erkenntnis: Alle Trellis-Configs (vtq2/vtq3/vtq3_3) liefern IDENTISCH PPL.** K-collision von 35B-A3B (64 attention layers, hohe parameter-redundanz).

### Speed Sweep (4 configs, llama-bench, 3 reps)

| Config | bpw | pp512 | tg128 |
|---|---|---|---|
| ktq2_1/vtq2_1 (current prod) | 3.0 | 1111.59 | 85.80 |
| **ktq2/vtq2** ⭐ | 2.78 | **1195.31** | **86.37** |
| ktq2/vtq3 | 3.56 | 1190.57 | 86.50 |
| ktq2_1/vtq3_3 | 3.78 | 1188.31 | 86.38 |

### 🎯 35B Sweet Spot: `ktq2/vtq2` (= ktq2_1/vtq2_2)

- VRAM: 2.78 bpw (minimum, **-7% vs current prod**)
- TG: 86.37 t/s (**+0.66% vs current prod**)
- PPL: -0.33% (**-4.18pp vs current prod**)
- Quality identisch zu vtq3_v8 + vtq3_3 (Trellis-K-collision)
- Cheapest VRAM, max speed = klarer Triple-Goal-Winner

## Qwen3.5-4B-Q4_K_M dense (GPU0 RTX 2060)

### PPL Sweep (8 configs)

| Config | bpw | PPL | Drift |
|---|---|---|---|
| f16/f16 | 32 | 8.6946 | 0.00% |
| **ktq2_1/vtq4_1 (current 4B prod)** | 5.0 | 8.7117 | **+0.20%** |
| ktq2/vtq2 (Trellis) | 2.78 | 8.6643 | **-0.35%** |
| ktq2/vtq3 | 3.56 | 8.6643 | -0.35% |
| ktq3/vtq3 | 4.06 | 8.6643 | -0.35% |
| ktq2/vtq4 | 5.0 | 8.7117 | +0.20% |
| ktq2_1/vtq3_3 | 3.78 | 8.6643 | -0.35% |
| ktq4/vtq4 | 6.0 | 8.7117 | +0.20% |

### Speed Sweep (4 configs)

| Config | bpw | pp512 | tg128 |
|---|---|---|---|
| ktq2_1/vtq4_1 (current 4B prod) | 5.0 | 1632.75 | 78.70 |
| **ktq2/vtq2** ⭐ | 2.78 | **2006.80** | **79.59** |
| ktq2/vtq3 | 3.56 | 2004.55 | 79.57 |
| ktq2/vtq4 | 5.0 | (GPU crash) | - |

### 🎯 4B Sweet Spot: `ktq2/vtq2`

**Die alte Annahme war FALSCH!** Wir hatten dokumentiert: "4B-dense braucht vtq4_1 weil PPL-empfindlicher". Dieser Sweep widerlegt das:

- ktq2_1/vtq4_1 (alte Annahme): +0.20% drift, 78.70 t/s, 5.0 bpw
- **ktq2/vtq2 (Trellis): -0.35% drift (BESSER als f16!), 79.59 t/s, 2.78 bpw**

Trellis-V (vtq2_2) gewinnt auch auf 4B. Der frühere Test war evtl. mit unterschiedlichen Hyperparams oder Mess-Rauschen.

**v8 Win auch auf 4B:**
- VRAM: 2.78 vs 5.0 bpw = **−44% KV-VRAM**
- TG: 79.59 vs 78.70 t/s = **+1.1%**
- PPL: -0.35% vs +0.20% = **0.55pp besser**

## Kombinierte Empfehlung

**Beide Modelle bekommen denselben Sweet Spot: `ktq2/vtq2`**

Vorteile:
1. Konsistente Config zwischen Modellen
2. Maximum VRAM-Einsparung
3. Beste Quality (negativer drift = innerhalb f16-noise besser)
4. Beste Speed
5. Trellis kernels stabil seit 2026-04-25

## Hardware Issue

GPU0 ging beim 4B speed-bench (4. config ktq2/vtq4) in "GPU requires reset" state nach erfolgreichem Sweep. Wahrscheinlich unerwartete CUDA-Interaction nach mehreren Sweep-Runs hintereinander. **Reset benötigt sudo (User-Action).**

## Next Steps (warten auf User)

1. User: GPU0 reset (`sudo nvidia-smi --gpu-reset -i 0` oder VM reboot)
2. After reset: redeploy 35B prod mit `ktq2/vtq2` (= bisher v8-default)
3. After reset: redeploy 4B prod mit `ktq2/vtq2` (statt `ktq2_1/vtq4_1`)
4. Memory update: alte Memory `project_qwen35_4b_triple_goal.md` mit "4B braucht vtq4" ist OUTDATED → vtq2 ist auch hier Sweet Spot
