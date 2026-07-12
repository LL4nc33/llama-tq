# Upstream-Adapt Round 2 — was llama-tq noch übernehmen kann

**Datum:** 2026-07-12
**Basis:** upstream/master `e3546c794` vs Gold-Branch `248f25b32` (nach den 3 Tier-1-Picks)
**Methode:** 2 adversarisch-verifizierte Workflows (8 Finder-Achsen + 6 Critic-Gaps), jeder
Kandidat gegen Fork-*Inhalt* (nicht SHA) und HW-Relevanz geprüft. 42 Agents gesamt.

Kontext-Filter: RTX 2060 Turing sm_75, CUDA-only, IQ2-MoE + TurboQuant-KV, Modelle
Qwen3.6-A35B-IQ2 / Gemma-4-12B/26B (SWA+Vision, teils MTP) / Ministral-3B, 200k-256k ctx, parallel 2.

`ADOPT` = sauber cherry-pickbar. `MANUAL_PORT` = Datei fork-diverged, Logik selbst nachbauen
(blinder cherry-pick gefährlich — teils meldet git `exit 0`, dupliziert aber still den Fix).

---

## TIER 1 — Korrektheits-Bugs auf Pfaden die wir REAL fahren (höchste Prio)

Das sind keine Nice-to-haves: latente Crashes / OOB / falsche Attention auf unseren Live-Pfaden.

| SHA | PR | Fix | Trifft uns | Verdict / Risiko |
|-----|-----|-----|-----------|------------------|
| `0eca4d490` | #24945 | KQ-mask strides `int32→int64` in `flash_attn_mask_to_KV_max` | 200k-256k ctx + parallel≥2 → int32-Overflow → **korrupte Attention**. Cold-path, NICHT die TQ-hot vec-dot. | MANUAL_PORT / med — 3 Stellen in `fattn-common.cuh` (607, 1016, **1361 = fork-TQ-dispatch**), `int→int64_t` |
| `a4107133a` | #25215 | `&& ->buffer`-Guard für K/V-rotation set_input | NULL-buffer-`GGML_ASSERT`-Crash auf **iswa/SWA-Pfad (Gemma-4)** + unser MTP-KV-inject. Nur `llama-graph.cpp`. | **ADOPT** / low |
| `a66d50588` | #24294 | iswa `kq_mask` auf eigenem Buffer guarden | SWA-only-draft-head (leere base sub-cache → null kq_mask) = **genau unser gemma4uv-MTP-assistant-head**, live. | **ADOPT** / low |
| `8a091c47a` | #24256 | spec vocab-compat-check: `bool→auto` (enum kollabierte zu true) | `common_speculative_are_compatible` — externer Draft-Model-Pfad. BPE-target+SPM-draft passte fälschlich → falsche Tokens. | **ADOPT** / low (in speculative.cpp, aber sauberer 2-Zeilen-Fix) |
| `defa95c30` | #23936 | OOB-Read in `ngram-map.cpp` bei Prompt-Shrink | **Live-ngram-spec** (alle Modelle) bei Reasoning-Chats / Prompt-Cache-Rewind. | MANUAL_PORT / high — fork hat `idx_last_check`-Divergenz, port_note in Doc |
| `1ee093937` | #25449 | `llama-batch` decreasing-pos-Guard (war tot, verglich nur gegen −1) | Multi-seq-Batching + parallel slots + unser Spec-Stack. 2 Zeilen. | **ADOPT** / low |
| `b000431a0` | #23857 | `#include <algorithm>` in `ngram-mod.cpp` | Build-Portabilität unseres NGRAM_MOD (2.28x repeat-boost). Bricht auf libc++/MSVC. | **ADOPT** / low |

## TIER 2 — Robustheit / Ressourcen auf unserem Deploy

| SHA | PR | Fix | Trifft uns | Verdict / Risiko |
|-----|-----|-----|-----------|------------------|
| `074944998` | #24776 | CUDA top_k/argsort in ~64MB-chunks → weniger temp-VRAM | 12GB-compute-buffer-budget bei 200k ctx. Nicht TQ-hot. | **ADOPT** / low |
| `60130d18f` | #24013 | `--sse-ping-interval` (SSE-keepalive bei langem Prefill) | 200k-Prefill ohne Token-Output → Clients droppen per Timeout. **Unser Failure-Mode.** | ADOPT / med |
| `40f3aafc4` | #24774 | `X-Accel-Buffering:no` Header | Deploy hinter Proxy → Nginx puffert SSE-Stream sonst. 2 Zeilen. | **ADOPT** / low |
| `236531595` | #23981 | SWA-checkpoints speichern nur non-masked cells | Gemma-4 SWA + session/checkpoint-state-save. | **ADOPT** / low |
| `db94854ff` | #24411 | server: checkpoints jenseits `pos_next` skippen | Prompt-Cache-Checkpoints. Fork-Selektions-Lambda diverged. | MANUAL_PORT / med |

## TIER 3 — VRAM/Perf mit Risiko-Abwägung (Bench-Gate zwingend)

| SHA | PR | Fix | Abwägung | Verdict / Risiko |
|-----|-----|-----|----------|------------------|
| `031ddb2e0` | #23764 | f16-mask für FA (statt f32) → VRAM-Ersparnis | Berührt hot `llama-kv-cache.cpp` + `llama-graph.cpp`. Genau 1 Konflikt. Könnte KV-VRAM spürbar senken. | MANUAL_PORT / med — nur mit Bench+PPL-Gate |
| `379ac6673` | #24277 | KV-cells-Kopien vermeiden (shared_ptr) | Ctor-init-list ist bei uns TQ-schwer umgebaut. Perf nur bei großem KV/draft-sharing. | MANUAL_PORT / high |
| `f0156d140` | #24267 | KV: source-cache-size beim cell-sharing folgen | **`git cherry-pick` meldet exit 0, DUPLIZIERT aber still den Fix-Block!** Nur manuell. | MANUAL_PORT / high |
| `f8f0a47a5` | #23907 | KV-quant-Space beim Startup reservieren | Patch kennt nur EINEN upstream-launch_fattn; unser TQ-Dispatch hat mehrere. Korrumpiert blind. | MANUAL_PORT / high |

## TIER 4 — Convert/Tooling QoL (kein Live-Impact, null Regression)

| SHA | PR | Fix | Nutzen | Verdict |
|-----|-----|-----|--------|---------|
| `c5229087a` | #23250 | `--fp8-as-q8` convert-Flag (default off) | FP8-Checkpoints (Qwen3.x/Mistral zunehmend) selbst konvertieren, halbe Zwischen-GGUF-Größe. | ADOPT / low |
| `f4043fec0` | #24833 | konsistenteres rope_parameters-handling im convert | Absichert künftige Qwen/Gemma/Mistral-Konvertierungen. Fork-convert diverged. | MANUAL_PORT / high |

---

## Bewusst VERWORFEN (aus Runde 1+2, dokumentiert damit nicht nochmal geprüft)

- **`e495d1e74` #25148** Gemma-E4B-MTP-FlashAttention-fix — berührt TQ-hot fattn für DKQ>256-Pfad
  den wir nicht nutzen; Gemma-4-MTP ist auf 2060 Netto-Verlust ([[project_gemma4_mtp_slower_than_baseline]]).
- **`7a63fdede` #24491** Turing-P2P-VMM — P2P ist bei uns tot (IOMMU, [[project_p2p_blocked]]).
- **`7c158fbb4` #24108** on-device spec-checkpoints disable — MANUAL_PORT möglich, aber unser
  DRAFT_MTP-Pfad hat den ON_DEVICE-Save laut `e359c740c` eh als dead-code (nur seq_rm-fail-fallback).
  Niedrige Prio, kein aktiver Bug beobachtet.
- **`cb295bf59`** K/V-type-validation-refactor — würde unsere KTQ/VTQ-Whitelist überschreiben.
- Alle Vulkan/SYCL/OpenCL/Metal/HIP/hexagon (137 Commits) — falsches Backend.
- DeepSeek-V4/V3.2, Mistral-Medium-128B — passen nie auf 12GB.

## Schon-im-Fork (adversarisch aussortiert — KEINE Doppelarbeit)

- `b3ce5cedf` #24986 quant-moe-mtp — fork-`llama-quant.cpp` hat den Fix bereits.
- `603300b00` #24208 n_gpu_layers off-by-one — fork nutzt bereits `>` an beiden Sites.
- `04eb4c446`/`7d2b45b4f` Gemma-4-MTP-core — via `feature/gemma4-mtp-23398` gemergt.

---

## Empfohlene Build-Reihenfolge

1. **Batch A (clean ADOPT, low-risk):** `a4107133a`, `a66d50588`, `8a091c47a`, `1ee093937`,
   `b000431a0`, `074944998`, `40f3aafc4`, `236531595`, `60130d18f`, `c5229087a` — ein CUDA-Build,
   Bench-Gate (kein Regression), die 3 crash-guards sind reine Korrektheit.
2. **Batch B (MANUAL_PORT Korrektheit):** `0eca4d490` (KQ-overflow, 3 Stellen), `defa95c30`
   (ngram OOB, port_note im journal). Sorgfältig, PPL+Coherence-Gate auf 35B-IQ2 langem ctx.
3. **Batch C (VRAM/Perf, optional, Bench-getrieben):** `031ddb2e0` (f16-mask), `f0156d140`,
   `379ac6673` — nur wenn VRAM/Perf-Bedarf, jeder einzeln mit Gate.

Volle port_notes: `journal.jsonl` unter
`subagents/workflows/wf_16429ae1-d1e/`.
