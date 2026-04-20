# Trick 2 — Per-head precision mixing: Design

**Datum:** 2026-04-20
**Status:** Design-Proposal, nicht implementiert
**Vorlage:** `tests/trellis-phase1/BACKLOG.md` Trick 2

## TL;DR

Per-head mixed precision (HV-heads → `vtq4_2`, LV-heads → `vtq2_2`) ist
im aktuellen llama-tq Layout **nicht ohne nichttrivialen Refactor möglich**:
ein V-Tensor pro Layer hat genau einen `ggml_type`, und sämtliche
View-/Stride-Rechnungen in `get_v()`/`cpy_v()` setzen
`ggml_row_size(v->type, ...)` voraus.

Sauberste Route: **Option A — zwei V-Sub-Tensoren pro Layer** (HV-Gruppe
+ LV-Gruppe), mit Head-Permutation die Heads nach Precision-Class
sortiert. Alles andere (echter Hybrid-Tensor, per-head row-type)
kollidiert mit ggml-Kernannahmen.

Erster Schritt (1 Woche): **Profiling-Hook** (Varianz/Kurtosis pro
Head auf Prefill) als separate PR — unabhängig vom Layout-Refactor.
Damit kriegen wir erstmal harte Daten, ob Varianz überhaupt heavy-tailed
genug ist, um den Refactor zu rechtfertigen.

## 1. Problem / Motivation

V-Varianz ist head-weise heavy-tailed. Wenn wenige Heads (z.B. 4 von 32)
80% der Varianz tragen, spart man durchschnittliche bpw, indem man nur
diese Heads auf `vtq4_2` hebt und den Rest auf `vtq2_2` drückt.
Ziel: **average ≈ 3 bpw bei PPL nahe vtq4_2**.

Präzedenzfall: Trick 1 (Sink-Protection) zeigt, dass ortsselektive
Precision funktioniert — ein f16 V-Layer von 24 senkt vtq2_2 Δ-PPL
von 9.0% auf 8.0%. Head-Selektivität ist die natürliche Erweiterung
in die Channel-Dimension.

## 2. State of the Art (aktueller Code)

### 2.1 V-Tensor Allocation (pro Layer, ein Typ)

`src/llama-kv-cache.cpp:191-360` — Loop über `il` erzeugt pro KV-Layer
**genau einen** `v` Tensor:

```cpp
// llama-kv-cache.cpp:309
ggml_tensor * v = has_v ? ggml_new_tensor_3d(
    ctx, eff_type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;
```

`eff_type_v` wird über drei orthogonale Policies pro Layer gewählt
(FA+TQ workaround, sink-protection, boundary-protection — Zeilen 244-306).
Alle drei sind **layer-granular**, nicht head-granular.

Layer-Struktur (`llama-kv-cache.h:229-245`):

```cpp
struct kv_layer {
    uint32_t il;
    ggml_tensor * k;
    ggml_tensor * v;  // EINER
    ggml_tensor * k_staging = nullptr;
    ggml_tensor * v_staging = nullptr;
    std::vector<ggml_tensor *> k_stream;
    std::vector<ggml_tensor *> v_stream;
    ...
};
```

### 2.2 View-Konstruktion setzt uniformen Type voraus

`llama-kv-cache.cpp:1414-1446` (`get_v`) — 4-D-Views rechnen Strides aus
**einem** `v->type`:

```cpp
return ggml_view_4d(ctx, v,
    hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
    ggml_row_size(v->type, hparams.n_embd_head_v(il)),
    ggml_row_size(v->type, n_embd_v_gqa),
    ggml_row_size(v->type, n_embd_v_gqa*kv_size),
    ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
```

**Härteste Constraint:** ein Tensor = ein Type = gleiche `row_size` für
alle Heads. Ein Hybrid-Tensor mit `vtq2_2` für Head-Slot 0-27 und
`vtq4_2` für Head-Slot 28-31 ist in ggml nicht repräsentierbar —
`nb[1]` ist konstant pro Tensor, Block-Größen zwischen VTQ-Varianten
(QK_VTQ2 = 256, unterschiedliche bpw → unterschiedliche block bytes)
decken sich nicht.

### 2.3 FA-Dispatch ist per-Type template-instanziiert

`ggml/src/ggml-cuda/fattn.cu:212-260` — jede `(type_K, type_V)`-Kombination
braucht eigene Template-Instanziierung via `FATTN_VEC_CASE`. Ein Forward-
Pass ruft pro V-Tensor genau einen FA-Kernel. Mixed-per-head innerhalb
eines Layers = **zwei FA-Aufrufe pro Layer**, Ergebnis-Softmax-Kombination
extern. Das ist der Kernel-Dispatch-Blocker.

### 2.4 Varianz-Profiling: existiert nicht

`grep -r "variance\|kurtosis\|v_stats" src/` — keine Treffer. Es gibt
keinen Hook der V-Werte pro Head inspiziert. Naheliegendster
Einfügungspunkt: `cpy_v()` (`llama-kv-cache.cpp:1485`). Dort läuft
`v_cur` durch — vor dem `ggml_set_rows`.

## 3. Konkurrenz

- **KIVI** (ICML 2024, arXiv:2402.02750) — asymmetrisch K/V, beide
  uniform 2-bit. Keine Mixed-Precision.
- **KVQuant** (NeurIPS 2024) — Outlier in FP16 + sparse. Element-Ebene.
- **KVTuner** (arXiv:2502.04420) — **Layer-wise** Mixed-Precision via
  Sensitivity-Grid. 3.25-bit lossless Llama-3.1-8B. Nah verwandt,
  aber Granularität eine Stufe gröber.
- **KITTY** (arXiv:2511.18643) — 2-bit V, uniform.

**Lücke:** Per-head Mixed-Precision ist öffentlich unbesetzt. Potentiell
publizierbar — vorausgesetzt Varianz ist heavy-tailed genug.

## 4. Options Analysis

### Option A — Sub-Tensor Split (empfohlen)

Pro Layer zwei V-Tensoren: `v_hi` (HV-Heads, `vtq4_2`), `v_lo` (LV-Heads,
`vtq2_2`). Heads werden beim Layer-Setup nach Class gruppiert.

**Änderungen:**
- `struct kv_layer` erhält `v_hi`, `v_lo`, `type_v_hi`, `type_v_lo`,
  `head_class[n_head_kv]` (uint8: 0=lo, 1=hi).
- `get_v()` liefert zwei Views oder virtuell concat'eten View
  (realistischer: zwei Views + zwei FA-Calls).
- `cpy_v()` routet Head-Slice → entsprechenden Tensor via
  `ggml_view` + `ggml_set_rows` × 2.
- FA-Dispatch in `llama-graph.cpp`: zwei getrennte `ggml_flash_attn_ext`-
  Calls, Outputs addiert (Partial-Softmax-Merge — äquivalent zu Multi-
  Device KV-Split, den llama.cpp bereits beherrscht).

**Pro:** Funktioniert mit existierenden FA-Kerneln. Keine neuen
Block-Layouts. Testbar schrittweise.

**Contra:** FA-Overhead ×2 pro Layer. Bei `n_head_kv ∈ {4, 8}`
spürbar, bei `n_head_kv = 32` (Qwen3.5-35B) vernachlässigbar.

**Effort:** ~400-600 LOC. 1-2 Wochen inkl. CUDA-Stabilität.

### Option B — Hybrid-Tensor mit Block-Interleaving

Ein V-Tensor mit custom Stride-Layout, per-row type.

**Blocker:**
- `ggml_row_size()` ist reine Funktion `(type, ne)` — kein per-row type
- `ggml_type` ist enum, nicht Vector
- Neue Quant-Type `GGML_TYPE_VTQ_MIX` + custom dequant-kernel mit
  Head-ID-Awareness → massiver Refactor in `ggml-cuda/fattn-*`,
  `convert.cu`, `getrows.cu`

**Verdikt:** Academisch interessant, unverhältnismäßig. Nicht empfohlen.

### Option C — Per-Head Stream

Jeder Head ist eigener `n_stream`. Missbrauch der existierenden
Multi-Stream-Infrastruktur.

**Blocker:**
- Streams sind semantisch Sequence-IDs. Code an vielen Stellen
  (`seq_to_stream`, `v_cells[stream]`, `sc_info`) hängt daran.
- FA würde `n_head_kv` sequenzielle Aufrufe machen — Performance-Desaster.

**Verdikt:** Falsche Abstraktionsachse.

## 5. Recommendation: drei Merge-Einheiten

**PR1 — Profiling-Hook (1-3 Tage, READ-ONLY, harmlos)**
- `cpy_v`-Callback, in `tq_profile_heads_first_n` Prefill-Calls Host-
  Kopie + Varianz/Kurtosis pro Head akkumuliert
- CLI: `--tq-profile-heads <N>` — dumpt JSON
  `{layer, head, variance, kurtosis}`
- Liefert **Daten**: ist Varianz heavy-tailed genug?

**PR2 — Static Head-Class Config (3-5 Tage)**
- GGUF-Metadata-Tensor
  `tq.head_precision_class[n_layer_kv × n_head_kv]` (uint8)
- Tool in `tools/` das aus PR1-Output Klassifikation nach Kurtosis-
  Quantil erstellt (z.B. top-12.5% → hi)
- `llama_kv_cache` liest Config, legt Sub-Tensor-Layout an

**PR3 — FA Dual-Dispatch (1-2 Wochen mit CUDA)**
- `llama-graph.cpp`: pro Layer zwei FA-Calls, Partial-Softmax-Merge
- Fallback: wenn `head_precision_class` fehlt → single-tensor

**Nur PR1 ist ohne Architektur-Risiko.** PR2+PR3 erst starten wenn
PR1-Daten die Hypothese stützen. Sonst Engineering ohne Ertrag.

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Varianz nicht heavy-tailed → kein PPL-Gewinn bei avg 3 bpw | Mittel | Hoch | PR1 zuerst; Go/No-Go Gate |
| FA Dual-Dispatch Overhead frisst Decode-Tok/s | Niedrig-Mittel | Mittel | Bei `n_head_kv ≥ 16` vernachlässigbar |
| Head-Class nicht stabil über lange Sequenzen | Mittel | Mittel | Rolling recompute auf 4k tokens; oder statisch aus Kalibrations-Corpus |
| GQA: Per-kv-Head, nicht per-q-Head | Hoch | Niedrig | Explizit dokumentieren |
| Sub-Tensor Layout bricht `state_write/read` | Mittel | Mittel | State-Format-Version-Bump |
| Interaktion mit `tq_protect_layers`, `tq_protect_sinks`, deferred-V | Hoch | Mittel | PR1+PR2 mit deferred-V aus; Integration als PR4 |
| Qwen3.5-35B MoE: V-Varianz pro Expert statt pro Head | Mittel | Hoch | Profiling pro Expert; Option A auf expert-granular erweitern |

## 7. Open Questions

1. **Threshold-Wahl:** Kurtosis vs. Varianz vs. top-K-magnitude — welche
   Metrik klassifiziert stabilsten? (PR1 empirisch)
2. **Ratio hi:lo:** 1:7 (12.5%, avg ~3 bpw) vs. 1:3 (25%, avg ~4 bpw)?
   CLI `--tq-head-hi-ratio 0.125`
3. **MoE-Interaktion:** Bei Qwen3.5-35B-A3B persistent über Experten,
   V-Writes aus Experten → positional statt head-positional. PR1
   muss mitmessen.
4. **Trick 16 (auto-bpw) Verhältnis:** Layer vs Head-Ebene. Zusammen
   designen um Doppel-Infrastruktur zu vermeiden.
5. **Training-Dynamik:** Nach Fine-Tuning dieselben HV-Heads?
   Head-Class als GGUF-Metadata an Checkpoint gebunden.

## 8. Next Steps

**Woche 1:**
- Review mit Maverick
- Bei Go: PR1 implementieren
  - Dateien: `src/llama-kv-cache.{h,cpp}`, `common/arg.cpp`,
    `include/llama.h`
  - ~150 LOC
- Test: Qwen3.5-0.8B (schnell) + Qwen3.5-35B-A3B (Production gpu00:8791)

**Woche 2:**
- PR1-Daten auswerten, Plot Kurtosis-Distribution
- **Decision Gate:** top-12.5% Heads > 50% der Varianz?
  - Ja → PR2
  - Nein → Trick 2 auf Eis, fokus Trick 3/4/7

## 9. Files Touched (projected)

| PR | Files |
|----|-------|
| PR1 | `src/llama-kv-cache.{h,cpp}`, `common/arg.cpp`, `include/llama.h`, `tests/trellis-phase1/profile_heads.sh` |
| PR2 | `src/llama-model.cpp` (GGUF-Read), `src/llama-kv-cache.{h,cpp}`, `tools/quantize/`, `gguf-py/` |
| PR3 | `src/llama-graph.cpp`, `ggml/src/ggml-cuda/fattn.cu`, `tests/trellis-phase1/RUN8_per_head.md` |

## Referenzen
- [KIVI](https://arxiv.org/abs/2402.02750)
- [KVTuner](https://arxiv.org/html/2502.04420)
- [KVQuant NeurIPS 2024](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf)
- [KITTY](https://www.arxiv.org/pdf/2511.18643)
