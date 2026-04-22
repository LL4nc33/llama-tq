# 400K ctx OOM Fix — Runtime Validation

**Datum:** 2026-04-20
**Server:** gpu00:8791 (Qwen3.5-35B-A3B IQ2_XS, ktq2_1 + vtq3_2 + deferred + sink)
**Binary:** master build (build-cuda/bin/llama-server, kein Rebuild nötig)

---

## Zusammenfassung

Runtime-Workaround für den parallel=2 + 400K ctx OOM auf 2× RTX 2060 12GB.
Kein neuer Build mit GGML_SCHED_MAX_COPIES=2 nötig.

**Ergebnis:** parallel=2 @ **200K ctx** (100K/slot) funktioniert mit marginalem
TG-Verlust (-2% vs parallel=1 @ 400K).

---

## Messungen

| Config | ctx | parallel | ub | -ts | tg tok/s | pp tok/s | Status |
|--------|-----|----------|----|----|----------|----------|--------|
| parallel=2 @ 400K ts 10,14 | 400K | 2 | 256 | 10,14 | — | — | ❌ OOM CUDA1 |
| parallel=2 @ 400K ts 11,13 | 400K | 2 | 256 | 11,13 | — | — | ❌ OOM CUDA1 (näher) |
| **parallel=2 @ 200K ts 12,12** | **200K** | **2** | **256** | **12,12** | **65.76** | **183.92** | **✅** |
| parallel=1 @ 400K (Baseline) | 400K | 1 | 512 | 12,12 | ~67 | — | ✅ |

Messkommando:
```bash
curl -X POST http://localhost:8791/completion -d '{"prompt":"Erkläre in 100 Worten...","n_predict":150}'
```
→ `timings.predicted_per_second = 65.76`

---

## Deploy-Empfehlung

**Für Single-User Entwicklung:** `parallel=1 -c 400000` (67 tok/s, 400K ctx verfügbar)

**Für Multi-User Testing:** `parallel=2 -c 200000 -ub 256 -ts 12,12` (65.8 tok/s × 2 slots)

---

## Warum parallel=2 + 400K auch mit -ts Shift nicht geht

OOM-Report (`2026-04-20-400k-oom-analysis.md`) nennt GGML_SCHED_MAX_COPIES=4 als
Root-Cause (4× Input-Replication bei pipeline parallelism). -ts shift schiebt
~250 MiB zwischen GPUs, reicht aber nicht um den 560 MiB Delta bei 400K + p=2 zu
kompensieren.

**Full-Fix wäre:** Rebuild mit `-DGGML_SCHED_MAX_COPIES=2` → halbierte Replication.
Aufwand: ~45min fattn.cu recompile. Nicht aktuell nötig da 200K pro slot bei 2
Usern ausreichend ist.

---

## Offene Frage

Lohnt sich der MAX_COPIES=2 Rebuild, um parallel=2 @ 400K zu ermöglichen?
Use-case: 2 User × 400K ctx (= Code-Agent mit riesigen Repos, lange Sessions).
Setting auf "Backlog" bis Bedarf klar ist.
