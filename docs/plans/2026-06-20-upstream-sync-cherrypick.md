# Upstream-Sync: Cherry-Pick & adaptiver Nachbau (2026-06-20)

**Fork-HEAD:** `a398c7135` (2026-06-15) · **upstream/master:** `84de01a1f` (2026-06-20)
**Letzter echter Sync:** PR #23398 (Gemma4-Assistant MTP), ~2026-06-09

## Kontext / Strategie

Der User will nicht nur literale `git cherry-pick`-Übernahmen, sondern auch
**Ideen/Patterns aus upstream adaptiv im Fork nachbauen**, wenn der Patch durch
Fork-Divergenz (OidaNice-Umbau, TurboQuant-KV, MTP, eigener Eagle3) nicht 1:1
appliziert.

### Wichtige Struktur-Erkenntnisse

1. **WebUI nicht entfernt, sondern umstrukturiert:** upstream verschob
   `tools/server/webui/` → `tools/ui/` (PR #23064, Naming `llama-ui`). Features
   bleiben im Haupt-Repo abrufbar → ideenweise Portierung in unser SvelteKit-WebUI
   möglich. Literale Picks scheitern dort aber an Pfad + abweichender Codebasis.
2. **Server-Split haben wir bereits** (server-{common,context,queue,http,models,task,tools}.cpp
   via #23398). upstream hat zusätzlich `main.cpp`, `server-chat.cpp/h`,
   `server-schema.cpp/h`.
3. **Eigener Eagle3** (Phasen 48-55, verteilt in bestehenden Dateien) ⇒ upstream-Eagle3
   (#18039, neue `src/models/eagle3.cpp`) ist **konkurrierend**, kein Pick.

## Verifizierte Cherry-Pick-Tests (empirisch, `cherry-pick --no-commit`)

| Commit | Subject | Apply-Status |
|---|---|---|
| `159d093a4` | server: fix non-bound n_discard (#24786) | ✅ **CLEAN** (4 Zeilen, server-context.cpp) |
| `10786217e` | server: HTTP 400 on invalid grammar (#24154) | ✅ **CLEAN** (3 Zeilen, common/sampling.cpp) |
| `e27f30859` | server: no auth-header in CORS proxy (#24373) | ✅ C++-Hunk (server-cors-proxy.h) CLEAN; UI-Teil (`tools/ui/`) weglassen |
| `d2462f8f7` | chat: LFM2 ignoring json_schema (#24377) | ⚠️ KONFLIKT common/chat.cpp (PEG-Parser-Umbau) → adaptiv |
| `fb83cc9a0` | CUDA: ssm_scan_f32 data-races (#24360) | ⚠️ KONFLIKT ssm-scan.cu (fehlende `kernel_launch_params`-Abstraktion) → adaptiv (3 Zeilen) |
| `aedb2a5e9` | chat: Cohere2MoE parser (#24615) | ⚠️ KONFLIKT common/chat.cpp → adaptiv |
| `4b48a53b6` | server: optimize get_token_probabilities (#24796) | ⚠️ KONFLIKT server-context.cpp → adaptiv (Idee: partial_sort, 12× top-k) |

## Batch A — sofort (literal + isoliert)

Branch: `upstream-sync-2026-06`. Reine Stabilität/Security, kein Fork-Feature-Risiko.

- [ ] `159d093a4` — n_discard bounds-clamp (KV-ctx-shift Crash-Edge-Case)
      ⚠️ vor Merge MTP-Draft-KV-Interaktion gegenprüfen
- [ ] `10786217e` — invalide Grammar → HTTP 400 statt Silent-Drop
- [ ] `e27f30859` — **Security**: CORS-Proxy leakt keine Auth-Header (nur C++-Hunk
      `server-cors-proxy.h` + `test_security.py`; `tools/ui/`-Hunks droppen)
- [ ] **adaptiv** `fb83cc9a0` — ssm-scan `__syncthreads`-Fix manuell nachbauen
      (3 Zeilen, ohne `ggml_cuda_kernel_launch_params`-Abstraktion)
- [ ] **adaptiv** `aedb2a5e9` — Cohere2MoE: `models/templates/Cohere2MoE.jinja`
      (neue Datei, sauber) + chat.cpp-Parser-Hunk manuell einpassen

**Gate:** CUDA-Build auf a local GPU host (setsid, -j1, ccache) → `make test` →
MTP-Acceptance-Regression-Check auf gemma-4 (lossless). Erst dann Merge.

## Batch B — WebUI-Quick-Wins (ideenweise in SvelteKit, kein CUDA)

Aus `tools/ui/`, adaptiv portiert. Reiner Frontend-Build.

- [ ] **Streaming-Error → Teil-Antwort behalten** (#23090) — höchster Alltags-Schmerz;
      `onError`-Pfad in `chat.service.ts` darf generierten Text nicht verwerfen
- [ ] **Autoscroll-Fixes** (#23026, #22977) — User-Scroll-Up bricht Auto-Scroll;
      Safari/WebKit-Form-Box-Bug; `use-auto-scroll.svelte.ts` + `ChatScreen.svelte`
- [ ] **Reasoning-Block als Markdown** (#24611) — `<MarkdownContent>` im Reasoning-Block
      (passt zum neuen Reasoning-Effort-Selektor)
- [ ] **EXIF-Orientation beim Bild-Upload** (#24196) — gedrehte Handyfotos korrigieren;
      optional **EXIF-Stripping (DSGVO)** als eigene Ergänzung
- [ ] zentrales Netzwerk-Fehler-Handling (#23431) — Aufräum-Pattern `chat.service.ts`/`api-fetch.ts`

Weitere Kandidaten (nach Bedarf): Mermaid/SVG inline + Source-Toggle (#24032/#24080/#24652),
Pinned Conversations (#21387), HEIC/HEIF (#24137), JSONL-Export (#24688),
A11y Keyboard/Touch (#23132/#24604).

## Batch C — Server-adaptiv (Design-Review nötig)

- [ ] **Token-Probs `partial_sort`** (`4b48a53b6`, #24796) — 12× schneller top-k
      logprobs; Signatur `get_token_probabilities(..., n_top)` + Call-Sites in
      server-context.cpp anpassen
- [ ] **Schema-Validierung** (`e1efd0991`, #24150) — Konzept (zentrale Request-
      Validierung + Alias `top_k`/`top-k`) in server-task.cpp einklinken, inkl.
      TurboQuant-Flags; nicht 1:1 `server-schema.cpp` übernehmen
- [ ] **Tool-Call-Parsing-Hardening** (`581e8eca8`) — robusteres OpenAI-Style-Parsing;
      adaptiv wg. MTP-Draft-State-Interaktion
- [ ] **LFM2 json_schema-Fix** (`d2462f8f7`) — adaptiv in unsere PEG-Parser-Region
- [ ] **Router Model-Management** (`4b4d13ae7`, #23976) — nur **SSE-Status-Pattern +
      load/unload-Endpoints** als Idee; eigener Multi-Model-Router bleibt Basis

## Bewusst NICHT übernehmen

- `88a39274e` Upstream-Eagle3 (#18039) — konkurriert mit eigener Eagle3-Implementierung
- `e95dae18d` MTP-Padding/D2D-Removal — 19 Backend-Dateien, kollidiert potenziell mit
  TurboQuant-fattn; erst analysieren
- `02810c7aa` NVFP4 edge-cases — nur relevant wenn NVFP4 aktiv genutzt
