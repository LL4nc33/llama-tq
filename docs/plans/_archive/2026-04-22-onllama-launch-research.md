# Research Report: Porting `ollama launch claude` to on-llama-tq

**Date:** 2026-04-22
**Status:** Research complete, no code changes.

---

## 1. What `ollama launch claude` actually does

Per docs (`docs.ollama.com/integrations/claude-code`) and source (`cmd/launch/claude.go`):

**Command surface:**
```
ollama launch claude
ollama launch claude --model kimi-k2.5:cloud
ollama launch claude --model X --yes -- -p "prompt"
ollama launch claude -- --channels plugin:telegram@claude-plugins-official
```

**What it does (verbatim from `claude.go`):** It is a thin launcher for Anthropic's **Claude Code CLI binary** (`claude`), not a model/weights shipment. It:
1. Resolves the `claude` binary via `exec.LookPath("claude")`, falling back to `~/.claude/local/claude[.exe]`.
2. `exec.Command`s it with `--model <model>` plus any pass-through args.
3. Injects env vars pointing Claude Code at Ollama's local server:
   ```
   ANTHROPIC_BASE_URL=<ollama host, default http://localhost:11434>
   ANTHROPIC_API_KEY=
   ANTHROPIC_AUTH_TOKEN=ollama
   CLAUDE_CODE_ATTRIBUTION_HEADER=0
   ANTHROPIC_DEFAULT_OPUS_MODEL=<model>
   ANTHROPIC_DEFAULT_SONNET_MODEL=<model>
   ANTHROPIC_DEFAULT_HAIKU_MODEL=<model>
   CLAUDE_CODE_SUBAGENT_MODEL=<model>
   CLAUDE_CODE_AUTO_COMPACT_WINDOW=<ctx from cloudModelLimits, if cloud>
   ```

**Network behavior:** Ollama itself proxies the Anthropic-compatible API. Claude Code thinks it's talking to Anthropic's API at `localhost:11434`. Ollama translates Anthropic `POST /v1/messages` shape to/from its internal inference. **No calls to Anthropic's real servers.**

**Not in the launcher:** no model download, no auth flow, no template hydration. Downloads are handled by `ensureModelsReady()` in `launch.go` (shared helper), triggered by `LaunchMissingModelMode` policy before the runner fires.

## 2. Ollama's internal implementation

**License:** MIT (confirmed via `LICENSE` on main). Compatible with llama.cpp/on-llama-tq MIT. **Safe to port code directly** with attribution.

**Directory:** `cmd/launch/` contains a plugin architecture:
- `launch.go` — Cobra command `LaunchCmd()`, three interfaces: `Runner`, `Editor`, `ManagedSingleModel`; `LauncherState`, `LaunchPolicy`, model-ready gates (`ensureModelsReady`, `showOrPullWithPolicy`), dispatch (`launchSingleIntegration` / `launchEditorIntegration` / `launchManagedSingleIntegration`).
- `claude.go` — 86 LOC, implements `Runner` (trivially small; full file reproduced in section 1).
- `models.go` — `isCloudModelName()`, `cloudModelLimits{}` (e.g. `kimi-k2.6` → 262144 ctx), `recommendedModels[]`, `recommendedVRAM{}`.
- `registry.go`, `selector_hooks.go` — TUI model picker.
- `copilot.go`, `codex.go`, `cline.go`, `droid.go`, `hermes.go`, `kimi.go`, `openclaw.go`, `opencode.go`, `pi.go`, `vscode.go` — sibling integrations following the same `Runner` pattern. Each is ~50–150 LOC.

**Load-bearing functions (Ollama side):**
1. `Claude.Run(model, args)` — `cmd/launch/claude.go:46` — the actual launcher (exec + env).
2. `Claude.findPath()` — `cmd/launch/claude.go:27` — resolves the external `claude` binary.
3. `LaunchCmd()` — `cmd/launch/launch.go` — Cobra wiring, subcommand dispatch.
4. `ensureModelsReady()` / `showOrPullWithPolicy()` — `cmd/launch/launch.go` — pull-before-run gate.
5. `ResolveRunModel()` — `cmd/launch/launch.go` — TUI selector.

**Dependency surface:** Pure Go stdlib (`os/exec`, `os`, `path/filepath`, `runtime`, `strconv`) + internal `github.com/ollama/ollama/envconfig`. **Zero Anthropic API calls.** **Zero network calls** inside `claude.go` itself. The Anthropic compatibility is entirely in Ollama's server (`/v1/messages` route), not the launcher.

## 3. Equivalent in on-llama-tq

### 3a. Server side: already 95% done

`on-llama-tq/tools/server/server.cpp:187–188` already registers:
```
POST /v1/messages
POST /v1/messages/count_tokens
```
routed to `routes.post_anthropic_messages` / `post_anthropic_count_tokens`. Conversion logic at `server-common.cpp:1436` (`convert_anthropic_to_oai`), `:1873` (`format_anthropic_sse`), and streaming at `server-task.cpp:728` (`TASK_RESPONSE_TYPE_ANTHROPIC` → `to_json_anthropic_stream`). Tests: `tools/server/tests/unit/test_compat_anthropic.py` (6+ cases). **`llama-server` is already an Anthropic-compatible endpoint.** Claude Code pointed at `http://localhost:8080` with `ANTHROPIC_AUTH_TOKEN=anything` should Just Work today.

This is the single most important finding: **the hard part (API surface) already exists.** The launcher is the 10% that's missing.

### 3b. Command dispatch: no subcommand layer

Confirmed by inspecting `tools/`: llama.cpp/on-llama-tq uses **separate binaries per tool** (`llama-server`, `llama-cli`, `llama-bench`, `llama-batched-bench`, `llama-perplexity`, ...). There is no `onllama <subcommand>` router. Options:

- **(A) New standalone binary** `tools/launch/launch.cpp` that execs `claude` with env vars and assumes a user-started `llama-server`. Simplest, most aligned with existing architecture. ~150–250 LOC C++.
- **(B) Subcommand inside `llama-cli`** — would require arg-parser surgery in `common/arg.cpp`, fights the existing convention. Not recommended.
- **(C) Wrapper script** `scripts/onllama-launch-claude.sh` (and `.ps1`) — ~40 LOC shell. Zero compilation, zero risk, but less polished. Good MVP.
- **(D) Meta-binary** `onllama` that dispatches to `llama-server`, `llama-cli`, `launch`, etc. — large refactor, touches build system. Out of scope unless project is pivoting to Ollama-style UX broadly.

### 3c. Model "registry" analogue

We have none. Three realistic options:
- **HuggingFace Hub download** via `curl`/`libcurl` into `~/.cache/on-llama-tq/models/` (llama.cpp already supports `--hf-repo` in `common_params_parse`). Reuse it.
- **User provides GGUF path** via `--model /path/file.gguf` (already standard). Ship a curated default URL for the `claude` alias.
- **No download, check-only** — require user to have run `llama-server -m ...` first; launcher just execs `claude` with env.

Option 3 + 2 hybrid matches Ollama's semantics closely without a registry build-out.

### 3d. "Claude" semantic mapping

Since we cannot ship Anthropic weights (and there would be no point — Ollama doesn't either):

- **Option a — Local code-model stand-in:** Default `claude` alias → pre-configured launch of Qwen2.5-Coder-32B-Instruct Q4_K_M (or DeepSeek-Coder-V2-Lite). Minimum viable; matches Ollama's pattern (Ollama ships open models under cloud-ish aliases too).
- **Option b — BYOK proxy to real Anthropic:** Pass user's `ANTHROPIC_API_KEY` through; `llama-server` reverse-proxies to `api.anthropic.com`. Defeats local-first ethos; skip unless asked.
- **Option c — Hybrid:** local by default, `--api` flag enables passthrough. Best UX, most code.

**Recommended:** (a). It's what Ollama actually does — Claude Code is the client; the "Claude" in `launch claude` refers to the *client binary*, not the weights. Our command would do the same.

## 4. Implementation size + time

Assuming familiar developer:

| Option | LOC | Hours | Notes |
|---|---|---|---|
| (C) shell wrappers for `onllama-launch-claude.sh/.ps1` | ~60 | 2–3 | MVP. Assumes user started `llama-server`. Ships today. |
| (A) new C++ binary `tools/launch/launch.cpp` | ~250 | 6–10 | Cobra-equiv via existing `common_params`; CMake entry; windows/unix path fallback; env injection. |
| (A) + model auto-pull via `--hf-repo` glue | +150 | +4 | Reuses existing HF download path in `common.cpp`. |
| (A) + auto-start `llama-server` subprocess if not running | +200 | +6 | Port detection, spawn, health probe, teardown. |
| (a) Claude-alias default model config (TOML/JSON) | ~50 | 2 | `models/aliases.json` with `claude → qwen2.5-coder-32b-instruct-q4_k_m.gguf` + HF URL. |
| Tests (python test suite, matches existing pattern) | ~150 | 3 | Mirror `test_compat_anthropic.py`. |
| Docs (README + integration guide) | — | 2 | |
| **Total MVP (shell only):** | **~60** | **~3h** | |
| **Total full C++ port with auto-pull + auto-server:** | **~800** | **~25h** | |

## 5. Legal / trademark

**"Claude" is a trademark of Anthropic, PBC.** Using it as a subcommand literal (`onllama launch claude`) is a gray area. Ollama gets away with it because:
- It's nominative use ("launches Claude Code", Anthropic's actual product).
- It doesn't imply endorsement.
- The command just runs Anthropic's own `claude` binary with env vars.

**If we do the same thing** (launch Anthropic's own `claude` CLI pointed at our server), nominative-use doctrine likely covers us — same as Ollama's risk profile. **If we replace the client with a different binary** or point "claude" at a non-Anthropic model without making that crystal-clear in UI output, that's closer to trademark dilution/confusion.

**Safer alternatives:**
1. `onllama launch claude-code` — explicitly references Anthropic's product by its full name, strongly nominative.
2. `onllama launch coder` — model-agnostic; could dispatch to claude-code, aider, continue, cursor-cli, etc.
3. `onllama run anthropic-compatible` — wordy but unambiguous.
4. `onllama serve --anthropic` — skip the launcher metaphor; just flip a server flag (we arguably already have this).

Practical recommendation: **ship as `onllama launch claude-code`** (matches Anthropic's product name exactly, least trademark risk, most descriptive), with shell alias `claude` documented. Add a one-line disclaimer in `--help` output: "Launches Anthropic's Claude Code CLI against a local Anthropic-compatible server. Claude is a trademark of Anthropic, PBC."

**Pending verification:** no IP lawyer consulted. Anthropic's actual position on this pattern is not publicly stated beyond their tacit acceptance of Ollama's integration (which they co-announced).

## Recommendation

**Worth doing — but scope-gated.**

Do the **shell MVP first** (~3 hours). It demonstrates the full UX using on-llama-tq's already-existing `/v1/messages` endpoint, which is the surprise finding of this research: **we are one 60-line script away from parity with Ollama's launcher.** The C++ binary is a nice-to-have; the `.sh`/`.ps1` script captures ~90% of user value.

Only promote to a C++ binary if (a) usage justifies it, or (b) we're doing the broader `onllama` meta-binary refactor anyway.

**Do NOT** ship this as `onllama launch claude` literally — use `claude-code` to minimize trademark surface area. Keep a `claude` shell alias in the docs so muscle memory works.

---

## Relevant file paths

- `/mnt/d/repos/on-llama-tq/tools/server/server.cpp` (lines 187–188: `/v1/messages` routes)
- `/mnt/d/repos/on-llama-tq/tools/server/server-common.cpp` (line 1436: `convert_anthropic_to_oai`; line 1873: `format_anthropic_sse`)
- `/mnt/d/repos/on-llama-tq/tools/server/server-task.cpp` (lines 728, 1105, 1171, 1653: Anthropic stream/final serializers)
- `/mnt/d/repos/on-llama-tq/tools/server/tests/unit/test_compat_anthropic.py` (existing test harness to mirror)
- `/mnt/d/repos/on-llama-tq/tools/server/README.md` (lines 1378–1440: existing Anthropic API docs)
- `/mnt/d/repos/on-llama-tq/tools/cli/cli.cpp` (reference for adding new `tools/launch/` binary pattern)
- Upstream reference: `github.com/ollama/ollama/cmd/launch/claude.go` (86 LOC, MIT, directly portable)
- Upstream reference: `github.com/ollama/ollama/cmd/launch/launch.go` (Cobra wiring + policies)
