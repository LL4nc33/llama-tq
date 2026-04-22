# Claude Code against a llama-tq server

llama-tq ships `/v1/messages` (Anthropic-compatible) in `tools/server/server.cpp`,
so Anthropic's official Claude Code CLI can talk to a local llama-tq server
instead of `api.anthropic.com`.

The `scripts/onllama-launch-claude.sh` wrapper sets the required environment
variables and execs the real `claude` binary — it does not re-implement
anything, it just points Claude Code at your server.

Inspired by Ollama's `ollama launch claude` (MIT).

---

## Prerequisites

- **Claude Code CLI** installed — see
  [docs.claude.com/en/docs/claude-code](https://docs.claude.com/en/docs/claude-code).
  The wrapper looks for `claude` on `PATH` and falls back to
  `~/.claude/local/claude` / `~/.claude/local/claude.exe`.
- **A running llama-tq server** exposing `POST /v1/messages`. Any build from
  this fork since 2026-04 qualifies. The server does not need to live on the
  same machine as Claude Code — see [Remote server](#remote-server) below.

Quick health check:

```bash
curl -fsS http://localhost:8080/health
# {"status":"ok"}
```

---

## Local server (same machine as Claude Code)

Start the server, then launch Claude Code against it:

```bash
# terminal 1 — llama-tq server
./build/bin/llama-server -m model.gguf --host 127.0.0.1 --port 8080 \
    -fa on -ngl 99 --jinja

# terminal 2 — Claude Code
./scripts/onllama-launch-claude.sh
```

Defaults: `--server http://localhost:8080`, `--model qwen3.6-35b-a3b`.

Override either:

```bash
./scripts/onllama-launch-claude.sh --server http://127.0.0.1:9000 --model my-model
```

Pass extra args to Claude Code after `--`:

```bash
./scripts/onllama-launch-claude.sh -- --print "sag hallo"
```

---

## Remote server

Running llama-tq on a GPU box, Claude Code on a laptop? Two options:

### Option A — SSH port-forward (recommended)

No credentials in the Claude config, server stays bound to `127.0.0.1`.

```bash
# terminal 1 — tunnel
ssh -N -L 8080:localhost:8080 user@gpu-host

# terminal 2 — Claude Code
./scripts/onllama-launch-claude.sh --server http://localhost:8080
```

### Option B — direct LAN

Bind the server with `--host 0.0.0.0` (or a specific LAN IP) and point
Claude Code at it:

```bash
./scripts/onllama-launch-claude.sh --server http://gpu-host.lan:8080
```

No auth is enforced by the wrapper; put the server on a trusted network
or front it with a reverse proxy that does TLS + auth if it leaves the LAN.

---

## How it works

The wrapper is a thin bash script (~60 LOC). Roughly:

```bash
export ANTHROPIC_BASE_URL="$SERVER_URL"
export ANTHROPIC_AUTH_TOKEN="llama-tq"      # any non-empty string works
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
exec claude --model "$MODEL" "$@"
```

All three `DEFAULT_*_MODEL` vars get the same alias because Claude Code maps
model tiers (opus/sonnet/haiku) to separate env vars and your llama-tq
server only serves one model at a time. The wrapper does a `curl /health`
probe first and warns if the server is unreachable.

Read the script: [`scripts/onllama-launch-claude.sh`](../scripts/onllama-launch-claude.sh).

---

## Server tuning for Claude Code (important)

Claude Code sends a large, stable system prompt (tool specs, agentic
instructions — typically 15–25 k tokens) plus the full conversation history
on every turn. Without prompt caching, each turn re-prefills all of it:

- First turn: `hallo` → ~1 min perceived latency (20 k-token prefill on a
  35 B model at PP512 ≈ 900 tok/s ⇒ ~22 s prefill, then decode).
- Multi-turn chats degrade further as history grows.

### `--cache-reuse N` — the single biggest win

llama.cpp can reuse KV-cache across turns via KV-shifting when the new prompt
shares a prefix with a previous one of at least `N` tokens. Add it to your
server invocation:

```bash
./build/bin/llama-server -m model.gguf \
    -c 400000 -ngl 99 -fa on --parallel 2 \
    --cache-type-k ktq2_1 --cache-type-v vtq2_1 \
    --cache-reuse 256
```

**Effect:** on the second turn of a Claude Code session, only the delta
(new user message + assistant response so far) is prefilled. A 20 k-token
system prompt is reused from the KV cache for free. Typical second-turn
latency drops from ~60 s to ~5–10 s on a 35 B model.

Requires prompt caching to be enabled (default in recent llama-tq builds).
`256` is a reasonable floor — smaller chunks aren't worth the shift cost.

### `--parallel N` and Claude Code

`--parallel 2` splits the context into two slots (each `c / 2` tokens), which
is right if two clients share the server. If only Claude Code uses it, drop
to `--parallel 1` to hand it the full context and avoid slot-ping between
turns. Prompt cache reuse works per-slot, so when `--parallel > 1`, requests
from the same client should ideally hit the same slot — llama-tq routes by
matching prefix automatically, no extra config needed.

### Model choice

Claude Code was designed for frontier-tier models. A 35 B runs it, but tool
chains feel sluggish. If latency matters more than depth, host a smaller
model (7 B / 14 B) on the same server and pass `--model your-alias`. The
wrapper sets the same alias for all three Anthropic tier env vars, so
Claude Code will pick it regardless of which internal tier it requests.

### `--keep N` for long sessions

Set `--keep 8192` (or higher) to pin the first N tokens of the prompt across
context shifts. Claude Code's system prompt is typically 15-25k tokens and
gets silently discarded when the ctx-window fills. `--keep 8192` protects
at least the first 8k of it, keeping tool specs and safety instructions alive.

### Prompt-caching (Anthropic `cache_control`)

The `/v1/messages` endpoint returns the full Anthropic-spec `usage` object,
including `cache_read_input_tokens`, `cache_creation_input_tokens`, and the
`cache_creation.ephemeral_5m/1h_input_tokens` breakdown. Today only the
read-path is populated (from llama.cpp's existing `--cache-reuse` KV-shift
mechanism); the creation-path counters are present but zeroed pending the
full `cache_control` implementation. Anthropic-strict clients won't error on
missing fields.

---

## Troubleshooting

**`claude: command not found`** — install from
[docs.claude.com/en/docs/claude-code](https://docs.claude.com/en/docs/claude-code),
or place the binary at `~/.claude/local/claude`.

**`warning: $SERVER_URL/health unreachable`** — the server isn't running,
isn't bound where you think, or the SSH tunnel died. Test with
`curl $SERVER_URL/health`.

**Claude Code hangs on first request** — the model isn't loaded yet
(llama-tq lazy-loads weights on the first `/v1/messages` call with some
builds). Send one request to warm it, then retry.

**Tool-use / function-calling edge cases** — llama-tq's `/v1/messages`
maps to the same OpenAI-compatible chat pipeline as `/v1/chat/completions`.
Use `--jinja` on the server for models whose chat template encodes tool
calls (most modern instruct models do).

---

## Trademark

"Claude" is a trademark of Anthropic, PBC. This wrapper runs Anthropic's
own Claude Code binary against a local Anthropic-compatible server; it
does not redistribute or modify Claude Code itself.
