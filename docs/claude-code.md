# Claude Code against a llama-tq server

The server has an Anthropic-compatible `/v1/messages` endpoint (from upstream llama.cpp; this fork adds the
`cache_control` handling described below), so Anthropic's Claude Code CLI can talk to a local llama-tq server instead of
`api.anthropic.com`.

The `scripts/llama-tq-launch-claude.sh` wrapper sets the required environment
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
./scripts/llama-tq-launch-claude.sh
```

Defaults: `--server http://localhost:8080`, `--model local` (also settable as `LLAMA_TQ_SERVER_URL` and
`LLAMA_TQ_MODEL`). The server serves its loaded model whatever alias is sent.

Override either:

```bash
./scripts/llama-tq-launch-claude.sh --server http://127.0.0.1:8081 --model my-model
```

Pass extra args to Claude Code after `--`:

```bash
./scripts/llama-tq-launch-claude.sh -- --print "say hello"
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
./scripts/llama-tq-launch-claude.sh --server http://localhost:8080
```

### Option B — direct LAN

Bind the server with `--host 0.0.0.0` (or a specific LAN IP) and point
Claude Code at it:

```bash
./scripts/llama-tq-launch-claude.sh --server http://<server-ip>:8080
```

No auth is enforced by the wrapper; put the server on a trusted network
or front it with a reverse proxy that does TLS + auth if it leaves the LAN.

---

## How it works

The wrapper is a short bash script. Roughly:

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

Read the script: [`scripts/llama-tq-launch-claude.sh`](../scripts/llama-tq-launch-claude.sh).

---

## Server tuning for Claude Code (important)

Claude Code sends a large, stable system prompt (tool specs, agentic
instructions — typically 15–25 k tokens) plus the full conversation history
on every turn. The first turn has to process all of it: at 900 tokens/s prompt processing a 20k-token prompt takes
about 22 s before the first token. Later turns reuse the KV cache of the common prefix (llama.cpp prompt caching, on by
default), so only the new part is processed.

### `--cache-reuse N`

When the prompt changes in the middle (for example after Claude Code compacts the history), `--cache-reuse N` also
reuses matching chunks of at least `N` tokens after the change by shifting their KV entries:

```bash
./build/bin/llama-server -m model.gguf \
    -c 131072 -ngl 99 -fa on --parallel 2 \
    --cache-type-k q8_0 --cache-type-v vtq4_1 \
    --cache-reuse 256
```

`256` is a reasonable floor; smaller chunks are not worth the shift. We have not measured how much it saves in a Claude
Code session.

### `--parallel N` and Claude Code

`--parallel 2` splits the context into two slots (each `c / 2` tokens), which
is right if two clients share the server. If only Claude Code uses it, drop
to `--parallel 1` to hand it the full context and avoid slot-ping between
turns. Prompt cache reuse works per-slot, so when `--parallel > 1`, requests
from the same client should ideally hit the same slot — llama-tq routes by
matching prefix automatically, no extra config needed.

### Model choice

Pass the model alias with `--model your-alias`. The wrapper sets the same alias for all three Anthropic tier env vars, so
Claude Code uses it regardless of which internal tier it requests. Claude Code needs reliable tool calls; test the model
on your own tasks.

### When the context is full

Context shift is off by default, so a request that does not fit the context fails instead of silently dropping the
start of the prompt. Give the server enough context (`-c`) for the system prompt
plus the conversation. If you enable `--context-shift`, also set `--keep N` so the system prompt at the start is kept.

### Prompt-caching (Anthropic `cache_control`)

With `--slot-save-path DIR` set, `/v1/messages` honours `cache_control` breakpoints for non-streaming requests: the
slot state is saved to disk under a key for the prefix up to the last breakpoint (TTL from the breakpoint, 5 min by
default, `--anthropic-cache-ttl-default`), a later request with the same prefix restores it before prefill, and `usage`
reports `cache_creation_input_tokens` and `cache_read_input_tokens`. `--anthropic-cache off` disables it.

Limits: Claude Code streams its requests, and streaming requests are not saved yet, so for Claude Code this path
currently does nothing beyond the normal in-memory prefix reuse. Only slot 0 is used. `--anthropic-cache-max-gb` is
stored but not enforced yet.

---

## Troubleshooting

**`claude: command not found`** — install from
[docs.claude.com/en/docs/claude-code](https://docs.claude.com/en/docs/claude-code),
or place the binary at `~/.claude/local/claude`.

**`warning: $SERVER_URL/health unreachable`** — the server isn't running,
isn't bound where you think, or the SSH tunnel died. Test with
`curl $SERVER_URL/health`.

**Claude Code seems to hang on the first request** — the first turn processes the whole system prompt and tool list
(see above); watch the server log for the prompt progress. If the server was just started, wait until `/health`
returns `{"status":"ok"}`.

**Tool-use / function-calling edge cases** — llama-tq's `/v1/messages`
maps to the same OpenAI-compatible chat pipeline as `/v1/chat/completions`.
Use `--jinja` on the server for models whose chat template encodes tool
calls (most modern instruct models do).

---

## Trademark

"Claude" is a trademark of Anthropic, PBC. This wrapper runs Anthropic's
own Claude Code binary against a local Anthropic-compatible server; it
does not redistribute or modify Claude Code itself.
