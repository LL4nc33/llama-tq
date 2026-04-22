#!/bin/bash
# onllama-launch-claude: launch Anthropic's Claude Code CLI pointed at a
# local llama-tq server instead of api.anthropic.com.
#
# Usage:
#   onllama-launch-claude                          # local server, default model
#   onllama-launch-claude --server URL             # custom server URL
#   onllama-launch-claude --model NAME             # override model name
#   onllama-launch-claude -- [claude-code args...] # pass-through after --
#
# Requires: Anthropic's `claude` CLI installed (https://docs.claude.com/en/docs/claude-code)
# and a llama-tq server running with /v1/messages support on the target URL.
#
# Works against:
#   - any llama-tq build with the Anthropic-compat routes (default since 2026-04)
#   - on-llama-tq
#   - any other llama.cpp fork exposing POST /v1/messages
#
# Inspired by ollama's `ollama launch claude` (MIT, github.com/ollama/ollama).
# "Claude" is a trademark of Anthropic, PBC. This launcher runs Anthropic's
# own Claude Code binary against a local Anthropic-compatible server.

set -euo pipefail

SERVER_URL="${ONLLAMA_SERVER_URL:-http://localhost:8080}"
MODEL="${ONLLAMA_MODEL:-qwen3.6-35b-a3b}"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server|--server-url)
            SERVER_URL="$2"; shift 2;;
        --model)
            MODEL="$2"; shift 2;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//' | head -25
            exit 0;;
        --)
            shift; PASSTHROUGH=("$@"); break;;
        *)
            PASSTHROUGH+=("$1"); shift;;
    esac
done

CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
if [[ -z "$CLAUDE_BIN" ]]; then
    for candidate in "$HOME/.claude/local/claude" "$HOME/.claude/local/claude.exe"; do
        [[ -x "$candidate" ]] && CLAUDE_BIN="$candidate" && break
    done
fi
if [[ -z "$CLAUDE_BIN" ]]; then
    echo "error: claude binary not found on PATH or at ~/.claude/local/" >&2
    echo "install from https://docs.claude.com/en/docs/claude-code" >&2
    exit 1
fi

if ! curl -fsS --max-time 3 "$SERVER_URL/health" >/dev/null 2>&1; then
    echo "warning: $SERVER_URL/health unreachable — claude may fail at first request" >&2
fi

export ANTHROPIC_BASE_URL="$SERVER_URL"
export ANTHROPIC_API_KEY=""
export ANTHROPIC_AUTH_TOKEN="llama-tq"
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"

echo "launching claude -> $SERVER_URL (model alias: $MODEL)" >&2
exec "$CLAUDE_BIN" --model "$MODEL" "${PASSTHROUGH[@]}"
