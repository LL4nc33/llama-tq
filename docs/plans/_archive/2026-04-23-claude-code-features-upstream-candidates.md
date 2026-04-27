# Phase4 Claude Code Features — Upstream Candidate Analysis

Draft analysis of which Phase4 features are candidates for submission to
upstream llama.cpp (ggml-org/llama.cpp) vs keeping fork-local. Gitignored
planning doc.

## Candidates

| Feature | Commit | Upstream-fit | Rationale |
|---------|--------|--------------|-----------|
| Anthropic usage response fields | `43660e4c4` | **Yes** | Pure compliance with Anthropic API spec. Zero risk, zero behavior change for non-Anthropic clients. Clean 16-LOC diff. |
| Tool-call early-stop | `5710a9e05` | **Yes, with scope widening** | Upstream maintainers likely want this as an opt-in flag or gated on detected template family. Needs generalization beyond Qwen-only. ~50 LOC total. |
| 400 error on bad anthropic request | `1a28e0ec3` | **Yes** | Correctness fix (was 500 for client error). Trivial. |
| Anthropic prompt-cache breakpoint parsing | `0d1d8d8d2` | **Depends on Phase 2** | Phase 1 alone emits `__anthropic_cache` for a consumer that doesn't exist upstream. Only ship together with the full KV persistence (Phase 2). |
| `--cache-reuse 256` config guidance | docs-only | N/A | Documentation, not code. Just note in README. |
| Single-GPU prod config (expert-offload + 200k ctx) | docs-only | N/A | Documentation of an existing upstream feature combo. |
| TCP_NODELAY for SSE | `1b0b3c1ac` | **Yes, strong candidate** | Fixes real streaming UX issue. Zero-risk. 5-line diff. |
| gzip response compression | `004ff57b9` | **Maybe** | Opt-in build flag. Some upstream maintainers dislike extra build dependencies. If ZLIB is found-but-optional, accepts. |
| `onllama-launch-claude.sh` wrapper | `e132e560e` | **No** | Project-specific branding + defaults (`qwen3.6-35b-a3b` model alias). Keep fork-local. Could contribute a generic `tools/server/examples/launch-claude.sh` template. |
| Full Anthropic prompt-cache (cache_control KV persistence) | Phase 2 WIP | **Yes, after validation** | Fits within existing `--slot-save-path` infrastructure. High-value feature for Anthropic-API consumers. Requires extensive testing before submission. |

## Submission Priority (if pursuing upstream)

1. **TCP_NODELAY for SSE** — smallest, most obvious, fixes real user pain. Good first PR.
2. **Anthropic usage response fields** — cleanup of existing Anthropic-compat code.
3. **400 instead of 500 on bad request** — bug fix.
4. **Full Anthropic prompt-cache (Phase 2)** — after working + tested + documented.
5. **Tool-call early-stop (generalized)** — after discussion with maintainers on how template-specific stop-injection should be exposed.

## Not candidates (fork-only)

- KTQ/VTQ quantization types — explicit research fork territory.
- `--keep 8192` default in `start-prod-single.sh` — deployment-level, not code.
- Single-GPU deploy script — operational.

## Non-goal

This is not a roadmap commitment to submit any of these. Just a clarity
document — if/when we have time to engage with upstream review, this is
the ordered list. Fork continues to be the primary home.
