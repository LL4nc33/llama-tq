# Anthropic-compatible Prompt Caching on `/v1/messages`

Date: 2026-04-23
Repo: `/mnt/d/repos/llama-tq` (branch `turboquant`, HEAD 7f4993cd5)
Target: real Anthropic clients (Claude Code CLI) sending `cache_control: {type: "ephemeral"}` on
content blocks in `system` / `messages` / `tools` against our `/v1/messages` endpoint.

## 1. What we already have

- `/v1/messages` handler: `tools/server/server-context.cpp:3762-3778` — parses Anthropic JSON →
  `convert_anthropic_to_oai()` → `oaicompat_chat_params_parse()` → `handle_completions_impl(..., TASK_RESPONSE_TYPE_ANTHROPIC)`.
- Anthropic → OAI converter: `tools/server/server-common.cpp:1490-1690` (strips `cache_control` today — it
  just drops anything it doesn't recognize inside `system` / content blocks).
- Anthropic response shape with partial cache accounting: `tools/server/server-task.cpp:1122-1186` and
  streaming `:1680-1710`. Already emits `cache_read_input_tokens` and `input_tokens` from
  `n_prompt_tokens_cache` (filled by automatic KV reuse — `--cache-reuse`, `--slot-prompt-similarity`).
  Missing: `cache_creation_input_tokens` and the `cache_creation.ephemeral_5m_input_tokens` /
  `ephemeral_1h_input_tokens` breakdown.
- Slot save/restore infra: `post_slots` (`:3478`), `handle_slots_save` (`:4072`),
  `handle_slots_restore` (`:4108`), `handle_slots_erase` (`:4145`); tasks `SERVER_TASK_TYPE_SLOT_SAVE/RESTORE/ERASE`
  in `server-task.h:24-25`. Requires `--slot-save-path`. Files written as `params.slot_save_path + filename`.
- Upstream already does best-effort prefix reuse across slots via `slot_prompt_similarity`
  (`server-context.cpp:968-999`). For many realistic caching flows this is almost enough —
  the Anthropic layer is mostly about **reporting cache accounting correctly and honoring TTL semantics**,
  not re-inventing KV reuse.

## 2. Design

### 2.1 Request parsing (augment `convert_anthropic_to_oai`)

In `server-common.cpp:1490` extend the converter to collect breakpoints:

- Walk `system[]`, `messages[*].content[]`, `tools[*]` in the ORIGINAL Anthropic order. Whenever a
  block has `cache_control`, record a `CacheBreakpoint { scope, prefix_end_byte_offset, ttl }` where
  `ttl ∈ {5m, 1h}` from `cache_control.ttl` (default `5m`).
- Also accept top-level `cache_control` → emits an automatic breakpoint on the last cacheable block.
- Cap at 4 breakpoints; reject with HTTP 400 (`ERROR_TYPE_INVALID_REQUEST`) if exceeded.
- Pass the breakpoint list back via a new field on the parsed body, e.g.
  `oai_body["__anthropic_cache"] = [{prefix_tokens: -1, ttl_sec: 300}, ...]`
  (tokens resolved post-tokenization). Keep the field internal (strip before sending to chat template).

### 2.2 Cache-key strategy

For each breakpoint `i`, compute:

```
key_i = SHA256(
    model_id_or_hash          // ggml model file sha or meta->model_name + params
    || chat_template_hash     // hash of the rendered template prefix used
    || prefix_bytes_up_to_i   // canonical: the tokenized prompt prefix, NOT raw JSON
)
```

Canonicalizing on **tokenized prefix** not raw JSON avoids false misses from whitespace / key order.
Tokenization already happens in `oaicompat_chat_params_parse`; we hash the token-id array slice
`[0..breakpoint_tokens_i]` plus a version byte.

### 2.3 Storage layout

Reuse `--slot-save-path` infrastructure (no new save/restore plumbing). Filenames:

```
<slot_save_path>/anthropic-cache/<key_hex>.bin
<slot_save_path>/anthropic-cache/<key_hex>.meta   (json: {n_tokens, ttl_sec, created_at, token_ids_hash})
```

New server-side in-memory index `std::unordered_map<std::string, CacheEntry>` guarded by a mutex,
seeded from directory listing on startup. Bound: LRU cap (e.g. 256 entries) + disk-quota config
`--anthropic-cache-max-gb` (default 32).

### 2.4 Request flow

In `post_anthropic_messages` BEFORE `handle_completions_impl`:

1. Tokenize prompt (done by `oaicompat_chat_params_parse` already).
2. Compute `key_i` for every breakpoint; look up in index. Pick the **longest-prefix hit** (mirrors
   Anthropic's "longest match wins").
3. If hit:
   a. Pick a slot (use existing slot-selection path). Call the same kernel as `handle_slots_restore`
      programmatically (factor out a `restore_from_file(id_slot, filepath)` helper in
      `server-context.cpp` around `:4108`) to load KV up to `n_tokens_hit`.
   b. Mark the task with `n_prompt_tokens_cache_hit = n_tokens_hit` so downstream reuse skips re-prefill
      for those tokens. Only the delta `[n_tokens_hit..n_prompt_tokens)` is prefilled.
4. Run inference as normal.
5. AFTER completion, for every breakpoint whose prefix was NOT already hit (i.e. newly created),
   issue a `SERVER_TASK_TYPE_SLOT_SAVE` with filepath `anthropic-cache/<key_i>.hex.bin`, write `.meta`.
   Track bytes in cache creation counters.

### 2.5 Response enrichment

Modify `server_task_result_cmpl_final` (new fields `n_prompt_tokens_cache_creation_5m`,
`n_prompt_tokens_cache_creation_1h`) plumbed from slot state. In `to_json_anthropic()`
(`server-task.cpp:1122`) replace the current `usage` object with:

```cpp
{"usage", {
  {"cache_read_input_tokens",     n_prompt_tokens_cache},
  {"cache_creation_input_tokens", n_creation_5m + n_creation_1h},
  {"input_tokens",                n_prompt_tokens - n_prompt_tokens_cache - n_creation_5m - n_creation_1h},
  {"output_tokens",               n_decoded},
  {"cache_creation", {
      {"ephemeral_5m_input_tokens", n_creation_5m},
      {"ephemeral_1h_input_tokens", n_creation_1h}
  }}
}}
```

Same for streaming at `server-task.cpp:1680-1710`.

### 2.6 TTL / eviction

Lazy-delete on access: when looking up a key, `stat` the `.meta` file; if
`now - created_at > ttl_sec`, `unlink` both files and treat as miss. Background sweeper is optional
(simple thread scanning every 60s, kick off in `server-context.cpp` init). "Refresh on hit" — Anthropic
resets TTL on read; mirror that by `utime()`-bumping `created_at` on hit.

### 2.7 Invalidation rules

Per Anthropic table: `tools` changes bust `tools+system+messages` caches. Implementation: include
`tools_hash` in the key material for breakpoints in `system`/`messages` too, and include
`system_hash` in `messages` breakpoint keys. Falls out naturally from "hash the tokenized prefix" since
tools/system are rendered before messages in the template.

### 2.8 CLI flags (in `common/arg.cpp`)

- `--anthropic-cache` (bool, default: on if `--slot-save-path` set)
- `--anthropic-cache-ttl-default` (sec, default 300)
- `--anthropic-cache-max-gb` (default 32)

## 3. File-change map

| File | Lines | Change |
|---|---|---|
| `tools/server/server-common.cpp` | 1490–1690 | parse `cache_control`, emit `__anthropic_cache` breakpoints |
| `tools/server/server-common.h`   | +1 struct | `struct anthropic_cache_bp { size_t n_tokens; int ttl_sec; };` + key-hash helper |
| `tools/server/server-context.h`  | new member | `anthropic_cache_manager cache_mgr;` |
| `tools/server/server-context.cpp` (NEW section) | ~+250 LOC | `class anthropic_cache_manager` (LRU, mtime TTL, key hashing, restore/save helpers that wrap existing SLOT_SAVE/RESTORE tasks) |
| `tools/server/server-context.cpp` | 3762 (`post_anthropic_messages`) | pre-lookup + slot restore; post-completion save scheduling |
| `tools/server/server-context.cpp` | 4072, 4108 | factor inner task dispatch into `save_slot_to(id, path)` / `restore_slot_from(id, path)` reused by cache mgr |
| `tools/server/server-task.h` | 349, 418 | add `int32_t n_prompt_tokens_cache_creation_5m`, `_1h` |
| `tools/server/server-task.cpp` | 1122, 1680 | emit full Anthropic usage incl. `cache_creation` breakdown |
| `common/arg.cpp` | new flags | `--anthropic-cache*` |
| `tools/server/tests/unit/test_compat_anthropic.py` | +tests | round-trip with `cache_control`, assert usage fields |

## 4. LOC estimate

- Converter + key hashing: ~120
- Cache manager class + LRU + TTL sweeper: ~250
- post_anthropic_messages wiring: ~80
- Slot task refactor (extract helpers): ~40
- Response usage plumbing: ~50
- Flags + tests: ~150

**Total: ~690 LOC** (~550 C++, ~140 Python tests).

## 5. Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Slot contention when restoring over an in-flight slot | HIGH | Require `slot_prompt_similarity=0` disabled when picking a cache-hit slot; explicitly erase slot before restore |
| Disk quota runaway | MED | LRU cap + byte-quota + sweeper |
| Hash collisions across models if user doesn't set `--alias` | LOW | include model file sha256 in key |
| Anthropic TTL "refresh on hit" semantics not matching `utime()` behavior on some FS | LOW | fall back to rewriting `.meta` |
| Race between two concurrent requests both missing same key → both create | LOW | dedupe map of "in-flight creation keys"; second request waits on first's future |
| Non-text blocks (images) in `cache_control` scope | MED | v1: only cache pure-text prefixes; skip breakpoint if any image precedes it (return as normal uncached) |
| Streaming: cache save must happen AFTER final chunk flushed | MED | hook into existing final-task path at `server-task.cpp:1122` / `:1680`; schedule SLOT_SAVE as post-completion task |
| `--slot-save-path` required — users without it | LOW | feature off by default, clear error message |

## 6. Non-goals (v1)

- Cross-node / distributed cache (single server only).
- 1h TTL pricing accounting (we just tag + report; no billing).
- Caching tool results / images.
- Partial-prefix fuzzy match (only exact token-prefix hash match).

## 7. Next step for implementer

Start with section 2.1 + 2.5 only (parse `cache_control`, plumb `cache_creation_input_tokens` field
zeroed, add `cache_creation` object to response). That's a 2h change that makes real Claude Code CLI
stop erroring on missing fields. Then layer 2.2-2.4 behind `--anthropic-cache` flag.
