# Upstream-PR submission guide (for LL4nc33)

## What we have ready

In `docs/plans/`:
- `upstream-PR-server-spec-mmproj-coexistence.md` — internal draft (DO NOT copy verbatim)
- `upstream-PR-server-spec-mmproj-coexistence.patch` — git format-patch ready to apply

On the build server:
- `${WORK_DIR}/llama-cpp-upstream/` cloned fresh from ggml-org/llama.cpp master
- Branch `phase41b-spec-mmproj-coexistence` committed (commit f28bc3740)
- Build verified clean, functional test passed

## What you need to do

The upstream contribution policy (CONTRIBUTING.md in llama.cpp) explicitly bans:
- PRs that are fully or predominantly AI-generated
- AI-written PR descriptions, GitHub posts, or maintainer responses

So before submitting, you (the human contributor) must:

### 1. Read the actual code change

```
ssh <build-server>
cd ${WORK_DIR}/llama-cpp-upstream
git checkout phase41b-spec-mmproj-coexistence
git diff master
```

Read every line. Understand why each `has_mtmd` → `has_media()` switch is safe.
Make sure you can answer maintainer questions about each change.

### 2. Write the PR title + description in your own words

Topic: enable speculative decoding (+ cache_reuse + ctx_shift) for text-only requests on
mmproj-loaded servers. Currently those features are globally disabled when mmproj loads,
forcing users to choose vision OR these features.

The fix is a per-request gate: distinguish the slot-capability flag (`has_mtmd`, set when
mmproj is loaded) from per-request content (new `has_media()` accessor that returns
`!map_idx_to_media.empty()`). Vision requests still skip those features; text-only
requests on the same server get the full feature set.

Keep the description short. Reference the patch's commits. Don't paraphrase the AI draft.

### 3. Disclose AI usage honestly

CONTRIBUTING.md asks for clarity. Something like:
> "AI was used to suggest the per-request gate pattern and to format the comments.
> I reviewed and rewrote the code. The architecture decision (introduce `has_media()`
> rather than thread per-request flags through every callsite) is mine."

…if that matches reality. Otherwise be more specific.

### 4. Push and open PR

From a local GPU host (or after pulling the branch to your local clone):

```
git push <your-fork> phase41b-spec-mmproj-coexistence
# then open PR via github web UI against ggml-org/llama.cpp:master
```

Maintainers may want:
- The functional test results (already in `upstream-PR-server-spec-mmproj-coexistence.md`)
- A ggml CI run
- Discussion of whether `has_media()` should live on `server_tokens` or be derived elsewhere

### 5. If you don't want to submit

That's fine. The patch lives in our fork. Production deployment doesn't depend on upstream
accepting it. The fork-local commits (240821bb4..cf0f6e4a2 on feature/mtp-shared-ctx) are
self-contained.

## Patch stats

3 files changed, +35 / -16 lines:
- `tools/server/server-common.h`: add `has_media()` accessor (7 lines)
- `tools/server/server-common.cpp`: relax 2 asserts to use `has_media()` (4 lines)
- `tools/server/server-context.cpp`: gate 5 paths per-request + lift 3 global blocks
  (24 lines net)

No new dependencies, no TQ/MTP-specific code, pure upstream-server-side change.

## Why this is worth submitting

A search of open PRs and issues (Jun 8 2026) found no existing work on this — closest are
#21815 (spec ngram reinit) and #22787 (spec ctx refactor), neither addresses the
mmproj-loaded server case.

Users with vision-capable models who want spec-decoding for their text-only requests
currently can't have both. This unblocks them with no API changes and no regression on
non-mmproj deploys.
