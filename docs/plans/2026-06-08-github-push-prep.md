# Pre-github-push scrub status (2026-06-08)

## Current state

- **gitea (private):** has all session commits 8f2754aa9..c63e585ea (Phase 27-45+).
- **github (public):** still at 193129086d (pre-session). Branch `feature/mtp-shared-ctx`
  is **not yet pushed to github** for this session.

## Private patterns found in session commits

A grep over Phase 27-45 commits (`8f2754aa9^..HEAD`) found:

| Pattern | Count | Where |
|---|---|---|
| `/home/lance` | 9 | scripts/deploy-35b-*.sh (MODEL/MMPROJ defaults), docs/plans/* (examples) |
| `/home/claude` | 12 | scripts/deploy-*.sh (LLAMA_BIN/SLOTS defaults), docs/plans/upstream-PR-* |
| `gpu00.node` | 1 | docs/plans/upstream-PR-submission-guide.md (ssh hostname example) |
| `192.168.178` | 0 | clean |
| `claude#00` | 0 | clean |
| `oidanice.at` | 0 | clean |
| `dbg@local` | 0 | clean |

## What needs to happen before any github-push

### Option A: scrub via filter-repo + force-push

```
git filter-repo --force \
  --path-glob "scripts/deploy-35b-*.sh" \
  --path-glob "docs/plans/upstream-PR-*" \
  --replace-text /tmp/scrub-rules.txt

# scrub-rules.txt:
# /home/lance ==> /models
# /home/claude ==> /workspace
# gpu00.node ==> SERVER

git push --force origin feature/mtp-shared-ctx
git push --force origin --tags
```

This rewrites history. Anyone who already pulled from gitea must re-pull.

### Option B: keep session commits gitea-only

Don't push `feature/mtp-shared-ctx` to github. Submit individual cherry-picked
changes to upstream (the upstream-PR patch in `docs/plans/`) directly to
ggml-org/llama.cpp.

The upstream-PR patch itself is **already scrubbed** (no private paths in the
3 server-side files: server-common.h, server-common.cpp, server-context.cpp).
Only the docs around it have private paths, and those don't go upstream.

### Option C: clean rewrite of the dirty files in a single squashed commit

If you want the session work on github but want to avoid filter-repo:
1. Create a fresh branch from the last-clean commit on github
2. Cherry-pick all session changes EXCEPT the dirty files
3. Re-create the dirty files (deploy scripts + plan docs) with `${SOMETHING_HOME}/...`
   placeholders instead of hardcoded paths
4. Single squash commit, push to github

## Recommendation

Option B is simplest. The fork-private deploy scripts and internal plan docs
are inherently personal-config files — they don't gain value from being on
github. Keep them gitea-only.

The upstream-PR patch is the only thing that has genuine public value, and
it's already scrub-clean by construction (no scripts, only the 3 source files).

## What the upstream-PR patch actually contains

```
$ git diff --stat 240821bb4^..cf0f6e4a2 -- tools/server/
 tools/server/server-common.cpp  |  7 +++++--
 tools/server/server-common.h    |  7 +++++++
 tools/server/server-context.cpp | 24 ++++++++++++++----------
 3 files changed, 27 insertions(+), 11 deletions(-)
```

Zero private paths. Ready for upstream as-is once human contributor rewrites
PR description per CONTRIBUTING.md AI policy.
