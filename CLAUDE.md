IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

---

# llama-tq Fork — Working Rules for Claude

> Upstream llama.cpp contributor rules live in [AGENTS.md](AGENTS.md). The rules below are **fork-specific** and apply to all work on this repository.

## 1. Language & Marketing

- **Banned words in own writing** (docs, blog posts, commit messages, comments):
  `production`, `prod`, `first-class`, `Karpathy-style`, `world's first`, `the only`,
  `First inference engine`, `likely the first`, `literally`, `groundbreaking`, `blazing`.
- **Replacements:** `production` / `prod` → `default`, `current`, `deploy`, `live`, `stable`, `recommended`, `service`. `first-class` → `independent`.
- **Vendored upstream content is untouched** — code identifiers (`int32x4_t prod`,
  `np.prod`, `out-prod.cuh`, Philox `prod` variable), `ggml/`, `src/`, `common/`,
  `examples/`, `tools/server/public/bundle.js`, `docs/android.md`, the
  `karpathy/tinyllamas` URLs in `.github/workflows/build.yml`, and any
  `production` strings inside `tests/test-chat.cpp`.
- **Headlines must be factual, not rhetorical:** *"What this fork adds on top of upstream"*
  beats *"What this fork does that no other does."*
- **Article matters:** *"A llama.cpp fork with..."* — never *"The llama.cpp fork..."*.

## 2. Public vs. Private

- `LEGION/` and `autoresearch/` are **gitignored and never tracked**. Internal
  brainstorming, ABORTED-pivot notes, and raw experiment logs stay local.
- When public-relevant findings emerge from those directories, write a **curated**
  doc under `docs/research/` or `docs/` — do not push the raw notes 1:1.
- Examples of curated docs: `docs/research/moe-expert-locality.md` (extracted from
  Phase 6 LEGION notes), `docs/autoresearch.md` (loop concept without internal data).

## 3. Bench Numbers & Claims

- **Math must be correct.** VRAM saved = `1 − bpw / 16`, ratio = `16 / bpw`.
  Past mistake: 2.78 bpw labelled "91% saved" — actual 83%. Always re-derive.
- **No headline harness mixing.** `llama-bench tg128`, live server-pps, and
  wallclock t/s are different numbers. If the headline says "86 t/s @ 100k+mmproj
  single-GPU", the configuration the number was measured under must match the claim.
- **bpw values stay consistent across the doc.** `vtq3_v8` is 3.625 bpw — not 3.56,
  not 3.6. Pick one form and use it everywhere.
- **Numbers come from `bench/plots/benchmarks.csv` and `bench/results/*.tsv`** — these
  are the single sources of truth. Running text must agree with them.

## 4. Git & GitHub

- **Commit email:** `85998242+LL4nc33@users.noreply.github.com`. This is the
  noreply form for GitHub user `LL4nc33` (id 85998242) — required for the avatar
  to render on GitHub. Already persisted in `.git/config` on `gpu00:~/llama-tq`.
- **Commit name:** `LL4nc33`.
- **No `Co-Authored-By:` lines.** Strip any tool-default trailer.
- **Push path:** `gpu00` has no GitHub credential. Workflow for push:
  1. clone the repo to a tmp dir on the WSL side,
  2. `git fetch claude@gpu00.node:/home/claude/llama-tq turboquant`,
  3. `git merge --ff-only FETCH_HEAD` (or `git reset --hard FETCH_HEAD`),
  4. `git push origin turboquant` (force-push if history was rewritten).
- **Force-pushes are explicit.** Only when the user asks to remove a commit
  "spurlos" or to rewrite. Otherwise FF-merge.

## 5. Hardware Phrasing (consistent vocabulary)

- "RTX 2060 12 GB", "Turing CC 7.5", "KVM-VM auf 3700X-Host", "asymmetric PCIe (x16/x4)".
- Two GPUs — `GPU0` is the x16 primary, `GPU1` is x4. Layer-split `-ts 17,7`
  reflects this asymmetry.

## 6. Verification Before Editing Docs

- Researcher / sub-agent line-number reports may come from a stale clone.
  Always verify with `ssh claude@gpu00.node "sed -n 'L,Lp' README.md"` before
  applying edits.
- Check internal links exist (`ls docs/...`) before referencing them in README.

## 7. Do Not Touch

- Vendored upstream code (`ggml/`, `src/`, `common/`, `examples/`, most of
  `tools/`, the `bundle.js` blob, the Karpathy llama2.c references in CI yaml,
  ggml header `first class` text, vendored chat-template tests).
- External proper names: `karpathy/tinyllamas`, `karpathy/llama2.c`, etc.
- C / Python / SIMD identifier variables that happen to be named `prod`.

## 8. Roadmap Discipline

- v6 work begins on a **clean base** (the current state after this round of
  cleanup). Do not start v6 implementation while there are still naming /
  marketing fixes pending on the README.
- Roadmap source of truth: `docs/plans/2026-05-05-v6-roadmap.md`. Verdict:
  Trellis-K is **dropped** (incompatible with the Hadamard-domain Q·K USP);
  performance levers (mmvq `__dp4a` tiling, S199 Sparse-K-Skip, FA register
  pressure audit) are the v6 scope.

## 9. SSH & Service Map (gpu00)

- `gpu00.node:8791` — `OidaNice-GPT-34B` (Qwen3.6-35B-A3B-IQ2_XXS hardlink),
  default deploy. KV-cache is `ktq2 + vtq2_2` since the 2026-05-03 EOS-cutoff fix
  (the legacy `ktq2_1 + vtq2_1` shows mid-generation cutoffs on long contexts
  with the S199-plumbing build).
- `gpu00.node:8793` — `nomic-embed-text-v1.5` F16 embeddings server, GPU1, 380 MB,
  systemd-user service with autostart.
- `gpu00.node:8792` — `functiongemma-270m`, CPU.
- VM `on-agents` (192.168.178.90, user `claude`) hosts opencode + Hermes Agent +
  Claude Code (with `cwe -l` alias for the llama-tq backend).

## 10. When in Doubt

- Read the full file before editing it (`Read` tool).
- Do not apply mass-`sed` to vendored upstream content.
- If a researcher / agent reports findings, **verify on the actual repo state**
  before acting. Stale clones are common.
