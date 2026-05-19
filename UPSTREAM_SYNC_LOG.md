# Upstream Sync Log — 2026-05-19

Branch: `upstream-sync-2026-05-19`
Base: `turboquant`
Upstream: `ggml-org/llama.cpp` master (fetched 2026-05-19)

## Picked Commits (8)

| SHA (cherry) | Upstream SHA | Subject |
|---|---|---|
| ac9ac1e75 | 8cef8201a | CUDA: directly include cuda/iterator (#22936) |
| f7202d9de | 84c678242 | CUDA: Continue directly including cuda/iterator (#23102) |
| 9c969ae7b | dd7cad719 | cmake : do not check for bin install dir (#23234) |
| 77a0c634e | e0de4c241 | cmake : do not install conversion script (#23204) |
| 956b26b67 | ff6b1062a | server : fix hardcoded proxy connection timeout in router mode (#22003) |
| a75de6e30 | e77056f9b | CUDA: use fastdiv for batch index split in get_rows (#22650) |
| b8a587b83 | 86db42e97 | CUDA: fuse relu + sqr (#22249) |
| d5b0796a0 | 726704a16 | feat: Support d_conv=15 for ssm-conv.cu (#23017) |

All commits include `(cherry picked from commit ...)` attribution via `-x`.

## Skipped Commits (Conflicts / Already-in-tree)

| Upstream SHA | Subject | Reason |
|---|---|---|
| 8e1f9d083 | CUDA: handle OW > 65535 in im2col | Already in tree |
| e93666076 | Ggml/cuda snake fusion hardening | Conflict in `ggml-cuda.cu` (custom code) |
| 7f3f843c3 | Fix issue #22974 cast to float | `allreduce.cu` deleted in our tree |
| 1a68ec937 | server : honor --embd-normalize CLI | Already in tree |
| c78fb909b | server: fix heap-buffer-overflow CVE-2026-21869 | Already in tree |
| ffdd983fb | server : fix swa-full logic | Conflict in `server-context.cpp` (Phase D) |
| ccee42642 | server-context: guarantee 1 token decode | Conflict in `server-context.cpp` (Phase D) |
| 64b38b561 | server: skip device enum in router mode | Conflicts in `server.cpp`, `common.cpp` |
| b64739ea3 | server: (router) alloc tmp buffer on heap | Conflict in `server-models.cpp` |
| 67b2b7f2f | logs : reduce | Conflict in `server.cpp` |
| b1a5bd4e0 | CUDA: better coalesce concat | Empty after merge (already-equivalent) |
| 9725a313b | CUDA: reduce MMQ stream-k overhead | Empty after merge (already-equivalent) |

## Build Verification

Local WSL environment: cmake not installed — full build verification deferred to test-rig.
Changes are limited to:
- Header includes (CUDA iterator) — compile-only impact
- CMakeLists tweaks (install dir checks, conversion script)
- Isolated CUDA op kernels: `getrows.cu`, `ssm-conv.cu`, `unary.cu` (relu+sqr fusion)
- Server: single-line timeout constant in `server-models.cpp`

None of the picked commits touch TurboQuant FA kernels, ggml-opt, or `llama-context.cpp` graph-sizing.

**Recommended next step:** `git push` to test-rig and run a CUDA build + smoke bench
(qwen3.5-0.8b-q8_0) before merging to `turboquant`.

## Deferred / Backlog (case-by-case)

These large/risky upstream features are NOT cherry-picked and should be reviewed
individually before integration:

- **MTP Support** (#22673, #23198) — multi-token prediction; large surface area
- **HIP RDNA3 mma FA** (#22880) — touches FA which is TurboQuant-owned territory
- **Snake fusion hardening** (#22912) — needs manual merge with our `ggml-cuda.cu`
- **Vulkan changes** — separate `vulkan` branch
- **CUDA graph LRU eviction** (#21611) — broad refactor
- **NCCL/TP refactors** (#21891, #22299) — TP is dormant here
- **Vertex AI / Codex compat** — server feature, evaluate need

## Notes

- No merge commits picked.
- No webui changes picked (separate velocity).
- All picked commits validated by upstream CI; local CUDA build verification
  pending on test-rig.
