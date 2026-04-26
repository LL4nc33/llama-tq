# Portfolio Devlog Seeds

Source markdown files für die `/devlog` section auf oidanice.at. Each file uses the format spec'd in `LEGION/portfolio-llamatq/2026-04-26_1118_devlog-section-collab.md`.

## Status der Seeds

| File | Topic | Status |
|---|---|---|
| `01-phase-4-stack-185-tg-80b.md` | Phase 4 +18.5% TG | shipped |
| `02-honest-failures.md` | QJL / Pie-PCIe / spec-decode-MoE | shipped (failure-log) |
| `03-e11-cuda-port-triton-detour.md` | E11 CUDA port + Triton detour | shipped (decision-log) |
| `04-phase-5-xquant-in-flight.md` | XQuant 1.69 bpw target | wip |

## Recommended publish order

1. **02-honest-failures.md** (trust-builder, low-stakes opener)
2. **01-phase-4-stack** (concrete win, headline number)
3. **03-e11-detour** (decision-log style, builds dev cred)
4. **04-phase-5-wip** (ongoing, invites readers to follow journey)

## Hero-number for devlog landing

`75.39 t/s on Qwen3.6-35B-A3B at 2.78 bpw KV cache, 200K ctx, 24 GB consumer VRAM`

(That's the most defensible single line: verified, deployed, peer-reviewed-paper-foundation, no asterisk pending the fattn-tq.cuh rebench.)
