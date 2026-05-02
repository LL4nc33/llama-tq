# v8 Final Deployment Status (2026-05-02 07:40)

## Triple-Goal Prio (User-Konditionen)

1. **Accuracy** (PPL drift) — höchste Prio
2. **Speed** (TG t/s) — Floor 70+ darf nicht regressed werden
3. **VRAM** — niedrigste Prio, aber **kein OOM**

## Aktueller Deploy-Status

### 35B-A3B GPU0:8791 — bleibt LEGACY ktq2_1/vtq2_1

**Why nicht v8 ktq2/vtq2?**
- v8 vtq2/vtq3 bracht **deferred-V staging fp16 buffer** at full kv_size
- Bei ctx=100k + mmproj (857 MB) + IQ2_XXS (9 GB) + KV → OOM beim mmproj-load
- Sweep-Daten waren bei ctx=2048 — bei 100k skaliert deferred-V buffer linear
- VRAM: 11699 used / 134 free — KEIN Headroom für extra deferred-V buffer

**Triple-Goal Trade-off:**
- Kondition #3 (kein OOM) blockt v8 auf 35B+100k+mmproj
- Bleibt auf legacy ktq2_1/vtq2_1 (3.0 bpw, +3.85% drift, 85.66 t/s)
- Alternative: ctx auf 65k reduzieren → würde v8 ermöglichen, verliert aber 35k ctx

**Files:**
- Active: `scripts/deploy-35b-singlegpu-100k.sh` (legacy, prod)
- Available: `scripts/deploy-35b-singlegpu-100k-v8.sh` (v8 sweet-spot, OOM bei 100k+mmproj)

### 4B-Q4_K_M GPU1:8793 — V8 SWEET SPOT live

**`ktq2/vtq2` (= ktq2_1/vtq2_2 trellis)**
- PPL: 8.6643 (-0.35% vs f16) — vs legacy +0.20%
- TG: 79.59 t/s — vs legacy 78.70
- VRAM: 2.78 bpw avg — vs legacy 5.0 bpw
- **Alle 3 Konditionen verbessert!**

**v8 Sweet Spot beweist:**
- 4B-dense braucht NICHT vtq4_1 (alte Annahme widerlegt)
- Trellis-V (vtq2_2) gewinnt auch auf 4B
- 44% VRAM-Einsparung erlaubt Chatterbox TTS coexist

**Files:**
- Active: `scripts/deploy-4b-gpu1-v8.sh` (v8 sweet-spot, prod)
- Legacy: `scripts/deploy-4b-gpu1.sh` (alte ktq2_1/vtq4_1)

## VRAM Status

| GPU | Used | Free | Service |
|---|---|---|---|
| GPU0 | 11699 MB | 134 MB | 35B-A3B legacy + mmproj 100k |
| GPU1 | 10293 MB | 1540 MB | 4B-Q4_K_M v8 + (Chatterbox space available) |

## Deploy-Scripts haben jetzt ALLE Flags

### 35B v8 (alle prod-flags + mmproj)

```
--mmproj $MMPROJ
--jinja --flash-attn on
-c 100000 -ngl 99 --no-mmap --parallel 1
--cache-type-k ktq2 --cache-type-v vtq2
--cache-reuse 25000
--predict 16384 -ub 64 --reasoning off
--moe-pin-experts --backend-sampling
--slot-save-path $SLOTS
--anthropic-cache 1 --anthropic-cache-ttl-default 300 --anthropic-cache-max-gb 32
--temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.15
--override-kv general.name=str:OidaNice-GPT-34B
```

### 4B v8 (alle prod-flags)

```
--jinja --flash-attn on
-c 100000 -ngl 99 --no-mmap --parallel 1
--cache-type-k ktq2 --cache-type-v vtq2
--cache-reuse 1024
--predict 8192 -ub 128 --reasoning off
--backend-sampling
--temp 0.7 --top-p 0.9 --repeat-penalty 1.1
--override-kv general.name=str:OidaNice-GPT-4B
```

## Future Work

1. **35B v8 auf 100k+mmproj möglich machen:**
   - Option A: Custom build mit reduziertem deferred-V buffer (Trick #4 Correction Overlay statt full staging)
   - Option B: ctx 65k v8 deployment als secondary port
   - Option C: vtq2_1 (codebook, kein deferred-V) statt vtq2_2 — würde bei 35B regressed sein
2. **4B Triple-Goal Memory update** — alte memory `project_qwen35_4b_triple_goal.md` mit "4B braucht vtq4" ist OUTDATED
3. **Pareto-Chart regenerate** mit neuen Daten
4. **80B / 122B / Gemma4 Sweet-Spot Sweep** (lower priority)
