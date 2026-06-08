# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --draft-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --draft-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-size-n 8 --spec-ngram-size-m 8 --spec-ngram-min-hits 2 --draft-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--draft-min` and `--draft-max`

llama-server ... --spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 48 --draft-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

If a draft model is combined with a draftless decoding the draftless decoding has higher precedence.

```
--draft, --draft-n, --draft-max N       number of tokens to draft for speculative decoding (default: 16)
                                        (env: LLAMA_ARG_DRAFT_MAX)
--draft-min, --draft-n-min N            minimum number of draft tokens to use for speculative decoding
                                        (default: 0)
                                        (env: LLAMA_ARG_DRAFT_MIN)
[...]
--spec-type [none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
                                        type of speculative decoding to use when no draft model is provided
                                        (default: none)
--spec-ngram-size-n N                   ngram size N for ngram-simple/ngram-map speculative decoding, length
                                        of lookup n-gram (default: 12)
--spec-ngram-size-m N                   ngram size M for ngram-simple/ngram-map speculative decoding, length
                                        of draft m-gram (default: 48)
--spec-ngram-min-hits N                 minimum hits for ngram-map speculative decoding (default: 1)
```

### `--spec-type TYPE`

Specifies a type of speculative decoding without draft model.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding (default) |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram-keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram-keys and up to four m-gram values (experimental) |
| `ngram-mod` | Use basic ngram hasher for speculative decoding with shared pool |

**Example:** Server-instance used to refactor source code.
```bash
./llama-server [...] --spec-type ngram-simple
```

### `--spec-ngram-size-n N`

Sets the size N of the lookup n-gram for n-gram map based speculative decoding.
The n-gram size N determines how many tokens in a row to look back when searching for matching patterns.

### `--spec-ngram-size-m M`

Sets the size M of the draft m-gram for n-gram map based speculative decoding.
The m-gram size determines how many tokens to draft when a match is found.
Larger values can provide more speedup but may reduce acceptance rate.

### `--spec-ngram-min-hits H`

This option defines how often a key has to appear in the token history to be used as a draft (default is 1).

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).

## llama-tq fork recommendations (2026-06)

### Model-class config matrix

Tested on 2x RTX 2060 12 GB + KTQ/VTQ KV cache. Boost vs `--spec-type none` baseline.

| Model | Quant | Baseline | Best spec config | Creative | Repeat |
|---|---|---|---|---|---|
| Qwen3.5-9B-MTP | IQ4_XS | 57.5 t/s | `--spec-type ngram-cache --draft-max 8 --draft-min 4` | 1.0-1.46x | up to 3.8x |
| Qwen3.6-27B-MTP-A3B | IQ2_XXS | 18.0 t/s | `--spec-type ngram-cache --draft-max 8 --draft-min 4` | 1.05-1.10x | 3.65x |
| Qwen3.6-35B-MTP-A3B | IQ2_XXS | 71.5 t/s | `--spec-type ngram-cache --draft-max 8 --draft-min 4` | 1.00-1.05x | 1.31x |
| Ministral-3-3B | Q4_K_M | ~110 t/s | `--spec-type ngram-cache --draft-max 8 --draft-min 4` | 1.0x | 1.5-2x |

All configs are **lossless** — output is byte-identical to baseline (verified via
diff on first 140 chars of greedy outputs across multiple prompts).

### DRAFT_MTP guidance

DRAFT_MTP is supported and respects `--draft-p-min` (default 0.75). On consumer
single-GPU + IQ2-class MoE models it consistently underperforms ngram-cache
because the single-layer MTP head caps per-step confidence so position 1-3 of a
dm=4 draft fail the p_min filter (observed: avg 1.30 tokens/draft, firing 32%
of TG-iterations with the Phase 29 decay patch — still net-negative).

When to enable DRAFT_MTP:
- Q4 or higher quantization (the MTP head retains useful sharpness)
- Bandwidth-bound hardware (Jetson Orin, M-series unified memory)
- Larger active-param models (>10B active) where 1 extra token amortises the draft setup

When NOT to enable DRAFT_MTP:
- IQ2/IQ3 quantization (use ngram-cache instead)
- Single GPU < 12 GB (compute-buffer fits but offers worse boost than ngram-cache)

### Tuning knobs (env-gated)

| Env var | Default | Range | Effect |
|---|---|---|---|
| `LLAMA_SPEC_RELAX` | 0 | 0..50 | Lower ngram-cache acceptance thresholds for noisier-quant models (Phase 28) |
| `LLAMA_MTP_DECAY` | 0.70 | 0.30..1.00 | DRAFT_MTP per-position p_min decay; 1.0 = strict at every step (Phase 29) |
| `FORK_MTP_PROFILE_ACC=1` | — | — | Print draft/accept stats per TG batch |
| `FORK_SPEC_TRACE=1` | — | — | Verbose spec-flow log (very noisy, debug only) |

### Static lookup-cache

The `--lookup-cache-static FILE` flag works in the server. Train a cache with:

```
llama-lookup-create -m TARGET.gguf -f corpus.txt -ngl 0 \
  --lookup-cache-static cache.bin
```

`LLAMA_NGRAM_STATIC` is set to 4 (was 2 upstream) for sharper keys. Static-cache
files trained with NGRAM_STATIC=2 must be regenerated. On IQ2 models the static
cache helps mostly with structured prompts; creative prompts see <5% boost from
static cache alone — the bottleneck is the model, not the lookup table.

### Universal-2x ceiling and Eagle3 path

On consumer 2x12 GB hardware with IQ2 MoE models, **universal 2x speculation
without quality regression is not achievable** with current draft-source options
(single-layer MTP-head + ngram-cache). Real universal 2x requires either:
1. Q4+ quantization (model doesn't fit in 24 GB at 27B+ context lengths)
2. **Eagle3-style draft-head** with 3-hidden-state fusion + training-time-test
3. Larger draft model (separate small GGUF; eats VRAM that's already maxed)

Option 2 is the realistic next step. Community Eagle3 checkpoints exist for
Qwen3-30B-A3B (`lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex`)
and Qwen3-VL-30B-A3B. Implementation is a 3-5 week engineering project on
top of the existing MTP infrastructure — see internal plan in
`docs/plans/2026-06-08-eagle3-integration.md` (local-only, not pushed).

Repeat-heavy workloads (lists, code boilerplate, log scanning) still hit
1.5-3.8x today via ngram-cache and do not need Eagle3.

### Critical fix (commit 78216a941)

Prior to 2026-06-08, ngram-* and draft-simple spec types silently corrupted
output on hybrid-recurrent models (qwen35) because `need_n_rs_seq()` only
returned a non-zero value for DRAFT_MTP. With `n_rs_seq = 0`, partial seq_rm
on hybrid models fails and the server falls back to `spec_ckpt.load_tgt`
with PARTIAL_ONLY which wipes mem_attn — destroying the target's prompt KV
context. Fix in `common/common.h::need_n_rs_seq()` extends the check to all
spec types that emit target-verifiable drafts.
