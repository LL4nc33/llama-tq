# Evaluation scripts

The scripts behind the measured tables in [docs/benchmarks/README.md](../../docs/benchmarks/README.md) and
[docs/turboquant.md](../../docs/turboquant.md). Each section below names a table and gives the commands that
reproduce it. Python scripts use only the standard library; shell scripts need bash, coreutils and, for the GPU idle
wait, `nvidia-smi`.

| Script | Measures |
|---|---|
| `retrieval.py` | Long-context copy task: builds a deterministic ~29k-token changelog prompt, sends it to `llama-server`, checks whether one section was copied exactly. Also records prefill and decode speed and draft acceptance per request. |
| `retrieval.sh` | Pass rate of the copy task per KV cache configuration over a prompt set (default 36 prompts), one server at a time. |
| `server-turn.py` | First prompt and follow-up turn on a running server: prompt tokens, prompt speed, wall time, decode speed, md5 of the output. |
| `spec-bench.py` | Speculative decoding (MTP, n-gram, draft model) with 1 or more concurrent requests: wall time, decode speed, accepted / generated draft tokens, md5 of the output. |
| `bench-abba.sh` | `llama-bench` of two builds in the order A, B, B, A per model and K/V pair, mean per build. |
| `bench-table.py` | Mean `llama-bench` speed per build, model, K/V pair and test from `-o jsonl` files (used by the two shell scripts that run `llama-bench`). |
| `release-check.sh` | `test-backend-ops` groups for the TurboQuant types on one GPU, then `llama-bench` and KL divergence. |

The options are in the header comment of each shell script and in `--help` of the Python scripts (`bench-table.py`
prints its usage when run without arguments).

## Setup

- Build: `cmake -B build -DGGML_CUDA=ON && cmake --build build -j` ([build.md](../../docs/build.md)); the binaries are
  in `build/bin`. `retrieval.sh` takes the server binary from `LLAMA_SERVER`, the other shell scripts take a
  directory (`BIN`, `BIN_A`, `BIN_B`) and add it to `LD_LIBRARY_PATH`.
- Models: the tables use Qwen3-4B-Instruct-2507 Q4_K_M (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`) unless they say
  otherwise; the file names below are the usual GGUF names of the models in the tables.
- Text: `scripts/get-wikitext-2.sh` downloads `wikitext-2-raw/wiki.test.raw`.
- GPUs: "one GPU" means `CUDA_VISIBLE_DEVICES=0` (`GPUS=0`), "two GPUs" `CUDA_VISIBLE_DEVICES=0,1` with the default
  layer split. Every run uses `-t 4` and offloads all layers.
- Order: on the GPUs of these tables the first run of a round is 2-3 % faster than the next one (warm-up and clocks).
  Builds are therefore compared in the order A, B, B, A, and pair lists run forwards and then backwards, so that every
  configuration is measured once early and once late.
- Run one measurement at a time on a GPU. `retrieval.sh` waits until the GPUs are idle before each server start.

## Long-context retrieval

[turboquant.md, Long-context retrieval](../../docs/turboquant.md#long-context-retrieval). The prompt is a synthetic
changelog of a fictional storage engine with about 100 sections; the request asks for one section verbatim between
`<<<BEGIN>>>` and `<<<END>>>`, followed by a 3-point summary. Each section depends only on the document seed and its
index, and `retrieval.py offline` estimates the size from characters, so the same seed, position and token target give
the same prompt (the same `prompt_sha256`) on every machine. `retrieval.sh` starts

```
llama-server -m MODEL -ngl 99 -fa on -c 40960 --jinja -np 1 --reasoning off --cache-ram 0 -t 4 -ctk K -ctv V FLAGS
```

once per configuration line and sends every prompt once (`/v1/chat/completions`, temperature 0, `max_tokens` 320,
`cache_prompt` false). A prompt counts when the copy is exact.

Qwen3-4B column, 36 prompts (6 documents x target sections at 10-85 % of the context, the defaults); one GPU for the
2026-10-14 rows, `GPUS=0,1` for the rows marked as two RTX 2060:

```bash
cat > qwen3-4b.cfg <<'EOF'
f16       f16    f16    - -
q8        q8_0   q8_0   - -
q8-v41    q8_0   vtq4_1 - -
q8-v21    q8_0   vtq2_1 - -
k41-v41   ktq4_1 vtq4_1 - -
k31-v31   ktq3_1 vtq3_1 - -
k21-v21   ktq2_1 vtq2_1 - -
k21-f16   ktq2_1 f16    - -
k21-q8    ktq2_1 q8_0   - -
k4-v4     ktq4   vtq4   - -
k3-v3     ktq3   vtq3   - -
k2-v2     ktq2   vtq2   - -
k2-q8     ktq2   q8_0   - -
q8-v3     q8_0   vtq3   - -
EOF
LLAMA_SERVER=build/bin/llama-server MODEL=models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf GPUS=0 \
    scripts/eval/retrieval.sh < qwen3-4b.cfg
```

The `ktq3_1` / `vtq3_1` 9/12 cell and the Qwen3.6-35B-A3B column use 12 prompts (3 documents, 15-90 %):

```bash
printf 'q8 q8_0 q8_0 - -\nq8-v41 q8_0 vtq4_1 - -\nk41-v41 ktq4_1 vtq4_1 - -\nk21-v21 ktq2_1 vtq2_1 - -\n' > qwen36.cfg
LLAMA_SERVER=build/bin/llama-server MODEL=models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf GPUS=0,1 \
    SEEDS="1234 2001 3001" FRACS="0.15 0.40 0.65 0.90" scripts/eval/retrieval.sh < qwen36.cfg
```

Gemma 4 26B-A4B (bullet below the table, 36 prompts, two GPUs): the same command with
`MODEL=models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf GPUS=0,1` and the lines for f16, `q8_0`, `ktq3_1` / `vtq3_1`,
`ktq3` / `vtq3`, `ktq4` / `vtq4` and `ktq3` / `q8_0`.

Output: one line per configuration, `LABEL K/V [ENV] [FLAGS]: exact N/M  mean ratio R  failed <hashes>`, also in
`OUT/summary.txt`; `OUT/LABEL.jsonl` has every request with timings, the generated text and the check, `OUT/LABEL.log`
the server log. `retrieval.py show --prompt OUT/prompts/p-1234-0.40.json` prints the section a prompt asks for.

## Trellis and codebook types, two GPUs

[benchmarks, Trellis and codebook types](../../docs/benchmarks/README.md#trellis-and-codebook-types-two-gpus).

Decode column (`-p 512 -n 64 -d 0,8192,32768 -r 2`, two GPUs). With one build, give it as both A and B: every pair
then runs four times and the A and B columns of the summary are two halves of the same build (their ratio shows the
run-to-run spread):

```bash
BIN_A=build/bin BIN_B=build/bin GPUS=0,1 MODEL=models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    KV="q8_0/q8_0 ktq4_1/vtq4_1 ktq4/vtq4 ktq3_1/vtq3_1 ktq3/vtq3" ARGS="-p 512 -n 64 -d 0,8192,32768 -r 2" \
    scripts/eval/bench-abba.sh
```

The Gemma 4 26B-A4B line below the table: the same with `MODEL=models/gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf`,
`KV="ktq3_1/vtq3_1 ktq3/vtq3"` and `ARGS="-p 0 -n 64 -d 0,8192,32768 -r 2"`.

First prompt and follow-up columns: a fresh server for each K/V pair and prompt length, the pairs in the order
`ktq4_1`, `ktq4`, `ktq3_1`, `ktq3`, `q8_0`, then backwards (V of the same family), default deferred K/V:

```bash
CUDA_VISIBLE_DEVICES=0,1 build/bin/llama-server -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -t 4 -ngl 99 -fa on \
    -c 40960 -np 1 --cache-ram 0 -ctk ktq4_1 -ctv vtq4_1 --host 127.0.0.1 --port 8899 &
python3 scripts/eval/retrieval.py health --url http://127.0.0.1:8899
python3 scripts/eval/server-turn.py --url http://127.0.0.1:8899 --text wikitext-2-raw/wiki.test.raw --chars 30000
kill %1
# the same with a newly started server and --chars 96000
```

30000 and 96000 characters of wikitext-2 (test) are the 6.8k and 23.7k-token prompts, the follow-up adds 2000
characters (about 460 tokens). "First prompt" is the wall time of the `A` line, "follow-up" the prompt speed of the `B`
line. Start a new server for each length: on one server the second prompt would reuse the cached text of the first
(that is what the first version of the table did). The `out` md5 shows whether the generated text is the same across
repetitions of a pair.

## Speculative decoding at long context, one GPU

[benchmarks, Speculative decoding at long context](../../docs/benchmarks/README.md#speculative-decoding-at-long-context-one-gpu).
The server runs with and without `--spec-type ngram-cache --draft-max 8 --draft-min 4`; `retrieval.py run` prints
prefill and decode speed, the accepted drafts and whether the copy is exact for each request.

Release-tree paragraph (two prompts of 27.6k / 27.9k tokens, 2 x 2 requests per cell, two GPUs as stated there),
per K/V pair (`q8_0` / `q8_0`, `q8_0` / `vtq4_1`, `ktq4_1` / `vtq4_1`, `ktq3` / `vtq3`) once without and once with
the speculation flags:

```bash
for p in "1234 0.40" "3001 0.70"; do set -- $p
    python3 scripts/eval/retrieval.py offline --out p-$1-$2.json --seed $1 --target-frac $2 --target-tokens 29000
done
SPEC="--spec-type ngram-cache --draft-max 8 --draft-min 4"   # empty for speculation off
CUDA_VISIBLE_DEVICES=0,1 build/bin/llama-server -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -t 4 -ngl 99 -fa on \
    -c 40960 -np 1 --jinja --reasoning off --cache-ram 0 -ctk q8_0 -ctv q8_0 $SPEC --host 127.0.0.1 --port 8899 &
python3 scripts/eval/retrieval.py health --url http://127.0.0.1:8899
for p in p-1234-0.40 p-3001-0.70 p-1234-0.40 p-3001-0.70; do
    python3 scripts/eval/retrieval.py run --url http://127.0.0.1:8899 --prompt $p.json --out spec.jsonl \
        --build release --pair q8_0/q8_0 --spec on --max-tokens 400 --timeout 600   # --spec off without SPEC
done
kill %1
```

Table of 2026-10-14 (one GPU, the 28,883-token prompt): the prompt was fitted to 29000 tokens on the server itself
(`retrieval.py calibrate --url http://127.0.0.1:8899 --out prompt.json --seed 1234 --target-tokens 29000`, target at
55 % of the document); per server start one warm-up request (`retrieval.py run ... --kind warmup --max-tokens 8`, not
counted) and two measured ones (`--max-tokens 400`), two server starts per build, K/V pair and speculation mode, the
two builds alternated A, B, B, A.

## Speculative decoding, two GPUs

[benchmarks, Speculative decoding, two GPUs](../../docs/benchmarks/README.md#speculative-decoding-two-gpus).
Qwen3.6-35B-A3B Q4_K_M on two GPUs, short prompts (rows "release tree, 2026-10-16" and the two-request sentence):

```bash
MTP="--spec-type draft-mtp --draft-max 3 --draft-min 0"
S="build/bin/llama-server -m models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf -t 4 -ngl 99 -fa on --jinja --reasoning off \
   --cache-ram 0 --host 127.0.0.1 --port 8899"
# without speculation, two slots: one request, then two concurrent ones
CUDA_VISIBLE_DEVICES=0,1 $S -ctk q8_0 -ctv q8_0 -c 16384 -np 2 &
python3 scripts/eval/retrieval.py health --url http://127.0.0.1:8899
python3 scripts/eval/spec-bench.py --url http://127.0.0.1:8899 --concurrent 1,2; kill %1
# MTP, one slot (the first batch warms up)
CUDA_VISIBLE_DEVICES=0,1 $S -ctk q8_0 -ctv q8_0 -c 8192 -np 1 $MTP &
python3 scripts/eval/retrieval.py health --url http://127.0.0.1:8899
python3 scripts/eval/spec-bench.py --url http://127.0.0.1:8899 --concurrent 1,1; kill %1
# MTP, two slots
CUDA_VISIBLE_DEVICES=0,1 $S -ctk q8_0 -ctv q8_0 -c 16384 -np 2 $MTP &
python3 scripts/eval/retrieval.py health --url http://127.0.0.1:8899
python3 scripts/eval/spec-bench.py --url http://127.0.0.1:8899 --concurrent 1,2; kill %1
```

The `ktq4_1` / `vtq4_1` row: the MTP one-slot command with `-ctk ktq4_1 -ctv vtq4_1`. Acceptance is accepted /
generated draft tokens of a request (`draft` in the output; the server log prints the same as "draft acceptance").

The rows of 2026-10-15 (29k-token prompt at 40k context, `--draft-max` 1 and 3) and the n-gram paragraph were taken
with an earlier request driver that is not part of this directory. The closest reproduction is the long-context
procedure above on this model: a 29k-token prompt from `retrieval.py calibrate`, a server with `-c 40960 -np 1` and
`$MTP` (or `--draft-max 1`) or the n-gram flags, and `retrieval.py run`, which prints prompt and decode speed and the
draft counts per request. For the `-b 8192` sentence compare the prefill speed with `-b 2048` and with the default.

## Against upstream, two GPUs

[benchmarks, Against upstream](../../docs/benchmarks/README.md#against-upstream-two-gpus). Upstream llama.cpp at
`23b0202` built the same way as the fork (`git clone https://github.com/ggml-org/llama.cpp upstream && git -C upstream
checkout 23b0202`, then `cmake -B build -DGGML_CUDA=ON` in it). The script's default arguments are the table's
(`-p 2048 -n 128 -b 2048 -ub 512 -d 0,32768 -r 1`, plus `-ngl 99 -fa 1 -t 4`):

```bash
M=models
BIN_A=upstream/build/bin BIN_B=build/bin GPUS=0,1 KV="f16/f16 q8_0/q8_0" \
    KV_B_ONLY="q8_0/vtq4_1 ktq4_1/vtq4_1 ktq3/vtq3" \
    MODEL="$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf $M/gpt-oss-20b-MXFP4.gguf $M/gemma-4-12b-it-Q4_K_M.gguf \
           $M/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf $M/Qwen3.8-27B-UD-Q4_K_M.gguf" \
    scripts/eval/bench-abba.sh
```

Per model each pair in `KV` runs upstream, fork, fork, upstream; `OUT/summary.md` has the mean of the two runs per
build (first table: `pp2048` and `pp2048@d32768` with f16, `tg128`, `tg128@d32768` with f16 and `q8_0`) and the
`KV_B_ONLY` pairs once on the fork (second table).

## Release check on newer GPUs

[benchmarks, Release check on newer GPUs](../../docs/benchmarks/README.md#release-check-on-newer-gpus). On each GPU:

```bash
BIN=build/bin MODEL=models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf WIKITEXT=wikitext-2-raw/wiki.test.raw \
    scripts/eval/release-check.sh
```

This runs the TurboQuant op tests (`SET_ROWS`, `GET_ROWS`, `CPY`, flash attention at head size 512, a head size 128
matrix and the GQA decode cases, each group with a time limit), then `llama-bench -fa 1 -t 4 -p 512 -n 128
-d 0,32768 -r 2` over the table's ten pairs forwards and backwards (`OUT/bench.md`: the mean of both rounds per pair,
`tg128` and `tg128@d32768` are the table cells), verify batches `-p 2,4,8,16 -n 0 -d 32768` (`pp16@d32768` is the
"verify batches of 16 queries" sentence) and the KL divergence of nine pairs against f16 (compare with
[turboquant.md, Accuracy](../../docs/turboquant.md#accuracy)). The RTX 5090 numbers are from one round. For the TITAN
RTX op test run `MODE=ops BIN=build/bin scripts/eval/release-check.sh`: after the TurboQuant groups it runs every
flash-attention case with a TurboQuant type, every flash-attention case and every op (several hours). `OUT/run.log`
lists per group the exit code (124 = time limit), passed tests and the counts of OK, FAIL and not supported cases,
followed by any line with FAIL, a CUDA error or an assertion; the exit code of the script is 1 if any group failed.

The release-tree rows (RTX 5090, RTX 4090, RTX A40) of the empty-context table in
[Decode speed, one GPU](../../docs/benchmarks/README.md#decode-speed-one-gpu) are the `tg128` cells of the same run.
