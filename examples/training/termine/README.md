# Appointment extraction task (evaluation of llama-finetune)

A small, fully synthetic and reproducible task used to check that LoRA training on quantized GGUF models learns a new
output format: extract an appointment from a German message into JSON with a fixed schema. No base model solves it
without training (0 % exact match: own keys, own date/time formats, code fences), so the score measures what the
adapter learned. Results per model are in [docs/finetune.md](../../../docs/finetune.md).

## Data

`gen_termine.py` writes the data (deterministic, `random.seed(42)`):

| File | Content | md5 |
|---|---|---|
| `termine-train.jsonl` | 700 chats (`messages`: system, user, assistant) | f1fc5656d900001f1cd2c68e5c0b70ed |
| `termine-test.jsonl` | 100 held-out prompts with the expected object (`system`, `user`, `expected`) | 0f5ce9372fbb0f720262a6633ffcaef3 |

800 unique messages are drawn; the first 700 are training data, the last 100 the test set. Every example:

- **System prompt:** `Extrahiere den Termin aus der Nachricht als JSON.`
- **User message:** one of six templates, e.g.
  - `Hallo! Ich möchte die {thema} mit {name} am {datum} um {zeit} {raum} in {stadt} fixieren.`
  - `Bitte trag ein: {thema}, {datum}, {zeit}, mit {name}, Ort: {stadt} ({raum}).`
  - `Servus, {name} hat sich gemeldet – die {thema} findet am {datum} um {zeit} in {stadt} {raum} statt.`
  - `Erinnerung: Am {datum} ist um {zeit} die {thema} mit {name}. Treffpunkt {raum} in {stadt}.`
  - `Kannst du für mich die {thema} mit {name} notieren? {stadt}, {raum}, {datum} ab {zeit}.`
  - `Termin bestätigt: {name}, {thema}, {zeit} am {datum}, {raum} in {stadt}. Danke!`
- **Answer:** one line of JSON with exactly these keys in this order:
  `{"person": "Eva Wagner", "datum": "2026-12-09", "uhrzeit": "10:00", "ort": "Innsbruck, Café Central", "thema": "Behördentermin"}`

Value ranges and conventions (the model has to learn them from the data):

- names: 30 first × 20 last names (Austrian); cities: 18 Austrian cities; 10 places, 15 topics;
- date written as `5. Juni 2026`, `5.6.2026` or `05.06.2026` (day first, Austrian month names such as `Jänner`) →
  `YYYY-MM-DD`; years 2026–2027;
- time written as `8:20 Uhr`, `8.20 Uhr` or, for full hours, `8 Uhr` → `HH:MM`; hours 7–19, so `7 Uhr` is 07:00;
- `ort` is `"{stadt}, {place without preposition}"` (`Innsbruck, Café Central`), only the city for `online per Videocall`.

## Training input (what the model sees)

`llama-finetune` renders each chat with the model's own chat template (`--jinja`), with the same `--reasoning` setting
as the server, and trains only the assistant answer including its end-of-turn token. The answer is trained after the
prompt exactly as the server renders it for generation. Two examples (`[[ ]]` = trained tokens,
`LLAMA_FINETUNE_SHOW_MASK=1` prints this):

ChatML (Qwen3-Instruct-2507):
```
<|im_start|>system
Extrahiere den Termin aus der Nachricht als JSON.<|im_end|>
<|im_start|>user
Hallo! Ich möchte die Behördentermin mit Eva Wagner am 09.12.2026 um 10 Uhr im Café Central in Innsbruck fixieren.<|im_end|>
<|im_start|>assistant
[[{"person": "Eva Wagner", "datum": "2026-12-09", "uhrzeit": "10:00", "ort": "Innsbruck, Café Central", "thema": "Behördentermin"}<|im_end|>]]
```

Gemma 4 with `--reasoning off` (the generation prompt contains an empty thought channel):
```
<bos><|turn>system
Extrahiere den Termin aus der Nachricht als JSON.<turn|>
<|turn>user
Hallo! Ich möchte die Behördentermin mit Eva Wagner am 09.12.2026 um 10 Uhr im Café Central in Innsbruck fixieren.<turn|>
<|turn>model
<|channel>thought
<channel|>[[{"person": "Eva Wagner", "datum": "2026-12-09", "uhrzeit": "10:00", "ort": "Innsbruck, Café Central", "thema": "Behördentermin"}<turn|>]]
```

The chats are packed whole into windows of the context size (`-c`); an example that does not fit into the rest of a
window starts the next one, the rest is padding without a label.

## Commands

```bash
python3 gen_termine.py

# training (attention LoRA; values per model in docs/finetune.md)
llama-finetune -m MODEL.gguf -f termine-train.jsonl -o termine.gguf \
  --lora-train-target '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$' --lora-train-rank 16 --lora-train-alpha 32 \
  -opt adamw -lr 2e-4 --lr-warmup 20 --epochs 2 -val-split 0.1 -ngl 99 -c 512 -b 512 -ub 512 --reasoning off

# evaluation
llama-server -m MODEL.gguf -ngl 99 -c 4096 --jinja --reasoning off --lora termine.lora.gguf --port 8080 &
python3 eval_termine.py 8080 --misses
```

`eval_termine.py` sends each test prompt through `/v1/chat/completions` (greedy, at most 160 tokens), takes the first
`{...}` of the answer and reports valid JSON, exact match of the whole object and accuracy per field. Without an
adapter every tested model scores 0 % exact.

`peft_termine.py` is the PyTorch reference (Hugging Face Transformers + PEFT + bitsandbytes nf4) with the same data,
LoRA setup (q/k/v/o, rank 16, alpha 32, no dropout), optimizer (AdamW 2e-4, 20 warmup steps, 2 epochs), loss mask
(assistant turn only) and metric: `peft_termine.py HF_MODEL_DIR termine-train.jsonl termine-test.jsonl OUT_DIR`.
