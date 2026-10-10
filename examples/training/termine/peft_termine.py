#!/usr/bin/env python3
# QLoRA reference run for the llama-finetune comparison: same data, same LoRA setup, same metric.
# Usage: peft_termine.py MODEL_DIR TRAIN_JSONL TEST_JSONL OUT_DIR
import json, math, random, re, sys, time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

model_dir, train_path, test_path, out_dir = sys.argv[1:5]
RANK, ALPHA, LR, WARMUP, EPOCHS, BATCH, VAL_SPLIT = 16, 32, 2e-4, 20, 2, 4, 0.1
torch.manual_seed(42); random.seed(42)

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map={"": 0}, torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

def encode(msgs):
    # loss only on the assistant turn: prompt tokens are masked with -100
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    return f_ids, [-100] * len(p_ids) + f_ids[len(p_ids):]

data = [encode(json.loads(l)["messages"]) for l in open(train_path)]
n_val = int(len(data) * VAL_SPLIT)
train, val = data[:len(data) - n_val], data[len(data) - n_val:]

def batches(rows, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        chunk = [rows[j] for j in idx[i:i + BATCH]]
        n = max(len(x) for x, _ in chunk)
        ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in chunk])
        lab = torch.tensor([y + [-100] * (n - len(y)) for _, y in chunk])
        att = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in chunk])
        yield ids.cuda(), lab.cuda(), att.cuda()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
steps = EPOCHS * math.ceil(len(train) / BATCH)
sched = get_linear_schedule_with_warmup(opt, WARMUP, steps)
scaler = torch.amp.GradScaler()

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model.train()
for ep in range(EPOCHS):
    tot = n = 0
    for ids, lab, att in batches(train, True):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        tot += loss.item(); n += 1
    model.eval()
    with torch.no_grad():
        vl = [model(input_ids=i, attention_mask=a, labels=l).loss.item() for i, l, a in batches(val, False)]
    model.train()
    print(f"epoch {ep + 1}: train loss {tot / n:.6f}  val loss {sum(vl) / len(vl):.6f}  t={time.time() - t0:.0f}s", flush=True)
train_s = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
model.save_pretrained(out_dir)

# eval: greedy, same metric as eval_termine.py
model.eval()
fields = ["person", "datum", "uhrzeit", "ort", "thema"]
n = valid = exact = 0
field_ok = {k: 0 for k in fields}
t1 = time.time()
for line in open(test_path):
    ex = json.loads(line)
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=160, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    n += 1
    m = re.search(r"\{.*\}", text, re.S)
    try:
        got = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        got = None
    if n <= 2:
        print("sample:", text.strip().replace("\n", " ")[:200])
    if not isinstance(got, dict):
        continue
    valid += 1
    exact += got == ex["expected"]
    for k in fields:
        field_ok[k] += str(got.get(k, "")).strip() == ex["expected"][k]
print(f"n={n} valid_json={100*valid/n:.0f}% exact={100*exact/n:.0f}% " +
      " ".join(f"{k}={100*v/n:.0f}%" for k, v in field_ok.items()))
print(f"PEFT train {train_s:.0f}s, peak allocated {peak:.2f} GiB, eval {time.time() - t1:.0f}s")
