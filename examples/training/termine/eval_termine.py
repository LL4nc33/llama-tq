#!/usr/bin/env python3
# Scores a running llama-server on termine-test.jsonl: valid JSON, exact match of the whole object, per-field accuracy.
# Greedy decoding, max 160 tokens, the server's chat template (start the server with --jinja and the same --reasoning
# setting as in training). --misses prints every example that is not an exact match, with the raw model output.
# Usage: eval_termine.py [PORT] [--misses]
import json, re, sys, urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
port = args[0] if args else "8080"
show_misses = "--misses" in sys.argv
fields = ["person", "datum", "uhrzeit", "ort", "thema"]
n = valid = exact = 0
field_ok = {k: 0 for k in fields}
for line in open("termine-test.jsonl", encoding="utf-8"):
    ex = json.loads(line)
    body = {"messages": [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}],
            "temperature": 0, "max_tokens": 160}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    n += 1
    m = re.search(r"\{.*\}", out, re.S)
    try:
        got = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        got = None
    ok = isinstance(got, dict) and got == ex["expected"]
    if show_misses and not ok:
        print("MISS  in: ", ex["user"])
        print("      out:", out.strip().replace("\n", " ")[:300])
        print("      exp:", json.dumps(ex["expected"], ensure_ascii=False))
    if not isinstance(got, dict):
        continue
    valid += 1
    exact += ok
    for k in fields:
        field_ok[k] += str(got.get(k, "")).strip() == ex["expected"][k]
print(f"n={n} valid_json={100*valid/n:.0f}% exact={100*exact/n:.0f}% " +
      " ".join(f"{k}={100*v/n:.0f}%" for k, v in field_ok.items()))
