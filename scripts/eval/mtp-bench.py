#!/usr/bin/env python3
"""Decode speed and draft acceptance of one running llama-server on three kinds of text (MTP, n-gram, draft model).

Sends three fixed prompts one after another to /v1/chat/completions: German prose, new code and the rewrite of a short
function, each at temperature 0 and at temperature 1 (seed 7), max_tokens --max-tokens, cache_prompt=false. A code
request first warms the server up. Per prompt it prints the mean decode speed over the two temperatures and --reps
repetitions and the share of accepted draft tokens:

  prose 51.3 t/s (draft 67 %) | code 81.6 t/s (draft 74 %) | rewrite 71.1 t/s (draft 70 %)

--jsonl FILE appends one JSON line per request (with --tag) for later tables. Start the server once per configuration
and alternate the configurations (A, B, B, A), e.g. Gemma 4 12B with its MTP assistant:

  LLAMA_MTP_DECAY=1.0 llama-server -m gemma-4-12b-it-Q4_K_M.gguf -ngl 99 -fa on -c 32768 -np 1 -ctk q8_0 -ctv q8_0 \\
      --jinja --reasoning off -md mtp-gemma-4-12B-it-Q8_0.gguf -ngld 99 --spec-type draft-mtp --draft-max 7 --port 8899

Stdlib only.
"""
import argparse
import json
import urllib.request

PROMPTS = {
    "prose": "Erkläre einem interessierten Laien in ganzen Sätzen, wie eine Wärmepumpe funktioniert, was sie im Winter in "
             "Österreich leistet und worauf man beim Kauf achten sollte. Etwa 350 Wörter, keine Aufzählungen.",
    "code": "Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), a method "
            "stats() returning hit/miss counts, full docstrings, type hints and a short pytest test suite. Output only code.",
    "rewrite": "Rewrite this function to be iterative, add type hints and a docstring, keep the behaviour identical, and "
               "explain the change in two sentences afterwards:\n\ndef flatten(x):\n    if not isinstance(x, list):\n"
               "        return [x]\n    out = []\n    for e in x:\n        out += flatten(e)\n    return out\n",
}


def ask(url: str, prompt: str, temp: float, max_tokens: int) -> dict:
    body = {"messages": [{"role": "user", "content": PROMPTS[prompt]}], "max_tokens": max_tokens, "temperature": temp,
            "seed": 7, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = json.loads(urllib.request.urlopen(req, timeout=900).read()).get("timings", {})
    return {"tps": t.get("predicted_per_second", 0.0), "n": t.get("predicted_n"), "draft_n": t.get("draft_n") or 0,
            "draft_acc": t.get("draft_n_accepted") or 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--max-tokens", type=int, default=450)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl")
    a = ap.parse_args()
    ask(a.url, "code", 0.0, 64)  # warm-up
    out = open(a.jsonl, "a") if a.jsonl else None
    cells = []
    for prompt in PROMPTS:
        runs = [ask(a.url, prompt, temp, a.max_tokens) for _ in range(a.reps) for temp in (0.0, 1.0)]
        if out:
            for r in runs:
                out.write(json.dumps({"tag": a.tag, "prompt": prompt, **r}) + "\n")
        tps = sum(r["tps"] for r in runs) / len(runs)
        drafted = sum(r["draft_n"] for r in runs)
        acc = f" (draft {100 * sum(r['draft_acc'] for r in runs) / drafted:.0f} %)" if drafted else ""
        cells.append(f"{prompt} {tps:.1f} t/s{acc}")
    print(" | ".join(cells))


if __name__ == "__main__":
    main()
