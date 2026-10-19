#!/usr/bin/env python3
"""Speculative decoding request driver for a running llama-server (MTP, n-gram, draft model).

Sends n of four fixed ~400-word technical prompts at the same time to /v1/chat/completions (greedy, max_tokens
--max-tokens, cache_prompt=false) and prints one line per batch:

  c<n>: p0 wall <s> tg <decode t/s> draft <accepted>/<generated> md5 <text> | p1 ... | total <s>

Run it once per entry of --concurrent (default "1,1": the first batch also warms the server up). With two slots
(-np 2) "--concurrent 1,2" measures one request and then two concurrent ones; the combined decode speed of two
requests is the sum of their tg values. The md5 shows whether the text with speculation equals the text without.
Server, e.g. MTP with a draft of up to 3 tokens:

  llama-server -m MODEL -ngl 99 -fa on --jinja --reasoning off --cache-ram 0 -ctk q8_0 -ctv q8_0 -c 8192 -np 1 \\
      --spec-type draft-mtp --draft-max 3 --draft-min 0 --port 8899

The server log also reports the draft acceptance ("draft acceptance"). Stdlib only.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Write a detailed technical explanation of how a hash map handles collisions, with a short C example. About 400 words.",
    "Explain how a B-tree keeps itself balanced on insert and delete, with a short pseudocode example. About 400 words.",
    "Describe how TCP congestion control works (slow start, congestion avoidance, fast recovery). About 400 words.",
    "Explain how a garbage collector with generational marking works, with a short example. About 400 words.",
]


def one(url, i, max_tokens, timeout, res):
    body = json.dumps({"messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except Exception as e:  # report and keep the other requests
        res[i] = "p%d FAILED %s" % (i, e)
        return
    tm = resp.get("timings", {})
    txt = resp["choices"][0]["message"]["content"] or ""
    res[i] = "p%d wall %.2f s tg %.1f t/s draft %s/%s md5 %s" % (
        i, time.time() - t0, tm.get("predicted_per_second", 0), tm.get("draft_n_accepted"), tm.get("draft_n"),
        hashlib.md5(txt.encode()).hexdigest()[:8])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8899", help="llama-server base URL (default %(default)s)")
    ap.add_argument("--concurrent", default="1,1",
                    help="comma-separated numbers of concurrent requests, one batch each (default %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=400, help="max_tokens per request (default 400)")
    ap.add_argument("--timeout", type=int, default=900, help="request timeout in seconds (default 900)")
    a = ap.parse_args()

    for n in (int(x) for x in a.concurrent.split(",") if x.strip()):
        res: list[str] = [""] * n
        t0 = time.time()
        threads = [threading.Thread(target=one, args=(a.url, i, a.max_tokens, a.timeout, res)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print("c%d: %s | total %.2f s" % (n, " | ".join(res), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
