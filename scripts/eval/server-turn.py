#!/usr/bin/env python3
"""First prompt and follow-up turn timing against a running llama-server.

For every N in --chars: send the first N characters of --text (tag A, the first prompt), then the first
N + --extra characters (tag B, the follow-up: the same prompt plus about 480 tokens of wikitext, computed on the
cached prefix of A). Requests go to /completion with cache_prompt=true, greedy, n_predict --n-predict.
One line per request:

  <tag><chars> prompt_n <tokens computed>  pp <prompt t/s>  wall <s>  tg <decode t/s>  out <md5 of the text>

prompt_n counts only the tokens the server computed, so for B it is the length of the new part. The wall time
includes everything the server does before it answers (for TurboQuant types with deferred K/V also the conversion of
the f16-staged prompt). The requests run in the order given and share one server slot: when a longer A follows a
shorter pair, its prompt starts with the cached text of the previous pair; prompt_n shows how much was computed.
Start the server with one slot and no host prompt cache, e.g.

  llama-server -m MODEL -ngl 99 -fa on -c 40960 -np 1 --cache-ram 0 -ctk K -ctv V --port 8899

Stdlib only.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request


def request(url, prompt, n_predict, timeout):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8899", help="llama-server base URL (default %(default)s)")
    ap.add_argument("--text", required=True, help="plain text file, e.g. wikitext-2-raw/wiki.test.raw")
    ap.add_argument("--chars", default="30000,90000",
                    help="comma-separated prompt sizes in characters (default %(default)s: about 7k and 22k tokens "
                         "of wikitext-2 with the Qwen3 tokenizer)")
    ap.add_argument("--extra", type=int, default=2000, help="characters added for the follow-up turn (default 2000)")
    ap.add_argument("--n-predict", type=int, default=32, help="tokens generated per request (default 32)")
    ap.add_argument("--timeout", type=int, default=1800, help="request timeout in seconds (default 1800)")
    a = ap.parse_args()

    with open(a.text, encoding="utf-8", errors="replace") as f:
        text = f.read()
    for n in (int(x) for x in a.chars.split(",") if x.strip()):
        for tag, size in (("A", n), ("B", n + a.extra)):
            if size > len(text):
                print("%s%d: the text has only %d characters" % (tag, size, len(text)), file=sys.stderr)
                return 1
            resp, wall = request(a.url, text[:size], a.n_predict, a.timeout)
            tm = resp["timings"]
            print("%s%d prompt_n %d  pp %.1f t/s  wall %.2f s  tg %.1f t/s  out %s" % (
                tag, size, tm["prompt_n"], tm["prompt_per_second"], wall, tm["predicted_per_second"],
                hashlib.md5(resp["content"].encode()).hexdigest()[:8]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
