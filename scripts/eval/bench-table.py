#!/usr/bin/env python3
"""Mean llama-bench speed per build, model, K/V pair and test from llama-bench JSONL files (-o jsonl).

usage: bench-table.py DIR_A/*.jsonl [DIR_B/*.jsonl ...]

The build label of a file is the name of its directory, so bench-abba.sh writes A/ and B/ and the table gets one
column per build (runs of the same build are averaged, the run count is in brackets) and, with two builds, the
ratio B / A. Tests: ppN = prompt of N tokens, tgN = N generated tokens, @dD = after D tokens of context.
Used by bench-abba.sh and release-check.sh. Stdlib only.
"""
import collections
import json
import os
import sys


def test_name(d):
    if d["n_prompt"] and d["n_gen"]:
        t = "pp%d+tg%d" % (d["n_prompt"], d["n_gen"])
    elif d["n_prompt"]:
        t = "pp%d" % d["n_prompt"]
    else:
        t = "tg%d" % d["n_gen"]
    return t + ("@d%d" % d["n_depth"] if d.get("n_depth") else "")


def main(paths):
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    runs = collections.defaultdict(list)  # (model, pair, test) -> build -> [t/s]
    builds, rows = [], []
    for p in paths:
        build = os.path.basename(os.path.dirname(os.path.abspath(p)))
        if build not in builds:
            builds.append(build)
        for line in open(p, errors="replace"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            key = (os.path.basename(d.get("model_filename", "?")), "%s/%s" % (d["type_k"], d["type_v"]), test_name(d))
            if key not in runs:
                rows.append(key)
                runs[key] = collections.defaultdict(list)
            runs[key][build].append(d["avg_ts"])

    head = ["model", "K/V", "test"] + builds + (["%s / %s" % (builds[1], builds[0])] if len(builds) == 2 else [])
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * 3 + "---:|" * (len(head) - 3))
    for key in rows:
        means = [sum(v) / len(v) if v else None for v in (runs[key].get(b, []) for b in builds)]
        cells = ["%.2f (%d)" % (m, len(runs[key][b])) if m is not None else "" for m, b in zip(means, builds)]
        if len(builds) == 2:
            cells.append("%.3f" % (means[1] / means[0]) if means[0] and means[1] else "")
        print("| " + " | ".join(list(key) + cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
