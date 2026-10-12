#!/usr/bin/env python3
"""Checks the fork's documentation before it is published.

- relative links and #anchors resolve
- no internal details: private IP addresses, internal host names, home directories, the word "production"
- every measured speed (t/s) appears in only one page; the changelog is history and may repeat numbers

usage: python3 scripts/check-docs.py   (from the repository root; exit code 1 on any problem)
"""

import glob
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

# pages written for this fork; upstream pages are left as they are
DOCS = ["README.md", "ROADMAP.md", "CHANGELOG.md", "SECURITY.md",
        "docs/turboquant.md", "docs/finetune.md", "docs/models.md", "docs/tensor-split.md",
        "docs/speculative.md", "docs/claude-code.md", "docs/vulkan.md", "docs/docker.md",
        "docs/calibrating-ktq.md", "docs/benchmarks/README.md"] + sorted(glob.glob("docs/history/*.md")) \
       + sorted(glob.glob("docs/benchmarks/history/*.md"))

# pages that may repeat numbers from other pages
HISTORY = {"CHANGELOG.md"} | {p for p in DOCS if "/history/" in p}

INTERNAL = [
    (re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"), "private IP address"),
    (re.compile(r"\bgpu0\d\b|\.node\b|\.lxc\b", re.I), "internal host name"),
    (re.compile(r"/home/\w+"), "home directory"),
    (re.compile(r"\bproduction\b|\bprod\b", re.I), "the word production"),
]

SPEED = re.compile(r"\b(\d{2,4}(?:\.\d)?) t/s\b")

# the same value on several pages that belongs to different measurements (value -> pages, reason)
SPEED_ALLOW = {
    "15.5": ({"docs/models.md", "docs/speculative.md"}, "Qwen3.8-Flash-Next at 118k vs the Qwen3.8-27B baseline"),
    "28": ({"docs/models.md", "docs/speculative.md"}, "Bonsai at 23k vs DFlash on Qwen3.8-27B"),
    "47": ({"docs/models.md", "docs/benchmarks/README.md"}, "K2-Horizon vs Qwen3-4B"),
    "57": ({"docs/models.md", "docs/benchmarks/README.md"}, "gpt-oss at 22k vs Qwen3-4B at 16k"),
}


def anchors(path):
    out, seen = set(), defaultdict(int)
    in_code = False
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r"#+\s+(.*)", line)
        if m:
            a = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-")
            out.add(a if seen[a] == 0 else f"{a}-{seen[a]}")
            seen[a] += 1
    return out


def main():
    problems = []
    speeds = defaultdict(set)
    for doc in DOCS:
        p = Path(doc)
        if not p.exists():
            problems.append(f"{doc}: listed in check-docs.py but missing")
            continue
        text = p.read_text(errors="ignore")
        for n, line in enumerate(text.splitlines(), 1):
            for rx, what in INTERNAL:
                if rx.search(line):
                    problems.append(f"{doc}:{n}: {what}: {line.strip()[:100]}")
            if doc not in HISTORY:
                for v in SPEED.findall(line):
                    speeds[v].add(f"{doc}:{n}")
        for m in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
            url = unquote(m.group(1))
            if url.startswith(("http://", "https://", "mailto:")):
                continue
            target, _, anchor = url.partition("#")
            t = (p.parent / target).resolve() if target else p.resolve()
            if not t.exists():
                problems.append(f"{doc}: broken link {url}")
            elif anchor and t.suffix == ".md" and anchor not in anchors(t):
                problems.append(f"{doc}: missing anchor {url}")
    for v, where in sorted(speeds.items()):
        pages = {w.split(":")[0] for w in where}
        if len(pages) > 1 and pages != SPEED_ALLOW.get(v, (None,))[0]:
            problems.append(f"{v} t/s appears in several pages (keep it in one, link from the others): "
                            + ", ".join(sorted(where)))
    for line in problems:
        print(line)
    print(f"{len(problems)} problem(s) in {len(DOCS)} pages")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
