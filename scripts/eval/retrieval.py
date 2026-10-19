#!/usr/bin/env python3
"""Long-context retrieval test for llama-server: copy one section of a ~29k-token document verbatim.

Stdlib only (urllib, json, difflib). Subcommands:

  calibrate  build the deterministic document and fit it to a token budget using a
             running server (/apply-template + /tokenize), write prompt.json
  offline    same, but estimate tokens from characters (no server needed); retrieval.sh
             uses this, so the prompts depend only on seed, fraction and token target
  health     wait until /health answers 200 (aborts early if the server PID died)
  run        send one chat request (cache_prompt=false), append one JSON line with
             timings, wall time, generated text and the verbatim check
  show       print the target section and the instruction of a prompt.json

The document is a synthetic but natural-looking changelog of a fictional storage
engine ("tessera"). Every section is generated from its own seeded RNG, so a section
only depends on (seed, index), never on how many sections there are. A request passes
when the copied section matches the source exactly (after trimming trailing spaces and
code fences). The many similar but not identical lines also make it a realistic prompt
for n-gram speculative decoding.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

BEGIN_MARK = "<<<BEGIN>>>"
END_MARK = "<<<END>>>"

# --------------------------------------------------------------------------------------
# Deterministic document generator
# --------------------------------------------------------------------------------------

MODULES = [
    "wal/segment_writer.rs", "wal/recovery.rs", "compaction/leveled.rs", "compaction/tiered.rs",
    "index/btree_page.rs", "index/bloom.rs", "net/rpc_server.rs", "net/tls_acceptor.rs",
    "query/planner.rs", "query/vectorized_scan.rs", "storage/block_cache.rs", "storage/mmap_file.rs",
    "replication/raft_log.rs", "replication/snapshot.rs", "admin/cli.rs", "admin/http_api.rs",
    "metrics/exporter.rs", "txn/mvcc.rs", "txn/lock_table.rs", "codec/zstd_frame.rs",
]
COMPONENTS = [
    "write-ahead log", "leveled compaction", "tiered compaction", "B-tree index", "bloom filters",
    "RPC server", "TLS acceptor", "query planner", "vectorized scan", "block cache",
    "mmap file layer", "Raft log", "snapshot transfer", "admin CLI", "HTTP admin API",
    "Prometheus exporter", "MVCC layer", "lock table", "zstd codec",
]
CONDITIONS = [
    "two flush threads rotated the same segment",
    "a follower rejoined the cluster during a snapshot install",
    "the block cache was resized while readers held pinned pages",
    "a client disconnected in the middle of a streaming scan",
    "the disk reported ENOSPC during a checkpoint",
    "a schema change raced with an open read transaction",
    "TLS session tickets expired under sustained load",
    "the compaction scheduler was paused and resumed within one tick",
    "the system clock jumped backwards by more than one second",
    "an empty key range was passed to a delete-range call",
    "more than 65535 column families were opened",
    "the leader stepped down while a batch commit was in flight",
]
EFFECTS = [
    "which could leave a torn record at the end of the log",
    "which caused the process to stall until the next heartbeat",
    "which produced duplicate rows in rare cases",
    "which leaked one file descriptor per occurrence",
    "which made recovery replay the same segment twice",
    "which reported a misleading error to the client",
    "which inflated the p99 read latency by up to 40 ms",
    "which could panic with an index-out-of-bounds message",
]
CONFIG_KEYS = [
    "storage.block_cache_mb", "wal.sync_interval_ms", "compaction.max_parallel_jobs",
    "net.max_inflight_requests", "replication.snapshot_chunk_kb", "query.max_scan_rows",
    "txn.lock_timeout_ms", "codec.zstd_level", "metrics.scrape_interval_s", "admin.http_bind",
]
METRICS = [
    "tessera_wal_fsync_seconds", "tessera_compaction_bytes_total", "tessera_cache_hit_ratio",
    "tessera_rpc_inflight", "tessera_raft_apply_lag", "tessera_txn_conflicts_total",
    "tessera_scan_rows_total", "tessera_snapshot_bytes_total",
]
HANDLES = [
    "mkovacs", "j.huber", "a-lindqvist", "pnovak", "s.brenner", "tkoller", "r-amari",
    "e.fischer", "d-oyelaran", "lgruber", "c.moreau", "hwinkler",
]
PLATFORMS = ["Linux x86_64", "Linux aarch64", "FreeBSD 14", "macOS arm64"]
KINDS = ["maintenance release", "feature release", "security release", "hotfix release"]
TEAMS = ["storage-core", "replication", "query-engine", "platform", "observability"]
PRIOS = ["low", "medium", "high"]

BULLET_TEMPLATES = [
    "Fixed a race condition in `{mod}` when {cond} (#{issue}).",
    "Fixed a bug in the {comp} where {cond}, {effect} (#{issue}).",
    "Improved {comp} throughput by {pct}% on {plat} by batching small writes (#{issue}, thanks @{who}).",
    "Added the configuration key `{key}` (default: {val}) to tune the {comp} (#{issue}).",
    "Changed the default of `{key}` from {val} to {val2}; existing configuration files are not affected (#{issue}).",
    "Added the metric `{metric}` to the Prometheus exporter (#{issue}).",
    "Deprecated `{key}`; it will be removed in {nextmajor}.0 (#{issue}).",
    "Reduced memory usage of the {comp} by about {mb} MiB per 1000 open tables (#{issue}).",
    "The {comp} now retries transient I/O errors up to {n} times before failing the request (#{issue}).",
    "Security: fixed a denial-of-service issue in `{mod}` when {cond} (CVE-{year}-{cve}).",
    "Documentation: clarified how `{key}` interacts with the {comp} (#{issue}, thanks @{who}).",
    "Removed an unnecessary lock from `{mod}`, {effect2} (#{issue}).",
]
EFFECTS2 = [
    "cutting tail latency under contention",
    "which shortens recovery after a crash",
    "so that background jobs no longer block foreground reads",
    "which halves the number of context switches per commit",
]
NOTES = [
    "Rolling upgrades from {prev}.x are supported.",
    "Rolling upgrades from {prev}.x are supported; downgrades require `tessera-admin compact --format=v{major}`.",
    "This release changes the on-disk format of the {comp}; take a backup before upgrading.",
    "No configuration changes are required.",
    "Operators running {plat} should upgrade within 7 days.",
]

BASE_DATE = _dt.date(2019, 1, 7)


def section_rng(seed: int, idx: int) -> random.Random:
    # derive a stable per-section seed (never use hash(), it is randomized per process)
    h = hashlib.sha256(("tessera:%d:%d" % (seed, idx)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def version_of(idx: int) -> tuple:
    return (1 + idx // 40, (idx // 8) % 5, idx % 8)


def make_section(seed: int, idx: int) -> str:
    """Section idx (1-based). Depends only on (seed, idx)."""
    rng = section_rng(seed, idx)
    n_bullets = rng.randint(5, 9)
    major, minor, patch = version_of(idx)
    date = BASE_DATE + _dt.timedelta(days=9 * idx + rng.randint(0, 4))
    kind = rng.choice(KINDS)
    head = "### [S%03d] tessera %d.%d.%d (%s) - %s" % (idx, major, minor, patch, date.isoformat(), kind)
    owner = "Owner: %s team. Upgrade priority: %s." % (rng.choice(TEAMS), rng.choice(PRIOS))
    lines = [head, owner, ""]
    used = set()
    for _ in range(n_bullets):
        # avoid identical template twice in a row inside one section
        t = rng.randrange(len(BULLET_TEMPLATES))
        while t in used and len(used) < len(BULLET_TEMPLATES):
            t = rng.randrange(len(BULLET_TEMPLATES))
        used.add(t)
        val = rng.choice([16, 32, 64, 128, 256, 512, 1024, 4096])
        fill = {
            "mod": rng.choice(MODULES), "cond": rng.choice(CONDITIONS), "effect": rng.choice(EFFECTS),
            "effect2": rng.choice(EFFECTS2), "comp": rng.choice(COMPONENTS), "key": rng.choice(CONFIG_KEYS),
            "metric": rng.choice(METRICS), "who": rng.choice(HANDLES), "plat": rng.choice(PLATFORMS),
            "issue": 1000 + idx * 37 + rng.randint(0, 36), "pct": rng.randint(4, 38),
            "val": val, "val2": val * 2, "mb": rng.randint(2, 90), "n": rng.randint(2, 6),
            "nextmajor": major + 1, "year": date.year, "cve": rng.randint(10000, 49999),
        }
        lines.append("- " + BULLET_TEMPLATES[t].format(**fill))
    lines.append("")
    note = rng.choice(NOTES).format(prev="%d.%d" % (major, max(minor - 1, 0)), major=major,
                                    comp=rng.choice(COMPONENTS), plat=rng.choice(PLATFORMS))
    lines.append("Notes: " + note)
    return "\n".join(lines)


def build_document(seed: int, n_sections: int) -> str:
    intro = (
        "# tessera - complete changelog\n\n"
        "tessera is an embedded, replicated key-value storage engine. This file lists every "
        "release in chronological order. Each section starts with a heading of the form "
        "`### [Snnn] tessera X.Y.Z (date) - kind`, followed by the owning team, a list of "
        "changes and upgrade notes.\n"
    )
    return intro + "\n" + "\n\n".join(make_section(seed, i) for i in range(1, n_sections + 1)) + "\n"


def pick_target(seed: int, n_sections: int, frac: float) -> int:
    """First section at/after frac*n whose text is 750-950 chars (~220-280 tokens), so the verbatim
    copy plus a 3-bullet summary fits into max_tokens=400."""
    start = max(1, int(round(frac * n_sections)))
    for idx in list(range(start, n_sections + 1)) + list(range(start - 1, 0, -1)):
        if 750 <= len(make_section(seed, idx)) <= 950:
            return idx
    return start


INSTRUCTION = (
    "The document above is the complete changelog of the tessera project.\n\n"
    "Task:\n"
    "1. Copy the section labelled [S{idx:03d}] verbatim - every line exactly as written, from its "
    "heading line through its \"Notes:\" line, keeping blank lines and punctuation. Put the copy "
    "between a line containing only " + BEGIN_MARK + " and a line containing only " + END_MARK + ". "
    "Do not use code fences.\n"
    "2. After " + END_MARK + ", summarize that section in exactly 3 bullet points."
)
SYSTEM = "You are a precise assistant. When asked to quote text, you reproduce it character for character."


def make_prompt(seed: int, n_sections: int, frac: float) -> dict:
    target = pick_target(seed, n_sections, frac)
    doc = build_document(seed, n_sections)
    user = "<document>\n" + doc + "</document>\n\n" + INSTRUCTION.format(idx=target)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    return {
        "seed": seed,
        "n_sections": n_sections,
        "target_index": target,
        "target_label": "[S%03d]" % target,
        "target_section": make_section(seed, target),
        "messages": messages,
        "doc_chars": len(doc),
        "prompt_sha256": hashlib.sha256(blob).hexdigest()[:16],
    }


# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------

def http_json(url: str, payload=None, timeout: float = 60.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", errors="replace") or "null")


def count_tokens(url: str, messages: list) -> int:
    _, tmpl = http_json(url + "/apply-template", {"messages": messages}, timeout=120)
    prompt = tmpl["prompt"]
    _, tok = http_json(url + "/tokenize", {"content": prompt, "add_special": True}, timeout=120)
    return len(tok["tokens"])


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie check (child that exited but was not reaped yet)
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


# --------------------------------------------------------------------------------------
# Verbatim check + text comparison
# --------------------------------------------------------------------------------------

def _norm(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # drop code fences the model may add despite the instruction
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def first_diff(a: str, b: str) -> int:
    """Index of the first differing character, -1 if identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def verbatim_check(text: str, section: str, label: str) -> dict:
    t = text.replace("\r\n", "\n")
    has_begin = BEGIN_MARK in t
    has_end = END_MARK in t
    if has_begin:
        body = t.split(BEGIN_MARK, 1)[1]
        body = body.split(END_MARK, 1)[0] if END_MARK in body else body
        how = "markers" if has_end else "begin-only"
    else:
        # fallback: start at the heading of the requested section
        pos = t.find("### " + label)
        if pos < 0:
            pos = t.find(label)
        if pos < 0:
            return {"how": "not-found", "exact": False, "ratio": 0.0, "first_diff": 0,
                    "copied_chars": 0, "section_chars": len(section)}
        body = t[pos:]
        body = body.split(END_MARK, 1)[0]
        body = body[: len(section) + 200]
        how = "heading"
    got = _norm(body)
    want = _norm(section)
    if how == "heading":
        got = got[: len(want)]
    return {
        "how": how,
        "exact": got == want,
        "ratio": round(similarity(got, want), 5),
        "first_diff": first_diff(got, want),
        "copied_chars": len(got),
        "section_chars": len(want),
    }


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------

def cmd_calibrate(a) -> int:
    url = a.url.rstrip("/")
    lo, hi = 10, a.max_sections
    best = None
    cache = {}

    def toks(n):
        if n not in cache:
            p = make_prompt(a.seed, n, a.target_frac)
            cache[n] = (count_tokens(url, p["messages"]), p)
            print("  calibrate: %4d sections -> %6d tokens" % (n, cache[n][0]), file=sys.stderr)
        return cache[n]

    # guess from ~3.4 chars/token and ~1000 chars/section, then binary search
    guess = max(lo, min(hi, int(a.target_tokens * 3.4 / 1000)))
    t, _ = toks(guess)
    if t <= a.target_tokens:
        lo = guess
    else:
        hi = guess - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        t, _ = toks(mid)
        if t <= a.target_tokens:
            lo = mid
        else:
            hi = mid - 1
    t, p = toks(lo)
    if t > a.target_tokens:
        print("calibrate: even %d sections exceed the budget" % lo, file=sys.stderr)
        return 2
    best = p
    best["prompt_tokens"] = t
    best["calibrated"] = True
    best["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1)
    print("calibrate: %d sections, %d prompt tokens, target %s (%d chars), sha %s -> %s"
          % (best["n_sections"], t, best["target_label"], len(best["target_section"]),
             best["prompt_sha256"], a.out), file=sys.stderr)
    return 0


def cmd_offline(a) -> int:
    # rough estimate only: ~3.4 chars per token for this kind of text
    n = 10
    while True:
        chars = len(build_document(a.seed, n + 1))
        if chars / 3.4 > a.target_tokens:
            break
        n += 1
        if n >= a.max_sections:
            break
    p = make_prompt(a.seed, n, a.target_frac)
    p["prompt_tokens"] = int(len(p["messages"][1]["content"]) / 3.4)
    p["calibrated"] = False
    p["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=1)
    print("offline: %d sections, ~%d tokens (ESTIMATE), target %s, sha %s -> %s"
          % (n, p["prompt_tokens"], p["target_label"], p["prompt_sha256"], a.out), file=sys.stderr)
    return 0


def cmd_health(a) -> int:
    url = a.url.rstrip("/") + "/health"
    t0 = time.time()
    last = ""
    while time.time() - t0 < a.timeout:
        if not pid_alive(a.pid):
            print("health: server PID %d died during startup" % a.pid, file=sys.stderr)
            return 3
        try:
            st, body = http_json(url, timeout=5)
            if st == 200 and (not isinstance(body, dict) or body.get("status", "ok") == "ok"):
                print("health: ok after %.1fs" % (time.time() - t0), file=sys.stderr)
                return 0
            last = str(body)
        except urllib.error.HTTPError as e:  # 503 while loading
            last = "HTTP %d" % e.code
        except Exception as e:  # connection refused etc.
            last = type(e).__name__
        time.sleep(0.5)
    print("health: timeout after %ds (last: %s)" % (a.timeout, last), file=sys.stderr)
    return 4


def read_loadavg() -> str:
    try:
        with open("/proc/loadavg") as f:
            return " ".join(f.read().split()[:3])
    except OSError:
        return ""


TIMING_KEYS = ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
               "predicted_per_second", "draft_n", "draft_n_accepted", "cache_n"]


def _rate(rec: dict, what: str) -> float:
    n, ms = rec.get(what + "_n"), rec.get(what + "_ms")
    return (n / ms * 1000.0) if n and ms else 0.0


def cmd_run(a) -> int:
    with open(a.prompt, encoding="utf-8") as f:
        P = json.load(f)
    url = a.url.rstrip("/") + "/v1/chat/completions"
    body = {
        "messages": P["messages"],
        "temperature": 0.0,
        "seed": a.seed,
        "max_tokens": a.max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": a.kind, "round": a.round, "slot": a.slot, "rep": a.rep,
        "build": a.build, "bindir": a.bindir, "pair": a.pair, "ctk": a.ctk, "ctv": a.ctv,
        "spec": a.spec, "server_log": a.server_log, "max_tokens": a.max_tokens,
        "prompt_sha256": P["prompt_sha256"], "expected_prompt_tokens": P.get("prompt_tokens"),
        "loadavg_before": read_loadavg(),
    }
    t0 = time.perf_counter()
    try:
        _, resp = http_json(url, body, timeout=a.timeout)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        rec["http_ok"] = True
    except Exception as e:
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        rec["http_ok"] = False
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:500])
        resp = None
    text = ""
    if resp:
        try:
            ch = resp["choices"][0]
            text = ch["message"].get("content") or ""
            rec["finish_reason"] = ch.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            rec["http_ok"] = False
            rec["error"] = "unexpected response: " + json.dumps(resp)[:500]
        tm = resp.get("timings") or {}
        for k in TIMING_KEYS:
            if k in tm:
                rec[k] = tm[k]
        rec["timings_raw"] = tm
        rec["usage"] = resp.get("usage")
        rec["system_fingerprint"] = resp.get("system_fingerprint")
    rec["text"] = text
    rec["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    if a.text_dir:
        os.makedirs(a.text_dir, exist_ok=True)
        name = "%s_%s_%s_%s_r%d_s%d_rep%d.txt" % (a.kind, a.build, a.pair.replace("/", "-"), a.spec,
                                                  a.round, a.slot, a.rep)
        path = os.path.join(a.text_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        rec["text_path"] = path
    rec["verbatim"] = verbatim_check(text, P["target_section"], P["target_label"])
    rec["loadavg_after"] = read_loadavg()
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if rec["http_ok"]:
        acc = ""
        if rec.get("draft_n"):
            n_acc, n_draft = int(rec.get("draft_n_accepted", 0)), int(rec["draft_n"])
            acc = " acc %d/%d (%.0f%%)" % (n_acc, n_draft, 100.0 * n_acc / n_draft)
        print("  %-7s %-6s %-14s spec=%-3s rep%d: prefill %6s tok %7.1f t/s | decode %4s tok %6.2f t/s%s"
              " | wall %6.1fs | verbatim %s %.3f"
              % (a.kind, a.build, a.pair, a.spec, a.rep, rec.get("prompt_n", "?"),
                 _rate(rec, "prompt"), rec.get("predicted_n", "?"), _rate(rec, "predicted"), acc, rec["wall_s"],
                 "EXACT" if rec["verbatim"]["exact"] else rec["verbatim"]["how"], rec["verbatim"]["ratio"]),
              file=sys.stderr)
        return 0
    print("  request FAILED: %s" % rec.get("error"), file=sys.stderr)
    return 5


def cmd_show(a) -> int:
    with open(a.prompt, encoding="utf-8") as f:
        P = json.load(f)
    print("sections: %d  prompt tokens: %s (calibrated: %s)  sha: %s"
          % (P["n_sections"], P.get("prompt_tokens"), P.get("calibrated"), P["prompt_sha256"]))
    print("target: %s (%d chars)\n" % (P["target_label"], len(P["target_section"])))
    print(P["target_section"])
    print("\n--- instruction ---")
    print(P["messages"][1]["content"].rsplit("</document>", 1)[1].strip())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("calibrate", "offline"):
        s = sub.add_parser(name, help="write a prompt.json (%s)" % (
            "token count from a running server" if name == "calibrate" else "token count estimated, no server"))
        if name == "calibrate":
            s.add_argument("--url", required=True, help="llama-server base URL, e.g. http://127.0.0.1:8080")
        s.add_argument("--out", required=True, help="prompt.json to write")
        s.add_argument("--seed", type=int, default=1234, help="document seed (default 1234)")
        s.add_argument("--target-tokens", type=int, default=29000, help="prompt size in tokens (default 29000)")
        s.add_argument("--target-frac", type=float, default=0.55,
                       help="relative position of the section to copy (0.55 = a bit past the middle)")
        s.add_argument("--max-sections", type=int, default=400, help="upper bound for the document length")

    s = sub.add_parser("health", help="wait for /health")
    s.add_argument("--url", required=True, help="llama-server base URL")
    s.add_argument("--pid", type=int, default=0, help="server PID; stop waiting when it exits (0 = do not check)")
    s.add_argument("--timeout", type=int, default=300, help="seconds (default 300)")

    s = sub.add_parser("run", help="send one request and append the result to a JSONL file")
    s.add_argument("--url", required=True, help="llama-server base URL")
    s.add_argument("--prompt", required=True, help="prompt.json from calibrate/offline")
    s.add_argument("--out", required=True, help="JSONL file to append to")
    s.add_argument("--text-dir", default="", help="also write the generated text to a file in this directory")
    s.add_argument("--kind", default="measure", choices=["measure", "warmup"])
    s.add_argument("--build", required=True, help="label of the build or configuration (stored in the record)")
    s.add_argument("--bindir", default="", help="label: binary directory (stored in the record)")
    s.add_argument("--pair", required=True, help="label: K/V types, e.g. q8_0/vtq4_1")
    s.add_argument("--ctk", default="", help="label: K type")
    s.add_argument("--ctv", default="", help="label: V type")
    s.add_argument("--spec", required=True, choices=["on", "off"], help="label: speculative decoding on the server")
    s.add_argument("--round", type=int, default=0, help="label: round number")
    s.add_argument("--slot", type=int, default=0, help="label: server start number")
    s.add_argument("--rep", type=int, default=0, help="label: repetition number")
    s.add_argument("--seed", type=int, default=42, help="sampling seed sent with the request (decoding is greedy)")
    s.add_argument("--max-tokens", type=int, default=400, help="max_tokens of the request (default 400)")
    s.add_argument("--timeout", type=int, default=900, help="request timeout in seconds (default 900)")
    s.add_argument("--server-log", default="", help="label: path of the server log (stored in the record)")

    s = sub.add_parser("show", help="print the target section and the instruction of a prompt.json")
    s.add_argument("--prompt", required=True)

    a = ap.parse_args()
    return {"calibrate": cmd_calibrate, "offline": cmd_offline, "health": cmd_health,
            "run": cmd_run, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
