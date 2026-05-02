#!/usr/bin/env python3
"""
TurboQuant v8 Unified Sim — round-trip MSE on synthetic K/V tensors.

Validates: is "Trellis + Outlier-Sidecar + Lloyd-Max" theoretically better than
the existing 15 KTQ/VTQ types BEFORE we write CUDA?

Pure numpy. No torch. No CUDA. Reproducible (seed=42).

Usage:
    python3 bench/sim/v8_unified_sim.py
    python3 bench/sim/v8_unified_sim.py > /tmp/v8_sim.log 2>&1

Output:
    bench/sim/v8_unified_results.md  (table + GO/NO-GO gate decision)

Constants are verified against ggml-common.h:
    QK_KTQ = 32, QK_VTQ = 32, QK_VTQ_TRELLIS = 128, VTQ_OUTLIER_K = 4
"""

from __future__ import annotations
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (verified against ggml-common.h and ggml-quants.c)
# ---------------------------------------------------------------------------
SEED = 42
QK_KTQ = 32
QK_VTQ = 32
QK_VTQ_TRELLIS = 128
VTQ_OUTLIER_K = 4

# Tensor dimensions (small enough to run < 5 min)
HEAD_DIM = 128
N_LAYERS = 32
N_TOKENS = 2048

# Lloyd-Max codebooks for Beta(15.5, 15.5) (post-RHT marginal at d=32)
PQ_CB_1BIT = np.array([-0.797885, 0.797885], dtype=np.float32)
PQ_CB_2BIT = np.array([-1.489560, -0.451428, 0.451428, 1.489560], dtype=np.float32)
PQ_CB_3BIT = np.array([
    -2.071926, -1.314996, -0.745325, -0.242405,
     0.242405,  0.745325,  1.314996,  2.071926,
], dtype=np.float32)
PQ_CB_4BIT = np.array([
    -2.732590, -2.069017, -1.618046, -1.256231,
    -0.942340, -0.656759, -0.388048, -0.128395,
     0.128395,  0.388048,  0.656759,  0.942340,
     1.256231,  1.618046,  2.069017,  2.732590,
], dtype=np.float32)

# Laplace-optimized 2-bit codebook for VTQ_1
VTQ_CB_2BIT = np.array([-1.810, -0.395, 0.395, 1.810], dtype=np.float32)

CB_SCALE_KTQ = 1.0 / math.sqrt(QK_KTQ)
CB_SCALE_VTQ = 1.0 / math.sqrt(QK_VTQ)
CB_SCALE_TRELLIS = 1.0 / math.sqrt(QK_VTQ_TRELLIS)


# ---------------------------------------------------------------------------
# Synthetic K/V generators
# ---------------------------------------------------------------------------
def gen_K_tensor(rng, n_tokens, n_heads, head_dim):
    raw = rng.standard_normal((n_tokens, n_heads, head_dim)).astype(np.float32)
    return raw


def gen_V_tensor(rng, n_tokens, n_heads, head_dim):
    base = rng.standard_normal((n_tokens, n_heads, head_dim)).astype(np.float32)
    laplace = rng.laplace(scale=0.7, size=base.shape).astype(np.float32)
    outlier_mask = rng.random(base.shape) < 0.01
    outliers = (rng.standard_normal(base.shape) * 5.0).astype(np.float32) * outlier_mask
    return base * 0.5 + laplace * 0.5 + outliers


# ---------------------------------------------------------------------------
# Philox-2x32 6-round
# ---------------------------------------------------------------------------
_PHILOX_M = np.uint64(0xD2511F53)
_PHILOX_W = np.uint32(0x9E3779B9)


def philox_6r(counter, key):
    lo = counter.astype(np.uint64)
    hi = np.full_like(lo, np.uint64(key))
    for i in range(6):
        lo_old = lo.copy()
        prod = lo_old * _PHILOX_M
        lo = (prod >> np.uint64(32)) ^ hi ^ (np.uint64(_PHILOX_W) * np.uint64(i + 1))
        hi = (prod & np.uint64(0xFFFFFFFF))
        lo = lo & np.uint64(0xFFFFFFFF)
    return lo.astype(np.uint32)


def rht_signs(seed, n=QK_KTQ):
    counters = np.arange(n, dtype=np.uint32)
    bits = philox_6r(counters, np.uint32(seed)) & np.uint32(1)
    return np.where(bits == 1, 1.0, -1.0).astype(np.float32)


def derive_seed(block_index):
    h = np.uint32(2166136261)
    bi = np.uint64(block_index)
    for shift in (0, 8, 16, 24):
        h ^= np.uint32((bi >> np.uint64(shift)) & np.uint64(0xFF))
        h = (h * np.uint32(16777619)).astype(np.uint32)
    return int(h & 0xFFFF)


# ---------------------------------------------------------------------------
# FWHT
# ---------------------------------------------------------------------------
def fwht_inplace(x):
    x = x.copy()
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i:i + h].copy()
            b = x[..., i + h:i + h * 2].copy()
            x[..., i:i + h] = a + b
            x[..., i + h:i + h * 2] = a - b
        h *= 2
    return x / math.sqrt(n)


def rht_forward(x, seed):
    signs = rht_signs(seed, x.shape[-1])
    return fwht_inplace(x * signs)


def rht_inverse(y, seed):
    signs = rht_signs(seed, y.shape[-1])
    return fwht_inplace(y) * signs


# ---------------------------------------------------------------------------
# Codebook quant
# ---------------------------------------------------------------------------
def quantize_codebook(values, codebook):
    diff = values[..., np.newaxis] - codebook
    return np.argmin(diff * diff, axis=-1).astype(np.int32)


def dequantize_codebook(indices, codebook):
    return codebook[indices]


# ---------------------------------------------------------------------------
# Existing types — round-trip implementations
# ---------------------------------------------------------------------------
def roundtrip_ktq(x, bits):
    cb = {1: PQ_CB_1BIT, 2: PQ_CB_2BIT, 3: PQ_CB_3BIT, 4: PQ_CB_4BIT}[bits]
    out = np.empty_like(x)
    flat = x.reshape(-1, QK_KTQ)
    for i in range(flat.shape[0]):
        block = flat[i]
        norm = np.linalg.norm(block)
        if norm < 1e-30:
            out.reshape(-1, QK_KTQ)[i] = 0
            continue
        x_hat = block / norm
        seed = derive_seed(i)
        rotated = rht_forward(x_hat, seed)
        idx = quantize_codebook(rotated, cb * CB_SCALE_KTQ)
        recon = dequantize_codebook(idx, cb * CB_SCALE_KTQ)
        result = rht_inverse(recon, seed)
        recon_norm = np.linalg.norm(result)
        d = norm / recon_norm if recon_norm > 1e-30 else norm
        out.reshape(-1, QK_KTQ)[i] = result * d
    return out


def roundtrip_vtq_1(x, bits):
    if bits == 1:
        cb = PQ_CB_1BIT
    elif bits == 2:
        cb = VTQ_CB_2BIT
    elif bits == 3:
        cb = PQ_CB_3BIT
    else:
        cb = PQ_CB_4BIT
    out = np.empty_like(x)
    flat = x.reshape(-1, QK_VTQ)
    for i in range(flat.shape[0]):
        block = flat[i]
        norm = np.linalg.norm(block)
        if norm < 1e-30:
            out.reshape(-1, QK_VTQ)[i] = 0
            continue
        x_hat = block / norm
        idx = quantize_codebook(x_hat, cb * CB_SCALE_VTQ)
        recon = dequantize_codebook(idx, cb * CB_SCALE_VTQ)
        recon_norm = np.linalg.norm(recon)
        d = norm / recon_norm if recon_norm > 1e-30 else norm
        out.reshape(-1, QK_VTQ)[i] = recon * d
    return out


def _build_trellis_table(S):
    s = np.arange(S, dtype=np.uint32)
    h = (s * np.uint32(0x9E3779B1) + np.uint32(0x7F4A7C15)).astype(np.uint64)
    p_real = ((h >> np.uint64(1)).astype(np.float64) + 0.5) / float(1 << 31)
    p_real = np.clip(p_real, 1e-12, 1.0 - 1e-12)
    return _inv_norm_cdf_acklam(p_real).astype(np.float32)


def _inv_norm_cdf_acklam(p):
    a = np.array([-3.969683028665376e+01,  2.209460984245205e+02,
                  -2.759285104469687e+02,  1.383577518672690e+02,
                  -3.066479806614716e+01,  2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01,  1.615858368580409e+02,
                  -1.556989798598866e+02,  6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                   4.374664141464968e+00,  2.938163982698783e+00])
    d = np.array([ 7.784695709041462e-03,  3.224671290700398e-01,
                   2.445134137142996e+00,  3.754408661907416e+00])
    plow, phigh = 0.02425, 1.0 - 0.02425
    out = np.empty_like(p)
    low = p < plow
    high = p > phigh
    mid = ~(low | high)
    if low.any():
        q = np.sqrt(-2 * np.log(p[low]))
        out[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    if high.any():
        q = np.sqrt(-2 * np.log(1 - p[high]))
        out[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    return out


def _viterbi_beam(xn, K, table, cb_scale, beam):
    L = 16
    S = 1 << L
    Lmask = S - 1
    N = xn.shape[0]
    INF = np.float32(np.inf)

    dp_cur = np.zeros(S, dtype=np.float32)
    bt = np.zeros((N, S), dtype=np.uint16)
    active_cur = np.arange(S, dtype=np.uint32)

    for i in range(N):
        xi = xn[i]
        prev_arr = active_cur
        prevs_expanded = np.repeat(prev_arr, 1 << K)
        kks = np.tile(np.arange(1 << K, dtype=np.uint32), prev_arr.shape[0])
        next_states = ((prevs_expanded >> np.uint32(K)) |
                       (kks << np.uint32(L - K))) & np.uint32(Lmask)
        codes = table[next_states] * cb_scale
        diff = xi - codes
        costs = dp_cur[prevs_expanded] + diff * diff

        dp_next = np.full(S, INF, dtype=np.float32)
        bt_i = np.zeros(S, dtype=np.uint16)
        order = np.argsort(next_states, kind="stable")
        ns_sorted = next_states[order]
        cs_sorted = costs[order]
        prev_sorted = prevs_expanded[order]
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(ns_sorted)) + 1, [len(ns_sorted)]))
        for b in range(len(boundaries) - 1):
            beg, end = boundaries[b], boundaries[b + 1]
            sl = slice(beg, end)
            best = np.argmin(cs_sorted[sl])
            s = ns_sorted[beg]
            dp_next[s] = cs_sorted[sl][best]
            bt_i[s] = prev_sorted[sl][best]

        touched = np.flatnonzero(np.isfinite(dp_next))
        if touched.size > beam:
            costs_t = dp_next[touched]
            keep = np.argpartition(costs_t, beam)[:beam]
            touched = touched[keep]

        bt[i] = bt_i
        dp_cur = dp_next
        active_cur = touched.astype(np.uint32)

    best_s = int(active_cur[np.argmin(dp_cur[active_cur])])
    states = np.empty(N + 1, dtype=np.uint32)
    states[N] = best_s
    for i in range(N - 1, -1, -1):
        states[i] = bt[i, states[i + 1]]
    return states


_TRELLIS_TABLE = None
def trellis_table():
    global _TRELLIS_TABLE
    if _TRELLIS_TABLE is None:
        _TRELLIS_TABLE = _build_trellis_table(1 << 16)
    return _TRELLIS_TABLE


def roundtrip_vtq_2_trellis(x, bits):
    BEAM = 1024
    table = trellis_table()
    cb_scale = CB_SCALE_TRELLIS

    out = np.empty_like(x)
    flat = x.reshape(-1, QK_VTQ_TRELLIS)
    for blk_i in range(flat.shape[0]):
        block = flat[blk_i]
        norm = np.linalg.norm(block)
        if norm < 1e-30:
            out.reshape(-1, QK_VTQ_TRELLIS)[blk_i] = 0
            continue
        xn = block / norm
        states_path = _viterbi_beam(xn, bits, table, cb_scale, beam=BEAM)
        decoded = table[states_path[1:]] * cb_scale
        recon_norm = np.linalg.norm(decoded)
        d = norm / recon_norm if recon_norm > 1e-30 else norm
        out.reshape(-1, QK_VTQ_TRELLIS)[blk_i] = decoded * d
    return out


def _fp16_roundtrip(v):
    return float(np.float32(np.float16(v)))


def roundtrip_vtq_3_outlier(x, bits):
    out = np.empty_like(x)
    flat = x.reshape(-1, QK_VTQ_TRELLIS)
    K_OUT = VTQ_OUTLIER_K
    for blk_i in range(flat.shape[0]):
        block = flat[blk_i]
        outlier_idx = np.argpartition(np.abs(block), -K_OUT)[-K_OUT:]
        masked = block.copy()
        masked[outlier_idx] = 0.0
        decoded_masked = roundtrip_vtq_2_trellis(masked.reshape(1, QK_VTQ_TRELLIS), bits).reshape(-1)
        decoded = decoded_masked.copy()
        for p in outlier_idx:
            decoded[p] = _fp16_roundtrip(block[p])
        out.reshape(-1, QK_VTQ_TRELLIS)[blk_i] = decoded
    return out


# ---------------------------------------------------------------------------
# v8 UNIFIED candidates
# ---------------------------------------------------------------------------
def quant_ktq_v8(x, bits, n_outliers=4):
    """V8 KTQ: RHT + Lloyd-Max + outlier-sidecar."""
    cb = {1: PQ_CB_1BIT, 2: PQ_CB_2BIT, 3: PQ_CB_3BIT, 4: PQ_CB_4BIT}[bits]
    out = np.empty_like(x)
    flat = x.reshape(-1, QK_KTQ)
    for i in range(flat.shape[0]):
        block = flat[i]
        if n_outliers > 0:
            outlier_idx = np.argpartition(np.abs(block), -n_outliers)[-n_outliers:]
        else:
            outlier_idx = np.array([], dtype=int)
        masked = block.copy()
        masked[outlier_idx] = 0.0
        norm = np.linalg.norm(masked)
        if norm < 1e-30:
            decoded = np.zeros_like(block)
        else:
            x_hat = masked / norm
            seed = derive_seed(i)
            rotated = rht_forward(x_hat, seed)
            idx = quantize_codebook(rotated, cb * CB_SCALE_KTQ)
            recon = dequantize_codebook(idx, cb * CB_SCALE_KTQ)
            result = rht_inverse(recon, seed)
            recon_norm = np.linalg.norm(result)
            d = norm / recon_norm if recon_norm > 1e-30 else norm
            decoded = result * d
        for p in outlier_idx:
            decoded[p] = _fp16_roundtrip(block[p])
        out.reshape(-1, QK_KTQ)[i] = decoded
    return out


def quant_vtq_v8(x, bits, n_outliers=2):
    """V8 VTQ: bits 2,3 = Trellis; bits=4 = Lloyd-Max codebook. With outliers."""
    out = np.empty_like(x)
    flat = x.reshape(-1, QK_VTQ_TRELLIS)
    for blk_i in range(flat.shape[0]):
        block = flat[blk_i]
        if n_outliers > 0:
            outlier_idx = np.argpartition(np.abs(block), -n_outliers)[-n_outliers:]
        else:
            outlier_idx = np.array([], dtype=int)
        masked = block.copy()
        masked[outlier_idx] = 0.0
        if bits in (2, 3):
            decoded = roundtrip_vtq_2_trellis(masked.reshape(1, QK_VTQ_TRELLIS), bits).reshape(-1)
        else:
            cb = PQ_CB_4BIT
            norm = np.linalg.norm(masked)
            if norm < 1e-30:
                decoded = np.zeros_like(masked)
            else:
                x_hat = masked / norm
                idx = quantize_codebook(x_hat, cb * CB_SCALE_TRELLIS)
                recon = dequantize_codebook(idx, cb * CB_SCALE_TRELLIS)
                recon_norm = np.linalg.norm(recon)
                d = norm / recon_norm if recon_norm > 1e-30 else norm
                decoded = recon * d
        for p in outlier_idx:
            decoded[p] = _fp16_roundtrip(block[p])
        out.reshape(-1, QK_VTQ_TRELLIS)[blk_i] = decoded
    return out


# ---------------------------------------------------------------------------
# bpw accounting
# ---------------------------------------------------------------------------
BPW_TABLE = {
    "ktq1_1": 2.5, "ktq2_1": 3.5, "ktq3_1": 4.5, "ktq4_1": 5.5,
    "vtq1_1": 1.5, "vtq2_1": 2.5, "vtq3_1": 4.0, "vtq4_1": 4.5,
    "vtq2_2": 2.25, "vtq3_2": 3.25, "vtq4_2": 4.25,
    "vtq2_3": 3.00, "vtq3_3": 4.00, "vtq4_3": 5.00,
    "ktq1_v8": 5.5, "ktq2_v8": 6.5, "ktq3_v8": 7.5, "ktq4_v8": 8.5,
    "vtq2_v8": 2.625, "vtq3_v8": 3.625, "vtq4_v8": 4.625,
}


# ---------------------------------------------------------------------------
# MSE measurement & gate
# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    role: str
    bpw: float
    mse: float
    rel_mse: float
    ppl_drift_pct: float


def measure_mse(name, role, x, decoded):
    mse = float(np.mean((x - decoded) ** 2))
    var = float(np.var(x))
    rel = mse / var if var > 1e-30 else mse
    ppl_drift = rel * 0.6 * 100
    return Result(name=name, role=role, bpw=BPW_TABLE[name], mse=mse, rel_mse=rel, ppl_drift_pct=ppl_drift)


def main():
    rng = np.random.default_rng(SEED)
    n_heads = N_LAYERS
    print(f"[v8-sim] generating K (head_dim={HEAD_DIM}, n_heads={n_heads}, n_tok={N_TOKENS})…", flush=True)
    K = gen_K_tensor(rng, N_TOKENS, n_heads, HEAD_DIM)
    V = gen_V_tensor(rng, N_TOKENS, n_heads, HEAD_DIM)
    n_K_blocks = K.size // QK_KTQ
    print(f"[v8-sim] K blocks(32):  {n_K_blocks}", flush=True)

    K32 = K.reshape(-1, QK_KTQ).astype(np.float32)
    V32 = V.reshape(-1, QK_VTQ).astype(np.float32)
    V128 = V.reshape(-1, QK_VTQ_TRELLIS).astype(np.float32)

    MAX_K_BLOCKS = 2048
    MAX_V32_BLOCKS = 2048
    MAX_V128_BLOCKS = 256
    if K32.shape[0] > MAX_K_BLOCKS: K32 = K32[:MAX_K_BLOCKS]
    if V32.shape[0] > MAX_V32_BLOCKS: V32 = V32[:MAX_V32_BLOCKS]
    if V128.shape[0] > MAX_V128_BLOCKS: V128 = V128[:MAX_V128_BLOCKS]
    print(f"[v8-sim] capped: K32={K32.shape[0]}, V32={V32.shape[0]}, V128={V128.shape[0]}", flush=True)

    results = []

    def time_run(fn, *a, **kw):
        t0 = time.time()
        r = fn(*a, **kw)
        return r, time.time() - t0

    # KTQ existing
    for bits in (1, 2, 3, 4):
        name = f"ktq{bits}_1"
        print(f"[v8-sim] running {name}…", flush=True)
        dec, dt = time_run(roundtrip_ktq, K32, bits)
        print(f"  {name} took {dt:.1f}s", flush=True)
        results.append(measure_mse(name, "K", K32, dec))

    # VTQ_1 existing
    for bits in (1, 2, 3, 4):
        name = f"vtq{bits}_1"
        print(f"[v8-sim] running {name}…", flush=True)
        dec, dt = time_run(roundtrip_vtq_1, V32, bits)
        print(f"  {name} took {dt:.1f}s", flush=True)
        results.append(measure_mse(name, "V", V32, dec))

    # VTQ_2 Trellis
    for bits in (2, 3, 4):
        name = f"vtq{bits}_2"
        print(f"[v8-sim] running {name} (trellis)…", flush=True)
        dec, dt = time_run(roundtrip_vtq_2_trellis, V128, bits)
        print(f"  {name} took {dt:.1f}s", flush=True)
        results.append(measure_mse(name, "V", V128, dec))

    # VTQ_3 Trellis + outlier
    for bits in (2, 3, 4):
        name = f"vtq{bits}_3"
        print(f"[v8-sim] running {name} (trellis+outlier)…", flush=True)
        dec, dt = time_run(roundtrip_vtq_3_outlier, V128, bits)
        print(f"  {name} took {dt:.1f}s", flush=True)
        results.append(measure_mse(name, "V", V128, dec))

    # v8 UNIFIED
    print("[v8-sim] === V8 UNIFIED ===", flush=True)
    for bits in (1, 2, 3, 4):
        name = f"ktq{bits}_v8"
        print(f"[v8-sim] running {name}…", flush=True)
        dec, dt = time_run(quant_ktq_v8, K32, bits, 4)
        print(f"  {name} took {dt:.1f}s", flush=True)
        results.append(measure_mse(name, "K", K32, dec))

    for bits in (2, 3, 4):
        name = f"vtq{bits}_v8"
        print(f"[v8-sim] running {name}…", flush=True)
        dec, dt = time_run(quant_vtq_v8, V128, bits, 2)
        print(f"  {name} took {dt:.1f}s", flush=True)
        results.append(measure_mse(name, "V", V128, dec))

    # Gate decisions per role
    by_role = {"K": [r for r in results if r.role == "K"],
               "V": [r for r in results if r.role == "V"]}
    bpw_buckets = {2.0: [], 2.5: [], 3.0: [], 3.5: [], 4.0: [], 4.5: [], 5.0: [], 5.5: [], 6.0: [], 6.5: [], 7.5: [], 8.5: []}
    for r in results:
        bucket = min(bpw_buckets.keys(), key=lambda b: abs(b - r.bpw))
        if abs(bucket - r.bpw) <= 0.4:
            bpw_buckets[bucket].append(r)

    decisions = []
    for bucket, items in bpw_buckets.items():
        existing = [x for x in items if "_v8" not in x.name]
        v8 = [x for x in items if "_v8" in x.name]
        if not v8 or not existing:
            continue
        best_e = min(existing, key=lambda r: r.mse)
        for v in v8:
            decisions.append((bucket, best_e.name, best_e.mse, v.name, v.mse, v.mse <= best_e.mse * 1.05))

    n_wins = sum(1 for d in decisions if d[5])
    n_losses = len(decisions) - n_wins
    overall_go = n_losses == 0

    out_path = Path(__file__).parent / "v8_unified_results.md"
    lines = []
    lines.append("# TurboQuant v8 Unified Sim — Results\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Seed={SEED}, head_dim={HEAD_DIM}, n_heads={N_LAYERS}, n_tokens={N_TOKENS}\n")
    lines.append(f"K-blocks tested: {K32.shape[0]} × 32 = {K32.size} samples\n")
    lines.append(f"V-blocks tested (32): {V32.shape[0]} × 32 = {V32.size} samples\n")
    lines.append(f"V-blocks tested (128): {V128.shape[0]} × 128 = {V128.size} samples\n\n")
    lines.append("## Round-trip MSE\n\n")
    lines.append("| Type | Role | bpw | MSE | rel_MSE | PPL-Δ-Estimate (%) |\n")
    lines.append("|------|:----:|:---:|----:|--------:|-------------------:|\n")
    for r in sorted(results, key=lambda x: (x.role, x.bpw)):
        lines.append(f"| `{r.name}` | {r.role} | {r.bpw:.2f} | {r.mse:.6e} | {r.rel_mse:.4f} | {r.ppl_drift_pct:+.2f} |\n")
    lines.append("\n## Per-bpw-class winners\n\n")
    lines.append("| bpw | Best existing | Best MSE | v8 candidate | v8 MSE | v8 wins (within 5%)? |\n")
    lines.append("|:---:|---------------|---------:|--------------|-------:|:--------------------:|\n")
    for bucket, exname, exmse, v8name, v8mse, win in decisions:
        marker = "YES" if win else "NO"
        lines.append(f"| {bucket:.2f} | `{exname}` | {exmse:.6e} | `{v8name}` | {v8mse:.6e} | {marker} |\n")
    lines.append("\n## Gate decision\n\n")
    if overall_go:
        lines.append(f"**GO** — all {len(decisions)} v8 candidates beat or tie the best existing type in their bpw class (within 5% MSE tolerance).\n\n")
        lines.append("Recommendation: proceed to CUDA implementation.\n")
    else:
        lines.append(f"**HYBRID** — {n_wins}/{len(decisions)} v8 candidates win; {n_losses} lose.\n\n")
        lines.append("Losing candidates:\n")
        for bucket, exname, exmse, v8name, v8mse, win in decisions:
            if not win:
                gap = (v8mse - exmse) / exmse * 100
                lines.append(f"- `{v8name}` ({bucket:.2f} bpw) loses to `{exname}` by +{gap:.2f}% MSE — keep existing for that tier.\n")
    lines.append("\n## Caveats\n\n")
    lines.append("- Trellis path uses beam-pruned Viterbi (beam=1024 instead of full 2^16). MSE is slightly pessimistic vs the full Viterbi CUDA path. SAFE bias for GO/NO-GO gate.\n")
    lines.append("- Synthetic V is Gauss + Laplace + 1% 5σ-outlier mix. Real V from a trained transformer may have heavier tails — re-run with real tensors before CUDA commit.\n")
    lines.append("- Calibration `PPL_drift_pct ≈ rel_mse * 0.6` is from `ktq2_1+vtq2_2` row in `bench/plots/benchmarks.csv` (PPL +0.16%).\n")

    with out_path.open("w") as f:
        f.writelines(lines)
    print(f"[v8-sim] report written to {out_path}", flush=True)
    print(f"[v8-sim] gate: {'GO' if overall_go else 'HYBRID'} ({n_wins}/{len(decisions)} v8 wins)", flush=True)
    return 0 if overall_go else 1


if __name__ == "__main__":
    sys.exit(main())
