#!/usr/bin/env python3
"""
Real-codebook QAT for the DiffusionGemma self_cond MLP (hybrid: numpy autograd + the
REAL llama.cpp quant codebook via the quant_roundtrip C shim).

Why this exists: the PyTorch STE-proxy QAT (dg_selfcond_qat.py) overfit its own symmetric
2-bit grid — the learned mid-phase contraction (map-ρ 1.5) vanished under the real q2_K
deploy codebook (ρ back to 13.2). gguf-py can't quantize k-/i-quants, so there was no way
to put the real codebook in the loss. This script does: it calls ggml_quantize_chunk (the
exact function llama-quantize uses) through libquant_roundtrip.so for the forward roundtrip,
and does the analytic FFN backward + STE (grad_full = grad_through_quant) + Adam in numpy.

The self_cond MLP is a gated FFN (gemma4, GELU-tanh, parallel gate/up):
    g = pre @ gate.T ;  u = pre @ up.T ;  h = gelu(g) * u ;  out = h @ down.T
weights row-major {out,in}; quantized along the last (in) dim per row.

Loss = MSE(student(pre), post)
     + hinge_lambda * relu(||f(pre_t)-f(pre_{t-1})|| - rho*||pre_t-pre_{t-1}||)^2   (mid-phase pairs)

Outputs trained gate/up/down as f32 .npy for the inject->llama-quantize deploy step.
"""
import argparse, ctypes, sys
import numpy as np

# ---- C shim (real codebook roundtrip) -------------------------------------------------
def load_shim(path):
    lib = ctypes.CDLL(path)
    lib.qrt_roundtrip.restype = ctypes.c_int
    lib.qrt_roundtrip.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_long, ctypes.c_long,
                                  ctypes.POINTER(ctypes.c_float)]
    lib.qrt_type_by_name.restype = ctypes.c_int
    lib.qrt_type_by_name.argtypes = [ctypes.c_char_p]
    return lib

def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def roundtrip(lib, type_id, w):
    """f32 [rows, cols] -> real quant(type) -> f32, same shape. cols = n_per_row (quant dim)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    out = np.empty_like(w)
    rc = lib.qrt_roundtrip(type_id, fptr(w), fptr(out), w.shape[0], w.shape[1], None)
    if rc != 0:
        raise RuntimeError(f"qrt_roundtrip failed rc={rc} (2=needs imatrix, 3=oom, 4=no to_float)")
    return out

# ---- data -----------------------------------------------------------------------------
def read_meta(path):
    kv = {}
    for tok in open(path + ".meta").read().split():
        if "=" in tok:
            k, v = tok.split("=", 1); kv[k] = v
    return int(kv["n_embd"]), int(kv["n_tokens"])

def load_trajectory(path):
    E, T = read_meta(path)
    pre  = np.fromfile(path + ".in.bin",  np.float32).reshape(-1, T, E)
    post = np.fromfile(path + ".out.bin", np.float32).reshape(-1, T, E)
    idx  = np.fromfile(path + ".idx",     np.int32).reshape(-1, 2)
    S = min(len(pre), len(post), len(idx))
    return pre[:S], post[:S], idx[:S], E, T

def adjacent_pairs(idx, lo=0, hi=0):
    pairs = []
    for s in range(len(idx) - 1):
        if idx[s,0] == idx[s+1,0] and idx[s,1] - idx[s+1,1] == 1:
            stb = int(idx[s+1,1])
            if lo and stb < lo: continue
            if hi and stb > hi: continue
            pairs.append((s, s+1))
    return pairs

# ---- gated FFN forward + analytic backward -------------------------------------------
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))

def dgelu(x):
    # derivative of the tanh-gelu approximation
    c = np.sqrt(2.0/np.pi)
    inner = c * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    dinner = c * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t*t) * dinner

def forward(pre, gate, up, down):
    g = pre @ gate.T           # [B, F]
    u = pre @ up.T             # [B, F]
    ge = gelu(g)
    h = ge * u                 # [B, F]
    out = h @ down.T           # [B, E]
    cache = (pre, g, u, ge, h)
    return out, cache

def backward(grad_out, cache, gate, up, down):
    pre, g, u, ge, h = cache
    grad_down = grad_out.T @ h            # [E, F]
    grad_h = grad_out @ down              # [B, F]
    grad_ge = grad_h * u
    grad_u = grad_h * ge
    grad_g = grad_ge * dgelu(g)
    grad_up   = grad_u.T @ pre            # [F, E]
    grad_gate = grad_g.T @ pre            # [F, E]
    return grad_gate, grad_up, grad_down

# ---- Adam -----------------------------------------------------------------------------
class Adam:
    def __init__(self, shapes, lr=1e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s, np.float32) for s in shapes]
        self.v = [np.zeros(s, np.float32) for s in shapes]
        self.t = 0
    def step(self, params, grads):
        self.t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*gr
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(gr*gr)
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", required=True, help="path to libquant_roundtrip.so")
    ap.add_argument("--qtype", default="q2_K", help="ggml quant type for gate/up (n_per_row=2816, %256==0)")
    ap.add_argument("--qtype-down", default="q4_0",
                    help="ggml quant type for down (n_per_row=2112, %256!=0 so k-quants produce nan; "
                         "use a 32-block type. Matches the real deploy fallback: down was Q4_0).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--gate", required=True); ap.add_argument("--up", required=True); ap.add_argument("--down", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hinge-lambda", type=float, default=0.0)
    ap.add_argument("--hinge-rho", type=float, default=0.9)
    ap.add_argument("--hinge-step-lo", type=int, default=0)
    ap.add_argument("--hinge-step-hi", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    lib = load_shim(args.shim)
    type_id = lib.qrt_type_by_name(args.qtype.encode())
    type_id_down = lib.qrt_type_by_name(args.qtype_down.encode())
    if type_id < 0 or type_id_down < 0:
        sys.exit(f"unknown ggml type '{args.qtype}' / '{args.qtype_down}'")
    print(f"real-codebook QAT: gate/up={args.qtype} (id={type_id})  down={args.qtype_down} (id={type_id_down})")

    pre, post, idx, E, T = load_trajectory(args.data)
    X = pre.reshape(-1, E); Y = post.reshape(-1, E)
    n = len(X)
    print(f"loaded {n} (pre,post) pairs, n_embd={E}")

    gate = np.load(args.gate).astype(np.float32)
    up   = np.load(args.up).astype(np.float32)
    down = np.load(args.down).astype(np.float32)
    print(f"teacher mlp: gate{gate.shape} up{up.shape} down{down.shape}")

    # hinge pairs (mid-phase consecutive steps) as flat per-token pre arrays
    pa = pb = None
    if args.hinge_lambda > 0:
        pairs = adjacent_pairs(idx, args.hinge_step_lo, args.hinge_step_hi)
        if not pairs:
            sys.exit("hinge-lambda>0 but no pairs in the phase window")
        pa = np.concatenate([pre[a] for a,b in pairs], 0).astype(np.float32)
        pb = np.concatenate([pre[b] for a,b in pairs], 0).astype(np.float32)
        print(f"hinge: {len(pairs)} step-pairs -> {pa.shape[0]} token-pairs, lambda={args.hinge_lambda} rho={args.hinge_rho}")

    def quant_all(g, u, d):
        return roundtrip(lib, type_id, g), roundtrip(lib, type_id, u), roundtrip(lib, type_id_down, d)

    # baseline MSE with teacher weights under the REAL codebook
    gq, uq, dq = quant_all(gate, up, down)
    pred0, _ = forward(X[:args.batch], gq, uq, dq)
    base = float(((pred0 - Y[:args.batch])**2).mean())
    print(f"baseline MSE (teacher weights, real {args.qtype}): {base:.6f}")

    opt = Adam([gate.shape, up.shape, down.shape], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    n_h = pa.shape[0] if pa is not None else 0

    for ep in range(args.epochs):
        perm = rng.permutation(n)
        tot = 0.0; tot_h = 0.0; nb = 0
        for i in range(0, n, args.batch):
            bi = perm[i:i+args.batch]
            xb = X[bi]; yb = Y[bi]
            # STE forward: quantize (real codebook), forward through quantized weights
            gq, uq, dq = quant_all(gate, up, down)
            pred, cache = forward(xb, gq, uq, dq)
            resid = pred - yb
            mse = float((resid**2).mean())
            grad_out = (2.0 / resid.size) * resid
            gg, gu, gd = backward(grad_out, cache, gq, uq, dq)  # grads wrt QUANTIZED weights

            h_val = 0.0
            if args.hinge_lambda > 0:
                hidx = rng.integers(0, n_h, size=min(args.batch, n_h))
                a = pa[hidx]; b = pb[hidx]
                fa, ca = forward(a, gq, uq, dq); fb, cb = forward(b, gq, uq, dq)
                diff = fb - fa                                  # [B, E]
                out_d = np.linalg.norm(diff, axis=1)            # ||f(b)-f(a)||
                in_d  = np.maximum(np.linalg.norm(b - a, axis=1), 1e-6)
                excess = np.maximum(out_d - args.hinge_rho * in_d, 0.0)
                h_val = float((excess**2).mean())
                # d/dfb of mean(excess^2) where excess = relu(||diff|| - rho*in_d), diff=fb-fa
                active = excess > 0
                # grad of ||diff|| wrt diff = diff / ||diff||
                safe = np.maximum(out_d, 1e-6)[:, None]
                gdiff = np.zeros_like(diff)
                coef = (2.0 * excess / len(excess))[:, None] * active[:, None]
                gdiff = coef * (diff / safe)
                # fb gets +gdiff, fa gets -gdiff; backprop both through the FFN
                gg_b, gu_b, gd_b = backward(args.hinge_lambda * gdiff,  cb, gq, uq, dq)
                gg_a, gu_a, gd_a = backward(args.hinge_lambda * -gdiff, ca, gq, uq, dq)
                gg += gg_b + gg_a; gu += gu_b + gu_a; gd += gd_b + gd_a

            # STE: gradient wrt quantized == gradient wrt full (identity passthrough)
            opt.step([gate, up, down], [gg, gu, gd])
            tot += mse; tot_h += h_val; nb += 1
        if args.hinge_lambda > 0:
            print(f"epoch {ep+1}/{args.epochs}  MSE {tot/nb:.6f}  hinge {tot_h/nb:.6f}")
        else:
            print(f"epoch {ep+1}/{args.epochs}  MSE {tot/nb:.6f}")

    np.save(args.out + ".gate.npy", gate.astype(np.float32))
    np.save(args.out + ".up.npy",   up.astype(np.float32))
    np.save(args.out + ".down.npy", down.astype(np.float32))
    print(f"saved trained weights -> {args.out}.{{gate,up,down}}.npy")

if __name__ == "__main__":
    main()
