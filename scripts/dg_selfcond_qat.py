#!/usr/bin/env python3
"""
Isolated self_cond_mlp QAT distillation for DiffusionGemma 2-bit.

Trains ONLY the 3 self_cond tensors (gate/up/down) to compensate the 2-bit weight
noise in the self-conditioning feedback path, using the f16 teacher's real
decoder-forward activations:

    loss = MSE( student_mlp(x_in), y_target )

where (x_in, y_target) are the self_cond_mlp input/output pairs dumped from the
coherent teacher (DG_DUMP_SELFCOND). The student starts from the BF16 self_cond
weights and is trained with optional fake-quant (straight-through estimator) so
the learned weights survive re-quantization to the deploy bit-width.

The self_cond MLP is a gated FFN (gemma4 style, GELU, parallel gate/up):
    h = gelu(x @ gate.T) * (x @ up.T)      # gate/up: {n_ff_sc, n_embd}
    y = h @ down.T                         # down:   {n_embd, n_ff_sc}
(weights stored row-major {out, in}; we apply x @ W.T)

Outputs the trained gate/up/down as raw f32 .npy for the GGUF merge step.
"""
import argparse, struct, sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("need torch: pip install torch")


def read_meta(path):
    # tolerate the trailing "idx=int32(block,cur_step)" token added for the hinge dump
    kv = {}
    for tok in open(path + ".meta").read().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return int(kv["n_embd"]), int(kv["n_tokens"])


def load_pairs(path):
    E, T = read_meta(path)
    x = np.fromfile(path + ".in.bin",  np.float32).reshape(-1, E)   # [N*T, E] flattened steps*tokens
    y = np.fromfile(path + ".out.bin", np.float32).reshape(-1, E)
    n = min(len(x), len(y))
    return x[:n], y[:n], E


def load_trajectory(path):
    """Load the step-structured trajectory: per collected denoise step a [T, E] block of
    (pre, post), plus its (block, cur_step) index. Returns (pre, post, idx, E, T) where
    pre/post are [S, T, E] and idx is [S, 2] = (block, cur_step). Used for the contraction
    hinge, which needs consecutive-step adjacency (same block, |Δcur_step| == 1)."""
    E, T = read_meta(path)
    pre  = np.fromfile(path + ".in.bin",  np.float32).reshape(-1, T, E)
    post = np.fromfile(path + ".out.bin", np.float32).reshape(-1, T, E)
    idx  = np.fromfile(path + ".idx",     np.int32).reshape(-1, 2)
    S = min(len(pre), len(post), len(idx))
    return pre[:S], post[:S], idx[:S], E, T


def adjacent_step_pairs(idx):
    """Indices (a, b) into the step axis where step b immediately follows step a within the
    SAME block. cur_step counts DOWN n_steps..1, so the next step has cur_step - 1. Pairing
    across a block boundary is invalid (self-cond resets to zero each block)."""
    pairs = []
    for s in range(len(idx) - 1):
        blk_a, st_a = int(idx[s, 0]),   int(idx[s, 1])
        blk_b, st_b = int(idx[s + 1, 0]), int(idx[s + 1, 1])
        if blk_a == blk_b and st_a - st_b == 1:   # consecutive within the same block
            pairs.append((s, s + 1))
    return pairs


def empirical_rho(pre, idx):
    """Diagnostic: empirical step-contraction factor rho_t = ||pre_{t+1}-pre_t|| /
    ||pre_t-pre_{t-1}|| along the real trajectory (per token, averaged). rho < 1 over the
    run => the loop is contractive (coherent). rho >= 1 => loop-gain divergence. This is the
    cheapest honest test of the Lipschitz>1 hypothesis — no quantize, no decoder needed."""
    pairs = adjacent_step_pairs(idx)
    if len(pairs) < 2:
        return None
    # group consecutive pair-of-pairs (a,b),(b,c) to form ||c-b||/||b-a||
    by_b = {b: a for (a, b) in pairs}
    ratios = []
    for (b, c) in pairs:
        if b in by_b:
            a = by_b[b]
            d_ab = np.linalg.norm((pre[b] - pre[a]).reshape(-1))
            d_bc = np.linalg.norm((pre[c] - pre[b]).reshape(-1))
            if d_ab > 1e-6:
                ratios.append(d_bc / d_ab)
    if not ratios:
        return None
    return float(np.mean(ratios)), float(np.median(ratios)), len(ratios)


def fake_quant_iq2_like(w, n_bits=2):
    """Crude symmetric per-row fake-quant (STE) to ~n_bits, to keep the trained
    weights robust to real IQ2 re-quant. Not the exact IQ2_XXS codebook, but a
    reasonable proxy: per-output-row absmax scaling to a (2^n_bits)-level grid."""
    levels = (1 << n_bits) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / (levels / 2)
    q = torch.clamp(torch.round(w / scale), -(levels // 2 + 1), levels // 2) * scale
    return w + (q - w).detach()   # STE: forward q, backward identity


class SelfCondMLP(torch.nn.Module):
    def __init__(self, gate, up, down, fake_quant=False, n_bits=2):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.tensor(gate, dtype=torch.float32))  # {n_ff, n_embd}
        self.up   = torch.nn.Parameter(torch.tensor(up,   dtype=torch.float32))  # {n_ff, n_embd}
        self.down = torch.nn.Parameter(torch.tensor(down, dtype=torch.float32))  # {n_embd, n_ff}
        self.fake_quant = fake_quant
        self.n_bits = n_bits

    def w(self, p):
        return fake_quant_iq2_like(p, self.n_bits) if self.fake_quant else p

    def forward(self, x):  # x: [B, n_embd]
        g = F.linear(x, self.w(self.gate))   # [B, n_ff]
        u = F.linear(x, self.w(self.up))     # [B, n_ff]
        h = F.gelu(g) * u
        return F.linear(h, self.w(self.down))  # [B, n_embd]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DG_DUMP_SELFCOND base path (.in/.out/.meta)")
    ap.add_argument("--gate", required=True, help=".npy of teacher self_cond_gate {n_ff,n_embd}")
    ap.add_argument("--up",   required=True, help=".npy of teacher self_cond_up   {n_ff,n_embd}")
    ap.add_argument("--down", required=True, help=".npy of teacher self_cond_down {n_embd,n_ff}")
    ap.add_argument("--out",  required=True, help="output prefix; writes <out>.{gate,up,down}.npy")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--fake-quant", action="store_true", help="STE fake-quant during training")
    ap.add_argument("--n-bits", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # contraction-hinge multi-step terms (loop-gain fix). lambda 0 => pure single-step distill.
    ap.add_argument("--hinge-lambda", type=float, default=0.0,
                    help="weight of the contraction hinge relu(||f(pre_t)-f(pre_{t-1})|| - rho*||pre_t-pre_{t-1}||)^2")
    ap.add_argument("--hinge-rho", type=float, default=0.9,
                    help="target contraction factor (<1). The map may expand up to rho before being penalized")
    ap.add_argument("--rho-diag", action="store_true",
                    help="print the empirical step-contraction rho of the trajectory and exit (no training)")
    args = ap.parse_args()

    # rho diagnostic: cheapest honest test of the Lipschitz>1 hypothesis. No training, no quantize.
    if args.rho_diag:
        pre, post, idx, E, T = load_trajectory(args.data)
        r = empirical_rho(pre, idx)
        if r is None:
            print("rho-diag: not enough consecutive-step pairs (need stride-1 dump with .idx)")
        else:
            mean, med, n = r
            print(f"rho-diag: empirical step-contraction over {n} consecutive triples: "
                  f"mean={mean:.4f} median={med:.4f}  ({'CONTRACTIVE <1' if mean < 1 else 'EXPANDING >=1 (loop-gain)'})")
        return

    x, y, E = load_pairs(args.data)
    print(f"loaded {len(x)} (x_in,y_target) pairs, n_embd={E}")
    gate = np.load(args.gate); up = np.load(args.up); down = np.load(args.down)
    print(f"teacher mlp: gate{gate.shape} up{up.shape} down{down.shape}")

    dev = torch.device(args.device)
    X = torch.tensor(x, device=dev); Y = torch.tensor(y, device=dev)
    model = SelfCondMLP(gate, up, down, fake_quant=args.fake_quant, n_bits=args.n_bits).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # contraction-hinge data: consecutive-step (pre_{t-1}, pre_t) pairs per token, formed only
    # within a block (never across the self-cond reset). Built once; flattened to [P, E] pairs.
    pa = pb = None
    if args.hinge_lambda > 0:
        pre, _post, idx, _E, T = load_trajectory(args.data)
        pairs = adjacent_step_pairs(idx)
        if not pairs:
            sys.exit("--hinge-lambda > 0 but no consecutive-step pairs found (need stride-1 .idx dump)")
        a_blocks = np.concatenate([pre[a] for (a, b) in pairs], axis=0)  # [P, E] = pre_{t-1} per token
        b_blocks = np.concatenate([pre[b] for (a, b) in pairs], axis=0)  # [P, E] = pre_t     per token
        pa = torch.tensor(a_blocks, device=dev)
        pb = torch.tensor(b_blocks, device=dev)
        print(f"hinge: {len(pairs)} consecutive step-pairs -> {pa.shape[0]} token-pairs, "
              f"lambda={args.hinge_lambda} rho={args.hinge_rho}")

    def hinge_term(idx_h):
        # relu(||f(pre_b)-f(pre_a)|| - rho*||pre_b-pre_a||)^2, averaged over the sampled pairs.
        a = pa[idx_h]; b = pb[idx_h]
        fa = model(a); fb = model(b)
        out_d = torch.linalg.vector_norm(fb - fa, dim=1)        # ||f(pre_b)-f(pre_a)||
        in_d  = torch.linalg.vector_norm(b - a, dim=1).clamp_min(1e-6)
        excess = F.relu(out_d - args.hinge_rho * in_d)
        return (excess * excess).mean()

    # baseline loss with teacher weights (should be ~0 without fake-quant; >0 with)
    with torch.no_grad():
        base = F.mse_loss(model(X[:args.batch]), Y[:args.batch]).item()
    print(f"baseline MSE (teacher weights, fake_quant={args.fake_quant}): {base:.6f}")

    n = len(X)
    n_h = pa.shape[0] if pa is not None else 0
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0; tot_h = 0.0; nb = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            mse = F.mse_loss(model(X[idx]), Y[idx])
            loss = mse
            h_val = 0.0
            if args.hinge_lambda > 0:
                # sample a hinge minibatch of the same size from the pair pool
                hidx = torch.randint(0, n_h, (min(args.batch, n_h),), device=dev)
                h = hinge_term(hidx)
                loss = loss + args.hinge_lambda * h
                h_val = h.item()
            loss.backward(); opt.step()
            tot += mse.item(); tot_h += h_val; nb += 1
        if args.hinge_lambda > 0:
            print(f"epoch {ep+1}/{args.epochs}  MSE {tot/nb:.6f}  hinge {tot_h/nb:.6f}")
        else:
            print(f"epoch {ep+1}/{args.epochs}  MSE {tot/nb:.6f}")

    g = model.gate.detach().cpu().numpy().astype(np.float32)
    u = model.up.detach().cpu().numpy().astype(np.float32)
    d = model.down.detach().cpu().numpy().astype(np.float32)
    np.save(args.out + ".gate.npy", g)
    np.save(args.out + ".up.npy", u)
    np.save(args.out + ".down.npy", d)
    print(f"saved trained weights -> {args.out}.{{gate,up,down}}.npy")


if __name__ == "__main__":
    main()
