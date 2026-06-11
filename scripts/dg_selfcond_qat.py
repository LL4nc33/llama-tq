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
    kv = dict(tok.split("=", 1) for tok in open(path + ".meta").read().split())
    return int(kv["n_embd"]), int(kv["n_tokens"])


def load_pairs(path):
    E, T = read_meta(path)
    x = np.fromfile(path + ".in.bin",  np.float32).reshape(-1, E)   # [N*T, E] flattened steps*tokens
    y = np.fromfile(path + ".out.bin", np.float32).reshape(-1, E)
    n = min(len(x), len(y))
    return x[:n], y[:n], E


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
    args = ap.parse_args()

    x, y, E = load_pairs(args.data)
    print(f"loaded {len(x)} (x_in,y_target) pairs, n_embd={E}")
    gate = np.load(args.gate); up = np.load(args.up); down = np.load(args.down)
    print(f"teacher mlp: gate{gate.shape} up{up.shape} down{down.shape}")

    dev = torch.device(args.device)
    X = torch.tensor(x, device=dev); Y = torch.tensor(y, device=dev)
    model = SelfCondMLP(gate, up, down, fake_quant=args.fake_quant, n_bits=args.n_bits).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # baseline loss with teacher weights (should be ~0 without fake-quant; >0 with)
    with torch.no_grad():
        base = F.mse_loss(model(X[:args.batch]), Y[:args.batch]).item()
    print(f"baseline MSE (teacher weights, fake_quant={args.fake_quant}): {base:.6f}")

    n = len(X)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0; nb = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.mse_loss(model(X[idx]), Y[idx])
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
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
