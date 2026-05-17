#!/usr/bin/env python3
"""Generate PyTorch reference outputs for MUL_MAT_ID backward.

ggml MUL_MAT_ID semantics (ne[0] is INNERMOST/fastest axis):
    as  : ggml shape [D_in, D_out, n_expert]   (ne[0]=D_in)
    b   : ggml shape [D_in, n_used_b, n_tokens]
    ids : ggml shape [n_used, n_tokens] (i32)
    c   : ggml shape [D_out, n_used, n_tokens]

Forward:
    c[i, e, t] = sum_j  as[j, i, ids[e, t]] * b[j, e mod n_used_b, t]

A ggml tensor [D_in, D_out, n_expert] in C-order = numpy (n_expert, D_out, D_in).
We compute the math in that numpy/torch shape, then dump with .tobytes() so the
byte layout matches ggml exactly.

IMPORTANT: dims here are intentionally ASYMMETRIC (D_out=48, D_in=32) so that
any future shape-layout regression cannot be hidden by symmetric dims giving
byte-identical tensors. The original Stage-1 bug was exactly this: symmetric
D=64 made [D_in, D_out, k] and [D_out, D_in, k] indistinguishable in bytes.
"""

import os
import torch

OUT_DIR = '/tmp/mul_mat_id_ref'
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)

D_out    = 48   # asymmetric on purpose
D_in     = 32
n_expert = 8
n_used   = 2
n_tokens = 4
n_used_b = n_used  # no broadcast

# ggml [D_in, D_out, n_expert]  -> torch (n_expert, D_out, D_in)
as_  = torch.randn(n_expert, D_out, D_in, dtype=torch.float32, requires_grad=True)
# ggml [D_in, n_used_b, n_tokens]  -> torch (n_tokens, n_used_b, D_in)
b    = torch.randn(n_tokens, n_used_b, D_in, dtype=torch.float32, requires_grad=True)
# ggml [n_used, n_tokens]  -> torch (n_tokens, n_used)
ids  = torch.randint(0, n_expert, (n_tokens, n_used), dtype=torch.int32)

# Forward in torch-shape:
#   c[t, e, i] = sum_j  as[ids[t,e], i, j] * b[t, e mod n_used_b, j]
# i.e. c[t, e] = as[ids[t,e]] @ b[t, e mod n_used_b]    (shape (D_out,))
c = torch.zeros(n_tokens, n_used, D_out, dtype=torch.float32)
for t in range(n_tokens):
    for e in range(n_used):
        k  = int(ids[t, e])
        eb = e % n_used_b
        c[t, e] = as_[k] @ b[t, eb]

# Loss = sum(c), so grad_c = ones
grad_c = torch.ones_like(c)
c.backward(grad_c)

# as_T: transpose last two dims of `as`, i.e. ggml [D_out, D_in, n_expert]
# In torch (n_expert, D_out, D_in) -> (n_expert, D_in, D_out).
as_T = as_.detach().transpose(-1, -2).contiguous()

def save(name, t):
    path = os.path.join(OUT_DIR, name + '.bin')
    arr = t.detach().contiguous().cpu().numpy()
    with open(path, 'wb') as f:
        f.write(arr.tobytes())
    print(f'  {name:14s} shape={tuple(arr.shape)} dtype={arr.dtype} bytes={arr.nbytes}')

print('Saving to', OUT_DIR)
save('as',      as_)
save('as_T',    as_T)
save('b',       b)
save('ids',     ids.to(torch.int32))
save('grad_c',  grad_c)
save('grad_as', as_.grad)
save('grad_b',  b.grad)

with open(os.path.join(OUT_DIR, 'meta.txt'), 'w') as f:
    f.write(f'D_out={D_out}\n')
    f.write(f'D_in={D_in}\n')
    f.write(f'n_expert={n_expert}\n')
    f.write(f'n_used={n_used}\n')
    f.write(f'n_used_b={n_used_b}\n')
    f.write(f'n_tokens={n_tokens}\n')

print()
print('grad_as sum:', float(as_.grad.sum()))
print('grad_b  sum:', float(b.grad.sum()))
