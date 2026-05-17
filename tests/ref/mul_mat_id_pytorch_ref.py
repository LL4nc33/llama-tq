#!/usr/bin/env python3
"""Generate PyTorch reference outputs for MUL_MAT_ID backward.

ggml MUL_MAT_ID semantics:
    as  : [D_out, D_in, n_expert]    (ggml shape, ne[0]=D_out fastest)
    b   : [D_in,  n_used_b, n_tokens]
    ids : [n_used, n_tokens]  (i32)
    c   : [D_out, n_used, n_tokens]

    c[i, e, t] = sum_j  as[i, j, ids[e, t]] * b[j, e mod n_used_b, t]

ggml stores tensors with ne[0] as the fastest-changing axis. A ggml tensor with
shape [D_out, D_in, n_expert] has C-layout:
    data[i + j*D_out + k*D_out*D_in]  for elem (i, j, k)
This matches a numpy/pytorch tensor shaped (n_expert, D_in, D_out) in C-order,
indexed [k, j, i].

So we compute the math in that PyTorch shape, then dump flat with .tobytes()
which gives the exact byte layout ggml expects.
"""

import os
import torch

OUT_DIR = '/tmp/mul_mat_id_ref'
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)

D_out    = 64
D_in     = 64
n_expert = 8
n_used   = 2
n_tokens = 4
n_used_b = n_used  # no broadcast

# Tensors in (outermost, ..., innermost) shape so that .contiguous().tobytes()
# matches ggml's [innermost, ..., outermost] layout.
# as:  ggml [D_out, D_in, n_expert]  -> torch (n_expert, D_in, D_out)
# b:   ggml [D_in,  n_used_b, n_tokens] -> torch (n_tokens, n_used_b, D_in)
# ids: ggml [n_used, n_tokens] -> torch (n_tokens, n_used)
# c:   ggml [D_out, n_used, n_tokens] -> torch (n_tokens, n_used, D_out)
as_   = torch.randn(n_expert, D_in, D_out, dtype=torch.float32, requires_grad=True)
b     = torch.randn(n_tokens, n_used_b, D_in, dtype=torch.float32, requires_grad=True)
ids   = torch.randint(0, n_expert, (n_tokens, n_used), dtype=torch.int32)

# Forward: c[t, e, i] = sum_j as[ids[t,e], j, i] * b[t, e mod n_used_b, j]
c = torch.zeros(n_tokens, n_used, D_out, dtype=torch.float32)
for t in range(n_tokens):
    for e in range(n_used):
        k = int(ids[t, e])
        eb = e % n_used_b
        # as[k] has shape (D_in, D_out), b[t, eb] has shape (D_in,)
        # result is (D_out,) = b @ as[k]
        c[t, e] = b[t, eb] @ as_[k]

# Loss = sum(c), so grad_c = ones
grad_c = torch.ones_like(c)
c.backward(grad_c)

# Dump in ggml byte order: contiguous().tobytes() of the torch tensor
# (which is already in the right shape).
def save(name, t):
    path = os.path.join(OUT_DIR, name + '.bin')
    arr = t.detach().contiguous().cpu().numpy()
    with open(path, 'wb') as f:
        f.write(arr.tobytes())
    print(f'  {name:14s} shape={tuple(arr.shape)} dtype={arr.dtype} bytes={arr.nbytes}')

print('Saving to', OUT_DIR)
save('as',      as_)
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
