#include "gated_delta_net.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int           K,
                                     const int32_t * s_ids,
                                     int64_t       s_row_stride,
                                     const float * raw_dt,
                                     const float * raw_a) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    // input state layout (D, K, n_seqs) — seq stride is K * D = K * H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // with a fused gather, the live state is the cache row s_ids[sequence] (K == 1 there)
    const int64_t state_in_offset      = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * K * H * S_v * S_v)
                                         + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    const int64_t state_size_per_token = S_v * S_v * H * n_seqs; // per-slot stride in output
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // slot mapping: target_slot = t - shift. When n_tokens < K only the last n_tokens slots
    // are written; earlier slots are left untouched (caller-owned).
    const int shift = (int) n_tokens - K;

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = *beta_t;
        if (raw_a) {
            // gate activations folded in: the same formulas as the sigmoid and softplus kernels
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            float g_raw = *g_t;
            if (raw_a) {
                const float x = g_raw + raw_dt[h_idx];
                g_raw = ((x > 20.0f) ? x : logf(1.0f + expf(x))) * raw_a[h_idx];
            }
            const float g_val = expf(g_raw);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = t - shift;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = (dst + attn_score_elems) + target_slot * state_size_per_token + state_out_offset;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, const int32_t * s_ids, int64_t s_row_stride,
        const float * raw_dt, const float * raw_a, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA, keep_rs_t><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K, s_ids, s_row_stride, raw_dt, raw_a);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA, keep_rs_t><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K, s_ids, s_row_stride, raw_dt, raw_a);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA, keep_rs_t><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K, s_ids, s_row_stride, raw_dt, raw_a);
            break;
        }
        case 128: {
            gated_delta_net_cuda<128, KDA, keep_rs_t><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K, s_ids, s_row_stride, raw_dt, raw_a);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    // gate activations folded in (ggml_gated_delta_net_set_raw_gates)
    const bool    raw    = ggml_get_op_params_i32(dst, 1) != 0;
    const float * raw_dt = raw ? (const float *) dst->src[7]->data : nullptr;
    const float * raw_a  = raw ? (const float *) dst->src[8]->data : nullptr;
    GGML_ASSERT(!raw || !kda);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // fused state gather, registered for this node by the graph evaluator (ggml_cuda_try_gdn_gather_skip)
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    if (const ggml_cuda_gated_delta_net_gather * gather = ctx.gdn_gathers().find(dst)) {
        s_d          = gather->base;
        s_ids        = gather->ids;
        s_row_stride = gather->row_stride;
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // state is 3D (S_v*S_v*H, K, n_seqs); K is the snapshot slot count.
    const int K = (int) src_state->ne[1];
    const bool keep_rs = K > 1;

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, s_ids, s_row_stride, raw_dt, raw_a, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, s_ids, s_row_stride, raw_dt, raw_a, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, s_ids, s_row_stride, raw_dt, raw_a, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, K, s_ids, s_row_stride, raw_dt, raw_a, stream);
        }
    }
}

// ====== backward pass (GGML_OP_GATED_DELTA_NET_BACK), see ggml_compute_forward_gated_delta_net_back ======
//
// As in the forward kernel each warp owns one column j of S (row j of M = S^T), the lanes hold its elements i.
// A column evolves independently of the others, so each warp recomputes its own history: first a forward pass
// that stores the column at the start of every segment, then per segment (in reverse) the column after each
// token, both in global scratch. Only dq, dk, dg (and dbeta) sum over the columns: per token the warps of a block
// add them up in shared memory and add the block's sum to the zero-initialised outputs atomically.


template <int S_v, bool KDA>
__global__ void gated_delta_net_back_cuda(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * state_in, const float * grad, float * dst, float * ckpt, float * seg_states,
        int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t seg, int64_t n_seg,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic, float scale) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = S_v / warp_size;

    const uint32_t h        = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;
    const bool     active   = col < S_v;

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    __shared__ float sh_dq[S_v];
    __shared__ float sh_dk[S_v];
    __shared__ float sh_da[S_v];
    __shared__ float sh_db;

    const int64_t SS    = (int64_t) S_v * S_v;
    const int64_t n_v   = (int64_t) S_v * H * n_tokens * n_seqs;
    float * out_dq = dst;
    float * out_dk = out_dq + n_v;
    float * out_dv = out_dk + n_v;
    float * out_dg = out_dv + n_v;
    float * out_db = out_dg + n_v / S_v * (KDA ? S_v : 1);

    const int64_t head    = (int64_t) sequence * H + h;
    float * my_ckpt = ckpt       + head * n_seg       * SS + (int64_t) col * S_v;
    float * my_seg  = seg_states + head * seg * SS + (int64_t) col * S_v;

    float m[rows_per_lane];  // the running column / the column after token t
    float dm[rows_per_lane]; // gradient of the column
    float p[rows_per_lane];  // the column before token t
    float a[rows_per_lane];
    float kr[rows_per_lane];

    auto load_gate = [&](int64_t t) {
        const int64_t gb = sequence * sb3 + t * sb2 + h * sb1;
        if constexpr (KDA) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                a[r] = expf(g[gb * S_v + r * warp_size + lane]);
            }
        } else {
            const float av = expf(g[gb]);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                a[r] = av;
            }
        }
    };
    // one forward step of this column: m := a*m + k*d with d = beta*(v_col - dot(a*m, k))
    auto step = [&](int64_t t) {
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float   b   = beta[sequence * sb3 + t * sb2 + h * sb1];
        load_gate(t);
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kr[r] = k_t[r * warp_size + lane];
            m[r] *= a[r];
            s += m[r] * kr[r];
        }
        s = warp_reduce_sum<warp_size>(s);
        const float d = b * (v[sequence * sv3 + t * sv2 + h * sv1 + col] - s);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            m[r] += kr[r] * d;
        }
    };

    if (active) {
        const float * s0 = state_in + sequence * H * SS + h * SS + (int64_t) col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            m[r] = s0[r * warp_size + lane];
        }
        for (int64_t t = 0; t < n_tokens; t++) {
            if (t % seg == 0) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    my_ckpt[(t / seg) * SS + r * warp_size + lane] = m[r];
                }
            }
            step(t);
        }
        const float * gs = grad + n_v + head * SS + (int64_t) col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            dm[r] = gs[r * warp_size + lane];
        }
    }

    for (int64_t is = n_seg - 1; is >= 0; is--) {
        const int64_t t0 = is * seg;
        const int64_t t1 = min(t0 + seg, n_tokens);

        if (active) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                m[r] = my_ckpt[is * SS + r * warp_size + lane];
            }
            for (int64_t t = t0; t < t1; t++) {
                step(t);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    my_seg[(t - t0) * SS + r * warp_size + lane] = m[r];
                }
            }
        }

        for (int64_t t = t1 - 1; t >= t0; t--) {
            for (int i = threadIdx.y * warp_size + lane; i < S_v; i += blockDim.y * warp_size) {
                sh_dq[i] = 0.0f;
                sh_dk[i] = 0.0f;
                sh_da[i] = 0.0f;
            }
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                sh_db = 0.0f;
            }
            __syncthreads();

            if (active) {
                const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
                const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
                const int64_t gb  = sequence * sb3 + t * sb2 + h * sb1;
                const float   b   = beta[gb];
                const float   vj  = v[sequence * sv3 + t * sv2 + h * sv1 + col];
                const int64_t io  = (((int64_t) sequence * n_tokens + t) * H + h) * S_v;
                const float   doj = grad[io + col];
                load_gate(t);

                float u = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    m[r]  = my_seg[(t - t0) * SS + i];
                    p[r]  = t > t0 ? my_seg[(t - t0 - 1) * SS + i] : my_ckpt[is * SS + i];
                    kr[r] = k_t[i];
                    u += a[r] * p[r] * kr[r];
                }
                u = warp_reduce_sum<warp_size>(u);
                const float d  = b * (vj - u);
                const float cj = scale * doj;

                // output: dM += c*do_j*q, dq += c*do_j*M
                float dd = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    atomicAdd(&sh_dq[i], cj * m[r]);
                    dm[r] += cj * q_t[i];
                    dd += dm[r] * kr[r];
                }
                dd = warp_reduce_sum<warp_size>(dd);
                if (lane == 0) {
                    out_dv[io + col] = b * dd;
                    atomicAdd(&sh_db, dd * (vj - u));
                }
                const float du = -b * dd;

                float da_sum = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    atomicAdd(&sh_dk[i], d * dm[r] + du * a[r] * p[r]);
                    dm[r] += du * kr[r];
                    const float dai = dm[r] * p[r];
                    if constexpr (KDA) {
                        atomicAdd(&sh_da[i], dai);
                    } else {
                        da_sum += dai;
                    }
                    dm[r] *= a[r];
                }
                if constexpr (!KDA) {
                    da_sum = warp_reduce_sum<warp_size>(da_sum);
                    if (lane == 0) {
                        atomicAdd(&sh_da[0], da_sum);
                    }
                }
            }
            __syncthreads();

            const int64_t io = (((int64_t) sequence * n_tokens + t) * H + h) * S_v;
            const int64_t gb = sequence * sb3 + t * sb2 + h * sb1;
            for (int i = threadIdx.y * warp_size + lane; i < S_v; i += blockDim.y * warp_size) {
                atomicAdd(&out_dq[io + i], sh_dq[i]);
                atomicAdd(&out_dk[io + i], sh_dk[i]);
                if constexpr (KDA) {
                    atomicAdd(&out_dg[io + i], sh_da[i] * expf(g[gb * S_v + i]));
                }
            }
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                if constexpr (!KDA) {
                    atomicAdd(&out_dg[(((int64_t) sequence * n_tokens + t) * H + h)], sh_da[0] * expf(g[gb]));
                }
                atomicAdd(&out_db[(((int64_t) sequence * n_tokens + t) * H + h)], sh_db);
            }
            __syncthreads();
        }
    }
}

template <bool KDA>
static void launch_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs, float * ckpt, float * seg_states, int64_t seg, int64_t n_seg,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    const dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    const dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const float * q    = (const float *) dst->src[0]->data;
    const float * k    = (const float *) dst->src[1]->data;
    const float * v    = (const float *) dst->src[2]->data;
    const float * g    = (const float *) dst->src[3]->data;
    const float * b    = (const float *) dst->src[4]->data;
    const float * s    = (const float *) dst->src[5]->data;
    const float * grad = (const float *) dst->src[6]->data;
    float *       out  = (float *) dst->data;

#define GDN_BACK_LAUNCH(SV) \
    gated_delta_net_back_cuda<SV, KDA><<<grid_dims, block_dims, 0, ctx.stream()>>>(q, k, v, g, b, s, grad, out, ckpt, seg_states, \
        H, n_tokens, n_seqs, seg, n_seg, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale)
    switch (S_v) {
        case 16:  GDN_BACK_LAUNCH(16);  break;
        case 32:  GDN_BACK_LAUNCH(32);  break;
        case 64:  GDN_BACK_LAUNCH(64);  break;
        case 128: GDN_BACK_LAUNCH(128); break;
        default:  GGML_ABORT("unsupported head size %d", (int) S_v);
    }
#undef GDN_BACK_LAUNCH
}

void ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_k    = dst->src[1];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_g    = dst->src[3];
    const ggml_tensor * src_beta = dst->src[4];

    GGML_ASSERT(ggml_is_contiguous_rows(src_q) && ggml_is_contiguous_rows(src_k) && ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k) && src_q->ne[1] == src_k->ne[1]);
    GGML_ASSERT(ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(dst->src[5]) && ggml_is_contiguous(dst->src[6]));

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t seg      = ggml_get_op_params_i32(dst, 0); // segment length, chosen by ggml_gated_delta_net_back
    const int64_t n_seg    = (n_tokens + seg - 1) / seg;
    const bool    kda      = src_g->ne[0] == S_v;

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    ggml_cuda_pool_alloc<float> ckpt(ctx.pool(), n_seqs * H * n_seg        * S_v * S_v);
    ggml_cuda_pool_alloc<float> seg_states(ctx.pool(), n_seqs * H * seg * S_v * S_v);

    const int64_t fs = sizeof(float);
    if (kda) {
        launch_gated_delta_net_back<true>(ctx, dst, S_v, H, n_tokens, n_seqs, ckpt.get(), seg_states.get(), seg, n_seg,
            src_q->nb[1]/fs, src_q->nb[2]/fs, src_q->nb[3]/fs, src_v->nb[1]/fs, src_v->nb[2]/fs, src_v->nb[3]/fs,
            src_beta->nb[1]/fs, src_beta->nb[2]/fs, src_beta->nb[3]/fs, src_q->ne[1], n_seqs / src_q->ne[3],
            1.0f / sqrtf((float) S_v));
    } else {
        launch_gated_delta_net_back<false>(ctx, dst, S_v, H, n_tokens, n_seqs, ckpt.get(), seg_states.get(), seg, n_seg,
            src_q->nb[1]/fs, src_q->nb[2]/fs, src_q->nb[3]/fs, src_v->nb[1]/fs, src_v->nb[2]/fs, src_v->nb[3]/fs,
            src_beta->nb[1]/fs, src_beta->nb[2]/fs, src_beta->nb[3]/fs, src_q->ne[1], n_seqs / src_q->ne[3],
            1.0f / sqrtf((float) S_v));
    }
}
