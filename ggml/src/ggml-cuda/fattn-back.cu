#include "fattn-back.cuh"

// Flash attention backward (GGML_OP_FLASH_ATTN_BACK), the math of the CPU reference in ggml-cpu/ops.cpp:
//   p_ij = exp(s_ij - L_i), D_i = dO_i . O_i, ds_ij = p_ij (dO_i . v_j - D_i) ds/dqk
//   dQ_i = sum_j ds_ij k_j, dK_j = sum_i ds_ij q_i, dV_j = sum_i p_ij dO_i, dSink_h = -sum_i p_i,sink D_i
// Two passes without atomics on dK/dV: rows_kernel (one warp per query row) computes L, D and dQ;
// keys_kernel (one warp per key row) computes dK and dV from them.

#define FATTN_BACK_MAX_CHUNKS 8 // head sizes up to 8*WARP_SIZE = 256
#define FATTN_BACK_WARPS      4

struct fattn_back_params {
    int64_t DK, DV, N, KV, HQ, HK, SEQ, rk2, rk3;
    // strides in elements
    int64_t q1, q2, q3, k1, k2, k3, v1, v2, v3, o1, o2, o3, m1, m2, m3;
    int64_t m_ne2, m_ne3;
    float scale_qk, scale, softcap, max_bias, m0, m1f;
    uint32_t n_head_log2;
};

template <typename T> static __device__ __forceinline__ float fb_load(const T * p, int64_t i);
template <> __device__ __forceinline__ float fb_load<float>(const float * p, int64_t i) { return p[i]; }
template <> __device__ __forceinline__ float fb_load<half>(const half * p, int64_t i) { return __half2float(p[i]); }

static __device__ __forceinline__ float fb_slope(const fattn_back_params & P, int64_t h) {
    if (P.max_bias <= 0.0f) {
        return 1.0f;
    }
    return h < P.n_head_log2 ? powf(P.m0, h + 1) : powf(P.m1f, 2*(h - P.n_head_log2) + 1);
}

// score of query row qr against key row kr (lane-strided), returns s and tanh for the softcap derivative
template <typename TK>
static __device__ __forceinline__ float fb_score(const fattn_back_params & P, const float * qreg, const TK * kr, float mv, float & t) {
    const int lane = threadIdx.x;
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        if (e < P.DK) {
            dot += qreg[c]*fb_load(kr, e);
        }
    }
    dot = warp_reduce_sum(dot)*P.scale_qk;
    if (P.softcap != 0.0f) {
        t = tanhf(dot);
        dot = P.softcap*t;
    }
    return dot + mv;
}

template <typename TK, typename TV>
static __global__ void fattn_back_rows_kernel(const fattn_back_params P,
        const float * q, const TK * k, const TV * v, const half * mask, const float * sinks,
        const float * o, const float * d, float * dq, float * dsinks, float * Lbuf, float * Dbuf) {
    const int64_t r = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= P.N*P.HQ*P.SEQ) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t iq1 = r % P.N;
    const int64_t iq2 = (r / P.N) % P.HQ;
    const int64_t iq3 = r / (P.N*P.HQ);
    const int64_t ik2 = iq2 / P.rk2;
    const int64_t ik3 = iq3 / P.rk3;

    const float * qr = q + iq1*P.q1 + iq2*P.q2 + iq3*P.q3;
    const float * dr = d + iq2*P.o1 + iq1*P.o2 + iq3*P.o3; // d and o have the layout of the forward output
    const float * orow = o + iq2*P.o1 + iq1*P.o2 + iq3*P.o3;
    const half  * mr = mask ? mask + iq1*P.m1 + (iq2 % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 : nullptr;
    const float slope = fb_slope(P, iq2);
    const float sink  = sinks ? sinks[iq2] : -INFINITY;

    float qreg[FATTN_BACK_MAX_CHUNKS], dreg[FATTN_BACK_MAX_CHUNKS], dqreg[FATTN_BACK_MAX_CHUNKS];
    float Dp = 0.0f;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        qreg[c]  = e < P.DK ? qr[e] : 0.0f;
        dreg[c]  = e < P.DV ? dr[e] : 0.0f;
        dqreg[c] = 0.0f;
        Dp += e < P.DV ? dreg[c]*orow[e] : 0.0f;
    }
    const float D = warp_reduce_sum(Dp);

    // logsumexp of the scores (online)
    float M = sink, S = sinks ? 1.0f : 0.0f;
    for (int64_t j = 0; j < P.KV; ++j) {
        const float mv = mr ? slope*__half2float(mr[j]) : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }
        float t;
        const float s = fb_score(P, qreg, k + j*P.k1 + ik2*P.k2 + ik3*P.k3, mv, t);
        if (s > M) {
            S = S*expf(M - s) + 1.0f;
            M = s;
        } else {
            S += expf(s - M);
        }
    }
    const float L = M == -INFINITY ? -INFINITY : M + logf(S);

    if (L != -INFINITY) {
        for (int64_t j = 0; j < P.KV; ++j) {
            const float mv = mr ? slope*__half2float(mr[j]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }
            const TK * kr = k + j*P.k1 + ik2*P.k2 + ik3*P.k3;
            const TV * vr = v + j*P.v1 + ik2*P.v2 + ik3*P.v3;
            float t = 0.0f;
            const float p = expf(fb_score(P, qreg, kr, mv, t) - L);
            float dp = 0.0f;
#pragma unroll
            for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                const int64_t e = lane + c*WARP_SIZE;
                if (e < P.DV) {
                    dp += dreg[c]*fb_load(vr, e);
                }
            }
            dp = warp_reduce_sum(dp);
            float g = p*(dp - D)*P.scale;
            if (P.softcap != 0.0f) {
                g *= 1.0f - t*t;
            }
#pragma unroll
            for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                const int64_t e = lane + c*WARP_SIZE;
                if (e < P.DK) {
                    dqreg[c] += g*fb_load(kr, e);
                }
            }
        }
    }

    float * dqr = dq + ((iq3*P.HQ + iq2)*P.N + iq1)*P.DK;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        if (e < P.DK) {
            dqr[e] = dqreg[c];
        }
    }
    if (lane == 0) {
        Lbuf[r] = L;
        Dbuf[r] = D;
        if (sinks && L != -INFINITY) {
            atomicAdd(dsinks + iq2, -expf(sink - L)*D);
        }
    }
}

template <typename TK, typename TV>
static __global__ void fattn_back_keys_kernel(const fattn_back_params P,
        const float * q, const TK * k, const TV * v, const half * mask,
        const float * d, const float * Lbuf, const float * Dbuf, float * dk, float * dv) {
    const int64_t r = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= P.KV*P.HK*P.SEQ / P.rk3) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t j   = r % P.KV;
    const int64_t ik2 = (r / P.KV) % P.HK;
    const int64_t ik3 = r / (P.KV*P.HK);

    const TK * kr = k + j*P.k1 + ik2*P.k2 + ik3*P.k3;
    const TV * vr = v + j*P.v1 + ik2*P.v2 + ik3*P.v3;
    float vreg[FATTN_BACK_MAX_CHUNKS], dkreg[FATTN_BACK_MAX_CHUNKS], dvreg[FATTN_BACK_MAX_CHUNKS];
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        vreg[c]  = e < P.DV ? fb_load(vr, e) : 0.0f;
        dkreg[c] = 0.0f;
        dvreg[c] = 0.0f;
    }

    for (int64_t iq3 = ik3*P.rk3; iq3 < (ik3 + 1)*P.rk3; ++iq3) {
        for (int64_t iq2 = ik2*P.rk2; iq2 < (ik2 + 1)*P.rk2; ++iq2) {
            const float slope = fb_slope(P, iq2);
            for (int64_t iq1 = 0; iq1 < P.N; ++iq1) {
                const int64_t row = (iq3*P.HQ + iq2)*P.N + iq1;
                const float L = Lbuf[row];
                if (L == -INFINITY) {
                    continue;
                }
                const float mv = mask ? slope*__half2float(mask[iq1*P.m1 + (iq2 % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 + j]) : 0.0f;
                if (mv == -INFINITY) {
                    continue;
                }
                const float * qr = q + iq1*P.q1 + iq2*P.q2 + iq3*P.q3;
                const float * dr = d + iq2*P.o1 + iq1*P.o2 + iq3*P.o3;
                float qreg[FATTN_BACK_MAX_CHUNKS];
#pragma unroll
                for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                    const int64_t e = lane + c*WARP_SIZE;
                    qreg[c] = e < P.DK ? qr[e] : 0.0f;
                }
                float t = 0.0f;
                const float p = expf(fb_score(P, qreg, kr, mv, t) - L);
                float dp = 0.0f;
#pragma unroll
                for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                    const int64_t e = lane + c*WARP_SIZE;
                    if (e < P.DV) {
                        dp += dr[e]*vreg[c];
                    }
                }
                dp = warp_reduce_sum(dp);
                float g = p*(dp - Dbuf[row])*P.scale;
                if (P.softcap != 0.0f) {
                    g *= 1.0f - t*t;
                }
#pragma unroll
                for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                    const int64_t e = lane + c*WARP_SIZE;
                    if (e < P.DK) {
                        dkreg[c] += g*qreg[c];
                    }
                    if (e < P.DV) {
                        dvreg[c] += p*dr[e];
                    }
                }
            }
        }
    }

    float * dkr = dk + ((ik3*P.HK + ik2)*P.KV + j)*P.DK;
    float * dvr = dv + ((ik3*P.HK + ik2)*P.KV + j)*P.DV;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        if (e < P.DK) {
            dkr[e] = dkreg[c];
        }
        if (e < P.DV) {
            dvr[e] = dvreg[c];
        }
    }
}

template <typename TK, typename TV>
static void launch_fattn_back(ggml_backend_cuda_context & ctx, const fattn_back_params & P, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * o = dst->src[5];
    const ggml_tensor * d = dst->src[6];

    float * out = (float *) dst->data;
    float * dq = out;
    float * dk = out + ggml_flash_attn_back_offset(dst, 1);
    float * dv = out + ggml_flash_attn_back_offset(dst, 2);
    float * ds = out + ggml_flash_attn_back_offset(dst, 3);

    const int64_t n_rows = P.N*P.HQ*P.SEQ;
    ggml_cuda_pool_alloc<float> Lbuf(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<float> Dbuf(ctx.pool(), n_rows);

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(ds, 0, P.HQ*sizeof(float), stream));

    const dim3 block(WARP_SIZE, FATTN_BACK_WARPS);
    fattn_back_rows_kernel<TK, TV><<<(n_rows + FATTN_BACK_WARPS - 1)/FATTN_BACK_WARPS, block, 0, stream>>>(P,
        (const float *) q->data, (const TK *) k->data, (const TV *) v->data, mask ? (const half *) mask->data : nullptr,
        sinks ? (const float *) sinks->data : nullptr, (const float *) o->data, (const float *) d->data,
        dq, ds, Lbuf.get(), Dbuf.get());

    const int64_t n_keys = P.KV*P.HK*(P.SEQ/P.rk3);
    fattn_back_keys_kernel<TK, TV><<<(n_keys + FATTN_BACK_WARPS - 1)/FATTN_BACK_WARPS, block, 0, stream>>>(P,
        (const float *) q->data, (const TK *) k->data, (const TV *) v->data, mask ? (const half *) mask->data : nullptr,
        (const float *) d->data, Lbuf.get(), Dbuf.get(), dk, dv);
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM path: the same math in blocks of R query rows, with the products done by cuBLAS. Per block and GQA group
// position: S = K^T Q (KV x R per head), a row kernel turns S into P (mask, ALiBi, softcap, sinks, logsumexp) and
// computes D = dO.O, dP = V^T dO, dS = P (dP - D) ds/dqk, dV += dO P^T, dQ = K dS, dK += Q dS^T. The memory is
// O(R x KV x heads) instead of the full attention matrix. GGML_CUDA_FA_BACK_NAIVE=1 uses the kernels above.

static __device__ __forceinline__ float fb_block_reduce(float v, bool is_max, float * sh) {
    v = is_max ? warp_reduce_max(v) : warp_reduce_sum(v);
    const int w = threadIdx.x / WARP_SIZE, l = threadIdx.x % WARP_SIZE;
    __syncthreads();
    if (l == 0) {
        sh[w] = v;
    }
    __syncthreads();
    const int nw = blockDim.x / WARP_SIZE;
    v = l < nw ? sh[l] : (is_max ? -INFINITY : 0.0f);
    v = is_max ? warp_reduce_max(v) : warp_reduce_sum(v);
    return v;
}

// one block per (row i of the block, query head h): S row -> P row (in place), L and D of the row, dSinks
static __global__ void fattn_back_gemm_rows(const fattn_back_params P, float * S, const half * mask, const float * sinks,
        const float * o, const float * d, float * Lbuf, float * Dbuf, float * dsinks, const int64_t q0, const int64_t R,
        const int64_t iq3) {
    __shared__ float sh[32];
    const int64_t i = blockIdx.x;
    const int64_t h = blockIdx.y;
    float * srow = S + (h*R + i)*P.KV;
    const half * mrow = mask ? mask + (q0 + i)*P.m1 + (h % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 : nullptr;
    const float slope = fb_slope(P, h);
    const float sink  = sinks ? sinks[h] : -INFINITY;

    float M = sink;
    for (int64_t j = threadIdx.x; j < P.KV; j += blockDim.x) {
        const float mv = mrow ? slope*__half2float(mrow[j]) : 0.0f;
        float s = -INFINITY;
        if (mv != -INFINITY) {
            s = srow[j]*P.scale_qk;
            if (P.softcap != 0.0f) {
                s = P.softcap*tanhf(s);
            }
            s += mv;
        }
        srow[j] = s;
        M = fmaxf(M, s);
    }
    M = fb_block_reduce(M, true, sh);
    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j < P.KV; j += blockDim.x) {
        sum += srow[j] == -INFINITY ? 0.0f : expf(srow[j] - M);
    }
    sum = fb_block_reduce(sum, false, sh);
    if (sinks && M != -INFINITY) {
        sum += expf(sink - M);
    }
    const float L = M == -INFINITY ? -INFINITY : M + logf(sum);
    for (int64_t j = threadIdx.x; j < P.KV; j += blockDim.x) {
        srow[j] = L == -INFINITY || srow[j] == -INFINITY ? 0.0f : expf(srow[j] - L);
    }

    const float * orow = o + h*P.o1 + (q0 + i)*P.o2 + iq3*P.o3;
    const float * drow = d + h*P.o1 + (q0 + i)*P.o2 + iq3*P.o3;
    float Dp = 0.0f;
    for (int64_t e = threadIdx.x; e < P.DV; e += blockDim.x) {
        Dp += orow[e]*drow[e];
    }
    const float D = fb_block_reduce(Dp, false, sh);
    if (threadIdx.x == 0) {
        Lbuf[h*R + i] = L;
        Dbuf[h*R + i] = D;
        if (sinks && L != -INFINITY) {
            atomicAdd(dsinks + h, -expf(sink - L)*D);
        }
    }
}

// dS = P (dP - D) scale (1 - t^2 with softcap; t recovered from P: s = log p + L, t = (s - mask)/softcap), into dP
static __global__ void fattn_back_gemm_ds(const fattn_back_params P, const float * S, float * dP, const half * mask,
        const float * Lbuf, const float * Dbuf, const int64_t q0, const int64_t R, const int64_t rows, const int64_t iq3) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= P.KV*R*P.HQ) {
        return;
    }
    const int64_t j = idx % P.KV;
    const int64_t i = (idx / P.KV) % R;
    const int64_t h = idx / (P.KV*R);
    if (i >= rows) {
        return;
    }
    const float p = S[idx];
    float g = 0.0f;
    if (p > 0.0f) {
        g = p*(dP[idx] - Dbuf[h*R + i])*P.scale;
        if (P.softcap != 0.0f) {
            const float mv = mask ? fb_slope(P, h)*__half2float(mask[(q0 + i)*P.m1 + (h % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 + j]) : 0.0f;
            const float t = (logf(p) + Lbuf[h*R + i] - mv)/P.softcap;
            g *= 1.0f - t*t;
        }
    }
    dP[idx] = g;
}

// strided (possibly F16) K or V -> contiguous F32 [D, KV, H, SEQ]
template <typename T>
static __global__ void fattn_back_to_f32(const T * x, float * y, const int64_t D, const int64_t KV, const int64_t H, const int64_t SEQ,
        const int64_t s1, const int64_t s2, const int64_t s3) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= D*KV*H*SEQ) {
        return;
    }
    const int64_t e = idx % D, j = (idx / D) % KV, h = (idx / (D*KV)) % H, sq = idx / (D*KV*H);
    y[idx] = fb_load(x, e + j*s1 + h*s2 + sq*s3);
}

static void fattn_back_gemm(ggml_backend_cuda_context & ctx, const fattn_back_params & P, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * o = dst->src[5];
    const ggml_tensor * d = dst->src[6];

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    float * out = (float *) dst->data;
    float * dq = out;
    float * dk = out + ggml_flash_attn_back_offset(dst, 1);
    float * dv = out + ggml_flash_attn_back_offset(dst, 2);
    float * ds = out + ggml_flash_attn_back_offset(dst, 3);
    CUDA_CHECK(cudaMemsetAsync(out, 0, ggml_nbytes(dst), stream));

    // K and V as F32 with known strides (training caches are F32 already; the flash attention forward may cast them)
    const int64_t SEQK = P.SEQ / P.rk3;
    const float * kf = (const float *) k->data;
    const float * vf = (const float *) v->data;
    int64_t k1 = P.k1, k2 = P.k2, k3 = P.k3, v1 = P.v1, v2 = P.v2, v3 = P.v3;
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool()), v_f32(ctx.pool());
    if (k->type != GGML_TYPE_F32) {
        const int64_t nk = P.DK*P.KV*P.HK*SEQK, nv = P.DV*P.KV*P.HK*SEQK;
        k_f32.alloc(nk); v_f32.alloc(nv);
        fattn_back_to_f32<half><<<(nk + 255)/256, 256, 0, stream>>>((const half *) k->data, k_f32.get(), P.DK, P.KV, P.HK, SEQK, P.k1, P.k2, P.k3);
        fattn_back_to_f32<half><<<(nv + 255)/256, 256, 0, stream>>>((const half *) v->data, v_f32.get(), P.DV, P.KV, P.HK, SEQK, P.v1, P.v2, P.v3);
        kf = k_f32.get(); vf = v_f32.get();
        k1 = P.DK; k2 = P.DK*P.KV; k3 = P.DK*P.KV*P.HK;
        v1 = P.DV; v2 = P.DV*P.KV; v3 = P.DV*P.KV*P.HK;
    }

    // block of query rows: S and dP of a block take at most ~128 MiB each
    const int64_t R = std::max<int64_t>(1, std::min<int64_t>(P.N, (int64_t(32) << 20) / std::max<int64_t>(1, P.KV*P.HQ)));
    ggml_cuda_pool_alloc<float> S(ctx.pool(), P.KV*R*P.HQ), dP(ctx.pool(), P.KV*R*P.HQ);
    ggml_cuda_pool_alloc<float> Lbuf(ctx.pool(), R*P.HQ), Dbuf(ctx.pool(), R*P.HQ);
    const float one = 1.0f, zero = 0.0f;
    const float * qd = (const float *) q->data;
    const float * dd = (const float *) d->data;

    for (int64_t iq3 = 0; iq3 < P.SEQ; ++iq3) {
        const int64_t ik3 = iq3 / P.rk3;
        for (int64_t q0 = 0; q0 < P.N; q0 += R) {
            const int64_t rows = std::min(R, P.N - q0);
            // S = K^T Q, per GQA group position g: query heads j*rk2 + g use K head j
            for (int64_t g = 0; g < P.rk2; ++g) {
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, P.KV, rows, P.DK,
                    &one,  kf + ik3*k3, k1, k2,
                           qd + iq3*P.q3 + q0*P.q1 + g*P.q2, P.q1, P.rk2*P.q2,
                    &zero, S.get() + g*P.KV*R, P.KV, P.rk2*P.KV*R, P.HK));
            }
            fattn_back_gemm_rows<<<dim3(rows, P.HQ), 256, 0, stream>>>(P, S.get(), mask ? (const half *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr, (const float *) o->data, dd, Lbuf.get(), Dbuf.get(), ds, q0, R, iq3);
            for (int64_t g = 0; g < P.rk2; ++g) {
                const float * dO = dd + iq3*P.o3 + q0*P.o2 + g*P.DV;
                // dP = V^T dO
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, P.KV, rows, P.DV,
                    &one,  vf + ik3*v3, v1, v2,
                           dO, P.o2, P.rk2*P.DV,
                    &zero, dP.get() + g*P.KV*R, P.KV, P.rk2*P.KV*R, P.HK));
                // dV += dO P^T
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, P.DV, P.KV, rows,
                    &one,  dO, P.o2, P.rk2*P.DV,
                           S.get() + g*P.KV*R, P.KV, P.rk2*P.KV*R,
                    &one,  dv + ik3*P.HK*P.KV*P.DV, P.DV, P.KV*P.DV, P.HK));
            }
            const int64_t n = P.KV*R*P.HQ;
            fattn_back_gemm_ds<<<(n + 255)/256, 256, 0, stream>>>(P, S.get(), dP.get(), mask ? (const half *) mask->data : nullptr,
                Lbuf.get(), Dbuf.get(), q0, R, rows, iq3);
            for (int64_t g = 0; g < P.rk2; ++g) {
                // dQ = K dS
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, P.DK, rows, P.KV,
                    &one,  kf + ik3*k3, k1, k2,
                           dP.get() + g*P.KV*R, P.KV, P.rk2*P.KV*R,
                    &zero, dq + (iq3*P.HQ + g)*P.N*P.DK + q0*P.DK, P.DK, P.rk2*P.N*P.DK, P.HK));
                // dK += Q dS^T
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, P.DK, P.KV, rows,
                    &one,  qd + iq3*P.q3 + q0*P.q1 + g*P.q2, P.q1, P.rk2*P.q2,
                           dP.get() + g*P.KV*R, P.KV, P.rk2*P.KV*R,
                    &one,  dk + ik3*P.HK*P.KV*P.DK, P.DK, P.KV*P.DK, P.HK));
            }
        }
    }
}

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    return q->type == GGML_TYPE_F32 && k->type == v->type && (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) &&
        k->ne[0] <= FATTN_BACK_MAX_CHUNKS*WARP_SIZE && v->ne[0] <= FATTN_BACK_MAX_CHUNKS*WARP_SIZE &&
        q->nb[0] == sizeof(float) && k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
        (!mask || (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask))) &&
        ggml_is_contiguous(dst->src[6]) && dst->src[5]->nb[0] == sizeof(float);
}

void ggml_cuda_op_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * o = dst->src[5];
    GGML_ASSERT(ggml_cuda_flash_attn_back_supported(dst));
    GGML_ASSERT(ggml_are_same_stride(o, dst->src[6]) || ggml_is_contiguous(o));

    fattn_back_params P;
    P.DK = k->ne[0]; P.DV = v->ne[0]; P.N = q->ne[1]; P.KV = k->ne[1];
    P.HQ = q->ne[2]; P.HK = k->ne[2]; P.SEQ = q->ne[3];
    P.rk2 = q->ne[2]/k->ne[2]; P.rk3 = q->ne[3]/k->ne[3];
    const size_t tk = ggml_type_size(k->type), tv = ggml_type_size(v->type), fs = sizeof(float);
    P.q1 = q->nb[1]/fs; P.q2 = q->nb[2]/fs; P.q3 = q->nb[3]/fs;
    P.k1 = k->nb[1]/tk; P.k2 = k->nb[2]/tk; P.k3 = k->nb[3]/tk;
    P.v1 = v->nb[1]/tv; P.v2 = v->nb[2]/tv; P.v3 = v->nb[3]/tv;
    // d is contiguous with the shape of o: [DV, HQ, N, SEQ]
    P.o1 = P.DV; P.o2 = P.DV*P.HQ; P.o3 = P.DV*P.HQ*P.N;
    GGML_ASSERT(o->nb[1] == P.o1*fs && o->nb[2] == P.o2*fs && o->nb[3] == P.o3*fs);
    P.m1 = mask ? mask->nb[1]/sizeof(half) : 0; P.m2 = mask ? mask->nb[2]/sizeof(half) : 0; P.m3 = mask ? mask->nb[3]/sizeof(half) : 0;
    P.m_ne2 = mask ? mask->ne[2] : 1; P.m_ne3 = mask ? mask->ne[3] : 1;
    P.scale    = ggml_get_op_params_f32(dst, 0);
    P.max_bias = ggml_get_op_params_f32(dst, 1);
    P.softcap  = ggml_get_op_params_f32(dst, 2);
    P.scale_qk = P.softcap != 0.0f ? P.scale/P.softcap : P.scale;
    P.n_head_log2 = 1u << (uint32_t) floorf(log2f((float) P.HQ));
    P.m0  = powf(2.0f, -(P.max_bias       )/P.n_head_log2);
    P.m1f = powf(2.0f, -(P.max_bias/2.0f)/P.n_head_log2);

    static const bool naive = getenv("GGML_CUDA_FA_BACK_NAIVE") != nullptr;
    if (!naive) {
        fattn_back_gemm(ctx, P, dst);
    } else if (k->type == GGML_TYPE_F16) {
        launch_fattn_back<half, half>(ctx, P, dst);
    } else {
        launch_fattn_back<float, float>(ctx, P, dst);
    }
}
