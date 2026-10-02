#include "allreduce-host.cuh"
#include "ggml-backend-impl.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

// host slots: [parity][device][max_ne] floats; two parities so that a device can write the next sum while
// another device may still read the previous one (it cannot be two sums behind: it raised the flag of the
// previous sum only after finishing the one before)
static constexpr int64_t ar_host_max_ne     = 4*1024*1024;
static constexpr int     ar_host_flag_stride = 32; // uint64 per flag, keeps flags of devices apart

struct ggml_cuda_allreduce_host_state {
    std::mutex mutex;
    bool       init_done = false;
    bool       usable    = false;
    bool       bf16      = false;
    float    * slots     = nullptr;
    uint64_t * flags     = nullptr;
    uint64_t   epoch     = 0;
    unsigned int * counters[GGML_CUDA_MAX_DEVICES] = {}; // per device: blocks done writing, [2..3] epoch ready
    float        * other[GGML_CUDA_MAX_DEVICES]    = {}; // per device: the other device's part (two devices)
};

static ggml_cuda_allreduce_host_state & ar_host_state() {
    static ggml_cuda_allreduce_host_state state;
    return state;
}

// One kernel per device: every thread copies its elements to this device's host slot, the last block to
// finish raises the device's flag, then every block waits for the flags of the other devices and adds their
// slots. Blocks that finished writing start reading while others still write, so on a slow link the two
// directions overlap. All blocks must be resident at once (the grid is kept below the SM capacity).
// slot values: f32, or bf16 to halve the traffic (then every device sums the rounded values of all devices,
// its own included, so the results stay identical across devices)
template <typename T> static __device__ __forceinline__ T    ar_host_to_slot(float x);
template <>           __device__ __forceinline__ float          ar_host_to_slot<float>(float x)          { return x; }
template <>           __device__ __forceinline__ unsigned short ar_host_to_slot<unsigned short>(float x) {
    return __bfloat16_as_ushort(__float2bfloat16(x));
}
static __device__ __forceinline__ float ar_host_from_slot(float x)          { return x; }
static __device__ __forceinline__ float ar_host_from_slot(unsigned short x) { return __bfloat162float(__ushort_as_bfloat16(x)); }

template <typename T>
static __global__ void k_ar_host(float * __restrict__ data, T * __restrict__ slots, const int64_t slot_stride,
        uint64_t * flags, unsigned int * counter, const int dev, const int n_dev, const uint64_t epoch, const int64_t ne) {
    const int64_t i0     = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;

    T * own_slot = slots + dev*slot_stride;
    for (int64_t i = i0; i < ne; i += stride) {
        own_slot[i] = ar_host_to_slot<T>(data[i]);
    }

    __shared__ bool last;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();

    // only the last block polls the host flags (polling crosses the link); the others wait on device memory
    volatile uint64_t * vflags = flags;
    volatile unsigned long long * vready = (volatile unsigned long long *) (counter + 2);
    if (threadIdx.x == 0) {
        if (last) {
            *counter = 0; // the next sum on this stream starts after this kernel
            __threadfence_system();
            vflags[dev*ar_host_flag_stride] = epoch;
            for (int k = 0; k < n_dev; ++k) {
                if (k != dev) {
                    while (vflags[k*ar_host_flag_stride] < epoch) {
                    }
                }
            }
            __threadfence_system();
            *vready = epoch;
        } else {
            while (*vready < epoch) {
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    for (int64_t i = i0; i < ne; i += stride) {
        const float own = ar_host_from_slot(ar_host_to_slot<T>(data[i]));
        float sum = 0.0f;
        for (int k = 0; k < n_dev; ++k) {
            // the slots change under the GPU's caches between sums: load without caching
            sum += k == dev ? own : ar_host_from_slot(__ldcv(slots + k*slot_stride + i));
        }
        data[i] = sum;
    }
}

// Two devices: writer blocks copy this device's part to its host slot while reader blocks wait for the other
// device's flag and copy its slot to a device buffer, so both directions of a slow link are busy at once.
// k_ar_host_add then adds the two (in the same rounding on both devices; a + b == b + a, so the results match).
template <typename T>
static __global__ void k_ar_host_xchg2(const float * __restrict__ data, T * __restrict__ slots, const int64_t slot_stride,
        float * __restrict__ other, uint64_t * flags, unsigned int * counter, const int dev, const int n_writers,
        const uint64_t epoch, const int64_t ne) {
    volatile uint64_t * vflags = flags;
    if ((int) blockIdx.x < n_writers) {
        T * own_slot = slots + dev*slot_stride;
        for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) n_writers*blockDim.x) {
            own_slot[i] = ar_host_to_slot<T>(data[i]);
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0 && atomicAdd(counter, 1u) == (unsigned int) n_writers - 1) {
            *counter = 0; // the next sum on this stream starts after this kernel
            __threadfence_system();
            vflags[dev*ar_host_flag_stride] = epoch;
        }
        return;
    }

    // readers: the first one polls the other device's host flag, the others wait on device memory
    const int reader    = blockIdx.x - n_writers;
    const int n_readers = gridDim.x - n_writers;
    volatile unsigned long long * vready = (volatile unsigned long long *) (counter + 2);
    if (threadIdx.x == 0) {
        if (reader == 0) {
            while (vflags[(1 - dev)*ar_host_flag_stride] < epoch) {
            }
            __threadfence_system();
            *vready = epoch;
        } else {
            while (*vready < epoch) {
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    const T * other_slot = slots + (1 - dev)*slot_stride;
    for (int64_t i = (int64_t) reader*blockDim.x + threadIdx.x; i < ne; i += (int64_t) n_readers*blockDim.x) {
        other[i] = ar_host_from_slot(__ldcv(other_slot + i));
    }
}

template <typename T>
static __global__ void k_ar_host_add(float * __restrict__ data, const float * __restrict__ other, const int64_t ne) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) gridDim.x*blockDim.x) {
        data[i] = ar_host_from_slot(ar_host_to_slot<T>(data[i])) + other[i];
    }
}

bool ggml_cuda_allreduce_host(ggml_backend_t * backends, ggml_tensor ** tensors, size_t n_backends) {
    if (n_backends < 2 || n_backends > GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne > ar_host_max_ne) {
        return false;
    }
    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne || !ggml_is_contiguously_allocated(tensors[i])) {
            return false;
        }
    }
    if (ne == 0) {
        return true;
    }

    ggml_cuda_allreduce_host_state & st = ar_host_state();
    std::lock_guard<std::mutex> lock(st.mutex);

    if (!st.init_done) {
        st.init_done = true;
        const char * env = getenv("GGML_CUDA_HOST_ALLREDUCE");
        if (env && atoi(env) == 0) {
            return false;
        }
        const size_t n_slots = 2*GGML_CUDA_MAX_DEVICES;
        if (cudaHostAlloc((void **) &st.slots, n_slots*ar_host_max_ne*sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void **) &st.flags, GGML_CUDA_MAX_DEVICES*ar_host_flag_stride*sizeof(uint64_t),
                cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_WARN("%s: cannot allocate mapped host memory, using the copy fallback\n", __func__);
            return false;
        }
        const char * env_bf16 = getenv("GGML_CUDA_HOST_ALLREDUCE_BF16");
        st.bf16 = env_bf16 && atoi(env_bf16) != 0;
        memset(st.flags, 0, GGML_CUDA_MAX_DEVICES*ar_host_flag_stride*sizeof(uint64_t));
        st.usable = true;
        GGML_LOG_INFO("%s: summing partial results across GPUs through mapped host memory%s\n", __func__,
            st.bf16 ? " (bf16 transfer)" : "");
    }
    if (!st.usable) {
        return false;
    }

    const uint64_t epoch  = ++st.epoch;
    float *        slots  = st.slots + (epoch % 2)*GGML_CUDA_MAX_DEVICES*ar_host_max_ne;
    const int      n_dev  = (int) n_backends;
    const int      block  = 256;
    const int      blocks = (int) std::min<int64_t>((ne + block - 1)/block, 64);

    for (int i = 0; i < n_dev; ++i) {
        ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backends[i]->context;
        ggml_cuda_set_device(ctx->device);
        if (st.counters[ctx->device] == nullptr) {
            CUDA_CHECK(cudaMalloc((void **) &st.counters[ctx->device], 4*sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(st.counters[ctx->device], 0, 4*sizeof(unsigned int)));
        }
        if (n_dev == 2) {
            if (st.other[ctx->device] == nullptr) {
                CUDA_CHECK(cudaMalloc((void **) &st.other[ctx->device], ar_host_max_ne*sizeof(float)));
            }
            const int n_half = (int) std::min<int64_t>((ne + block - 1)/block, 32);
            float * data = (float *) tensors[i]->data;
            if (st.bf16) {
                k_ar_host_xchg2<<<2*n_half, block, 0, ctx->stream()>>>(data, (unsigned short *) slots, ar_host_max_ne,
                    st.other[ctx->device], st.flags, st.counters[ctx->device], i, n_half, epoch, ne);
                k_ar_host_add<unsigned short><<<blocks, block, 0, ctx->stream()>>>(data, st.other[ctx->device], ne);
            } else {
                k_ar_host_xchg2<<<2*n_half, block, 0, ctx->stream()>>>(data, slots, ar_host_max_ne,
                    st.other[ctx->device], st.flags, st.counters[ctx->device], i, n_half, epoch, ne);
                k_ar_host_add<float><<<blocks, block, 0, ctx->stream()>>>(data, st.other[ctx->device], ne);
            }
        } else if (st.bf16) {
            k_ar_host<<<blocks, block, 0, ctx->stream()>>>((float *) tensors[i]->data, (unsigned short *) slots, ar_host_max_ne,
                st.flags, st.counters[ctx->device], i, n_dev, epoch, ne);
        } else {
            k_ar_host<<<blocks, block, 0, ctx->stream()>>>((float *) tensors[i]->data, slots, ar_host_max_ne,
                st.flags, st.counters[ctx->device], i, n_dev, epoch, ne);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
