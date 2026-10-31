#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend-share.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Source of the buffer epochs below: a value is never handed out twice, so equal epochs mean an unchanged buffer.
static std::atomic<uint64_t> ggml_backend_meta_epoch_counter{0};

struct ggml_backend_meta_buffer_context {
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);

    // hash maps: the lookups run for every node of every graph, the order of the entries is never used
    struct tensor_flag_hash {
        size_t operator()(const std::pair<const ggml_tensor *, bool> & k) const {
            return std::hash<const ggml_tensor *>()(k.first) ^ size_t(k.second);
        }
    };

    // split states of views and results, which come and go with the graphs
    std::unordered_map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>, tensor_flag_hash> split_state_cache;
    // split states of the buffer's own tensors (weights, caches), which depend on nothing but the tensor itself
    std::unordered_map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>, tensor_flag_hash> split_state_cache_own;
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>>                               simple_tensors;

    // new whenever a tensor of the buffer is (re)initialized or the buffer is reset, i.e. whenever a graph in it may have
    // new tensors; graph_compute reuses the subgraphs of a graph only while the epochs of its buffers stand still
    uint64_t epoch = ggml_backend_meta_epoch_counter.fetch_add(1, std::memory_order_relaxed) + 1;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;
    bool owns_bufs = true; // false: an alias of another meta buffer's memory (ggml_backend_meta_buffer_alias)

    int debug;

    ggml_backend_meta_buffer_context() {
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf] : buf_ctx->buf_configs) {
        if (buf_ctx->owns_bufs) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
    }
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->buf_configs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());
    return buf_ctx->buf_configs[index].buf;
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

// a view of a tensor in a host buffer: ggml-alloc never allocates views, so it is not in a meta buffer even when it lands
// in a meta split (the scheduler copies its source); the view node itself is a no-op (upstream #30217)
static bool ggml_backend_meta_is_host_view(const struct ggml_tensor * tensor) {
    return ggml_is_view(tensor) && ggml_backend_buffer_is_host(tensor->view_src->buffer);
}

// A split state is 2 KiB (ne has room for 16 segments on 16 devices), almost all of it unused: copying or zeroing whole states
// dominated the graph rebuilds and every compute with -sm tensor. Copies take only the used part of ne.
// A state with one segment on the given axis; only the entries of one segment are zeroed, as only those are read.
static ggml_backend_meta_split_state ggml_backend_meta_split_state_of(const ggml_backend_meta_split_axis axis) {
    ggml_backend_meta_split_state ret;
    ret.axis       = axis;
    ret.n_segments = 1;
    std::fill_n(ret.ne, GGML_BACKEND_META_MAX_DEVICES, int64_t(0));
    return ret;
}

static void ggml_backend_meta_split_state_copy(
        ggml_backend_meta_split_state & dst, const ggml_backend_meta_split_state & src, const size_t n_bufs) {
    dst.axis       = src.axis;
    dst.n_segments = src.n_segments;
    memcpy(dst.ne, src.ne, src.n_segments*n_bufs*sizeof(int64_t));
}

// Scratch for the source states of one split-state computation, one set per recursion level, reused so that the states are
// neither allocated nor zeroed for every tensor.
struct ggml_backend_meta_src_states {
    ggml_backend_meta_split_state * ss;

    ggml_backend_meta_src_states() {
        auto & lv = levels();
        if (depth() == lv.size()) {
            lv.emplace_back(new ggml_backend_meta_split_state[GGML_MAX_SRC]);
        }
        ss = lv[depth()++].get();
    }
    ~ggml_backend_meta_src_states() {
        depth()--;
    }

    static std::vector<std::unique_ptr<ggml_backend_meta_split_state[]>> & levels() {
        static thread_local std::vector<std::unique_ptr<ggml_backend_meta_split_state[]>> lv;
        return lv;
    }
    static size_t & depth() {
        static thread_local size_t d = 0;
        return d;
    }
};

// The split state of a tensor from the cache of its buffer, computed on a miss. The reference stays valid until the next lookup
// that computes a state (a changed view empties the cache), so callers take what they need from it right away.
static const ggml_backend_meta_split_state & ggml_backend_meta_split_state_ref(const struct ggml_tensor * tensor, bool assume_sync) {
    if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
        GGML_ASSERT(ggml_backend_meta_is_host_view(tensor));
        // not allocated in the meta buffer, not split across the sub-devices
        static const ggml_backend_meta_split_state not_split = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_NONE);
        return not_split;
    }
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const ggml_backend_meta_split_state * src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_NONE);
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // set by the handlers that give the elements per device themselves instead of taking the ratio of a source
    bool ne_explicit = false;

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Top-k of rows split along axis 0 (with -sm tensor the logits are split over the vocabulary): each device takes the
    // top ne[0]/n_bufs of its part, as indices into its part. The caller asks for k per device times the devices and maps
    // the indices to the whole row; get_rows with them gathers within the same parts (handle_get_rows).
    auto handle_top_k = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) {
            return handle_per_row(src_ss);
        }
        GGML_ASSERT(src_ss[0].n_segments == 1 && tensor->ne[0] % int64_t(n_bufs) == 0);
        ggml_backend_meta_split_state ret = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_0);
        for (size_t j = 0; j < n_bufs; j++) {
            ret.ne[j] = tensor->ne[0] / int64_t(n_bufs);
        }
        ne_explicit = true;
        return ret;
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.n_segments = 1;
            return ret;
        }
        if ((src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 ||
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_3) && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            // a mirrored src0 broadcast over src1: the result is split along the same src1 axis
            ggml_backend_meta_split_state ret = src_ss[1];
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return ggml_backend_meta_split_state_of(assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL);
        }
        GGML_ABORT("fatal error");
        //return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    };

    auto handle_cpy = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            int64_t ne_split_src = tensor->src[0]->ne[0];
            for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                ne_split_src *= tensor->src[0]->ne[dim];
            }
            int64_t ne_split_dst = 1;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ne_split_dst *= tensor->ne[dim];
                if (ne_split_dst == ne_split_src) {
                    return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(dim));
                }
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    // the split axis of a reshape, chosen assuming each device holds a contiguous slice of the source axis
    auto handle_reshape_contiguous = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]));
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1) {
                    return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1));
                }
                std::vector<int64_t> base_ne_in;
                base_ne_in.reserve(GGML_MAX_DIMS - src_ss[0].axis);
                {
                    base_ne_in.push_back(1);
                    int dim = 0;
                    for (; dim <= src_ss[0].axis; dim++) {
                        base_ne_in[0] *= tensor->src[0]->ne[dim];
                    }
                    for (; dim <= GGML_MAX_DIMS; dim++) {
                        base_ne_in.push_back(base_ne_in.back() * tensor->src[0]->ne[dim]);
                    }
                }
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    for (const int64_t & bni : base_ne_in) {
                        if (bni == base_ne_out_next) {
                            return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(dim));
                        }
                    }
                    if (base_ne_out_next > base_ne_in[0]) {
                        GGML_ASSERT(dim + 1 < GGML_MAX_DIMS);
                        return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(dim + 1));
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
            }
        }
    };

    // A reshape that splits the source axis into [inner, outer] dims may not be expressible as a contiguous
    // slice: a tensor that was split in equal segments (e.g. a GDN value axis [head_dim, n_k_heads, head_ratio]
    // with n_k_heads split in every head_ratio segment) holds a part of every outer block on each device.
    // When the contiguous choice gives fractional per-device sizes, use the innermost axis that divides evenly.
    auto handle_reshape = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = handle_reshape_contiguous(src_ss);
        const int src_axis = src_ss[0].axis;
        if (src_axis < 0 || src_axis >= GGML_MAX_DIMS || ret.axis < 0 || ret.axis >= GGML_MAX_DIMS ||
                ggml_nelements(tensor) != ggml_nelements(tensor->src[0])) {
            return ret;
        }
        const int64_t ne_src = tensor->src[0]->ne[src_axis];
        auto divides_evenly = [&](int dim) -> bool {
            for (size_t j = 0; j < n_bufs; j++) {
                int64_t ne_j = 0;
                for (size_t s = 0; s < src_ss[0].n_segments; s++) {
                    ne_j += src_ss[0].ne[s*n_bufs + j];
                }
                if ((ne_j * tensor->ne[dim]) % ne_src != 0) {
                    return false;
                }
            }
            return true;
        };
        if (divides_evenly(ret.axis)) {
            return ret;
        }
        int64_t inner_src = 1;
        for (int dim = 0; dim < src_axis; dim++) {
            inner_src *= tensor->src[0]->ne[dim];
        }
        int64_t inner_dst = 1;
        for (int dim = 0; dim < ret.axis; dim++) {
            inner_dst *= tensor->ne[dim];
        }
        for (int dim = ret.axis - 1; dim >= 0; dim--) {
            inner_dst /= tensor->ne[dim];
            if (inner_dst % inner_src == 0 && divides_evenly(dim)) {
                return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(dim));
            }
        }
        return ret;
    };

    auto handle_view = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] && tensor->nb[0] == tensor->src[0]->nb[0] &&
                tensor->view_offs % tensor->src[0]->nb[1] == 0) {
            // whole rows of a tensor split along its rows (e.g. the slices of a MUL_MAT_ID result that are summed up)
            return src_ss[0];
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    if (tensor->ne[dim] == tensor->src[0]->ne[axis]) {
                        // the whole split axis is in the view: keep the slices of the devices
                        ggml_backend_meta_split_state ret = src_ss[0];
                        ret.axis = ggml_backend_meta_split_axis(dim);
                        return ret;
                    }
                    return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(dim));
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    };

    auto handle_permute = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
            }
        }
    };

    auto handle_transpose = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                return ggml_backend_meta_split_state_of(ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1));
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
            }
        }
    };

    auto handle_get_rows = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        // rows split over the devices, indices into each device's part (handle_top_k): gather within the parts
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(src_ss[1].n_segments == 1);
            ggml_backend_meta_split_state ret = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_1);
            for (size_t j = 0; j < n_bufs; j++) {
                ret.ne[j] = src_ss[1].ne[j];
            }
            ne_explicit = true;
            return ret;
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_1);
    };

    auto handle_ssm_conv = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_1);
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_0);
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const ggml_backend_meta_split_state * src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_0);
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t sj = 0; sj < ret.n_segments*n_bufs; sj++) {
                    GGML_ASSERT(ret.ne[sj] % granularity == 0);
                    ne_sum += ret.ne[sj];
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        const ggml_backend_meta_src_states src_states;
        ggml_backend_meta_split_state * src_ss = src_states.ss;
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i].axis       = GGML_BACKEND_SPLIT_AXIS_UNKNOWN;
                src_ss[i].n_segments = 1;
                continue;
            }
            ggml_backend_meta_split_state_copy(src_ss[i], ggml_backend_meta_split_state_ref(tensor->src[i], /*assume_sync =*/ true), n_bufs);
            if (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
                GGML_LOG_ERROR("%s: %s (%s) has src %zu %s (%s) with an unknown split\n", __func__,
                        tensor->name, ggml_op_name(tensor->op), i, tensor->src[i]->name, ggml_op_name(tensor->src[i]->op));
            }
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT: {
                // a cont of the same shape (e.g. of a permute) keeps the logical split axis
                split_state = ggml_are_same_shape(tensor, tensor->src[0]) ? src_ss[0] : handle_reshape(src_ss);
            } break;
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_TOP_K: {
                split_state = handle_top_k(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = ggml_backend_meta_split_state_of(GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS && !ne_explicit) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            if (split_state.ne[j] % tensor->src[i]->ne[src_ss[i].axis] != 0) {
                                GGML_LOG_ERROR("%s: %s (%s, ne %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 ") cannot take the split of src %zu %s "
                                        "(%s, axis %d, ne %" PRId64 "): %" PRId64 " elements on device %zu\n", __func__,
                                        tensor->name, ggml_op_name(tensor->op), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                                        i, tensor->src[i]->name, ggml_op_name(tensor->src[i]->op), src_ss[i].axis,
                                        tensor->src[i]->ne[src_ss[i].axis], split_state.ne[j], j);
                            }
                            GGML_ASSERT(split_state.ne[j] % tensor->src[i]->ne[src_ss[i].axis] == 0);
                            split_state.ne[j] /= tensor->src[i]->ne[src_ss[i].axis];
                        }
                    }
                } else {
                    for (size_t j = 0; j < n_bufs; j++) {
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j];
                        }
                        // Assert that ratio is consistent:
                        GGML_ASSERT(split_state.ne[j] * tensor->src[i]->ne[src_ss[i].axis]
                                               == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    // a view that changed with a new graph empties only the states of views and results: emptying the states of the weights and
    // caches as well would send all of them through the split callback again, at every change of the graph shape (MTP verify)
    const bool own_tensor = ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr;
    auto & cache = own_tensor ? buf_ctx->split_state_cache_own : buf_ctx->split_state_cache;

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = cache.find(key);
    if (it != cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        if (own_tensor) {
            buf_ctx->split_state_cache_own.clear();
        }
        buf_ctx->split_state_cache.clear();
        it = cache.end();
    }

    if (it != cache.end()) {
        return it->second.first;
    }

    const ggml_backend_meta_split_state ret = calculate_split_state();
    {
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state & split_state = ggml_backend_meta_split_state_ref(tensor->src[i], true);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ne_info += std::to_string(ret.ne[j]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(ret.axis), ne_info.c_str());
        }
    }

    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t sj = 0; sj < ret.n_segments*n_bufs; sj++) {
            ne_ret += ret.ne[sj];
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG

    // the states of the sources computed on the way (and the debug output) can have emptied the cache, so the entry is only made now
    auto & entry = cache[key];
    ggml_backend_meta_split_state_copy(entry.first, ret, n_bufs);
    memcpy(entry.second, tensor, sizeof(entry.second));
    return entry.first;
}

static ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    return ggml_backend_meta_split_state_ref(tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    ggml_backend_meta_split_state split_state;
    ggml_backend_meta_split_state_copy(split_state, ggml_backend_meta_split_state_ref(tensor, /*assume_sync =*/ true), n_simple_bufs);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(ne[split_dim] != 0 && tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }
    buf_ctx->simple_tensors[tensor] = simple_tensors;
    buf_ctx->epoch = ggml_backend_meta_epoch_counter.fetch_add(1, std::memory_order_relaxed) + 1;

    return GGML_STATUS_SUCCESS;
}

// Copies whole rows of a 2D tensor split along axis 0 in several segments: within a row the data is
// [seg0_dev0, seg0_dev1, ..., seg1_dev0, ...], while device j holds [seg0_dev_j, seg1_dev_j, ...] per row.
// Used for reads/writes of rows (e.g. the recurrent state of a sequence for prompt caches and checkpoints).
static bool ggml_backend_meta_rows_segmented(const ggml_tensor * tensor, const ggml_backend_meta_split_state & split_state,
        size_t n_bufs, void * data, size_t offset, size_t size, bool set) {
    if (split_state.n_segments == 1 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_0 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return false;
    }
    const size_t row_size = tensor->nb[1];
    GGML_ASSERT(offset % row_size == 0 && size % row_size == 0);
    const int64_t i_start   = offset / row_size;
    const int64_t n_rows    = size / row_size;
    const int64_t blck_size = ggml_blck_size(tensor->type);

    size_t offset_row = 0;
    std::vector<size_t> simple_offsets(n_bufs, 0);
    for (size_t s = 0; s < split_state.n_segments; s++) {
        for (size_t j = 0; j < n_bufs; j++) {
            ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
            const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
            const size_t simple_offset = i_start*simple_tensor->nb[1] + simple_offsets[j];
            if (nbytes > 0) {
                if (set) {
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_row, simple_offset,
                        nbytes, n_rows, simple_tensor->nb[1], row_size);
                } else {
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_row, simple_offset,
                        nbytes, n_rows, simple_tensor->nb[1], row_size);
                }
            }
            offset_row        += nbytes;
            simple_offsets[j] += nbytes;
        }
    }
    GGML_ASSERT(offset_row == row_size);
    return true;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    if (!(offset == 0 && size == ggml_nbytes(tensor)) &&
            ggml_backend_meta_rows_segmented(tensor, split_state, n_bufs, const_cast<void *>(data), offset, size, /*set =*/ true)) {
        return;
    }

    if (split_state.n_segments != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(offset == 0);
        GGML_ASSERT(size == ggml_nbytes(tensor));
        GGML_ASSERT(tensor->ne[3] == 1);
        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);
            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                    const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data, simple_offsets[j], nbytes,
                        tensor->ne[1], simple_tensor->nb[1], tensor->nb[1]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
            GGML_ASSERT(offset_data*tensor->ne[1] == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data, simple_offsets[j], nbytes,
                    tensor->ne[2], simple_tensor->nb[2], tensor->nb[2]);
                offset_data       += nbytes;
                simple_offsets[j] += nbytes;
            }
        }
        GGML_ASSERT(offset_data*tensor->ne[2] == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (ggml_backend_meta_rows_segmented(tensor, split_state, n_bufs, data, offset, size, /*set =*/ false)) {
        return;
    }
    GGML_ASSERT(split_state.n_segments == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    // only the graph allocator resets (compute) buffers, before it initializes the tensors of the next graph
    // again: drop the per-device tensor objects of the previous graph, otherwise every evaluation adds a full
    // set of objects to the per-device contexts until their memory pool is exhausted
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->simple_tensors.clear();
    buf_ctx->split_state_cache.clear();
    buf_ctx->split_state_cache_own.clear();
    buf_ctx->epoch = ggml_backend_meta_epoch_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    for (auto & bc : buf_ctx->buf_configs) {
        ggml_reset(bc.ctx);
        ggml_backend_buffer_reset(bc.buf);
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context();
    size_t max_size = 0;
    buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_backend_buffer_t simple_buf = ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(simple_buf));
        buf_ctx->buf_configs.emplace_back(ggml_init(params), simple_buf);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

ggml_backend_buffer_t ggml_backend_meta_buffer_alias(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    const ggml_backend_meta_buffer_context * src = (const ggml_backend_meta_buffer_context *) buffer->context;

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    // own per-device tensor objects, split states and epoch, the memory of buffer: a graph allocator resets its
    // buffers, which for a meta buffer drops the tensor objects of the graph that was placed there before
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context();
    buf_ctx->owns_bufs = false;
    buf_ctx->buf_configs.reserve(src->buf_configs.size());
    for (const auto & bc : src->buf_configs) {
        buf_ctx->buf_configs.emplace_back(ggml_init(params), bc.buf);
    }
    return ggml_backend_buffer_init(buffer->buft, ggml_backend_meta_buffer_iface, buf_ctx, buffer->size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context();
    meta_buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr);
    }

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct backend_config {
        ggml_backend_t backend;

        std::vector<ggml_tensor *> nodes; // the simple tensors of the graph being split
        ggml_backend_buffer_ptr    buf;

        backend_config(ggml_backend_t backend) : backend(backend) {}
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;

    // the mappings onto the devices and the subgraphs of the graphs evaluated last, one entry per graph, with the epochs
    // of the meta buffers of its nodes and their sources at that time (see ggml_backend_meta_graph_compute). Several
    // entries because a context keeps several graphs (llama_context: one per batch size of up to 16 tokens).
    struct split_entry {
        const ggml_cgraph * cgraph      = nullptr;
        int                 n_nodes     = -1;
        const ggml_tensor * first       = nullptr;
        const ggml_tensor * last        = nullptr;
        size_t              n_subgraphs = 0;
        struct buf_epoch {
            const ggml_tensor *         tensor; // a node or source in the buffer, so the buffer is read only through the graph
            ggml_backend_buffer_t       buffer;
            uint64_t                    epoch;
        };
        std::vector<buf_epoch>     buf_epochs;
        std::vector<ggml_cgraph *> subgraphs; // subgraph i of device j at j*n_subgraphs + i
        ggml_context_ptr           ctx;       // holds the subgraphs
        uint64_t                   last_use = 0;
    };
    static constexpr size_t  SPLIT_CACHE_SIZE = 16;
    std::vector<split_entry> splits;
    uint64_t                 split_tick = 0;


    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        name = "Meta(";
        backend_configs.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            backend_configs.emplace_back(ggml_backend_dev_init(simple_dev, params));
        }
        name += ")";
    }

    ~ggml_backend_meta_context() {
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }

    size_t n_reduce_steps() const {
        return std::ceil(std::log2(backend_configs.size()));
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend);

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (split_state.n_segments != 1) {
        // segmented tensors: synchronous row copy
        ggml_backend_meta_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }
    GGML_ASSERT(offset == 0);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (split_state.n_segments != 1) {
        // segmented tensors: synchronous row copy
        ggml_backend_meta_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks"; a read from an offset
            // (rows of one sequence of a batch) starts at the same chunk on every device.
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, i_start*chunk_size_j, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// Maps the nodes of a graph onto the simple tensors of each device and cuts the graph into the subgraphs between two
// reductions; returns the number of subgraphs.
// map cgraph onto the devices and cut it into the subgraphs between the reductions, into entry
static void ggml_backend_meta_graph_split(ggml_backend_t backend, struct ggml_cgraph * cgraph, ggml_backend_meta_context::split_entry & entry) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            backend_ctx->backend_configs[j].nodes.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
    }
    for (size_t j = 0; j < n_backends; j++) {
        auto & bcj = backend_ctx->backend_configs[j];

        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (!ggml_backend_buffer_is_meta(node->buffer)) {
                // a host view (see ggml_backend_meta_is_host_view): keep the node, it has no simple tensor to map to
                GGML_ASSERT(ggml_backend_meta_is_host_view(node));
                bcj.nodes[i] = node;
                continue;
            }
            bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
            GGML_ASSERT(bcj.nodes[i]);
        }
    }

    std::vector<int> offsets; // first node of each subgraph
    size_t max_tmp_size = 0;
    {
        // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
        auto get_i_delayed = [&](const int i) -> int {
            int id = i; // i_delayed
            int idr = i; // i_delayed return, last safe return value

            ggml_tensor * node = cgraph->nodes[id];
            int32_t n_used = ggml_node_get_use_count(cgraph, id);
            if (id + 1 >= cgraph->n_nodes) {
                return idr;
            }
            {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                        ggml_backend_meta_split_state_ref(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                        ggml_backend_meta_split_state_ref(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    node = next;
                    id++;
                    idr = id;
                    n_used = ggml_node_get_use_count(cgraph, id);
                }
            }
            if (id + 1 >= cgraph->n_nodes) {
                return idr;
            }
            {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op == GGML_OP_MUL && next->src[0] == node &&
                        ggml_backend_meta_split_state_ref(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    node = next;
                    id++;
                    idr = id;
                    n_used = ggml_node_get_use_count(cgraph, id);
                }
            }

            if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                return idr;
            }
            for (int32_t k = 0; k < n_used; k++) {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                        next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                        ggml_node_get_use_count(cgraph, id+1) != 1) {
                    return idr;
                }
                id++;
            }
            {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                        next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                    return idr;
                }
                id++;
            }
            for (int32_t k = 0; k < n_used - 2; k++) {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                        next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                    return idr;
                }
                id++;
            }
            idr = id;
            return idr;
        };

        int i_start = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                continue;
            }
            const ggml_backend_meta_split_axis axis = ggml_backend_meta_split_state_ref(node, /*assume_sync =*/ false).axis;
            if (axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
            }
            const bool new_subgraph = i + 1 == cgraph->n_nodes || axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
            if (!new_subgraph) {
                continue;
            }

            i = get_i_delayed(i);

            offsets.push_back(i_start);
            i_start = i + 1;
        }
        GGML_ASSERT(i_start == cgraph->n_nodes);
    }

    if (max_tmp_size > backend_ctx->max_tmp_size) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.buf.reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
        }
        backend_ctx->max_tmp_size = max_tmp_size;
    }


    const size_t n_subgraphs = offsets.size();

    // the nodes and graphs of the reductions, built at each evaluation
    if (n_subgraphs > backend_ctx->max_subgraphs) {
        backend_ctx->max_subgraphs = n_subgraphs;
        const size_t n_reduce_steps = backend_ctx->n_reduce_steps();
        const size_t n_nodes_per_device = 2 * n_reduce_steps; // tmp + ADD per step
        const size_t n_cgraphs_per_device = n_reduce_steps;    // 1 ADD graph per step
        const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
        const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
        ggml_init_params params = {
            /*.mem_size   =*/ n_backends * (mem_per_device_graphs_aux + mem_per_device_nodes_aux),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        backend_ctx->ctx.reset(ggml_init(params));
        backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
        for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
            backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
        }
        backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
        for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
            backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
        }
    }

    // the subgraphs, each sized for its own nodes
    auto n_nodes_of = [&](size_t i_graph) -> size_t {
        return (i_graph + 1 < n_subgraphs ? offsets[i_graph + 1] : cgraph->n_nodes) - offsets[i_graph];
    };
    size_t mem_subgraphs = 0;
    for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
        mem_subgraphs += ggml_graph_overhead_custom(n_nodes_of(i_graph), /*grads =*/ false);
    }
    ggml_init_params params = {
        /*.mem_size   =*/ n_backends * mem_subgraphs,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    entry.ctx.reset(ggml_init(params));
    entry.n_subgraphs = n_subgraphs;
    entry.subgraphs.resize(n_backends*n_subgraphs);
    for (size_t j = 0; j < n_backends; j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
            const size_t i_node_start = offsets[i_graph];
            const size_t n_nodes_ij   = n_nodes_of(i_graph);
            ggml_cgraph * cgraph_ij = ggml_new_graph_custom(entry.ctx.get(), n_nodes_ij, /*grads =*/ false);
            entry.subgraphs[j*n_subgraphs + i_graph] = cgraph_ij;
            cgraph_ij->n_nodes = n_nodes_ij;
            for (size_t i_node = i_node_start; i_node < i_node_start + n_nodes_ij; i_node++) {
                ggml_tensor * node_ij = bcj.nodes[i_node];
                cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
            }
        }
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // The mapping and the subgraphs depend only on the graph and on the tensors of the meta buffers, so a graph that is
    // evaluated again without being reallocated (a decode without graph rebuild) reuses them: that saves a split-state
    // lookup per node and a map lookup per node and device. One entry per graph, for contexts that switch between graphs.
    using split_entry = ggml_backend_meta_context::split_entry;
    auto & splits = backend_ctx->splits;
    auto valid = [&](const split_entry & e) {
        bool same = cgraph == e.cgraph && cgraph->n_nodes == e.n_nodes && cgraph->n_nodes > 0 &&
            cgraph->nodes[0] == e.first && cgraph->nodes[cgraph->n_nodes - 1] == e.last;
        for (size_t k = 0; same && k < e.buf_epochs.size(); k++) {
            const auto & be = e.buf_epochs[k];
            same = be.tensor->buffer == be.buffer &&
                ((const ggml_backend_meta_buffer_context *) be.buffer->context)->epoch == be.epoch;
        }
        return same;
    };
    split_entry * entry = nullptr;
    for (auto & e : splits) {
        if (valid(e)) {
            entry = &e;
            break;
        }
    }
    if (entry == nullptr) {
        // the entry of this graph from before its reallocation, a new entry, or the one used longest ago
        for (auto & e : splits) {
            if (e.cgraph == cgraph) {
                entry = &e;
                break;
            }
        }
        if (entry == nullptr && splits.size() < ggml_backend_meta_context::SPLIT_CACHE_SIZE) {
            splits.emplace_back();
            entry = &splits.back();
        }
        if (entry == nullptr) {
            entry = &splits[0];
            for (auto & e : splits) {
                if (e.last_use < entry->last_use) {
                    entry = &e;
                }
            }
        }
        ggml_backend_meta_graph_split(backend, cgraph, *entry);
        entry->cgraph  = cgraph;
        entry->n_nodes = cgraph->n_nodes;
        entry->first   = cgraph->n_nodes > 0 ? cgraph->nodes[0] : nullptr;
        entry->last    = cgraph->n_nodes > 0 ? cgraph->nodes[cgraph->n_nodes - 1] : nullptr;
        entry->buf_epochs.clear();
        auto add_buf = [&](const ggml_tensor * t) {
            if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_meta(t->buffer)) {
                return;
            }
            for (const auto & be : entry->buf_epochs) {
                if (be.buffer == t->buffer) {
                    return;
                }
            }
            entry->buf_epochs.push_back({t, t->buffer, ((const ggml_backend_meta_buffer_context *) t->buffer->context)->epoch});
        };
        for (int i = 0; i < cgraph->n_nodes; i++) {
            add_buf(cgraph->nodes[i]);
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                add_buf(cgraph->nodes[i]->src[k]);
            }
        }
    }
    entry->last_use = ++backend_ctx->split_tick;
    const size_t n_subgraphs = entry->n_subgraphs;
    auto subgraph = [&](size_t j, size_t i) -> ggml_cgraph * {
        return entry->subgraphs[j*n_subgraphs + i];
    };

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };
    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        for (size_t offset_j = 1; offset_j < n_backends; offset_j *= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < n_backends; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other > j) {
                    continue;
                }

                auto & bcj1 = backend_ctx->backend_configs[j];
                auto & bcj2 = backend_ctx->backend_configs[j_other];

                ggml_tensor * node1 = subgraph(j, i)->nodes[subgraph(j, i)->n_nodes - 1];
                ggml_tensor * node2 = subgraph(j_other, i)->nodes[subgraph(j_other, i)->n_nodes - 1];
                GGML_ASSERT(ggml_is_contiguous(node1));
                GGML_ASSERT(ggml_is_contiguous(node2));

                // Tmp tensors to receive P2P copies
                ggml_tensor * node_tmp_1 = get_node_aux(node1);
                node_tmp_1->buffer = bcj1.buf.get();
                node_tmp_1->data = ggml_backend_buffer_get_base(bcj1.buf.get());

                ggml_tensor * node_tmp_2 = get_node_aux(node2);
                node_tmp_2->buffer = bcj2.buf.get();
                node_tmp_2->data = ggml_backend_buffer_get_base(bcj2.buf.get());

                // 2 P2P copies: exchange full buffers
                ggml_backend_tensor_copy_async(bcj1.backend, bcj2.backend, node1, node_tmp_2);
                ggml_backend_tensor_copy_async(bcj2.backend, bcj1.backend, node2, node_tmp_1);

                // Local ADD: node1 += tmp1 (in-place via view)
                ggml_tensor * node_red_1 = get_node_aux(node1);
                node_red_1->view_src = node1->view_src == nullptr ? node1 : node1->view_src;
                node_red_1->view_offs = node1->view_offs;
                node_red_1->op = GGML_OP_ADD;
                node_red_1->src[0] = node1;
                node_red_1->src[1] = node_tmp_1;
                node_red_1->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_red_1);

                // Local ADD: node2 += tmp2 (in-place via view)
                ggml_tensor * node_red_2 = get_node_aux(node2);
                node_red_2->view_src = node2->view_src == nullptr ? node2 : node2->view_src;
                node_red_2->view_offs = node2->view_offs;
                node_red_2->op = GGML_OP_ADD;
                node_red_2->src[0] = node2;
                node_red_2->src[1] = node_tmp_2;
                node_red_2->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_red_2);

                // Build 1-node cgraphs for the ADD ops
                ggml_cgraph * cgraph_aux_1 = get_cgraph_aux();
                cgraph_aux_1->nodes[0] = node_red_1;
                cgraph_aux_1->n_nodes = 1;
                step_cgraphs[j] = cgraph_aux_1;

                ggml_cgraph * cgraph_aux_2 = get_cgraph_aux();
                cgraph_aux_2->nodes[0] = node_red_2;
                cgraph_aux_2->n_nodes = 1;
                step_cgraphs[j_other] = cgraph_aux_2;
            }

            // Execute local ADDs for this step
            for (size_t j = 0; j < n_backends; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    };


    for (size_t i = 0; i < n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, subgraph(j, i));
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 && i < n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            ggml_backend_allreduce_tensor_t allreduce_tensor = (ggml_backend_allreduce_tensor_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ctx->backend_configs[0].backend)), "ggml_backend_allreduce_tensor");
            if (allreduce_tensor) {
                std::vector<ggml_backend_t> backends;
                backends.reserve(n_backends);
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    backends.push_back(bcj.backend);
                    ggml_cgraph * cgraph_ij = subgraph(j, i);
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = allreduce_tensor(backends.data(), nodes.data(), n_backends);
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
