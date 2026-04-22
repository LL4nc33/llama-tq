#include "server-anthropic-cache.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

// Cache-key schema version. Bump this whenever the hashing recipe changes so
// that old on-disk entries are effectively invalidated.
static constexpr uint32_t ANTHROPIC_CACHE_KEY_VERSION = 1;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static int64_t now_unix_seconds() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

static std::string to_hex16(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return std::string(buf, 16);
}

// FNV-1a 64-bit. Matches the style already used in server-common.cpp for
// bitmap hashing; adequate for a local disk cache of <<1M entries.
static inline void fnv_update(uint64_t & hash, const uint8_t * data, size_t len) {
    static constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= FNV_PRIME;
    }
}

// ---------------------------------------------------------------------------
// anthropic_cache_manager
// ---------------------------------------------------------------------------

bool anthropic_cache_manager::init(const std::string & slot_save_path,
                                   int32_t              ttl_default_sec,
                                   int32_t              max_gb) {
    std::lock_guard<std::mutex> lk(mu_);
    enabled_ = false;
    if (slot_save_path.empty()) {
        return false;
    }

    // slot_save_path is expected to already end with a separator.
    dir_ = slot_save_path + "anthropic-cache/";
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
        fprintf(stderr, "anthropic-cache: failed to create directory '%s': %s\n",
                dir_.c_str(), ec.message().c_str());
        dir_.clear();
        return false;
    }

    ttl_default_sec_ = ttl_default_sec > 0 ? ttl_default_sec : 300;
    max_gb_          = max_gb > 0 ? max_gb : 32;

    seed_from_disk_unlocked();

    enabled_ = true;
    fprintf(stderr, "anthropic-cache: enabled (dir=%s, seeded=%zu, ttl_default=%ds, max=%dGiB)\n",
            dir_.c_str(), index_.size(), ttl_default_sec_, max_gb_);
    return true;
}

std::string anthropic_cache_manager::bin_path(const std::string & key_hex) const {
    return dir_ + key_hex + ".bin";
}

std::string anthropic_cache_manager::meta_path(const std::string & key_hex) const {
    return dir_ + key_hex + ".meta";
}

std::string anthropic_cache_manager::compute_key(const std::string              & model_fp,
                                                 const std::vector<llama_token> & tokens,
                                                 size_t                           n_prefix) {
    if (n_prefix > tokens.size()) n_prefix = tokens.size();

    uint64_t hi = 0xcbf29ce484222325ULL;
    uint64_t lo = 0x84222325cbf29ce4ULL; // second stream with different seed

    const uint32_t version = ANTHROPIC_CACHE_KEY_VERSION;
    fnv_update(hi, reinterpret_cast<const uint8_t*>(&version), sizeof(version));
    fnv_update(lo, reinterpret_cast<const uint8_t*>(&version), sizeof(version));

    fnv_update(hi, reinterpret_cast<const uint8_t*>(model_fp.data()), model_fp.size());
    fnv_update(lo, reinterpret_cast<const uint8_t*>(model_fp.data()), model_fp.size());

    const uint64_t n_prefix_u64 = static_cast<uint64_t>(n_prefix);
    fnv_update(hi, reinterpret_cast<const uint8_t*>(&n_prefix_u64), sizeof(n_prefix_u64));
    fnv_update(lo, reinterpret_cast<const uint8_t*>(&n_prefix_u64), sizeof(n_prefix_u64));

    if (n_prefix > 0) {
        const auto * tok_bytes = reinterpret_cast<const uint8_t*>(tokens.data());
        const size_t tok_len   = n_prefix * sizeof(llama_token);
        fnv_update(hi, tok_bytes,             tok_len);
        // Stir lo with reversed byte order so the two streams diverge quickly
        // even for degenerate token sequences (e.g. long runs of same id).
        for (size_t i = 0; i < tok_len; ++i) {
            uint8_t b = tok_bytes[tok_len - 1 - i];
            fnv_update(lo, &b, 1);
        }
    }

    return to_hex16(hi) + to_hex16(lo); // 32 hex chars
}

std::optional<anthropic_cache_hit> anthropic_cache_manager::lookup(const std::string & key_hex) {
    if (!enabled_) return std::nullopt;

    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(key_hex);
    if (it == index_.end()) {
        // Maybe we have a stale in-memory miss; try to re-read meta from disk.
        anthropic_cache_entry e;
        if (!read_meta_unlocked(key_hex, e)) {
            return std::nullopt;
        }
        index_[key_hex] = e;
        it = index_.find(key_hex);
    }

    const int64_t now = now_unix_seconds();
    const int64_t age = now - it->second.created_at;
    if (age > it->second.ttl_sec) {
        // Expired — lazy-delete both files and drop from index.
        erase_unlocked(key_hex);
        return std::nullopt;
    }

    // Refresh-on-hit: bump created_at and rewrite meta.
    it->second.created_at = now;
    write_meta_unlocked(it->second, /*scope=*/"");

    anthropic_cache_hit h;
    h.key_hex   = it->second.key_hex;
    h.filepath  = bin_path(key_hex);
    h.n_tokens  = it->second.n_tokens;
    h.ttl_sec   = it->second.ttl_sec;
    h.ttl_label = it->second.ttl_label;

    // Sanity: make sure the .bin file actually exists.
    std::error_code ec;
    if (!std::filesystem::exists(h.filepath, ec) || ec) {
        erase_unlocked(key_hex);
        return std::nullopt;
    }
    return h;
}

void anthropic_cache_manager::record(const std::string & key_hex,
                                     int32_t             n_tokens,
                                     int32_t             ttl_sec,
                                     const std::string & ttl_label,
                                     const std::string & scope) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(mu_);
    anthropic_cache_entry e;
    e.key_hex    = key_hex;
    e.n_tokens   = n_tokens;
    e.ttl_sec    = ttl_sec > 0 ? ttl_sec : ttl_default_sec_;
    e.ttl_label  = ttl_label.empty() ? std::string("5m") : ttl_label;
    e.created_at = now_unix_seconds();
    index_[key_hex] = e;
    write_meta_unlocked(e, scope);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void anthropic_cache_manager::seed_from_disk_unlocked() {
    index_.clear();
    std::error_code ec;
    std::filesystem::directory_iterator it(dir_, ec);
    if (ec) return;
    for (auto & entry : it) {
        if (!entry.is_regular_file()) continue;
        const auto & p = entry.path();
        if (p.extension() != ".meta") continue;
        std::string key = p.stem().string();
        anthropic_cache_entry e;
        if (read_meta_unlocked(key, e)) {
            index_[key] = e;
        }
    }
}

bool anthropic_cache_manager::read_meta_unlocked(const std::string & key_hex,
                                                 anthropic_cache_entry & out) const {
    const std::string path = meta_path(key_hex);
    std::ifstream ifs(path);
    if (!ifs.good()) return false;
    std::stringstream ss;
    ss << ifs.rdbuf();
    try {
        auto j = nlohmann::json::parse(ss.str());
        out.key_hex    = key_hex;
        out.n_tokens   = j.value("n_tokens",   0);
        out.ttl_sec    = j.value("ttl_sec",    ttl_default_sec_);
        out.created_at = j.value("created_at", now_unix_seconds());
        out.ttl_label  = j.value("ttl_label",  std::string("5m"));
        return true;
    } catch (const std::exception & ex) {
        fprintf(stderr, "anthropic-cache: failed to parse %s: %s\n",
                path.c_str(), ex.what());
        return false;
    }
}

void anthropic_cache_manager::write_meta_unlocked(const anthropic_cache_entry & e,
                                                  const std::string & scope) const {
    nlohmann::json j;
    j["n_tokens"]   = e.n_tokens;
    j["ttl_sec"]    = e.ttl_sec;
    j["ttl_label"]  = e.ttl_label;
    j["created_at"] = e.created_at;
    if (!scope.empty()) j["scope"] = scope;
    std::ofstream ofs(meta_path(e.key_hex), std::ios::trunc);
    ofs << j.dump();
}

void anthropic_cache_manager::erase_unlocked(const std::string & key_hex) {
    std::error_code ec;
    std::filesystem::remove(bin_path(key_hex),  ec);
    std::filesystem::remove(meta_path(key_hex), ec);
    index_.erase(key_hex);
}
