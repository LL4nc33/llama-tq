#pragma once

// Anthropic-compatible prompt-cache manager.
//
// Maintains a disk-backed, TTL-tracked index of KV-cache snapshots keyed by
// (model fingerprint, token prefix). Designed to sit on top of the existing
// llama.cpp slot save/restore API (`llama_state_seq_save_file` /
// `llama_state_seq_load_file`) so we do not duplicate any KV-serialization
// logic.
//
// Storage layout (under `<slot_save_path>/anthropic-cache/`):
//     <key_hex>.bin     — raw slot KV dump produced by llama_state_seq_save_file
//     <key_hex>.meta    — small JSON: { n_tokens, ttl_sec, created_at, scope, ttl_label }
//
// Thread-safety: the in-memory index is guarded by an internal mutex; the
// class is safe to share across request handlers.
//
// Design: docs/plans/2026-04-23-anthropic-prompt-caching-design.md §2.2-2.6

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h" // for llama_token

struct anthropic_cache_entry {
    std::string key_hex;      // 16-hex-char FNV-1a-based key
    int32_t     n_tokens = 0; // number of prompt tokens the snapshot covers
    int32_t     ttl_sec  = 300;
    int64_t     created_at = 0; // unix seconds
    std::string ttl_label;    // "5m" or "1h" for reporting
};

struct anthropic_cache_hit {
    std::string key_hex;
    std::string filepath;
    int32_t     n_tokens  = 0;
    int32_t     ttl_sec   = 300;
    std::string ttl_label;
};

class anthropic_cache_manager {
public:
    anthropic_cache_manager() = default;

    // Initialise the on-disk directory. `slot_save_path` must already include
    // a trailing separator (same convention as common_params::slot_save_path).
    // Returns true on success, false if the feature should be disabled.
    bool init(const std::string & slot_save_path,
              int32_t ttl_default_sec,
              int32_t max_gb);

    bool is_enabled() const { return enabled_; }

    // Derive a stable, versioned cache key from a model fingerprint and the
    // first `n_prefix` tokens. Uses FNV-1a 64 over (version || model_fp ||
    // n_prefix || token bytes). Collision risk is negligible for local caches.
    static std::string compute_key(const std::string           & model_fp,
                                   const std::vector<llama_token> & tokens,
                                   size_t                         n_prefix);

    // Resolve the on-disk `.bin` path for a given key. The caller is
    // responsible for passing this to llama_state_seq_*_file.
    std::string bin_path(const std::string & key_hex) const;
    std::string meta_path(const std::string & key_hex) const;

    // Look up a key and, if present, check TTL. Expired entries are deleted
    // and reported as a miss. On hit the `created_at` timestamp is refreshed
    // ("refresh-on-hit", mirroring Anthropic semantics).
    std::optional<anthropic_cache_hit> lookup(const std::string & key_hex);

    // Record a newly-created entry after a successful slot-save. Writes the
    // `.meta` file and updates the in-memory index.
    void record(const std::string & key_hex,
                int32_t             n_tokens,
                int32_t             ttl_sec,
                const std::string & ttl_label,
                const std::string & scope);

    // Directory holding the `.bin` / `.meta` files (may be empty when disabled).
    const std::string & dir() const { return dir_; }

    int32_t ttl_default_sec() const { return ttl_default_sec_; }

private:
    void seed_from_disk_unlocked();
    bool read_meta_unlocked(const std::string & key_hex, anthropic_cache_entry & out) const;
    void write_meta_unlocked(const anthropic_cache_entry & e, const std::string & scope) const;
    void erase_unlocked(const std::string & key_hex);

    mutable std::mutex mu_;
    bool        enabled_         = false;
    std::string dir_;                   // e.g. "/tmp/slots/anthropic-cache/"
    int32_t     ttl_default_sec_ = 300;
    int32_t     max_gb_          = 32;
    std::unordered_map<std::string, anthropic_cache_entry> index_;
};
