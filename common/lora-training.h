#pragma once

// LoRA training adapter bootstrapping for llama-finetune.
//
// Instead of loading A and B from a pre-trained .gguf adapter file, this
// initialises fresh A (gauss / sqrt(rank)) and B (zeros) adapter tensors for
// every base-model tensor whose name matches a user-supplied regex. The
// adapter is registered with the model so the existing forward graph builders
// (`build_lora_mm` / `build_lora_mm_id`) pick it up automatically, and the
// A/B tensors are marked as trainable so the autodiff includes them in the
// backward pass.
//
// Frozen base-model weights remain frozen — only A and B receive gradients,
// which keeps the trainable parameter count to ~r·(in+out) per target tensor.

#include "llama.h"

#include <string>

struct llama_adapter_lora;

struct lora_training_config {
    std::string target_regex;     // tensors whose names match get LoRA adapters
    int         rank   = 16;      // rank of the A/B decomposition
    float       alpha  = 32.0f;   // effective scale = alpha / rank
};

// Build a fresh trainable LoRA adapter for the model. The returned adapter
// is owned by the model and freed via the normal llama_adapter_lora_free
// path on shutdown.
//
// Returns nullptr if no tensor matched the regex.
llama_adapter_lora * lora_training_init(
        llama_model * model,
        const lora_training_config & cfg);
