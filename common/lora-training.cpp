#include "lora-training.h"

#include "llama.h"

llama_adapter_lora * lora_training_init(
        llama_model * model,
        const lora_training_config & cfg) {
    return llama_adapter_lora_init_for_training(
            model, cfg.target_regex.c_str(), cfg.rank, cfg.alpha);
}
