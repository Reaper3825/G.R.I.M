#pragma once

#include <cstddef>

namespace GRIM {

class LanguageModel;

struct ModelStats {
    size_t total_params = 0;
    size_t embedding_params = 0;
    size_t encoder_params = 0;
    size_t lm_head_params = 0;
    size_t scratchblock_params = 0;  // Atom type embeddings + projection
    float model_size_mb = 0.0f;
};

ModelStats computeModelStats(const LanguageModel& model);

} // namespace GRIM