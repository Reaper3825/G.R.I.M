#pragma once

#include <cstddef>

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM {

struct ModelStats {
    size_t total_params = 0;
    size_t embedding_params = 0;
    size_t encoder_params = 0;
    size_t lm_head_params = 0;
    size_t scratchblock_params = 0;  // Atom type embeddings + projection
    float model_size_mb = 0.0f;
};

ModelStats computeModelStats(const ::ParameterRegistry::StartupParameterRegistry& parameter_registry);

} // namespace GRIM