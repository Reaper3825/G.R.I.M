#pragma once

#include <cstdint>

namespace GRIMText::Training {

struct TrainingContext;

struct ModelAllocationState {
    int model_max_cached_batch = 0;
    std::uint32_t model_max_cached_seq_len = 0;
    int model_max_tokens_per_batch = 0;
};

ModelAllocationState captureAndValidateModelAllocationOrThrow(const TrainingContext& ctx);
void ModelAllocated(TrainingContext& ctx);

} // namespace GRIMText::Training

