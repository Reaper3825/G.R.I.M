#pragma once

#include <cstdint>

namespace GRIMText::Training {

struct TrainingContext;

struct ModelAllocationState {
    int model_max_tokens_per_batch = 0;
};

ModelAllocationState captureAndValidateModelAllocationOrThrow(const TrainingContext& ctx);
void ModelAllocated(TrainingContext& ctx);

} // namespace GRIMText::Training

