#pragma once

#include <cuda_runtime.h>

#include "../TensorContract/TensorContract_GPU.hpp"

#include <vector>

namespace GRIM {

struct MeanPoolSequenceSpan {
    int sequence_index = 0;
    int token_begin = 0; // inclusive, local to the sequence
    int token_end = 0;   // exclusive, local to the sequence
};

// Generic token-wise mean pooling. Span selection belongs to the caller; this
// operation has no knowledge of prompts, responses, goals, or model roles.
Tensor meanPoolHiddenStates(
    const Tensor& hidden_states,
    int tokens_per_sequence,
    const std::vector<MeanPoolSequenceSpan>& spans,
    cudaStream_t stream);

} // namespace GRIM
