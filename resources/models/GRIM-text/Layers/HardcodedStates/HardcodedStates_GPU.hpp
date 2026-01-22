#pragma once

#include <cuda_runtime.h>

namespace GRIM {

// Hardcoded Hidden States Diagnostic Pattern (Issue #42)
// MUST match the enum in grim_language_model_cuda.hpp line ~292-298
enum class HardcodedPattern {
    DISABLED = 0,
    RANDOM_CENTERED = 1,
    ORTHOGONAL_W277 = 2,
    ALIGNED_W277 = 3,
    CONSTANT_UNIFORM = 4,
    ZERO_MEAN_SINE = 5
};

/**
 * @brief Generate hardcoded hidden state patterns for diagnostic testing (Issue #42)
 * 
 * Replaces encoder output with synthetic patterns to isolate whether mode collapse
 * originates from the encoder or from the LM head/gradient system.
 */
void generateHardcodedStates(
    float* output,                                      // [total_tokens, d_model] - destination
    const float* lm_head_weights,                       // [vocab_size, d_model] - for W[277] patterns
    HardcodedPattern pattern,                           // Which pattern to generate
    int total_tokens,
    int d_model,
    int vocab_size,
    int batch_idx,                                      // For logging/seeding
    cudaStream_t stream
);

/**
 * @brief Log diagnostic info about hardcoded states and resulting logits
 */
void logHardcodedStateDiagnostics(
    const float* hidden_states,                         // [total_tokens, d_model]
    const float* lm_head_weights,                       // [vocab_size, d_model]
    const float* logits,                                // [total_tokens, vocab_size]
    HardcodedPattern pattern,
    int total_tokens,
    int d_model,
    int vocab_size,
    int batch_idx,
    cudaStream_t stream
);

} // namespace GRIM
