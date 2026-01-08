#pragma once
#include <cstdint>

namespace GRIMText::TrainingEvents {

struct OptimizerStepEvent {
    int epoch = 0;
    int batch_index = 0;
    int total_batches = 0;
    int global_step = 0;
    int effective_batch_size = 0;
    float batch_loss = 0.0f;
    float learning_rate = 0.0f;
};

struct ValidationEvent {
    int epoch = 0;
    float loss = 0.0f;
    float perplexity = 0.0f;
    int sequences = 0;
};

struct MicroValidationEvent {
    int global_step = 0;
    int batches = 0;
    int sequences = 0;
    float loss = 0.0f;
    float perplexity = 0.0f;
    float learning_rate = 0.0f;
    float duration_ms = 0.0f;
    float cache_fill_ratio = 0.0f;
};

// NOTE: Event delegates removed per Rule 20 - no code subscribed to these.
// Re-add if/when actual subscribers are implemented.

} // namespace GRIMText::TrainingEvents
