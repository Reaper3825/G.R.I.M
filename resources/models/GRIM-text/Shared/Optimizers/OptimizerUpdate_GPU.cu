#ifndef USE_CUDA
#define USE_CUDA
#endif
//======================================================//
//  OptimizerUpdate_GPU.cu
//  Optimizer Window configured update dispatch boundary
//======================================================//

#include "OptimizerUpdate_GPU.hpp"

#include "AdamW/AdamW_Kernal_GPU.hpp"
#include "RAdamW/RAdamW_Kernal_GPU.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM {

void launchOptimizerUpdate(std::vector<ParameterGroup>& groups,
                           const HyperParameters::OptimizerUpdateHP& hp,
                           float learning_rate,
                           int step,
                           cudaStream_t stream) {
    if (!std::isfinite(learning_rate) || learning_rate < 0.0f) {
        throw std::runtime_error("[launchOptimizerUpdate] learning_rate must be finite and >= 0, got " +
                                 std::to_string(learning_rate));
    }
    if (step < 0) {
        throw std::runtime_error("[launchOptimizerUpdate] step must be >= 0, got " +
                                 std::to_string(step));
    }
    if (stream == nullptr) {
        throw std::runtime_error("[launchOptimizerUpdate] stream is NULL - caller MUST provide valid CUDA stream");
    }

    switch (hp.kind) {
        case HyperParameters::OptimizerKind::ADAMW:
            launchAdamWStep(groups,
                            learning_rate,
                            hp.weight_decay,
                            step,
                            stream,
                            hp.embedding_freeze_after_step);
            return;
        case HyperParameters::OptimizerKind::RADAMW:
            launchRAdamWStep(groups,
                             learning_rate,
                             hp.weight_decay,
                             step,
                             hp.beta1,
                             hp.beta2,
                             hp.epsilon,
                             stream,
                             hp.embedding_freeze_after_step);
            return;
    }

    throw std::runtime_error("[launchOptimizerUpdate] unknown optimizer kind enum value");
}

} // namespace GRIM
