//======================================================//
//  ModelForwardRuntimePayload.hpp
//
//  Explicit mutable runtime payload for one shared forward call.
//  Mirrors the boundary style used by BatchPayload / BatchDeviceBindings:
//  - model forward receives this payload explicitly,
//  - the payload owns no storage,
//  - the underlying owners remain TrainingState / GenerationState /
//    AutogradIntermediates depending on the caller.
//
//  Rule 20: this payload is the ONLY runtime sink surface visible to
//  Shared/Forward/ModelForward_GPU.*. Shared forward must not rediscover
//  mutable runtime owners by reaching through TrainingState or any other
//  god object.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <string>
#include <vector>

#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace Autograd {
struct AutogradIntermediates;
}

namespace Forward {

struct ModelForwardRuntimePayload {
    Autograd::AutogradIntermediates* autograd_intermediates = nullptr;
    std::vector<std::vector<ExecutionRecord>>* execution_trace_by_row = nullptr;
    std::vector<Tensor>* trace_state_by_row = nullptr;
    Tensor* read_gate_accum_tensor = nullptr;

    void validate(const char* caller, bool execution_block_active) const {
        if (!autograd_intermediates) {
            throw std::runtime_error(std::string(caller) + ": runtime payload autograd_intermediates is NULL");
        }
        if (execution_block_active) {
            if (!execution_trace_by_row) {
                throw std::runtime_error(std::string(caller) + ": runtime payload execution_trace_by_row is NULL");
            }
            if (!trace_state_by_row) {
                throw std::runtime_error(std::string(caller) + ": runtime payload trace_state_by_row is NULL");
            }
        }
    }
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
