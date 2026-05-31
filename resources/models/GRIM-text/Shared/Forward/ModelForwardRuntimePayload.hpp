//======================================================//
//  ModelForwardRuntimePayload.hpp
//
//  Explicit mutable runtime payload for one shared forward call.
//  Mirrors the boundary style used by BatchPayload / BatchDeviceBindings:
//  - model forward receives this payload explicitly,
//  - the payload owns no storage,
//  - the underlying owners remain the caller-owned execution runtime /
//    read-gate workspace for the current call.
//
//  Rule 20: this payload is the ONLY runtime sink surface visible to
//  Shared/Forward/ModelForward_GPU.*. Shared forward must not rediscover
//  mutable runtime owners by reaching through TrainingState or any other
//  god object.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <string>

#include "ModelForwardExecutionRuntime.hpp"
#include "ModelForwardOutputs.hpp"

namespace GRIM {
namespace Forward {

struct ModelForwardRuntimePayload {
    ModelForwardExecutionRuntime* execution_runtime = nullptr;
    Tensor* read_gate_accum_tensor = nullptr;

    void validate(const char* caller, bool execution_block_active) const {
        if (execution_block_active) {
            if (!execution_runtime) {
                throw std::runtime_error(std::string(caller) + ": runtime payload execution_runtime is NULL");
            }
        }
    }
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
