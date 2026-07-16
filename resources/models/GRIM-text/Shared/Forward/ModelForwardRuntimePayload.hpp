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

    // Optional single-row register file owned by the caller's generation
    // session. Shared forward borrows it only for downstream cross-attention
    // readback; it never bootstraps, clears, or executes steps against it.
    // Cross-attention may update ExecutionMemory::usage telemetry, but semantic
    // register values and validity remain unchanged.
    ExecutionMemory* persistent_execution_memory = nullptr;
    bool persistent_execution_memory_was_read = false;

    void validate(const char* caller, bool execution_block_active) const {
        if (execution_block_active) {
            if (!execution_runtime) {
                throw std::runtime_error(std::string(caller) + ": runtime payload execution_runtime is NULL");
            }
        } else if (persistent_execution_memory) {
            throw std::runtime_error(
                std::string(caller) +
                ": persistent_execution_memory supplied while execution block is inactive");
        }
        if (persistent_execution_memory) {
            const auto& memory = *persistent_execution_memory;
            if (!memory.values.data || !memory.valid_mask.data ||
                !memory.usage.data || !memory.key_embeds.data ||
                !memory.state_embeds.data) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": persistent_execution_memory is missing required register/readback tensors");
            }
        }
    }
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
