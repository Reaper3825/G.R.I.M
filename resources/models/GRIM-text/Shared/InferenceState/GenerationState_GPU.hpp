//======================================================//
//  GenerationState_GPU.hpp
//  Explicit owner for autoregressive inference/generation state
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../Shared/Forward/ModelForwardExecutionRuntime.hpp"
#include "../../Shared/Execution/DecodeTimeResolveResult.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

namespace GRIM {

struct GenerationState {
    // Persistent inference execution state. Survives prefill -> decode steps
    // within a generation session and is invalidated only at session reset.
    ExecutionMemory exec_memory;
    bool has_exec_memory = false;

    // Decode-time ExecutionBlock trace state for autoregressive generation.
    // Training forward traces remain TrainingState-owned; these are session state.
    Forward::ModelForwardExecutionRuntime execution_runtime;

    // Decode-time <NUM> selector result consumed by sampling.
    DecodeTimeResolveResult decode_selector;

    void resetSession() {
        has_exec_memory = false;
        execution_runtime.clear();
        decode_selector.reset();
    }
};

} // namespace GRIM

#endif // USE_CUDA
