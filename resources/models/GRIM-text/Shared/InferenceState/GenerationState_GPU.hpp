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
#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

namespace GRIM {

struct GenerationState {
    // Persistent inference execution state. Survives prefill -> decode steps
    // within a generation session and is invalidated only at session reset.
    Forward::ExecutionMemoryOwnedStorage exec_memory_storage;
    ExecutionMemory exec_memory;
    bool has_exec_memory = false;

    // Decode-time ExecutionBlock trace state for autoregressive generation.
    // Training forward traces remain TrainingState-owned; these are session state.
    Forward::ModelForwardExecutionRuntime execution_runtime;

    // Session-scoped KV cache for autoregressive decode. Buffers are allocated
    // lazily on first prefill and reused across decode steps; resetSession()
    // re-zeroes the fill counter (host side) while keeping the buffers.
    KvCacheState kv_cache;

    void resetSession() {
        exec_memory = ExecutionMemory();
        exec_memory_storage = Forward::ExecutionMemoryOwnedStorage();
        has_exec_memory = false;
        execution_runtime.clear();
        kv_cache.resetSession();
    }
};

} // namespace GRIM

#endif // USE_CUDA
