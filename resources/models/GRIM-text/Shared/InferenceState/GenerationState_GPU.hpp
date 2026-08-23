//======================================================//
//  GenerationState_GPU.hpp
//  Explicit owner for autoregressive inference/generation state
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"

namespace GRIM {

struct GenerationState {
    // Session-scoped KV cache for autoregressive decode. Buffers are allocated
    // lazily on first prefill and reused across decode steps; resetSession()
    // re-zeroes the fill counter (host side) while keeping the buffers.
    KvCacheState kv_cache;

    void resetSession() {
        kv_cache.resetSession();
    }
};

} // namespace GRIM

#endif // USE_CUDA
