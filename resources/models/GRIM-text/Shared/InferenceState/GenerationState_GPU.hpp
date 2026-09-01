//======================================================//
//  GenerationState_GPU.hpp
//  Explicit owner for autoregressive inference/generation state
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Shared/InferenceState/LocalAtomRetrievalInferenceState.hpp"

namespace GRIM {

struct GenerationState {
    // Session-scoped state for autoregressive decode. Device buffers are
    // allocated lazily on first prefill and reused across decode steps;
    // resetSession() clears logical contents while retaining capacities.
    KvCacheState kv_cache;
    LocalAtomRetrievalInferenceState local_atom_retrieval;

    void resetSession() {
        kv_cache.resetSession();
        local_atom_retrieval.resetSession();
    }
};

} // namespace GRIM

#endif // USE_CUDA
