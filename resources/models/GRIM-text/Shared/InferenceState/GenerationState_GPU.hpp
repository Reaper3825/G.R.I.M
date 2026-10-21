//======================================================//
//  GenerationState_GPU.hpp
//  Explicit owner for autoregressive inference/generation state
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Shared/InferenceState/LocalAtomRetrievalInferenceState.hpp"
#include "InferenceGenerationSession.hpp"
#include "../Batching/BatchPayload.hpp"

#include <optional>

namespace GRIM {
namespace Tokenizer { class UniByte; }

struct GenerationState {
    // Session-scoped state for autoregressive decode. Device buffers are
    // allocated lazily on first prefill and reused across decode steps;
    // resetSession() clears logical contents while retaining capacities.
    KvCacheState kv_cache;
    LocalAtomRetrievalInferenceState local_atom_retrieval;
    // Retained across span returns. kv_cache.host_seqlen is the authoritative
    // processed-token cursor; the newest committed token can still be pending.
    std::optional<Batching::BatchPayload> prefill;
    std::optional<Inference::InferenceGenerationSession> active_session;
    std::shared_ptr<Batching::BatchDeviceStorage> inference_device_storage;
    const Tokenizer::UniByte* tokenizer = nullptr;

    void resetSession() {
        kv_cache.resetSession();
        local_atom_retrieval.resetSession();
        active_session.reset();
        prefill.reset();
        inference_device_storage.reset();
        tokenizer = nullptr;
    }
};

} // namespace GRIM

#endif // USE_CUDA
