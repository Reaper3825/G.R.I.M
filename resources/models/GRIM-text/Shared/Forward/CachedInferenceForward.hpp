#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "ModelForward_GPU.hpp"

namespace GRIM {
struct LocalAtomRetrievalInferenceState;
}

namespace GRIM::Batching {
struct BatchPayload;
struct BatchDeviceStorage;
}

namespace GRIM::Forward {

// Selects which results are copied to the CPU.
struct CachedInferenceReadback {
    bool last_logits = true;
    bool encoder_row = false;
    bool retrieval_logits = false;
};

// Host-owned outputs that survive GPU forward cleanup.
struct CachedInferenceResult {
    std::vector<float> last_logits;
    std::vector<float> encoder_row;
    std::vector<float> retrieval_logits;

    // Preserve existing forward capture-handle types.
    decltype(std::declval<ModelForwardOutputs&>().lens_snapshot)
        lens_snapshot{};

    decltype(std::declval<ModelForwardOutputs&>().lens_capture_result)
        lens_capture_result{};
};

// Execute one prefill or decode window against an
// already-initialized KV cache. The request supplies model and capture settings;
// this function supplies payload/bindings, enforces inference graph policy, and
// owns temporary forward outputs.
// A retrieval state enables prefill seeding and atom-open decode scoring.
// Requested readbacks must be produced; request retrieval logits only for a
// forward that emits them (atom-opening decode tokens).
CachedInferenceResult runCachedInferenceForward(
    ModelForwardRequest request,
    GRIM::Batching::BatchPayload& payload,
    std::shared_ptr<GRIM::Batching::BatchDeviceStorage>& reusable_storage,
    GRIM::LocalAtomRetrievalInferenceState* retrieval_state,
    CachedInferenceReadback readback = {});

} // namespace GRIM::Forward
