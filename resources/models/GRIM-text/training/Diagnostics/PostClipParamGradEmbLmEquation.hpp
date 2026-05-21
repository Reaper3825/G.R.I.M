//======================================================//
//  PostClipParamGradEmbLmEquation.hpp
//  Single public operation for the post-clip embedding/LM
//  parameter-gradient equation diagnostic.
//======================================================//

#pragma once

#include <cuda_runtime.h>

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText::Training {
    struct TrainingContext;
    struct TrainingLoopState;
}

namespace GRIM::Diagnostics {

inline constexpr const char* kPostClipParamGradEmbLmEquationOp =
    "POSTCLIP_PARAM_GRAD_EMB_LM_EQUATION";

// Emits POSTCLIP_PARAM_GRAD_EMB_LM_EQUATION when sync diagnostics and the
// debug tape gate both allow it. This is a post-clipping parameter-gradient
// diagnostic, not an EmbeddingGradFn hook: it inspects the completed registered
// LM-head and embedding gradient buffers (shared when tied, aggregated host-side
// when untied) and updates diagnostic trend state.
void runPostClipParamGradEmbLmEquation(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    float emb_rms_pre,
    int batch_idx,
    bool sync_diag,
    cudaStream_t stream);

} // namespace GRIM::Diagnostics