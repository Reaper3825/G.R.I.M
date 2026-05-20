//======================================================//
//  GradientNormDiagnostic.hpp
//  Post-clipping gradient-norm logging (Issue #138, #139, #150).
//  Consumes the clipping path's already-measured ClipResult and never
//  launches GradNorm kernels. Post-clip parameter-gradient equation logging
//  is delegated to PostClipParamGradEmbLmEquation.
//======================================================//

#pragma once

#include "../../Shared/Gradients/GradientCC_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct TrainingLoopState;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

// Logs gradient norm diagnostics from the ClipResult produced by the global
// clipping owner. This function does not measure gradients, allocate GradNorm
// scratch, or mutate BatchResult. The delegated post-clip embedding/LM equation
// path consumes clip_stream and performs blocking host inspection only when
// shouldSyncDiagnostics() plus the debug tape gate are both enabled.
void runGradientNormClipDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::GradClip::ClipResult& clip,
    int batch_idx,
    cudaStream_t clip_stream);

} // namespace GRIM::Diagnostics
