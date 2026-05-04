//======================================================//
//  GradientNormDiagnostic.hpp
//  Post-clipping gradient-norm logging plus
//  the [EMB_GRAD_EQUATION] embedding-spike diagnostic
//  (Issue #138, #139, #141, #150). The diagnostic consumes
//  the clipping path's already-measured ClipResult. It never
//  launches GradNorm kernels. The embedding equation branch is
//  explicitly sync-safe only under the gated diagnostic interval.
//======================================================//

#pragma once

#include "../../Shared/Gradients/GradientCC_GPU.hpp"

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct TrainingLoopState;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

// Logs gradient norm diagnostics from the ClipResult produced by the global
// clipping owner. This function only formats/validates/logs. It does not
// measure gradients, allocate scratch, or mutate BatchResult. The optional
// embedding equation path performs blocking host inspection only when
// shouldSyncDiagnostics() is true.
void runGradientNormClipDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::GradClip::ClipResult& clip,
    int batch_idx);

} // namespace GRIM::Diagnostics
