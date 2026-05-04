//======================================================//
//  GradientNormDiagnostic.hpp
//  Synced post-backward gradient-norm measurement plus
//  the [EMB_GRAD_EQUATION] embedding-spike diagnostic
//  (Issue #138, #139, #141, #150). The diagnostic reads
//  gradients/metrics and may throw, but result ownership
//  remains in Phase2_TrainingLoop.
//======================================================//

#pragma once

#include "../../Shared/GradNorm/GradNormGPU.hpp"

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct TrainingLoopState;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

// Snapshot of the per-component RMS values produced by
// measureGradientNorms. Returned to the caller so that the
// telemetry input struct can read preclip_grad_rms / enc_rms_pre
// without needing access to the raw GradMetrics buffer.
struct GradNormSnapshot {
    float grad_rms = 0.0f;          // total RMS across all parameter groups
    float preclip_grad_rms = 0.0f;  // alias of grad_rms (pre-clipping)
    float emb_rms_pre = 0.0f;
    float enc_rms_pre = 0.0f;
    float sb_rms_pre = 0.0f;
    GRIM::GradNorm::GradMetrics metrics{};  // raw per-component sums for telemetry
};

// Runs the grad-norm sync block:
//   1. validate caller-owned grad_norm_scratch and all registered grad buffers
//   2. measureGradientNormsLaunch + one required stream sync + Finalize
//   3. derive RMS from the same registered groups consumed by clipping
//   4. throw on empty/non-finite gradients (Rule 20)
//   5. emit gated [GradTrace] logs
//   6. run [EMB_GRAD_EQUATION] diagnostic on the shared sync interval
//
// Reads state.last_grad_rms only for the PRE-GRADNORM log line.
GradNormSnapshot runGradientNormDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    int batch_idx);

} // namespace GRIM::Diagnostics
