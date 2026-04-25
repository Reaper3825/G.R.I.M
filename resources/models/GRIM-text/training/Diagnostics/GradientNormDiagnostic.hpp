//======================================================//
//  GradientNormDiagnostic.hpp
//  Synced post-backward gradient-norm measurement plus
//  the [EMB_GRAD_EQUATION] embedding-spike diagnostic
//  (Issue #138, #139, #141, #150). Lifted verbatim from
//  Phase2_TrainingLoop.cu processBatch.
//======================================================//

#pragma once

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
};

// Runs the full grad-norm sync block:
//   1. PRE-MEASURE [GRAD_DIAG] sample (LM head pointer match check)
//   2. lazy-allocate grad_norm_scratch
//   3. measureGradientNormsLaunch + Finalize (with CUDA event timing)
//   4. derive per-component RMS values
//   5. throw on NaN/Inf (Rule 20)
//   6. emit [GradTrace] COMPONENTS / POST-GRADNORM logs
//   7. run [EMB_GRAD_EQUATION] diagnostic on diag interval
//
// Side effects on result: sets result.grad_rms and result.normalized_grad_rms.
// Reads state.last_grad_rms only for the PRE-GRADNORM log line.
GradNormSnapshot runGradientNormDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIMText::Training::TrainingLoopState& state,
    GRIMText::Training::BatchResult& result,
    int batch_idx);

} // namespace GRIM::Diagnostics
