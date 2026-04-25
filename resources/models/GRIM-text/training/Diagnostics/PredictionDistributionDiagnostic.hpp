//======================================================//
//  PredictionDistributionDiagnostic.hpp
//  Two diagnostics:
//   - runTargetDistributionLog: BATCH_TARGET_DIST log
//     of the per-batch target-token histogram (top-10).
//   - runPredictionDistributionAndLogitTrace: post-loss
//     argmax distribution over cached logits + the
//     per-position [LogitTrace][PostLoss] equation.
//  Both lifted verbatim from Phase2_TrainingLoop.cu.
//======================================================//

#pragma once

namespace GRIM { namespace Batching { struct BatchPayload; } }
namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runTargetDistributionLog(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

void runPredictionDistributionAndLogitTrace(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx);

} // namespace GRIM::Diagnostics
