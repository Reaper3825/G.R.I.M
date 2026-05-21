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

#include "../../GRIM/grim_language_model_cuda.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }
namespace GRIM { struct Tensor; }

namespace GRIM::Diagnostics {

void runTargetDistributionLog(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

void runPredictionDistributionAndLogitTrace(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Tensor& logits_tensor,
    float loss,
    int batch_idx);

} // namespace GRIM::Diagnostics
