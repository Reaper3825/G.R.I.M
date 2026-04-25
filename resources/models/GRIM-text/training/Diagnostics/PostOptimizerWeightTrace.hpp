//======================================================//
//  PostOptimizerWeightTrace.hpp
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  post-optimizer LM-head weight sample, GradTrace
//  POST-OPTIMIZER log line, [UpdateMag] emit, and the
//  per-component Adam update_rms trace (Issue #150).
//
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//
//  Caller still owns capturing `pre_sample` before the
//  optimizer step (cheaper than re-sampling here).
//======================================================//

#pragma once

#include "TrainingDiagnostics.hpp"      // WeightSample, formatWeightSample, computePerComponentUpdateTrace

namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

void runPostOptimizerWeightTrace(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    const WeightSample& pre_sample,
    int batch_idx,
    bool sync_diag);

} // namespace GRIM::Diagnostics
