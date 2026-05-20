//======================================================//
//  PostOptimizerWeightTrace.hpp
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  post-optimizer LM-head weight sample, GradTrace
//  POST-OPTIMIZER log line, [UpdateMag] emit, and the
//  optimizer-boundary adaptive update trace.
//
//  Update trace is an optimizer operation: it runs after the optimizer
//  window update and reads live ParameterGroup moment buffers. It does not
//  cache batch/autograd state.
//
//  Caller still owns capturing `pre_sample` before the
//  optimizer step (cheaper than re-sampling here).
//======================================================//

#pragma once

#include "TrainingDiagnostics.hpp"      // WeightSample, formatWeightSample
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

void runPostOptimizerWeightTrace(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    const GRIM::HyperParameters::OptimizerUpdateHP& optimizer_hp,
    const WeightSample& pre_sample,
    bool sync_diag);

} // namespace GRIM::Diagnostics
