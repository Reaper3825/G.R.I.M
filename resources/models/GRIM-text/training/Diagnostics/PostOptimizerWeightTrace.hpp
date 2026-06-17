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

#include "../Phases/Startup/Model/ParameterRegistry.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"

#include <string>
#include <vector>

namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

//======================================================//
//  WeightSample — small LM-head weight snapshot for pre/post optimizer delta
//======================================================//

constexpr int kWeightSampleSize = 10;

struct WeightSample {
    bool valid = false;
    float values[kWeightSampleSize] = {0.0f};
    float rms = 0.0f;
};

WeightSample sampleWeightStats(const GRIM::Tensor& lm_head_weights, const GRIM::TrainingState& ts, bool sync_for_host = false);
std::string formatWeightSample(const WeightSample& sample);
float computeUpdateRms(const WeightSample& before, const WeightSample& after);

void runPostOptimizerWeightTrace(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::TrainingState& training_state,
    const std::vector<GRIM::ParameterGroup>& parameter_groups,
    const GRIM::HyperParameters::OptimizerUpdateHP& optimizer_hp,
    const WeightSample& pre_sample,
    bool sync_diag);

} // namespace GRIM::Diagnostics
