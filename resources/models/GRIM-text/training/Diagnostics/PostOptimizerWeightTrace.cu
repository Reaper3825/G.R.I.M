//======================================================//
//  PostOptimizerWeightTrace.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  post-optimizer LM-head weight sample, GradTrace
//  POST-OPTIMIZER log line, [UpdateMag] emit, and the
//  optimizer-boundary adaptive update trace.
//
//  Update trace is an optimizer operation: it runs after the optimizer
//  window update and reads live ParameterGroup moment buffers. It does not
//  cache batch/autograd state.
//======================================================//

#include "PostOptimizerWeightTrace.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Optimizers/OptimizerUpdateTrace.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"  // EmitModuleInfo, ModuleId

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace GRIM::Diagnostics {

WeightSample sampleWeightStats(const GRIM::Tensor& lm_head_weights, const GRIM::TrainingState& ts, bool sync_for_host) {
    WeightSample sample{};
    if (!lm_head_weights.data) {
        return sample;
    }

    if (!sync_for_host) {
        return sample;
    }

    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    cudaMemcpyAsync(sample.values, lm_head_weights.data,
                    kWeightSampleSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float sum_sq = 0.0f;
    for (int i = 0; i < kWeightSampleSize; ++i) {
        sum_sq += sample.values[i] * sample.values[i];
    }
    sample.rms = std::sqrt(sum_sq / kWeightSampleSize);
    sample.valid = true;
    return sample;
}

std::string formatWeightSample(const WeightSample& sample) {
    if (!sample.valid) {
        return "lm_head_weights=nullptr";
    }

    std::ostringstream oss;
    oss << "lm_w[0:10]=[";
    for (int i = 0; i < kWeightSampleSize; ++i) {
        oss << std::fixed << std::setprecision(6) << sample.values[i];
        if (i + 1 < kWeightSampleSize) oss << ",";
    }
    oss << "] rms=" << std::scientific << std::setprecision(4) << sample.rms;
    return oss.str();
}

float computeUpdateRms(const WeightSample& before, const WeightSample& after) {
    if (!before.valid || !after.valid) {
        return 0.0f;
    }

    float sum_sq = 0.0f;
    for (int i = 0; i < kWeightSampleSize; ++i) {
        const float delta = after.values[i] - before.values[i];
        sum_sq += delta * delta;
    }
    return std::sqrt(sum_sq / kWeightSampleSize);
}

void runPostOptimizerWeightTrace(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::TrainingState& training_state,
    const std::vector<GRIM::ParameterGroup>& parameter_groups,
    const GRIM::HyperParameters::OptimizerUpdateHP& optimizer_hp,
    const WeightSample& pre_sample,
    bool sync_diag)
{
    namespace Internal = ::GRIMText::Training::Internal;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;
    const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_step.step);
    const int iteration = optimizer_step + 1;
    const GRIM::Tensor& lm_head_weights =
        parameter_registry.requireLmHeadParameters("runPostOptimizerWeightTrace").weights;

    WeightSample post_sample{};
    std::string post_weights = "lm_head_weights=skipped";
    if (sync_diag) {
        post_sample = sampleWeightStats(lm_head_weights, training_state, true);
        if (post_sample.valid) {
            post_weights = formatWeightSample(post_sample);
        }
    }
    ctx.logging.logger->log("[GradTrace] POST-OPTIMIZER optimizer_step=" + std::to_string(optimizer_step) +
                            " iteration=" + std::to_string(iteration) +
                            " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                            " " + post_weights);

    if (pre_sample.valid && post_sample.valid) {
        const float update_rms = computeUpdateRms(pre_sample, post_sample);
        const std::string update_msg = "[UpdateMag] optimizer_step=" + std::to_string(optimizer_step) +
                                       " iteration=" + std::to_string(iteration) +
                                       " update_rms=" + Internal::formatScalar(update_rms, 8) +
                                       " param_rms=" + Internal::formatScalar(pre_sample.rms, 8);
        ctx.logging.logger->log(update_msg);
        EmitModuleInfo(ModuleId::Optimizer, update_msg, ctx.global_step);
    }

    // Optimizer-boundary adaptive update diagnostic. This is not a batch or
    // TensorContract grad-flow op: it samples live optimizer moment buffers
    // after launchOptimizerUpdate() and owns no cached state.
    if (sync_diag) {
        const auto update_trace = computeOptimizerUpdateTrace(
            parameter_groups,
            optimizer_hp,
            result.learning_rate,
            optimizer_step,
            training_state.stream_ctrl.getPrimaryStream()
        );
        if (update_trace.valid) {
            const auto trace_lines = formatOptimizerUpdateTraceLines(
                update_trace,
                optimizer_step,
                GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings"));
            for (const auto& trace_line : trace_lines) {
                ctx.logging.logger->log(trace_line);
            }
        }
    }
}

} // namespace GRIM::Diagnostics
