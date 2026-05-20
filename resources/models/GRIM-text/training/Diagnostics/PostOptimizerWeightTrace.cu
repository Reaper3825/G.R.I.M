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

namespace GRIM::Diagnostics {

void runPostOptimizerWeightTrace(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    const GRIM::HyperParameters::OptimizerUpdateHP& optimizer_hp,
    const WeightSample& pre_sample,
    bool sync_diag)
{
    namespace Internal = ::GRIMText::Training::Internal;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;
    const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_step.step);
    const int iteration = optimizer_step + 1;

    WeightSample post_sample{};
    std::string post_weights = "lm_head_weights=skipped";
    if (sync_diag) {
        post_sample = sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
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
            ctx.model->parameterGroups(),
            optimizer_hp,
            result.learning_rate,
            optimizer_step,
            ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()
        );
        if (update_trace.valid) {
            const auto trace_lines = formatOptimizerUpdateTraceLines(
                update_trace,
                optimizer_step,
                ctx.model_config.tie_embeddings);
            for (const auto& trace_line : trace_lines) {
                ctx.logging.logger->log(trace_line);
            }
        }
    }
}

} // namespace GRIM::Diagnostics
