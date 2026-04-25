//======================================================//
//  PostOptimizerWeightTrace.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  post-optimizer LM-head weight sample, GradTrace
//  POST-OPTIMIZER log line, [UpdateMag] emit, and the
//  per-component Adam update_rms trace (Issue #150).
//
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "PostOptimizerWeightTrace.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"  // EmitModuleInfo, ModuleId

#include <string>

namespace GRIM::Diagnostics {

void runPostOptimizerWeightTrace(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    const WeightSample& pre_sample,
    int batch_idx,
    bool sync_diag)
{
    namespace Internal = ::GRIMText::Training::Internal;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    WeightSample post_sample{};
    std::string post_weights = "lm_head_weights=skipped";
    if (sync_diag) {
        post_sample = sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
        if (post_sample.valid) {
            post_weights = formatWeightSample(post_sample);
        }
    }
    ctx.logging.logger->log("[GradTrace] POST-OPTIMIZER batch=" + std::to_string(batch_idx + 1) +
                            " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                            " step=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                            " t=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                            " " + post_weights);

    if (pre_sample.valid && post_sample.valid) {
        const float update_rms = computeUpdateRms(pre_sample, post_sample);
        const std::string update_msg = "[UpdateMag] batch=" + std::to_string(batch_idx + 1) +
                                       " update_rms=" + Internal::formatScalar(update_rms, 8) +
                                       " param_rms=" + Internal::formatScalar(pre_sample.rms, 8);
        ctx.logging.logger->log(update_msg);
        EmitModuleInfo(ModuleId::Optimizer, update_msg, ctx.global_step);
    }

    // Per-component Adam update_rms diagnostic (Issue #150)
    // Answers: "Does Adam normalize the gradient gap across component types?"
    // Only on diagnostic-sync batches to avoid blocking the pipeline.
    if (sync_diag) {
        const auto update_trace = computePerComponentUpdateTrace(
            ctx.model->parameterGroups(),
            result.learning_rate,
            ctx.optimizer.optimizer_state.step + 1,  // 1-based iteration count (matches AdamW bias correction)
            ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()
        );
        if (update_trace.valid) {
            const std::string trace_str = formatUpdateTrace(
                update_trace, batch_idx + 1, ctx.model->getConfig().tie_embeddings);
            ctx.logging.logger->log(trace_str);
        }
    }
}

} // namespace GRIM::Diagnostics
