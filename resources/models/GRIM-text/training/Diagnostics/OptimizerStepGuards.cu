//======================================================//
//  OptimizerStepGuards.cu
//  Implementations lifted verbatim from
//  Phase2_TrainingLoop.cu processBatch.
//======================================================//

#include "OptimizerStepGuards.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Gradients/GradientCC_GPU.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

namespace {

std::string formatScalar(float value, int precision = 4) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
        if (value == 0.0f) value = 0.0f;
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

} // namespace

// ────────────────────────────────────────────────────────────────────────────
// Issue #149: Zero PAD/UNK gradients before optimizer step
// ────────────────────────────────────────────────────────────────────────────
void zeroNonTrainableTokenGrads(GRIMText::Training::TrainingContext& ctx) {
    auto& zero_ts = ctx.model->getTrainingState();
    cudaStream_t zero_stream = zero_ts.stream_ctrl.getPrimaryStream();
    const auto& zero_cfg = ctx.model->getConfig();
    const size_t row_bytes = static_cast<size_t>(zero_cfg.d_model) * sizeof(float);

    constexpr int NON_TRAINABLE_TOKENS[] = {
        GRIM::Tokenizer::UNK_TOKEN_ID,  // 0
        GRIM::Tokenizer::PAD_TOKEN_ID   // 1
    };

    // Zero embedding gradients for non-trainable tokens
    float* emb_grads = ctx.model->getEmbeddingLayer()->tokenWeights().grad_data();
    if (emb_grads) {
        for (int tok : NON_TRAINABLE_TOKENS) {
            cudaMemsetAsync(
                emb_grads + static_cast<size_t>(tok) * zero_cfg.d_model,
                0, row_bytes, zero_stream);
        }
    }

    // Zero LM head gradients for non-trainable tokens
    float* lm_grads = ctx.model->getLmHeadLayer()->weights().grad_data();
    if (lm_grads) {
        for (int tok : NON_TRAINABLE_TOKENS) {
            cudaMemsetAsync(
                lm_grads + static_cast<size_t>(tok) * zero_cfg.d_model,
                0, row_bytes, zero_stream);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// POST-ACCUMULATION global gradient clipping
// ────────────────────────────────────────────────────────────────────────────
void clipPostAccumulationGradients(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    float per_token_limit,
    int batch_idx,
    float& clipping_elapsed_ms)
{
    auto clipping_start = std::chrono::steady_clock::now();

    auto& clip_ts = ctx.model->getTrainingState();
    cudaStream_t clip_stream = clip_ts.stream_ctrl.getPrimaryStream();
    auto& clip_groups = ctx.model->parameterGroups();

    if (!clip_ts.grad_norm_scratch) {
        throw std::runtime_error("[FATAL] grad_norm_scratch is NULL at clipping stage - "
                                 "diagnostic norm should have allocated it");
    }

    GRIM::GradClip::ClipConfig clip_cfg;
    clip_cfg.max_rms = per_token_limit;

    const auto clip = GRIM::GradClip::clipGradientNorms(
        clip_groups.data(), clip_groups.size(),
        clip_ts.grad_norm_scratch, clip_cfg, clip_stream);

    result.grad_rms = clip.global_rms_post;
    result.gradient_clipped = clip.any_clipped();

    ctx.logging.logger->log("[PostAccumClip] batch=" + std::to_string(batch_idx + 1) +
                            " pre_clip_global_rms=" + formatScalar(clip.global_rms_pre, 6) +
                            " clipped=" + (clip.clipped ? "YES" : "NO") +
                            " post_clip_global_rms=" + formatScalar(clip.global_rms_post, 6));

    clipping_elapsed_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - clipping_start).count();
}

// ────────────────────────────────────────────────────────────────────────────
// Rule 20: Post-optimizer weight NaN spot check
// ────────────────────────────────────────────────────────────────────────────
void checkPostOptimizerWeightsFinite(
    GRIMText::Training::TrainingContext& ctx,
    const GRIMText::Training::BatchResult& result,
    int batch_idx)
{
    cudaStreamSynchronize(ctx.model->getTrainingState().stream_ctrl.getPrimaryStream());
    const auto& groups = ctx.model->parameterGroups();
    for (size_t g = 0; g < groups.size(); ++g) {
        if (!groups[g].weights() || groups[g].size() == 0) continue;
        // Sample first element of each parameter group (fast: 1 float per group)
        float h_sample = 0.0f;
        cudaMemcpy(&h_sample, groups[g].weights(), sizeof(float), cudaMemcpyDeviceToHost);
        if (!std::isfinite(h_sample)) {
            throw std::runtime_error("[FATAL] Post-optimizer NaN/Inf in parameter group '" +
                groups[g].name + "' (group " + std::to_string(g) + ") at batch " +
                std::to_string(batch_idx + 1) + " optimizer_step=" +
                std::to_string(ctx.optimizer.optimizer_state.step) +
                " lr=" + std::to_string(result.learning_rate) +
                " — THIS batch's optimizer step corrupted weights. "
                "Check gradient magnitude and clipping for this group.");
        }
    }
}

} // namespace GRIM::Diagnostics
