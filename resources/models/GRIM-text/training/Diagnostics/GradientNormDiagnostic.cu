//======================================================//
//  GradientNormDiagnostic.cu
//  Implementation of runGradientNormClipDiagnostic.
//  Consumes the existing global clipping measurement, logs gated diagnostics,
//  and throws on hard validation failures. It never launches GradNorm kernels,
//  allocates scratch, or mutates training-loop results. The optional
//  EMB_GRAD_EQUATION path is an explicitly gated sync diagnostic because it
//  performs host-side inspection of device data.
//======================================================//

#include "GradientNormDiagnostic.hpp"

#include "DiagnosticGates.hpp"
#include "TrainingDiagnostics.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/LogRecorder/LogTypes.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

namespace {

std::string formatScalar(float value, int precision = 4) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
        if (value == 0.0f) value = 0.0f; // kill -0.0
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

float rmsOrThrow(double sum_sq, uint64_t count, const char* label, int batch_idx) {
    if (count == 0) {
        throw std::runtime_error(std::string("[FATAL] ") + label +
                                 " gradient count is zero at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (!std::isfinite(sum_sq)) {
        throw std::runtime_error(std::string("[FATAL] ") + label +
                                 " gradient sum_sq is non-finite at batch " +
                                 std::to_string(batch_idx + 1));
    }
    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
}

void validateMeasuredMetricsOrThrow(
    const GRIM::GradNorm::GradMetrics& gm,
    size_t expected_groups,
    int batch_idx)
{
    if (gm.groups_processed != expected_groups) {
        throw std::runtime_error("[FATAL] GradNorm processed group count mismatch at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " expected=" + std::to_string(expected_groups) +
                                 " actual=" + std::to_string(gm.groups_processed));
    }
    if (gm.has_nan || gm.has_inf) {
        std::ostringstream oss;
        oss << "[FATAL] NaN/Inf detected in gradients at batch " << (batch_idx + 1)
            << " nan=" << (gm.has_nan ? "true" : "false")
            << " inf=" << (gm.has_inf ? "true" : "false")
            << " first_nan_group=" << gm.first_nan_group
            << " first_nan_value=" << gm.first_nan_value
            << " first_inf_group=" << gm.first_inf_group
            << " first_inf_value=" << gm.first_inf_value;
        throw std::runtime_error(oss.str());
    }
}

void validateClipResultOrThrow(const GRIM::GradClip::ClipResult& clip, int batch_idx) {
    if (clip.measured_group_count == 0) {
        throw std::runtime_error("[FATAL] ClipResult has zero measured_group_count at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (!std::isfinite(clip.global_rms_pre)) {
        throw std::runtime_error("[FATAL] ClipResult global_rms_pre is non-finite at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (!std::isfinite(clip.global_rms_post)) {
        throw std::runtime_error("[FATAL] ClipResult global_rms_post is non-finite at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (clip.global_rms_pre < 0.0f || clip.global_rms_post < 0.0f) {
        throw std::runtime_error("[FATAL] ClipResult RMS is negative at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " pre=" + std::to_string(clip.global_rms_pre) +
                                 " post=" + std::to_string(clip.global_rms_post));
    }

    constexpr float kRmsEps = 1.0e-6f;
    const float equality_tolerance = kRmsEps * std::max(
        1.0f, std::max(clip.global_rms_pre, clip.global_rms_post));
    const float required_decrease = std::max(1.0e-12f, kRmsEps * clip.global_rms_pre);
    if (clip.global_rms_post > clip.global_rms_pre + equality_tolerance) {
        throw std::runtime_error("[FATAL] ClipResult post RMS exceeds pre RMS at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " pre=" + std::to_string(clip.global_rms_pre) +
                                 " post=" + std::to_string(clip.global_rms_post));
    }
    if (clip.clipped && (clip.global_rms_pre - clip.global_rms_post) <= required_decrease) {
        throw std::runtime_error("[FATAL] ClipResult marked clipped but RMS did not decrease at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " pre=" + std::to_string(clip.global_rms_pre) +
                                 " post=" + std::to_string(clip.global_rms_post));
    }
    if (!clip.clipped && std::fabs(clip.global_rms_post - clip.global_rms_pre) > equality_tolerance) {
        throw std::runtime_error("[FATAL] ClipResult marked unclipped but pre/post RMS differ at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " pre=" + std::to_string(clip.global_rms_pre) +
                                 " post=" + std::to_string(clip.global_rms_post));
    }
}

float computeEmbeddingDiagnosticRmsOrThrow(
    const GRIM::GradNorm::GradMetrics& gm,
    bool tied,
    int batch_idx)
{
    const double sum_sq = tied ? gm.lm_head_sum_sq : (gm.lm_head_sum_sq + gm.embedding_sum_sq);
    const uint64_t count = tied ? gm.lm_head_count : (gm.lm_head_count + gm.embedding_count);
    return rmsOrThrow(sum_sq, count, tied ? "emb_lm_tied" : "emb_lm_untied", batch_idx);
}

float computeEncoderTelemetryRms(const GRIM::GradNorm::GradMetrics& gm, int batch_idx) {
    const double sum_sq = gm.attention_sum_sq + gm.ffn_sum_sq + gm.rmsnorm_sum_sq +
        gm.scratchblock_sum_sq + gm.reasoning_head_sum_sq + gm.execution_block_sum_sq;
    const uint64_t count = gm.attention_count + gm.ffn_count +
        gm.rmsnorm_count + gm.scratchblock_count + gm.reasoning_head_count +
        gm.execution_block_count;
    if (count == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return rmsOrThrow(sum_sq, count, "encoder_telemetry", batch_idx);
}

float computeScratchBlockTelemetryRms(const GRIM::GradNorm::GradMetrics& gm) {
    if (gm.scratchblock_count <= 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(std::sqrt(gm.scratchblock_sum_sq / static_cast<double>(gm.scratchblock_count)));
}

std::string formatTopGradientGroups(
    const GRIM::GradClip::ClipResult& clip,
    const std::vector<GRIM::ParameterGroup>& groups)
{
    std::ostringstream oss;
    oss << "[GradTrace] TOP-GRAD-GROUPS";
    for (const auto& top : clip.top_groups) {
        if (!top.valid) {
            continue;
        }
        const std::string group_name = top.index < groups.size()
            ? groups[top.index].name
            : std::string("<out-of-range>");
        oss << " #" << top.index
            << "(" << group_name
            << ",type=" << top.type
            << ",layer=" << top.layer_index
            << ",rms=" << formatScalar(top.rms, 6)
            << ",count=" << top.count
            << ")";
    }
    return oss.str();
}

} // namespace

void runGradientNormClipDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::GradClip::ClipResult& clip,
    int batch_idx)
{
    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);

    const auto& gm = clip.metrics;

    validateClipResultOrThrow(clip, batch_idx);

    ctx.logging.logger->log("[GradTrace] CLIP-MEASURED batch=" + std::to_string(batch_idx + 1) +
                            " preclip_registered_global=" + formatScalar(clip.global_rms_pre, 6));

    validateMeasuredMetricsOrThrow(gm, clip.measured_group_count, batch_idx);

    const auto lm_head_hp =
        GRIM::HyperParameters::lmHeadLayerConstructionHP(ctx.config.hyperparameters.architecture);
    const bool tied = lm_head_hp.tie_embeddings;
    const float preclip_grad_rms = clip.global_rms_pre;
    const float emb_rms_pre = computeEmbeddingDiagnosticRmsOrThrow(gm, tied, batch_idx);
    const float enc_rms_pre = computeEncoderTelemetryRms(gm, batch_idx);
    const float sb_rms_pre = computeScratchBlockTelemetryRms(gm);
    const auto& groups = ctx.model->parameterGroups();

    ctx.logging.logger->log("[GradTrace] POST-CLIP-MEASURE preclip_registered_global=" +
                            formatScalar(preclip_grad_rms, 6) +
                            " postclip_registered_global=" + formatScalar(clip.global_rms_post, 6) +
                            " clipped=" + (clip.clipped ? "YES" : "NO") +
                            " emb_rms_pre=" + formatScalar(emb_rms_pre, 6) +
                            " enc_rms_pre=" + formatScalar(enc_rms_pre, 6) +
                            " sb_rms_pre=" + formatScalar(sb_rms_pre, 6));
    ctx.logging.logger->log(formatTopGradientGroups(clip, groups));

    // ========================================================================
    // DIAGNOSTIC: [EMB_GRAD_EQUATION] Embedding gradient spike analysis (Issue #141)
    // Rule 21 equation-based logging for tied-weight gradient decomposition.
    // Runs every diag_interval batches (same cadence as other sync diagnostics).
    // This is sync-safe only under shouldSyncDiagnostics(): computeEmbGradEquation()
    // performs blocking D2H copies for host-side row/frequency analysis.
    // Identifies which token rows concentrate gradient mass and whether
    // atomicAdd scatter density correlates with spike magnitude.
    // ========================================================================
    {
        const bool kEmbGradDiagEnabled = sync_diag && ctx.logging.tape &&
            ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug);

        if (kEmbGradDiagEnabled) {
            const auto& ts = ctx.model->getTrainingState();
            const auto model_arch_hp =
                GRIM::HyperParameters::modelArchitectureHP(ctx.config.hyperparameters.architecture);
            cudaStream_t emb_stream = ts.stream_ctrl.getPrimaryStream();

            const int total_tokens_diag = payload.total_tokens;
            const int* d_tok_ids = reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data);

            if (d_tok_ids && total_tokens_diag > 0) {
                const float prev_emb_rms = state.diagnostics.has_prev_emb_rms
                    ? state.diagnostics.prev_emb_rms
                    : emb_rms_pre;
                GRIM::Diagnostics::EmbGradEquationDiag emb_diag = GRIM::Diagnostics::computeEmbGradEquation(
                    ctx.model->getEmbeddingLayer(), d_tok_ids, total_tokens_diag,
                    model_arch_hp.d_model, static_cast<int>(ctx.config.actual_vocab_size),
                    prev_emb_rms, emb_rms_pre,
                    emb_stream);

                std::string emb_eq_str = GRIM::Diagnostics::formatEmbGradEquation(emb_diag, batch_idx);
                ctx.logging.logger->log(emb_eq_str);

                EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Embedding, GRIM::Logging::LogPhase::GRADIENT_CLIP, 0, "EMB_GRAD_EQUATION", emb_eq_str.c_str());

                state.diagnostics.prev_emb_rms = emb_rms_pre;
                state.diagnostics.has_prev_emb_rms = true;
            }
        }
    }

}

} // namespace GRIM::Diagnostics
