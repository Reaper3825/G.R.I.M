//======================================================//
//  GradientNormDiagnostic.cu
//  Implementation of runGradientNormDiagnostic.
//  Reads gradients/metrics, logs gated diagnostics, and throws on hard
//  validation failures. Training-loop result mutation and scratch ownership
//  stay outside this file.
//======================================================//

#include "GradientNormDiagnostic.hpp"

#include "DiagnosticGates.hpp"
#include "TrainingDiagnostics.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/GradNorm/GradNormGPU.hpp"
#include "../../Shared/LogRecorder/LogTypes.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"

#include <chrono>
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

void throwCudaFailure(cudaError_t err, const char* operation, int batch_idx) {
    if (err == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string("[FATAL] ") + operation + " failed at batch " +
                             std::to_string(batch_idx + 1) + ": " + cudaGetErrorString(err));
}

struct ScopedCudaEvent {
    cudaEvent_t event = nullptr;

    ScopedCudaEvent(const char* name, int batch_idx) {
        throwCudaFailure(cudaEventCreate(&event), name, batch_idx);
    }

    ~ScopedCudaEvent() {
        if (event) {
            cudaEventDestroy(event);
        }
    }

    ScopedCudaEvent(const ScopedCudaEvent&) = delete;
    ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;
};

void validateGradNormInputsOrThrow(
    const std::vector<GRIM::ParameterGroup>& groups,
    const GRIM::GradNorm::GradNormScratch* scratch,
    int batch_idx)
{
    if (groups.empty()) {
        throw std::runtime_error("[FATAL] No parameter groups registered for grad norm at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (!scratch) {
        throw std::runtime_error("[FATAL] grad_norm_scratch is NULL at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " - Phase2 must allocate TrainingState-owned scratch before diagnostics");
    }
    if (!scratch->d_partial_sums || !scratch->h_partial_sums || !scratch->h_metrics) {
        throw std::runtime_error("[FATAL] grad_norm_scratch buffers are incomplete at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (scratch->max_groups < groups.size()) {
        throw std::runtime_error("[FATAL] grad_norm_scratch capacity mismatch at batch " +
                                 std::to_string(batch_idx + 1) +
                                 " required_groups=" + std::to_string(groups.size()) +
                                 " scratch_max_groups=" + std::to_string(scratch->max_groups));
    }

    for (size_t g = 0; g < groups.size(); ++g) {
        if (groups[g].size() == 0) {
            throw std::runtime_error("[FATAL] Parameter group '" + groups[g].name +
                                     "' has size=0 during grad norm at batch " +
                                     std::to_string(batch_idx + 1));
        }
        if (!groups[g].grads()) {
            throw std::runtime_error("[FATAL] Parameter group '" + groups[g].name +
                                     "' has NULL gradients during grad norm at batch " +
                                     std::to_string(batch_idx + 1));
        }
    }
}

float rmsOrThrow(float sum_sq, int64_t count, const char* label, int batch_idx) {
    if (count <= 0) {
        throw std::runtime_error(std::string("[FATAL] ") + label +
                                 " gradient count is zero at batch " +
                                 std::to_string(batch_idx + 1));
    }
    if (!std::isfinite(sum_sq)) {
        throw std::runtime_error(std::string("[FATAL] ") + label +
                                 " gradient sum_sq is non-finite at batch " +
                                 std::to_string(batch_idx + 1));
    }
    return std::sqrt(sum_sq / static_cast<float>(count));
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

float computeRegisteredGlobalRmsOrThrow(
    const std::vector<GRIM::ParameterGroup>& groups,
    const GRIM::GradNorm::GradNormScratch* scratch,
    int batch_idx)
{
    float global_sum_sq = 0.0f;
    int64_t global_count = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
        const float sq = scratch->h_partial_sums[g];
        if (!std::isfinite(sq)) {
            throw std::runtime_error("[FATAL] Non-finite grad norm partial sum for group '" +
                                     groups[g].name + "' at batch " +
                                     std::to_string(batch_idx + 1));
        }
        global_sum_sq += sq;
        global_count += static_cast<int64_t>(groups[g].size());
    }
    return rmsOrThrow(global_sum_sq, global_count, "registered_global", batch_idx);
}

float computeEmbeddingBucketRmsOrThrow(
    const GRIM::GradNorm::GradMetrics& gm,
    bool tied,
    int batch_idx)
{
    const float sum_sq = tied ? gm.lm_head_sum_sq : (gm.lm_head_sum_sq + gm.embedding_sum_sq);
    const int64_t count = tied ? gm.lm_head_count :
        (static_cast<int64_t>(gm.lm_head_count) + static_cast<int64_t>(gm.embedding_count));
    return rmsOrThrow(sum_sq, count, tied ? "emb_lm_tied" : "emb_lm_untied", batch_idx);
}

float computeEncoderTelemetryRmsOrThrow(const GRIM::GradNorm::GradMetrics& gm, int batch_idx) {
    const float sum_sq = gm.attention_sum_sq + gm.ffn_sum_sq + gm.rmsnorm_sum_sq +
        gm.scratchblock_sum_sq + gm.reasoning_head_sum_sq + gm.execution_block_sum_sq;
    const int64_t count = static_cast<int64_t>(gm.attention_count) + gm.ffn_count +
        gm.rmsnorm_count + gm.scratchblock_count + gm.reasoning_head_count +
        gm.execution_block_count;
    return rmsOrThrow(sum_sq, count, "encoder_telemetry", batch_idx);
}

float computeScratchBlockTelemetryRms(const GRIM::GradNorm::GradMetrics& gm) {
    if (gm.scratchblock_count <= 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::sqrt(gm.scratchblock_sum_sq / static_cast<float>(gm.scratchblock_count));
}

} // namespace

GradNormSnapshot runGradientNormDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    int batch_idx)
{
    auto& training_state = ctx.model->getTrainingState();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    const auto& groups = ctx.model->parameterGroups();
    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);

    if (sync_diag) {
        ctx.logging.logger->log("[GradTrace] PRE-GRADNORM batch=" + std::to_string(batch_idx + 1) +
                                " cached_preclip=" + formatScalar(state.last_grad_rms));
    }

    validateGradNormInputsOrThrow(groups, training_state.grad_norm_scratch, batch_idx);

    ScopedCudaEvent pre_norm_event("cudaEventCreate(pre_norm_event)", batch_idx);
    ScopedCudaEvent post_norm_event("cudaEventCreate(post_norm_event)", batch_idx);
    throwCudaFailure(cudaEventRecord(pre_norm_event.event, stream), "cudaEventRecord(pre_norm_event)", batch_idx);

    auto norm_status = GRIM::GradNorm::measureGradientNormsLaunch(
        groups.data(), groups.size(), training_state.grad_norm_scratch, stream);
    if (norm_status != GRIM::GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[FATAL] measureGradientNormsLaunch failed: " +
                                 std::string(GRIM::GradNorm::statusToString(norm_status)) +
                                 " at batch " + std::to_string(batch_idx + 1));
    }

    throwCudaFailure(cudaEventRecord(post_norm_event.event, stream), "cudaEventRecord(post_norm_event)", batch_idx);

    const auto sync_start = std::chrono::steady_clock::now();
    throwCudaFailure(cudaStreamSynchronize(stream), "cudaStreamSynchronize(grad_norm)", batch_idx);
    const float sync_wait_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - sync_start).count();

    float gpu_measure_ms = 0.0f;
    throwCudaFailure(cudaEventElapsedTime(&gpu_measure_ms, pre_norm_event.event, post_norm_event.event),
                     "cudaEventElapsedTime(grad_norm)", batch_idx);

    const auto finalize_start = std::chrono::steady_clock::now();
    norm_status = GRIM::GradNorm::measureGradientNormsFinalize(
        groups.data(), groups.size(), training_state.grad_norm_scratch);
    const float finalize_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - finalize_start).count();
    if (norm_status != GRIM::GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[FATAL] measureGradientNormsFinalize failed: " +
                                 std::string(GRIM::GradNorm::statusToString(norm_status)) +
                                 " at batch " + std::to_string(batch_idx + 1));
    }

    const auto& gm = *training_state.grad_norm_scratch->h_metrics;
    validateMeasuredMetricsOrThrow(gm, groups.size(), batch_idx);

    const bool tied = ctx.model->getConfig().tie_embeddings;
    const float preclip_grad_rms = computeRegisteredGlobalRmsOrThrow(
        groups, training_state.grad_norm_scratch, batch_idx);
    const float emb_rms_pre = computeEmbeddingBucketRmsOrThrow(gm, tied, batch_idx);
    const float enc_rms_pre = computeEncoderTelemetryRmsOrThrow(gm, batch_idx);
    const float sb_rms_pre = computeScratchBlockTelemetryRms(gm);

    if (sync_diag) {
        ctx.logging.logger->log("[GradTrace] POST-BACKWARD measureGradientNorms timing "
                                "gpu_event=" + formatScalar(gpu_measure_ms, 2) + "ms " +
                                "sync_wait=" + formatScalar(sync_wait_ms, 2) + "ms " +
                                "finalize=" + formatScalar(finalize_ms, 2) + "ms");

        ctx.logging.logger->log("[GradTrace] POST-GRADNORM preclip_registered_global=" +
                                formatScalar(preclip_grad_rms, 6) +
                                " emb_bucket_rms=" + formatScalar(emb_rms_pre, 6) +
                                " enc_telemetry_rms=" + formatScalar(enc_rms_pre, 6));
    }

    // ========================================================================
    // DIAGNOSTIC: [EMB_GRAD_EQUATION] Embedding gradient spike analysis (Issue #141)
    // Rule 21 equation-based logging for tied-weight gradient decomposition.
    // Runs every diag_interval batches (same cadence as other sync diagnostics).
    // Identifies which token rows concentrate gradient mass and whether
    // atomicAdd scatter density correlates with spike magnitude.
    // ========================================================================
    {
        const bool kEmbGradDiagEnabled = sync_diag && ctx.logging.tape &&
            ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug);

        if (kEmbGradDiagEnabled) {
            const auto& ts = ctx.model->getTrainingState();
            const auto& cfg = ctx.model->getConfig();
            cudaStream_t emb_stream = ts.stream_ctrl.getPrimaryStream();

            const int total_tokens_diag = ts.cached_batch_size * ts.cached_seq_len;
            const int* d_tok_ids = reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data);

            if (d_tok_ids && total_tokens_diag > 0) {
                const float prev_emb_rms = state.has_prev_emb_rms_for_spike_diag
                    ? state.prev_emb_rms_for_spike_diag
                    : emb_rms_pre;
                GRIM::Diagnostics::EmbGradEquationDiag emb_diag = GRIM::Diagnostics::computeEmbGradEquation(
                    ctx.model->getEmbeddingLayer(), d_tok_ids, total_tokens_diag,
                    cfg.d_model, cfg.vocab_size,
                    prev_emb_rms, emb_rms_pre,
                    emb_stream);

                std::string emb_eq_str = GRIM::Diagnostics::formatEmbGradEquation(emb_diag, batch_idx);
                ctx.logging.logger->log(emb_eq_str);

                EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Embedding, GRIM::Logging::LogPhase::GRADIENT_CLIP, 0, "EMB_GRAD_EQUATION", emb_eq_str.c_str());

                state.prev_emb_rms_for_spike_diag = emb_rms_pre;
                state.has_prev_emb_rms_for_spike_diag = true;
            }
        }
    }

    GradNormSnapshot snap;
    snap.grad_rms = preclip_grad_rms;
    snap.preclip_grad_rms = preclip_grad_rms;
    snap.emb_rms_pre = emb_rms_pre;
    snap.enc_rms_pre = enc_rms_pre;
    snap.sb_rms_pre = sb_rms_pre;
    snap.metrics = gm;
    return snap;
}

} // namespace GRIM::Diagnostics
