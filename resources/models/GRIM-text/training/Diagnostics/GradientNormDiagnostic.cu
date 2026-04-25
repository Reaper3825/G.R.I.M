//======================================================//
//  GradientNormDiagnostic.cu
//  Implementation of runGradientNormDiagnostic.
//  Lifted verbatim from Phase2_TrainingLoop.cu processBatch
//  (PRE-GRADNORM log → end of [EMB_GRAD_EQUATION] block).
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
#include <cstdio>
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
        if (value == 0.0f) value = 0.0f; // kill -0.0
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

// Verbatim copy of Internal::formatGradientComponents from
// Phase2_TrainingLoop.cu (kept local to preserve byte-equivalent log output).
std::string formatGradientComponents(const GRIM::GradNorm::GradMetrics& gm, bool tied) {
    using GM = GRIM::GradNorm::GradMetrics;
    std::ostringstream comp_msg;
    comp_msg << "[GradTrace] COMPONENTS(rms):";

    constexpr int kComponentPrecision = 10;

    if (tied) {
        comp_msg << " emb_lm_tied=" << formatScalar(GM::rms(gm.lm_head_sum_sq, gm.lm_head_count), kComponentPrecision);
    } else {
        comp_msg << " emb=" << formatScalar(GM::rms(gm.embedding_sum_sq, gm.embedding_count), kComponentPrecision)
                 << " lm=" << formatScalar(GM::rms(gm.lm_head_sum_sq, gm.lm_head_count), kComponentPrecision);
    }

    comp_msg << " attn=" << formatScalar(GM::rms(gm.attention_sum_sq, gm.attention_count), kComponentPrecision)
             << " ffn=" << formatScalar(GM::rms(gm.ffn_sum_sq, gm.ffn_count), kComponentPrecision)
             << " rmsnorm=" << formatScalar(GM::rms(gm.rmsnorm_sum_sq, gm.rmsnorm_count), kComponentPrecision);

    comp_msg << " tied=" << (tied ? "yes" : "no");

    if (gm.scratchblock_sum_sq > 0.0f) {
        comp_msg << " sb=" << formatScalar(GM::rms(gm.scratchblock_sum_sq, gm.scratchblock_count), kComponentPrecision);
    }

    return comp_msg.str();
}

} // namespace

GradNormSnapshot runGradientNormDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIMText::Training::TrainingLoopState& state,
    GRIMText::Training::BatchResult& result,
    int batch_idx)
{
    ctx.logging.logger->log("[GradTrace] PRE-GRADNORM batch=" + std::to_string(batch_idx + 1) +
                            " cached_preclip=" + formatScalar(state.last_grad_rms));

    // === TIMING: Track wall time of grad-norm sync ===
    auto norm_start = std::chrono::steady_clock::now();

    auto& training_state = ctx.model->getTrainingState();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    const auto& groups = ctx.model->parameterGroups();

    // ════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: Sample gradient values BEFORE measurement to verify
    // backward results survive to this point.  Companion to [GRAD_DIAG]
    // POST-BACKWARD in AutogradTraining.cu.
    // ════════════════════════════════════════════════════════════════════
    {
        cudaStreamSynchronize(stream);
        float lm_sample = 0.0f;
        float* lm_grads = ctx.model->getLmHeadLayer()->weights().grad_data();
        if (lm_grads) {
            cudaMemcpy(&lm_sample, lm_grads, sizeof(float), cudaMemcpyDeviceToHost);
        }
        // Also verify the ParameterGroup sees the same pointer
        float pg_sample = 0.0f;
        for (size_t g = 0; g < groups.size(); ++g) {
            if (groups[g].type == GRIM::ParamGroupType::LM_HEAD) {
                float* pg_grads = groups[g].grads();
                if (pg_grads) {
                    cudaMemcpy(&pg_sample, pg_grads, sizeof(float), cudaMemcpyDeviceToHost);
                }
                fprintf(stderr,
                    "[GRAD_DIAG] PRE-MEASURE batch=%d micro=%d "
                    "lm_grad[0]=%.10e lm_ptr=%p pg_grad[0]=%.10e pg_ptr=%p match=%s\n",
                    batch_idx + 1, ctx.optimizer.current_micro_step,
                    lm_sample, static_cast<void*>(lm_grads),
                    pg_sample, static_cast<void*>(pg_grads),
                    (lm_grads == pg_grads) ? "YES" : "NO");
                break;
            }
        }
    }

    // Lazy-allocate scratch buffers on first use
    if (!training_state.grad_norm_scratch) {
        training_state.grad_norm_scratch = GRIM::GradNorm::allocateGradNormScratch(
            groups.size() * 2, stream);
        if (!training_state.grad_norm_scratch) {
            throw std::runtime_error("[FATAL] Failed to allocate grad_norm_scratch at batch " +
                                     std::to_string(batch_idx + 1));
        }
    }

    // Issue #138: Record CUDA event BEFORE launching grad norm kernels
    cudaEvent_t pre_norm_event = nullptr, post_norm_event = nullptr;
    cudaEventCreate(&pre_norm_event);
    cudaEventRecord(pre_norm_event, stream);

    auto norm_status = GRIM::GradNorm::measureGradientNormsLaunch(
        groups.data(), groups.size(), training_state.grad_norm_scratch, stream);
    if (norm_status != GRIM::GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[FATAL] measureGradientNormsLaunch failed: " +
                                 std::string(GRIM::GradNorm::statusToString(norm_status)) +
                                 " at batch " + std::to_string(batch_idx + 1));
    }
    cudaEventCreate(&post_norm_event);
    cudaEventRecord(post_norm_event, stream);
    cudaStreamSynchronize(stream);  // Wait for D2H before Finalize
    norm_status = GRIM::GradNorm::measureGradientNormsFinalize(
        groups.data(), groups.size(), training_state.grad_norm_scratch);
    if (norm_status != GRIM::GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[FATAL] measureGradientNormsFinalize failed: " +
                                 std::string(GRIM::GradNorm::statusToString(norm_status)) +
                                 " at batch " + std::to_string(batch_idx + 1));
    }

    const auto& gm = *training_state.grad_norm_scratch->h_metrics;
    // Compute per-component RMS matching the clipping strategy (Issue #139)
    const bool tied_for_norm = ctx.model->getConfig().tie_embeddings;
    const float emb_sum_sq_pre = tied_for_norm ? gm.lm_head_sum_sq : (gm.lm_head_sum_sq + gm.embedding_sum_sq);
    const int64_t emb_count_pre = tied_for_norm ? gm.lm_head_count : (gm.lm_head_count + gm.embedding_count);
    const float enc_sum_sq_pre = gm.attention_sum_sq + gm.ffn_sum_sq + gm.rmsnorm_sum_sq + gm.scratchblock_sum_sq + gm.reasoning_head_sum_sq + gm.execution_block_sum_sq;
    const int64_t enc_count_pre = gm.attention_count + gm.ffn_count + gm.rmsnorm_count + gm.scratchblock_count + gm.reasoning_head_count + gm.execution_block_count;
    const float emb_rms_pre = (emb_count_pre > 0) ? std::sqrt(emb_sum_sq_pre / static_cast<float>(emb_count_pre)) : 0.0f;
    const float enc_rms_pre = (enc_count_pre > 0) ? std::sqrt(enc_sum_sq_pre / static_cast<float>(enc_count_pre)) : 0.0f;
    // Separate sb_rms for POST-GRADNORM visibility (Issue #150)
    const float sb_rms_pre = (gm.scratchblock_count > 0) ? std::sqrt(gm.scratchblock_sum_sq / static_cast<float>(gm.scratchblock_count)) : 0.0f;
    // Combined RMS across all parameter groups
    const float total_sum_sq_pre = emb_sum_sq_pre + enc_sum_sq_pre;
    const int64_t total_count_pre = emb_count_pre + enc_count_pre;
    result.grad_rms = (total_count_pre > 0) ? std::sqrt(total_sum_sq_pre / static_cast<float>(total_count_pre)) : 0.0f;
    result.normalized_grad_rms = result.grad_rms;
    const float preclip_grad_rms = result.grad_rms;
    auto norm_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - norm_start).count();

    // Issue #138: Decompose wall time into kernel time + backward drain time
    float gpu_kernel_ms = 0.0f;
    cudaEventElapsedTime(&gpu_kernel_ms, pre_norm_event, post_norm_event);
    cudaEventDestroy(pre_norm_event);
    cudaEventDestroy(post_norm_event);
    const float drain_ms = norm_elapsed_ms - gpu_kernel_ms;
    ctx.logging.logger->log("[GradTrace] POST-BACKWARD synced measureGradientNorms in " +
                                formatScalar(norm_elapsed_ms, 2) + "ms (kernel=" +
                                formatScalar(gpu_kernel_ms, 2) + "ms drain=" +
                                formatScalar(drain_ms, 2) + "ms)");

    // NaN/Inf check — RULE 20: Fail loud!
    if (gm.has_nan || gm.has_inf) {
        std::ostringstream nf_log;
        nf_log << "[GradTrace] NON-FINITE grads detected"
               << " nan=" << (gm.has_nan ? "true" : "false")
               << " inf=" << (gm.has_inf ? "true" : "false")
               << " first_nan_group=" << gm.first_nan_group
               << " first_nan_value=" << gm.first_nan_value
               << " first_inf_group=" << gm.first_inf_group
               << " first_inf_value=" << gm.first_inf_value
               << " groups_processed=" << gm.groups_processed;
        ctx.logging.logger->log(nf_log.str());

        throw std::runtime_error("[FATAL] NaN/Inf detected in gradients at batch " +
                                std::to_string(batch_idx + 1) +
                                " first_nan_group=" + std::to_string(gm.first_nan_group) +
                                " - investigate the backward pass!");
    }

    const bool tied = ctx.model->getConfig().tie_embeddings;
    std::string comp_log = formatGradientComponents(gm, tied);
    ctx.logging.logger->log(comp_log);

    ctx.logging.logger->log("[GradTrace] POST-GRADNORM preclip=" + formatScalar(preclip_grad_rms) +
                            " emb_rms=" + formatScalar(emb_rms_pre, 6) +
                            " enc_rms=" + formatScalar(enc_rms_pre, 6) +
                            " sb_rms=" + formatScalar(sb_rms_pre, 6));

    // ========================================================================
    // DIAGNOSTIC: [EMB_GRAD_EQUATION] Embedding gradient spike analysis (Issue #141)
    // Rule 21 equation-based logging for tied-weight gradient decomposition.
    // Runs every diag_interval batches (same cadence as other sync diagnostics).
    // Identifies which token rows concentrate gradient mass and whether
    // atomicAdd scatter density correlates with spike magnitude.
    // ========================================================================
    {
        static float prev_emb_rms_for_spike_diag = 0.0f;

        // Gate behind shouldSyncDiagnostics so full vocab gradient D2H only runs on diagnostic sync interval
        static int emb_grad_diag_interval = 10;
        const bool kEmbGradDiagEnabled = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx) &&
            ctx.logging.tape && ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug) &&
            (batch_idx == 0 || (batch_idx + 1) % std::max(emb_grad_diag_interval, 1) == 0);

        // gm is already in scope from measureGradientNorms above
        const float curr_emb_rms = (gm.lm_head_count > 0)
            ? std::sqrt(gm.lm_head_sum_sq / static_cast<float>(gm.lm_head_count)) : 0.0f;

        if (kEmbGradDiagEnabled) {
            const auto& ts = ctx.model->getTrainingState();
            const auto& cfg = ctx.model->getConfig();
            cudaStream_t emb_stream = ts.stream_ctrl.getPrimaryStream();

            const int total_tokens_diag = ts.cached_batch_size * ts.cached_seq_len;
            const int* d_tok_ids = reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data);

            if (d_tok_ids && total_tokens_diag > 0) {
                GRIM::Diagnostics::EmbGradEquationDiag emb_diag = GRIM::Diagnostics::computeEmbGradEquation(
                    ctx.model->getEmbeddingLayer(), d_tok_ids, total_tokens_diag,
                    cfg.d_model, cfg.vocab_size,
                    prev_emb_rms_for_spike_diag, curr_emb_rms,
                    emb_stream);

                std::string emb_eq_str = GRIM::Diagnostics::formatEmbGradEquation(emb_diag, batch_idx);
                ctx.logging.logger->log(emb_eq_str);

                EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Embedding, GRIM::Logging::LogPhase::GRADIENT_CLIP, 0, "EMB_GRAD_EQUATION", emb_eq_str.c_str());
            }
        }

        prev_emb_rms_for_spike_diag = curr_emb_rms;
    }

    GradNormSnapshot snap;
    snap.grad_rms = result.grad_rms;
    snap.preclip_grad_rms = preclip_grad_rms;
    snap.emb_rms_pre = emb_rms_pre;
    snap.enc_rms_pre = enc_rms_pre;
    snap.sb_rms_pre = sb_rms_pre;
    return snap;
}

} // namespace GRIM::Diagnostics
