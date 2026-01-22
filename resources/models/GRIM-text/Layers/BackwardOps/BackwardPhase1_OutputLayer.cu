/**
 * @file BackwardPhase1_OutputLayer.cu
 * @brief Phase 1: Output Layer Backward Pass Implementation
 *
 * This phase computes:
 * 1. Cross-entropy gradient: dL/d_logits = softmax(logits) - one_hot(targets)
 * 2. LM Head backward: grad_encoder_out = grad_logits @ W_lm_head^T
 * 3. LM Head weight gradient: grad_W_lm_head = grad_logits^T @ encoder_output
 *
 * TENSOR LAYOUT: All tensors are row-major (BSM format)
 * 
 * OUTPUT: ctx.current_grad points to grad_encoder_out for Phase 2
 */

#include "BackwardPhase1_OutputLayer.hpp"
#include "../../Shared/Loss/ComputeLoss/ComputeLossHost_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"
#include "../../Layers/LayernNorm/RMSNorm_Kernel_GPU.hpp"
#include "../../Common/grim_scale_buffer.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <vector>

namespace GRIM {
namespace Backward {

//======================================================//
//  Logging Setup
//======================================================//

namespace {
constexpr auto kModule = GRIM::Logging::ModuleId::BackwardPass;

#define P1_INFO(msg) do { std::ostringstream _oss; _oss << "[Phase1] " << msg; GRIM::Logging::EmitModuleInfo(kModule, _oss.str()); } while (0)
#define P1_WARN(msg) do { std::ostringstream _oss; _oss << "[Phase1] " << msg; GRIM::Logging::EmitModuleWarning(kModule, _oss.str()); } while (0)
#define P1_ERROR(msg) do { std::ostringstream _oss; _oss << "[Phase1] " << msg; GRIM::Logging::EmitModuleError(kModule, _oss.str()); } while (0)

// Diagnostic flag (can be disabled for production)
constexpr bool kEnableLayerDiagnostics = false;

inline void queueGradStats(const char* name,
    int layer,
    const float* grad_ptr,
    size_t size,
    float explosion_threshold,
    cudaStream_t stream) {
    if (!grad_ptr || size == 0) {
        return;
    }
    GRIM::GradStats::enqueue(name, layer, grad_ptr, size, explosion_threshold, stream);
}

} // anonymous namespace

//======================================================//
//  Phase 1 Entry Point
//======================================================//

BackwardStatus executePhase1_OutputLayer(BackwardContext& ctx) {
    P1_INFO("START batch=" << ctx.batch_size << " seq=" << ctx.seq_len 
            << " tokens=" << ctx.total_tokens);
    
    //--------------------------------------------------//
    // Validation
    //--------------------------------------------------//
    
    BackwardStatus validation = ctx.validate();
    if (validation != BackwardStatus::SUCCESS) {
        ctx.error_message = "Context validation failed";
        P1_ERROR("Context validation failed: " << statusToString(validation));
        return validation;
    }
    
    // Validate required pointers
    BWD_CHECK_PTR(ctx, ctx.training_state->grad_logits_tensor.data, "grad_logits", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->grad_encoder_tensor.data, "grad_encoder_out", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->cached_logits, "cached_logits", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->cached_targets, "cached_targets", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->cached_encoder_outputs, "cached_encoder_outputs", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->lm_head_weights.data, "lm_head_weights", -1);  // Tensor API
    
    //--------------------------------------------------//
    // Step 1: Cross-Entropy Gradient
    //--------------------------------------------------//
    
    BackwardStatus ce_status = computeCrossEntropyGradient(ctx);
    if (ce_status != BackwardStatus::SUCCESS) {
        return ce_status;
    }
    
    //--------------------------------------------------//
    // Step 2: LM Head Backward
    //--------------------------------------------------//
    
    BackwardStatus lm_status = computeLMHeadBackward(ctx);
    if (lm_status != BackwardStatus::SUCCESS) {
        return lm_status;
    }
    
    //--------------------------------------------------//
    // Step 3: Final RMSNorm Backward (Issue #33 fix)
    //--------------------------------------------------//
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    
    if (ts->final_rms_gamma.data && ts->cached_final_rms_input && ts->final_rms_gamma_grads()) {
        P1_INFO("Computing final RMSNorm backward...");
        
        const float eps = 1e-5f;
        
        // grad_encoder_out currently holds gradient from LM head (w.r.t. normalized output)
        // We need to backprop through RMSNorm:
        //   - Update grad_encoder_out to be gradient w.r.t. PRE-normalized input
        //   - Accumulate gamma gradients
        RMSNormBackwardParams rms_params{};
        rms_params.input = TensorContract::TensorView::make_BSM(
            ts->cached_final_rms_input, ctx.total_tokens, cfg->d_model, "cached_final_rms_input");
        rms_params.grad_output = TensorContract::TensorView::make_BSM(
            ts->grad_encoder_tensor.data, ctx.total_tokens, cfg->d_model, "grad_encoder_out");
        rms_params.gamma = TensorContract::TensorView::make_BSM(
            ts->final_rms_gamma.data, 1, cfg->d_model, "final_rms_gamma");
        rms_params.grad_input = TensorContract::TensorView::make_BSM(
            ts->grad_encoder_tensor.data, ctx.total_tokens, cfg->d_model, "grad_encoder_out_inplace");
        rms_params.grad_gamma = TensorContract::TensorView::make_BSM(
            ts->final_rms_gamma_grads(), 1, cfg->d_model, "final_rms_gamma_grads");
        rms_params.epsilon = eps;
        rms_params.stream = ts->stream_ctrl.getPrimaryStream();
        
        launchRMSNormBackward(rms_params);
        
        if (ctx.enable_grad_checks) {
            queueGradStats(
                "final_rms_gamma_grads",
                -1,
                ts->final_rms_gamma_grads(),
                static_cast<size_t>(cfg->d_model),
                ctx.explosion_threshold,
                ts->stream_ctrl.getPrimaryStream());
        }
        
        P1_INFO("Final RMSNorm backward complete");
    }
    
    //--------------------------------------------------//
    // Set output for Phase 2
    //--------------------------------------------------//
    
    ctx.current_grad = ctx.training_state->grad_encoder_tensor.data;
    ctx.phase1_status = BackwardStatus::SUCCESS;
    
    // DIAGNOSTIC: Queue grad_encoder_out stats before passing to Phase 2 (no sync)
    if (kEnableLayerDiagnostics || ctx.enable_grad_checks) {
        queueGradStats(
            "phase1_grad_encoder_out",
            -1,
            ctx.training_state->grad_encoder_tensor.data,
            static_cast<size_t>(ctx.total_tokens) * ctx.config->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    P1_INFO("COMPLETE - grad_encoder_out ready for Phase 2");
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  Cross-Entropy Gradient
//======================================================//

BackwardStatus computeCrossEntropyGradient(BackwardContext& ctx) {
    P1_INFO("Computing cross-entropy gradient...");
    
    const auto* cfg = ctx.config;
    auto* ts = ctx.training_state;
    
    //--------------------------------------------------//
    // CRITICAL: UnifiedLoss already computed grad_logits during forward pass!
    // The unifiedLossKernel computes Focal+LabelSmoothing+CE gradients directly.
    // DO NOT recompute here - it would overwrite the unified gradients with
    // simple CE gradients, breaking the loss function.
    //
    // The grad_logits buffer is already populated by computeLossBatch() → 
    // computeLossHost() → launchLossPipeline() → computeUnifiedLoss().
    //--------------------------------------------------//
    
    P1_INFO("Skipping gradient recomputation - UnifiedLoss already computed grad_logits");
    
    // Validate that grad_logits buffer exists (it should, from forward pass)
    if (!ts->grad_logits_tensor.data) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::FATAL_ERROR, 
                      "grad_logits buffer is NULL - forward pass did not compute gradients", -1);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  LOGITS LAYOUT VALIDATION (TensorContract Integration)
    //  Verify grad_logits has the expected LOGITS layout [tokens x vocab_size]
    // ═══════════════════════════════════════════════════════════════════════════
    {
        auto grad_logits_view = TensorContract::TensorView::make_LOGITS(
            ts->grad_logits_tensor.data,
            ctx.total_tokens,
            cfg->vocab_size,
            "grad_logits"
        );
        P1_INFO("grad_logits validated: " << grad_logits_view.to_string());
    }
    
    // Clear sequence weights after use (they were already applied in forward)
    ts->sequence_weight_count = 0;
    
    //--------------------------------------------------//
    // Apply gradient scale (token normalization)
    //--------------------------------------------------//
    
    if (std::abs(ctx.grad_scale - 1.0f) > 1e-7f) {
        P1_INFO("Scaling grad_logits by " << ctx.grad_scale);
        
        scaleDeviceBuffer(
            ts->grad_logits_tensor.data,
            static_cast<size_t>(ctx.total_tokens) * static_cast<size_t>(cfg->vocab_size),
            ctx.grad_scale,
            ctx.training_state->stream_ctrl.getPrimaryStream()
        );

        if (cfg->numeric_head_enabled && ts->grad_numeric_tensor.data) {
            scaleDeviceBuffer(
                ts->grad_numeric_tensor.data,
                static_cast<size_t>(ctx.total_tokens),
                ctx.grad_scale,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
    }
    
    //--------------------------------------------------//
    // Gradient validation
    //--------------------------------------------------//
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_logits",
            -1,
            ts->grad_logits_tensor.data,
            static_cast<size_t>(ctx.total_tokens) * cfg->vocab_size,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  LM Head Backward
//======================================================//

BackwardStatus computeLMHeadBackward(BackwardContext& ctx) {
    P1_INFO("Computing LM Head backward...");
    
    const auto* cfg = ctx.config;
    auto* ts = ctx.training_state;
    
    //--------------------------------------------------//
    // TensorContract: Validate tensors
    //--------------------------------------------------//
    
    auto encoder_out_view = TensorContract::TensorView::make_BSM(
        ts->grad_encoder_tensor.data, 
        ctx.total_tokens, 
        cfg->d_model, 
        "grad_encoder_out");
    
    //--------------------------------------------------//
    // Prepare LM Head backward params using TensorViews
    //--------------------------------------------------//
    
    LMHeadBackwardParams lm_params{};
    
    // Input tensor views (read-only during backward)
    lm_params.grad_logits = TensorContract::TensorView::make_LOGITS(
        ts->grad_logits_tensor.data, 
        ctx.total_tokens, 
        cfg->vocab_size,
        "lm_backward_grad_logits"
    );
    
    lm_params.encoder_output = TensorContract::TensorView::make_BSM(
        ts->cached_encoder_outputs, 
        ctx.total_tokens, 
        cfg->d_model,
        "lm_backward_encoder_output"
    );
    
    lm_params.weights = TensorContract::TensorView::make_BSM(
        ts->lm_head_weights.data, 
        cfg->vocab_size, 
        cfg->d_model,
        "lm_backward_weights"
    );
    
    // Output tensor views (mutable during backward)
    lm_params.grad_encoder = TensorContract::TensorView::make_BSM(
        ts->grad_encoder_tensor.data, 
        ctx.total_tokens, 
        cfg->d_model,
        "lm_backward_grad_encoder"
    );
    
    lm_params.grad_weight = TensorContract::TensorView::make_BSM(
        ts->lm_head_weight_grads(), 
        cfg->vocab_size, 
        cfg->d_model,
        "lm_backward_grad_weight"
    );
    
    // Optional bias gradient
    if (cfg->use_bias && ts->lm_head_bias.grad) {
        lm_params.grad_bias = TensorContract::TensorView::make_BSM(
            ts->lm_head_bias.grad, 
            cfg->vocab_size, 
            1,
            "lm_backward_grad_bias"
        );
    }
    
    // Centered encoder scratch for Issue #37 / #40 fix
    if (ts->encoder_workspace) {
        lm_params.centered_encoder = TensorContract::TensorView::make_BSM(
            ts->encoder_workspace, 
            ctx.total_tokens, 
            cfg->d_model,
            "lm_backward_centered_encoder"
        );
    }
    
    // Operation flags
    lm_params.accumulate = ctx.accumulate;
    lm_params.use_bias = cfg->use_bias && ts->lm_head_bias.grad != nullptr;
    lm_params.use_centering = cfg->lm_head_center_hidden_states;
    lm_params.recenter_gradients = cfg->lm_head_recenter_gradients;
    
    // Execution context
    lm_params.handle = ctx.cublas_handle;
    lm_params.stream = ts->stream_ctrl.getPrimaryStream();
    
    //--------------------------------------------------//
    // ISSUE #42 DIAGNOSTIC: Log grad_logits and encoder_output BEFORE LM head backward
    // These are the two inputs to grad_W = h^T @ grad_logits
    //--------------------------------------------------//
    {
        static int s_issue42_diag = 0;
        constexpr int kToken277 = 277;
        if (false && ++s_issue42_diag <= 5) { // DISABLED
            cudaStreamSynchronize(lm_params.stream);
            
            // Read grad_logits[t, 277] for first 10 tokens
            std::vector<float> grad_logits_277(std::min(10, ctx.total_tokens));
            for (int t = 0; t < static_cast<int>(grad_logits_277.size()); ++t) {
                const size_t offset = static_cast<size_t>(t) * cfg->vocab_size + kToken277;
                cudaMemcpy(&grad_logits_277[t], ts->grad_logits_tensor.data + offset, sizeof(float), cudaMemcpyDeviceToHost);
            }
            
            // Read encoder_output[t, 0:3] for first 5 tokens (to see h values)
            std::vector<float> enc_sample(std::min(5, ctx.total_tokens) * 3);
            for (int t = 0; t < std::min(5, ctx.total_tokens); ++t) {
                cudaMemcpy(&enc_sample[t*3], ts->cached_encoder_outputs + static_cast<size_t>(t) * cfg->d_model, 
                           3 * sizeof(float), cudaMemcpyDeviceToHost);
            }
            
            // Sum all grad_logits[t, 277]
            double grad_logits_277_sum = 0.0;
            std::vector<float> all_grad_277(ctx.total_tokens);
            for (int t = 0; t < ctx.total_tokens; ++t) {
                const size_t offset = static_cast<size_t>(t) * cfg->vocab_size + kToken277;
                cudaMemcpy(&all_grad_277[t], ts->grad_logits_tensor.data + offset, sizeof(float), cudaMemcpyDeviceToHost);
                grad_logits_277_sum += all_grad_277[t];
            }
            
            // Count targets == 277
            std::vector<int> targets(ctx.total_tokens);
            cudaMemcpy(targets.data(), ts->cached_targets, ctx.total_tokens * sizeof(int), cudaMemcpyDeviceToHost);
            int target_277_count = 0;
            double grad_when_target_277 = 0.0, grad_when_not_target_277 = 0.0;
            for (int t = 0; t < ctx.total_tokens; ++t) {
                if (targets[t] == kToken277) {
                    target_277_count++;
                    grad_when_target_277 += all_grad_277[t];
                } else if (targets[t] >= 0) {
                    grad_when_not_target_277 += all_grad_277[t];
                }
            }
            
            std::ostringstream oss;
            oss << "[Issue42-INPUTS] call=" << s_issue42_diag << " tokens=" << ctx.total_tokens << "\n";
            oss << "  grad_logits[0:10, 277]: ";
            for (size_t i = 0; i < grad_logits_277.size(); ++i) {
                oss << std::scientific << std::setprecision(4) << grad_logits_277[i] << " ";
            }
            oss << "\n  sum(grad_logits[:, 277])=" << std::scientific << std::setprecision(6) << grad_logits_277_sum;
            oss << "\n  target_277_count=" << target_277_count << "/" << ctx.total_tokens 
                << " (" << std::fixed << std::setprecision(1) << (100.0 * target_277_count / ctx.total_tokens) << "%)";
            oss << "\n  grad_when_target_277=" << std::scientific << std::setprecision(6) << grad_when_target_277 
                << " (SHOULD BE NEGATIVE)";
            oss << "\n  grad_when_NOT_target_277=" << std::scientific << std::setprecision(6) << grad_when_not_target_277
                << " (positive is OK)";
            oss << "\n  encoder_output[0:5, 0:3]: ";
            for (int t = 0; t < std::min(5, ctx.total_tokens); ++t) {
                oss << "[" << enc_sample[t*3] << "," << enc_sample[t*3+1] << "," << enc_sample[t*3+2] << "] ";
            }
            P1_INFO(oss.str());
        }
    }
    
    //--------------------------------------------------//
    // Execute LM Head backward (throws on error)
    //--------------------------------------------------//
    
    launchLMHeadBackward(lm_params);

    // === ISSUE #39 DIAGNOSTIC: Check grad_W[277] IMMEDIATELY after LM head backward ===
    // This is BEFORE any accumulation scaling or other processing
    {
        static int s_p1_diag = 0;
        constexpr int kToken277 = 277;
        if (false && ++s_p1_diag <= 10 && ts->lm_head_weight_grads()) { // DISABLED
            cudaStreamSynchronize(lm_params.stream);
            const size_t row_offset = static_cast<size_t>(kToken277) * cfg->d_model;
            std::vector<float> grad_row(cfg->d_model);
            cudaMemcpy(grad_row.data(), ts->lm_head_weight_grads() + row_offset,
                       cfg->d_model * sizeof(float), cudaMemcpyDeviceToHost);
            double sum = 0.0;
            for (int i = 0; i < cfg->d_model; ++i) sum += grad_row[i];
            P1_INFO("[Issue39-P1-DIAG] IMMEDIATELY after LMHeadBackward call=" << s_p1_diag 
                    << " grad_W[277].sum=" << std::fixed << std::setprecision(9) << sum);
        }
    }

    if (cfg->numeric_head_enabled) {
        // Tensor API: check .data and .grad fields instead of raw pointers
        if (!ts->grad_numeric_tensor.data || !ts->numeric_head_weights.data) {
            BWD_FAIL_LOUD(ctx, BackwardStatus::FATAL_ERROR,
                          "numeric head enabled but grad_predictions/weights missing", -1);
        }

        NumericHeadBackwardParams num_params{};
        
        // Input tensors
        num_params.grad_predictions = TensorContract::TensorView::make_BSM(
            ts->grad_numeric_tensor.data,
            ctx.total_tokens,
            1,
            "numeric_bwd_grad_predictions"
        );
        
        num_params.encoder_output = TensorContract::TensorView::make_BSM(
            ts->cached_encoder_outputs,
            ctx.total_tokens,
            cfg->d_model,
            "numeric_bwd_encoder_output"
        );
        
        num_params.weights = TensorContract::TensorView::make_BSM(
            ts->numeric_head_weights.data,
            cfg->d_model,
            1,
            "numeric_bwd_weights"
        );
        
        // Output tensors
        num_params.grad_encoder = TensorContract::TensorView::make_BSM(
            ts->grad_encoder_tensor.data,
            ctx.total_tokens,
            cfg->d_model,
            "numeric_bwd_grad_encoder"
        );
        
        num_params.grad_weight = TensorContract::TensorView::make_BSM(
            ts->numeric_head_weights.grad,
            cfg->d_model,
            1,
            "numeric_bwd_grad_weight"
        );
        
        // Optional bias gradient
        if (cfg->use_bias && ts->numeric_head_bias.grad) {
            num_params.grad_bias = TensorContract::TensorView::make_BSM(
                ts->numeric_head_bias.grad,
                1,
                1,
                "numeric_bwd_grad_bias"
            );
            num_params.use_bias = true;
        }
        
        // Operation flags and execution context
        num_params.accumulate = ctx.accumulate;
        num_params.handle = ctx.cublas_handle;
        num_params.stream = ts->stream_ctrl.getPrimaryStream();

        launchNumericHeadBackward(num_params);
    }
    
    //--------------------------------------------------//
    // Check for CUDA errors (no sync - use peek to avoid blocking)
    //--------------------------------------------------//
    
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), 
                   "cudaPeekAtLastError after LM Head backward", -1);
    
    //--------------------------------------------------//
    // Gradient validation
    //--------------------------------------------------//
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_encoder_out",
            -1,
            ts->grad_encoder_tensor.data,
            static_cast<size_t>(ctx.total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        if (ts->lm_head_weight_grads()) {
            queueGradStats(
                "lm_head_weight_grads",
                -1,
                ts->lm_head_weight_grads(),
                static_cast<size_t>(cfg->vocab_size) * cfg->d_model,
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
        if (cfg->numeric_head_enabled && ts->numeric_head_weights.grad) {
            queueGradStats(
                "numeric_head_weight_grads",
                -1,
                ts->numeric_head_weights.grad,
                static_cast<size_t>(cfg->d_model),
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
    }
    
    return BackwardStatus::SUCCESS;
}

} // namespace Backward
} // namespace GRIM
