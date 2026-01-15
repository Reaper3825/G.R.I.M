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
constexpr bool kEnableLayerDiagnostics = true;

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
    BWD_CHECK_PTR(ctx, ctx.training_state->grad_logits, "grad_logits", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->grad_encoder_out, "grad_encoder_out", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->cached_logits, "cached_logits", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->cached_targets, "cached_targets", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->cached_encoder_outputs, "cached_encoder_outputs", -1);
    BWD_CHECK_PTR(ctx, ctx.training_state->lm_head_weights, "lm_head_weights", -1);
    
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
    
    if (ts->final_rms_gamma && ts->cached_final_rms_input && ts->final_rms_gamma_grads) {
        P1_INFO("Computing final RMSNorm backward...");
        
        const float eps = 1e-5f;
        
        // grad_encoder_out currently holds gradient from LM head (w.r.t. normalized output)
        // We need to backprop through RMSNorm:
        //   - Update grad_encoder_out to be gradient w.r.t. PRE-normalized input
        //   - Accumulate gamma gradients
        launchRMSNormBackward(
            ts->cached_final_rms_input,      // Input to RMSNorm (pre-norm)
            ts->grad_encoder_out,             // Gradient from LM head (w.r.t. normalized output)
            ts->final_rms_gamma,             // Gamma weights
            ts->grad_encoder_out,             // Output: grad w.r.t. input (in-place OK)
            ts->final_rms_gamma_grads,       // Output: grad w.r.t. gamma
            ctx.total_tokens,
            cfg->d_model,
            eps,
            ts->stream_ctrl.getPrimaryStream()
        );
        
        if (ctx.enable_grad_checks) {
            queueGradStats(
                "final_rms_gamma_grads",
                -1,
                ts->final_rms_gamma_grads,
                static_cast<size_t>(cfg->d_model),
                ctx.explosion_threshold,
                ts->stream_ctrl.getPrimaryStream());
        }
        
        P1_INFO("Final RMSNorm backward complete");
    }
    
    //--------------------------------------------------//
    // Set output for Phase 2
    //--------------------------------------------------//
    
    ctx.current_grad = ctx.training_state->grad_encoder_out;
    ctx.phase1_status = BackwardStatus::SUCCESS;
    
    // DIAGNOSTIC: Queue grad_encoder_out stats before passing to Phase 2 (no sync)
    if (kEnableLayerDiagnostics || ctx.enable_grad_checks) {
        queueGradStats(
            "phase1_grad_encoder_out",
            -1,
            ctx.training_state->grad_encoder_out,
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
    if (!ts->grad_logits) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::FATAL_ERROR, 
                      "grad_logits buffer is NULL - forward pass did not compute gradients", -1);
    }
    
    // Clear sequence weights after use (they were already applied in forward)
    ts->sequence_weight_count = 0;
    
    //--------------------------------------------------//
    // Apply gradient scale (token normalization)
    //--------------------------------------------------//
    
    if (std::abs(ctx.grad_scale - 1.0f) > 1e-7f) {
        P1_INFO("Scaling grad_logits by " << ctx.grad_scale);
        
        scaleDeviceBuffer(
            ts->grad_logits,
            static_cast<size_t>(ctx.total_tokens) * static_cast<size_t>(cfg->vocab_size),
            ctx.grad_scale,
            ctx.training_state->stream_ctrl.getPrimaryStream()
        );

        if (cfg->numeric_head_enabled && ts->grad_numeric_predictions) {
            scaleDeviceBuffer(
                ts->grad_numeric_predictions,
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
            ts->grad_logits,
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
        ts->grad_encoder_out, 
        ctx.total_tokens, 
        cfg->d_model, 
        "grad_encoder_out");
    
    //--------------------------------------------------//
    // Prepare LM Head backward params
    //--------------------------------------------------//
    
    LMHeadBackwardParams lm_params{};
    lm_params.grad_logits = ts->grad_logits;
    lm_params.encoder_output = ts->cached_encoder_outputs;
    lm_params.grad_encoder = ts->grad_encoder_out;
    lm_params.grad_weight = ts->lm_head_weight_grads;
    lm_params.grad_bias = ts->lm_head_bias_grads;
    lm_params.weights = ts->lm_head_weights;
    lm_params.batch_size = ctx.batch_size;
    lm_params.seq_len = ctx.seq_len;
    lm_params.d_model = cfg->d_model;
    lm_params.vocab_size = cfg->vocab_size;
    lm_params.accumulate = ctx.accumulate;
    lm_params.use_bias = cfg->use_bias && ts->lm_head_bias_grads != nullptr;
    lm_params.handle = ctx.cublas_handle;
    lm_params.stream = ctx.training_state->stream_ctrl.getPrimaryStream();
    
    // Issue #37 FIX: Use centered encoder output for grad_weight computation
    // MUST match forward pass centering to ensure correct gradient!
    // We use encoder_workspace as scratch (same as forward) and recompute centering
    lm_params.centered_encoder = ts->encoder_workspace;
    lm_params.use_centering = true;
    
    //--------------------------------------------------//
    // Execute LM Head backward (throws on error)
    //--------------------------------------------------//
    
    launchLMHeadBackward(lm_params);

    if (cfg->numeric_head_enabled) {
        if (!ts->grad_numeric_predictions || !ts->numeric_head_weights) {
            BWD_FAIL_LOUD(ctx, BackwardStatus::FATAL_ERROR,
                          "numeric head enabled but grad_predictions/weights missing", -1);
        }

        NumericHeadBackwardParams num_params{};
        num_params.grad_predictions = ts->grad_numeric_predictions;
        num_params.encoder_output = ts->cached_encoder_outputs;
        num_params.grad_encoder = ts->grad_encoder_out;
        num_params.grad_weight = ts->numeric_head_weight_grads;
        num_params.grad_bias = ts->numeric_head_bias_grads;
        num_params.weights = ts->numeric_head_weights;
        num_params.total_tokens = ctx.total_tokens;
        num_params.d_model = cfg->d_model;
        num_params.accumulate = ctx.accumulate;
        num_params.use_bias = cfg->use_bias && ts->numeric_head_bias_grads != nullptr;
        num_params.handle = ctx.cublas_handle;
        num_params.stream = ctx.training_state->stream_ctrl.getPrimaryStream();

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
            ts->grad_encoder_out,
            static_cast<size_t>(ctx.total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        if (ts->lm_head_weight_grads) {
            queueGradStats(
                "lm_head_weight_grads",
                -1,
                ts->lm_head_weight_grads,
                static_cast<size_t>(cfg->vocab_size) * cfg->d_model,
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
        if (cfg->numeric_head_enabled && ts->numeric_head_weight_grads) {
            queueGradStats(
                "numeric_head_weight_grads",
                -1,
                ts->numeric_head_weight_grads,
                static_cast<size_t>(cfg->d_model),
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
    }
    
    return BackwardStatus::SUCCESS;
}

} // namespace Backward
} // namespace GRIM
