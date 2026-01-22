//======================================================//
//  ComputeLoss_GPU.cu
//  UNIFIED LOSS PIPELINE - Production Ready
//  
//  REPLACES: Old broken sequential pipeline that overwrote gradients
//  NOW: Single unified kernel for Focal + LabelSmoothing + CrossEntropy
//======================================================//

#include "ComputeLoss_GPU.hpp"

#include "Shared/Loss/UnifiedLoss/UnifiedLoss_GPU.hpp"
#include "Shared/Loss/CrossEntropy/CrossEntropy_GPU.hpp"  // For reduceLossBuffer (legacy)
#include "Shared/TensorContract/TensorContract_GPU.hpp"  // LOGITS layout validation

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace GRIM::Loss {
namespace {

inline int totalTokens(const LossContext& ctx)
{
    return ctx.batch_size * ctx.seq_len;
}

// Store last telemetry for external access
static UnifiedLossTelemetry g_lastTelemetry = {};

} // namespace

// Accessor for telemetry
const UnifiedLossTelemetry& getLastTelemetry() {
    return g_lastTelemetry;
}

LossBreakdown launchLossPipeline(const LossContext& ctx,
                                  const LossConfig& cfg,
                                  DeviceBuffers buffers)
{
    LossBreakdown breakdown{};
    
    //=========================================================================
    // VALIDATION - Fail loud (Rule 20)
    //=========================================================================
    
    validate(ctx);  // Throws on failure
    
    const int tokens = totalTokens(ctx);
    if (tokens <= 0) {
        throw std::runtime_error("[ComputeLoss] tokens=" + std::to_string(tokens) +
            " (batch=" + std::to_string(ctx.batch_size) +
            ", seq=" + std::to_string(ctx.seq_len) + ")");
    }
    
    if (!buffers.token_losses) {
        throw std::runtime_error("[ComputeLoss] buffers.token_losses is NULL");
    }
    
    if (!buffers.grad_logits) {
        throw std::runtime_error("[ComputeLoss] buffers.grad_logits is NULL");
    }

    //=========================================================================
    // BUILD UNIFIED CONFIG
    //=========================================================================
    
    UnifiedLossConfig unified_cfg;
    unified_cfg.focal_enabled = cfg.focal.enabled;
    unified_cfg.focal_alpha = cfg.focal.alpha;
    unified_cfg.focal_gamma = cfg.focal.gamma;
    unified_cfg.smoothing_enabled = cfg.label_smoothing.enabled;
    unified_cfg.smoothing_epsilon = cfg.label_smoothing.epsilon;
    unified_cfg.strict_mode = false;  // Disabled - sync causes 14s stall per batch true=debugging
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    unified_cfg.entropy_reg_enabled = cfg.entropy_reg.enabled;
    unified_cfg.entropy_reg_lambda = cfg.entropy_reg.lambda;
    
    //=========================================================================
    // BUILD UNIFIED INPUTS
    //=========================================================================
    
    UnifiedLossInputs unified_inputs;
    unified_inputs.logits = ctx.logits;
    unified_inputs.targets = ctx.targets;
    unified_inputs.batch_size = ctx.batch_size;
    unified_inputs.seq_len = ctx.seq_len;
    unified_inputs.vocab_size = ctx.vocab_size;
    unified_inputs.sequence_weights = ctx.sequence_weights;
    unified_inputs.weight_count = ctx.sequence_weight_count;
    unified_inputs.position_byte_lengths = ctx.position_byte_lengths;  // GRMT v6
    unified_inputs.logit_bias = ctx.logit_bias;        // Issue #39: Output logit bias correction
    unified_inputs.logit_bias_update = ctx.logit_bias_update;  // Issue #39: scratch for EMA update
    unified_inputs.logit_bias_ema_alpha = ctx.logit_bias_ema_alpha;  // Issue #39: EMA decay rate
    unified_inputs.stream = ctx.stream;
    
    //=========================================================================
    // BUILD UNIFIED OUTPUTS
    //=========================================================================
    
    UnifiedLossOutputs unified_outputs;
    unified_outputs.token_losses = buffers.token_losses;
    unified_outputs.grad_logits = buffers.grad_logits;
    unified_outputs.loss_sum = buffers.scratch;  // Use scratch for reduction
    
    //=========================================================================
    // LOGITS LAYOUT VALIDATION (TensorContract Integration)
    // Verify both input logits and output grad_logits have proper LOGITS layout
    //=========================================================================
    {
        auto logits_view = TensorContract::TensorView::make_LOGITS(
            const_cast<float*>(unified_inputs.logits),  // safe: validation only
            tokens,
            ctx.vocab_size,
            "loss_input_logits"
        );
        auto grad_logits_view = TensorContract::TensorView::make_LOGITS(
            unified_outputs.grad_logits,
            tokens,
            ctx.vocab_size,
            "loss_output_grad_logits"
        );
        if (!logits_view.is_valid() || !grad_logits_view.is_valid()) {
            throw std::runtime_error("[ComputeLoss] LOGITS layout validation failed");
        }
    }
    
    //=========================================================================
    // COMPUTE UNIFIED LOSS (Rule 22 compliant - reuses context)
    //=========================================================================
    
    // Static context persists across calls (Rule 22: allocate once, reuse)
    static UnifiedLossContext loss_context;
    
    g_lastTelemetry = loss_context.compute(unified_cfg, unified_inputs, unified_outputs);
    
    if (g_lastTelemetry.error_code != UnifiedLossTelemetry::OK) {
        // Error already logged by UnifiedLossContext::compute
        fprintf(stderr, "[ComputeLoss] FATAL: UnifiedLoss failed: %s\n",
                getErrorMessage(g_lastTelemetry.error_code));
        throw std::runtime_error("ComputeLoss: UnifiedLoss failed");
    }
    
    //=========================================================================
    // POPULATE BREAKDOWN FROM TELEMETRY
    //=========================================================================
    
    // Total loss from telemetry (single sync in UnifiedLossContext::compute).
    if (g_lastTelemetry.valid_tokens > 0) {
        breakdown.cross_entropy = g_lastTelemetry.loss_mean *
            static_cast<float>(g_lastTelemetry.valid_tokens);
    } else {
        breakdown.cross_entropy = 0.0f;
    }
    
    // These components are integrated into the unified loss computation
    breakdown.label_smoothing = 0.0f;  // Integrated into CE
    breakdown.focal = 0.0f;            // Integrated into CE
    
    // Distillation/preference not supported yet (need teacher model)
    breakdown.distillation_kl = 0.0f;
    breakdown.preference_kl = 0.0f;
    
    breakdown.total = breakdown.cross_entropy;
    
    return breakdown;
}

void launchReduceLoss(const float* losses,
                      float* output,
                      int n,
                      cudaStream_t stream)
{
    reduceLossBuffer(losses, output, n, stream);
}

void launchCrossEntropyGradient(const LossContext& ctx,
                                  DeviceBuffers buffers)
{
    computeCrossEntropyGradient(ctx, buffers);
}

} // namespace GRIM::Loss
