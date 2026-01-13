#include "ForwardPhase1_OutputLayer.hpp"
#include "ForwardDiagnostics.cuh"

#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"
#include "../../Layers/LayernNorm/RMSNorm_Kernel_GPU.hpp"

namespace GRIM {
namespace Forward {

ForwardStatus executePhase1_OutputLayer(ForwardContext& ctx) {
    FWD_INFO("[ForwardPhase1] START logits=" << (ctx.logits_target == ForwardLogitsTarget::FullSequence ? "full" : "last"));

    ForwardStatus validation = ctx.validate();
    if (validation != ForwardStatus::SUCCESS) {
        ctx.error_message = "Phase 1 context validation failed";
        FWD_ERROR("[ForwardPhase1] Context validation failed: " << statusToString(validation));
        return validation;
    }

    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;

    float* encoder_output = ctx.encoder_output ? ctx.encoder_output : ts->cached_encoder_outputs;
    float* logits_output = ctx.logits_output;

    if (ctx.logits_target == ForwardLogitsTarget::FullSequence) {
        if (!logits_output) {
            logits_output = ts->cached_logits;
        }
    } else {
        if (!logits_output) {
            logits_output = ts->single_token_logits;
        }
    }

    FWD_CHECK_PTR(ctx, encoder_output, "encoder_output", -1);
    FWD_CHECK_PTR(ctx, logits_output, "logits_output", -1);

    // ═══════════════════════════════════════════════════════════════════════════
    //  FINAL RMSNORM (Issue #33 fix: prevents variance explosion)
    //  Standard architecture: embedding → encoder → **final_norm** → LM head
    //  Cache the PRE-norm input for backward pass (grad w.r.t. gamma computation)
    //
    //  BUG FIX Issue #34: For LastToken mode, we must normalize the LAST token,
    //  not the first token. The pointer offset must match what LM head reads.
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Compute pointer to the encoder output that LM head will use
    float* encoder_for_lm_head;
    int lm_head_batch_size;
    int lm_head_seq_len;
    
    if (ctx.logits_target == ForwardLogitsTarget::FullSequence) {
        encoder_for_lm_head = encoder_output;
        lm_head_batch_size = ctx.batch_size;
        lm_head_seq_len = ctx.seq_len;
    } else {
        // LastToken mode - we need the last token's encoder output
        if (ctx.mode == ForwardMode::DecodeIncremental) {
            // DecodeIncremental: encoder_output is already just the single new token
            encoder_for_lm_head = encoder_output;
        } else {
            // Prefill/DecodeFull: encoder_output is the full sequence, we need the last token
            if (ctx.seq_len <= 0) {
                FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "seq_len <= 0 for last-token logits", -1);
            }
            encoder_for_lm_head = encoder_output + static_cast<size_t>(ctx.seq_len - 1) * cfg->d_model;
        }
        lm_head_batch_size = 1;
        lm_head_seq_len = 1;
    }
    
    const int total_tokens = lm_head_batch_size * lm_head_seq_len;
    
    if (ts->final_rms_gamma && ts->cached_final_rms_input) {
        // Cache pre-norm input for backward pass
        // BUG FIX Issue #34: Cache and normalize encoder_for_lm_head, not encoder_output
        cudaMemcpyAsync(ts->cached_final_rms_input, encoder_for_lm_head,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
        
        // Apply final RMSNorm in-place to the encoder output that LM head will use
        const float eps = 1e-5f;  // Standard epsilon for RMSNorm
        launchRMSNormForward(ts->cached_final_rms_input, ts->final_rms_gamma, encoder_for_lm_head,
                             total_tokens, cfg->d_model, eps, ctx.stream);
        
        FWD_INFO("[ForwardPhase1] Applied final RMSNorm (tokens=" << total_tokens << ")");
    }

    LMHeadForwardParams lm_params{};
    lm_params.weights = ts->lm_head_weights;
    lm_params.bias = ts->lm_head_bias;
    lm_params.d_model = cfg->d_model;
    lm_params.vocab_size = cfg->vocab_size;
    lm_params.use_bias = cfg->use_bias && ts->lm_head_bias != nullptr;
    lm_params.handle = ctx.cublas_handle;
    lm_params.stream = ctx.stream;
    lm_params.logits = logits_output;
    lm_params.encoder_output = encoder_for_lm_head;  // Already computed above
    lm_params.batch_size = lm_head_batch_size;
    lm_params.seq_len = lm_head_seq_len;

    try {
        launchLMHeadForward(lm_params);
    } catch (const std::exception& ex) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, ex.what(), -1);
    }

    // === DIAGNOSTIC: Log encoder output and logits ===
    // Expected encoder_output: After final RMSNorm, should have mean≈0, var≈1
    // Expected logits: Should be in reasonable range [-2, 2] with final norm active
    {
        const size_t enc_elements = static_cast<size_t>(lm_params.batch_size) * 
                                    static_cast<size_t>(lm_params.seq_len) * 
                                    static_cast<size_t>(cfg->d_model);
        const size_t logit_elements = static_cast<size_t>(lm_params.batch_size) * 
                                      static_cast<size_t>(lm_params.seq_len) * 
                                      static_cast<size_t>(cfg->vocab_size);
        
        FWD_DIAG_BUFFER_EXPECTED("encoder_output (LM head input)", 
            lm_params.encoder_output, enc_elements,
            0.0f, 1.5f,    // Expected mean≈0, var≈1-1.5 (grows slightly through layers)
            -6.0f, 6.0f,   // Expected range after RMSNorm: ~6σ covers 99.9999% of normal dist
            ctx.stream);
        
        // Logits = encoder_output @ lm_head_weights.T
        // logit_var ≈ d_model × encoder_var × weight_var = 768 × 1.2 × 0.00004 ≈ 0.037
        // With vocab_size=50376, Xavier weight stddev = sqrt(2/(768+50376)) = 0.00625
        FWD_DIAG_BUFFER_EXPECTED("logits (LM head output)", 
            logits_output, logit_elements,
            0.0f, 0.05f,      // Expected var≈0.04 from matrix multiply math
            -2.0f, 2.0f,      // Tight range due to tiny weight variance
            ctx.stream);
        
        // LM head weights: Xavier init with large fan_out (vocab_size)
        // stddev = sqrt(2/(d_model+vocab_size)) = sqrt(2/(768+50376)) ≈ 0.00625
        // variance ≈ 0.00004
        const size_t weight_elements = static_cast<size_t>(cfg->d_model) * 
                                       static_cast<size_t>(cfg->vocab_size);
        const float xavier_weight_var = 2.0f / (cfg->d_model + cfg->vocab_size);  // ≈ 0.00004
        FWD_DIAG_BUFFER_EXPECTED("lm_head_weights",
            ts->lm_head_weights, weight_elements,
            0.0f, xavier_weight_var,    // Xavier: var = 2/(fan_in+fan_out)
            -0.04f, 0.04f,              // ~6 stddev range
            ctx.stream);
    }
    // === END DIAGNOSTIC ===

    if (cfg->numeric_head_enabled) {
        if (!ts->numeric_head_weights || !ts->cached_numeric_predictions) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::NULL_POINTER,
                          "numeric head enabled but weights/predictions buffer missing", -1);
        }

        NumericHeadForwardParams num_params{};
        num_params.weights = ts->numeric_head_weights;
        num_params.bias = ts->numeric_head_bias;
        num_params.d_model = cfg->d_model;
        num_params.use_bias = cfg->use_bias && ts->numeric_head_bias != nullptr;
        num_params.handle = ctx.cublas_handle;
        num_params.stream = ctx.stream;

        if (ctx.logits_target == ForwardLogitsTarget::FullSequence) {
            num_params.encoder_output = encoder_output;
            num_params.predictions = ts->cached_numeric_predictions;
            num_params.total_tokens = ctx.total_tokens;
        } else {
            if (ctx.seq_len <= 0) {
                FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "seq_len <= 0 for numeric head", -1);
            }
            const int last_index = ctx.seq_len - 1;
            num_params.encoder_output = (ctx.mode == ForwardMode::DecodeIncremental)
                ? encoder_output
                : encoder_output + static_cast<size_t>(last_index) * cfg->d_model;
            num_params.predictions = ts->cached_numeric_predictions + last_index;
            num_params.total_tokens = 1;
        }

        try {
            launchNumericHeadForward(num_params);
        } catch (const std::exception& ex) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, ex.what(), -1);
        }
    }

    if (ctx.enable_activation_quantization && cfg->activation_quantization.enabled &&
        cfg->activation_quantization.apply_to_logits && ctx.model) {
        size_t elements = 0;
        if (ctx.logits_target == ForwardLogitsTarget::FullSequence) {
            elements = static_cast<size_t>(ctx.total_tokens) * static_cast<size_t>(cfg->vocab_size);
        } else {
            elements = static_cast<size_t>(cfg->vocab_size);
        }
        if (elements > 0) {
            ctx.model->applyActivationQuantization(logits_output, elements);
        }
    }

    ctx.phase1_status = ForwardStatus::SUCCESS;
    FWD_INFO("[ForwardPhase1] COMPLETE");
    return ForwardStatus::SUCCESS;
}

} // namespace Forward
} // namespace GRIM
