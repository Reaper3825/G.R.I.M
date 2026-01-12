#include "ForwardPhase1_OutputLayer.hpp"
#include "ForwardDiagnostics.cuh"

#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"

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

    LMHeadForwardParams lm_params{};
    lm_params.weights = ts->lm_head_weights;
    lm_params.bias = ts->lm_head_bias;
    lm_params.d_model = cfg->d_model;
    lm_params.vocab_size = cfg->vocab_size;
    lm_params.use_bias = cfg->use_bias && ts->lm_head_bias != nullptr;
    lm_params.handle = ctx.cublas_handle;
    lm_params.stream = ctx.stream;
    lm_params.logits = logits_output;

    if (ctx.logits_target == ForwardLogitsTarget::FullSequence) {
        lm_params.encoder_output = encoder_output;
        lm_params.batch_size = ctx.batch_size;
        lm_params.seq_len = ctx.seq_len;
    } else {
        if (ctx.mode == ForwardMode::DecodeIncremental) {
            lm_params.encoder_output = encoder_output;
        } else {
            if (ctx.seq_len <= 0) {
                FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "seq_len <= 0 for last-token logits", -1);
            }
            lm_params.encoder_output = encoder_output + static_cast<size_t>(ctx.seq_len - 1) * cfg->d_model;
        }
        lm_params.batch_size = 1;
        lm_params.seq_len = 1;
    }

    try {
        launchLMHeadForward(lm_params);
    } catch (const std::exception& ex) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, ex.what(), -1);
    }

    // === DIAGNOSTIC: Log encoder output and logits ===
    // Expected encoder_output: After RMSNorm, should have mean≈0, rms≈1
    // Expected logits: Should be in reasonable range [-15, 15], mean near 0 if well-initialized
    {
        const size_t enc_elements = static_cast<size_t>(lm_params.batch_size) * 
                                    static_cast<size_t>(lm_params.seq_len) * 
                                    static_cast<size_t>(cfg->d_model);
        const size_t logit_elements = static_cast<size_t>(lm_params.batch_size) * 
                                      static_cast<size_t>(lm_params.seq_len) * 
                                      static_cast<size_t>(cfg->vocab_size);
        
        FWD_DIAG_BUFFER_EXPECTED("encoder_output (LM head input)", 
            lm_params.encoder_output, enc_elements,
            0.0f, 1.0f,    // Expected mean≈0, var≈1 after RMSNorm
            -10.0f, 10.0f, // Expected range
            ctx.stream);
        
        FWD_DIAG_BUFFER_EXPECTED("logits (LM head output)", 
            logits_output, logit_elements,
            0.0f, 10.0f,      // Expected mean≈0, var depends on init
            -20.0f, 20.0f,    // Expected range (can be larger for vocab projection)
            ctx.stream);
        
        // Also log weight stats to verify initialization
        const size_t weight_elements = static_cast<size_t>(cfg->d_model) * 
                                       static_cast<size_t>(cfg->vocab_size);
        FWD_DIAG_BUFFER_EXPECTED("lm_head_weights",
            ts->lm_head_weights, weight_elements,
            0.0f, 1.0f / cfg->d_model,  // Xavier: var = 1/d_model
            -0.1f, 0.1f,                // Expected small values
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
