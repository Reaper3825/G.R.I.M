#include "ForwardPhase1_OutputLayer.hpp"

#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"

#include <exception>
#include <chrono>

namespace GRIM {
namespace Forward {

ForwardStatus executePhase1_OutputLayer(ForwardContext& ctx) {
    auto phase_start = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[PHASE_TIMING] Phase1 (Output) START\n");
    
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

    auto lmhead_start = std::chrono::high_resolution_clock::now();
    try {
        launchLMHeadForward(lm_params);
    } catch (const std::exception& ex) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, ex.what(), -1);
    }
    auto lmhead_end = std::chrono::high_resolution_clock::now();
    auto lmhead_ms = std::chrono::duration<double, std::milli>(lmhead_end - lmhead_start).count();

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
    auto phase_end = std::chrono::high_resolution_clock::now();
    auto phase_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
    fprintf(stderr, "[PHASE_TIMING] Phase1 (Output) COMPLETE: %.2f ms\n", phase_ms);
    FWD_INFO("[ForwardPhase1] COMPLETE");
    return ForwardStatus::SUCCESS;
}

} // namespace Forward
} // namespace GRIM
