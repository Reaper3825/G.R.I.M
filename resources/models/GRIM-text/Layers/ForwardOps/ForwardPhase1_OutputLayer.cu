#include "ForwardPhase1_OutputLayer.hpp"
#include "ForwardDiagnostics.cuh"

#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"
#include "../../Layers/LayernNorm/RMSNorm_Kernel_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"  // LOGITS layout validation
#include "../../Layers/HardcodedStates/HardcodedStates_GPU.hpp"  // Issue #42 diagnostic

#include <vector>
#include <algorithm>

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
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  HARDCODED HIDDEN STATES DIAGNOSTIC (Issue #42)
    //  When enabled, replaces encoder output with synthetic patterns to isolate
    //  whether mode collapse originates from encoder or LM head/gradient system.
    // ═══════════════════════════════════════════════════════════════════════════
    if (cfg->hardcoded_hidden_pattern != GRIM::LanguageModelConfig::HardcodedPattern::DISABLED) {
        static int s_batch_counter = 0;  // Track batch number for logging
        ++s_batch_counter;
        
        FWD_INFO("[ForwardPhase1] HARDCODED HIDDEN STATES: Replacing encoder output with pattern "
                 << static_cast<int>(cfg->hardcoded_hidden_pattern));
        
        GRIM::generateHardcodedStates(
            encoder_for_lm_head,                  // Overwrite the encoder output
            ts->lm_head_weights.data,             // For W[277]-based patterns
            static_cast<GRIM::HardcodedPattern>(cfg->hardcoded_hidden_pattern),
            total_tokens,
            cfg->d_model,
            cfg->vocab_size,
            s_batch_counter,
            ctx.stream
        );
        
        // Skip final RMSNorm when using hardcoded states (patterns already normalized)
        goto skip_final_rmsnorm;
    }
    
    // Tensor API: check .data field for final_rms_gamma
    if (ts->final_rms_gamma.data && ts->cached_final_rms_input) {
        // Cache pre-norm input for backward pass
        // BUG FIX Issue #34: Cache and normalize encoder_for_lm_head, not encoder_output
        cudaMemcpyAsync(ts->cached_final_rms_input, encoder_for_lm_head,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
        
        // Apply final RMSNorm in-place to the encoder output that LM head will use
        // REFACTORED: Use TensorView-based RMSNormForwardParams (Rule 20: Fail Loud)
        RMSNormForwardParams rms_fwd_params{};
        rms_fwd_params.input = TensorContract::TensorView::make_BSM(
            ts->cached_final_rms_input, total_tokens, cfg->d_model, "final_rms_input");
        rms_fwd_params.gamma = TensorContract::TensorView::make_BSM(
            ts->final_rms_gamma.data, 1, cfg->d_model, "final_rms_gamma");
        rms_fwd_params.output = TensorContract::TensorView::make_BSM(
            encoder_for_lm_head, total_tokens, cfg->d_model, "final_rms_output");
        rms_fwd_params.epsilon = 1e-5f;
        rms_fwd_params.stream = ctx.stream;
        launchRMSNormForward(rms_fwd_params);
        
        FWD_INFO("[ForwardPhase1] Applied final RMSNorm (tokens=" << total_tokens << ")");
        
        // Issue #37: Track W[277] alignment after final RMSNorm (just before LM head)
        constexpr int kToken277 = 277;  // SPACE token
        // Tensor API: check .data field for lm_head_weights
        if (ts->lm_head_weights.data && kToken277 < cfg->vocab_size) {
            const float* w277 = ts->lm_head_weights.data + static_cast<size_t>(kToken277) * cfg->d_model;
            FWD_DIAG_TOKEN277_ALIGNMENT("after_final_rmsnorm", 
                encoder_for_lm_head, w277, total_tokens, cfg->d_model, ctx.stream);
        }
    }

skip_final_rmsnorm:

    // Construct TensorViews for LMHead parameters (Rule 20: Fail Loud type safety)
    LMHeadForwardParams lm_params{};
    
    // Input tensors - BSM layout [total_tokens, d_model] for encoder output
    lm_params.encoder_output = TensorContract::TensorView::make_BSM(
        encoder_for_lm_head,
        total_tokens,
        cfg->d_model,
        "encoder_for_lm_head"
    );
    
    // Weights - BSM layout [vocab_size, d_model] for weight matrix
    lm_params.weights = TensorContract::TensorView::make_BSM(
        ts->lm_head_weights.data,
        cfg->vocab_size,
        cfg->d_model,
        "lm_head_weights"
    );
    
    // Output - LOGITS layout [total_tokens, vocab_size]
    lm_params.logits = TensorContract::TensorView::make_LOGITS(
        logits_output,
        total_tokens,
        cfg->vocab_size,
        "lm_head_logits_output"
    );
    
    // Optional bias - BSM layout [vocab_size, 1] (conceptually a column vector)
    if (cfg->use_bias && ts->lm_head_bias.data) {
        lm_params.bias = TensorContract::TensorView::make_BSM(
            ts->lm_head_bias.data,
            cfg->vocab_size,
            1,
            "lm_head_bias"
        );
        lm_params.use_bias = true;
    } else {
        lm_params.use_bias = false;
    }
    
    // Issue #37 FIX: Use encoder_workspace as scratch for zero-mean centering
    // encoder_workspace is used by Phase2_Encoder but free during Phase1_OutputLayer
    // NOW CONFIGURABLE via ai_config.json -> lm_head_centering
    if (cfg->lm_head_center_hidden_states && ts->encoder_workspace) {
        lm_params.centered_scratch = TensorContract::TensorView::make_BSM(
            ts->encoder_workspace,
            total_tokens,
            cfg->d_model,
            "centered_scratch"
        );
        lm_params.use_centering = true;
    } else {
        lm_params.use_centering = false;
    }
    
    // Execution context
    lm_params.handle = ctx.cublas_handle;
    lm_params.stream = ctx.stream;

    try {
        launchLMHeadForward(lm_params);
    } catch (const std::exception& ex) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, ex.what(), -1);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  HARDCODED HIDDEN STATES DIAGNOSTIC LOGGING (Issue #42)
    //  Log detailed diagnostics when using hardcoded patterns
    // ═══════════════════════════════════════════════════════════════════════════
    if (cfg->hardcoded_hidden_pattern != GRIM::LanguageModelConfig::HardcodedPattern::DISABLED) {
        static int s_log_batch_counter = 0;
        ++s_log_batch_counter;
        
        if ((s_log_batch_counter % cfg->hardcoded_log_every_n_batches) == 0) {
            GRIM::logHardcodedStateDiagnostics(
                encoder_for_lm_head,               // The hardcoded hidden states (raw pointer)
                ts->lm_head_weights.data,          // For W[277] analysis
                logits_output,                     // Resulting logits
                static_cast<GRIM::HardcodedPattern>(cfg->hardcoded_hidden_pattern),
                total_tokens,
                cfg->d_model,
                cfg->vocab_size,
                s_log_batch_counter,
                ctx.stream
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  LOGITS LAYOUT VALIDATION (TensorContract Integration)
    //  Create a TensorView to track the logits buffer with proper layout typing.
    //  This enables compile-time layout safety and runtime dimension validation.
    // ═══════════════════════════════════════════════════════════════════════════
    {
        auto logits_view = TensorContract::TensorView::make_LOGITS(
            logits_output,
            total_tokens,
            cfg->vocab_size
        );
        FWD_INFO("[ForwardPhase1] Logits tensor validated: " << logits_view.to_string());
    }

    // === DIAGNOSTIC: Log encoder output and logits ===
    // Expected encoder_output: After final RMSNorm, should have mean≈0, var≈1
    // Expected logits: Should be in reasonable range [-2, 2] with final norm active
    {
        const size_t enc_elements = static_cast<size_t>(total_tokens) * 
                                    static_cast<size_t>(cfg->d_model);
        const size_t logit_elements = static_cast<size_t>(total_tokens) * 
                                      static_cast<size_t>(cfg->vocab_size);
        
        FWD_DIAG_BUFFER_EXPECTED("encoder_output (LM head input)", 
            encoder_for_lm_head, enc_elements,
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
            ts->lm_head_weights.data, weight_elements,  // Tensor API
            0.0f, xavier_weight_var,    // Xavier: var = 2/(fan_in+fan_out)
            -0.04f, 0.04f,              // ~6 stddev range
            ctx.stream);
        
        // Issue #37: Log actual logit[277] values after LM head
        // This is the FINAL value that determines if model predicts SPACE
        constexpr int kToken277 = 277;
        if (kToken277 < cfg->vocab_size && total_tokens > 0) {
            // Read logit[277] for first few tokens to see actual prediction strength
            const int num_sample_tokens = std::min(10, total_tokens);
            std::vector<float> sample_logits(num_sample_tokens);
            
            // Each token's logits are vocab_size apart
            for (int t = 0; t < num_sample_tokens; ++t) {
                float logit_277;
                cudaMemcpyAsync(&logit_277, 
                    logits_output + static_cast<size_t>(t) * cfg->vocab_size + kToken277,
                    sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                sample_logits[t] = logit_277;
            }
            cudaStreamSynchronize(ctx.stream);
            
            // Find max logit and argmax for first token
            std::vector<float> first_token_logits(cfg->vocab_size);
            cudaMemcpy(first_token_logits.data(), logits_output, 
                       cfg->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            float max_logit = first_token_logits[0];
            int argmax = 0;
            float logit_277_val = first_token_logits[kToken277];
            for (int v = 1; v < cfg->vocab_size; ++v) {
                if (first_token_logits[v] > max_logit) {
                    max_logit = first_token_logits[v];
                    argmax = v;
                }
            }
            
            fprintf(stderr, "[Token277Logit] after_lm_head: logit_277=%.4f max_logit=%.4f argmax=%d is_277=%s\n",
                    logit_277_val, max_logit, argmax, (argmax == kToken277) ? "YES" : "no");
            fprintf(stderr, "[Token277Logit] sample_logits_277=[");
            for (int t = 0; t < num_sample_tokens; ++t) {
                fprintf(stderr, "%.3f%s", sample_logits[t], (t < num_sample_tokens - 1) ? ", " : "");
            }
            fprintf(stderr, "]\n");
        }
    }
    // === END DIAGNOSTIC ===

    if (cfg->numeric_head_enabled) {
        // Tensor API: check .data field
        if (!ts->numeric_head_weights.data || !ts->cached_numeric_predictions) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::NULL_POINTER,
                          "numeric head enabled but weights/predictions buffer missing", -1);
        }

        NumericHeadForwardParams num_params{};
        
        // Weights: [d_model, 1] column vector
        num_params.weights = TensorContract::TensorView::make_BSM(
            ts->numeric_head_weights.data,
            cfg->d_model,
            1,
            "numeric_fwd_weights"
        );
        
        // Optional bias: [1, 1] scalar
        if (cfg->use_bias && ts->numeric_head_bias.data) {
            num_params.bias = TensorContract::TensorView::make_BSM(
                ts->numeric_head_bias.data,
                1,
                1,
                "numeric_fwd_bias"
            );
            num_params.use_bias = true;
        }
        
        // Execution context
        num_params.handle = ctx.cublas_handle;
        num_params.stream = ctx.stream;

        if (ctx.logits_target == ForwardLogitsTarget::FullSequence) {
            // Encoder output: [total_tokens, d_model]
            num_params.encoder_output = TensorContract::TensorView::make_BSM(
                encoder_output,
                ctx.total_tokens,
                cfg->d_model,
                "numeric_fwd_encoder_output"
            );
            // Predictions: [total_tokens, 1]
            num_params.predictions = TensorContract::TensorView::make_BSM(
                ts->cached_numeric_predictions,
                ctx.total_tokens,
                1,
                "numeric_fwd_predictions"
            );
        } else {
            if (ctx.seq_len <= 0) {
                FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "seq_len <= 0 for numeric head", -1);
            }
            const int last_index = ctx.seq_len - 1;
            float* last_encoder_output = (ctx.mode == ForwardMode::DecodeIncremental)
                ? encoder_output
                : encoder_output + static_cast<size_t>(last_index) * cfg->d_model;
            
            // Last token encoder output: [1, d_model]
            num_params.encoder_output = TensorContract::TensorView::make_BSM(
                last_encoder_output,
                1,
                cfg->d_model,
                "numeric_fwd_encoder_output_last"
            );
            // Single prediction: [1, 1]
            num_params.predictions = TensorContract::TensorView::make_BSM(
                ts->cached_numeric_predictions + last_index,
                1,
                1,
                "numeric_fwd_predictions_last"
            );
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
