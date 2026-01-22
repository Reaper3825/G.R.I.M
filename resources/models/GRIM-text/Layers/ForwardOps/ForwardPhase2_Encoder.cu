#include "ForwardPhase2_Encoder.hpp"
#include "ForwardKernels.hpp"
#include "ForwardDiagnostics.cuh"

#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio>

namespace GRIM {
namespace Forward {

namespace {

//======================================================//
// runFullEncoder - CONSOLIDATED IMPLEMENTATION
// Iterates through encoder layers directly (no delegation)
// Uses ForwardContext stream (centralized controller pattern)
//======================================================//
ForwardStatus runFullEncoder(ForwardContext& ctx) {
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;

    FWD_CHECK_PTR(ctx, ctx.gpu_encoder, "gpu_encoder", -1);
    FWD_CHECK_PTR(ctx, ts->cached_embeddings, "cached_embeddings", -1);
    FWD_CHECK_PTR(ctx, ts->encoder_workspace, "encoder_workspace", -1);

    float* encoder_output = ctx.encoder_output ? ctx.encoder_output : ts->cached_encoder_outputs;
    FWD_CHECK_PTR(ctx, encoder_output, "encoder_output", -1);

    // Zero entropy output if enabled
    if (ctx.enable_entropy_output && ts->d_entropy_output && ts->entropy_output_capacity > 0) {
        cudaMemsetAsync(ts->d_entropy_output, 0,
                        ts->entropy_output_capacity * sizeof(float),
                        ctx.stream);
    }

    // Setup Flash Attention BF16 scratch buffers
    FlashAttentionBF16Scratch fa_scratch{};
    fa_scratch.q = ts->fa_q_bf16;
    fa_scratch.k = ts->fa_k_bf16;
    fa_scratch.v = ts->fa_v_bf16;
    fa_scratch.out = ts->fa_out_bf16;
    fa_scratch.q_elems = ts->fa_q_bf16_elems;
    fa_scratch.kv_elems = ts->fa_kv_bf16_elems;

    //======================================================//
    // LAYER ITERATION - Previously in Forward_GPU.cu::forwardGPU()
    // Now consolidated here using ForwardContext stream
    //======================================================//
    const float* d_layer_input = ts->cached_embeddings;
    const int total_tokens = ctx.total_tokens;
    const int num_layers = ctx.gpu_encoder->getNumLayers();

    fprintf(stderr, "[PHASE_TIMING] Phase2 (Encoder) START: num_layers=%d total_tokens=%d\n", num_layers, total_tokens);

    // Issue #37 DIAGNOSTIC: Set W[277] reference for hidden state alignment tracking
    // Tensor API: use .data field
    if (ts->lm_head_weights.data) {
        setEncoderW277Reference(ts->lm_head_weights.data, cfg->vocab_size, cfg->d_model, ctx.stream);
        resetEncoderDiagCount();
    }
    fprintf(stderr, "[PHASE_TIMING] Phase2: W277 reference set, entering layer loop\n");

    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d/%d START\n", layer_idx, num_layers);

        // Get the encoder layer
        auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
        if (!enc_layer) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::NULL_POINTER, "encoder layer is null", layer_idx);
        }
        
        // Guarded verbose layer timing logs (set true to debug layer execution)
        constexpr bool kEnableVerboseLayerTiming = false;
        if constexpr (kEnableVerboseLayerTiming) {
            fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d: got encoder layer, setting workspace\n", layer_idx);
        }

        // Setup workspace
        LayerWorkspace<float> ws{};
        ws.data = ts->encoder_workspace;
        ws.bytes = enc_layer->requiredWorkspaceBytes(total_tokens, ctx.seq_len);
        enc_layer->setWorkspace(ws.data, ws.bytes);
        
        if constexpr (kEnableVerboseLayerTiming) {
            fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d: workspace set (%zu bytes), creating input tensor\n", layer_idx, ws.bytes);
        }

        // Wrap input as Tensor (non-owning view)
        TensorContract::TensorShape input_shape = TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model);
        Tensor input_tensor = Tensor::from_ptr(
            const_cast<float*>(d_layer_input), input_shape, false, false);  // not owned, no grad
        
        if constexpr (kEnableVerboseLayerTiming) {
            fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d: calling forward()...\n", layer_idx);
        }
        
        // HYBRID FIX: Get the layer cache for legacy backward pass
        // Pass cache pointer so forward() populates cached activations
        EncoderLayerCache* layer_cache = nullptr;
        if (ctx.layer_caches && layer_idx < ctx.layer_cache_count) {
            layer_cache = &ctx.layer_caches[layer_idx];
            if constexpr (kEnableVerboseLayerTiming) {
                fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d: using layer cache at %p\n", 
                        layer_idx, (void*)layer_cache);
            }
        }
        
        // Execute layer forward - returns Tensor, optionally populates cache
        Tensor output_tensor = enc_layer->forward(input_tensor, ctx.seq_len, ctx.stream, layer_cache);
        
        if constexpr (kEnableVerboseLayerTiming) {
            fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d: forward() returned\n", layer_idx);
        }
        
        // Copy output to encoder_output buffer for next layer / final output
        cudaMemcpyAsync(encoder_output, output_tensor.data,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
        
        if constexpr (kEnableVerboseLayerTiming) {
            fprintf(stderr, "[PHASE_TIMING] Phase2: Layer %d COMPLETE\n", layer_idx);
        }

        // === DIAGNOSTIC: Log per-layer h values ===
        // LOG THE ACTUAL h VALUES for debugging mode collapse
        constexpr bool kEnableHiddenStateDiag = false;  // Set true to enable [h_LOG] / [Token277Align] logs
        if constexpr (kEnableHiddenStateDiag) {
            const int d_model = cfg->d_model;
            
            // Copy first few h vectors to host for logging
            constexpr int kLogPositions = 5;  // Log first 5 positions
            const int positions_to_log = std::min(kLogPositions, total_tokens);
            std::vector<float> h_sample(positions_to_log * d_model);
            cudaMemcpyAsync(h_sample.data(), encoder_output, 
                           positions_to_log * d_model * sizeof(float),
                           cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            
            // Compute h_sum (sum across d_model) for each position
            for (int t = 0; t < positions_to_log; ++t) {
                const float* h_t = &h_sample[t * d_model];
                double h_sum = 0.0, h_sq_sum = 0.0;
                float h_min = h_t[0], h_max = h_t[0];
                for (int i = 0; i < d_model; ++i) {
                    h_sum += h_t[i];
                    h_sq_sum += h_t[i] * h_t[i];
                    h_min = std::min(h_min, h_t[i]);
                    h_max = std::max(h_max, h_t[i]);
                }
                double h_mean = h_sum / d_model;
                double h_norm = std::sqrt(h_sq_sum);
                
                // Log first 8 and last 4 values of h
                fprintf(stderr, "[h_LOG] layer=%d pos=%d h_sum=%.6f h_mean=%.6f h_norm=%.4f min=%.4f max=%.4f\n",
                        layer_idx, t, h_sum, h_mean, h_norm, h_min, h_max);
                fprintf(stderr, "[h_LOG] layer=%d pos=%d h[0:7]=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                        layer_idx, t, h_t[0], h_t[1], h_t[2], h_t[3], h_t[4], h_t[5], h_t[6], h_t[7]);
                fprintf(stderr, "[h_LOG] layer=%d pos=%d h[%d:%d]=[%.4f,%.4f,%.4f,%.4f]\n",
                        layer_idx, t, d_model-4, d_model-1,
                        h_t[d_model-4], h_t[d_model-3], h_t[d_model-2], h_t[d_model-1]);
            }
            
            // Issue #37: Track W[277] alignment after each encoder layer
            // Tensor API: use .data field
            constexpr int kToken277 = 277;  // SPACE token
            if (ts->lm_head_weights.data && kToken277 < cfg->vocab_size) {
                const float* w277 = ts->lm_head_weights.data + static_cast<size_t>(kToken277) * cfg->d_model;
                char align_buf[128];
                snprintf(align_buf, sizeof(align_buf), "after_encoder_layer_%d", layer_idx);
                FWD_DIAG_TOKEN277_ALIGNMENT(align_buf, 
                    encoder_output, w277, total_tokens, cfg->d_model, ctx.stream);
            }
        }
        // === END DIAGNOSTIC ===

        // Next layer's input is this layer's output
        d_layer_input = encoder_output;
    }
    fprintf(stderr, "[PHASE_TIMING] Phase2: All %d layers complete, running diagnostics\n", num_layers);

    // === DIAGNOSTIC: Final encoder output ===
    // Expected: After all 12 layers, variance should be ~1.2 (RMSNorm constrains growth)
    {
        const size_t total_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(cfg->d_model);
        FWD_DIAG_BUFFER_EXPECTED("encoder_final_output",
            encoder_output, total_elements,
            0.0f, 1.5f,       // var≈1.2 after 12 layers with RMSNorm
            -6.0f, 6.0f,      // ~6σ covers 99.9999% of normal dist
            ctx.stream);
    }
    // === END DIAGNOSTIC ===
    fprintf(stderr, "[PHASE_TIMING] Phase2: Diagnostics complete, checking CUDA errors\n");

    FWD_CHECK_CUDA(ctx, cudaGetLastError(), "encoder layer forward", -1);
    fprintf(stderr, "[PHASE_TIMING] Phase2: CUDA error check passed\n");

    if (ctx.enable_activation_quantization && cfg->activation_quantization.enabled) {
        const size_t tokens = static_cast<size_t>(ctx.total_tokens);
        if (tokens == 0) {
            return ForwardStatus::SUCCESS;
        }

        const auto quantize_buffer = [&](float* ptr, size_t elements) {
            if (ptr && elements > 0 && ctx.model) {
                ctx.model->applyActivationQuantization(ptr, elements);
            }
        };

        if (cfg->activation_quantization.apply_to_layer_caches && cfg->activation_quantization.enabled) {
            const size_t model_elements = tokens * static_cast<size_t>(cfg->d_model);
            const size_t ffn_elements = tokens * static_cast<size_t>(cfg->d_ff);
            for (int layer = 0; layer < cfg->num_layers; ++layer) {
                quantize_buffer(ts->cached_ln1_outputs[layer], model_elements);
                quantize_buffer(ts->cached_attn_inputs[layer], model_elements);
                quantize_buffer(ts->cached_attn_bhsd[layer], model_elements);
                quantize_buffer(ts->cached_attn_outputs[layer], model_elements);
                quantize_buffer(ts->cached_residual1_outputs[layer], model_elements);
                quantize_buffer(ts->cached_ln2_outputs[layer], model_elements);
                quantize_buffer(ts->cached_ffn_pre_gelu[layer], ffn_elements);
                quantize_buffer(ts->cached_ffn_outputs[layer], ffn_elements);
                quantize_buffer(ts->cached_layer_outputs[layer], model_elements);
            }
        }

        if (cfg->activation_quantization.apply_to_qkv_cache && cfg->activation_quantization.enabled) {
            const int head_dim = cfg->d_model / cfg->num_heads;
            int kv_heads = (ts->num_kv_heads > 0) ? ts->num_kv_heads : cfg->num_kv_heads;
            if (kv_heads <= 0) kv_heads = cfg->num_heads;
            const size_t q_elements = tokens * static_cast<size_t>(cfg->d_model);
            const size_t kv_elements = tokens * static_cast<size_t>(kv_heads * head_dim);
            for (int layer = 0; layer < cfg->num_layers; ++layer) {
                quantize_buffer(ts->cached_Q[layer], q_elements);
                quantize_buffer(ts->cached_K[layer], kv_elements);
                quantize_buffer(ts->cached_V[layer], kv_elements);
            }
        }

        if (cfg->activation_quantization.apply_to_encoder_outputs && cfg->activation_quantization.enabled) {
            const size_t elements = tokens * static_cast<size_t>(cfg->d_model);
            quantize_buffer(encoder_output, elements);
        }
    }
    fprintf(stderr, "[PHASE_TIMING] Phase2 (Encoder) COMPLETE\n");

    return ForwardStatus::SUCCESS;
}

ForwardStatus runIncrementalEncoder(ForwardContext& ctx) {
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    cudaStream_t stream = ctx.stream;

    FWD_CHECK_PTR(ctx, ctx.gpu_encoder, "gpu_encoder", -1);
    FWD_CHECK_PTR(ctx, ts->encoder_workspace, "encoder_workspace", -1);
    FWD_CHECK_PTR(ctx, ts->single_token_hidden, "single_token_hidden", -1);

    const int num_heads = cfg->num_heads;
    const int num_kv_heads = (ts->num_kv_heads > 0) ? ts->num_kv_heads :
                             (cfg->num_kv_heads > 0 ? cfg->num_kv_heads : num_heads);
    if (num_heads <= 0 || num_kv_heads <= 0 || cfg->d_model <= 0) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "Invalid head configuration", -1);
    }
    if (cfg->d_model % num_heads != 0) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "d_model not divisible by num_heads", -1);
    }

    const int d_head = cfg->head_dim;  // Use pre-computed value from config
    const int kv_dim = num_kv_heads * d_head;
    const int max_kv_len = ts->kv_cache_capacity;
    const int kv_len = ctx.query_pos + 1;

    const bool qk_norm_enabled = HyperParameters::QK_NORMALIZATION_ENABLED;
    const float qk_norm_scale = HyperParameters::QK_NORM_SCALE;
    const float softmax_temperature = HyperParameters::SOFTMAX_TEMPERATURE;
    const float base_scale = qk_norm_enabled ? qk_norm_scale : (1.0f / sqrtf(static_cast<float>(d_head)));
    const float inv_temperature = 1.0f / fmaxf(softmax_temperature, 0.1f);

    int rotary_dim = 0;
    const float* rope_inv_freq = nullptr;
    const float* alibi_slopes = nullptr;
    if (ctx.alibi && ctx.alibi->isInitialized()) {
        const auto pos_type = ctx.alibi->getType();
        if (pos_type == PositionalEncodingType::ROPE || pos_type == PositionalEncodingType::ALIBI_ROPE) {
            rope_inv_freq = ctx.alibi->getRoPEFreqs();
            rotary_dim = d_head;
        }
        if (pos_type == PositionalEncodingType::ALIBI || pos_type == PositionalEncodingType::ALIBI_ROPE) {
            alibi_slopes = ctx.alibi->getSlopes();
        }
    }

    float* d_hidden = ts->single_token_hidden;
    float* d_logits = ts->single_token_logits;

    float* d_Q = ts->encoder_workspace;
    float* d_new_K = d_Q + cfg->d_model;
    float* d_new_V = d_new_K + kv_dim;
    float* d_attn_scores = d_new_V + kv_dim;  // [num_heads, max_kv_len]
    float* d_attn_out = d_attn_scores + num_heads * max_kv_len;
    float* d_ffn_hidden = d_attn_out + cfg->d_model;
    float* d_temp = d_ffn_hidden + cfg->d_ff;

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;

    for (int layer = 0; layer < cfg->num_layers; ++layer) {
        auto* enc_layer = ctx.gpu_encoder->getLayer(layer);
        if (!enc_layer) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "gpu_encoder layer is null", layer);
        }

        const float* rms1_gamma = enc_layer->getRMS1Gamma();
        const float* rms2_gamma = enc_layer->getRMS2Gamma();
        const float* W_qkv = enc_layer->getAttnWqkv();
        const float* b_qkv = enc_layer->getAttnBqkv();
        const float* W_o = enc_layer->getAttnWo();
        const float* b_o = enc_layer->getAttnBo();
        const float* W_1 = enc_layer->getFFNW1();
        const float* b_1 = enc_layer->getFFNB1();
        const float* W_2 = enc_layer->getFFNW2();
        const float* b_2 = enc_layer->getFFNB2();

        const float* alpha_q = (layer < static_cast<int>(ts->attn_alpha_q.size())) ? ts->attn_alpha_q[layer] : nullptr;
        const float* alpha_k = (layer < static_cast<int>(ts->attn_alpha_k.size())) ? ts->attn_alpha_k[layer] : nullptr;

        launchSingleTokenRMSNorm(d_hidden, rms1_gamma, d_temp, cfg->d_model, 1e-5f, stream);

        FWD_CHECK_CUBLAS(ctx, cublasSgemv(ctx.cublas_handle,
                                          CUBLAS_OP_T,
                                          cfg->d_model,
                                          cfg->d_model,
                                          &alpha,
                                          W_qkv,
                                          cfg->d_model,
                                          d_temp,
                                          1,
                                          &beta_zero,
                                          d_Q,
                                          1),
                         "Q projection", layer);
        if (b_qkv && cfg->use_bias) {
            launchSingleTokenResidual(d_Q, b_qkv, d_Q, cfg->d_model, stream);
        }

        const float* W_k = W_qkv + static_cast<size_t>(cfg->d_model) * cfg->d_model;
        FWD_CHECK_CUBLAS(ctx, cublasSgemv(ctx.cublas_handle,
                                          CUBLAS_OP_T,
                                          cfg->d_model,
                                          kv_dim,
                                          &alpha,
                                          W_k,
                                          cfg->d_model,
                                          d_temp,
                                          1,
                                          &beta_zero,
                                          d_new_K,
                                          1),
                         "K projection", layer);
        if (b_qkv && cfg->use_bias) {
            const float* b_k = b_qkv + cfg->d_model;
            launchSingleTokenResidual(d_new_K, b_k, d_new_K, kv_dim, stream);
        }

        const float* W_v = W_k + static_cast<size_t>(kv_dim) * cfg->d_model;
        FWD_CHECK_CUBLAS(ctx, cublasSgemv(ctx.cublas_handle,
                                          CUBLAS_OP_T,
                                          cfg->d_model,
                                          kv_dim,
                                          &alpha,
                                          W_v,
                                          cfg->d_model,
                                          d_temp,
                                          1,
                                          &beta_zero,
                                          d_new_V,
                                          1),
                         "V projection", layer);
        if (b_qkv && cfg->use_bias) {
            const float* b_v = b_qkv + cfg->d_model + kv_dim;
            launchSingleTokenResidual(d_new_V, b_v, d_new_V, kv_dim, stream);
        }

        float* layer_K_cache = ts->cached_K[layer];
        float* layer_V_cache = ts->cached_V[layer];
        FWD_CHECK_PTR(ctx, layer_K_cache, "cached_K layer", layer);
        FWD_CHECK_PTR(ctx, layer_V_cache, "cached_V layer", layer);

        launchAppendKVCacheGQA(d_new_K, layer_K_cache, num_kv_heads, d_head, max_kv_len, ctx.query_pos, stream);
        launchAppendKVCacheGQA(d_new_V, layer_V_cache, num_kv_heads, d_head, max_kv_len, ctx.query_pos, stream);

        launchIncrementalAttentionScoresFull(
            d_Q,
            layer_K_cache,
            d_attn_scores,
            alpha_q,
            alpha_k,
            alibi_slopes,
            rope_inv_freq,
            num_heads,
            num_kv_heads,
            d_head,
            kv_len,
            max_kv_len,
            ctx.query_pos,
            rotary_dim,
            base_scale,
            inv_temperature,
            qk_norm_enabled,
            stream);

        launchIncrementalSoftmax(d_attn_scores, num_heads, kv_len, stream);

        launchIncrementalAttentionOutputGQA(
            d_attn_scores,
            layer_V_cache,
            d_attn_out,
            num_heads,
            num_kv_heads,
            d_head,
            kv_len,
            max_kv_len,
            stream);

        FWD_CHECK_CUBLAS(ctx, cublasSgemv(ctx.cublas_handle,
                                          CUBLAS_OP_T,
                                          cfg->d_model,
                                          cfg->d_model,
                                          &alpha,
                                          W_o,
                                          cfg->d_model,
                                          d_attn_out,
                                          1,
                                          &beta_zero,
                                          d_temp,
                                          1),
                         "W_o projection", layer);
        if (b_o && cfg->use_bias) {
            launchSingleTokenResidual(d_temp, b_o, d_temp, cfg->d_model, stream);
        }

        launchSingleTokenResidual(d_hidden, d_temp, d_hidden, cfg->d_model, stream);

        launchSingleTokenRMSNorm(d_hidden, rms2_gamma, d_temp, cfg->d_model, 1e-5f, stream);

        FWD_CHECK_CUBLAS(ctx, cublasSgemv(ctx.cublas_handle,
                                          CUBLAS_OP_T,
                                          cfg->d_model,
                                          cfg->d_ff,
                                          &alpha,
                                          W_1,
                                          cfg->d_model,
                                          d_temp,
                                          1,
                                          &beta_zero,
                                          d_ffn_hidden,
                                          1),
                         "FFN W1", layer);
        if (b_1 && cfg->use_bias) {
            launchSingleTokenResidual(d_ffn_hidden, b_1, d_ffn_hidden, cfg->d_ff, stream);
        }

        launchSingleTokenGELU(d_ffn_hidden, d_ffn_hidden, cfg->d_ff, stream);

        FWD_CHECK_CUBLAS(ctx, cublasSgemv(ctx.cublas_handle,
                                          CUBLAS_OP_T,
                                          cfg->d_ff,
                                          cfg->d_model,
                                          &alpha,
                                          W_2,
                                          cfg->d_ff,
                                          d_ffn_hidden,
                                          1,
                                          &beta_zero,
                                          d_temp,
                                          1),
                         "FFN W2", layer);
        if (b_2 && cfg->use_bias) {
            launchSingleTokenResidual(d_temp, b_2, d_temp, cfg->d_model, stream);
        }

        launchSingleTokenResidual(d_hidden, d_temp, d_hidden, cfg->d_model, stream);
    }

    ts->kv_cache_len = kv_len;
    ctx.encoder_output = d_hidden;
    ctx.logits_output = d_logits;
    return ForwardStatus::SUCCESS;
}

} // anonymous namespace

ForwardStatus executePhase2_Encoder(ForwardContext& ctx) {
    FWD_INFO("[ForwardPhase2] START mode=" << modeToString(ctx.mode));

    ForwardStatus validation = ctx.validate();
    if (validation != ForwardStatus::SUCCESS) {
        ctx.error_message = "Phase 2 context validation failed";
        FWD_ERROR("[ForwardPhase2] Context validation failed: " << statusToString(validation));
        return validation;
    }

    ForwardStatus status = ForwardStatus::SUCCESS;
    try {
        if (ctx.mode == ForwardMode::DecodeIncremental) {
            status = runIncrementalEncoder(ctx);
        } else {
            status = runFullEncoder(ctx);
        }
    } catch (const std::exception& ex) {
        FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, ex.what(), -1);
    }

    ctx.phase2_status = status;
    if (status == ForwardStatus::SUCCESS) {
        FWD_INFO("[ForwardPhase2] COMPLETE");
    }
    return status;
}

} // namespace Forward
} // namespace GRIM
