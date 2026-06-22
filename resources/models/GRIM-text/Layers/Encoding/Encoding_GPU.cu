//======================================================//
//  Encoding_GPU.cu
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  This implementation:
//    - Uses Flash Attention directly (NOT GPUMultiHeadAttention)
//    - Uses modular QKV projection from Layers/Attention/
//    - Uses RMSNorm (NOT LayerNorm - no beta parameter!)
//    - Validates EVERYTHING and screams on errors
//    - GQA-native from the start
//======================================================//

#include "Encoding_GPU.hpp"
#include "EncoderDiagnostics.hpp"
#include "AblationFlags.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../FlashAttention/EncoderSelfAttention_GPU.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>     // fprintf, snprintf


namespace {
    constexpr bool kEnableEncoderStepLogs = false;  // Set true to enable [EncoderFwd] step logs
}
//======================================================// 


namespace GRIM {

//======================================================//
//  CUDA Error Checking
//======================================================//

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "CUDA ERROR at %s:%d - %s: %s", \
                 __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "cuBLAS ERROR at %s:%d - %s: status=%d", \
                 __FILE__, __LINE__, #call, static_cast<int>(status)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

namespace {

uint64_t mixEncoderForwardSeed(uint64_t batch_idx, uint64_t layer_nonce) {
    uint64_t x = (batch_idx + 1ULL) * 0x9E3779B97F4A7C15ULL;
    x ^= layer_nonce + 0xBF58476D1CE4E5B9ULL + (x << 6) + (x >> 2);
    return x;
}

void validateLayerScaleGamma(const Tensor& gamma, const char* name, int d_model, const char* context) {
    if (!gamma.data) {
        throw std::runtime_error(std::string(context) + ": " + name + " is NULL while LayerScale is enabled");
    }
    const std::string shape_context = std::string(context) + " " + name;
    gamma.shape.require(shape_context.c_str());
    if (!gamma.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(context) + ": " + name + " must be a 2D [1,d_model] gamma vector");
    }
    const auto dims = gamma.shape.as_2d();
    if (dims.rows != 1 || dims.cols != d_model) {
        throw std::runtime_error(std::string(context) + ": " + name + " must have shape [1,d_model]. expected=[1," +
                                 std::to_string(d_model) + "] got=[" + std::to_string(dims.rows) + "," +
                                 std::to_string(dims.cols) + "]");
    }
}

void validateEncodingParameters(const EncodingLayerParameterTensors* parameters,
                                const HyperParameters::EncoderLayerConstructionHP& hp,
                                const char* context) {
    if (!parameters) {
        throw std::runtime_error(std::string(context) + ": encoding_parameters is NULL - caller must pass registry-derived encoder parameter tensors");
    }
    if (!parameters->rms1_gamma.data) {
        throw std::runtime_error(std::string(context) + ": rms1_gamma parameter tensor is required");
    }
    if (!parameters->rms2_gamma.data) {
        throw std::runtime_error(std::string(context) + ": rms2_gamma parameter tensor is required");
    }
    if (!parameters->W_qkv.data) {
        throw std::runtime_error(std::string(context) + ": W_qkv parameter tensor is required");
    }
    if (!parameters->W_o.data) {
        throw std::runtime_error(std::string(context) + ": W_o parameter tensor is required");
    }
    if (hp.use_bias) {
        if (!parameters->b_qkv.data) {
            throw std::runtime_error(std::string(context) + ": hp.use_bias=true requires b_qkv parameter tensor");
        }
        if (!parameters->b_o.data) {
            throw std::runtime_error(std::string(context) + ": hp.use_bias=true requires b_o parameter tensor");
        }
    } else {
        if (parameters->b_qkv.data || parameters->b_o.data) {
            throw std::runtime_error(std::string(context) + ": hp.use_bias=false but bias parameter tensors were provided");
        }
    }
    if (hp.use_layer_scale) {
        if (!parameters->layer_scale1.data) {
            throw std::runtime_error(std::string(context) + ": hp.use_layer_scale=true requires layer_scale1 parameter tensor");
        }
        if (!parameters->layer_scale2.data) {
            throw std::runtime_error(std::string(context) + ": hp.use_layer_scale=true requires layer_scale2 parameter tensor");
        }
    } else {
        if (parameters->layer_scale1.data || parameters->layer_scale2.data) {
            throw std::runtime_error(std::string(context) + ": hp.use_layer_scale=false but layer-scale parameter tensors were provided");
        }
    }
}

void validateEncodingConstructionHP(const HyperParameters::EncoderLayerConstructionHP& hp,
                                    const char* context) {
    if (hp.d_model <= 0 || hp.num_layers <= 0 || hp.qkv_dim <= 0) {
        throw std::invalid_argument(std::string(context) +
            ": invalid encoder construction snapshot d_model=" + std::to_string(hp.d_model) +
            " num_layers=" + std::to_string(hp.num_layers) +
            " qkv_dim=" + std::to_string(hp.qkv_dim));
    }
}

}  // anonymous namespace



// NOTE: RMSNorm forward/backward are declared in RMSNorm_Kernel_GPU.hpp
// Include that header if needed. The extern declarations below are REMOVED
// as backward pass is now handled via TensorView-based RMSNormBackwardParams.

//======================================================//
//  EncodingLayer Implementation
//======================================================//

// ═══════════════════════════════════════════════════════════════════════════
//  Compute-layer constructor
//  Durable encoder tensors are registry-owned. This layer only captures HP
//  and instantiates the FFN compute sublayer.
// ═══════════════════════════════════════════════════════════════════════════
EncodingLayer::EncodingLayer(const HyperParameters::EncoderLayerConstructionHP& hp_snapshot)
    : hp_(hp_snapshot) {
    validateEncodingConstructionHP(hp_, "EncodingLayer::EncodingLayer");
    const HyperParameters::FeedForwardLayerConstructionHP ffn_hp =
        HyperParameters::feedForwardLayerConstructionHP(hp_);
    ffn_ = std::make_unique<FeedForwardLayer>(ffn_hp);
    if (!ffn_) {
        throw std::runtime_error("EncodingLayer::EncodingLayer: failed to create FeedForwardLayer compute shell");
    }
}

// NOTE: requiredWorkspaceBytes() DELETED (Rule 20/26)
// REASON: encoder_workspace was allocated by InitTrainingState.cu based on this method's output,
// but NOTHING in the forward/backward path ever consumed it. The autograd forward pass creates
// its own intermediate Tensors. This was orphaned GPU memory.

//======================================================//
//  Forward Pass - Autograd Implementation writing into ModelForwardOutputs
//======================================================//

void forwardEncodingLayer(const HyperParameters::EncoderLayerConstructionHP& hp,
                          FeedForwardLayer& ffn_compute,
                          const Tensor& input,
                          const BatchPayload& payload,
                          const PBM::PBMState& pos_encoding,
                          cudaStream_t stream,
                          cublasHandle_t cublas_handle,
                          Forward::ModelForwardOutputs& forward_outputs,
                          uint64_t batch_idx,
                          bool dropout_enabled,
                          int layer_idx,
                          const EncodingLayerParameterTensors* encoding_parameters,
                          const FeedForwardParameterTensors* ffn_parameters,
                          const KvCacheLayerView* kv_cache_view) {
    if (layer_idx < 0) {
        throw std::runtime_error("forwardEncodingLayer: layer_idx must be >= 0, got " + std::to_string(layer_idx));
    }
    if (kv_cache_view && dropout_enabled) {
        throw std::runtime_error("forwardEncodingLayer: KV-cache decode path is read-only and cannot run with dropout_enabled=true");
    }
    const size_t layer_slot = static_cast<size_t>(layer_idx);
    forward_outputs.validateLayerIndex(layer_slot, "forwardEncodingLayer");
    validateEncodingParameters(encoding_parameters, hp, "forwardEncodingLayer");
    if (!ffn_parameters) {
        throw std::runtime_error("forwardEncodingLayer: ffn_parameters is NULL - caller must pass registry-derived FFN parameter tensors");
    }
    Tensor& ln1_out = forward_outputs.ln1_out_per_layer[layer_slot];
    Tensor& residual1 = forward_outputs.residual1_per_layer[layer_slot];
    Tensor& ln2_out = forward_outputs.ln2_out_per_layer[layer_slot];
    Tensor& scaled_proj = forward_outputs.scaled_proj_per_layer[layer_slot];
    Tensor& scaled_ffn = forward_outputs.scaled_ffn_per_layer[layer_slot];
    Tensor& proj_out = forward_outputs.proj_out_per_layer[layer_slot];
    Tensor& ffn_out = forward_outputs.ffn_out_per_layer[layer_slot];
    Tensor& output = forward_outputs.output_per_layer[layer_slot];
    const float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(hp.num_layers));
    const Tensor& rms1_gamma = encoding_parameters->rms1_gamma;
    const Tensor& rms2_gamma = encoding_parameters->rms2_gamma;
    const Tensor& W_qkv = encoding_parameters->W_qkv;
    const Tensor& W_o = encoding_parameters->W_o;
    const Tensor* b_qkv = hp.use_bias ? &encoding_parameters->b_qkv : nullptr;
    const Tensor* b_o = hp.use_bias ? &encoding_parameters->b_o : nullptr;
    const Tensor* layer_scale1 = hp.use_layer_scale ? &encoding_parameters->layer_scale1 : nullptr;
    const Tensor* layer_scale2 = hp.use_layer_scale ? &encoding_parameters->layer_scale2 : nullptr;
    Tensor empty_b_qkv;
    Tensor empty_b_o;
    const Tensor& b_qkv_ref = b_qkv ? *b_qkv : empty_b_qkv;
    const Tensor& b_o_ref = b_o ? *b_o : empty_b_o;
    
    // Validate input
    if (!input.data) {
        throw std::runtime_error("forwardEncodingLayer: input.data is NULL");
    }
    if (!stream) {
        throw std::runtime_error("forwardEncodingLayer: stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error("forwardEncodingLayer: cublas_handle is NULL");
    }
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    input.shape.require("forwardEncodingLayer input");
    if (!input.shape.is_2d_layout()) {
        throw std::runtime_error("forwardEncodingLayer: input must be a 2D [total_tokens,d_model] tensor");
    }
    const auto& input_shape = input.shape.as_2d();
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] START total_tokens=%d seq_len=%d\n", 
                input_shape.rows, payload.max_seq_len);
    }
    
    // CRITICAL: Set autograd cuBLAS handle before any autograd::matmul calls.
    // The handle is supplied by the caller's forward payload/request and is thread_local.
    autograd::set_autograd_cublas_handle(cublas_handle);
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] autograd cuBLAS handle set: %p\n", (void*)cublas_handle);
    }
    
    if (input_shape.cols != hp.d_model) {
        throw std::runtime_error("forwardEncodingLayer: input d_model mismatch. "
                                 "Expected " + std::to_string(hp.d_model) + 
                                 ", got " + std::to_string(input_shape.cols));
    }
    if (input_shape.rows != payload.total_tokens) {
        throw std::runtime_error("forwardEncodingLayer: input rows (" + std::to_string(input_shape.rows) +
                                 ") != BatchPayload.total_tokens (" + std::to_string(payload.total_tokens) + ")");
    }
    
    if (hp.center_encoder_residuals) {
        if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
            throw std::runtime_error("forwardEncodingLayer: payload.seq_lengths size (" +
                                     std::to_string(payload.seq_lengths.size()) +
                                     ") != payload.batch_size (" + std::to_string(payload.batch_size) + ")");
        }
        for (int b = 0; b < payload.batch_size; ++b) {
            const int row_len = payload.seq_lengths[static_cast<size_t>(b)];
            if (row_len <= 1 || row_len > payload.max_seq_len) {
                throw std::runtime_error("forwardEncodingLayer: center_encoder_residuals invalid seq_lengths[" +
                                         std::to_string(b) + "]=" + std::to_string(row_len) +
                                         " for payload.max_seq_len=" + std::to_string(payload.max_seq_len));
            }
        }
    }
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] validated: batch=%d heads=%d kv_heads=%d head_dim=%d\n", 
                payload.batch_size, hp.num_heads, hp.num_kv_heads, hp.head_dim);
    }
    const uint64_t dropout_batch_seed = mixEncoderForwardSeed(batch_idx, static_cast<uint64_t>(layer_idx) + 1ULL);
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    // Retained in ModelForwardOutputs.
    //--------------------------------------------------
    ln1_out = autograd::rms_norm(input, rms1_gamma, hp.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1 DONE\n");
    
    //--------------------------------------------------
    // 2. Attention sublayer: ln1_out -> proj_out
    // Attention owns QKV projection, RoPE/ALiBi, SDPA, diagnostics, dropout seed,
    // BHSD flattening, and output projection. Encoder only supplies the PBM spec.
    //--------------------------------------------------
    const HyperParameters::EncoderSelfAttentionHP attention_hp =
        HyperParameters::encoderSelfAttentionHP(hp, dropout_enabled);
    Attention::EncoderSelfAttentionForwardRequest attention_request{
        payload,
        attention_hp,
        stream,
        cublas_handle,
        dropout_batch_seed,
        layer_idx
    };
    if (kv_cache_view) {
        // Inference KV-cache decode/prefill: fused-rotary attention over the
        // per-layer cache. cache_seqlens is advanced once per forward by the
        // shared forward primitive, AFTER all layers.
        Attention::encoderSelfAttentionForwardCached(
            ln1_out,
            W_qkv,
            b_qkv_ref,
            W_o,
            b_o_ref,
            pos_encoding,
            attention_request,
            *kv_cache_view,
            forward_outputs);
    } else {
        Attention::encoderSelfAttentionForward(
            ln1_out,
            W_qkv,
            b_qkv_ref,
            W_o,
            b_o_ref,
            pos_encoding,
            attention_request,
            forward_outputs);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: Attention facade DONE\n");
    
    //--------------------------------------------------
    // 6b. Post-attention sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (hp.dropout_rate > 0.0f && dropout_enabled) {
        const uint64_t attn_proj_dropout_seed = dropout_batch_seed * 2654435761ULL + 100 + layer_idx;
        const uint64_t attn_proj_dropout_mask_stream = 0x0001000000000000ULL + static_cast<uint64_t>(layer_idx);
        proj_out = autograd::dropout(proj_out, hp.dropout_rate,
                                     attn_proj_dropout_seed, stream,
                                     attn_proj_dropout_mask_stream);
    }
    proj_out = autograd::mul_scalar(proj_out, residual_scale, stream);
    
    //--------------------------------------------------
    // 7. Residual1: input + proj_out -> residual1
    // Issue #56: Store on ModelForwardOutputs
    // Issue #109: Apply LayerScale to proj_out before residual addition
    // Note: Standard pre-norm architecture (Issue #148: Sandwich Norm removed).
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1...\n");
    
    // Issue #109: LayerScale gating for attention sublayer
    const Tensor* proj_for_residual = &proj_out;
    if (hp.use_layer_scale) {
        validateLayerScaleGamma(*layer_scale1, "layer_scale1_", hp.d_model, "forwardEncodingLayer");
        scaled_proj = autograd::layer_scale(proj_out, *const_cast<Tensor*>(layer_scale1), stream);
        proj_for_residual = &scaled_proj;
    }

    // Experimental ablation (AblationFlags.hpp): zero attention's contribution
    // to the residual while keeping the autograd graph intact. The chosen
    // branch tensor is multiplied by 0 so residual1 == input (+ centering),
    // and attention parameters receive zero gradient (sublayer frozen).
    if (GRIM::Ablation::kZeroAttnResidual) {
        Tensor& attn_branch = hp.use_layer_scale ? scaled_proj : proj_out;
        attn_branch = autograd::mul_scalar(attn_branch, 0.0f, stream);
    }
    
    // ========================================================================
    // STANDARD PRE-NORM RESIDUAL (Issue #148: Sandwich Norm REMOVED)
    //
    // Architecture: residual1 = input + LayerScale(attn_out)
    //
    // Standard pre-norm transformer residual connection. Hidden state norms
    // are FREE to vary across tokens, providing an additional degree of freedom
    // that prevents premature cosine alignment (mode collapse).
    //
    // Why Sandwich Norm was removed:
    //   1. Forced ||h[t]|| = sqrt(D) for ALL tokens → removed magnitude diversity
    //   2. All tokens on a hypersphere → cosine alignment is the ONLY axis
    //   3. Initial rho(0)=0.21 (vs PyTorch 0.05-0.08) → started halfway to collapse
    //   4. Combined with causal attention prefix averaging → mode collapse by batch 3
    // ========================================================================
    residual1 = autograd::add(input, *proj_for_residual, stream);
    
    // ========================================================================
    // RESIDUAL CENTERING (Issue #118 / Mode Collapse Fix)
    //
    // WHY: Causal attention creates a shared output component from prefix tokens.
    //   Residual accumulation + RMSNorm direction preservation amplifies this
    //   shared direction through layers: ρ grows +0.01-0.04 per layer.
    //   Over 12 layers: ρ(emb)=0.05 → ρ(final)=0.44 → mode collapse.
    //
    // WHAT: center_columns_by_causal_prefix_lengths subtracts the strict-past
    //   prefix mean over VALID (unpadded) tokens for each feature WITHIN EACH
    //   BATCH ROW:
    //   h[b,0,d] = h[b,0,d]
    //   h[b,t,d] -= mean_{u < t}(h[b,u,d])   for valid t > 0
    //   h[b,t,d] = 0                         for padded t
    //   This removes the running shared direction at each layer without making
    //   sample A depend on sample B, including PAD rows, or leaking future
    //   positions into token t. Strict-past also preserves the first token
    //   instead of erasing it with mean_{u <= 0}.
    //
    // GRADIENT COST: The centering projection is lower-triangular. Backward
    //   applies its transpose within each sequence; no gradient path crosses
    //   from future inputs into earlier forward positions.
    // ========================================================================
    if (hp.center_encoder_residuals) {
        if (payload.max_seq_len <= 1) {
            throw std::runtime_error("forwardEncodingLayer: center_encoder_residuals requires payload.max_seq_len > 1; single-row column centering would erase the residual stream");
        }
        residual1 = autograd::center_columns_by_causal_prefix_lengths(
            residual1, payload.seq_lengths, payload.batch_size, payload.max_seq_len, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1 (pre-norm, no sandwich) DONE\n");
    
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    // Retained in ModelForwardOutputs.
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2...\n");
    ln2_out = autograd::rms_norm(residual1, rms2_gamma, hp.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2 DONE\n");
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out (already using autograd)
    // Retained in ModelForwardOutputs.
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN...\n");
    ffn_compute.forward(ln2_out,
                        stream, cublas_handle,
                        forward_outputs,
                        layer_idx,
                        *ffn_parameters);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN DONE\n");

    ffn_out = autograd::mul_scalar(ffn_out, residual_scale, stream);
    
    //--------------------------------------------------
    // 10. Residual2: residual1 + ffn_out -> output
    // Issue #56: The final output IS stored in intermediates too
    // for consistency, but we also return it
    // Issue #109: Apply LayerScale to ffn_out before residual addition
    // Issue #118: Apply centering to remove common direction before residual add
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2...\n");
    
    // Issue #109: LayerScale gating for FFN sublayer
    const Tensor* ffn_for_residual = &ffn_out;
    if (hp.use_layer_scale) {
        validateLayerScaleGamma(*layer_scale2, "layer_scale2_", hp.d_model, "forwardEncodingLayer");
        scaled_ffn = autograd::layer_scale(ffn_out, *const_cast<Tensor*>(layer_scale2), stream);
        ffn_for_residual = &scaled_ffn;
    }

    // Experimental ablation (AblationFlags.hpp): zero FFN's contribution to the
    // residual while keeping the autograd graph intact. The chosen branch tensor
    // is multiplied by 0 so output == residual1, and FFN parameters receive zero
    // gradient (sublayer frozen).
    if (GRIM::Ablation::kZeroFfnResidual) {
        Tensor& ffn_branch = hp.use_layer_scale ? scaled_ffn : ffn_out;
        ffn_branch = autograd::mul_scalar(ffn_branch, 0.0f, stream);
    }
    
    // ========================================================================
    // PRE-NORM RESIDUAL 
    // Architecture: output = residual1 + LayerScale(ffn_out)
    // ========================================================================
    output = autograd::add(residual1, *ffn_for_residual, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2 (pre-norm, no sandwich) DONE - layer COMPLETE\n");
    bool emitLayerResidualDiag = false;
    if (emitLayerResidualDiag){
    EncoderDiagnostics::emitLayerResidualDiagnostic({
        input,
        proj_out,
        scaled_proj,
        residual1,
        ffn_out,
        *ffn_for_residual,
        output,
        hp.use_layer_scale ? layer_scale1 : nullptr,
        hp.use_layer_scale ? layer_scale2 : nullptr,
        hp,
        payload,
        stream,
        layer_idx,
        emitLayerResidualDiag
    });
    }
    if (!output.data) {
        throw std::runtime_error("forwardEncodingLayer: result.output.data is NULL before return");
    }
} 

} // namespace GRIM
