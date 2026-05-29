//======================================================//
//  Encoding_GPU.hpp
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  Architecture: Standard Pre-Norm with RMSNorm (NOT LayerNorm)
//    Input -> RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual -> Output
//  Issue #148: Sandwich Norm (post-residual RMSNorm) REMOVED to allow hidden
//  state norms to vary freely, preventing mode collapse from premature cosine alignment.
//  
//  Features:
//    - GQA (Grouped Query Attention) native support
//    - Flash Attention for memory efficiency
//    - Modular components from Layers/
//    - Strict validation with clear error messages
//    - Zero tolerance for misconfiguration
//  
//  WARNING: This file does NOT use grim_transformer_gpu.hpp
//           All attention is done via Flash Attention kernels
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <memory>

#include "../grim_layer_gpu.hpp"
#include "../FlashAttention/EncoderSelfAttention_GPU.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIM {

using Batching::BatchPayload;

struct EncodingLayerParameterViews {
    const Tensor* rms1_gamma = nullptr;
    const Tensor* rms2_gamma = nullptr;
    const Tensor* W_qkv = nullptr;
    const Tensor* b_qkv = nullptr;
    const Tensor* W_o = nullptr;
    const Tensor* b_o = nullptr;
    const Tensor* layer_scale1 = nullptr;
    const Tensor* layer_scale2 = nullptr;
    FeedForwardParameterViews ffn{};
};

//======================================================//
//  NOTE: EncodingForwardArgs DELETED per Rule 20
//  
//  Autograd forward takes Tensor directly:
//    Tensor forward(const Tensor& input, int seq_len, cudaStream_t stream);
//  
//  Autograd tape handles all caching for backward pass automatically.
//======================================================//

//======================================================//
//  NOTE: EncodingBackwardArgs DELETED per Rule 20
//  
//  Training backward pass uses BackwardPhase2_Encoder.cu
//  which has its own argument handling via BackwardContext.
//  This struct was dead code - never called by production training.
//======================================================//

//======================================================//
//  EncodingLayer - Clean Implementation (Forward-Only)
//======================================================//

class EncodingLayer final : public Layer<EncodingLayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kEncoding;
    
    //--------------------------------------------------
    // Construction
    //--------------------------------------------------
    EncodingLayer() = default;
    
    /// Self-allocating constructor — layer owns its weights
    /// @param hp_snapshot Grouped encoder construction HP from HyperparameterGroupings.hpp
    /// @param pos_encoding Positional encoding state initialized before encoder construction
    /// @param seed Base PRNG seed. Offsets: +0 W_qkv, +1 W_o, +2 FFN W1, +3 FFN W2
    /// @param init_stream CUDA stream for self-allocation during startup/model assembly
    EncodingLayer(const HyperParameters::EncoderLayerConstructionHP& hp_snapshot,
                  const PBM::PBMState& pos_encoding,
                  uint64_t seed,
                  cudaStream_t init_stream);
    
    ~EncodingLayer();
    
    // No copy (cuBLAS handle)
    EncodingLayer(const EncodingLayer&) = delete;
    EncodingLayer& operator=(const EncodingLayer&) = delete;
    
    // Move OK
    EncodingLayer(EncodingLayer&& other) noexcept;
    EncodingLayer& operator=(EncodingLayer&& other) noexcept;
    
    //--------------------------------------------------
    // Grouped HP snapshot
    //--------------------------------------------------
    const HyperParameters::EncoderLayerConstructionHP& hp() const noexcept { return hp_; }
    
   //--------------------------------------------------
   // Forward Pass - Pure Autograd writing retained layer tensors into the sink
   // backward() handled automatically via output.backward()
   //--------------------------------------------------
    /**
     * Encoder forward with autograd tracking (Issue #56 Fix)
     * 
     * Architecture:
     *   ln1 = RMSNorm(input)
     *   attn_out = Attention(ln1) + input  (residual)
     *   ln2 = RMSNorm(attn_out)
     *   output = FFN(ln2) + attn_out  (residual)
     * 
      * All intermediate tensors are written directly into the canonical per-call
      * forward sink for the active layer slot.
     * 
     * @param input [total_tokens, d_model] - encoder input (from embedding or prev layer)
        * @param payload Host-side batch geometry and sequence lengths
        * @param stream CUDA stream from the caller's forward payload/request
        * @param cublas_handle cuBLAS handle from the caller's forward payload/request
        * @param forward_outputs Canonical per-call forward sink
        * @param batch_idx Current batch index for deterministic dropout masks
        * @param dropout_enabled Explicit mode gate for dropout; batch_idx never controls mode
     * @param layer_idx Layer index within encoder stack (for equation logging, sink slot, and dropout seed)
     */
        void forward(const Tensor& input, const BatchPayload& payload,
                        cudaStream_t stream, cublasHandle_t cublas_handle,
                        Forward::ModelForwardOutputs& forward_outputs,
                        uint64_t batch_idx = 0,
                        bool dropout_enabled = false,
                        int layer_idx = 0,
                        const EncodingLayerParameterViews* parameter_views = nullptr);
    
    //--------------------------------------------------
    // Weight Management (Pattern B: self-allocated)
    //--------------------------------------------------
    bool weightsReady() const noexcept { return weights_ready_; }
    
    //--------------------------------------------------
    // Tensor Accessors (use these for ALL access)
    //--------------------------------------------------
    
    // RMSNorm (pre-norm)
    Tensor& rms1Gamma() { return rms1_gamma_; }
    Tensor& rms2Gamma() { return rms2_gamma_; }
    
    // Issue #148: Sandwich norm accessors REMOVED — post-residual RMSNorm deleted.
    // Old checkpoints with sandwich norm weights are loaded but weights are ignored.
    
    // Attention weights/biases
    Tensor& attnWqkv() { return W_qkv_; }
    Tensor& attnBqkv() { return b_qkv_; }
    Tensor& attnWo() { return W_o_; }
    Tensor& attnBo() { return b_o_; }
    
    // FFN weights (delegates to FeedForwardLayer, SwiGLU)
    // Rule 20: ffn_ MUST be initialized - crash if null
    Tensor& ffnWGate() { return ffn_->W_gate(); }
    Tensor& ffnW1() { return ffn_->W1(); }
    Tensor& ffnW2() { return ffn_->W2(); }
    Tensor& ffnB2() { return ffn_->b2(); }
    
    // Direct access to FFN layer (for autograd forward)
    FeedForwardLayer* getFfnLayer() { return ffn_.get(); }
    
    // LayerScale (Issue #109)
    Tensor& layerScale1() { return layer_scale1_; }
    Tensor& layerScale2() { return layer_scale2_; }
    
    //--------------------------------------------------
    // Workspace Budget — DELETED (Rule 20/26)
    // The autograd forward creates its own Tensors; encoder_workspace was never consumed.
    //--------------------------------------------------
    
    // NOTE: CRTP interface (forwardImpl, backwardImpl, applyGradientsImpl) DELETED per Rule 20
    // Use Tensor forward(const Tensor& input, int seq_len, cudaStream_t) directly.
    // Training backward uses BackwardPhase2_Encoder.cu.
    
private:
    void freeWeights();
    void validateReady(const char* context) const;
    void validateConstructionSnapshot(const char* context) const;
    
    /// Pattern B: self-allocate and Xavier-init all weights + create FFN
    void allocateWeights(uint64_t seed, cudaStream_t init_stream);
    
    HyperParameters::EncoderLayerConstructionHP hp_{};
    const PBM::PBMState* pos_encoding_ = nullptr;
    bool weights_ready_ = false;  // Set by allocateWeights()
    
    // RMSNorm weights (Tensor with requires_grad=true)
    Tensor rms1_gamma_;    // [d_model] - pre-attention norm
    Tensor rms2_gamma_;    // [d_model] - pre-FFN norm
    
    // Issue #148: Sandwich norm weights REMOVED (rms_post_attn_gamma_, rms_post_ffn_gamma_)
    // Standard pre-norm architecture does not use post-residual normalization.
    
    // Attention weights (Tensor with requires_grad=true)
    // W_qkv layout: [W_q: d_model x d_model][W_k: kv_dim x d_model][W_v: kv_dim x d_model]
    Tensor W_qkv_;         // [(d_model + 2*kv_dim), d_model]
    Tensor b_qkv_;         // [qkv_dim] = [d_model + 2*kv_dim]
    Tensor W_o_;           // [d_model, d_model]
    Tensor b_o_;           // [d_model]
    
    // FFN layer (owns its own weights as Tensors)
    std::unique_ptr<FeedForwardLayer> ffn_;
    
    // LayerScale parameters (Issue #109: reduces correlation buildup)
    // These are learnable per-channel gamma vectors applied before residual addition:
    //   residual1[t,d] = input[t,d] + layer_scale1[d] * attn_output[t,d]
    //   residual2[t,d] = residual1[t,d] + layer_scale2[d] * ffn_output[t,d]
    Tensor layer_scale1_;  // [1, d_model] per-channel gamma for attention
    Tensor layer_scale2_;  // [1, d_model] per-channel gamma for FFN
};

} // namespace GRIM
