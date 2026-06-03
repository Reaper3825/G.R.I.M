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
    
    /// Compute-layer constructor.
    /// Durable encoder/FFN parameters are owned by ParameterRegistry::StartupParameterRegistry;
    /// callers must pass explicit registry-derived views into forward().
    /// @param hp_snapshot Grouped encoder construction HP from HyperparameterGroupings.hpp
    explicit EncodingLayer(const HyperParameters::EncoderLayerConstructionHP& hp_snapshot);
    
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
          * @param pos_encoding Borrowed Phase1-owned PBM state for the attention sublayer
        * @param stream CUDA stream from the caller's forward payload/request
        * @param cublas_handle cuBLAS handle from the caller's forward payload/request
        * @param forward_outputs Canonical per-call forward sink
        * @param batch_idx Current batch index for deterministic dropout masks
        * @param dropout_enabled Explicit mode gate for dropout; batch_idx never controls mode
     * @param layer_idx Layer index within encoder stack (for equation logging, sink slot, and dropout seed)
     */
        void forward(const Tensor& input, const BatchPayload& payload,
                const PBM::PBMState& pos_encoding,
                        cudaStream_t stream, cublasHandle_t cublas_handle,
                        Forward::ModelForwardOutputs& forward_outputs,
                        uint64_t batch_idx = 0,
                        bool dropout_enabled = false,
                        int layer_idx = 0,
                        const EncodingLayerParameterViews* parameter_views = nullptr);
    
    //--------------------------------------------------
    // Compute-layer readiness
    //--------------------------------------------------
    bool weightsReady() const noexcept { return weights_ready_; }

    // Durable encoder trainable tensors are registry-owned. Shared forward,
    // serialization, telemetry, and autograd diagnostics must read them through
    // ParameterRegistry::StartupParameterRegistry instead of this compute layer.
    
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

    /// Create FFN compute sublayer; durable trainable tensors live on the registry owner.
    void initializeComputeLayer();
    
    HyperParameters::EncoderLayerConstructionHP hp_{};
    bool weights_ready_ = false;  // Set by initializeComputeLayer()
    
    // FFN compute layer. Durable FFN tensors are supplied through explicit forward views.
    std::unique_ptr<FeedForwardLayer> ffn_;
};

} // namespace GRIM
