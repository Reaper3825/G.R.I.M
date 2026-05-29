#pragma once
//======================================================//
//  EncoderSelfAttention_GPU.hpp
//  Encoder self-attention facade for GRIM-text.
//
//  Ownership boundary:
//    - Encoding_GPU.cu orchestrates the encoder block only.
//    - This attention facade owns the QKV -> RoPE -> SDPA -> output projection path.
//    - TensorContract/autograd owns the actual tape nodes and GradFns.
//======================================================//

#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM::Attention {

struct EncoderSelfAttentionWeights {
    const Tensor& W_qkv;
    const Tensor& b_qkv;
    const Tensor& W_o;
    const Tensor& b_o;
};

struct EncoderSelfAttentionForwardRequest {
    const Batching::BatchPayload& payload;
    const HyperParameters::EncoderSelfAttentionHP& hp;
    const HyperParameters::FlashAttentionRuntimeHP& flash_attention;
    const PBM::PBMState& pbm;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    std::uint64_t batch_idx = 0;
    bool dropout_enabled = false;
    int layer_idx = 0;
};

void encoderSelfAttentionForward(
    const Tensor& norm_input,
    EncoderSelfAttentionWeights weights,
    const EncoderSelfAttentionForwardRequest& request,
    Forward::ModelForwardOutputs& forward_outputs);

}  // namespace GRIM::Attention
