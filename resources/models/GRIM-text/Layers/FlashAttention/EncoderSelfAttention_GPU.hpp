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
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM::Attention {

struct EncoderSelfAttentionWeights {
    Tensor& W_qkv;
    Tensor& b_qkv;
    Tensor& W_o;
    Tensor& b_o;
};

struct EncoderSelfAttentionIntermediates {
    Tensor& qkv_out;
    Tensor& Q_bhsd;
    Tensor& K_bhsd;
    Tensor& V_bhsd;
    Tensor& attn_out_bhsd;
    Tensor& attn_out;
    Tensor& proj_out;
};

struct EncoderSelfAttentionForwardRequest {
    const Batching::BatchPayload& payload;
    const HyperParameters::EncoderSelfAttentionHP& hp;
    const PBM::PBMSpec& pbm;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    std::uint64_t training_step = 0;
    bool dropout_enabled = false;
    int layer_idx = 0;
};

void encoderSelfAttentionForward(const Tensor& norm_input,
                                 EncoderSelfAttentionWeights weights,
                                 EncoderSelfAttentionIntermediates intermediates,
                                 const EncoderSelfAttentionForwardRequest& request);

}  // namespace GRIM::Attention
