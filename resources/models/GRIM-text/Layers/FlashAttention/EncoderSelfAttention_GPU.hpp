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
#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM::Attention {

struct EncoderSelfAttentionForwardRequest {
    const Batching::BatchPayload& payload;
    const HyperParameters::EncoderSelfAttentionHP& hp;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    std::uint64_t batch_idx = 0;
    int layer_idx = 0;
};

void encoderSelfAttentionForward(
    const Tensor& norm_input,
    const Tensor& W_qkv,
    const Tensor& b_qkv,
    const Tensor& W_o,
    const Tensor& b_o,
    const PBM::PBMState& pbm,
    const EncoderSelfAttentionForwardRequest& request,
    Forward::ModelForwardOutputs& forward_outputs);

// Inference-only KV-cache decode/prefill attention. Identical QKV projection and
// output projection as encoderSelfAttentionForward, but the SDPA core is replaced
// by flash_attn_fwd_kvcache_rotary: RoPE is fused inside the kernel, the new
// token's rotated K / raw V are appended to the per-layer cache in place, and the
// query attends over the full cached prefix. Read-only over durable parameters
// (no grad_fn). The caller advances cache.cache_seqlens once per forward, AFTER
// all layers (see executeModelForward). Used for both prefill (seqlen_q=prompt_len,
// cache fill 0) and decode (seqlen_q=1 or K+1 for MTP verification).
void encoderSelfAttentionForwardCached(
    const Tensor& norm_input,
    const Tensor& W_qkv,
    const Tensor& b_qkv,
    const Tensor& W_o,
    const Tensor& b_o,
    const PBM::PBMState& pbm,
    const EncoderSelfAttentionForwardRequest& request,
    const KvCacheLayerView& cache,
    Forward::ModelForwardOutputs& forward_outputs);

}  // namespace GRIM::Attention
