//======================================================//
//  AttentionEpilogue.hpp
//  Shared declaration for the attention off-by-one (softmax1) epilogue.
//
//  Defined in AutogradAttention.cu (GRIM::autograd). Declared here so BOTH the
//  training SDPA forward and the inference KV-cache decode facade reapply the
//  identical softmax1 post-process: O_obo = O_std·σ(lse), lse_obo = softplus(lse).
//  Without this, decode under attention_off_by_one=true would diverge from the
//  trained model.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace GRIM {
namespace autograd {

// out_bshd: [B, S, H, D] bf16 (scaled in place); lse_bhs: [B, H, S] fp32 (softplus'd in place).
void launchAttentionOffByOneEpilogue(
    __nv_bfloat16* out_bshd, float* lse_bhs,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM

#endif  // USE_CUDA
