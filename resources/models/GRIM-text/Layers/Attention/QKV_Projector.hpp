#pragma once
//======================================================//
//  QKV_Projector.hpp
//  Tensor reshape utility for attention output
//
//  History: QKVProjectionConfig, QKVProjectionWeights, and projection/
//  reshape-to-BHSD functions removed (Rule 26) — superseded by autograd.
//  Only launchReshapeFromBHSD remains.
//======================================================//

#include <cuda_runtime.h>

namespace GRIM {

// Reshape from [batch, heads, seq, head_dim] back to [tokens, d_model]
// Used by autograd::reshape_bhsd_to_flat (TensorContract_GPU.cu)
void launchReshapeFromBHSD(const float* src,
                           float* dst,
                           int batch_size,
                           int seq_len,
                           int num_heads,
                           int head_dim,
                           cudaStream_t stream);

} // namespace GRIM
