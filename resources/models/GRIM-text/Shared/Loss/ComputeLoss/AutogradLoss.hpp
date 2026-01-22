//======================================================//
//  AutogradLoss.hpp
//  Autograd-enabled cross-entropy loss computation
//  
//  This replaces the legacy float-returning computeLossHost() with
//  a Tensor-returning version that builds the computation graph
//  for automatic gradient propagation.
//======================================================//

#pragma once

#include "../../TensorContract/TensorContract_GPU.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM {
namespace autograd {

/**
 * Compute cross-entropy loss with autograd support
 * 
 * Forward: loss = mean(-log(softmax(logits)[target]))
 * Backward: grad_logits = (softmax(logits) - one_hot(target)) / num_valid_tokens
 * 
 * @param logits      [total_tokens, vocab_size] - raw logits from LM head
 * @param targets     [total_tokens] - target token IDs (on GPU)
 * @param valid_mask  [total_tokens] - 1.0 for valid tokens, 0.0 for padding (optional)
 * @param num_tokens  Number of tokens
 * @param vocab_size  Vocabulary size
 * @param stream      CUDA stream
 * @return Scalar loss tensor with grad_fn attached (if logits.requires_grad)
 */
Tensor cross_entropy_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
);

/**
 * Kernel to compute cross-entropy forward pass (loss value)
 * Uses numerically stable log-softmax computation
 */
void launchCrossEntropyForward(
    const float* logits,        // [tokens, vocab_size]
    const int* targets,         // [tokens]
    const float* valid_mask,    // [tokens] optional (nullptr = all valid)
    float* per_token_loss,      // [tokens] output per-token losses
    float* loss_sum,            // scalar output
    int* valid_count,           // scalar output
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
);

/**
 * Kernel to compute cross-entropy backward pass (gradient w.r.t. logits)
 * Computes: grad_logits[t,v] = (softmax[t,v] - (v==target[t])) * valid_mask[t] / valid_count
 */
void launchCrossEntropyBackward(
    const float* logits,        // [tokens, vocab_size]
    const int* targets,         // [tokens]
    const float* valid_mask,    // [tokens] optional
    float* grad_logits,         // [tokens, vocab_size] output
    int num_tokens,
    int vocab_size,
    int valid_count,            // for normalization
    cudaStream_t stream
);

}  // namespace autograd
}  // namespace GRIM
