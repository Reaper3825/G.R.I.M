//======================================================//
//  CrossEntropyNLL.hpp
//  Internal cross-entropy / NLL implementation for unified autograd loss
//======================================================//

#pragma once

#include "AutogradLoss.hpp"
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct CrossEntropyForwardResult {
    float mean_loss;
    int valid_count;
    float weight_sum;
};

/**
 * Compute mean cross-entropy-family loss from log-probabilities.
 *
 * This is the single forward call used by unified_loss() after log_softmax().
 * It owns per-token scratch allocation, reduction buffer allocation, host
 * readback, fail-loud validation, and mean normalization.
 */
CrossEntropyForwardResult computeCrossEntropyForwardFromLogProbs(
    const float* log_probs,
    const int* targets,
    int num_tokens,
    int vocab_size,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
);

/**
 * Compute gradient of the cross-entropy-family loss w.r.t. log-probabilities.
 *
 * This is the single backward call used by NLLLossGradFn before chaining into
 * LogSoftmaxGradFn.
 */
void computeCrossEntropyBackwardToLogProbs(
    const float* log_probs,
    const int* targets,
    float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float weight_sum,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    float grad_output_scale,
    cudaStream_t stream
);

}  // namespace autograd
}  // namespace GRIM
