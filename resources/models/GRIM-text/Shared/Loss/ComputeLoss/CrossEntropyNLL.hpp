//======================================================//
//  CrossEntropyNLL.hpp
//  Internal cross-entropy / NLL implementation for unified autograd loss
//======================================================//

#pragma once

#include "../../Batching/BatchDeviceBindings.hpp"
#include "../../Batching/BatchPayload.hpp"
#include "../../HyperParameters/HyperparameterGroupings.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace GRIM {
namespace autograd {

enum class CrossEntropyTargetSource {
    PrimaryLm
};

struct CrossEntropyTargetSelection {
    CrossEntropyTargetSource source;

    static CrossEntropyTargetSelection primaryLm() {
        return CrossEntropyTargetSelection{CrossEntropyTargetSource::PrimaryLm};
    }
};

struct CrossEntropyForwardResult {
    float mean_loss;
    int valid_count;
    float weight_sum;
};

struct CrossEntropyForwardWorkspace {
    float* loss_sum = nullptr;      // Device scalar [1], NLLLossGradFn-owned forward scratch
    int* valid_count = nullptr;     // Device scalar [1], NLLLossGradFn-owned forward scratch
    float* weight_sum = nullptr;    // Device scalar [1], NLLLossGradFn-owned forward scratch
    std::size_t loss_sum_bytes = 0;
    std::size_t valid_count_bytes = 0;
    std::size_t weight_sum_bytes = 0;
    cudaStream_t owner_stream = nullptr;
};

/**
 * Compute mean cross-entropy-family loss from log-probabilities.
 *
 * This is the single forward call used by unified_loss() after log_softmax().
 * It consumes caller-owned scalar reduction workspace, performs host readback,
 * fail-loud validation, and mean normalization. In the autograd path the caller
 * is NLLLossGradFn::capture_inputs(), so the scratch is Category 1 tape state.
 */
CrossEntropyForwardResult computeCrossEntropyForwardFromLogProbs(
    const float* log_probs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    const CrossEntropyForwardWorkspace& workspace,
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
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    float* grad_log_probs,
    int valid_count,
    float weight_sum,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    float grad_output_scale,
    cudaStream_t stream
);

}  // namespace autograd
}  // namespace GRIM
