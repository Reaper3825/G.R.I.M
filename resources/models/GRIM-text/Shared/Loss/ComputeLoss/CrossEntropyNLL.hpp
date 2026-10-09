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
    float* loss_sum = nullptr;      // Device scalar [1], TextLossGradFn-owned forward scratch
    int* valid_count = nullptr;     // Device scalar [1], TextLossGradFn-owned forward scratch
    float* weight_sum = nullptr;    // Device scalar [1], TextLossGradFn-owned forward scratch
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
 * loss-resource/result validation, and mean normalization. Host payload/targets
 * must already be validated by batching and the matching upload completed.
 * In the autograd path the caller
 * is TextLossGradFn::capture_inputs(), so the scratch is Category 1 tape state.
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
 * Accumulate the cross-entropy-family gradient w.r.t. logits into a live destination.
 *
 * This fuses NLL and log-softmax backward. The caller owns/initializes the
 * destination; masked rows are untouched and valid rows use +=.
 * Requires the validated forward payload and its still-live device bindings.
 */
void computeCrossEntropyBackwardToLogits(
    const float* log_probs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    float* grad_logits,
    int valid_count,
    float weight_sum,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    float grad_output_scale,
    cudaStream_t stream
);

}  // namespace autograd
}  // namespace GRIM
