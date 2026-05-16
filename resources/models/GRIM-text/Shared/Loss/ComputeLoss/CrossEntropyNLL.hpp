//======================================================//
//  CrossEntropyNLL.hpp
//  Internal cross-entropy / NLL implementation for unified autograd loss
//======================================================//

#pragma once

#include "../../Batching/BatchDeviceBindings.hpp"
#include "../../Batching/BatchPayload.hpp"
#include "../../HyperParameters/HyperparameterGroupings.hpp"
#include <cuda_runtime.h>
#include <optional>

namespace GRIM {
namespace autograd {

enum class CrossEntropyTargetSource {
    PrimaryLm,
    MtpShiftedHead
};

struct CrossEntropyTargetSelection {
    CrossEntropyTargetSource source;
    std::optional<int> mtp_head_idx;

    static CrossEntropyTargetSelection primaryLm() {
        return CrossEntropyTargetSelection{CrossEntropyTargetSource::PrimaryLm, std::nullopt};
    }

    static CrossEntropyTargetSelection mtpShiftedHead(int head_idx) {
        return CrossEntropyTargetSelection{CrossEntropyTargetSource::MtpShiftedHead, head_idx};
    }
};

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
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
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
