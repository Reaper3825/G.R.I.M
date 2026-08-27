//======================================================//
//  AtomInsertionLoss.hpp
//  OPEN-type + EXIT loss assembled from autograd primitives
//======================================================//

#pragma once

#include "../../Batching/BatchDeviceBindings.hpp"
#include "../../Batching/BatchPayload.hpp"
#include "../../Forward/ModelForwardOutputs.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM::AtomInsertion {

// Task-local weighting input. The compiled model gate selects this loss; these
// weights remain explicit caller behavior rather than durable parameters.
struct AtomInsertionLossConfig {
    float positive_label_weight = 1.0f;
    float negative_label_weight = 1.0f;

    void validate(const char* caller) const;
};

// Structural statistics are available without synchronizing the CUDA stream
// or copying the scalar loss back to the host. The normal loss boundary owns
// any eventual scalar readback.
struct AtomInsertionLossStats {
    int valid_gap_count = 0;
    int positive_label_count = 0;
    int negative_label_count = 0;
    float normalization_weight = 0.0f;
};

// full_gap_vocab_logits is [B * (S - 1), vocab_size]. This wrapper composes:
//
//   slice_columns(full logits, decision_vocab_offset, decision_class_count)
//       -> masked_binary_cross_entropy_with_logits(...)
//
// The compact OPEN-type + EXIT decision slice is retained by
// ModelForwardOutputs as the ordinary autograd input. The BCE GradFn owns saved
// sigmoid probabilities for backward; it does not borrow the slice's value
// buffer. Supervision is resolved from the scheduler-provided payload/bindings
// during backward. The returned scalar belongs at the caller's ordinary
// loss-state lifetime boundary.
Tensor atomInsertionLoss(
    Tensor& full_gap_vocab_logits,
    Forward::ModelForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool EnableAtomIdentification,
    const AtomInsertionLossConfig& config,
    AtomInsertionLossStats* out_stats,
    cudaStream_t stream);

} // namespace GRIM::AtomInsertion
