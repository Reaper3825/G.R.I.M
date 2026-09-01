#pragma once
//======================================================//
//  LocalAtomRetrievalLoss.hpp
//  Causal sequence-local atom retrieval objective.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../Forward/ModelForwardOutputs.hpp"

#include <cuda_runtime.h>

namespace GRIM::LocalAtomRetrieval {

// Mean cross-entropy over payload-authored local-atom queries.
//
// logits [Q, K]
//   Q = payload.localAtomQueryCount()
//   K = 1 + maximum payload-authored row/type candidate-bank width
//
// AtomType is not the class axis. Each query's AtomType selects its own
// [sequence row, AtomType] bank. Within that bank:
//
//   target/slot 0               -> NO_REFERENCE
//   target/slot local_index + 1 -> that typed local candidate
//
// Only candidates whose first CLOSE position is strictly before the query
// opening position participate in the softmax denominator. Invalid/padded or
// future columns receive exactly zero gradient even if their input logits are
// finite. The reduction is a mean over all Q queries, including slot-0 rows.
//
// Returns a scalar [1, 1] Tensor. When Q is zero, returns a detached zero.
Tensor LocalAtomRetrievalLoss(
    Forward::ModelForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

} // namespace GRIM::LocalAtomRetrieval
