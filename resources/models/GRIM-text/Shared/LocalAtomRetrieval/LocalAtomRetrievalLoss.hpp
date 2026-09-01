#pragma once
//======================================================//
//  LocalAtomRetrievalLoss.hpp
//  Standalone causal selector objective.
//
//  This file intentionally has no model-forward or training-loop wiring. It
//  documents the proposed objective boundary for LocalAtomRetrieval.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

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
//   target/class 0               -> NO_REFERENCE
//   target/class local_index + 1 -> that typed local candidate
//
// Only candidates whose first CLOSE position is strictly before the query
// opening position participate in the softmax denominator. Invalid/padded or
// future columns receive exactly zero gradient even if their input logits are
// finite. The reduction is a mean over all Q queries, including class-0 rows.
//
// Returns a scalar [1, 1] Tensor. When Q is zero, returns a detached zero.
Tensor LocalAtomRetrievalLoss(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

} // namespace GRIM::LocalAtomRetrieval
