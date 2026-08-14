//======================================================//
//  NumericAtomForward.hpp
//  Stub typed auxiliary forward branch for numeric atoms.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace Forward {

// Numeric auxiliary-head branch from the shared model hidden state.
//
// BatchPayload remains the semantic owner of numeric atom routing and targets;
// this operation must not accept exploded atom masks, positions, digit arrays,
// or pow10 arrays as independent arguments. The implementation is intentionally
// a no-op scaffold until the numeric reconstruction head is authored.
void NumericAtomForward(
    const Tensor& shared_hidden_state,
    const Batching::BatchPayload& payload,
    cudaStream_t stream);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
