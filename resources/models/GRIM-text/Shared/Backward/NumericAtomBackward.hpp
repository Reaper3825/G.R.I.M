//======================================================//
//  NumericAtomBackward.hpp
//  Dedicated NumericAtom autograd boundary.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../Forward/NumericAtomForward.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace autograd {

// Attaches the single NumericAtom backward node to an already-computed scalar
// loss. Forward computation, loss computation, graph scheduling, and storage
// ownership remain outside this boundary.
void attachNumericAtomBackward(
    Tensor& numeric_atom_loss,
    Tensor& shared_hidden_state,
    Tensor& digit_embedding,
    Tensor& pow10_embedding,
    Tensor& slot_embedding,
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM

#endif  // USE_CUDA
