//======================================================//
//  NumericAtomBackward.hpp
//  Dedicated recurrent NumericAtom autograd boundary.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../Forward/NumericAtomForward.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct NumberEncoderParameterTensors;

namespace autograd {

// Attaches the single reverse-time NumericAtom node to the detached scalar
// loss. The node borrows forward activations and registry-owned parameters;
// allocation, loss composition, scheduling, and ownership stay outside it.
void attachNumericAtomBackward(
    Tensor& numeric_atom_loss,
    Tensor& shared_hidden_state,
    NumberEncoderParameterTensors& parameters,
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM

#endif  // USE_CUDA
