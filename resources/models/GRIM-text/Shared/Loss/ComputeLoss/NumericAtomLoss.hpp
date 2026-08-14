#pragma once
//======================================================//
//  NumericAtomLoss.hpp
//  Loss boundary for numeric atom reconstruction.
//======================================================//

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../../Batching/BatchDeviceBindings.hpp"
#include "../../Batching/BatchPayload.hpp"
#include "../../Forward/NumericAtomForward.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace autograd {

// Computes the detached scalar NumericAtom reconstruction loss. Numeric target
// device memory is owned by BatchDeviceStorage and borrowed through bindings;
// this operation allocates only its one-element result Tensor.
Tensor NumericAtomLoss(
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM

#endif  // USE_CUDA
