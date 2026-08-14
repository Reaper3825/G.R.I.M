#pragma once
//======================================================//
//  NumericAtomLoss.hpp
//  Loss boundary for numeric atom reconstruction.
//======================================================//

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../../Batching/BatchPayload.hpp"
#include "../../Forward/NumericAtomForward.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace autograd {

// Computes the scalar numeric reconstruction loss from the typed numeric-head
// outputs and BatchPayload-owned targets. This is currently a wiring stub; an
// empty Tensor means that no numeric term is ready for loss composition yet.
Tensor NumericAtomLoss(
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM

#endif  // USE_CUDA
