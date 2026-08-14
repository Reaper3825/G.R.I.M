//======================================================//
//  NumericAtomLoss.cu
//  Loss boundary for numeric atom reconstruction.
//======================================================//

#include "NumericAtomLoss.hpp"

namespace GRIM {
namespace autograd {

Tensor NumericAtomLoss(
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    cudaStream_t stream) {
    // Wiring stub only. Target selection, reconstruction terms, reduction, and
    // autograd composition will be implemented here.
    (void)forward_outputs;
    (void)payload;
    (void)stream;
    return Tensor();
}

}  // namespace autograd
}  // namespace GRIM
