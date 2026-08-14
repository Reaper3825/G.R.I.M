//======================================================//
//  NumericAtomForward.cu
//  Stub typed auxiliary forward branch for numeric atoms.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "NumericAtomForward.hpp"

namespace GRIM {
namespace Forward {

void NumericAtomForward(
    const Tensor& shared_hidden_state,
    const Batching::BatchPayload& payload,
    cudaStream_t stream) {
    // Future implementation boundary:
    //   shared hidden state
    //     -> payload-authored numeric AUX row routing
    //     -> mantissa digit / pow10 reconstruction
    //     -> normalized decoded-value comparison
    //
    // Keep this stub side-effect free so introducing the architectural branch
    // does not alter the current LM-forward result.
    (void)shared_hidden_state;
    (void)payload;
    (void)stream;
}

}  // namespace Forward
}  // namespace GRIM
