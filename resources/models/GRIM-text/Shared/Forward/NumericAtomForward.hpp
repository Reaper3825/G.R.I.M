//======================================================//
//  NumericAtomForward.hpp
//  Typed auxiliary forward branch for numeric atoms.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct NumberEncoderParameterTensors;

namespace Forward {

// Dense, autograd-connected numeric-head predictions. Target selection and
// reduction remain loss-side responsibilities; BatchPayload row routing masks
// select the supervised rows from these tensors later.
struct NumericAtomForwardOutputs {
    int total_rows = 0;
    int digit_classes = 0;
    int pow10_buckets = 0;

    Tensor digit_logits;  // [payload.total_tokens, 10]
    Tensor pow10_logits;  // [payload.total_tokens, pow10_buckets]

    void clear() {
        digit_logits = Tensor();
        pow10_logits = Tensor();
        total_rows = 0;
        digit_classes = 0;
        pow10_buckets = 0;
    }

    bool populated() const {
        return digit_logits.data != nullptr && pow10_logits.data != nullptr;
    }
};

// Numeric auxiliary-head branch from the shared model hidden state.
//
// BatchPayload remains the semantic owner of numeric atom routing and targets;
// this operation must not accept exploded atom masks, positions, digit arrays,
// or pow10 arrays as independent arguments. The implementation is intentionally
// a tied output classifier over the NumberEncoder digit/pow10 embeddings.
NumericAtomForwardOutputs NumericAtomForward(
    const Tensor& shared_hidden_state,
    const NumberEncoderParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    cudaStream_t stream);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
