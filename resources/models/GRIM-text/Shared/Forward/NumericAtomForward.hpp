//======================================================//
//  NumericAtomForward.hpp
//  Row-aligned numeric decoder heads.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct NumberEncoderParameterTensors;

namespace Forward {

// Numeric decoder predictions use compact atom-step rows independent of token
// geometry. The LM head owns the typed OPEN delimiter; NumericAtom owns sign,
// digit, place, and STOP decisions after that opening state.
struct NumericAtomForwardOutputs {
    bool evaluated = false;
    int decoder_row_count = 0;
    int atom_count = 0;
    int decoder_step_capacity = 0;
    int digit_classes = 0;
    int pow10_buckets = 0;

    Tensor digit_logits;  // [atom_count * (decoder_step_capacity + 1), 10]
    Tensor pow10_logits;  // [atom_count * (decoder_step_capacity + 1), pow10_buckets]
    Tensor stop_logits;   // [atom_count * (decoder_step_capacity + 1), 1]
    Tensor sign_logits;   // [atom_count, 1], projected from opening state_0
    Tensor final_states;  // [atom_count, d_model], numeric entries populated
    Tensor step_states;   // [atom_count * decoder_step_capacity, d_model], state_t
    Tensor update_gates;  // [atom_count * decoder_step_capacity, d_model]
    Tensor reset_gates;   // [atom_count * decoder_step_capacity, d_model]
    Tensor candidates;    // [atom_count * decoder_step_capacity, d_model]

    void clear() {
        digit_logits = Tensor();
        pow10_logits = Tensor();
        stop_logits = Tensor();
        sign_logits = Tensor();
        final_states = Tensor();
        step_states = Tensor();
        update_gates = Tensor();
        reset_gates = Tensor();
        candidates = Tensor();
        evaluated = false;
        decoder_row_count = 0;
        atom_count = 0;
        decoder_step_capacity = 0;
        digit_classes = 0;
        pow10_buckets = 0;
    }

    bool populated() const {
        return evaluated && decoder_row_count > 0 &&
               digit_logits.data != nullptr && pow10_logits.data != nullptr &&
               stop_logits.data != nullptr && sign_logits.data != nullptr;
    }
};

// One inference-time projection from a persistent NumericAtom recurrent state.
// Selection policy stays with generation orchestration; this primitive only
// exposes the learned decisions for the current state.
struct NumericAtomInferenceLogits {
    Tensor digit_logits;  // [1, 10]
    Tensor pow10_logits;  // [1, pow10_buckets]
    Tensor stop_logits;   // [1, 1]
    Tensor sign_logits;   // [1, 1], consumed once from opening state_0

    bool populated() const {
        return digit_logits.data != nullptr && pow10_logits.data != nullptr &&
             stop_logits.data != nullptr && sign_logits.data != nullptr;
    }
};

// Initializes state_0 from each numeric OPEN row, emits digit/pow10 logits from
// state_t, then applies the teacher-forced GRU transition to produce
// state_t+1. Payload and bindings provide semantic row routing and borrowed
// target addresses; this operation owns neither storage nor orchestration.
NumericAtomForwardOutputs NumericAtomForward(
    const Tensor& shared_hidden_state,
    const NumberEncoderParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

NumericAtomInferenceLogits NumericAtomInferenceProject(
    const Tensor& recurrent_state,
    const NumberEncoderParameterTensors& parameters,
    cudaStream_t stream);

// Advances the persistent state after generation selects one digit/place pair.
// STOP has no transition; callers terminate without invoking this primitive.
Tensor NumericAtomInferenceTransition(
    const Tensor& recurrent_state,
    const NumberEncoderParameterTensors& parameters,
    int selected_digit,
    int selected_pow10_index,
    cudaStream_t stream);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
