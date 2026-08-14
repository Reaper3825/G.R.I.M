//======================================================//
//  NumericAtomForward.hpp
//  Typed auxiliary forward branch for numeric atoms.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <vector>

#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct NumberEncoderParameterTensors;

namespace Forward {

// Atom-aligned numeric-head predictions. The physical Tensor layout remains
// 2D; each row is one (numeric atom, digit slot) pair. This boundary remains
// forward-only; the dedicated backward node is attached during loss assembly.
struct NumericAtomForwardOutputs {
    bool evaluated = false;
    int numeric_atom_count = 0;
    int digit_slots = 0;
    int total_rows = 0;
    int digit_classes = 0;
    int pow10_buckets = 0;

    // Maps numeric output atom index back to the compact payload atom index.
    std::vector<int> payload_atom_indices;

    Tensor digit_logits;  // [numeric_atom_count * digit_slots, 10]
    Tensor pow10_logits;  // [numeric_atom_count * digit_slots, pow10_buckets]

    void clear() {
        digit_logits = Tensor();
        pow10_logits = Tensor();
        payload_atom_indices.clear();
        evaluated = false;
        numeric_atom_count = 0;
        digit_slots = 0;
        total_rows = 0;
        digit_classes = 0;
        pow10_buckets = 0;
    }

    bool populated() const {
        if (!evaluated) return false;
        if (numeric_atom_count == 0) {
            return total_rows == 0 && payload_atom_indices.empty();
        }
        return digit_logits.data != nullptr && pow10_logits.data != nullptr;
    }
};

// Numeric auxiliary-head branch from the shared model hidden state.
//
// BatchPayload remains the semantic owner of numeric atom routing and targets;
// this operation must not accept exploded atom masks, positions, digit arrays,
// or pow10 arrays as independent arguments. Each numeric opening-anchor hidden
// state is expanded across numeric_atom_slot_emb before the tied digit/pow10
// classifiers are applied.
NumericAtomForwardOutputs NumericAtomForward(
    const Tensor& shared_hidden_state,
    const NumberEncoderParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    cudaStream_t stream);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
