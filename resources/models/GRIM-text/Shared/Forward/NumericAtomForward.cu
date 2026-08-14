//======================================================//
//  NumericAtomForward.cu
//  Typed auxiliary forward branch for numeric atoms.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "NumericAtomForward.hpp"

#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Forward {

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelGatherAnchorAndAddSlot(
    const float* __restrict__ hidden,
    const float* __restrict__ slot_emb,
    const int* __restrict__ anchor_positions,
    float* __restrict__ output,
    int numeric_atom_count,
    int digit_slots,
    int d_model) {
    const size_t count = static_cast<size_t>(numeric_atom_count) * digit_slots * d_model;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const int feature = static_cast<int>(index % d_model);
    const int atom_slot = static_cast<int>(index / d_model);
    const int slot = atom_slot % digit_slots;
    const int atom = atom_slot / digit_slots;
    const int anchor = anchor_positions[atom];
    output[index] = hidden[static_cast<size_t>(anchor) * d_model + feature] +
                    slot_emb[static_cast<size_t>(slot) * d_model + feature];
}

Tensor gatherAnchorAndAddSlot(
    const Tensor& hidden,
    const Tensor& slot_emb,
    const std::vector<int>& anchor_positions,
    cudaStream_t stream) {
    const auto hidden_shape = hidden.shape.as_2d();
    const auto slot_shape = slot_emb.shape.as_2d();
    const int numeric_atom_count = static_cast<int>(anchor_positions.size());
    const int digit_slots = slot_shape.rows;
    const int d_model = hidden_shape.cols;
    const int output_rows = numeric_atom_count * digit_slots;

    Tensor output = Tensor::empty(
        TensorContract::TensorShape::make_BSM(output_rows, d_model),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.slot_states");

    int* device_anchor_positions = nullptr;
    const size_t anchor_bytes = anchor_positions.size() * sizeof(int);
    const cudaError_t alloc_error = cudaMalloc(
        reinterpret_cast<void**>(&device_anchor_positions), anchor_bytes);
    if (alloc_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("NumericAtomForward: anchor allocation failed: ") +
            cudaGetErrorString(alloc_error));
    }
    const cudaError_t copy_error = cudaMemcpyAsync(
        device_anchor_positions,
        anchor_positions.data(),
        anchor_bytes,
        cudaMemcpyHostToDevice,
        stream);
    if (copy_error != cudaSuccess) {
        cudaFree(device_anchor_positions);
        throw std::runtime_error(
            std::string("NumericAtomForward: anchor upload failed: ") +
            cudaGetErrorString(copy_error));
    }

    const size_t count = static_cast<size_t>(output_rows) * d_model;
    const int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
    kernelGatherAnchorAndAddSlot<<<blocks, kBlockSize, 0, stream>>>(
        hidden.data,
        slot_emb.data,
        device_anchor_positions,
        output.data,
        numeric_atom_count,
        digit_slots,
        d_model);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        cudaFree(device_anchor_positions);
        throw std::runtime_error(
            std::string("NumericAtomForward: anchor-slot kernel launch failed: ") +
            cudaGetErrorString(launch_error));
    }

    cudaFree(device_anchor_positions);
    return output;
}

}  // namespace

NumericAtomForwardOutputs NumericAtomForward(
    const Tensor& shared_hidden_state,
    const NumberEncoderParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    cudaStream_t stream) {
    if (!stream) {
        throw std::runtime_error("NumericAtomForward: stream is NULL");
    }
    shared_hidden_state.require("NumericAtomForward.shared_hidden_state");
    parameters.digit_emb.require("NumericAtomForward.parameters.digit_emb");
    parameters.pow10_emb.require("NumericAtomForward.parameters.pow10_emb");
    parameters.numeric_atom_slot_emb.require(
        "NumericAtomForward.parameters.numeric_atom_slot_emb");
    if (!shared_hidden_state.shape.is_2d_layout() ||
        !parameters.digit_emb.shape.is_2d_layout() ||
        !parameters.pow10_emb.shape.is_2d_layout() ||
        !parameters.numeric_atom_slot_emb.shape.is_2d_layout()) {
        throw std::runtime_error(
            "NumericAtomForward: hidden state and numeric embeddings must be 2D");
    }

    const auto hidden_shape = shared_hidden_state.shape.as_2d();
    const auto digit_shape = parameters.digit_emb.shape.as_2d();
    const auto pow10_shape = parameters.pow10_emb.shape.as_2d();
    const auto slot_shape = parameters.numeric_atom_slot_emb.shape.as_2d();
    if (hidden_shape.rows != payload.total_tokens) {
        throw std::runtime_error(
            "NumericAtomForward: shared hidden rows=" +
            std::to_string(hidden_shape.rows) + " != payload.total_tokens=" +
            std::to_string(payload.total_tokens));
    }
    if (digit_shape.rows != 10) {
        throw std::runtime_error(
            "NumericAtomForward: digit_emb rows=" +
            std::to_string(digit_shape.rows) + " != 10 digit classes");
    }
    if (digit_shape.cols != hidden_shape.cols ||
        pow10_shape.cols != hidden_shape.cols ||
        slot_shape.cols != hidden_shape.cols ||
        pow10_shape.rows <= 0 || slot_shape.rows <= 0) {
        throw std::runtime_error(
            "NumericAtomForward: numeric embedding geometry does not match shared hidden width");
    }
    if (payload.number_aux_target_digit_slots > 0) {
        if (slot_shape.rows != payload.number_aux_target_digit_slots) {
            throw std::runtime_error(
                "NumericAtomForward: numeric_atom_slot_emb rows=" +
                std::to_string(slot_shape.rows) +
                " != payload-authored digit slots=" +
                std::to_string(payload.number_aux_target_digit_slots));
        }
        const int expected_pow10_buckets =
            2 * payload.number_aux_target_max_abs_pow10 + 1;
        if (pow10_shape.rows != expected_pow10_buckets) {
            throw std::runtime_error(
                "NumericAtomForward: pow10_emb rows=" +
                std::to_string(pow10_shape.rows) +
                " != payload-authored pow10 buckets=" +
                std::to_string(expected_pow10_buckets));
        }
    }

    NumericAtomForwardOutputs outputs;
    outputs.evaluated = true;
    outputs.digit_slots = slot_shape.rows;
    outputs.digit_classes = digit_shape.rows;
    outputs.pow10_buckets = pow10_shape.rows;

    std::vector<int> anchor_positions;
    anchor_positions.reserve(payload.atom_positions.size());
    outputs.payload_atom_indices.reserve(payload.atom_positions.size());
    for (std::size_t atom = 0; atom < payload.atom_positions.size(); ++atom) {
        const auto atom_type = static_cast<Tokenizer::AtomType>(payload.atom_types[atom]);
        if (!Tokenizer::isNumericAtom(atom_type)) continue;

        const int anchor = payload.atom_positions[atom];
        if (anchor < 0 || anchor >= payload.total_tokens) {
            throw std::runtime_error(
                "NumericAtomForward: numeric atom opening anchor is outside payload geometry");
        }
        anchor_positions.push_back(anchor);
        outputs.payload_atom_indices.push_back(static_cast<int>(atom));
    }

    outputs.numeric_atom_count = static_cast<int>(anchor_positions.size());
    outputs.total_rows = outputs.numeric_atom_count * outputs.digit_slots;
    if (outputs.numeric_atom_count == 0) {
        return outputs;
    }

    Tensor slot_states = gatherAnchorAndAddSlot(
        shared_hidden_state,
        parameters.numeric_atom_slot_emb,
        anchor_positions,
        stream);

    // The dedicated NumericAtom backward boundary owns this branch's single
    // gradient operation. Keep these classifier matmuls forward-only so they
    // do not attach independent MatMulGradFn nodes.
    Tensor digit_classifier = parameters.digit_emb.detach(stream);
    Tensor pow10_classifier = parameters.pow10_emb.detach(stream);
    outputs.digit_logits = autograd::matmul(
        slot_states,
        digit_classifier,
        stream,
        /*transpose_b=*/true);
    outputs.pow10_logits = autograd::matmul(
        slot_states,
        pow10_classifier,
        stream,
        /*transpose_b=*/true);

    return outputs;
}

}  // namespace Forward
}  // namespace GRIM
