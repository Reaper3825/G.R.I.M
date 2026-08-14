//======================================================//
//  NumericAtomBackward.cu
//  Dedicated NumericAtom autograd boundary.
//======================================================//

#include "NumericAtomBackward.hpp"

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace GRIM {
namespace autograd {

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelNumericAtomBackward(
    const float* __restrict__ upstream_gradient,
    const float* __restrict__ hidden,
    const float* __restrict__ digit_embedding,
    const float* __restrict__ pow10_embedding,
    const float* __restrict__ slot_embedding,
    const float* __restrict__ digit_logits,
    const float* __restrict__ pow10_logits,
    const int* __restrict__ digit_targets,
    const int* __restrict__ pow10_targets,
    const uint8_t* __restrict__ digit_mask,
    float* hidden_gradient,
    float* digit_embedding_gradient,
    float* pow10_embedding_gradient,
    float* slot_embedding_gradient,
    int anchor_position,
    int digit_slots,
    int digit_classes,
    int pow10_buckets,
    int d_model,
    float normalization) {
    const size_t count = static_cast<size_t>(digit_slots) * d_model;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const int slot = static_cast<int>(index / d_model);
    const int feature = static_cast<int>(index % d_model);
    if (digit_mask[slot] == 0) return;

    const int digit_target = digit_targets[slot];
    const int pow10_target = pow10_targets[slot];
    const float* digit_row =
        digit_logits + static_cast<size_t>(slot) * digit_classes;
    const float* pow10_row =
        pow10_logits + static_cast<size_t>(slot) * pow10_buckets;

    float digit_max = digit_row[0];
    for (int cls = 1; cls < digit_classes; ++cls) {
        digit_max = fmaxf(digit_max, digit_row[cls]);
    }
    float digit_exp_sum = 0.0f;
    for (int cls = 0; cls < digit_classes; ++cls) {
        digit_exp_sum += __expf(digit_row[cls] - digit_max);
    }
    const float digit_inv_sum = 1.0f / fmaxf(digit_exp_sum, 1e-20f);

    float pow10_max = pow10_row[0];
    for (int cls = 1; cls < pow10_buckets; ++cls) {
        pow10_max = fmaxf(pow10_max, pow10_row[cls]);
    }
    float pow10_exp_sum = 0.0f;
    for (int cls = 0; cls < pow10_buckets; ++cls) {
        pow10_exp_sum += __expf(pow10_row[cls] - pow10_max);
    }
    const float pow10_inv_sum = 1.0f / fmaxf(pow10_exp_sum, 1e-20f);

    const float scale = upstream_gradient[0] * normalization;
    const float slot_state =
        hidden[static_cast<size_t>(anchor_position) * d_model + feature] +
        slot_embedding[static_cast<size_t>(slot) * d_model + feature];
    float slot_state_gradient = 0.0f;

    for (int cls = 0; cls < digit_classes; ++cls) {
        const float probability =
            __expf(digit_row[cls] - digit_max) * digit_inv_sum;
        const float logit_gradient =
            (probability - (cls == digit_target ? 1.0f : 0.0f)) * scale;
        slot_state_gradient +=
            logit_gradient *
            digit_embedding[static_cast<size_t>(cls) * d_model + feature];
        if (digit_embedding_gradient) {
            atomicAdd(
                &digit_embedding_gradient[
                    static_cast<size_t>(cls) * d_model + feature],
                logit_gradient * slot_state);
        }
    }

    for (int cls = 0; cls < pow10_buckets; ++cls) {
        const float probability =
            __expf(pow10_row[cls] - pow10_max) * pow10_inv_sum;
        const float logit_gradient =
            (probability - (cls == pow10_target ? 1.0f : 0.0f)) * scale;
        slot_state_gradient +=
            logit_gradient *
            pow10_embedding[static_cast<size_t>(cls) * d_model + feature];
        if (pow10_embedding_gradient) {
            atomicAdd(
                &pow10_embedding_gradient[
                    static_cast<size_t>(cls) * d_model + feature],
                logit_gradient * slot_state);
        }
    }

    if (hidden_gradient) {
        atomicAdd(
            &hidden_gradient[
                static_cast<size_t>(anchor_position) * d_model + feature],
            slot_state_gradient);
    }
    if (slot_embedding_gradient) {
        atomicAdd(
            &slot_embedding_gradient[
                static_cast<size_t>(slot) * d_model + feature],
            slot_state_gradient);
    }
}

struct NumericAtomBackwardFn final : public GradFn {
    std::shared_ptr<Tensor> hidden_gradient;
    std::shared_ptr<Tensor> digit_embedding_gradient;
    std::shared_ptr<Tensor> pow10_embedding_gradient;
    std::shared_ptr<Tensor> slot_embedding_gradient;

    const float* hidden = nullptr;
    const float* digit_embedding = nullptr;
    const float* pow10_embedding = nullptr;
    const float* slot_embedding = nullptr;
    const float* digit_logits = nullptr;
    const float* pow10_logits = nullptr;

    int numeric_atom_count = 0;
    int digit_slots = 0;
    int digit_classes = 0;
    int pow10_buckets = 0;
    int hidden_rows = 0;
    int d_model = 0;

    NumericAtomBackwardFn() { op_name = "numeric_atom"; }
    ~NumericAtomBackwardFn() override { release_saved(); }

    void capture_inputs(
        Tensor& shared_hidden_state,
        Tensor& digit_embedding_tensor,
        Tensor& pow10_embedding_tensor,
        Tensor& slot_embedding_tensor,
        const Forward::NumericAtomForwardOutputs& forward_outputs,
        cudaStream_t stream) {
        hidden = shared_hidden_state.data;
        digit_embedding = digit_embedding_tensor.data;
        pow10_embedding = pow10_embedding_tensor.data;
        slot_embedding = slot_embedding_tensor.data;
        digit_logits = forward_outputs.digit_logits.data;
        pow10_logits = forward_outputs.pow10_logits.data;

        if (shared_hidden_state.requires_grad) {
            hidden_gradient = capture_input_gradient(
                shared_hidden_state,
                stream,
                "NumericAtomBackward.hidden");
        }
        if (digit_embedding_tensor.requires_grad) {
            digit_embedding_gradient = capture_input_gradient(
                digit_embedding_tensor,
                stream,
                "NumericAtomBackward.digit_embedding");
        }
        if (pow10_embedding_tensor.requires_grad) {
            pow10_embedding_gradient = capture_input_gradient(
                pow10_embedding_tensor,
                stream,
                "NumericAtomBackward.pow10_embedding");
        }
        if (slot_embedding_tensor.requires_grad) {
            slot_embedding_gradient = capture_input_gradient(
                slot_embedding_tensor,
                stream,
                "NumericAtomBackward.slot_embedding");
        }
    }

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override {
        if (applied) return;
        applied = true;
        if (!backward_payload || !backward_bindings) {
            throw std::runtime_error(
                "NumericAtomBackward: payload and device bindings are required");
        }
        if (!grad_output.data || grad_output.numel() != 1) {
            throw std::runtime_error(
                "NumericAtomBackward: upstream gradient must be scalar");
        }
        if (!hidden || !digit_embedding || !pow10_embedding || !slot_embedding ||
            !digit_logits || !pow10_logits) {
            throw std::runtime_error(
                "NumericAtomBackward: borrowed forward data is incomplete");
        }
        if (!backward_bindings->d_number_aux_target_digits ||
            !backward_bindings->d_number_aux_target_pow10_index ||
            !backward_bindings->d_number_aux_target_digit_mask) {
            throw std::runtime_error(
                "NumericAtomBackward: numeric target device bindings are NULL");
        }

        backward_payload->validate("NumericAtomBackward");
        if (backward_payload->total_tokens != hidden_rows ||
            backward_payload->number_aux_target_digit_slots != digit_slots ||
            2 * backward_payload->number_aux_target_max_abs_pow10 + 1 !=
                pow10_buckets) {
            throw std::runtime_error(
                "NumericAtomBackward: payload geometry disagrees with the captured graph");
        }
        int valid_slot_count = 0;
        int routed_numeric_atoms = 0;
        const size_t payload_digit_slots = static_cast<size_t>(digit_slots);
        for (size_t atom = 0; atom < backward_payload->atom_types.size(); ++atom) {
            const auto atom_type = static_cast<Tokenizer::AtomType>(
                backward_payload->atom_types[atom]);
            if (!Tokenizer::isNumericAtom(atom_type)) continue;
            ++routed_numeric_atoms;
            if (backward_payload->number_aux_target_base[atom] != 10) {
                throw std::runtime_error(
                    "NumericAtomBackward: only base-10 numeric targets are supported");
            }
            if (backward_payload->number_aux_target_valid[atom] == 0) continue;
            const size_t target_offset = atom * payload_digit_slots;
            for (int slot = 0; slot < digit_slots; ++slot) {
                valid_slot_count +=
                    backward_payload->number_aux_target_digit_mask[
                        target_offset + static_cast<size_t>(slot)] != 0;
            }
        }
        if (routed_numeric_atoms != numeric_atom_count || valid_slot_count <= 0) {
            throw std::runtime_error(
                "NumericAtomBackward: payload routing disagrees with captured forward geometry");
        }

        const float normalization =
            1.0f / (2.0f * static_cast<float>(valid_slot_count));
        const size_t per_atom_count =
            static_cast<size_t>(digit_slots) * d_model;
        const int blocks = static_cast<int>(
            (per_atom_count + kBlockSize - 1) / kBlockSize);

        int numeric_atom = 0;
        for (size_t atom = 0; atom < backward_payload->atom_types.size(); ++atom) {
            const auto atom_type = static_cast<Tokenizer::AtomType>(
                backward_payload->atom_types[atom]);
            if (!Tokenizer::isNumericAtom(atom_type)) continue;

            const int output_atom = numeric_atom++;
            if (backward_payload->number_aux_target_valid[atom] == 0) continue;
            const int anchor_position = backward_payload->atom_positions[atom];
            if (anchor_position < 0 ||
                anchor_position >= backward_payload->total_tokens) {
                throw std::runtime_error(
                    "NumericAtomBackward: numeric anchor is outside payload geometry");
            }

            const size_t logit_row_offset =
                static_cast<size_t>(output_atom) * digit_slots;
            const size_t target_offset = atom * payload_digit_slots;
            kernelNumericAtomBackward<<<blocks, kBlockSize, 0, stream>>>(
                grad_output.data,
                hidden,
                digit_embedding,
                pow10_embedding,
                slot_embedding,
                digit_logits + logit_row_offset * digit_classes,
                pow10_logits + logit_row_offset * pow10_buckets,
                backward_bindings->d_number_aux_target_digits + target_offset,
                backward_bindings->d_number_aux_target_pow10_index + target_offset,
                backward_bindings->d_number_aux_target_digit_mask + target_offset,
                hidden_gradient ? hidden_gradient->data : nullptr,
                digit_embedding_gradient ? digit_embedding_gradient->data : nullptr,
                pow10_embedding_gradient ? pow10_embedding_gradient->data : nullptr,
                slot_embedding_gradient ? slot_embedding_gradient->data : nullptr,
                anchor_position,
                digit_slots,
                digit_classes,
                pow10_buckets,
                d_model,
                normalization);
            trackKernelLaunch("kernelNumericAtomBackward", stream);
        }

        if (hidden_gradient) {
            propagate_input_gradient(
                hidden_gradient,
                stream,
                backward_payload,
                backward_bindings,
                "NumericAtomBackward.hidden");
        }
        if (digit_embedding_gradient) {
            propagate_input_gradient(
                digit_embedding_gradient,
                stream,
                backward_payload,
                backward_bindings,
                "NumericAtomBackward.digit_embedding");
        }
        if (pow10_embedding_gradient) {
            propagate_input_gradient(
                pow10_embedding_gradient,
                stream,
                backward_payload,
                backward_bindings,
                "NumericAtomBackward.pow10_embedding");
        }
        if (slot_embedding_gradient) {
            propagate_input_gradient(
                slot_embedding_gradient,
                stream,
                backward_payload,
                backward_bindings,
                "NumericAtomBackward.slot_embedding");
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        hidden_gradient.reset();
        digit_embedding_gradient.reset();
        pow10_embedding_gradient.reset();
        slot_embedding_gradient.reset();
        hidden = nullptr;
        digit_embedding = nullptr;
        pow10_embedding = nullptr;
        slot_embedding = nullptr;
        digit_logits = nullptr;
        pow10_logits = nullptr;
    }
};

}  // namespace

void attachNumericAtomBackward(
    Tensor& numeric_atom_loss,
    Tensor& shared_hidden_state,
    Tensor& digit_embedding,
    Tensor& pow10_embedding,
    Tensor& slot_embedding,
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    cudaStream_t stream) {
    numeric_atom_loss.require("attachNumericAtomBackward.loss");
    shared_hidden_state.require("attachNumericAtomBackward.hidden");
    digit_embedding.require("attachNumericAtomBackward.digit_embedding");
    pow10_embedding.require("attachNumericAtomBackward.pow10_embedding");
    slot_embedding.require("attachNumericAtomBackward.slot_embedding");
    if (!stream) {
        throw std::runtime_error("attachNumericAtomBackward: stream is NULL");
    }
    if (numeric_atom_loss.numel() != 1 || numeric_atom_loss.grad_fn ||
        numeric_atom_loss.requires_grad) {
        throw std::runtime_error(
            "attachNumericAtomBackward: loss must be a detached scalar");
    }
    if (!forward_outputs.populated() ||
        forward_outputs.numeric_atom_count <= 0 ||
        forward_outputs.digit_slots <= 0 ||
        forward_outputs.digit_classes <= 0 ||
        forward_outputs.pow10_buckets <= 0) {
        throw std::runtime_error(
            "attachNumericAtomBackward: forward outputs are incomplete");
    }
    if (!shared_hidden_state.shape.is_2d_layout() ||
        !digit_embedding.shape.is_2d_layout() ||
        !pow10_embedding.shape.is_2d_layout() ||
        !slot_embedding.shape.is_2d_layout() ||
        !forward_outputs.digit_logits.shape.is_2d_layout() ||
        !forward_outputs.pow10_logits.shape.is_2d_layout()) {
        throw std::runtime_error(
            "attachNumericAtomBackward: hidden, embeddings, and logits must be 2D");
    }

    const auto hidden_shape = shared_hidden_state.shape.as_2d();
    const auto digit_embedding_shape = digit_embedding.shape.as_2d();
    const auto pow10_embedding_shape = pow10_embedding.shape.as_2d();
    const auto slot_embedding_shape = slot_embedding.shape.as_2d();
    const auto digit_logit_shape = forward_outputs.digit_logits.shape.as_2d();
    const auto pow10_logit_shape = forward_outputs.pow10_logits.shape.as_2d();
    const int expected_logit_rows =
        forward_outputs.numeric_atom_count * forward_outputs.digit_slots;
    if (forward_outputs.total_rows != expected_logit_rows ||
        digit_logit_shape.rows != expected_logit_rows ||
        digit_logit_shape.cols != forward_outputs.digit_classes ||
        pow10_logit_shape.rows != expected_logit_rows ||
        pow10_logit_shape.cols != forward_outputs.pow10_buckets) {
        throw std::runtime_error(
            "attachNumericAtomBackward: logit geometry disagrees with forward metadata");
    }
    if (hidden_shape.cols <= 0 ||
        digit_embedding_shape.rows != forward_outputs.digit_classes ||
        pow10_embedding_shape.rows != forward_outputs.pow10_buckets ||
        slot_embedding_shape.rows != forward_outputs.digit_slots ||
        digit_embedding_shape.cols != hidden_shape.cols ||
        pow10_embedding_shape.cols != hidden_shape.cols ||
        slot_embedding_shape.cols != hidden_shape.cols) {
        throw std::runtime_error(
            "attachNumericAtomBackward: embedding geometry disagrees with the shared hidden state");
    }
    if (forward_outputs.digit_logits.requires_grad ||
        forward_outputs.digit_logits.grad_fn ||
        forward_outputs.pow10_logits.requires_grad ||
        forward_outputs.pow10_logits.grad_fn) {
        throw std::runtime_error(
            "attachNumericAtomBackward: logits must remain detached from independent backward nodes");
    }
    if (!shared_hidden_state.requires_grad ||
        !digit_embedding.requires_grad ||
        !pow10_embedding.requires_grad ||
        !slot_embedding.requires_grad) {
        throw std::runtime_error(
            "attachNumericAtomBackward: all training inputs must require gradients");
    }

    auto grad_fn = std::make_shared<NumericAtomBackwardFn>();
    grad_fn->numeric_atom_count = forward_outputs.numeric_atom_count;
    grad_fn->digit_slots = forward_outputs.digit_slots;
    grad_fn->digit_classes = forward_outputs.digit_classes;
    grad_fn->pow10_buckets = forward_outputs.pow10_buckets;
    grad_fn->hidden_rows = hidden_shape.rows;
    grad_fn->d_model = hidden_shape.cols;
    grad_fn->capture_inputs(
        shared_hidden_state,
        digit_embedding,
        pow10_embedding,
        slot_embedding,
        forward_outputs,
        stream);

    numeric_atom_loss.requires_grad = true;
    numeric_atom_loss.is_leaf = false;
    numeric_atom_loss.grad_fn = std::move(grad_fn);
}

}  // namespace autograd
}  // namespace GRIM
