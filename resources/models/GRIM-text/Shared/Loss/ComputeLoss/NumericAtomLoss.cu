//======================================================//
//  NumericAtomLoss.cu
//  Loss boundary for numeric atom reconstruction.
//======================================================//

#include "NumericAtomLoss.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace GRIM {
namespace autograd {

namespace {

constexpr int kRowBlock = 256;

__global__ void kernelNumericAtomLoss(
    const float* __restrict__ digit_logits,
    const float* __restrict__ pow10_logits,
    const int* __restrict__ digit_targets,
    const int* __restrict__ pow10_targets,
    const uint8_t* __restrict__ digit_mask,
    float* __restrict__ loss_sum,
    int digit_slots,
    int digit_classes,
    int pow10_buckets,
    float scale) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= digit_slots || digit_mask[slot] == 0) return;

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

    float pow10_max = pow10_row[0];
    for (int cls = 1; cls < pow10_buckets; ++cls) {
        pow10_max = fmaxf(pow10_max, pow10_row[cls]);
    }
    float pow10_exp_sum = 0.0f;
    for (int cls = 0; cls < pow10_buckets; ++cls) {
        pow10_exp_sum += __expf(pow10_row[cls] - pow10_max);
    }

    const float digit_nll =
        digit_max + logf(fmaxf(digit_exp_sum, 1e-20f)) - digit_row[digit_target];
    const float pow10_nll =
        pow10_max + logf(fmaxf(pow10_exp_sum, 1e-20f)) - pow10_row[pow10_target];
    atomicAdd(loss_sum, (digit_nll + pow10_nll) * scale);
}

void requireLogitShape(
    const Tensor& tensor,
    int expected_rows,
    int expected_classes,
    const char* name) {
    tensor.require(name);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(name) + " must be 2D");
    }
    const auto shape = tensor.shape.as_2d();
    if (shape.rows != expected_rows || shape.cols != expected_classes) {
        throw std::runtime_error(
            std::string(name) + " shape=[" + std::to_string(shape.rows) + "," +
            std::to_string(shape.cols) + "] expected=[" +
            std::to_string(expected_rows) + "," +
            std::to_string(expected_classes) + "]");
    }
}

}  // namespace

Tensor NumericAtomLoss(
    const Forward::NumericAtomForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream) {
    if (!stream) {
        throw std::runtime_error("NumericAtomLoss: stream is NULL");
    }
    payload.validate("NumericAtomLoss");
    if (!forward_outputs.evaluated) {
        throw std::runtime_error("NumericAtomLoss: NumericAtomForward was not evaluated");
    }
    if (forward_outputs.numeric_atom_count == 0) {
        return Tensor();
    }
    if (!bindings.d_number_aux_target_digits ||
        !bindings.d_number_aux_target_pow10_index ||
        !bindings.d_number_aux_target_digit_mask) {
        throw std::runtime_error(
            "NumericAtomLoss: numeric target device bindings are NULL");
    }
    if (forward_outputs.digit_slots != payload.number_aux_target_digit_slots ||
        forward_outputs.digit_slots <= 0) {
        throw std::runtime_error(
            "NumericAtomLoss: forward digit-slot geometry disagrees with BatchPayload");
    }
    if (forward_outputs.digit_classes != 10 ||
        forward_outputs.pow10_buckets !=
            2 * payload.number_aux_target_max_abs_pow10 + 1) {
        throw std::runtime_error(
            "NumericAtomLoss: forward class geometry disagrees with BatchPayload");
    }
    if (static_cast<int>(forward_outputs.payload_atom_indices.size()) !=
        forward_outputs.numeric_atom_count) {
        throw std::runtime_error(
            "NumericAtomLoss: payload atom routing size disagrees with numeric_atom_count");
    }

    const int total_slot_rows =
        forward_outputs.numeric_atom_count * forward_outputs.digit_slots;
    if (forward_outputs.total_rows != total_slot_rows) {
        throw std::runtime_error(
            "NumericAtomLoss: flattened output row count is inconsistent");
    }
    requireLogitShape(
        forward_outputs.digit_logits,
        total_slot_rows,
        forward_outputs.digit_classes,
        "NumericAtomLoss.digit_logits");
    requireLogitShape(
        forward_outputs.pow10_logits,
        total_slot_rows,
        forward_outputs.pow10_buckets,
        "NumericAtomLoss.pow10_logits");

    int valid_slot_count = 0;
    const size_t payload_atom_count = payload.atom_positions.size();
    const size_t digit_slots = static_cast<size_t>(forward_outputs.digit_slots);
    for (int numeric_atom = 0;
         numeric_atom < forward_outputs.numeric_atom_count;
         ++numeric_atom) {
        const int payload_atom =
            forward_outputs.payload_atom_indices[static_cast<size_t>(numeric_atom)];
        if (payload_atom < 0 || static_cast<size_t>(payload_atom) >= payload_atom_count) {
            throw std::runtime_error(
                "NumericAtomLoss: payload atom index is out of range");
        }
        if (payload.number_aux_target_base[static_cast<size_t>(payload_atom)] != 10) {
            throw std::runtime_error(
                "NumericAtomLoss: only base-10 numeric targets are supported");
        }
        if (payload.number_aux_target_valid[static_cast<size_t>(payload_atom)] == 0) {
            continue;
        }
        const size_t target_offset =
            static_cast<size_t>(payload_atom) * digit_slots;
        for (int slot = 0; slot < forward_outputs.digit_slots; ++slot) {
            valid_slot_count +=
                payload.number_aux_target_digit_mask[
                    target_offset + static_cast<size_t>(slot)] != 0;
        }
    }

    if (valid_slot_count == 0) {
        return Tensor();
    }

    Tensor result = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.loss");
    const float scale = 1.0f / (2.0f * static_cast<float>(valid_slot_count));
    const int blocks =
        (forward_outputs.digit_slots + kRowBlock - 1) / kRowBlock;

    for (int numeric_atom = 0;
         numeric_atom < forward_outputs.numeric_atom_count;
         ++numeric_atom) {
        const int payload_atom =
            forward_outputs.payload_atom_indices[static_cast<size_t>(numeric_atom)];
        if (payload.number_aux_target_valid[static_cast<size_t>(payload_atom)] == 0) {
            continue;
        }

        const size_t logit_row_offset =
            static_cast<size_t>(numeric_atom) * digit_slots;
        const size_t target_offset =
            static_cast<size_t>(payload_atom) * digit_slots;
        kernelNumericAtomLoss<<<blocks, kRowBlock, 0, stream>>>(
            forward_outputs.digit_logits.data +
                logit_row_offset * forward_outputs.digit_classes,
            forward_outputs.pow10_logits.data +
                logit_row_offset * forward_outputs.pow10_buckets,
            bindings.d_number_aux_target_digits + target_offset,
            bindings.d_number_aux_target_pow10_index + target_offset,
            bindings.d_number_aux_target_digit_mask + target_offset,
            result.data,
            forward_outputs.digit_slots,
            forward_outputs.digit_classes,
            forward_outputs.pow10_buckets,
            scale);
    }

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("NumericAtomLoss: kernel launch failed: ") +
            cudaGetErrorString(launch_error));
    }
    return result;
}

}  // namespace autograd
}  // namespace GRIM
