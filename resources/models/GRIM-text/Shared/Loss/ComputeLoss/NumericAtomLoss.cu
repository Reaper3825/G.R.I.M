//======================================================//
//  NumericAtomLoss.cu
//  Row-routed recurrent NumericAtom loss.
//======================================================//

#include "NumericAtomLoss.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace autograd {

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelNumericAtomLoss(
    const float* __restrict__ digit_logits,
    const float* __restrict__ pow10_logits,
    const int* __restrict__ row_atom_index,
    const uint8_t* __restrict__ row_mask,
    const int* __restrict__ row_step_index,
    const int* __restrict__ digit_targets,
    const int* __restrict__ pow10_targets,
    const uint8_t* __restrict__ digit_mask,
    float* __restrict__ loss_sum,
    int total_rows,
    int atom_count,
    int digit_slots,
    int digit_classes,
    int pow10_buckets,
    float scale) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows || row_mask[row] == 0) return;

    const int atom = row_atom_index[row];
    const int step = row_step_index[row];
    if (atom < 0 || atom >= atom_count || step < 0 || step >= digit_slots) return;

    const size_t target_index = static_cast<size_t>(atom) * digit_slots + step;
    if (digit_mask[target_index] == 0) return;

    const int digit_target = digit_targets[target_index];
    const int pow10_target = pow10_targets[target_index];
    const float* digit_row =
        digit_logits + static_cast<size_t>(row) * digit_classes;
    const float* pow10_row =
        pow10_logits + static_cast<size_t>(row) * pow10_buckets;

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
    int rows,
    int classes,
    const char* name) {
    tensor.require(name);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(name) + " must be 2D");
    }
    const auto shape = tensor.shape.as_2d();
    if (shape.rows != rows || shape.cols != classes) {
        throw std::runtime_error(
            std::string(name) + " shape=[" + std::to_string(shape.rows) + "," +
            std::to_string(shape.cols) + "] expected=[" + std::to_string(rows) +
            "," + std::to_string(classes) + "]");
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
    if (!forward_outputs.populated()) {
        throw std::runtime_error("NumericAtomLoss: recurrent forward outputs are incomplete");
    }
    if (forward_outputs.row_count != payload.total_tokens ||
        forward_outputs.atom_count != static_cast<int>(payload.atom_positions.size()) ||
        forward_outputs.decoder_step_capacity !=
            payload.number_aux_target_digit_slots ||
        forward_outputs.digit_classes != 10 ||
        forward_outputs.pow10_buckets !=
            2 * payload.number_aux_target_max_abs_pow10 + 1) {
        throw std::runtime_error(
            "NumericAtomLoss: forward geometry disagrees with BatchPayload");
    }
    requireLogitShape(
        forward_outputs.digit_logits,
        forward_outputs.row_count,
        forward_outputs.digit_classes,
        "NumericAtomLoss.digit_logits");
    requireLogitShape(
        forward_outputs.pow10_logits,
        forward_outputs.row_count,
        forward_outputs.pow10_buckets,
        "NumericAtomLoss.pow10_logits");

    if (!bindings.d_number_aux_target_atom_index ||
        !bindings.d_number_aux_target_row_mask ||
        !bindings.d_number_aux_target_step_index ||
        !bindings.d_number_aux_target_digits ||
        !bindings.d_number_aux_target_pow10_index ||
        !bindings.d_number_aux_target_digit_mask) {
        throw std::runtime_error(
            "NumericAtomLoss: numeric target device bindings are incomplete");
    }

    const int digit_slots = payload.number_aux_target_digit_slots;
    int valid_steps = 0;
    for (int row = 0; row < payload.total_tokens; ++row) {
        if (payload.number_aux_target_row_mask[static_cast<size_t>(row)] == 0) continue;
        const int atom =
            payload.number_aux_target_atom_index[static_cast<size_t>(row)];
        const int step =
            payload.number_aux_target_step_index[static_cast<size_t>(row)];
        if (atom < 0 || atom >= forward_outputs.atom_count ||
            step < 0 || step >= digit_slots) {
            continue;
        }
        const size_t target_index = static_cast<size_t>(atom) * digit_slots + step;
        valid_steps += payload.number_aux_target_digit_mask[target_index] != 0;
    }
    if (valid_steps == 0) {
        return Tensor();
    }

    Tensor result = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.loss");
    const float scale = 1.0f / (2.0f * static_cast<float>(valid_steps));
    const int blocks =
        (forward_outputs.row_count + kBlockSize - 1) / kBlockSize;
    kernelNumericAtomLoss<<<blocks, kBlockSize, 0, stream>>>(
        forward_outputs.digit_logits.data,
        forward_outputs.pow10_logits.data,
        bindings.d_number_aux_target_atom_index,
        bindings.d_number_aux_target_row_mask,
        bindings.d_number_aux_target_step_index,
        bindings.d_number_aux_target_digits,
        bindings.d_number_aux_target_pow10_index,
        bindings.d_number_aux_target_digit_mask,
        result.data,
        forward_outputs.row_count,
        forward_outputs.atom_count,
        digit_slots,
        forward_outputs.digit_classes,
        forward_outputs.pow10_buckets,
        scale);
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
