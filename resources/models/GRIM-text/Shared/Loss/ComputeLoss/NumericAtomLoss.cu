//======================================================//
//  NumericAtomLoss.cu
//  Compact atom-step recurrent NumericAtom loss.
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
    const float* __restrict__ stop_logits,
    const uint8_t* __restrict__ atom_valid,
    const int* __restrict__ digit_targets,
    const int* __restrict__ pow10_targets,
    const uint8_t* __restrict__ digit_mask,
    float* __restrict__ loss_sum,
    int decoder_rows,
    int atom_count,
    int digit_slots,
    int digit_classes,
    int pow10_buckets,
    float scale) {
    const int decoder_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (decoder_row >= decoder_rows) return;
    const int decoder_stride = digit_slots + 1;
    const int atom = decoder_row / decoder_stride;
    const int step = decoder_row % decoder_stride;
    if (atom >= atom_count || atom_valid[atom] == 0) return;

    const size_t target_base = static_cast<size_t>(atom) * digit_slots;
    int digit_count = 0;
    while (digit_count < digit_slots && digit_mask[target_base + digit_count] != 0) {
        ++digit_count;
    }
    if (step > digit_count) return;

    const float stop_target = step == digit_count ? 1.0f : 0.0f;
    const float stop_logit = stop_logits[decoder_row];
    const float stop_bce = fmaxf(stop_logit, 0.0f) - stop_logit * stop_target +
        log1pf(__expf(-fabsf(stop_logit)));
    if (step == digit_count) {
        atomicAdd(loss_sum, stop_bce * scale);
        return;
    }

    const size_t target_index = target_base + step;
    const int digit_target = digit_targets[target_index];
    const int pow10_target = pow10_targets[target_index];
    const float* digit_row =
        digit_logits + static_cast<size_t>(decoder_row) * digit_classes;
    const float* pow10_row =
        pow10_logits + static_cast<size_t>(decoder_row) * pow10_buckets;

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
    atomicAdd(loss_sum, (digit_nll + pow10_nll + stop_bce) * scale);
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
    if (forward_outputs.atom_count != static_cast<int>(payload.atom_positions.size()) ||
        forward_outputs.decoder_step_capacity !=
            payload.number_aux_target_digit_slots ||
        forward_outputs.decoder_row_count != forward_outputs.atom_count *
            (forward_outputs.decoder_step_capacity + 1) ||
        forward_outputs.digit_classes != 10 ||
        forward_outputs.pow10_buckets !=
            2 * payload.number_aux_target_max_abs_pow10 + 1) {
        throw std::runtime_error(
            "NumericAtomLoss: forward geometry disagrees with BatchPayload");
    }
    requireLogitShape(
        forward_outputs.digit_logits,
        forward_outputs.decoder_row_count,
        forward_outputs.digit_classes,
        "NumericAtomLoss.digit_logits");
    requireLogitShape(
        forward_outputs.pow10_logits,
        forward_outputs.decoder_row_count,
        forward_outputs.pow10_buckets,
        "NumericAtomLoss.pow10_logits");
    requireLogitShape(
        forward_outputs.stop_logits,
        forward_outputs.decoder_row_count,
        1,
        "NumericAtomLoss.stop_logits");

    if (!bindings.d_number_aux_target_valid ||
        !bindings.d_number_aux_target_digits ||
        !bindings.d_number_aux_target_pow10_index ||
        !bindings.d_number_aux_target_digit_mask) {
        throw std::runtime_error(
            "NumericAtomLoss: numeric target device bindings are incomplete");
    }

    const int digit_slots = payload.number_aux_target_digit_slots;
    int valid_digit_steps = 0;
    for (int atom = 0; atom < forward_outputs.atom_count; ++atom) {
        if (payload.number_aux_target_valid[static_cast<size_t>(atom)] == 0) continue;
        valid_digit_steps += static_cast<int>(
            payload.number_aux_target_digit_count[static_cast<size_t>(atom)]);
    }
    const int valid_stop_steps = payload.number_aux_target_valid_count;
    if (valid_digit_steps == 0) {
        return Tensor();
    }
    if (valid_stop_steps == 0) {
        throw std::runtime_error(
            "NumericAtomLoss: numeric digit supervision has no typed-CLOSE stop target");
    }

    Tensor result = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.loss");
    const float scale = 1.0f /
        static_cast<float>(3 * valid_digit_steps + valid_stop_steps);
    const int blocks =
        (forward_outputs.decoder_row_count + kBlockSize - 1) / kBlockSize;
    kernelNumericAtomLoss<<<blocks, kBlockSize, 0, stream>>>(
        forward_outputs.digit_logits.data,
        forward_outputs.pow10_logits.data,
        forward_outputs.stop_logits.data,
        bindings.d_number_aux_target_valid,
        bindings.d_number_aux_target_digits,
        bindings.d_number_aux_target_pow10_index,
        bindings.d_number_aux_target_digit_mask,
        result.data,
        forward_outputs.decoder_row_count,
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
