//======================================================//
//  SelectFixedGroupRowsGradFn.cu
//  Primitive row selection within a fixed rectangular group layout.
//======================================================//

#include "SelectFixedGroupRowsGradFn.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIM::autograd {

namespace {

constexpr int kBlockSize = 256;

int blocksFor(std::size_t element_count, const char* caller) {
    if (element_count == 0) {
        throw std::runtime_error(std::string(caller) + ": element count is zero");
    }
    const std::size_t block_count =
        1 + (element_count - 1) / static_cast<std::size_t>(kBlockSize);
    if (block_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string(caller) + ": CUDA grid size is outside int range");
    }
    return static_cast<int>(block_count);
}

void throwIfCudaFailed(cudaError_t status, const char* caller) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + cudaGetErrorString(status));
    }
}

__device__ std::size_t selectedInputIndex(
    std::size_t output_index,
    int rows_per_group,
    int row_offset,
    int selected_rows_per_group,
    int feature_count) {
    const std::size_t output_row = output_index / feature_count;
    const int feature = static_cast<int>(output_index % feature_count);
    const int group = static_cast<int>(
        output_row / static_cast<std::size_t>(selected_rows_per_group));
    const int selected_row = static_cast<int>(
        output_row % static_cast<std::size_t>(selected_rows_per_group));
    const int input_row =
        group * rows_per_group + row_offset + selected_row;
    return static_cast<std::size_t>(input_row) * feature_count + feature;
}

__global__ void kernelSelectFixedGroupRowsForward(
    const float* __restrict__ input,
    float* __restrict__ output,
    std::size_t output_count,
    int rows_per_group,
    int row_offset,
    int selected_rows_per_group,
    int feature_count) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) {
        return;
    }
    output[output_index] = input[selectedInputIndex(
        output_index,
        rows_per_group,
        row_offset,
        selected_rows_per_group,
        feature_count)];
}

__global__ void kernelSelectFixedGroupRowsBackward(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    std::size_t output_count,
    int rows_per_group,
    int row_offset,
    int selected_rows_per_group,
    int feature_count) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) {
        return;
    }
    const std::size_t input_index = selectedInputIndex(
        output_index,
        rows_per_group,
        row_offset,
        selected_rows_per_group,
        feature_count);
    grad_input[input_index] += grad_output[output_index];
}

} // namespace

SelectFixedGroupRowsGradFn::SelectFixedGroupRowsGradFn() {
    op_name = "select_fixed_group_rows";
}

void SelectFixedGroupRowsGradFn::captureInput(
    Tensor& input,
    cudaStream_t stream) {
    if (input.requires_grad) {
        input_gradient = capture_input_gradient(
            input,
            stream,
            "SelectFixedGroupRowsGradFn::captureInput");
    }
}

void SelectFixedGroupRowsGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("select_fixed_group_rows", this);
    if (applied) {
        return;
    }
    applied = true;
    if (!input_gradient) {
        return;
    }

    grad_output.require("SelectFixedGroupRowsGradFn::apply grad_output");
    if (!grad_output.shape.is_2d_layout()) {
        throw std::runtime_error(
            "SelectFixedGroupRowsGradFn::apply: grad_output must be 2D");
    }
    const int expected_rows = group_count * selected_rows_per_group;
    const auto grad_shape = grad_output.shape.as_2d();
    if (grad_shape.rows != expected_rows ||
        grad_shape.cols != feature_count) {
        throw std::runtime_error(
            "SelectFixedGroupRowsGradFn::apply: grad_output shape mismatch");
    }

    const std::size_t output_count =
        static_cast<std::size_t>(expected_rows) * feature_count;
    kernelSelectFixedGroupRowsBackward<<<
        blocksFor(output_count, "SelectFixedGroupRowsGradFn::apply"),
        kBlockSize,
        0,
        stream>>>(
            grad_output.data,
            input_gradient->data,
            output_count,
            rows_per_group,
            row_offset,
            selected_rows_per_group,
            feature_count);
    throwIfCudaFailed(
        cudaGetLastError(),
        "SelectFixedGroupRowsGradFn::apply backward launch");

    propagate_input_gradient(
        input_gradient,
        stream,
        backward_payload,
        backward_bindings,
        "SelectFixedGroupRowsGradFn::apply");
}

void SelectFixedGroupRowsGradFn::release_saved() {
    GradFn::release_saved();
    input_gradient.reset();
}

Tensor select_fixed_group_rows(
    const Tensor& input,
    int group_count,
    int rows_per_group,
    int row_offset,
    int selected_rows_per_group,
    cudaStream_t stream) {
    constexpr const char* caller = "autograd::select_fixed_group_rows";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    input.require(caller);
    if (!input.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(caller) + ": input must be 2D");
    }
    if (group_count <= 0 || rows_per_group <= 0 ||
        row_offset < 0 || selected_rows_per_group <= 0 ||
        selected_rows_per_group > rows_per_group ||
        row_offset > rows_per_group - selected_rows_per_group) {
        throw std::runtime_error(
            std::string(caller) + ": invalid fixed-group selection geometry");
    }
    if (group_count >
        std::numeric_limits<int>::max() / rows_per_group) {
        throw std::runtime_error(
            std::string(caller) + ": input row geometry overflows int");
    }
    if (group_count >
        std::numeric_limits<int>::max() / selected_rows_per_group) {
        throw std::runtime_error(
            std::string(caller) + ": output row geometry overflows int");
    }

    const auto input_shape = input.shape.as_2d();
    if (input_shape.rows != group_count * rows_per_group) {
        throw std::runtime_error(
            std::string(caller) + ": input row count does not match "
            "group_count * rows_per_group");
    }

    const int output_rows = group_count * selected_rows_per_group;
    const int feature_count = input_shape.cols;
    const std::size_t output_count =
        static_cast<std::size_t>(output_rows) * feature_count;
    Tensor result = Tensor::empty(
        TensorContract::TensorShape::make_BSM(output_rows, feature_count),
        input.requires_grad,
        stream,
        "select_fixed_group_rows_result");

    kernelSelectFixedGroupRowsForward<<<
        blocksFor(output_count, caller),
        kBlockSize,
        0,
        stream>>>(
            input.data,
            result.data,
            output_count,
            rows_per_group,
            row_offset,
            selected_rows_per_group,
            feature_count);
    throwIfCudaFailed(
        cudaGetLastError(),
        "autograd::select_fixed_group_rows forward launch");

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<SelectFixedGroupRowsGradFn>();
        grad_fn->group_count = group_count;
        grad_fn->rows_per_group = rows_per_group;
        grad_fn->row_offset = row_offset;
        grad_fn->selected_rows_per_group = selected_rows_per_group;
        grad_fn->feature_count = feature_count;
        grad_fn->captureInput(const_cast<Tensor&>(input), stream);
        result.grad_fn = std::move(grad_fn);
    }

    return result;
}

} // namespace GRIM::autograd
