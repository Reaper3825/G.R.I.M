#include "LoRALinear.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {
namespace {

const TensorContract::Shape2D& requireMatrix(
    const Tensor& tensor,
    const char* name) {
    tensor.require(name);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(
            std::string("lora_linear: ") + name + " must be a 2D tensor");
    }
    const auto& shape = tensor.shape.as_2d();
    if (shape.rows <= 0 || shape.cols <= 0) {
        throw std::runtime_error(
            std::string("lora_linear: ") + name + " has non-positive dimensions");
    }
    return shape;
}

void requireShape(
    const TensorContract::Shape2D& shape,
    int expected_rows,
    int expected_cols,
    const char* name) {
    if (shape.rows != expected_rows || shape.cols != expected_cols) {
        throw std::runtime_error(
            std::string("lora_linear: ") + name + " shape mismatch expected=[" +
            std::to_string(expected_rows) + "," +
            std::to_string(expected_cols) + "] actual=[" +
            std::to_string(shape.rows) + "," +
            std::to_string(shape.cols) + "]");
    }
}

} // namespace

Tensor lora_linear(
    const Tensor& x,
    const Tensor& W_base,
    const LoRAProjectionView* lora,
    MatmulOrientation orientation,
    cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error(
            "lora_linear: stream is NULL - caller MUST provide valid stream");
    }

    const auto& x_shape = requireMatrix(x, "x");
    const auto& base_shape = requireMatrix(W_base, "W_base");
    const bool transpose_weight =
        orientation == MatmulOrientation::TRANSPOSED_WEIGHT;
    if (!transpose_weight && orientation != MatmulOrientation::DIRECT_WEIGHT) {
        throw std::runtime_error("lora_linear: unknown MatmulOrientation");
    }

    const int input_width = x_shape.cols;
    const int output_width = transpose_weight
        ? base_shape.rows
        : base_shape.cols;
    const int base_input_width = transpose_weight
        ? base_shape.cols
        : base_shape.rows;
    if (input_width != base_input_width) {
        throw std::runtime_error(
            "lora_linear: x and W_base input dimensions do not match");
    }

    if (!lora) {
        return matmul(x, W_base, stream, transpose_weight);
    }
    if (!lora->A || !lora->B) {
        throw std::runtime_error(
            "lora_linear: enabled adapter requires non-NULL A and B tensors");
    }
    if (lora->rank == 0 ||
        lora->rank > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        !std::isfinite(lora->scale) || lora->scale <= 0.0f) {
        throw std::runtime_error(
            "lora_linear: adapter rank and scale must be positive");
    }
    if (W_base.requires_grad) {
        throw std::runtime_error(
            "lora_linear: adapted W_base must be frozen");
    }

    Tensor& A = *lora->A;
    Tensor& B = *lora->B;
    const auto& a_shape = requireMatrix(A, "A");
    const auto& b_shape = requireMatrix(B, "B");
    if (A.requires_grad != B.requires_grad) {
        throw std::runtime_error(
            "lora_linear: A and B must have identical trainability");
    }
    if (A.precision() != TensorContract::PrecisionType::FP32 ||
        B.precision() != TensorContract::PrecisionType::FP32) {
        throw std::runtime_error(
            "lora_linear: v1 adapter tensors must use FP32 compute precision");
    }

    const int rank = static_cast<int>(lora->rank);
    if (transpose_weight) {
        requireShape(a_shape, rank, input_width, "A");
        requireShape(b_shape, output_width, rank, "B");
    } else {
        requireShape(b_shape, input_width, rank, "B");
        requireShape(a_shape, rank, output_width, "A");
    }

    Tensor base_out = matmul(x, W_base, stream, transpose_weight);
    Tensor rank_out;
    Tensor delta_out;
    if (transpose_weight) {
        rank_out = matmul(x, A, stream, true);
        delta_out = matmul(rank_out, B, stream, true);
    } else {
        rank_out = matmul(x, B, stream, false);
        delta_out = matmul(rank_out, A, stream, false);
    }

    const auto& base_out_shape = requireMatrix(base_out, "base_out");
    const auto& delta_out_shape = requireMatrix(delta_out, "delta_out");
    requireShape(base_out_shape, x_shape.rows, output_width, "base_out");
    requireShape(delta_out_shape, x_shape.rows, output_width, "delta_out");

    Tensor scaled_delta = mul_scalar(delta_out, lora->scale, stream);
    return add(base_out, scaled_delta, stream);
}

} // namespace GRIM::autograd