//======================================================//
//  NumericAtomForward.cu
//  Typed auxiliary forward branch for numeric atoms.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "NumericAtomForward.hpp"

#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <stdexcept>
#include <string>

namespace GRIM {
namespace Forward {

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
    if (!shared_hidden_state.shape.is_2d_layout() ||
        !parameters.digit_emb.shape.is_2d_layout() ||
        !parameters.pow10_emb.shape.is_2d_layout()) {
        throw std::runtime_error(
            "NumericAtomForward: hidden state and classifier embeddings must be 2D");
    }

    const auto hidden_shape = shared_hidden_state.shape.as_2d();
    const auto digit_shape = parameters.digit_emb.shape.as_2d();
    const auto pow10_shape = parameters.pow10_emb.shape.as_2d();
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
        pow10_shape.cols != hidden_shape.cols || pow10_shape.rows <= 0) {
        throw std::runtime_error(
            "NumericAtomForward: classifier embedding geometry does not match shared hidden width");
    }
    if (payload.number_aux_target_digit_slots > 0) {
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
    outputs.total_rows = hidden_shape.rows;
    outputs.digit_classes = digit_shape.rows;
    outputs.pow10_buckets = pow10_shape.rows;

    // Tied classifiers: the same digit and place embeddings that represent
    // authored numbers provide the output class directions. These primitive
    // matmuls attach the numeric branch to the existing autograd graph.
    outputs.digit_logits = autograd::matmul(
        shared_hidden_state,
        parameters.digit_emb,
        stream,
        /*transpose_b=*/true);
    outputs.pow10_logits = autograd::matmul(
        shared_hidden_state,
        parameters.pow10_emb,
        stream,
        /*transpose_b=*/true);

    return outputs;
}

}  // namespace Forward
}  // namespace GRIM
