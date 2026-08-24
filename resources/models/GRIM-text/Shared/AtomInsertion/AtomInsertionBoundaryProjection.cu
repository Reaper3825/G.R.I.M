//======================================================//
//  AtomInsertionBoundaryProjection.cu
//  Primitive-composed adjacent-state boundary projection
//======================================================//

#include "AtomInsertionBoundaryProjection.hpp"

#include <stdexcept>
#include <string>

namespace GRIM::AtomInsertion {

namespace {

std::string requireCaller(const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error(
            "AtomInsertionBoundaryProjection requires a non-empty caller label");
    }
    return std::string(caller) + ": AtomInsertionBoundaryProjection";
}

void requireShape(
    const Tensor& tensor,
    int expected_rows,
    int expected_cols,
    const std::string& name,
    const std::string& prefix) {
    tensor.require(prefix.c_str());
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(prefix + ": " + name + " must be 2D");
    }
    const auto shape = tensor.shape.as_2d();
    if (shape.rows != expected_rows || shape.cols != expected_cols) {
        throw std::runtime_error(
            prefix + ": " + name + " shape mismatch; expected [" +
            std::to_string(expected_rows) + "," +
            std::to_string(expected_cols) + "] got [" +
            std::to_string(shape.rows) + "," +
            std::to_string(shape.cols) + "]");
    }
}

void validateParameterTensors(
    const AtomInsertionBoundaryParameterTensors& parameters,
    const HyperParameters::AtomInsertionBoundaryProjectionHP& hp,
    const char* caller) {
    const std::string prefix = requireCaller(caller);
    if (!hp.enabled) {
        throw std::runtime_error(prefix + ": atom insertion is disabled");
    }
    if (hp.d_model <= 0) {
        throw std::runtime_error(prefix + ": d_model must be positive");
    }
    requireShape(
        parameters.left_projection_weight,
        hp.d_model,
        hp.d_model,
        "left_projection_weight",
        prefix);
    requireShape(
        parameters.right_projection_weight,
        hp.d_model,
        hp.d_model,
        "right_projection_weight",
        prefix);
    requireShape(
        parameters.projection_bias,
        1,
        hp.d_model,
        "projection_bias",
        prefix);
}

} // namespace

void forwardAtomInsertionBoundaryProjection(
    const HyperParameters::AtomInsertionBoundaryProjectionHP& hp,
    const AtomInsertionBoundaryParameterTensors& parameters,
    const Tensor& contextual_states,
    const Batching::BatchPayload& payload,
    bool EnableAtomIdentification,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs) {
    constexpr const char* caller = "forwardAtomInsertionBoundaryProjection";
    if (!EnableAtomIdentification) {
        throw std::runtime_error(
            "forwardAtomInsertionBoundaryProjection: "
            "EnableAtomIdentification is false");
    }
    if (!stream) {
        throw std::runtime_error(
            "forwardAtomInsertionBoundaryProjection: stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error(
            "forwardAtomInsertionBoundaryProjection: cublas_handle is NULL");
    }

    validateParameterTensors(parameters, hp, caller);
    payload.validate(caller);
    if (payload.max_seq_len < 2) {
        throw std::runtime_error(
            "forwardAtomInsertionBoundaryProjection: max_seq_len must include "
            "at least BOS and EOS rows");
    }
    requireShape(
        contextual_states,
        payload.total_tokens,
        hp.d_model,
        "contextual_states",
        requireCaller(caller));

    autograd::set_autograd_cublas_handle(cublas_handle);
    const int gap_rows_per_sequence = payload.max_seq_len - 1;

    // These are ordinary primitive results committed to the caller-owned
    // forward lifetime boundary. The two selectors share the same contextual
    // producer; AutogradEngine discovers both edges and performs their fan-in
    // through GradFn::receive_gradient/accumulate_grad before that producer runs.
    forward_outputs.atom_insertion_left_contextual_states =
        autograd::select_fixed_group_rows(
            contextual_states,
            payload.batch_size,
            payload.max_seq_len,
            0,
            gap_rows_per_sequence,
            stream);
    forward_outputs.atom_insertion_right_contextual_states =
        autograd::select_fixed_group_rows(
            contextual_states,
            payload.batch_size,
            payload.max_seq_len,
            1,
            gap_rows_per_sequence,
            stream);

    forward_outputs.atom_insertion_left_projected = autograd::matmul(
        forward_outputs.atom_insertion_left_contextual_states,
        parameters.left_projection_weight,
        stream);
    forward_outputs.atom_insertion_right_projected = autograd::matmul(
        forward_outputs.atom_insertion_right_contextual_states,
        parameters.right_projection_weight,
        stream);
    forward_outputs.atom_insertion_projection_sum = autograd::add(
        forward_outputs.atom_insertion_left_projected,
        forward_outputs.atom_insertion_right_projected,
        stream);
    forward_outputs.atom_insertion_gap_states = autograd::broadcast_add(
        forward_outputs.atom_insertion_projection_sum,
        parameters.projection_bias,
        stream);
}

} // namespace GRIM::AtomInsertion
