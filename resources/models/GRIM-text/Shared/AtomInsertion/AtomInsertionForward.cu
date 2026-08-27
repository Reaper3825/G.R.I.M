//======================================================//
//  AtomInsertionForward.cu
//  Downstream forward entry for the atom-identification model
//======================================================//

#include "AtomInsertionForward.hpp"

#include <stdexcept>
#include <string>

namespace GRIM::AtomInsertion {

namespace {

std::string callerPrefix(const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error(
            "AtomInsertionForward requires a non-empty caller label");
    }
    return std::string(caller) + ": AtomInsertionForward";
}

void requireMatrix(
    const Tensor& tensor,
    int expected_rows,
    int expected_cols,
    const char* name,
    const std::string& prefix) {
    tensor.require(prefix.c_str());
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(
            prefix + ": " + name + " must be a 2D tensor");
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

void requireFreshAtomOutputs(
    const Forward::ModelForwardOutputs& outputs,
    const std::string& prefix) {
    if (outputs.atom_insertion_left_contextual_states.data ||
        outputs.atom_insertion_right_contextual_states.data ||
        outputs.atom_insertion_left_projected.data ||
        outputs.atom_insertion_right_projected.data ||
        outputs.atom_insertion_projection_sum.data ||
        outputs.atom_insertion_gap_states.data ||
        outputs.atom_insertion_decision_logits.data ||
        outputs.logits_tensor.data) {
        throw std::runtime_error(
            prefix + ": atom/logit outputs are already populated; caller must "
            "provide the fresh ModelForwardOutputs for this forward pass");
    }
}

} // namespace

void forwardAtomInsertion(
    const HyperParameters::AtomInsertionBoundaryProjectionHP& boundary_hp,
    const AtomInsertionBoundaryParameterTensors& boundary_parameters,
    const HyperParameters::LMHeadLayerConstructionHP& lm_head_hp,
    const LMHeadParameterTensors& lm_head_parameters,
    const Tensor& contextual_states,
    const Batching::BatchPayload& payload,
    bool EnableAtomIdentification,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs) {
    constexpr const char* caller = "forwardAtomInsertion";
    const std::string prefix = callerPrefix(caller);
    if (!EnableAtomIdentification || !payload.EnableAtomIdentification) {
        throw std::runtime_error(
            prefix + ": EnableAtomIdentification must be true on both the "
            "entry point and BatchPayload");
    }
    if (!stream) {
        throw std::runtime_error(prefix + ": stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error(prefix + ": cublas_handle is NULL");
    }

    payload.validate(caller);
    if (!lm_head_hp.atom_insertion_enabled) {
        throw std::runtime_error(
            prefix + ": compiled LM-head grouping does not enable atom insertion");
    }
    if (lm_head_hp.atom_insertion_enabled != boundary_hp.enabled) {
        throw std::runtime_error(
            prefix + ": LM-head and atom boundary groupings disagree on task mode");
    }
    if (lm_head_hp.d_model != boundary_hp.d_model) {
        throw std::runtime_error(
            prefix + ": LM-head d_model does not match boundary projection");
    }
    if (payload.vocab_size != lm_head_hp.vocab_size) {
        throw std::runtime_error(
            prefix + ": LM-head vocab_size does not match BatchPayload");
    }
    if (lm_head_hp.vocab_size < Tokenizer::UNIGRAM_VOCAB_OFFSET) {
        throw std::runtime_error(
            prefix + ": vocabulary does not contain all atom delimiter IDs");
    }
    requireMatrix(
        contextual_states,
        payload.total_tokens,
        boundary_hp.d_model,
        "contextual_states",
        prefix);
    requireFreshAtomOutputs(forward_outputs, prefix);

    autograd::set_autograd_cublas_handle(cublas_handle);
    forwardAtomInsertionBoundaryProjection(
        boundary_hp,
        boundary_parameters,
        contextual_states,
        payload,
        EnableAtomIdentification,
        stream,
        cublas_handle,
        forward_outputs);

    // Reuse the complete existing LM head and its registry-owned parameter
    // bundle. The atom loss later selects the compact OPEN-type + EXIT window.
    forwardLmHead(
        lm_head_hp,
        lm_head_parameters,
        forward_outputs.atom_insertion_gap_states,
        payload,
        stream,
        cublas_handle,
        forward_outputs);

    requireMatrix(
        forward_outputs.logits_tensor,
        payload.atomInsertionGapRowCount(),
        lm_head_hp.vocab_size,
        "logits_tensor",
        prefix);
}

} // namespace GRIM::AtomInsertion
