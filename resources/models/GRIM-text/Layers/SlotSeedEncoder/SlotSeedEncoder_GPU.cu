//======================================================//
//  SlotSeedEncoder_GPU.cu
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "SlotSeedEncoder_GPU.hpp"

#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace SlotSeedEncoder {
namespace {

const TensorContract::Shape2D& require2D(
    const Tensor& tensor,
    const char* label)
{
    tensor.require(label);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(
            std::string("SlotSeedEncoder::forward: ") + label +
            " must be a 2D tensor");
    }
    return tensor.shape.as_2d();
}

void requireShape(
    const Tensor& tensor,
    int expected_rows,
    int expected_cols,
    const char* label)
{
    const auto& shape = require2D(tensor, label);
    if (shape.rows != expected_rows || shape.cols != expected_cols) {
        throw std::runtime_error(
            std::string("SlotSeedEncoder::forward: ") + label + " shape=[" +
            std::to_string(shape.rows) + ", " + std::to_string(shape.cols) +
            "] expected=[" + std::to_string(expected_rows) + ", " +
            std::to_string(expected_cols) + "]");
    }
}

}  // namespace

void forward(
    const HyperParameters::SlotSeedEncoderConstructionHP& hp,
    const SlotSeedEncoderParameterTensors& parameter_tensors,
    const Tensor& contextual_hidden_states,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int num_slots,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs)
{
    if (!hp.enabled) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: called while SlotSeedEncoder is disabled");
    }
    if (!stream) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: stream is NULL");
    }
    if (payload.batch_size <= 0 || payload.total_tokens <= 0 ||
        payload.max_seq_len <= 0) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: BatchPayload geometry must be positive");
    }
    if (num_slots <= 0) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: num_slots must be > 0");
    }
    if (static_cast<int>(payload.token_to_slot_map.size()) != payload.total_tokens) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: BatchPayload.token_to_slot_map size does "
            "not match total_tokens");
    }
    if (payload.atom_positions.size() != payload.atom_types.size()) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: BatchPayload atom position/type channels "
            "have different sizes");
    }
    if (forward_outputs.slot_seed_contextual_input.data ||
        forward_outputs.slot_seeds.data) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: ModelForwardOutputs already owns slot-seed "
            "state for this forward pass");
    }

    requireShape(
        contextual_hidden_states,
        payload.total_tokens,
        hp.d_model,
        "contextual_hidden_states");
    if (!parameter_tensors.W_seed_in.data ||
        !parameter_tensors.W_seed_out.data) {
        throw std::runtime_error(
            "SlotSeedEncoder::forward: required weight tensor is not initialized");
    }
    requireShape(
        parameter_tensors.W_seed_in,
        hp.d_model,
        hp.d_hidden,
        "W_seed_in");
    requireShape(
        parameter_tensors.W_seed_out,
        hp.d_hidden,
        hp.d_model,
        "W_seed_out");
    if (hp.bias_enabled) {
        if (!parameter_tensors.b_seed_in.data ||
            !parameter_tensors.b_seed_out.data) {
            throw std::runtime_error(
                "SlotSeedEncoder::forward: bias is enabled but a bias tensor is "
                "not initialized");
        }
        requireShape(
            parameter_tensors.b_seed_in, 1, hp.d_hidden, "b_seed_in");
        requireShape(
            parameter_tensors.b_seed_out, 1, hp.d_model, "b_seed_out");
    }
    if (hp.type_embedding_enabled) {
        if (!parameter_tensors.type_embeddings.data) {
            throw std::runtime_error(
                "SlotSeedEncoder::forward: type embeddings are enabled but the "
                "parameter tensor is not initialized");
        }
        requireShape(
            parameter_tensors.type_embeddings,
            2,
            hp.d_model,
            "type_embeddings");
    }

    int authored_slot_count = 0;
    for (int token = 0; token < payload.total_tokens; ++token) {
        const int slot = payload.token_to_slot_map[static_cast<std::size_t>(token)];
        if (slot < 0) {
            continue;
        }
        if (slot >= num_slots) {
            throw std::runtime_error(
                "SlotSeedEncoder::forward: token_to_slot_map[" +
                std::to_string(token) + "]=" + std::to_string(slot) +
                " is outside num_slots=" + std::to_string(num_slots));
        }
        const int batch_row = token / payload.max_seq_len;
        if (batch_row < 0 || batch_row >= payload.batch_size) {
            throw std::runtime_error(
                "SlotSeedEncoder::forward: routed token resolves outside batch geometry");
        }
        if (payload.atom_mask[static_cast<std::size_t>(token)] == 0 ||
            !Tokenizer::isAtomTokenId(
                payload.input_ids[static_cast<std::size_t>(token)])) {
            throw std::runtime_error(
                "SlotSeedEncoder::forward: routed token is not an authored atom");
        }
        const auto atom_type = Tokenizer::tokenIdToAtomType(
            payload.input_ids[static_cast<std::size_t>(token)]);
        if (!Tokenizer::isNumericAtom(atom_type)) {
            throw std::runtime_error(
                "SlotSeedEncoder::forward: routed atom is not <INT> or <FLOAT>");
        }
        ++authored_slot_count;
    }

    if (authored_slot_count == 0) {
        return;
    }

    forward_outputs.slot_seed_contextual_input =
        autograd::gather_slot_seed_inputs(
            contextual_hidden_states,
            parameter_tensors.type_embeddings,
            hp.type_embedding_enabled,
            payload,
            bindings,
            num_slots,
            stream);
    const Tensor* contextual_input =
        &forward_outputs.slot_seed_contextual_input;

    forward_outputs.slot_seed_hidden_pre_activation = autograd::matmul(
        *contextual_input, parameter_tensors.W_seed_in, stream);
    if (hp.bias_enabled) {
        forward_outputs.slot_seed_hidden_pre_activation = autograd::broadcast_add(
            forward_outputs.slot_seed_hidden_pre_activation,
            parameter_tensors.b_seed_in,
            stream);
    }
    forward_outputs.slot_seed_hidden_activation = autograd::silu(
        forward_outputs.slot_seed_hidden_pre_activation,
        stream,
        forward_outputs.slot_seed_hidden_pre_activation.data);

    forward_outputs.slot_seed_residual_delta = autograd::matmul(
        forward_outputs.slot_seed_hidden_activation,
        parameter_tensors.W_seed_out,
        stream);
    if (hp.bias_enabled) {
        forward_outputs.slot_seed_residual_delta = autograd::broadcast_add(
            forward_outputs.slot_seed_residual_delta,
            parameter_tensors.b_seed_out,
            stream);
    }
    forward_outputs.slot_seed_unmasked = autograd::residual_add(
        *contextual_input,
        forward_outputs.slot_seed_residual_delta,
        stream);

    // Biases may make absent dense slot rows non-zero. Reapply the existing
    // payload slot topology without materializing a second validity mask.
    forward_outputs.slot_seeds = autograd::mask_unauthored_slot_rows(
        forward_outputs.slot_seed_unmasked,
        payload,
        bindings,
        num_slots,
        stream);
}

}  // namespace SlotSeedEncoder
}  // namespace GRIM
