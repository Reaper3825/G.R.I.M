#pragma once

//======================================================//
//  ParameterRegistry — adjacent startup parameter-owner
//  declarations plus header-only migrated tensor-group
//  inventory slices.
//
//  Current migrated surface:
//    - Embedding durable tensor owner
//    - LM head durable tensor owner
//    - Encoder durable per-layer parameter tensor owner
//    - NumberEncoder durable tensor owner
//    - NumberEncoder parameter-group inventory
//    - SlotSeedEncoder durable tensor owner
//    - SlotSeedEncoder parameter-group inventory
//    - ExecutionBlock durable parameter tensor owner
//    - ExecutionBlock parameter-group inventory
//    - FeedForward durable per-layer parameter tensor owner
//    - FeedForward parameter-group inventory
//    - Durable ParameterGroup inventory owner
//
//  Registration/validation/transaction ownership stays
//  in ParameterGroupRegistration.cu.
//======================================================//

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct EmbeddingParameterTensors {
    Tensor token_weights;  // [vocab_size, d_model]
};

// LM head durable tensor owner. Registry-owned like every other parameter
// tensor bundle; LM-head forward/view helpers in Layers/LMHead borrow these.
//   weights         [vocab_size, d_model] (tied to embedding when owns_weights=false)
//   bias            [vocab_size] when config.use_bias=true
//   final_rms_gamma [d_model] pre-LM-head normalization gain
//
// Head-side residual SwiGLU adapter (config.lm_head_mlp_enabled):
//   u = z + alpha * ( SiLU(z @ mlp_W_gate) ⊙ (z @ mlp_W_up) ) @ mlp_W_down
// where z = RMSNorm(encoder_output). alpha is the authored config scalar
// lm_head_mlp_alpha (carried on LMHeadLayerConstructionHP, not a tensor).
//   mlp_W_gate [d_model, lm_head_mlp_d_ff]
//   mlp_W_up   [d_model, lm_head_mlp_d_ff]
//   mlp_W_down [lm_head_mlp_d_ff, d_model]
struct LMHeadParameterTensors {
    Tensor weights;
    Tensor bias;
    Tensor final_rms_gamma;
    Tensor mlp_W_gate;
    Tensor mlp_W_up;
    Tensor mlp_W_down;
    bool owns_weights = true;
};

// NumericAtom decoder parameters. The decoder reuses digit_emb/pow10_emb as
// tied input/output tables and owns Wz/Uz, Wr/Ur, Wh/Uh for its transition.
struct NumberEncoderParameterTensors {
    Tensor digit_emb;   // [10, d_model] digit identity embedding
    Tensor pow10_emb;   // [pow10_buckets, d_model] place identity embedding
    Tensor numeric_atom_Wz; // [2 * d_model, d_model] GRU update-gate input projection
    Tensor numeric_atom_Uz; // [d_model, d_model] GRU update-gate state projection
    Tensor numeric_atom_Wr; // [2 * d_model, d_model] GRU reset-gate input projection
    Tensor numeric_atom_Ur; // [d_model, d_model] GRU reset-gate state projection
    Tensor numeric_atom_Wh; // [2 * d_model, d_model] GRU candidate input projection
    Tensor numeric_atom_Uh; // [d_model, d_model] GRU candidate state projection
    Tensor numeric_atom_sign_classifier; // [1, d_model] negative-sign projection
    Tensor numeric_atom_stop_classifier; // [1, d_model] typed CLOSE / stop projection
};

// Arg/option selector head parameters (execution-INDEPENDENT). A single query
// projection W_q maps encoder hidden states into an externally owned
// candidate-key space.
struct SelectorParameterTensors {
    Tensor W_q;   // [d_model, d_model] query projection
};

// SlotSeedEncoder — contextual numeric-placeholder representation.
//   hidden = SiLU(context @ W_seed_in + b_seed_in)
//   seed   = context + hidden @ W_seed_out + b_seed_out
// The optional type embedding is added to context before the residual MLP.
struct SlotSeedEncoderParameterTensors {
    Tensor W_seed_in;       // [d_model, d_hidden]
    Tensor b_seed_in;       // [1, d_hidden] when bias_enabled=true
    Tensor W_seed_out;      // [d_hidden, d_model]
    Tensor b_seed_out;      // [1, d_model] when bias_enabled=true
    Tensor type_embeddings; // [kAtomTypeCount, d_model] when enabled
};

struct ExecutionBlockParameterTensors {
    Tensor w_decode_1;
    Tensor b_decode_1;
    Tensor w_decode_2;
    Tensor w_arg1_select;
    Tensor w_arg2_select;
    Tensor W_arg1_to_arg2;
    Tensor W_op_select;
    Tensor W_key_proj;
    Tensor W_write_query;
    Tensor W_write_key;
    Tensor alpha;
    Tensor beta;
    Tensor step_embeddings;
    Tensor type_num_embed;
    Tensor W_value_to_emb;
    Tensor b_value_to_emb;
    Tensor w_inject_gate;
    Tensor W_Q_read;
    Tensor W_K_read;
    Tensor W_V_read;
    Tensor W_O_read;
    Tensor W_gate_read;
    Tensor tau;
    Tensor E_slot;
    Tensor E_op;
    Tensor W_scal;
    Tensor b_scal;
    Tensor W_trace;
    Tensor b_trace;
    Tensor W_reason_gate;
    Tensor W_trace_gate;
    Tensor W_execute;
    Tensor b_execute;
    Tensor W_stop;
    Tensor b_stop;
};

struct AttentionResidualGateParameterTensors {
    Tensor W_gate;  // [d_model, 1] when config.attention_residual_gate_enabled=true
    Tensor b_gate;  // [1] when config.attention_residual_gate_enabled=true
};

struct EncodingLayerParameterTensors {
    Tensor rms1_gamma;     // [d_model]
    Tensor rms2_gamma;     // [d_model]
    Tensor W_qkv;          // [qkv_dim, d_model]
    Tensor b_qkv;          // [qkv_dim] when config.use_bias=true
    Tensor W_o;            // [d_model, d_model]
    Tensor b_o;            // [d_model] when config.use_bias=true
    Tensor layer_scale1;   // [1, d_model] when config.use_layer_scale=true
    Tensor layer_scale2;   // [1, d_model] when config.use_layer_scale=true
    AttentionResidualGateParameterTensors attention_residual_gate;
};

struct FeedForwardParameterTensors {
    Tensor W_gate;  // [d_model, d_ff]
    Tensor W1;      // [d_model, d_ff]
    Tensor W2;      // [d_ff, d_model]
    Tensor b2;      // [1, d_model] when config.use_bias=true
};

} // namespace GRIM

namespace ParameterRegistry {

struct StartupParameterRegistry {
    std::unique_ptr<GRIM::EmbeddingParameterTensors> embedding_parameters;
    std::unique_ptr<GRIM::LMHeadParameterTensors> lm_head_parameters;
    std::vector<GRIM::EncodingLayerParameterTensors> encoding_layer_parameter_tensors;
    std::unique_ptr<GRIM::NumberEncoderParameterTensors> number_encoder_parameters;
    std::unique_ptr<GRIM::SelectorParameterTensors> selector_parameters;
    std::unique_ptr<GRIM::SlotSeedEncoderParameterTensors> slot_seed_encoder_parameters;
    std::unique_ptr<GRIM::ExecutionBlockParameterTensors> execution_block_parameters;
    std::vector<GRIM::FeedForwardParameterTensors> feed_forward_parameter_tensors;
    // Single durable optimizer/autograd parameter inventory owner.
    // ParameterGroup entries are non-owning views into the tensor owners in
    // this registry and startup-owned layer topology. Do not mirror this
    // vector elsewhere on TrainingContext.
    std::vector<GRIM::ParameterGroup> parameter_groups;

    GRIM::EmbeddingParameterTensors* getEmbeddingParameters() {
        return embedding_parameters.get();
    }

    const GRIM::EmbeddingParameterTensors* getEmbeddingParameters() const {
        return embedding_parameters.get();
    }

    GRIM::EmbeddingParameterTensors& requireEmbeddingParameters(const char* caller) {
        if (!embedding_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.embedding_parameters is NULL");
        }
        return *embedding_parameters;
    }

    const GRIM::EmbeddingParameterTensors& requireEmbeddingParameters(const char* caller) const {
        if (!embedding_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.embedding_parameters is NULL");
        }
        return *embedding_parameters;
    }

    GRIM::LMHeadParameterTensors* getLmHeadParameters() {
        return lm_head_parameters.get();
    }

    const GRIM::LMHeadParameterTensors* getLmHeadParameters() const {
        return lm_head_parameters.get();
    }

    GRIM::LMHeadParameterTensors& requireLmHeadParameters(const char* caller) {
        if (!lm_head_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.lm_head_parameters is NULL");
        }
        return *lm_head_parameters;
    }

    const GRIM::LMHeadParameterTensors& requireLmHeadParameters(const char* caller) const {
        if (!lm_head_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.lm_head_parameters is NULL");
        }
        return *lm_head_parameters;
    }

    std::vector<GRIM::EncodingLayerParameterTensors>& encodingLayerParameterTensors() {
        return encoding_layer_parameter_tensors;
    }

    const std::vector<GRIM::EncodingLayerParameterTensors>& encodingLayerParameterTensors() const {
        return encoding_layer_parameter_tensors;
    }

    GRIM::EncodingLayerParameterTensors& requireEncodingLayerParameters(int layer, const char* caller) {
        if (layer < 0 || layer >= static_cast<int>(encoding_layer_parameter_tensors.size())) {
            throw std::runtime_error(std::string(caller) + ": missing EncodingLayerParameterTensors for layer " +
                                     std::to_string(layer) + " registry_size=" +
                                     std::to_string(encoding_layer_parameter_tensors.size()));
        }
        return encoding_layer_parameter_tensors[static_cast<std::size_t>(layer)];
    }

    const GRIM::EncodingLayerParameterTensors& requireEncodingLayerParameters(int layer, const char* caller) const {
        if (layer < 0 || layer >= static_cast<int>(encoding_layer_parameter_tensors.size())) {
            throw std::runtime_error(std::string(caller) + ": missing EncodingLayerParameterTensors for layer " +
                                     std::to_string(layer) + " registry_size=" +
                                     std::to_string(encoding_layer_parameter_tensors.size()));
        }
        return encoding_layer_parameter_tensors[static_cast<std::size_t>(layer)];
    }

    GRIM::AttentionResidualGateParameterTensors& requireAttentionResidualGateParameters(
        int layer,
        const char* caller) {
        auto& encoding_parameters = requireEncodingLayerParameters(layer, caller);
        auto& gate_parameters = encoding_parameters.attention_residual_gate;
        if (!gate_parameters.W_gate.data || !gate_parameters.b_gate.data) {
            throw std::runtime_error(
                std::string(caller) + ": attention residual gate parameters are unavailable for layer " +
                std::to_string(layer) +
                " (config.attention_residual_gate_enabled=false or startup initialization incomplete)");
        }
        return gate_parameters;
    }

    const GRIM::AttentionResidualGateParameterTensors& requireAttentionResidualGateParameters(
        int layer,
        const char* caller) const {
        const auto& encoding_parameters = requireEncodingLayerParameters(layer, caller);
        const auto& gate_parameters = encoding_parameters.attention_residual_gate;
        if (!gate_parameters.W_gate.data || !gate_parameters.b_gate.data) {
            throw std::runtime_error(
                std::string(caller) + ": attention residual gate parameters are unavailable for layer " +
                std::to_string(layer) +
                " (config.attention_residual_gate_enabled=false or startup initialization incomplete)");
        }
        return gate_parameters;
    }

    GRIM::ExecutionBlockParameterTensors* getExecutionBlockParameters() {
        return execution_block_parameters.get();
    }

    const GRIM::ExecutionBlockParameterTensors* getExecutionBlockParameters() const {
        return execution_block_parameters.get();
    }

    GRIM::NumberEncoderParameterTensors* getNumberEncoderParameters() {
        return number_encoder_parameters.get();
    }

    const GRIM::NumberEncoderParameterTensors* getNumberEncoderParameters() const {
        return number_encoder_parameters.get();
    }

    GRIM::NumberEncoderParameterTensors& requireNumberEncoderParameters(const char* caller) {
        if (!number_encoder_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.number_encoder_parameters is NULL");
        }
        return *number_encoder_parameters;
    }

    const GRIM::NumberEncoderParameterTensors& requireNumberEncoderParameters(const char* caller) const {
        if (!number_encoder_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.number_encoder_parameters is NULL");
        }
        return *number_encoder_parameters;
    }

    GRIM::SelectorParameterTensors* getSelectorParameters() { return selector_parameters.get(); }
    const GRIM::SelectorParameterTensors* getSelectorParameters() const { return selector_parameters.get(); }

    GRIM::SelectorParameterTensors& requireSelectorParameters(const char* caller) {
        if (!selector_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.selector_parameters is NULL");
        }
        return *selector_parameters;
    }

    const GRIM::SelectorParameterTensors& requireSelectorParameters(const char* caller) const {
        if (!selector_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.selector_parameters is NULL");
        }
        return *selector_parameters;
    }

    GRIM::SlotSeedEncoderParameterTensors* getSlotSeedEncoderParameters() {
        return slot_seed_encoder_parameters.get();
    }

    const GRIM::SlotSeedEncoderParameterTensors* getSlotSeedEncoderParameters() const {
        return slot_seed_encoder_parameters.get();
    }

    GRIM::SlotSeedEncoderParameterTensors& requireSlotSeedEncoderParameters(const char* caller) {
        if (!slot_seed_encoder_parameters) {
            throw std::runtime_error(
                std::string(caller) +
                ": StartupParameterRegistry.slot_seed_encoder_parameters is NULL");
        }
        return *slot_seed_encoder_parameters;
    }

    const GRIM::SlotSeedEncoderParameterTensors& requireSlotSeedEncoderParameters(
        const char* caller) const {
        if (!slot_seed_encoder_parameters) {
            throw std::runtime_error(
                std::string(caller) +
                ": StartupParameterRegistry.slot_seed_encoder_parameters is NULL");
        }
        return *slot_seed_encoder_parameters;
    }

    GRIM::ExecutionBlockParameterTensors& requireExecutionBlockParameters(const char* caller) {
        if (!execution_block_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.execution_block_parameters is NULL");
        }
        return *execution_block_parameters;
    }

    const GRIM::ExecutionBlockParameterTensors& requireExecutionBlockParameters(const char* caller) const {
        if (!execution_block_parameters) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.execution_block_parameters is NULL");
        }
        return *execution_block_parameters;
    }

    std::vector<GRIM::FeedForwardParameterTensors>& feedForwardParameterTensors() {
        return feed_forward_parameter_tensors;
    }

    const std::vector<GRIM::FeedForwardParameterTensors>& feedForwardParameterTensors() const {
        return feed_forward_parameter_tensors;
    }

    GRIM::FeedForwardParameterTensors& requireFeedForwardParameters(int layer, const char* caller) {
        if (layer < 0 || layer >= static_cast<int>(feed_forward_parameter_tensors.size())) {
            throw std::runtime_error(std::string(caller) + ": missing FeedForwardParameterTensors for layer " +
                                     std::to_string(layer) + " registry_size=" +
                                     std::to_string(feed_forward_parameter_tensors.size()));
        }
        return feed_forward_parameter_tensors[static_cast<std::size_t>(layer)];
    }

    const GRIM::FeedForwardParameterTensors& requireFeedForwardParameters(int layer, const char* caller) const {
        if (layer < 0 || layer >= static_cast<int>(feed_forward_parameter_tensors.size())) {
            throw std::runtime_error(std::string(caller) + ": missing FeedForwardParameterTensors for layer " +
                                     std::to_string(layer) + " registry_size=" +
                                     std::to_string(feed_forward_parameter_tensors.size()));
        }
        return feed_forward_parameter_tensors[static_cast<std::size_t>(layer)];
    }

    std::vector<GRIM::ParameterGroup>& parameterGroups() {
        return parameter_groups;
    }

    const std::vector<GRIM::ParameterGroup>& parameterGroups() const {
        return parameter_groups;
    }

    std::vector<GRIM::ParameterGroup>& requireParameterGroups(const char* caller) {
        if (parameter_groups.empty()) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.parameter_groups is empty");
        }
        return parameter_groups;
    }

    const std::vector<GRIM::ParameterGroup>& requireParameterGroups(const char* caller) const {
        if (parameter_groups.empty()) {
            throw std::runtime_error(std::string(caller) + ": StartupParameterRegistry.parameter_groups is empty");
        }
        return parameter_groups;
    }

    GRIM::ParameterGroup& requireParameterGroupForTensor(
        GRIM::Tensor& tensor,
        const std::string& label,
        const char* caller)
    {
        auto& groups = requireParameterGroups(caller);
        for (auto& group : groups) {
            if (group.tensor == &tensor ||
                (group.tensor && group.tensor->grad_data() == tensor.grad_data())) {
                return group;
            }
        }
        throw std::runtime_error(
            std::string(caller) + ": expected gradient tensor '" + label +
            "' is absent from the parameter registry");
    }
};

template <typename OwnerT>
struct TensorParameterSpec {
    const char* name;
    GRIM::Tensor OwnerT::* tensor_member;
    GRIM::ParamGroupType type;
    GRIM::ParamStatsBucket stats_bucket;
    int layer = -1;
};

using ExecutionBlockTensorParameterSpec =
    TensorParameterSpec<GRIM::ExecutionBlockParameterTensors>;

using NumberEncoderTensorParameterSpec =
    TensorParameterSpec<GRIM::NumberEncoderParameterTensors>;

using SelectorTensorParameterSpec =
    TensorParameterSpec<GRIM::SelectorParameterTensors>;

using SlotSeedEncoderTensorParameterSpec =
    TensorParameterSpec<GRIM::SlotSeedEncoderParameterTensors>;

using EncodingLayerTensorParameterSpec =
    TensorParameterSpec<GRIM::EncodingLayerParameterTensors>;

using AttentionResidualGateTensorParameterSpec =
    TensorParameterSpec<GRIM::AttentionResidualGateParameterTensors>;

using FeedForwardTensorParameterSpec =
    TensorParameterSpec<GRIM::FeedForwardParameterTensors>;

using EmbeddingTensorParameterSpec =
    TensorParameterSpec<GRIM::EmbeddingParameterTensors>;

inline constexpr std::array<EmbeddingTensorParameterSpec, 1>
    kEmbeddingTensorParameters = {{
        {"embedding", &GRIM::EmbeddingParameterTensors::token_weights,
         GRIM::ParamGroupType::EMBEDDING, GRIM::ParamStatsBucket::EMBEDDING},
    }};

inline constexpr std::array<NumberEncoderTensorParameterSpec, 10>
    kNumberEncoderTensorParameters = {{
        {"number_encoder_digit_emb", &GRIM::NumberEncoderParameterTensors::digit_emb,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::EMBEDDING},
        {"number_encoder_pow10_emb", &GRIM::NumberEncoderParameterTensors::pow10_emb,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::EMBEDDING},
        {"numeric_atom_Wz", &GRIM::NumberEncoderParameterTensors::numeric_atom_Wz,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_Uz", &GRIM::NumberEncoderParameterTensors::numeric_atom_Uz,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_Wr", &GRIM::NumberEncoderParameterTensors::numeric_atom_Wr,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_Ur", &GRIM::NumberEncoderParameterTensors::numeric_atom_Ur,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_Wh", &GRIM::NumberEncoderParameterTensors::numeric_atom_Wh,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_Uh", &GRIM::NumberEncoderParameterTensors::numeric_atom_Uh,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_sign_classifier", &GRIM::NumberEncoderParameterTensors::numeric_atom_sign_classifier,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"numeric_atom_stop_classifier", &GRIM::NumberEncoderParameterTensors::numeric_atom_stop_classifier,
         GRIM::ParamGroupType::NUMBER_ENCODER, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<SelectorTensorParameterSpec, 1>
    kSelectorTensorParameters = {{
        {"selector_W_q", &GRIM::SelectorParameterTensors::W_q,
         GRIM::ParamGroupType::ARG_SELECTOR, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<SlotSeedEncoderTensorParameterSpec, 5>
    kSlotSeedEncoderTensorParameters = {{
        {"slot_seed_encoder_W_seed_in",
         &GRIM::SlotSeedEncoderParameterTensors::W_seed_in,
         GRIM::ParamGroupType::SLOT_SEED_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"slot_seed_encoder_b_seed_in",
         &GRIM::SlotSeedEncoderParameterTensors::b_seed_in,
         GRIM::ParamGroupType::SLOT_SEED_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"slot_seed_encoder_W_seed_out",
         &GRIM::SlotSeedEncoderParameterTensors::W_seed_out,
         GRIM::ParamGroupType::SLOT_SEED_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"slot_seed_encoder_b_seed_out",
         &GRIM::SlotSeedEncoderParameterTensors::b_seed_out,
         GRIM::ParamGroupType::SLOT_SEED_ENCODER, GRIM::ParamStatsBucket::ENCODER},
        {"slot_seed_encoder_type_embeddings",
         &GRIM::SlotSeedEncoderParameterTensors::type_embeddings,
         GRIM::ParamGroupType::SLOT_SEED_ENCODER, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<ExecutionBlockTensorParameterSpec, 35>
    kExecutionBlockTensorParameters = {{
        {"exec_block_w_decode_1", &GRIM::ExecutionBlockParameterTensors::w_decode_1,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_b_decode_1", &GRIM::ExecutionBlockParameterTensors::b_decode_1,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_w_decode_2", &GRIM::ExecutionBlockParameterTensors::w_decode_2,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_w_arg1_select", &GRIM::ExecutionBlockParameterTensors::w_arg1_select,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_w_arg2_select", &GRIM::ExecutionBlockParameterTensors::w_arg2_select,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_arg1_to_arg2", &GRIM::ExecutionBlockParameterTensors::W_arg1_to_arg2,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_op_select", &GRIM::ExecutionBlockParameterTensors::W_op_select,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_key_proj", &GRIM::ExecutionBlockParameterTensors::W_key_proj,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_write_query", &GRIM::ExecutionBlockParameterTensors::W_write_query,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_write_key", &GRIM::ExecutionBlockParameterTensors::W_write_key,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_alpha", &GRIM::ExecutionBlockParameterTensors::alpha,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_beta", &GRIM::ExecutionBlockParameterTensors::beta,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_step_embeddings", &GRIM::ExecutionBlockParameterTensors::step_embeddings,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_type_num_embed", &GRIM::ExecutionBlockParameterTensors::type_num_embed,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_value_to_emb", &GRIM::ExecutionBlockParameterTensors::W_value_to_emb,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_b_value_to_emb", &GRIM::ExecutionBlockParameterTensors::b_value_to_emb,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_w_inject_gate", &GRIM::ExecutionBlockParameterTensors::w_inject_gate,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_Q_read", &GRIM::ExecutionBlockParameterTensors::W_Q_read,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_K_read", &GRIM::ExecutionBlockParameterTensors::W_K_read,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_V_read", &GRIM::ExecutionBlockParameterTensors::W_V_read,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_O_read", &GRIM::ExecutionBlockParameterTensors::W_O_read,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_gate_read", &GRIM::ExecutionBlockParameterTensors::W_gate_read,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_tau", &GRIM::ExecutionBlockParameterTensors::tau,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_E_slot", &GRIM::ExecutionBlockParameterTensors::E_slot,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_E_op", &GRIM::ExecutionBlockParameterTensors::E_op,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_scal", &GRIM::ExecutionBlockParameterTensors::W_scal,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_b_scal", &GRIM::ExecutionBlockParameterTensors::b_scal,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_trace", &GRIM::ExecutionBlockParameterTensors::W_trace,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_b_trace", &GRIM::ExecutionBlockParameterTensors::b_trace,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_reason_gate", &GRIM::ExecutionBlockParameterTensors::W_reason_gate,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_trace_gate", &GRIM::ExecutionBlockParameterTensors::W_trace_gate,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_execute", &GRIM::ExecutionBlockParameterTensors::W_execute,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_b_execute", &GRIM::ExecutionBlockParameterTensors::b_execute,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_W_stop", &GRIM::ExecutionBlockParameterTensors::W_stop,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
        {"exec_block_b_stop", &GRIM::ExecutionBlockParameterTensors::b_stop,
         GRIM::ParamGroupType::EXECUTION_BLOCK, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<EncodingLayerTensorParameterSpec, 8>
    kEncodingLayerTensorParameters = {{
        {"qkv_weight", &GRIM::EncodingLayerParameterTensors::W_qkv,
         GRIM::ParamGroupType::ATTENTION, GRIM::ParamStatsBucket::ENCODER},
        {"qkv_bias", &GRIM::EncodingLayerParameterTensors::b_qkv,
         GRIM::ParamGroupType::ATTENTION, GRIM::ParamStatsBucket::ENCODER},
        {"wo_weight", &GRIM::EncodingLayerParameterTensors::W_o,
         GRIM::ParamGroupType::ATTENTION, GRIM::ParamStatsBucket::ENCODER},
        {"wo_bias", &GRIM::EncodingLayerParameterTensors::b_o,
         GRIM::ParamGroupType::ATTENTION, GRIM::ParamStatsBucket::ENCODER},
        {"rms1_gamma", &GRIM::EncodingLayerParameterTensors::rms1_gamma,
         GRIM::ParamGroupType::RMSNORM, GRIM::ParamStatsBucket::ENCODER},
        {"rms2_gamma", &GRIM::EncodingLayerParameterTensors::rms2_gamma,
         GRIM::ParamGroupType::RMSNORM, GRIM::ParamStatsBucket::ENCODER},
        {"layer_scale1", &GRIM::EncodingLayerParameterTensors::layer_scale1,
         GRIM::ParamGroupType::RMSNORM, GRIM::ParamStatsBucket::ENCODER},
        {"layer_scale2", &GRIM::EncodingLayerParameterTensors::layer_scale2,
         GRIM::ParamGroupType::RMSNORM, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<AttentionResidualGateTensorParameterSpec, 2>
    kAttentionResidualGateTensorParameters = {{
        {"attention_residual_gate_weight", &GRIM::AttentionResidualGateParameterTensors::W_gate,
         GRIM::ParamGroupType::ATTENTION, GRIM::ParamStatsBucket::ENCODER},
        {"attention_residual_gate_bias", &GRIM::AttentionResidualGateParameterTensors::b_gate,
         GRIM::ParamGroupType::ATTENTION, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<FeedForwardTensorParameterSpec, 4>
    kFeedForwardTensorParameters = {{
        {"ffn_w_gate", &GRIM::FeedForwardParameterTensors::W_gate,
         GRIM::ParamGroupType::FFN, GRIM::ParamStatsBucket::ENCODER},
        {"ffn_w1", &GRIM::FeedForwardParameterTensors::W1,
         GRIM::ParamGroupType::FFN, GRIM::ParamStatsBucket::ENCODER},
        {"ffn_w2", &GRIM::FeedForwardParameterTensors::W2,
         GRIM::ParamGroupType::FFN, GRIM::ParamStatsBucket::ENCODER},
        {"ffn_b2", &GRIM::FeedForwardParameterTensors::b2,
         GRIM::ParamGroupType::FFN, GRIM::ParamStatsBucket::ENCODER},
    }};

template <typename RegistrarT>
inline void registerEmbeddingParameters(
    GRIM::EmbeddingParameterTensors& embedding_parameters,
    RegistrarT& registrar) {
    for (const auto& spec : kEmbeddingTensorParameters) {
        registrar.addTensor(spec.name,
                            embedding_parameters.*(spec.tensor_member),
                            spec.type,
                            spec.stats_bucket,
                            spec.layer);
    }
}

template <typename RegistrarT>
inline void registerExecutionBlockParameters(
    GRIM::ExecutionBlockParameterTensors& execution_block_parameters,
    const GRIM::HyperParameters::ExecutionBlockConstructionHP& hp,
    RegistrarT& registrar) {
    for (const auto& spec : kExecutionBlockTensorParameters) {
        bool enabled = true;
        if (spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_decode_1) {
            enabled = hp.decode_bias_enabled;
        } else if (spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_value_to_emb) {
            enabled = hp.value_embedding_bias_enabled;
        } else if (spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_scal) {
            enabled = hp.scalar_bias_enabled;
        } else if (spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_trace) {
            enabled = hp.trace_bias_enabled;
        }
        if (!enabled || spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_decode_1 ||
            spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_value_to_emb ||
            spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_scal ||
            spec.tensor_member == &GRIM::ExecutionBlockParameterTensors::b_trace) {
            registrar.addConfigGatedTensor(spec.name,
                                           execution_block_parameters.*(spec.tensor_member),
                                           spec.type,
                                           spec.stats_bucket,
                                           spec.layer,
                                           enabled,
                                           "corresponding execution-block bias gate is false");
            continue;
        }
        registrar.addTensor(spec.name,
                            execution_block_parameters.*(spec.tensor_member),
                            spec.type,
                            spec.stats_bucket,
                            spec.layer);
    }
}

template <typename RegistrarT>
inline void registerNumberEncoderParameters(
    GRIM::NumberEncoderParameterTensors& number_encoder_parameters,
    RegistrarT& registrar) {
    for (const auto& spec : kNumberEncoderTensorParameters) {
        registrar.addTensor(spec.name,
                            number_encoder_parameters.*(spec.tensor_member),
                            spec.type,
                            spec.stats_bucket,
                            spec.layer);
    }
}

template <typename RegistrarT>
inline void registerSelectorParameters(
    GRIM::SelectorParameterTensors& selector_parameters,
    RegistrarT& registrar) {
    for (const auto& spec : kSelectorTensorParameters) {
        registrar.addTensor(spec.name,
                            selector_parameters.*(spec.tensor_member),
                            spec.type,
                            spec.stats_bucket,
                            spec.layer);
    }
}

template <typename RegistrarT>
inline void registerSlotSeedEncoderParameters(
    GRIM::SlotSeedEncoderParameterTensors& slot_seed_encoder_parameters,
    const GRIM::HyperParameters::SlotSeedEncoderConstructionHP& hp,
    RegistrarT& registrar) {
    for (const auto& spec : kSlotSeedEncoderTensorParameters) {
        const bool is_bias =
            spec.tensor_member == &GRIM::SlotSeedEncoderParameterTensors::b_seed_in ||
            spec.tensor_member == &GRIM::SlotSeedEncoderParameterTensors::b_seed_out;
        const bool is_type_embedding =
            spec.tensor_member == &GRIM::SlotSeedEncoderParameterTensors::type_embeddings;
        if (is_bias || is_type_embedding) {
            const bool enabled = is_bias ? hp.bias_enabled : hp.type_embedding_enabled;
            registrar.addConfigGatedTensor(
                spec.name,
                slot_seed_encoder_parameters.*(spec.tensor_member),
                spec.type,
                spec.stats_bucket,
                spec.layer,
                enabled,
                is_bias
                    ? "slot_seed_encoder_bias_enabled=false"
                    : "slot_seed_encoder_type_embedding_enabled=false");
            continue;
        }
        registrar.addTensor(
            spec.name,
            slot_seed_encoder_parameters.*(spec.tensor_member),
            spec.type,
            spec.stats_bucket,
            spec.layer);
    }
}

template <typename RegistrarT>
inline void registerEncodingLayerParameters(
    GRIM::EncodingLayerParameterTensors& encoding_parameters,
    int layer_index,
    bool qkv_bias_enabled,
    bool output_bias_enabled,
    bool freeze_learned_rms_gammas,
    bool use_layer_scale,
    bool attention_residual_gate_enabled,
    RegistrarT& registrar) {
    if (layer_index < 0) {
        throw std::runtime_error("registerEncodingLayerParameters: layer_index must be non-negative");
    }

    const std::string prefix = "layer" + std::to_string(layer_index) + "_";
    for (const auto& spec : kEncodingLayerTensorParameters) {
        if (spec.tensor_member == &GRIM::EncodingLayerParameterTensors::b_qkv ||
            spec.tensor_member == &GRIM::EncodingLayerParameterTensors::b_o) {
            const bool enabled = spec.tensor_member == &GRIM::EncodingLayerParameterTensors::b_qkv
                ? qkv_bias_enabled
                : output_bias_enabled;
            registrar.addConfigGatedTensor(prefix + spec.name,
                                           encoding_parameters.*(spec.tensor_member),
                                           spec.type,
                                           spec.stats_bucket,
                                           layer_index,
                                           enabled,
                                           "corresponding attention bias gate is false");
            continue;
        }

        if (spec.tensor_member == &GRIM::EncodingLayerParameterTensors::layer_scale1 ||
            spec.tensor_member == &GRIM::EncodingLayerParameterTensors::layer_scale2) {
            registrar.addConfigGatedTensor(prefix + spec.name,
                                           encoding_parameters.*(spec.tensor_member),
                                           spec.type,
                                           spec.stats_bucket,
                                           layer_index,
                                           use_layer_scale,
                                           "config.use_layer_scale=false");
            continue;
        }

        if (spec.tensor_member == &GRIM::EncodingLayerParameterTensors::rms1_gamma ||
            spec.tensor_member == &GRIM::EncodingLayerParameterTensors::rms2_gamma) {
            if (freeze_learned_rms_gammas) {
                continue;
            }
        }

        registrar.addTensor(prefix + spec.name,
                            encoding_parameters.*(spec.tensor_member),
                            spec.type,
                            spec.stats_bucket,
                            layer_index);
    }

    for (const auto& spec : kAttentionResidualGateTensorParameters) {
        registrar.addConfigGatedTensor(
            prefix + spec.name,
            encoding_parameters.attention_residual_gate.*(spec.tensor_member),
            spec.type,
            spec.stats_bucket,
            layer_index,
            attention_residual_gate_enabled,
            "config.attention_residual_gate_enabled=false");
    }
}

template <typename RegistrarT>
inline void registerFeedForwardParameters(
    GRIM::FeedForwardParameterTensors& feed_forward_parameters,
    int layer_index,
    bool output_bias_enabled,
    RegistrarT& registrar) {
    if (layer_index < 0) {
        throw std::runtime_error("registerFeedForwardParameters: layer_index must be non-negative");
    }

    const std::string prefix = "layer" + std::to_string(layer_index) + "_";
    for (const auto& spec : kFeedForwardTensorParameters) {
        if (spec.tensor_member == &GRIM::FeedForwardParameterTensors::b2) {
            registrar.addConfigGatedTensor(prefix + spec.name,
                                           feed_forward_parameters.*(spec.tensor_member),
                                           spec.type,
                                           spec.stats_bucket,
                                           layer_index,
                                           output_bias_enabled,
                                           "config.ffn_output_bias_enabled=false");
            continue;
        }

        registrar.addTensor(prefix + spec.name,
                            feed_forward_parameters.*(spec.tensor_member),
                            spec.type,
                            spec.stats_bucket,
                            layer_index);
    }
}

} // namespace ParameterRegistry
