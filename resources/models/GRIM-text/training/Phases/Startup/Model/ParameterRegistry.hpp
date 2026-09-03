#pragma once

//======================================================//
//  ParameterRegistry — adjacent startup parameter-owner
//  declarations plus header-only migrated tensor-group
//  inventory slices.
//
//  Current migrated surface:
//    - Embedding durable tensor owner
//    - LM head durable tensor owner
//    - Optional atom insertion boundary-projection tensor owner
//    - Encoder durable per-layer parameter tensor owner
//    - LocalAtomRetrieval durable tensor owner
//    - LocalAtomRetrieval parameter-group inventory
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

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
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

// Atom-insertion boundary projection durable tensor owner. The registry owns
// this optional bundle; atom-insertion compute paths only borrow it.
struct AtomInsertionBoundaryParameterTensors {
    Tensor left_projection_weight;   // [d_model, d_model]
    Tensor right_projection_weight;  // [d_model, d_model]
    Tensor projection_bias;          // [1, d_model]
};

// Arg/option selector head parameters (execution-INDEPENDENT). A single query
// projection W_q maps encoder hidden states into an externally owned
// candidate-key space.
struct SelectorParameterTensors {
    Tensor W_q;   // [d_model, d_model] query projection
};

// Sequence-local atom retrieval parameters. The retrieval compute path borrows
// this tensor through StartupParameterRegistry and never owns model weights.
struct LocalAtomRetrievalParameterTensors {
    Tensor type_no_reference_key;  // [kAtomTypeCount, d_model]
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
    std::unique_ptr<GRIM::AtomInsertionBoundaryParameterTensors>
        atom_insertion_boundary_parameters;
    std::vector<GRIM::EncodingLayerParameterTensors> encoding_layer_parameter_tensors;
    std::unique_ptr<GRIM::SelectorParameterTensors> selector_parameters;
    std::unique_ptr<GRIM::LocalAtomRetrievalParameterTensors>
        local_atom_retrieval_parameters;
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

    GRIM::AtomInsertionBoundaryParameterTensors*
    getAtomInsertionBoundaryParameters() {
        return atom_insertion_boundary_parameters.get();
    }

    const GRIM::AtomInsertionBoundaryParameterTensors*
    getAtomInsertionBoundaryParameters() const {
        return atom_insertion_boundary_parameters.get();
    }

    GRIM::AtomInsertionBoundaryParameterTensors&
    requireAtomInsertionBoundaryParameters(const char* caller) {
        if (!atom_insertion_boundary_parameters) {
            throw std::runtime_error(
                std::string(caller) +
                ": StartupParameterRegistry.atom_insertion_boundary_parameters is NULL");
        }
        return *atom_insertion_boundary_parameters;
    }

    const GRIM::AtomInsertionBoundaryParameterTensors&
    requireAtomInsertionBoundaryParameters(const char* caller) const {
        if (!atom_insertion_boundary_parameters) {
            throw std::runtime_error(
                std::string(caller) +
                ": StartupParameterRegistry.atom_insertion_boundary_parameters is NULL");
        }
        return *atom_insertion_boundary_parameters;
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

    GRIM::LocalAtomRetrievalParameterTensors*
    getLocalAtomRetrievalParameters() {
        return local_atom_retrieval_parameters.get();
    }

    const GRIM::LocalAtomRetrievalParameterTensors*
    getLocalAtomRetrievalParameters() const {
        return local_atom_retrieval_parameters.get();
    }

    GRIM::LocalAtomRetrievalParameterTensors&
    requireLocalAtomRetrievalParameters(const char* caller) {
        if (!local_atom_retrieval_parameters) {
            throw std::runtime_error(
                std::string(caller) +
                ": StartupParameterRegistry.local_atom_retrieval_parameters is NULL");
        }
        return *local_atom_retrieval_parameters;
    }

    const GRIM::LocalAtomRetrievalParameterTensors&
    requireLocalAtomRetrievalParameters(const char* caller) const {
        if (!local_atom_retrieval_parameters) {
            throw std::runtime_error(
                std::string(caller) +
                ": StartupParameterRegistry.local_atom_retrieval_parameters is NULL");
        }
        return *local_atom_retrieval_parameters;
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

using SelectorTensorParameterSpec =
    TensorParameterSpec<GRIM::SelectorParameterTensors>;

using EncodingLayerTensorParameterSpec =
    TensorParameterSpec<GRIM::EncodingLayerParameterTensors>;

using AttentionResidualGateTensorParameterSpec =
    TensorParameterSpec<GRIM::AttentionResidualGateParameterTensors>;

using FeedForwardTensorParameterSpec =
    TensorParameterSpec<GRIM::FeedForwardParameterTensors>;

using EmbeddingTensorParameterSpec =
    TensorParameterSpec<GRIM::EmbeddingParameterTensors>;

using AtomInsertionBoundaryTensorParameterSpec =
    TensorParameterSpec<GRIM::AtomInsertionBoundaryParameterTensors>;

using LocalAtomRetrievalTensorParameterSpec =
    TensorParameterSpec<GRIM::LocalAtomRetrievalParameterTensors>;

inline constexpr std::array<EmbeddingTensorParameterSpec, 1>
    kEmbeddingTensorParameters = {{
        {"embedding", &GRIM::EmbeddingParameterTensors::token_weights,
         GRIM::ParamGroupType::EMBEDDING, GRIM::ParamStatsBucket::EMBEDDING},
    }};

inline constexpr std::array<AtomInsertionBoundaryTensorParameterSpec, 3>
    kAtomInsertionBoundaryTensorParameters = {{
        {"atom_insertion_left_projection_weight",
         &GRIM::AtomInsertionBoundaryParameterTensors::left_projection_weight,
         GRIM::ParamGroupType::LM_HEAD, GRIM::ParamStatsBucket::LM_HEAD},
        {"atom_insertion_right_projection_weight",
         &GRIM::AtomInsertionBoundaryParameterTensors::right_projection_weight,
         GRIM::ParamGroupType::LM_HEAD, GRIM::ParamStatsBucket::LM_HEAD},
        {"atom_insertion_projection_bias",
         &GRIM::AtomInsertionBoundaryParameterTensors::projection_bias,
         GRIM::ParamGroupType::LM_HEAD, GRIM::ParamStatsBucket::LM_HEAD},
    }};

inline constexpr std::array<SelectorTensorParameterSpec, 1>
    kSelectorTensorParameters = {{
        {"selector_W_q", &GRIM::SelectorParameterTensors::W_q,
         GRIM::ParamGroupType::ARG_SELECTOR, GRIM::ParamStatsBucket::ENCODER},
    }};

inline constexpr std::array<LocalAtomRetrievalTensorParameterSpec, 1>
    kLocalAtomRetrievalTensorParameters = {{
        {"local_atom_retrieval_type_no_reference_key",
         &GRIM::LocalAtomRetrievalParameterTensors::type_no_reference_key,
         GRIM::ParamGroupType::ARG_SELECTOR, GRIM::ParamStatsBucket::ENCODER},
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
inline void registerAtomInsertionBoundaryParameters(
    GRIM::AtomInsertionBoundaryParameterTensors& parameters,
    RegistrarT& registrar) {
    for (const auto& spec : kAtomInsertionBoundaryTensorParameters) {
        registrar.addTensor(spec.name,
                            parameters.*(spec.tensor_member),
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
inline void registerLocalAtomRetrievalParameters(
    GRIM::LocalAtomRetrievalParameterTensors& parameters,
    RegistrarT& registrar) {
    for (const auto& spec : kLocalAtomRetrievalTensorParameters) {
        registrar.addTensor(spec.name,
                            parameters.*(spec.tensor_member),
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
