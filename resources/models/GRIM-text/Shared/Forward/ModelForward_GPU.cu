//======================================================//
//  ModelForward_GPU.cu
//  Shared full-model forward primitive
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelForward_GPU.hpp"
#include "ModelForwardArgBootstrap_GPU.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ArgSelector/ArgSelector_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../InferenceState/KvCacheState_GPU.hpp"
#include "../CudaAllocUtils.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../TensorContract/GradFns/NumberEncoderGradFn.hpp"
#include "ModelForwardOutputs.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../VerboseLogging.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIM {
namespace Forward {

#define MFWD_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[ModelForward] INFO: " << msg << std::endl; \
    } \
} while(0)

namespace {

const TensorContract::Shape2D& requireTensor2DShape(
    const Tensor& tensor,
    const char* caller,
    const char* label);

const char* graphPolicyName(const ModelForwardGraphPolicy& graph) {
    return graph.connect_parameter_graph ? "autograd_connected" : "read_only";
}

const TensorContract::Shape2D& requireTensor2DShape(
    const Tensor& tensor,
    const char* caller,
    const char* label) {
    tensor.require(label);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(caller) + ": " + label + " must be a 2D tensor");
    }
    return tensor.shape.as_2d();
}

void requireCenteringSequenceLengths(const Batching::BatchPayload& payload,
                                     const char* caller) {
    if (payload.batch_size <= 0 || payload.max_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": invalid payload geometry batch=" +
                                 std::to_string(payload.batch_size) + " seq=" +
                                 std::to_string(payload.max_seq_len));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(payload.batch_size) + ")");
    }
    for (int b = 0; b < payload.batch_size; ++b) {
        const int row_len = payload.seq_lengths[static_cast<size_t>(b)];
        if (row_len <= 1 || row_len > payload.max_seq_len) {
            throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                     std::to_string(b) + "]=" + std::to_string(row_len) +
                                     " for padding-aware centering over max_seq_len=" +
                                     std::to_string(payload.max_seq_len));
        }
    }
}

Tensor viewCommittedTensor(const Tensor& owned,
                           cudaStream_t stream,
                           const char* debug_name,
                           const char* caller) {
    if (!owned.data) {
        throw std::runtime_error(std::string(caller) + ": committed tensor data is NULL for " + debug_name);
    }
    Tensor view = Tensor::from_ptr(
        owned.data,
        owned.shape,
        false,
        owned.requires_grad,
        debug_name);
    view.is_leaf = false;
    view.grad_fn = owned.grad_fn;
    view.stream = stream;
    return view;
}

// Encode the candidate atom-entry pool once for token-level selector scoring.
// Candidate keys are independent of encoder hidden state, so preparing them
// before the encoder loop does not change selector causality.
void materializeForwardSelectorCandidateKeys(
    const ModelForwardRequest& request,
    const Batching::BatchPayload& payload,
    ModelForwardOutputs& forward_outputs) {
    forward_outputs.selector_candidate_keys = Tensor();
    if (!request.graph.emit_selector_logits) {
        return;
    }

    const int num_pool_atoms = request.bindings->num_pool_atoms;
    if (num_pool_atoms <= 0) {
        return;  // No candidate entries in this batch — nothing to select among.
    }

    const auto number_encoder_hp = HyperParameters::numberEncoderConstructionHP(*request.config);
    if (!number_encoder_hp.enabled) {
        throw std::runtime_error(
            "executeModelForward: selector candidate keys require the NumberEncoder "
            "to be enabled");
    }
    if (!request.bindings->d_pool_digit_values || !request.bindings->d_pool_digit_pow10_index ||
        !request.bindings->d_pool_digit_mask || !request.bindings->d_pool_digit_slot_features ||
        !request.bindings->d_pool_global_features || !request.bindings->d_row_atom_offset) {
        throw std::runtime_error(
            "executeModelForward: selector candidate keys requested but candidate-pool "
            "device bindings are NULL");
    }

    auto& ne = request.parameter_registry->requireNumberEncoderParameters("executeModelForward(selector)");
    // Connected (training): encode keys against the registered NumberEncoder
    // leaves so the selection loss accumulates gradient into them — the selector
    // teaches the encoder which candidate entries to keep distinguishable.
    // Read-only (inference): detached copies so the keys are forward-only and no
    // graph edges are retained (mirrors the W_q detach below).
    GRIM::NumberEncoderParameterTensors ne_detached{};
    const GRIM::NumberEncoderParameterTensors* ne_src = &ne;
    if (!request.graph.connect_parameter_graph) {
        ne_detached.digit_emb = ne.digit_emb.detach(request.stream);
        ne_detached.pow10_emb = ne.pow10_emb.detach(request.stream);
        ne_detached.W_c1 = ne.W_c1.detach(request.stream);
        ne_detached.b_c1 = ne.b_c1.detach(request.stream);
        ne_detached.W_c2 = ne.W_c2.detach(request.stream);
        ne_detached.W_g1 = ne.W_g1.detach(request.stream);
        ne_detached.b_g1 = ne.b_g1.detach(request.stream);
        ne_detached.W_g2 = ne.W_g2.detach(request.stream);
        ne_src = &ne_detached;
    }
    autograd::NumberEncoderForwardParams ne_params{};
    ne_params.digit_emb = &ne_src->digit_emb;
    ne_params.pow10_emb = &ne_src->pow10_emb;
    ne_params.W_c1 = &ne_src->W_c1;
    ne_params.b_c1 = &ne_src->b_c1;
    ne_params.W_c2 = &ne_src->W_c2;
    ne_params.W_g1 = &ne_src->W_g1;
    ne_params.b_g1 = &ne_src->b_g1;
    ne_params.W_g2 = &ne_src->W_g2;

    forward_outputs.selector_candidate_keys = autograd::encodeAtomEntryPoolKeys(
        ne_params, number_encoder_hp,
        request.bindings->d_pool_digit_values,
        request.bindings->d_pool_digit_pow10_index,
        request.bindings->d_pool_digit_mask,
        request.bindings->d_pool_digit_slot_features,
        request.bindings->d_pool_global_features,
        num_pool_atoms, request.stream);
}

// Arg/option selector head: score the live LM-head hidden state against the
// already-materialized candidate pool. Emits selector_logits
// [total_tokens, num_pool_atoms].
void materializeForwardSelectorLogits(
    const ModelForwardRequest& request,
    const Batching::BatchPayload& payload,
    ModelForwardOutputs& forward_outputs) {
    forward_outputs.selector_logits = Tensor();
    if (!request.graph.emit_selector_logits) {
        return;
    }

    const int num_pool_atoms = request.bindings->num_pool_atoms;
    if (num_pool_atoms <= 0) {
        return;
    }
    if (!forward_outputs.selector_candidate_keys.data) {
        throw std::runtime_error(
            "executeModelForward: selector logits requested before candidate keys "
            "were materialized");
    }

    Tensor* sel_input = forward_outputs.liveLmHeadInputOrNull();
    if (!sel_input || !sel_input->data) {
        throw std::runtime_error(
            "executeModelForward: live LM-head input is NULL before selector materialization");
    }

    auto& sel = request.parameter_registry->requireSelectorParameters("executeModelForward(selector)");
    // Connected (training): score against the registered W_q leaf so gradient
    // accumulates into the optimizer-visible buffer. Read-only (inference): a
    // detached copy so no graph edges are retained.
    Tensor W_q_detached;
    const Tensor* W_q_ptr = &sel.W_q;
    if (!request.graph.connect_parameter_graph) {
        W_q_detached = sel.W_q.detach(request.stream);
        W_q_ptr = &W_q_detached;
    }

    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*request.config, "d_model");
    const float selector_scale = 1.0f / std::sqrt(static_cast<float>(d_model));

    forward_outputs.selector_logits = ArgSelector::argSelectorForward(
        *sel_input, *W_q_ptr, forward_outputs.selector_candidate_keys, payload,
        request.bindings->d_row_atom_offset, num_pool_atoms, selector_scale, request.stream);
}

GRIM::FeedForwardParameterTensors detachFeedForwardParameters(
    const GRIM::FeedForwardParameterTensors& parameters,
    bool output_bias_enabled,
    cudaStream_t stream) {
    GRIM::FeedForwardParameterTensors detached{};
    detached.W_gate = parameters.W_gate.detach(stream);
    detached.W1 = parameters.W1.detach(stream);
    detached.W2 = parameters.W2.detach(stream);
    if (output_bias_enabled) {
        detached.b2 = parameters.b2.detach(stream);
    }
    return detached;
}

GRIM::EncodingLayerParameterTensors detachEncodingLayerParameters(
    const GRIM::EncodingLayerParameterTensors& parameters,
    bool qkv_bias_enabled,
    bool output_bias_enabled,
    bool use_layer_scale,
    bool attention_residual_gate_enabled,
    cudaStream_t stream) {
    GRIM::EncodingLayerParameterTensors detached{};
    detached.rms1_gamma = parameters.rms1_gamma.detach(stream);
    detached.rms2_gamma = parameters.rms2_gamma.detach(stream);
    detached.W_qkv = parameters.W_qkv.detach(stream);
    detached.W_o = parameters.W_o.detach(stream);
    if (qkv_bias_enabled) {
        detached.b_qkv = parameters.b_qkv.detach(stream);
    }
    if (output_bias_enabled) {
        detached.b_o = parameters.b_o.detach(stream);
    }
    if (use_layer_scale) {
        detached.layer_scale1 = parameters.layer_scale1.detach(stream);
        detached.layer_scale2 = parameters.layer_scale2.detach(stream);
    }
    if (attention_residual_gate_enabled) {
        detached.attention_residual_gate.W_gate =
            parameters.attention_residual_gate.W_gate.detach(stream);
        detached.attention_residual_gate.b_gate =
            parameters.attention_residual_gate.b_gate.detach(stream);
    }
    return detached;
}

GRIM::LMHeadParameterTensors detachLmHeadParameters(
    const GRIM::LMHeadParameterTensors& parameters,
    cudaStream_t stream) {
    GRIM::LMHeadParameterTensors detached{};
    detached.owns_weights = false;
    detached.weights = parameters.weights.detach(stream);
    if (parameters.bias.data) {
        detached.bias = parameters.bias.detach(stream);
    }
    if (parameters.final_rms_gamma.data) {
        detached.final_rms_gamma = parameters.final_rms_gamma.detach(stream);
    }
    if (parameters.mlp_W_gate.data) {
        detached.mlp_W_gate = parameters.mlp_W_gate.detach(stream);
    }
    if (parameters.mlp_W_up.data) {
        detached.mlp_W_up = parameters.mlp_W_up.detach(stream);
    }
    if (parameters.mlp_W_down.data) {
        detached.mlp_W_down = parameters.mlp_W_down.detach(stream);
    }
    return detached;
}

}  // namespace

void ModelForwardRequest::validate(const char* caller) const {
    if (!config) throw std::runtime_error(std::string(caller) + ": config is NULL");
    if (!gpu_encoder) throw std::runtime_error(std::string(caller) + ": gpu_encoder is NULL");
    if (!parameter_registry) throw std::runtime_error(std::string(caller) + ": parameter_registry is NULL");
    (void)parameter_registry->requireEmbeddingParameters(caller);
    (void)parameter_registry->requireLmHeadParameters(caller);
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*config, "num_layers");
    if (static_cast<int>(parameter_registry->encodingLayerParameterTensors().size()) != num_layers) {
        throw std::runtime_error(std::string(caller) + ": parameter_registry encoder tensor count mismatch. size=" +
                                 std::to_string(parameter_registry->encodingLayerParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }
    if (static_cast<int>(parameter_registry->feedForwardParameterTensors().size()) != num_layers) {
        throw std::runtime_error(std::string(caller) + ": parameter_registry FFN tensor count mismatch. size=" +
                                 std::to_string(parameter_registry->feedForwardParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }
    if (!this->pbm) throw std::runtime_error(std::string(caller) + ": pbm is NULL");
    if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
    if (!stream) throw std::runtime_error(std::string(caller) + ": stream is NULL");
    if (!payload) throw std::runtime_error(std::string(caller) + ": payload is NULL");
    if (!bindings) throw std::runtime_error(std::string(caller) + ": bindings is NULL");
    (void)graphPolicyName(graph);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*config);
    if (execution_hp.enabled) {
        if (!execution_block_enabled) {
            throw std::runtime_error(std::string(caller) + ": execution_block_enabled=false while argument bootstrap is configured");
        }
    } else if (execution_block_enabled) {
        throw std::runtime_error(std::string(caller) + ": execution_block_enabled=true while argument bootstrap is not configured");
    }
    if (payload->batch_size <= 0) throw std::runtime_error(std::string(caller) + ": BatchPayload.batch_size <= 0");
    if (payload->max_seq_len <= 0) throw std::runtime_error(std::string(caller) + ": BatchPayload.max_seq_len <= 0");
    if (static_cast<int>(payload->seq_lengths.size()) != payload->batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload->seq_lengths.size()) +
                                 ") != payload.batch_size (" + std::to_string(payload->batch_size) + ")");
    }
    for (int b = 0; b < payload->batch_size; ++b) {
        const int row_len = payload->seq_lengths[static_cast<size_t>(b)];
        if (row_len <= 0 || row_len > payload->max_seq_len) {
            throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                     std::to_string(b) + "]=" + std::to_string(row_len) +
                                     " for payload.max_seq_len=" + std::to_string(payload->max_seq_len));
        }
    }
    if (graph.emit_selector_logits) {
        const bool selector_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*config, "selector_enabled");
        if (!selector_enabled) {
            throw std::runtime_error(std::string(caller) + ": graph.emit_selector_logits=true while config.selector_enabled=false");
        }
        const bool number_encoder_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*config, "number_encoder_enabled");
        if (!number_encoder_enabled) {
            throw std::runtime_error(std::string(caller) + ": graph.emit_selector_logits=true requires number_encoder_enabled=true (selector keys are NumberEncoder-derived)");
        }
        (void)parameter_registry->requireSelectorParameters(caller);
    }
    if (kv_cache) {
        if (graph.connect_parameter_graph) {
            throw std::runtime_error(std::string(caller) + ": kv_cache requires connect_parameter_graph == false");
        }
        if (graph.enable_dropout) {
            throw std::runtime_error(std::string(caller) + ": kv_cache decode is read-only and cannot run with dropout");
        }
        if (payload->batch_size != 1) {
            throw std::runtime_error(std::string(caller) + ": kv_cache decode requires payload.batch_size == 1");
        }
        if (!kv_cache->allocated) {
            throw std::runtime_error(std::string(caller) + ": kv_cache is not allocated (call ensureAllocated/beginSession before the forward)");
        }
        const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*config, "num_layers");
        if (kv_cache->num_layers != num_layers) {
            throw std::runtime_error(std::string(caller) + ": kv_cache.num_layers=" +
                                     std::to_string(kv_cache->num_layers) + " != config.num_layers=" +
                                     std::to_string(num_layers));
        }
        if (kv_cache->host_seqlen + payload->total_tokens > kv_cache->cache_max_seq) {
            throw std::runtime_error(std::string(caller) + ": kv_cache overflow: current fill=" +
                                     std::to_string(kv_cache->host_seqlen) + " + q_len=" +
                                     std::to_string(payload->total_tokens) + " > cache_max_seq=" +
                                     std::to_string(kv_cache->cache_max_seq));
        }
    }
}

ModelForwardOutputs executeModelForward(const ModelForwardRequest& request,
                                        ModelForwardRuntimePayload& runtime_payload) {
    request.validate("executeModelForward");
    const auto* cfg = request.config;
    const auto embedding_hp = HyperParameters::embeddingLayerConstructionHP(*cfg);
    const auto encoder_hp = HyperParameters::encoderLayerConstructionHP(*cfg);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*cfg);
    const auto slot_seed_encoder_hp =
        HyperParameters::slotSeedEncoderConstructionHP(*cfg);
    const auto lm_head_hp = HyperParameters::lmHeadLayerConstructionHP(*cfg);
    const bool center_encoder_residuals = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "center_encoder_residuals");
    const bool lm_head_center_hidden_states = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "lm_head_center_hidden_states");
    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "d_model");
    const auto positional_encoding = HyperParameters::snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(*cfg, "positional_encoding");
    const float dropout_rate = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "dropout_rate");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "num_layers");
    const bool use_layer_scale = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "use_layer_scale");
    const int execution_block_layer = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_layer");
    const bool arg_bootstrap_active =
        execution_hp.enabled &&
        request.execution_block_enabled &&
        slot_seed_encoder_hp.enabled;

    const auto& payload = *request.payload;
    (void)runtime_payload;
    ModelForwardOutputs forward_outputs;
    const auto* bindings = request.bindings;
    const auto& embedding_parameters = request.parameter_registry->requireEmbeddingParameters("executeModelForward");
    const auto& lm_head_parameters = request.parameter_registry->requireLmHeadParameters("executeModelForward");
    const bool connect_parameter_graph = request.graph.connect_parameter_graph;
    const bool dropout_enabled = request.graph.enable_dropout;

    if (center_encoder_residuals || lm_head_center_hidden_states) {
        requireCenteringSequenceLengths(payload, "ModelForward");
    }

    const int total_tokens = payload.total_tokens;

    MFWD_INFO("forward: batch=" << payload.batch_size << " seq=" << payload.max_seq_len
              << " tokens=" << total_tokens << " vocab=" << payload.vocab_size
              << " graph=" << graphPolicyName(request.graph));

    autograd::set_autograd_cublas_handle(request.cublas_handle);

    int* token_ids = bindings->d_input_ids;
    if (!token_ids) {
        throw std::runtime_error("ModelForward: input token device pointer is NULL");
    }

    Tensor emb_weights_view;
    const Tensor* emb_weights = &embedding_parameters.token_weights;
    if (!connect_parameter_graph) {
        emb_weights_view = embedding_parameters.token_weights.detach(request.stream);
        emb_weights = &emb_weights_view;
    }
    if (!emb_weights->data) {
        throw std::runtime_error("ModelForward: embedding token_weights.data is NULL");
    }

    if (!emb_weights->shape.is_valid()) {
        throw std::runtime_error("ModelForward: embedding token_weights.shape is INVALID - EmbeddingLayer MUST initialize with correct shape [vocab_size="
                                + std::to_string(payload.vocab_size) + ", d_model=" + std::to_string(d_model) + "]");
    }

    const float embedding_scale = embedding_hp.embedding_scale;
    Tensor emb_output = autograd::embedding(
        *emb_weights,
        payload,
        *bindings,
        request.stream,
        embedding_scale);

    MFWD_INFO("Step 1b: No position embeddings (using "
              << HyperParameters::positionalEncodingTypeToString(positional_encoding)
              << " inside attention)");

    forward_outputs.embedding_tensor = std::move(emb_output);
    MFWD_INFO("Step 1: Token embedding complete, shape=[" << total_tokens << ", " << d_model
              << "] scale=" << embedding_scale);

    // ─── Step 1n: NumberEncoder numeric-meaning fusion ──────────────────────
    // x_t = token_embedding[<INT>/<FLOAT>] + number_embedding(arg_number).
    // Selection-side input path (docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md):
    // digit-place contribution slots are pooled per numeric atom and added at
    // that atom's token position; all non-atom rows receive exact zero. The
    // channels are CURRENT-token metadata only — next-token atom metadata is
    // supervision and never enters this input boundary.
    const auto number_encoder_hp = HyperParameters::numberEncoderConstructionHP(*cfg);
    std::vector<Tensor> number_encoder_detached_params;  // keep-alive across the call window
    Tensor number_encoder_out;                           // keep-alive across the call window
    if (number_encoder_hp.enabled) {
        if (payload.number_encoder_digit_slots != number_encoder_hp.max_digit_slots) {
            throw std::runtime_error(
                "ModelForward: payload.number_encoder_digit_slots=" +
                std::to_string(payload.number_encoder_digit_slots) +
                " != config max_digit_slots=" +
                std::to_string(number_encoder_hp.max_digit_slots) +
                " — payload was built against a different NumberEncoder config");
        }
        if (payload.authoredAtomCount() > 0) {
            auto& number_encoder_parameters =
                request.parameter_registry->requireNumberEncoderParameters("executeModelForward");
            autograd::NumberEncoderForwardParams ne_params{};
            if (connect_parameter_graph) {
                ne_params.digit_emb = &number_encoder_parameters.digit_emb;
                ne_params.pow10_emb = &number_encoder_parameters.pow10_emb;
                ne_params.W_c1 = &number_encoder_parameters.W_c1;
                ne_params.b_c1 = &number_encoder_parameters.b_c1;
                ne_params.W_c2 = &number_encoder_parameters.W_c2;
                ne_params.W_g1 = &number_encoder_parameters.W_g1;
                ne_params.b_g1 = &number_encoder_parameters.b_g1;
                ne_params.W_g2 = &number_encoder_parameters.W_g2;
            } else {
                number_encoder_detached_params.reserve(8);
                number_encoder_detached_params.push_back(number_encoder_parameters.digit_emb.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.pow10_emb.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_c1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.b_c1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_c2.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_g1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.b_g1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_g2.detach(request.stream));
                ne_params.digit_emb = &number_encoder_detached_params[0];
                ne_params.pow10_emb = &number_encoder_detached_params[1];
                ne_params.W_c1 = &number_encoder_detached_params[2];
                ne_params.b_c1 = &number_encoder_detached_params[3];
                ne_params.W_c2 = &number_encoder_detached_params[4];
                ne_params.W_g1 = &number_encoder_detached_params[5];
                ne_params.b_g1 = &number_encoder_detached_params[6];
                ne_params.W_g2 = &number_encoder_detached_params[7];
            }
            number_encoder_out = autograd::number_encode(
                ne_params, number_encoder_hp, payload, *bindings, request.stream);
            forward_outputs.embedding_tensor = autograd::residual_add(
                forward_outputs.embedding_tensor, number_encoder_out, request.stream);
            MFWD_INFO("Step 1n: NumberEncoder fused into " << payload.authoredAtomCount()
                      << " numeric atom positions (digit_slots=" << number_encoder_hp.max_digit_slots
                      << ", d_hidden=" << number_encoder_hp.d_hidden << ")");
        }
    }

    if (arg_bootstrap_active && payload.execution_slot_count > 0) {
        if (payload.execution_slot_count != execution_hp.num_slots) {
            throw std::runtime_error(
                "ModelForward: payload.execution_slot_count=" +
                std::to_string(payload.execution_slot_count) +
                " does not match argument-bootstrap num_slots=" +
                std::to_string(execution_hp.num_slots));
        }
        if (!bindings->d_token_to_slot_index_map ||
            bindings->execution_slot_count != execution_hp.num_slots) {
            throw std::runtime_error(
                "ModelForward: argument-bootstrap slot bindings are missing or "
                "have incompatible geometry");
        }
    }
    materializeForwardSelectorCandidateKeys(
        request,
        payload,
        forward_outputs);

    if (dropout_enabled && dropout_rate > 0.0f) {
        const uint64_t emb_dropout_seed = request.batch_idx * 2654435761ULL + 500;
        constexpr uint64_t kEmbeddingDropoutMaskStream = 0x0005000000000001ULL;
        forward_outputs.embedding_tensor = autograd::dropout(
            forward_outputs.embedding_tensor,
            dropout_rate,
            emb_dropout_seed,
            request.stream,
            kEmbeddingDropoutMaskStream);
        MFWD_INFO("Step 1c: Embedding-fusion dropout applied"
                  << " (p=" << dropout_rate << ", batch_idx=" << request.batch_idx << ")");
    }

    if (!request.gpu_encoder) {
        throw std::runtime_error("ModelForward: gpu_encoder is NULL - pass encoder in request");
    }

    forward_outputs.encoder_layer_outputs.clear();
    forward_outputs.clearRetainedLayerOutputs();
    forward_outputs.embedding_tensor.is_leaf = false;

    int bootstrap_layer = -1;
    if (arg_bootstrap_active) {
        bootstrap_layer = execution_block_layer;
        if (bootstrap_layer < 0) bootstrap_layer = num_layers - 2;
        if (bootstrap_layer < 0) bootstrap_layer = 0;
        if (bootstrap_layer >= num_layers) bootstrap_layer = num_layers - 1;
    }

    if (!connect_parameter_graph) {
        Tensor running;
        forward_outputs.reserveLayerOutputs(num_layers);
        MFWD_INFO("Step 2: Running " << num_layers << " encoder layers (no_grad)...");

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }
            if (layer_idx > 0) {
                cudaError_t sync_err = cudaStreamSynchronize(request.stream);
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error("ModelForward(no_grad): cudaStreamSynchronize failed after layer " +
                        std::to_string(layer_idx - 1) + ": " + cudaGetErrorString(sync_err));
                }
            }

            forward_outputs.pushLayerOutputs();

            Tensor* layer_input = (layer_idx == 0) ? &forward_outputs.embedding_tensor : &running;

            const auto& encoding_parameters = request.parameter_registry->requireEncodingLayerParameters(
                layer_idx,
                "executeModelForward(no_grad)");
            if (encoder_hp.attention_residual_gate_enabled) {
                request.parameter_registry->requireAttentionResidualGateParameters(
                    layer_idx, "executeModelForward(no_grad)");
            }
            const auto& ffn_parameters = request.parameter_registry->requireFeedForwardParameters(
                layer_idx,
                "executeModelForward(no_grad)");
            const GRIM::EncodingLayerParameterTensors* encoding_parameter_ptr = &encoding_parameters;
            const GRIM::FeedForwardParameterTensors* ffn_parameter_ptr = &ffn_parameters;
            GRIM::EncodingLayerParameterTensors detached_encoding_parameters{};
            GRIM::FeedForwardParameterTensors detached_ffn_parameters{};
            if (!connect_parameter_graph) {
                detached_encoding_parameters = detachEncodingLayerParameters(
                    encoding_parameters,
                    encoder_hp.attention_qkv_bias_enabled,
                    encoder_hp.attention_output_bias_enabled,
                    use_layer_scale,
                    encoder_hp.attention_residual_gate_enabled,
                    request.stream);
                detached_ffn_parameters = detachFeedForwardParameters(
                    ffn_parameters,
                    encoder_hp.ffn_output_bias_enabled,
                    request.stream);
                encoding_parameter_ptr = &detached_encoding_parameters;
                ffn_parameter_ptr = &detached_ffn_parameters;
            }
            // Inference KV-cache path: every layer reads/appends its own cache at
            // the SAME cache_seqlens offset (the fill before this forward). The
            // counter is advanced once, after the layer loop.
            KvCacheLayerView cache_view{};
            const KvCacheLayerView* cache_view_ptr = nullptr;
            if (request.kv_cache) {
                cache_view = request.kv_cache->layerView(layer_idx);
                cache_view_ptr = &cache_view;
            }
            forwardEncodingLayer(
                enc_layer->hp(),
                enc_layer->requireFeedForwardCompute("executeModelForward(no_grad)"),
                *layer_input,
                payload,
                *request.pbm,
                request.stream,
                request.cublas_handle,
                forward_outputs,
                request.batch_idx,
                false,
                layer_idx,
                encoding_parameter_ptr,
                ffn_parameter_ptr,
                cache_view_ptr);
            Tensor layer_output_view = viewCommittedTensor(
                forward_outputs.output_per_layer[static_cast<size_t>(layer_idx)],
                request.stream,
                "enc_layer_output",
                "executeModelForward(no_grad)");

            Tensor owned = Tensor::empty(layer_output_view.shape, false, request.stream, "no_grad_layer_output");
            const size_t bytes = static_cast<size_t>(layer_output_view.shape.total_elements()) * sizeof(float);
            cudaError_t cp_err = cudaMemcpyAsync(owned.data, layer_output_view.data, bytes, cudaMemcpyDeviceToDevice, request.stream);
            if (cp_err != cudaSuccess) {
                throw std::runtime_error("ModelForward(no_grad): copy layer output failed: " +
                    std::string(cudaGetErrorString(cp_err)));
            }
            cudaError_t sync_err = cudaStreamSynchronize(request.stream);
            if (sync_err != cudaSuccess) {
                throw std::runtime_error("ModelForward(no_grad): sync after layer output copy failed: " +
                    std::string(cudaGetErrorString(sync_err)));
            }

            materializeForwardArgBootstrapSeeds(
                request,
                slot_seed_encoder_hp,
                execution_hp.num_slots,
                layer_idx,
                bootstrap_layer,
                arg_bootstrap_active,
                owned,
                forward_outputs);
            running = std::move(owned);
        }

        cudaError_t enc_sync = cudaStreamSynchronize(request.stream);
        if (enc_sync != cudaSuccess) {
            throw std::runtime_error("ModelForward(no_grad): CUDA error after encoder layers: " +
                std::string(cudaGetErrorString(enc_sync)) + " (illegal access usually means a kernel wrote/read out of bounds)");
        }

        // KV-cache path: all layers have appended q_len tokens at the prior fill
        // offset. Advance the shared cache_seqlens exactly once so the next forward
        // (decode step or speculative verification) attends over the new prefix.
        if (request.kv_cache) {
            request.kv_cache->advance(total_tokens, request.stream);
        }

        forward_outputs.encoder_output_tensor = std::move(running);
        forward_outputs.encoder_output_tensor.requires_grad = false;
        forward_outputs.encoder_output_tensor.grad_fn.reset();
        forward_outputs.encoder_output_tensor.stream = request.stream;
        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete (no_grad)");
    } else {
        forward_outputs.encoder_layer_outputs.reserve(num_layers);
        forward_outputs.reserveLayerOutputs(num_layers);

        MFWD_INFO("Step 2: Running " << num_layers << " encoder layers with retained graph...");
        MFWD_INFO("  embedding_tensor.grad_fn=" << (void*)forward_outputs.embedding_tensor.grad_fn.get()
                  << " requires_grad=" << forward_outputs.embedding_tensor.requires_grad);

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }

            forward_outputs.pushLayerOutputs();

            const Tensor* layer_input = (layer_idx == 0)
                ? &forward_outputs.embedding_tensor
                : &forward_outputs.encoder_layer_outputs.back();
            const auto& encoding_parameters = request.parameter_registry->requireEncodingLayerParameters(
                layer_idx,
                "executeModelForward(retained_graph)");
            if (encoder_hp.attention_residual_gate_enabled) {
                request.parameter_registry->requireAttentionResidualGateParameters(
                    layer_idx, "executeModelForward(retained_graph)");
            }
            const auto& ffn_parameters = request.parameter_registry->requireFeedForwardParameters(
                layer_idx,
                "executeModelForward(retained_graph)");

            forwardEncodingLayer(
                enc_layer->hp(),
                enc_layer->requireFeedForwardCompute("executeModelForward(retained_graph)"),
                *layer_input,
                payload,
                *request.pbm,
                request.stream,
                request.cublas_handle,
                forward_outputs,
                request.batch_idx,
                dropout_enabled,
                layer_idx,
                &encoding_parameters,
                &ffn_parameters);
            Tensor layer_output = viewCommittedTensor(
                forward_outputs.output_per_layer[static_cast<size_t>(layer_idx)],
                request.stream,
                "enc_layer_output",
                "executeModelForward(retained_graph)");

            materializeForwardArgBootstrapSeeds(
                request,
                slot_seed_encoder_hp,
                execution_hp.num_slots,
                layer_idx,
                bootstrap_layer,
                arg_bootstrap_active,
                layer_output,
                forward_outputs);

            forward_outputs.encoder_layer_outputs.push_back(std::move(layer_output));
        }

        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete");
        Tensor& last = forward_outputs.encoder_layer_outputs.back();
        forward_outputs.encoder_output_tensor = Tensor::from_ptr(
            last.data,
            TensorContract::TensorShape::make_BSM(total_tokens, d_model),
            false,
            true,
            "encoder_output_for_lmhead");
        forward_outputs.encoder_output_tensor.is_leaf = false;
        forward_outputs.encoder_output_tensor.stream = request.stream;
        forward_outputs.encoder_output_tensor.grad_fn = last.grad_fn;
    }

    const GRIM::LMHeadParameterTensors* lm_head_parameter_ptr = &lm_head_parameters;
    GRIM::LMHeadParameterTensors detached_lm_head_parameters{};
    if (!connect_parameter_graph) {
        detached_lm_head_parameters = detachLmHeadParameters(lm_head_parameters, request.stream);
        lm_head_parameter_ptr = &detached_lm_head_parameters;
    }

    forwardLmHead(
        lm_head_hp,
        *lm_head_parameter_ptr,
        forward_outputs.encoder_output_tensor,
        payload,
        request.stream,
        request.cublas_handle,
        forward_outputs);
    if (!forward_outputs.logits_tensor.data) {
        throw std::runtime_error("ModelForward: LMHeadLayer::forward returned logits tensor with NULL data");
    }

    const Tensor* live_lm_head_input = forward_outputs.liveLmHeadInputOrNull();
    if (!live_lm_head_input || !live_lm_head_input->data) {
        throw std::runtime_error("ModelForward: LM-head input snapshot is NULL after LMHeadLayer::forward");
    }

    materializeForwardSelectorLogits(request, payload, forward_outputs);

    MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << payload.vocab_size << "]");

    return forward_outputs;
}

}  // namespace Forward
}  // namespace GRIM
