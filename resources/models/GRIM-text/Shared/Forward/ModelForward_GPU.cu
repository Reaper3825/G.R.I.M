//======================================================//
//  ModelForward_GPU.cu
//  Shared full-model forward primitive
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelForward_GPU.hpp"
#include "NumericAtomForward.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../InferenceState/KvCacheState_GPU.hpp"
#include "../CudaAllocUtils.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "ModelForwardOutputs.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../VerboseLogging.hpp"

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

GRIM::NumberEncoderParameterTensors detachNumericAtomParameters(
    const GRIM::NumberEncoderParameterTensors& parameters,
    cudaStream_t stream) {
    GRIM::NumberEncoderParameterTensors detached{};
    detached.digit_emb = parameters.digit_emb.detach(stream);
    detached.pow10_emb = parameters.pow10_emb.detach(stream);
    detached.numeric_atom_Wz = parameters.numeric_atom_Wz.detach(stream);
    detached.numeric_atom_Uz = parameters.numeric_atom_Uz.detach(stream);
    detached.numeric_atom_Wr = parameters.numeric_atom_Wr.detach(stream);
    detached.numeric_atom_Ur = parameters.numeric_atom_Ur.detach(stream);
    detached.numeric_atom_Wh = parameters.numeric_atom_Wh.detach(stream);
    detached.numeric_atom_Uh = parameters.numeric_atom_Uh.detach(stream);
    detached.numeric_atom_stop_classifier =
        parameters.numeric_atom_stop_classifier.detach(stream);
    return detached;
}

}  // namespace

GoalSpanView ModelForwardRequest::goalSpansForRow(std::size_t row) const {
    if (!payload) {
        throw std::runtime_error(
            "ModelForwardRequest::goalSpansForRow: payload is NULL");
    }
    return payload->goalSpansForRow(row);
}

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
    const auto lm_head_hp = HyperParameters::lmHeadLayerConstructionHP(*cfg);
    const auto number_encoder_hp = HyperParameters::numberEncoderConstructionHP(*cfg);
    const bool center_encoder_residuals = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "center_encoder_residuals");
    const bool lm_head_center_hidden_states = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "lm_head_center_hidden_states");
    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "d_model");
    const auto positional_encoding = HyperParameters::snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(*cfg, "positional_encoding");
    const float dropout_rate = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "dropout_rate");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "num_layers");
    const bool use_layer_scale = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "use_layer_scale");

    const auto& payload = *request.payload;
    (void)runtime_payload;
    ModelForwardOutputs forward_outputs;
    forward_outputs.setGoalMetadata(
        static_cast<std::size_t>(payload.batch_size), payload.goals);
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

    // The encoder output is the shared model hidden state. Typed auxiliary
    // heads branch from it independently of the LM head and consume the
    // complete BatchPayload semantic contract directly.
    // The recurrent NumericAtom training forward consumes teacher-forced
    // digit/pow10 targets. Inference owns a persistent per-open-atom state and
    // enters through the dedicated NumericAtom inference step boundary.
    if (number_encoder_hp.enabled && payload.number_aux_target_digit_slots > 0) {
        const auto& numeric_parameters =
            request.parameter_registry->requireNumberEncoderParameters(
                "executeModelForward.NumericAtomForward");
        const GRIM::NumberEncoderParameterTensors* numeric_parameter_ptr =
            &numeric_parameters;
        GRIM::NumberEncoderParameterTensors detached_numeric_parameters{};
        if (!connect_parameter_graph) {
            detached_numeric_parameters = detachNumericAtomParameters(
                numeric_parameters,
                request.stream);
            numeric_parameter_ptr = &detached_numeric_parameters;
        }
        forward_outputs.numeric_atom = NumericAtomForward(
            forward_outputs.encoder_output_tensor,
            *numeric_parameter_ptr,
            payload,
            *bindings,
            request.stream);
        if (!forward_outputs.numeric_atom.populated()) {
            throw std::runtime_error(
                "executeModelForward: NumericAtomForward returned empty logits");
        }
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

    MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << payload.vocab_size << "]");

    return forward_outputs;
}

}  // namespace Forward
}  // namespace GRIM
