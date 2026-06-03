//======================================================//
//  ModelForward_GPU.cu
//  Shared full-model forward primitive
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelForward_GPU.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "ModelForwardOutputs.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../VerboseLogging.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Forward {

#define MFWD_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[ModelForward] INFO: " << msg << std::endl; \
    } \
} while(0)

namespace {

const char* graphPolicyName(const ModelForwardGraphPolicy& graph) {
    if (graph.connect_parameter_graph && graph.retain_backward_graph) {
        return graph.emit_mtp_logits ? "autograd_connected+mtp" : "autograd_connected";
    }
    if (!graph.connect_parameter_graph && !graph.retain_backward_graph) {
        return graph.emit_mtp_logits ? "read_only+mtp" : "read_only";
    }
    throw std::runtime_error("ModelForward: invalid graph policy — connect_parameter_graph and retain_backward_graph must agree at this boundary");
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

int requirePayloadRowLength(const Batching::BatchPayload& payload,
                            int row,
                            const char* caller) {
    if (row < 0 || row >= payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": row index " +
                                 std::to_string(row) + " out of range for batch_size=" +
                                 std::to_string(payload.batch_size));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(payload.batch_size) + ")");
    }
    const int row_len = payload.seq_lengths[static_cast<size_t>(row)];
    if (row_len <= 0 || row_len > payload.max_seq_len) {
        throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                 std::to_string(row) + "]=" + std::to_string(row_len) +
                                 " for payload.max_seq_len=" + std::to_string(payload.max_seq_len));
    }
    return row_len;
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

void materializeForwardMtpLogits(
    const ModelForwardRequest& request,
    const Batching::BatchPayload& payload,
    ModelForwardOutputs& forward_outputs) {
    forward_outputs.mtp_logits_tensors.clear();
    if (!request.graph.emit_mtp_logits) {
        return;
    }

    const bool mtp_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*request.config, "mtp_enabled");
    const int mtp_k = HyperParameters::snapshotTrainingConfigField<int>(*request.config, "mtp_k");
    if (!mtp_enabled || mtp_k <= 0) {
        throw std::runtime_error("executeModelForward: graph.emit_mtp_logits=true but config MTP is disabled");
    }
    if (static_cast<int>(request.mtp_heads.size()) != mtp_k) {
        throw std::runtime_error("executeModelForward: request.mtp_heads.size()=" +
            std::to_string(request.mtp_heads.size()) + " != config.mtp_k=" + std::to_string(mtp_k));
    }

    Tensor* mtp_input = forward_outputs.liveLmHeadInputOrNull();
    if (!mtp_input || !mtp_input->data) {
        throw std::runtime_error("executeModelForward: live LM-head input snapshot is NULL before MTP logits materialization");
    }

    const auto& input_shape = requireTensor2DShape(*mtp_input, "executeModelForward", "mtp_input");
    if (input_shape.rows != payload.total_tokens) {
        throw std::runtime_error("executeModelForward: mtp_input rows=" +
            std::to_string(input_shape.rows) + " != payload.total_tokens=" +
            std::to_string(payload.total_tokens));
    }

    forward_outputs.mtp_logits_tensors.reserve(static_cast<size_t>(mtp_k));
    for (int k = 0; k < mtp_k; ++k) {
        const auto& head = request.mtp_heads[static_cast<size_t>(k)];
        if (!head.weight.data || !head.bias.data) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " weight or bias tensor is NULL");
        }

        const auto& weight_shape = requireTensor2DShape(head.weight, "executeModelForward", "MTP head weight");
        const auto& bias_shape = requireTensor2DShape(head.bias, "executeModelForward", "MTP head bias");
        if (weight_shape.rows != payload.vocab_size) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " weight rows=" + std::to_string(weight_shape.rows) +
                " != payload.vocab_size=" + std::to_string(payload.vocab_size));
        }
        if (weight_shape.cols != input_shape.cols) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " weight cols=" + std::to_string(weight_shape.cols) +
                " != mtp_input cols=" + std::to_string(input_shape.cols));
        }
        if (head.bias.numel() != static_cast<size_t>(payload.vocab_size)) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " bias elements=" + std::to_string(head.bias.numel()) +
                " != payload.vocab_size=" + std::to_string(payload.vocab_size));
        }
        if (bias_shape.cols != payload.vocab_size && bias_shape.rows != payload.vocab_size) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " bias shape=[" + std::to_string(bias_shape.rows) + "," + std::to_string(bias_shape.cols) +
                "] cannot represent vocab_size=" + std::to_string(payload.vocab_size));
        }

        Tensor logits_k = autograd::matmul(
            *mtp_input,
            head.weight,
            request.stream,
            true);
        logits_k = autograd::broadcast_add(logits_k, head.bias, request.stream);
        const auto& logits_shape = requireTensor2DShape(logits_k, "executeModelForward", "MTP head logits");
        if (logits_shape.rows != payload.total_tokens || logits_shape.cols != payload.vocab_size) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " logits shape=[" + std::to_string(logits_shape.rows) + "," + std::to_string(logits_shape.cols) +
                "] does not match payload [total_tokens=" + std::to_string(payload.total_tokens) +
                ", vocab_size=" + std::to_string(payload.vocab_size) + "]");
        }
        forward_outputs.mtp_logits_tensors.push_back(std::move(logits_k));
    }
}

GRIM::FeedForwardParameterTensors detachFeedForwardParameters(
    const GRIM::FeedForwardParameterTensors& parameters,
    bool use_bias,
    cudaStream_t stream) {
    GRIM::FeedForwardParameterTensors detached{};
    detached.W_gate = parameters.W_gate.detach(stream);
    detached.W1 = parameters.W1.detach(stream);
    detached.W2 = parameters.W2.detach(stream);
    if (use_bias && parameters.b2.data) {
        detached.b2 = parameters.b2.detach(stream);
    }
    return detached;
}

GRIM::EncodingLayerParameterTensors detachEncodingLayerParameters(
    const GRIM::EncodingLayerParameterTensors& parameters,
    bool use_bias,
    bool use_layer_scale,
    cudaStream_t stream) {
    GRIM::EncodingLayerParameterTensors detached{};
    detached.rms1_gamma = parameters.rms1_gamma.detach(stream);
    detached.rms2_gamma = parameters.rms2_gamma.detach(stream);
    detached.W_qkv = parameters.W_qkv.detach(stream);
    detached.W_o = parameters.W_o.detach(stream);
    if (use_bias) {
        detached.b_qkv = parameters.b_qkv.detach(stream);
        detached.b_o = parameters.b_o.detach(stream);
    }
    if (use_layer_scale) {
        detached.layer_scale1 = parameters.layer_scale1.detach(stream);
        detached.layer_scale2 = parameters.layer_scale2.detach(stream);
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
            throw std::runtime_error(std::string(caller) + ": execution_block_enabled=false while ExecutionBlockConstructionHP.enabled=true");
        }
        (void)parameter_registry->requireExecutionBlockParameters(caller);
    } else if (execution_block_enabled) {
        throw std::runtime_error(std::string(caller) + ": execution_block_enabled=true while ExecutionBlockConstructionHP.enabled=false");
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
    if (graph.emit_mtp_logits) {
        const bool mtp_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*config, "mtp_enabled");
        const int mtp_k = HyperParameters::snapshotTrainingConfigField<int>(*config, "mtp_k");
        if (!mtp_enabled || mtp_k <= 0) {
            throw std::runtime_error(std::string(caller) + ": graph.emit_mtp_logits=true while config MTP is disabled");
        }
        if (static_cast<int>(mtp_heads.size()) != mtp_k) {
            throw std::runtime_error(std::string(caller) + ": mtp_heads.size()=" +
                                     std::to_string(mtp_heads.size()) + " != config.mtp_k=" +
                                     std::to_string(mtp_k));
        }
    } else if (!mtp_heads.empty()) {
        throw std::runtime_error(std::string(caller) + ": mtp_heads is non-empty while graph.emit_mtp_logits=false");
    }
}

ModelForwardOutputs executeModelForward(const ModelForwardRequest& request,
                                        ModelForwardRuntimePayload& runtime_payload) {
    request.validate("executeModelForward");
    const auto* cfg = request.config;
    const auto embedding_hp = HyperParameters::embeddingLayerConstructionHP(*cfg);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*cfg);
    const auto lm_head_hp = HyperParameters::lmHeadLayerConstructionHP(*cfg);
    const bool center_encoder_residuals = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "center_encoder_residuals");
    const bool lm_head_center_hidden_states = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "lm_head_center_hidden_states");
    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "d_model");
    const auto positional_encoding = HyperParameters::snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(*cfg, "positional_encoding");
    const float dropout_rate = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "dropout_rate");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "num_layers");
    const bool use_bias = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "use_bias");
    const bool use_layer_scale = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "use_layer_scale");
    const int execution_block_layer = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_layer");
    const int execution_block_num_steps = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_steps");
    const int execution_block_num_slots = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_slots");
    const int execution_block_num_ops = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_ops");
    const float execution_block_temp_start = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "execution_block_temp_start");
    const bool execution_block_active = execution_hp.enabled && request.execution_block_enabled;
    auto* execution_block_parameters = execution_block_active
        ? &request.parameter_registry->requireExecutionBlockParameters("executeModelForward")
        : nullptr;

    runtime_payload.validate(
        "executeModelForward",
        execution_block_active);

    const auto& payload = *request.payload;
    auto& runtime = runtime_payload;
    ModelForwardOutputs forward_outputs;
    if (execution_block_active) {
        forward_outputs.ensureExecutionBatchGeometry(
            static_cast<size_t>(payload.batch_size),
            "executeModelForward");
        runtime.execution_runtime->ensureBatchGeometry(
            static_cast<size_t>(payload.batch_size),
            "executeModelForward");
    }
    const auto* bindings = request.bindings;
    const auto& embedding_parameters = request.parameter_registry->requireEmbeddingParameters("executeModelForward");
    const auto& lm_head_parameters = request.parameter_registry->requireLmHeadParameters("executeModelForward");
    const bool connect_parameter_graph = request.graph.connect_parameter_graph;
    const bool retain_backward_graph = request.graph.retain_backward_graph;
    const bool dropout_enabled = request.graph.enable_dropout;
    const bool emit_mtp_logits = request.graph.emit_mtp_logits;

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

    float* encoder_output = nullptr;

    if (!retain_backward_graph) {
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

            Tensor& layer_input = (layer_idx == 0) ? forward_outputs.embedding_tensor : running;

            const auto& encoding_parameters = request.parameter_registry->requireEncodingLayerParameters(
                layer_idx,
                "executeModelForward(no_grad)");
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
                    use_bias,
                    use_layer_scale,
                    request.stream);
                detached_ffn_parameters = detachFeedForwardParameters(
                    ffn_parameters,
                    use_bias,
                    request.stream);
                encoding_parameter_ptr = &detached_encoding_parameters;
                ffn_parameter_ptr = &detached_ffn_parameters;
            }
            forwardEncodingLayer(
                enc_layer->hp(),
                enc_layer->requireFeedForwardCompute("executeModelForward(no_grad)"),
                layer_input,
                payload,
                *request.pbm,
                request.stream,
                request.cublas_handle,
                forward_outputs,
                request.batch_idx,
                false,
                layer_idx,
                encoding_parameter_ptr,
                ffn_parameter_ptr);
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

        encoder_output = running.data;
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

        int exec_layer = -1;
        int exec_K = 0;
        if (execution_block_active) {
            exec_layer = execution_block_layer;
            if (exec_layer < 0) exec_layer = num_layers - 2;
            if (exec_layer < 0) exec_layer = 0;
            if (exec_layer >= num_layers) exec_layer = num_layers - 1;
            exec_K = execution_block_num_steps;
        }

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }

            forward_outputs.pushLayerOutputs();

            const Tensor* layer_input = (layer_idx == 0)
                ? &forward_outputs.embedding_tensor
                : &forward_outputs.encoder_layer_outputs.back();
            Tensor execution_read_augmented_input;
            const auto& encoding_parameters = request.parameter_registry->requireEncodingLayerParameters(
                layer_idx,
                "executeModelForward(retained_graph)");
            const auto& ffn_parameters = request.parameter_registry->requireFeedForwardParameters(
                layer_idx,
                "executeModelForward(retained_graph)");

            if (exec_layer >= 0 && layer_idx > exec_layer
                && execution_block_active
                && !forward_outputs.exec_memories.empty()) {
                // executeStep(...) may export the immediate step result on the
                // execution layer output, but persistent ExecutionMemory is a
                // downstream side channel. Its first consumer is the next
                // layer input at the row-final token only.
                bool has_execution_readback = false;
                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];
                    if (!row_exec_active) continue;
                    const Tensor& read_source = has_execution_readback
                        ? execution_read_augmented_input
                        : *layer_input;
                    const int row_len = requirePayloadRowLength(
                        payload, b, "ModelForward ExecutionBlock next-layer input readback");
                    const int final_token_offset = b * payload.max_seq_len + row_len - 1;
                    Tensor row_delta = GRIM::executionBlockCrossAttentionRead(
                        execution_hp, read_source, forward_outputs.exec_memories[b], *execution_block_parameters,
                        total_tokens, request.stream,
                        final_token_offset, 1,
                        runtime.read_gate_accum_tensor
                            ? runtime.read_gate_accum_tensor->data
                            : nullptr);
                    Tensor padded = autograd::zero_pad(
                        row_delta, final_token_offset, total_tokens, request.stream);
                    execution_read_augmented_input = autograd::add(
                        read_source, padded, request.stream);
                    has_execution_readback = true;
                }
                if (has_execution_readback) {
                    layer_input = &execution_read_augmented_input;
                }
            }

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

            if (layer_idx == exec_layer && execution_block_active) {
                const int V = execution_block_num_slots;
                const int nop = execution_block_num_ops;

                float T = execution_block_temp_start;

                auto& execution_runtime = *runtime.execution_runtime;
                Forward::provisionExecutionForwardRuntime(
                    payload.execution_active,
                    payload.batch_size,
                    execution_hp.num_slots,
                    execution_hp.atom_embedding_dim,
                    execution_hp.d_model,
                    execution_hp.d_key,
                    execution_hp.d_type,
                    connect_parameter_graph,
                    request.stream,
                    forward_outputs,
                    execution_runtime);
                execution_runtime.ensureDiagnostics(request.stream);

                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];

                    if (!row_exec_active) continue;

                    auto& M_b = forward_outputs.exec_memories[b];

                    const int tok_off = b * payload.max_seq_len;
                    const int row_len = requirePayloadRowLength(payload, b, "ModelForward ExecutionBlock bootstrap");

                    if (!request.bindings || !request.bindings->d_token_to_slot_map
                        || !request.bindings->d_numeric_values) {
                        throw std::runtime_error(
                            "ModelForward: execution-active row " + std::to_string(b)
                            + " has no slot map or numeric values for bootstrap — "
                            "compiled payload marks row active but bootstrap data is missing");
                    }
                    GRIM::executionBlockBootstrapMemoryFromSlotMap(
                        execution_hp,
                        M_b,
                        *execution_block_parameters,
                        request.bindings->d_numeric_values + tok_off,
                        request.bindings->d_token_to_slot_map + tok_off,
                        row_len, request.stream);

                    for (int step = 0; step < exec_K; ++step) {
                        ExecutionBlockStepOutput step_diag;

                        GRIM::executionBlockStep(
                            execution_hp, execution_runtime.execution_diag,
                            layer_output, M_b, *execution_block_parameters,
                            payload, *request.bindings, b,
                            step, T, request.stream,
                            &step_diag,
                            runtime.execution_runtime->trace_state_by_row[b],
                            runtime.execution_runtime->execution_trace_by_row[b]);
                        runtime.execution_runtime->execution_trace_by_row[b].push_back(step_diag.record);
                        forward_outputs.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
                    }
                }
            }

            forward_outputs.encoder_layer_outputs.push_back(std::move(layer_output));
        }

        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete");
        encoder_output = forward_outputs.encoder_layer_outputs.back().data;
    }

    if (retain_backward_graph) {
        const bool lmhead_track_grad = true;
        Tensor encoder_output_tensor = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, d_model),
            false,
            lmhead_track_grad,
            "encoder_output_for_lmhead");
        encoder_output_tensor.is_leaf = false;
        encoder_output_tensor.stream = request.stream;
        encoder_output_tensor.grad_fn = forward_outputs.encoder_layer_outputs.back().grad_fn;
        forward_outputs.encoder_output_tensor = std::move(encoder_output_tensor);
    }

    LMHeadParameterViews lm_head_parameter_views{};
    const LMHeadParameterViews* lm_head_parameter_view_ptr = nullptr;
    Tensor lm_head_weights_view;
    Tensor lm_head_bias_view;
    Tensor lm_head_gamma_view;
    if (!connect_parameter_graph) {
        lm_head_weights_view = lm_head_parameters.weights.detach(request.stream);
        lm_head_parameter_views.weights = &lm_head_weights_view;
        if (lm_head_parameters.bias.data) {
            lm_head_bias_view = lm_head_parameters.bias.detach(request.stream);
            lm_head_parameter_views.bias = &lm_head_bias_view;
        }
        if (lm_head_parameters.final_rms_gamma.data) {
            lm_head_gamma_view = lm_head_parameters.final_rms_gamma.detach(request.stream);
            lm_head_parameter_views.final_rms_gamma = &lm_head_gamma_view;
        }
        lm_head_parameter_view_ptr = &lm_head_parameter_views;
    }

    forwardLmHead(
        lm_head_hp,
        lm_head_parameters,
        forward_outputs.encoder_output_tensor,
        payload,
        request.stream,
        request.cublas_handle,
        forward_outputs,
        lm_head_parameter_view_ptr);
    if (!forward_outputs.logits_tensor.data) {
        throw std::runtime_error("ModelForward: LMHeadLayer::forward returned logits tensor with NULL data");
    }

    const Tensor* live_lm_head_input = forward_outputs.liveLmHeadInputOrNull();
    if (!live_lm_head_input || !live_lm_head_input->data) {
        throw std::runtime_error("ModelForward: LM-head input snapshot is NULL after LMHeadLayer::forward");
    }

    materializeForwardMtpLogits(request, payload, forward_outputs);
 
    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        constexpr int kSamplePositions = 1024;
        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_logits(sample_size * payload.vocab_size);

        cudaMemcpyAsync(h_encoder.data(), live_lm_head_input->data,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaMemcpyAsync(h_logits.data(), forward_outputs.logits_tensor.data,
                        sample_size * payload.vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaStreamSynchronize(request.stream);

        for (int pos = 0; pos < sample_size; ++pos) {
            const float* h = h_encoder.data() + pos * d_model;
            const float* logits_row = h_logits.data() + pos * payload.vocab_size;

            int argmax_token = 0;
            float max_logit_val = logits_row[0];
            for (int v = 1; v < payload.vocab_size; ++v) {
                if (logits_row[v] > max_logit_val) {
                    max_logit_val = logits_row[v];
                    argmax_token = v;
                }
            }

            std::vector<float> h_weights_argmax(d_model);
            cudaMemcpyAsync(h_weights_argmax.data(),
                            lm_head_parameters.weights.data + static_cast<size_t>(argmax_token) * d_model,
                            d_model * sizeof(float),
                            cudaMemcpyDeviceToHost, request.stream);
            cudaStreamSynchronize(request.stream);

            float h_sum = 0.0f, h_sum_sq = 0.0f, h_min = h[0], h_max = h[0];
            for (int d = 0; d < d_model; ++d) {
                h_sum += h[d];
                h_sum_sq += h[d] * h[d];
                h_min = std::min(h_min, h[d]);
                h_max = std::max(h_max, h[d]);
            }
            float h_mean = h_sum / d_model;
            float h_rms = std::sqrt(h_sum_sq / d_model);

            float w_sum = 0.0f, w_sum_sq = 0.0f, w_min = h_weights_argmax[0], w_max = h_weights_argmax[0];
            for (int d = 0; d < d_model; ++d) {
                w_sum += h_weights_argmax[d];
                w_sum_sq += h_weights_argmax[d] * h_weights_argmax[d];
                w_min = std::min(w_min, h_weights_argmax[d]);
                w_max = std::max(w_max, h_weights_argmax[d]);
            }
            float w_mean = w_sum / d_model;
            float w_rms = std::sqrt(w_sum_sq / d_model);

            float dot_product_argmax = 0.0f;
            float positive_contrib = 0.0f, negative_contrib = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                float contrib = h[d] * h_weights_argmax[d];
                dot_product_argmax += contrib;
                if (contrib > 0) positive_contrib += contrib;
                else negative_contrib += contrib;
            }

            float h_rms_val = std::sqrt(h_sum_sq / d_model);
            float w_rms_val = std::sqrt(w_sum_sq / d_model);
            float cosine_sim = (h_rms_val > 1e-8f && w_rms_val > 1e-8f)
                               ? (dot_product_argmax / (h_rms_val * w_rms_val * d_model)) : 0.0f;

            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            fprintf(stderr, "[LOGIT_ANALYSIS] Position %d: Why does logit[v] = Σ_d h[d] × W[v,d] choose token %d?\n", pos, argmax_token);
            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            fprintf(stderr, "HIDDEN STATE h[pos=%d]:\n", pos);
            fprintf(stderr, "  Statistics: mean=%.6f (offset) rms=%.6f (magnitude) range=[%.6f, %.6f]\n",
                    h_mean, h_rms, h_min, h_max);
            fprintf(stderr, "WEIGHT ROW W[%d] (predicted token):\n", argmax_token);
            fprintf(stderr, "  Statistics: mean=%.6f rms=%.6f range=[%.6f, %.6f]\n",
                    w_mean, w_rms, w_min, w_max);
            fprintf(stderr, "DOT_PRODUCT ANALYSIS Σ_d h[d]×W[%d,d]:\n", argmax_token);
            fprintf(stderr, "  Raw computation: %.6f\n", dot_product_argmax);
            fprintf(stderr, "  ├─ Positive contributions (h×W>0): %.6f (%.1f%%)\n",
                    positive_contrib, 100.0f * positive_contrib / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Negative contributions (h×W<0): %.6f (%.1f%%)\n",
                    negative_contrib, 100.0f * std::abs(negative_contrib) / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Cosine alignment: %.6f (1.0=perfect alignment, 0=orthogonal, -1=opposite)\n", cosine_sim);
            fprintf(stderr, "RESULT:\n");
            fprintf(stderr, "  logit[%d]=%.6f (PREDICTED argmax token)\n", argmax_token, max_logit_val);
            fprintf(stderr, "\n");
        }
    }

    if (emit_mtp_logits) {
        MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << payload.vocab_size
                  << "] mtp_heads=" << forward_outputs.mtp_logits_tensors.size());
    } else {
        MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << payload.vocab_size << "]");
    }

    return forward_outputs;
}

}  // namespace Forward
}  // namespace GRIM
