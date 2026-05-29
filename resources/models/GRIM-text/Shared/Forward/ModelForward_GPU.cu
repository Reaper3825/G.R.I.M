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
#include "../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
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
            mtp_input->data,
            nullptr,
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

}  // namespace

void ModelForwardRequest::validate(const char* caller) const {
    if (!config) throw std::runtime_error(std::string(caller) + ": config is NULL");
    if (!gpu_encoder) throw std::runtime_error(std::string(caller) + ": gpu_encoder is NULL");
    if (!embedding_layer) throw std::runtime_error(std::string(caller) + ": embedding_layer is NULL");
    if (!lm_head) throw std::runtime_error(std::string(caller) + ": lm_head is NULL");
    if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
    if (!stream) throw std::runtime_error(std::string(caller) + ": stream is NULL");
    if (!payload) throw std::runtime_error(std::string(caller) + ": payload is NULL");
    if (!bindings) throw std::runtime_error(std::string(caller) + ": bindings is NULL");
    (void)graphPolicyName(graph);
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

void executeModelForward(const ModelForwardRequest& request,
                         ModelForwardRuntimePayload& runtime_payload) {
    request.validate("executeModelForward");
    const auto* cfg = request.config;
    const auto scratch_hp = HyperParameters::scratchBlockConstructionHP(*cfg);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*cfg);
    const bool center_encoder_residuals = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "center_encoder_residuals");
    const bool lm_head_center_hidden_states = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "lm_head_center_hidden_states");
    const int vocab_size = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "vocab_size");
    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "d_model");
    const auto positional_encoding = HyperParameters::snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(*cfg, "positional_encoding");
    const bool scratch_block_execution_first_type_only = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "scratch_block_execution_first_type_only");
    const float dropout_rate = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "dropout_rate");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "num_layers");
    const bool use_layer_scale = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "use_layer_scale");
    const int execution_block_layer = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_layer");
    const int execution_block_num_steps = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_steps");
    const int execution_block_num_slots = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_slots");
    const int execution_block_num_ops = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_ops");
    const float execution_block_temp_start = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "execution_block_temp_start");
    const int scratch_block_atom_embedding_dim = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "scratch_block_atom_embedding_dim");
    const bool scratch_block_active = scratch_hp.enabled && request.scratch_block != nullptr;
    const bool execution_block_active = execution_hp.enabled && request.execution_block != nullptr;

    if (scratch_hp.enabled && !request.scratch_block) {
        throw std::runtime_error("ModelForward: ScratchBlockConstructionHP.enabled=true but request.scratch_block is NULL");
    }
    if (!scratch_hp.enabled && request.scratch_block) {
        throw std::runtime_error("ModelForward: request.scratch_block is non-null while ScratchBlockConstructionHP.enabled=false");
    }

    runtime_payload.validate(
        "executeModelForward",
        execution_block_active);

    auto& runtime = runtime_payload;
    auto& forward_outputs = *runtime.forward_outputs;
    const auto& payload = *request.payload;
    const auto* bindings = request.bindings;
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
    const Tensor* emb_weights = &request.embedding_layer->tokenWeights();
    if (!connect_parameter_graph) {
        emb_weights_view = request.embedding_layer->tokenWeights().detach(request.stream);
        emb_weights = &emb_weights_view;
    }
    if (!emb_weights->data) {
        throw std::runtime_error("ModelForward: embedding token_weights.data is NULL");
    }

    if (!emb_weights->shape.is_valid()) {
        throw std::runtime_error("ModelForward: embedding token_weights.shape is INVALID - EmbeddingLayer MUST initialize with correct shape [vocab_size="
                                + std::to_string(vocab_size) + ", d_model=" + std::to_string(d_model) + "]");
    }

    const float embedding_scale = 1.0f;
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
    MFWD_INFO("Step 1: Token embedding complete, shape=[" << total_tokens << ", " << d_model << "]");

    if (scratch_block_active) {
        MFWD_INFO("Step 1.5: Running all-token ScratchBlock vector gate...");
        (void)cudaGetLastError();

        if (!bindings->d_numeric_values) {
            throw std::runtime_error("executeModelForward: ScratchBlock requires BatchDeviceBindings.d_numeric_values");
        }
        if (!bindings->d_token_to_slot_map) {
            throw std::runtime_error("executeModelForward: ScratchBlock requires BatchDeviceBindings.d_token_to_slot_map");
        }

        const bool exec_first_type_only =
            execution_hp.enabled && scratch_block_execution_first_type_only;
        if (execution_hp.enabled && !exec_first_type_only) {
            throw std::runtime_error(
                "executeModelForward: execution_block_enabled requires "
                "scratch_block_execution_first_type_only=true to prevent value leakage "
                "into hidden states on arithmetic batches");
        }

        ScratchBlockProjectionParameterViews scratch_param_views{};
        const ScratchBlockProjectionParameterViews* scratch_param_view_ptr = nullptr;
        Tensor scratch_atom_type_embeddings_view;
        Tensor scratch_atom_projection_view;
        if (!connect_parameter_graph) {
            scratch_atom_type_embeddings_view = request.scratch_block->atomTypeEmbeddings().detach(request.stream);
            scratch_atom_projection_view = request.scratch_block->atomProjection().detach(request.stream);
            scratch_param_views.atom_type_embeddings = &scratch_atom_type_embeddings_view;
            scratch_param_views.atom_projection = &scratch_atom_projection_view;
            scratch_param_view_ptr = &scratch_param_views;
        }

        forward_outputs.embedding_structured_state = autograd::scratch_block_project_all_tokens(
            *request.scratch_block,
            token_ids,
            bindings->d_numeric_values,
            bindings->d_atom_mask,
            bindings->d_atom_flags,
            bindings->d_token_to_slot_map,
            total_tokens,
            request.stream,
            exec_first_type_only,
            connect_parameter_graph,
            scratch_param_view_ptr);

        forward_outputs.embedding_gate_concat = autograd::concat(
            forward_outputs.embedding_tensor,
            forward_outputs.embedding_structured_state,
            request.stream);

        if (connect_parameter_graph) {
            forward_outputs.embedding_gate_logits = autograd::matmul(
                forward_outputs.embedding_gate_concat,
                request.scratch_block->structuredGateWeight(),
                request.stream,
                forward_outputs.embedding_gate_concat.data,
                nullptr);
        } else {
            Tensor gate_weight_view = request.scratch_block->structuredGateWeight().detach(request.stream);
            forward_outputs.embedding_gate_logits = autograd::matmul(
                forward_outputs.embedding_gate_concat,
                gate_weight_view,
                request.stream,
                nullptr,
                nullptr);
        }

        forward_outputs.embedding_gate_values = autograd::sigmoid(
            forward_outputs.embedding_gate_logits,
            request.stream,
            forward_outputs.embedding_gate_logits.data);

        forward_outputs.embedding_gate_delta = autograd::elementwise_mul(
            forward_outputs.embedding_gate_values,
            forward_outputs.embedding_structured_state,
            request.stream);

        forward_outputs.embedding_tensor = autograd::add(
            forward_outputs.embedding_tensor,
            forward_outputs.embedding_gate_delta,
            request.stream);

        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("ModelForward: ScratchBlock vector gate CUDA error: " +
                                     std::string(cudaGetErrorString(cuda_err)));
        }

        MFWD_INFO("Step 1.5: ScratchBlock vector gate complete");
    }

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

            EncodingLayerParameterViews enc_param_views{};
            const EncodingLayerParameterViews* enc_param_view_ptr = nullptr;
            Tensor rms1_view;
            Tensor rms2_view;
            Tensor wqkv_view;
            Tensor bqkv_view;
            Tensor wo_view;
            Tensor bo_view;
            Tensor layer_scale1_view;
            Tensor layer_scale2_view;
            Tensor ffn_w_gate_view;
            Tensor ffn_w1_view;
            Tensor ffn_w2_view;
            Tensor ffn_b2_view;
            if (!connect_parameter_graph) {
                rms1_view = enc_layer->rms1Gamma().detach(request.stream);
                rms2_view = enc_layer->rms2Gamma().detach(request.stream);
                wqkv_view = enc_layer->attnWqkv().detach(request.stream);
                wo_view = enc_layer->attnWo().detach(request.stream);
                enc_param_views.rms1_gamma = &rms1_view;
                enc_param_views.rms2_gamma = &rms2_view;
                enc_param_views.W_qkv = &wqkv_view;
                enc_param_views.W_o = &wo_view;
                if (enc_layer->attnBqkv().data) {
                    bqkv_view = enc_layer->attnBqkv().detach(request.stream);
                    enc_param_views.b_qkv = &bqkv_view;
                }
                if (enc_layer->attnBo().data) {
                    bo_view = enc_layer->attnBo().detach(request.stream);
                    enc_param_views.b_o = &bo_view;
                }
                if (use_layer_scale) {
                    layer_scale1_view = enc_layer->layerScale1().detach(request.stream);
                    layer_scale2_view = enc_layer->layerScale2().detach(request.stream);
                    enc_param_views.layer_scale1 = &layer_scale1_view;
                    enc_param_views.layer_scale2 = &layer_scale2_view;
                }
                FeedForwardLayer* ffn_layer = enc_layer->getFfnLayer();
                if (!ffn_layer) {
                    throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " FFN layer is NULL");
                }
                ffn_w_gate_view = ffn_layer->W_gate().detach(request.stream);
                ffn_w1_view = ffn_layer->W1().detach(request.stream);
                ffn_w2_view = ffn_layer->W2().detach(request.stream);
                enc_param_views.ffn.W_gate = &ffn_w_gate_view;
                enc_param_views.ffn.W1 = &ffn_w1_view;
                enc_param_views.ffn.W2 = &ffn_w2_view;
                if (ffn_layer->b2().data) {
                    ffn_b2_view = ffn_layer->b2().detach(request.stream);
                    enc_param_views.ffn.b2 = &ffn_b2_view;
                }
                enc_param_view_ptr = &enc_param_views;
            }
            enc_layer->forward(
                layer_input, payload, request.stream, request.cublas_handle,
                forward_outputs,
                request.batch_idx, false, layer_idx, enc_param_view_ptr);
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
        if (execution_block_active && scratch_block_active) {
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

            if (exec_layer >= 0 && layer_idx > exec_layer
                && request.execution_block
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
                    Tensor row_delta = request.execution_block->crossAttentionRead(
                        read_source, forward_outputs.exec_memories[b],
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

            enc_layer->forward(
                *layer_input, payload, request.stream, request.cublas_handle,
                forward_outputs,
                request.batch_idx, dropout_enabled, layer_idx, nullptr);
            Tensor layer_output = viewCommittedTensor(
                forward_outputs.output_per_layer[static_cast<size_t>(layer_idx)],
                request.stream,
                "enc_layer_output",
                "executeModelForward(retained_graph)");

            if (layer_idx == exec_layer && request.execution_block) {
                const int V = execution_block_num_slots;
                const int nop = execution_block_num_ops;

                float T = execution_block_temp_start;

                auto& execution_runtime = *runtime.execution_runtime;
                request.execution_block->prepareForwardRuntime(
                    payload,
                    connect_parameter_graph,
                    request.stream,
                    forward_outputs.exec_memories,
                    forward_outputs.exec_outputs_per_row,
                    execution_runtime.execution_trace_by_row,
                    execution_runtime.trace_state_by_row);

                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];

                    if (!row_exec_active) continue;

                    auto& M_b = forward_outputs.exec_memories[b];

                    const int tok_off = b * payload.max_seq_len;
                    const int row_len = requirePayloadRowLength(payload, b, "ModelForward ExecutionBlock bootstrap");

                    auto row_atom_view = request.scratch_block->extractRowLocalAtomView(
                        tok_off, row_len, request.stream);

                    if (!request.bindings || !request.bindings->d_token_to_slot_map
                        || !request.bindings->d_numeric_values) {
                        throw std::runtime_error(
                            "ModelForward: execution-active row " + std::to_string(b)
                            + " has no slot map or numeric values for bootstrap — "
                            "compiled payload marks row active but bootstrap data is missing");
                    }
                    request.execution_block->bootstrapMemoryFromSlotMap(
                        M_b,
                        request.bindings->d_numeric_values + tok_off,
                        request.bindings->d_token_to_slot_map + tok_off,
                        row_len, request.stream);

                    for (int step = 0; step < exec_K; ++step) {
                        ExecutionBlockStepOutput step_diag;

                        request.execution_block->executeStep(
                            layer_output, M_b,
                            reinterpret_cast<const int*>(row_atom_view.atom_positions.data),
                            row_atom_view.num_atoms, payload, *request.bindings, b,
                            step, T, request.stream,
                            &step_diag,
                            runtime.execution_runtime->trace_state_by_row[b],
                            runtime.execution_runtime->execution_trace_by_row[b]);
                        runtime.execution_runtime->execution_trace_by_row[b].push_back(step_diag.record);
                        forward_outputs.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
                    }
                }
            }

            if (center_encoder_residuals) {
                if (payload.max_seq_len <= 1) {
                    throw std::runtime_error("ModelForward: center_encoder_residuals requires payload.max_seq_len > 1; single-row column centering would erase the residual stream");
                }
                layer_output = autograd::center_columns_by_causal_prefix_lengths(
                    layer_output, payload.seq_lengths, payload.batch_size, payload.max_seq_len, request.stream);
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
        lm_head_weights_view = request.lm_head->weights().detach(request.stream);
        lm_head_parameter_views.weights = &lm_head_weights_view;
        if (request.lm_head->bias().data) {
            lm_head_bias_view = request.lm_head->bias().detach(request.stream);
            lm_head_parameter_views.bias = &lm_head_bias_view;
        }
        if (request.lm_head->finalRmsGamma().data) {
            lm_head_gamma_view = request.lm_head->finalRmsGamma().detach(request.stream);
            lm_head_parameter_views.final_rms_gamma = &lm_head_gamma_view;
        }
        lm_head_parameter_view_ptr = &lm_head_parameter_views;
    }

    request.lm_head->forward(
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
        const int vocab_size_local = vocab_size;

        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_logits(sample_size * vocab_size_local);

        cudaMemcpyAsync(h_encoder.data(), live_lm_head_input->data,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaMemcpyAsync(h_logits.data(), forward_outputs.logits_tensor.data,
                        sample_size * vocab_size_local * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaStreamSynchronize(request.stream);

        for (int pos = 0; pos < sample_size; ++pos) {
            const float* h = h_encoder.data() + pos * d_model;
            const float* logits_row = h_logits.data() + pos * vocab_size_local;

            int argmax_token = 0;
            float max_logit_val = logits_row[0];
            for (int v = 1; v < vocab_size_local; ++v) {
                if (logits_row[v] > max_logit_val) {
                    max_logit_val = logits_row[v];
                    argmax_token = v;
                }
            }

            std::vector<float> h_weights_argmax(d_model);
            cudaMemcpyAsync(h_weights_argmax.data(),
                            request.lm_head->weights().data + static_cast<size_t>(argmax_token) * d_model,
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
        MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << vocab_size
                  << "] mtp_heads=" << forward_outputs.mtp_logits_tensors.size());
    } else {
        MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << vocab_size << "]");
    }
}

}  // namespace Forward
}  // namespace GRIM
