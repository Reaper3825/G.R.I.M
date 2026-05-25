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
#include "../../Layers/ReasoningHead/reasoning_head_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../TensorContract/ForwardIntermediates.hpp"
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
        return "autograd_connected";
    }
    if (!graph.connect_parameter_graph && !graph.retain_backward_graph) {
        return "read_only";
    }
    throw std::runtime_error("ModelForward: invalid graph policy — connect_parameter_graph and retain_backward_graph must agree at this boundary");
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
}

void executeModelForward(const ModelForwardRequest& request,
                         ModelForwardRuntimePayload& runtime_payload) {
    request.validate("executeModelForward");
    runtime_payload.validate(
        "executeModelForward",
        request.execution_block && request.config->execution_block_enabled);

    const auto* cfg = request.config;
    auto& runtime = runtime_payload;
    auto& intermediates = *runtime.autograd_intermediates;
    const auto& payload = *request.payload;
    const auto* bindings = request.bindings;
    const bool connect_parameter_graph = request.graph.connect_parameter_graph;
    const bool retain_backward_graph = request.graph.retain_backward_graph;
    const bool dropout_enabled = request.graph.enable_dropout;

    if (cfg->center_encoder_residuals || cfg->lm_head_center_hidden_states) {
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
                                + std::to_string(cfg->vocab_size) + ", d_model=" + std::to_string(cfg->d_model) + "]");
    }

    const float embedding_scale = 1.0f;
    Tensor emb_output = autograd::embedding(
        *emb_weights,
        token_ids,
        total_tokens,
        request.stream,
        embedding_scale);

    MFWD_INFO("Step 1b: No position embeddings (using "
              << HyperParameters::positionalEncodingTypeToString(cfg->positional_encoding)
              << " inside attention)");

    intermediates.embedding_tensor = std::move(emb_output);
    MFWD_INFO("Step 1: Token embedding complete, shape=[" << total_tokens << ", " << cfg->d_model << "]");

    if (request.scratch_block && request.scratch_block->isEnabled()) {
        MFWD_INFO("Step 1.5: Running all-token ScratchBlock vector gate...");
        (void)cudaGetLastError();

        if (!bindings->d_numeric_values) {
            throw std::runtime_error("executeModelForward: ScratchBlock requires BatchDeviceBindings.d_numeric_values");
        }
        if (!bindings->d_token_to_slot_map) {
            throw std::runtime_error("executeModelForward: ScratchBlock requires BatchDeviceBindings.d_token_to_slot_map");
        }

        const bool exec_first_type_only =
            cfg->execution_block_enabled && cfg->scratch_block_execution_first_type_only;
        if (cfg->execution_block_enabled && !exec_first_type_only) {
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

        intermediates.embedding_structured_state = autograd::scratch_block_project_all_tokens(
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

        intermediates.embedding_gate_concat = autograd::concat(
            intermediates.embedding_tensor,
            intermediates.embedding_structured_state,
            request.stream);

        if (connect_parameter_graph) {
            intermediates.embedding_gate_logits = autograd::matmul(
                intermediates.embedding_gate_concat,
                request.scratch_block->structuredGateWeight(),
                request.stream,
                intermediates.embedding_gate_concat.data,
                nullptr);
        } else {
            Tensor gate_weight_view = request.scratch_block->structuredGateWeight().detach(request.stream);
            intermediates.embedding_gate_logits = autograd::matmul(
                intermediates.embedding_gate_concat,
                gate_weight_view,
                request.stream,
                nullptr,
                nullptr);
        }

        intermediates.embedding_gate_values = autograd::sigmoid(
            intermediates.embedding_gate_logits,
            request.stream,
            intermediates.embedding_gate_logits.data);

        intermediates.embedding_gate_delta = autograd::elementwise_mul(
            intermediates.embedding_gate_values,
            intermediates.embedding_structured_state,
            request.stream);

        intermediates.embedding_tensor = autograd::add(
            intermediates.embedding_tensor,
            intermediates.embedding_gate_delta,
            request.stream);

        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("ModelForward: ScratchBlock vector gate CUDA error: " +
                                     std::string(cudaGetErrorString(cuda_err)));
        }

        MFWD_INFO("Step 1.5: ScratchBlock vector gate complete");
    }

    if (dropout_enabled && cfg->dropout_rate > 0.0f) {
        const uint64_t emb_dropout_seed = request.batch_idx * 2654435761ULL + 500;
        constexpr uint64_t kEmbeddingDropoutMaskStream = 0x0005000000000001ULL;
        intermediates.embedding_tensor = autograd::dropout(
            intermediates.embedding_tensor,
            cfg->dropout_rate,
            emb_dropout_seed,
            request.stream,
            kEmbeddingDropoutMaskStream);
        MFWD_INFO("Step 1c: Embedding-fusion dropout applied"
                  << " (p=" << cfg->dropout_rate << ", batch_idx=" << request.batch_idx << ")");
    }

    if (!request.gpu_encoder) {
        throw std::runtime_error("ModelForward: gpu_encoder is NULL - pass encoder in request");
    }

    const int num_layers = cfg->num_layers;
    intermediates.encoder_layer_outputs.clear();
    intermediates.layer_intermediates.layers.clear();
    intermediates.embedding_tensor.is_leaf = false;

    float* encoder_output = nullptr;

    if (!retain_backward_graph) {
        Tensor running;
        intermediates.layer_intermediates.layers.reserve(num_layers);
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

            intermediates.layer_intermediates.layers.emplace_back();
            ForwardIntermediates& layer_storage = intermediates.layer_intermediates.layers.back();

            Tensor& layer_input = (layer_idx == 0) ? intermediates.embedding_tensor : running;

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
                if (cfg->use_layer_scale) {
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
            Tensor layer_output_view = enc_layer->forward(
                layer_input, payload, request.stream, request.cublas_handle, layer_storage,
                request.batch_idx, false, layer_idx, enc_param_view_ptr);

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
        intermediates.encoder_output_tensor = std::move(running);
        intermediates.encoder_output_tensor.requires_grad = false;
        intermediates.encoder_output_tensor.grad_fn.reset();
        intermediates.encoder_output_tensor.stream = request.stream;
        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete (no_grad)");
    } else {
        intermediates.encoder_layer_outputs.reserve(num_layers);
        intermediates.layer_intermediates.layers.reserve(num_layers);

        MFWD_INFO("Step 2: Running " << num_layers << " encoder layers with retained graph...");
        MFWD_INFO("  embedding_tensor.grad_fn=" << (void*)intermediates.embedding_tensor.grad_fn.get()
                  << " requires_grad=" << intermediates.embedding_tensor.requires_grad);

        int exec_layer = -1;
        int exec_K = 0;
        if (request.execution_block && cfg->execution_block_enabled && request.scratch_block && request.scratch_block->isEnabled()) {
            exec_layer = cfg->execution_block_layer;
            if (exec_layer < 0) exec_layer = num_layers - 2;
            if (exec_layer < 0) exec_layer = 0;
            if (exec_layer >= num_layers) exec_layer = num_layers - 1;
            exec_K = cfg->execution_block_num_steps;
        }

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }

            intermediates.layer_intermediates.layers.emplace_back();
            ForwardIntermediates& layer_storage = intermediates.layer_intermediates.layers.back();

            Tensor& layer_input = (layer_idx == 0)
                ? intermediates.embedding_tensor
                : intermediates.encoder_layer_outputs.back();

            Tensor layer_output = enc_layer->forward(
                layer_input, payload, request.stream, request.cublas_handle, layer_storage,
                request.batch_idx, dropout_enabled, layer_idx, nullptr);

            if (layer_idx == exec_layer && request.execution_block) {
                const int ae = cfg->scratch_block_atom_embedding_dim;
                const int V = cfg->execution_block_num_slots;
                const int nop = cfg->execution_block_num_ops;
                const int dk = cfg->execution_block_d_key;
                const int dt = cfg->execution_block_d_type;

                intermediates.exec_memories.resize(payload.batch_size);
                intermediates.exec_outputs_per_row.resize(payload.batch_size);

                float T = cfg->execution_block_temp_start;

                auto& execution_trace_by_row = *runtime.execution_trace_by_row;
                auto& trace_state_by_row = *runtime.trace_state_by_row;
                execution_trace_by_row.resize(payload.batch_size);
                trace_state_by_row.resize(payload.batch_size);
                for (int b = 0; b < payload.batch_size; ++b) {
                    execution_trace_by_row[b].clear();
                    const bool row_active = !payload.execution_active.empty()
                        && payload.execution_active[b];
                    if (row_active) {
                        trace_state_by_row[b] = Tensor::zeros({1, cfg->d_model}, request.stream, "trace_state_row");
                        if (connect_parameter_graph) {
                            trace_state_by_row[b].requires_grad_();
                            trace_state_by_row[b].ensure_grad();
                        } else {
                            trace_state_by_row[b].requires_grad = false;
                        }
                    } else {
                        trace_state_by_row[b] = Tensor();
                    }
                }

                const bool have_exec_teacher = !payload.teacher_steps.empty();
                const int B_teacher = have_exec_teacher
                    ? static_cast<int>(payload.teacher_steps.size()) : 0;
                intermediates.exec_expected_target_tensors.clear();
                intermediates.exec_expected_target_tensors.reserve(
                    static_cast<size_t>(payload.batch_size) * static_cast<size_t>(std::max(0, exec_K)));

                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];

                    intermediates.exec_outputs_per_row[b].steps.clear();

                    if (!row_exec_active) continue;

                    auto& M_b = intermediates.exec_memories[b];
                    M_b.allocate(V, ae, cfg->d_model, dk, dt, request.stream);
                    M_b.clear(request.stream);

                    const int tok_off = b * payload.max_seq_len;

                    auto row_atom_view = request.scratch_block->extractRowLocalAtomView(
                        tok_off, payload.max_seq_len, request.stream);

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
                        payload.max_seq_len, request.stream);

                    for (int step = 0; step < exec_K; ++step) {
                        ExecutionBlockStepOutput step_diag;

                        const float* d_expected_target = nullptr;
                        TeacherSelectionTargets selection_targets;

                        if (have_exec_teacher && b < B_teacher) {
                            const auto& teacher_row = payload.teacher_steps[b];
                            if (step < static_cast<int>(teacher_row.size())) {
                                const auto& ts_k = teacher_row[step];
                                float h_val = ts_k.expected_value;
                                Tensor expected_target_tensor = Tensor::empty(
                                    TensorContract::TensorShape::make_BSM(1, 1),
                                    false,
                                    request.stream,
                                    "exec_expected_target_owned");
                                cudaMemcpyAsync(expected_target_tensor.data, &h_val,
                                                sizeof(float), cudaMemcpyHostToDevice, request.stream);
                                d_expected_target = expected_target_tensor.data;
                                intermediates.exec_expected_target_tensors.push_back(std::move(expected_target_tensor));

                                if (cfg->structured_ce_enabled) {
                                    const int S_scratch = 0;
                                    const int V_val = V - S_scratch;

                                    const int arg1_idx = ts_k.arg1_slot - S_scratch;
                                    const int arg2_idx = ts_k.arg2_slot - S_scratch;
                                    const int write_idx = ts_k.write_slot;

                                    if (ts_k.op_id < 0 || ts_k.op_id >= nop)
                                        throw std::runtime_error(
                                            "ModelForward: teacher op_id=" + std::to_string(ts_k.op_id)
                                            + " out of range [0," + std::to_string(nop) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));
                                    if (arg1_idx < 0 || arg1_idx >= V_val)
                                        throw std::runtime_error(
                                            "ModelForward: teacher arg1_slot=" + std::to_string(ts_k.arg1_slot)
                                            + " maps to index " + std::to_string(arg1_idx)
                                            + " out of range [0," + std::to_string(V_val) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));
                                    if (arg2_idx < 0 || arg2_idx >= V_val)
                                        throw std::runtime_error(
                                            "ModelForward: teacher arg2_slot=" + std::to_string(ts_k.arg2_slot)
                                            + " maps to index " + std::to_string(arg2_idx)
                                            + " out of range [0," + std::to_string(V_val) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));
                                    if (write_idx < 0 || write_idx >= V)
                                        throw std::runtime_error(
                                            "ModelForward: teacher write_slot=" + std::to_string(ts_k.write_slot)
                                            + " out of range [0," + std::to_string(V) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));

                                    selection_targets.op_target    = ts_k.op_id;
                                    selection_targets.arg1_target  = arg1_idx;
                                    selection_targets.arg2_target  = arg2_idx;
                                    selection_targets.write_target = write_idx;
                                    selection_targets.valid = true;

                                    const bool have_step_mask_here =
                                        (!payload.teacher_step_mask.empty()
                                         && b < static_cast<int>(payload.teacher_step_mask.size())
                                         && step < static_cast<int>(payload.teacher_step_mask[b].size()));
                                    if (have_step_mask_here && payload.teacher_step_mask[b][step] == 0) {
                                        selection_targets.valid = false;
                                    }
                                }
                            }
                        }

                        request.execution_block->executeStep(
                            layer_output, M_b,
                            reinterpret_cast<const int*>(row_atom_view.atom_positions.data),
                            row_atom_view.num_atoms, payload, *request.bindings, b,
                            step, T, request.stream,
                            &step_diag,
                            (*runtime.trace_state_by_row)[b],
                            (*runtime.execution_trace_by_row)[b],
                            d_expected_target,
                            cfg->structured_ce_enabled ? &selection_targets : nullptr);
                        (*runtime.execution_trace_by_row)[b].push_back(step_diag.record);
                        intermediates.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
                    }
                }
            }

            if (exec_layer >= 0 && layer_idx >= exec_layer
                && request.execution_block
                && !intermediates.exec_memories.empty()) {
                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];
                    if (!row_exec_active) continue;
                    Tensor row_delta = request.execution_block->crossAttentionRead(
                        layer_output, intermediates.exec_memories[b],
                        total_tokens, request.stream,
                        b * payload.max_seq_len, payload.max_seq_len,
                        runtime.read_gate_accum_tensor
                            ? runtime.read_gate_accum_tensor->data
                            : nullptr);
                    Tensor padded = autograd::zero_pad(row_delta, b * payload.max_seq_len, total_tokens, request.stream);
                    layer_output = autograd::add(layer_output, padded, request.stream);
                }
            }

            if (cfg->center_encoder_residuals) {
                if (payload.max_seq_len <= 1) {
                    throw std::runtime_error("ModelForward: center_encoder_residuals requires payload.max_seq_len > 1; single-row column centering would erase the residual stream");
                }
                layer_output = autograd::center_columns_by_sequence_lengths(
                    layer_output, payload.seq_lengths, payload.batch_size, payload.max_seq_len, request.stream);
            }

            intermediates.encoder_layer_outputs.push_back(std::move(layer_output));
        }

        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete");
        encoder_output = intermediates.encoder_layer_outputs.back().data;
    }

    if (retain_backward_graph) {
        const bool lmhead_track_grad = true;
        Tensor encoder_output_tensor = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,
            lmhead_track_grad,
            "encoder_output_for_lmhead");
        encoder_output_tensor.is_leaf = false;
        encoder_output_tensor.stream = request.stream;
        encoder_output_tensor.grad_fn = intermediates.encoder_layer_outputs.back().grad_fn;
        intermediates.encoder_output_tensor = std::move(encoder_output_tensor);
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

    Tensor logits_tensor = request.lm_head->forward(
        intermediates.encoder_output_tensor,
        intermediates.centered_encoder_output,
        payload,
        request.stream,
        request.cublas_handle,
        lm_head_parameter_view_ptr);
    if (!logits_tensor.data) {
        throw std::runtime_error("ModelForward: LMHeadLayer::forward returned logits tensor with NULL data");
    }

    const float* lm_input_ptr = nullptr;
    if (intermediates.centered_encoder_output.data) {
        lm_input_ptr = intermediates.centered_encoder_output.data;
    } else if (intermediates.encoder_output_tensor.data) {
        lm_input_ptr = intermediates.encoder_output_tensor.data;
    }
    if (!lm_input_ptr) {
        throw std::runtime_error("ModelForward: LM-head input snapshot is NULL after LMHeadLayer::forward");
    }

    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        constexpr int kSamplePositions = 1024;
        const int d_model = cfg->d_model;
        const int vocab_size_local = cfg->vocab_size;

        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_logits(sample_size * vocab_size_local);

        cudaMemcpyAsync(h_encoder.data(), lm_input_ptr,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaMemcpyAsync(h_logits.data(), logits_tensor.data,
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

    intermediates.logits_tensor = std::move(logits_tensor);

    if (request.reasoning_head && request.scratch_block && request.scratch_block->isEnabled()
        && !cfg->execution_block_enabled) {
        int num_atoms = 0;
        cudaMemcpyAsync(&num_atoms, request.scratch_block->numAtomsBuffer(),
                        sizeof(int), cudaMemcpyDeviceToHost, request.stream);
        cudaStreamSynchronize(request.stream);

        if (num_atoms > 0) {
            const int atom_dim = cfg->scratch_block_atom_embedding_dim;

            intermediates.scratch_atom_embeddings = Tensor::empty(
                TensorContract::TensorShape::make_BSM(num_atoms, atom_dim),
                connect_parameter_graph,
                request.stream,
                "scratch_atom_embeddings");
            cudaMemcpyAsync(
                intermediates.scratch_atom_embeddings.data,
                request.scratch_block->atomEmbeddingsBuffer(),
                static_cast<size_t>(num_atoms) * atom_dim * sizeof(float),
                cudaMemcpyDeviceToDevice,
                request.stream);

            ReasoningHeadParameterViews reasoning_parameter_views{};
            const ReasoningHeadParameterViews* reasoning_parameter_view_ptr = nullptr;
            Tensor reasoning_w_op_view;
            Tensor reasoning_b_op_view;
            Tensor reasoning_w_arg1_view;
            Tensor reasoning_w_arg2_view;
            if (!connect_parameter_graph) {
                reasoning_w_op_view = request.reasoning_head->W_op().detach(request.stream);
                reasoning_b_op_view = request.reasoning_head->b_op().detach(request.stream);
                reasoning_w_arg1_view = request.reasoning_head->w_arg1().detach(request.stream);
                reasoning_w_arg2_view = request.reasoning_head->w_arg2().detach(request.stream);
                reasoning_parameter_views.w_op = &reasoning_w_op_view;
                reasoning_parameter_views.b_op = &reasoning_b_op_view;
                reasoning_parameter_views.w_arg1 = &reasoning_w_arg1_view;
                reasoning_parameter_views.w_arg2 = &reasoning_w_arg2_view;
                reasoning_parameter_view_ptr = &reasoning_parameter_views;
            }

            ReasoningHeadOutput rh_out = request.reasoning_head->forward(
                intermediates.encoder_output_tensor,
                intermediates.scratch_atom_embeddings,
                request.scratch_block->atomPositionsBuffer(),
                num_atoms,
                total_tokens,
                request.stream,
                request.cublas_handle,
                reasoning_parameter_view_ptr);
            intermediates.reasoning_output = std::move(rh_out);
        }
    }

    MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << cfg->vocab_size << "]");
}

}  // namespace Forward
}  // namespace GRIM
