//======================================================//
//  InferenceForward_GPU.cu
//  Explicit inference-prefill forward boundary
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "InferenceForward_GPU.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../../Layers/ReasoningHead/reasoning_head_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../TensorContract/ForwardIntermediates.hpp"
#include "../TrainingState/TrainingState_GPU.hpp"
#include "../VerboseLogging.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Forward {

namespace {

#define IFWD_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[InferenceForward] INFO: " << msg << std::endl; \
    } \
} while(0)

Batching::BatchPayload makeInferenceGeometryPayload(const InferenceForwardRequest& request) {
    Batching::BatchPayload payload;
    payload.batch_size = request.batch_size;
    payload.max_seq_len = request.seq_len;
    payload.total_tokens = request.batch_size * request.seq_len;
    payload.actual_tokens = payload.total_tokens;
    payload.padding_tokens = 0;
    payload.valid_tokens = payload.total_tokens;
    payload.lm_valid_tokens = payload.total_tokens;
    payload.vocab_size = request.config->vocab_size;
    payload.seq_lengths.assign(static_cast<size_t>(request.batch_size), request.seq_len);
    payload.valid_target_counts.assign(static_cast<size_t>(request.batch_size), request.seq_len);
    payload.min_seq_len = request.seq_len;
    payload.packing_efficiency = 1.0f;
    payload.fits_in_cache = true;
    return payload;
}

void checkCuda(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

void InferenceForwardRequest::validate(const char* caller) const {
    if (!config) throw std::runtime_error(std::string(caller) + ": config is NULL");
    if (!runtime_state) throw std::runtime_error(std::string(caller) + ": runtime_state is NULL");
    if (!gpu_encoder) throw std::runtime_error(std::string(caller) + ": gpu_encoder is NULL");
    if (!embedding_layer) throw std::runtime_error(std::string(caller) + ": embedding_layer is NULL");
    if (!lm_head) throw std::runtime_error(std::string(caller) + ": lm_head is NULL");
    if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
    if (!stream) throw std::runtime_error(std::string(caller) + ": stream is NULL");
    if (!bindings) throw std::runtime_error(std::string(caller) + ": bindings is NULL");
    if (batch_size <= 0) throw std::runtime_error(std::string(caller) + ": batch_size <= 0");
    if (seq_len <= 0) throw std::runtime_error(std::string(caller) + ": seq_len <= 0");
    if (bindings->batch_size != batch_size || bindings->max_seq_len != seq_len) {
        throw std::runtime_error(
            std::string(caller) + ": BatchDeviceBindings geometry (" +
            std::to_string(bindings->batch_size) + "x" + std::to_string(bindings->max_seq_len) +
            ") does not match request (" + std::to_string(batch_size) + "x" +
            std::to_string(seq_len) + ")");
    }
    if (!bindings->d_input_ids) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings.d_input_ids is NULL");
    }
}

InferenceForwardResult executeInferencePrefillForward(InferenceForwardRequest& request) {
    request.validate("executeInferencePrefillForward");

    InferenceForwardResult result{};
    result.success = false;

    auto* ts = request.runtime_state;
    const auto* cfg = request.config;
    auto& intermediates = ts->autograd_intermediates;
    const auto* bindings = request.bindings;
    Batching::BatchPayload payload = makeInferenceGeometryPayload(request);

    const int total_tokens = payload.total_tokens;
    result.total_tokens = total_tokens;
    result.vocab_size = payload.vocab_size;

    IFWD_INFO("prefill forward: batch=" << request.batch_size
              << " seq=" << request.seq_len
              << " tokens=" << total_tokens
              << " vocab=" << payload.vocab_size);

    autograd::set_autograd_cublas_handle(request.cublas_handle);

    int* token_ids = bindings->d_input_ids;
    if (!token_ids) {
        throw std::runtime_error("InferenceForward: input token device pointer is NULL");
    }

    Tensor& emb_weights = request.embedding_layer->tokenWeights();
    if (!emb_weights.data) {
        throw std::runtime_error("InferenceForward: embedding token_weights.data is NULL");
    }
    emb_weights.requires_grad = false;
    if (!emb_weights.shape.is_valid()) {
        throw std::runtime_error("InferenceForward: embedding token_weights.shape is INVALID - EmbeddingLayer MUST initialize with correct shape [vocab_size="
                                + std::to_string(cfg->vocab_size) + ", d_model=" + std::to_string(cfg->d_model) + "]");
    }

    const float embedding_scale = 1.0f;
    Tensor emb_output = autograd::embedding(
        emb_weights,
        token_ids,
        total_tokens,
        request.stream,
        embedding_scale);

    intermediates.embedding_tensor = std::move(emb_output);
    if (cfg->dropout_rate > 0.0f) {
        const uint64_t emb_dropout_seed = request.step * 2654435761ULL + 500;
        intermediates.embedding_tensor = autograd::dropout(
            intermediates.embedding_tensor,
            cfg->dropout_rate,
            emb_dropout_seed,
            false,
            request.stream);
    }

    if (request.scratch_block && request.scratch_block->isEnabled()) {
        (void)cudaGetLastError();

        if (!bindings->d_numeric_values) {
            throw std::runtime_error("executeInferencePrefillForward: ScratchBlock requires BatchDeviceBindings.d_numeric_values");
        }
        if (!bindings->d_token_to_slot_map) {
            throw std::runtime_error("executeInferencePrefillForward: ScratchBlock requires BatchDeviceBindings.d_token_to_slot_map");
        }

        const bool exec_first_type_only =
            cfg->execution_block_enabled && cfg->scratch_block_execution_first_type_only;
        if (cfg->execution_block_enabled && !exec_first_type_only) {
            throw std::runtime_error(
                "executeInferencePrefillForward: execution_block_enabled requires "
                "scratch_block_execution_first_type_only=true to prevent value leakage "
                "into hidden states on arithmetic batches");
        }

        intermediates.embedding_tensor = autograd::scratch_block_inject(
            intermediates.embedding_tensor,
            *request.scratch_block,
            token_ids,
            bindings->d_numeric_values,
            bindings->d_text_features,
            bindings->d_atom_mask,
            bindings->d_atom_flags,
            bindings->d_token_to_slot_map,
            total_tokens,
            request.stream,
            exec_first_type_only);

        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("InferenceForward: ScratchBlock CUDA error: " +
                                     std::string(cudaGetErrorString(cuda_err)));
        }
    }

    if (!request.gpu_encoder) {
        throw std::runtime_error("InferenceForward: gpu_encoder is NULL - pass encoder in request");
    }

    const int num_layers = request.gpu_encoder->getNumLayers();
    intermediates.encoder_layer_outputs.clear();
    intermediates.layer_intermediates.layers.clear();
    intermediates.embedding_tensor.is_leaf = false;

    ForwardIntermediates no_grad_layer_storage;
    Tensor running;
    float* encoder_output = nullptr;

    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
        if (!enc_layer) {
            throw std::runtime_error("InferenceForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
        }

        if (layer_idx > 0) {
            checkCuda(cudaStreamSynchronize(request.stream),
                      ("InferenceForward: sync after layer " + std::to_string(layer_idx - 1)).c_str());
        }

        no_grad_layer_storage.clear();
        Tensor& layer_input = (layer_idx == 0) ? intermediates.embedding_tensor : running;
        running = enc_layer->forward(layer_input, payload, request.stream, no_grad_layer_storage, request.step, layer_idx);

        Tensor owned = Tensor::empty(running.shape, false, request.stream, "inference_no_grad_layer_output");
        const size_t bytes = static_cast<size_t>(running.shape.total_elements()) * sizeof(float);
        checkCuda(cudaMemcpyAsync(owned.data, running.data, bytes, cudaMemcpyDeviceToDevice, request.stream),
                  "InferenceForward: copy layer output");
        checkCuda(cudaStreamSynchronize(request.stream),
                  "InferenceForward: sync after layer output copy");
        running = std::move(owned);
    }

    checkCuda(cudaStreamSynchronize(request.stream),
              "InferenceForward: sync after encoder layers");

    encoder_output = running.data;
    intermediates.encoder_output_tensor = std::move(running);
    intermediates.encoder_output_tensor.requires_grad = false;
    intermediates.encoder_output_tensor.grad_fn.reset();
    intermediates.encoder_output_tensor.stream = request.stream;
    result.encoder_output = encoder_output;

    if (ts->cached_encoder_output.data) {
        checkCuda(cudaMemcpyAsync(ts->cached_encoder_output.data, encoder_output,
                                  static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                                  cudaMemcpyDeviceToDevice, request.stream),
                  "InferenceForward: cached_encoder_output copy");
    }

    request.lm_head->setStream(request.stream);
    request.lm_head->setCublasHandle(request.cublas_handle);

    float* logits_output = ts->cached_logits_tensor.data;
    if (!logits_output) {
        throw std::runtime_error("InferenceForward: cached_logits_tensor buffer is NULL - runtime state MUST allocate logits buffer");
    }

    Tensor logits_tensor = request.lm_head->forward(
        intermediates.encoder_output_tensor,
        intermediates.centered_encoder_output);

    if (cfg->lm_head_center_hidden_states && ts->cached_encoder_output.data && intermediates.centered_encoder_output.data) {
        checkCuda(cudaMemcpyAsync(ts->cached_encoder_output.data, intermediates.centered_encoder_output.data,
                                  static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                                  cudaMemcpyDeviceToDevice, request.stream),
                  "InferenceForward: centered cached_encoder_output copy");
    }

    checkCuda(cudaMemcpyAsync(logits_output, logits_tensor.data,
                              logits_tensor.shape.total_elements() * sizeof(float),
                              cudaMemcpyDeviceToDevice, request.stream),
              "InferenceForward: cached logits copy");

    intermediates.logits_tensor = std::move(logits_tensor);

    if (request.reasoning_head && request.scratch_block && request.scratch_block->isEnabled()
        && !cfg->execution_block_enabled) {
        int num_atoms = 0;
        checkCuda(cudaMemcpyAsync(&num_atoms, request.scratch_block->numAtomsBuffer(),
                                  sizeof(int), cudaMemcpyDeviceToHost, request.stream),
                  "InferenceForward: copy num_atoms");
        checkCuda(cudaStreamSynchronize(request.stream),
                  "InferenceForward: sync num_atoms");

        if (num_atoms > 0) {
            const int atom_dim = cfg->scratch_block_atom_embedding_dim;
            intermediates.scratch_atom_embeddings = Tensor::empty(
                TensorContract::TensorShape::make_BSM(num_atoms, atom_dim),
                false,
                request.stream,
                "inference_scratch_atom_embeddings");
            checkCuda(cudaMemcpyAsync(
                          intermediates.scratch_atom_embeddings.data,
                          request.scratch_block->atomEmbeddingsBuffer(),
                          static_cast<size_t>(num_atoms) * atom_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice,
                          request.stream),
                      "InferenceForward: copy atom embeddings");

            request.reasoning_head->setStream(request.stream);
            request.reasoning_head->setCublasHandle(request.cublas_handle);

            ReasoningHeadOutput rh_out = request.reasoning_head->forward(
                intermediates.encoder_output_tensor,
                intermediates.scratch_atom_embeddings,
                request.scratch_block->atomPositionsBuffer(),
                num_atoms,
                total_tokens,
                request.stream);
            intermediates.reasoning_output = std::move(rh_out);
        }
    }

    result.success = true;
    return result;
}

}  // namespace Forward
}  // namespace GRIM
