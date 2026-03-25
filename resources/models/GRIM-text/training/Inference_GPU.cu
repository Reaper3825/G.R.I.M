//======================================================//
//  Inference_GPU.cu
//  Inference using autograd forward pass
//  
//  Single inference entry point: executeInferenceForward_()
//  All public inference methods copy data to cached_* tensors
//  then call this private method.
//
//  Rule 20: No backwards compatibility - uses autograd only
//  Rule 26: One inference path, not two
//======================================================//

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "Autograd/AutogradTraining.hpp"

namespace GRIM {

namespace {

void copyTokenSlotMapH2D(TrainingState& ts, cudaStream_t stream, int seq_len,
                         const std::vector<int32_t>& prompt_map) {
    if (!ts.cached_token_to_slot_map.data || seq_len <= 0)
        return;
    auto* dst = reinterpret_cast<int32_t*>(ts.cached_token_to_slot_map.data);
    if (prompt_map.empty()) {
        std::vector<int32_t> neg(static_cast<size_t>(seq_len), -1);
        cudaError_t err = cudaMemcpyAsync(dst, neg.data(),
            static_cast<size_t>(seq_len) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("copyTokenSlotMapH2D (sentinel): ") +
                                     cudaGetErrorString(err));
        }
        return;
    }
    if (static_cast<int>(prompt_map.size()) != seq_len) {
        throw std::runtime_error(
            "token_to_slot_map size must match sequence length (or pass empty for all -1)");
    }
    cudaError_t err = cudaMemcpyAsync(dst, prompt_map.data(),
        static_cast<size_t>(seq_len) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("copyTokenSlotMapH2D: ") + cudaGetErrorString(err));
    }
}

}  // namespace

//======================================================//
//  executeInferenceForward_ - THE single inference forward path
//  Assumes all data already in cached_* tensors.
//  Creates autograd context, runs forward, returns last-token logits.
//======================================================//
Vector LanguageModel::executeInferenceForward_(int seq_len) {
    if (!training_state_.initialized) {
        throw std::runtime_error("executeInferenceForward_: training state not initialized");
    }
    if (seq_len <= 0 || seq_len > config_.max_seq_len) {
        throw std::runtime_error("executeInferenceForward_: seq_len=" + std::to_string(seq_len) +
                                 " out of range [1, " + std::to_string(config_.max_seq_len) + "]");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    // Store dimensions for downstream consumers
    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;

    // Initialize autograd context (is_training=false disables dropout)
    Autograd::AutogradContext ctx = Autograd::initAutogradContext(
        &config_,
        &training_state_,
        &getGpuEncoder(),
        getEmbeddingLayer(),
        getLmHeadLayer(),
        getScratchBlockLayer(),
        getReasoningHeadLayer(),
        getExecutionBlockLayer(),
        training_state_.cublas_handle,
        stream,
        1,          // batch_size = 1 for inference
        seq_len,
        1.0f,       // grad_scale (unused for inference)
        0,          // step
        false       // is_training (disable dropout)
    );

    // Run autograd forward
    Autograd::ForwardResult result = Autograd::executeAutogradForward(ctx);
    if (!result.success) {
        throw std::runtime_error("executeInferenceForward_: forward failed - " + result.error_message);
    }

    // Extract last token logits
    if (!training_state_.cached_logits_tensor.data) {
        throw std::runtime_error("executeInferenceForward_: cached_logits_tensor not initialized");
    }
    Vector logits(config_.vocab_size);
    const size_t last_token_offset = static_cast<size_t>(seq_len - 1) * config_.vocab_size;
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.cached_logits_tensor.data + last_token_offset,
                    config_.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Persist ExecutionMemory for Step Z <NUM> slot binding during generation.
    if (!training_state_.autograd_intermediates.exec_memories.empty()) {
        training_state_.inference_exec_memory =
            std::move(training_state_.autograd_intermediates.exec_memories[0]);
        training_state_.has_inference_exec_memory = true;
        // Track last write slot from the final execution step
        const auto& row_out = training_state_.autograd_intermediates.exec_outputs_per_row;
        if (!row_out.empty() && !row_out[0].steps.empty()) {
            const auto& last_step = row_out[0].steps.back();
            const int V = config_.execution_block_num_slots;
            if (last_step.p_write.data && V > 0) {
                std::vector<float> h_pw(V);
                cudaMemcpyAsync(h_pw.data(), last_step.p_write.data,
                                V * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                int best = 0;
                for (int i = 1; i < V; ++i)
                    if (h_pw[i] > h_pw[best]) best = i;
                training_state_.inference_exec_last_write_slot = best;
            }
        }
    }

    // Free all autograd intermediates — inference never runs backward.
    // Without this, grad_fn chains and cached tensors leak across generation steps.
    training_state_.autograd_intermediates.clear();

    return logits;
}

//======================================================//
//  forwardInit - Prefill phase: copy prompt to device, run forward
//======================================================//
Vector LanguageModel::forwardInit(const std::vector<int>& prompt_tokens,
                                  const std::vector<float>& prompt_numeric_values,
                                  const std::vector<uint8_t>& prompt_atom_mask,
                                  const std::vector<int32_t>& prompt_token_to_slot_map) {
    const int seq_len = static_cast<int>(prompt_tokens.size());
    
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
    }

    if (seq_len <= 0) {
        throw std::runtime_error("forwardInit: seq_len <= 0");
    }
    if (seq_len > config_.max_seq_len) {
        throw std::runtime_error("forwardInit: seq_len exceeds max_seq_len");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Copy tokens to device
    cudaMemcpyAsync(reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data),
                    prompt_tokens.data(),
                    seq_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Copy numeric side-channel
    if (!prompt_numeric_values.empty()) {
        cudaMemcpyAsync(training_state_.cached_token_numeric_values.data,
                        prompt_numeric_values.data(),
                        seq_len * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    if (!prompt_atom_mask.empty()) {
        cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data),
                        prompt_atom_mask.data(),
                        seq_len * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, stream);
    }

    copyTokenSlotMapH2D(training_state_, stream, seq_len, prompt_token_to_slot_map);
    
    // Store sequence length for subsequent forwardStep() calls
    training_state_.kv_cache_len = seq_len;
    
    return executeInferenceForward_(seq_len);
}

//======================================================//
//  forwardStep - Decode phase: append token, recompute full sequence
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t atom_mask,
                                  int32_t new_token_slot_id) {
    if (!training_state_.initialized) {
        throw std::runtime_error("forwardStep: training state not initialized");
    }
    if (training_state_.kv_cache_len == 0) {
        throw std::runtime_error("forwardStep: call forwardInit first");
    }

    const int new_pos = training_state_.kv_cache_len;
    const int new_seq_len = new_pos + 1;
    
    if (new_seq_len > config_.max_seq_len) {
        throw std::runtime_error("forwardStep: sequence exceeds max_seq_len");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Append new token
    int* token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
    cudaMemcpyAsync(token_ids_ptr + new_pos,
                    &new_token, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Append numeric side-channel
    cudaMemcpyAsync(training_state_.cached_token_numeric_values.data + new_pos,
                    &numeric_value, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data) + new_pos,
                    &atom_mask, sizeof(uint8_t),
                    cudaMemcpyHostToDevice, stream);

    if (training_state_.cached_token_to_slot_map.data) {
        auto* slot_dst = reinterpret_cast<int32_t*>(training_state_.cached_token_to_slot_map.data) + new_pos;
        cudaError_t serr = cudaMemcpyAsync(slot_dst, &new_token_slot_id, sizeof(int32_t),
                                           cudaMemcpyHostToDevice, stream);
        if (serr != cudaSuccess) {
            throw std::runtime_error(std::string("forwardStep slot map: ") + cudaGetErrorString(serr));
        }
    }

    training_state_.kv_cache_len = new_seq_len;
    
    return executeInferenceForward_(new_seq_len);
}

//======================================================//
//  resetKVCache - Clear sequence state
//======================================================//
void LanguageModel::resetKVCache() {
    if (training_state_.initialized) {
        training_state_.kv_cache_len = 0;
        training_state_.has_inference_exec_memory = false;
        training_state_.inference_exec_last_write_slot = -1;
    }
}

//======================================================//
//  getKVCacheLength - Query current sequence length
//======================================================//
int LanguageModel::getKVCacheLength() const {
    return training_state_.kv_cache_len;
}

//======================================================//
//  predictNumericValue — removed (execution-first spec; no value head)
//======================================================//
float LanguageModel::predictNumericValue() const {
    throw std::runtime_error(
        "predictNumericValue: NumericHead removed; numeric values must come from ExecutionBlock "
        "and slot binding (spec Step Z), not hidden-state regression");
}

} // namespace GRIM
