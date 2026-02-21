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
#include <cuda_runtime.h>
#include <stdexcept>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "Autograd/AutogradTraining.hpp"

namespace GRIM {

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

    return logits;
}

//======================================================//
//  forwardInit - Prefill phase: copy prompt to device, run forward
//======================================================//
Vector LanguageModel::forwardInit(const std::vector<int>& prompt_tokens,
                                  const std::vector<float>& prompt_numeric_values,
                                  const std::vector<uint8_t>& prompt_atom_mask) {
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
    
    // Store sequence length for subsequent forwardStep() calls
    training_state_.kv_cache_len = seq_len;
    
    return executeInferenceForward_(seq_len);
}

//======================================================//
//  forwardStep - Decode phase: append token, recompute full sequence
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t atom_mask) {
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

    training_state_.kv_cache_len = new_seq_len;
    
    return executeInferenceForward_(new_seq_len);
}

//======================================================//
//  resetKVCache - Clear sequence state
//======================================================//
void LanguageModel::resetKVCache() {
    if (training_state_.initialized) {
        training_state_.kv_cache_len = 0;
    }
}

//======================================================//
//  getKVCacheLength - Query current sequence length
//======================================================//
int LanguageModel::getKVCacheLength() const {
    return training_state_.kv_cache_len;
}

} // namespace GRIM
