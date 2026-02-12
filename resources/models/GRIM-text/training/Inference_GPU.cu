//======================================================//
//  Inference_GPU.cu
//  Inference using autograd forward pass
//  
//  Rule 20: No backwards compatibility - uses autograd only
//======================================================//

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "Autograd/AutogradTraining.hpp"

namespace GRIM {

//======================================================//
//  executeInferenceForward - Single forward pass for inference
//  Returns logits for the last token (for generation)
//======================================================//
static Vector executeInferenceForward(
    LanguageModel& model,
    const int* d_token_ids,
    int seq_len,
    cudaStream_t stream
) {
    auto& ts = model.getTrainingState();
    const auto& cfg = model.getConfig();
    
    // Initialize autograd context
    Autograd::AutogradContext ctx = Autograd::initAutogradContext(
        &cfg,
        &ts,
        &model.getGpuEncoder(),
        model.getScratchBlockLayer(),
        ts.cublas_handle,
        stream,
        1,          // batch_size = 1 for inference
        seq_len,
        1.0f,       // grad_scale (unused for inference)
        0,          // step
        false       // is_training (disable dropout)
    );
    
    // Copy token IDs to cached buffer - Rule 20: use Tensor .data accessor
    cudaMemcpyAsync(reinterpret_cast<int*>(ts.cached_token_ids_tensor.data), d_token_ids,
                    seq_len * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Run autograd forward
    Autograd::ForwardResult result = Autograd::executeAutogradForward(ctx);
    
    if (!result.success) {
        throw std::runtime_error("Inference forward failed: " + result.error_message);
    }
    
    // Extract last token logits
    Vector logits(cfg.vocab_size);
    const size_t last_token_offset = static_cast<size_t>(seq_len - 1) * cfg.vocab_size;
    cudaMemcpyAsync(logits.data.data(),
                    ts.cached_logits_tensor.data + last_token_offset,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return logits;
}

//======================================================//
//  forwardInit - Prefill phase using autograd
//======================================================//
Vector LanguageModel::forwardInit(const std::vector<int>& prompt_tokens,
                                  const std::vector<float>& prompt_numeric_values,
                                  const std::vector<uint8_t>& prompt_numeric_mask) {
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
    
    // Copy tokens to device - Rule 20: use Tensor .data accessor
    cudaMemcpyAsync(reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data),
                    prompt_tokens.data(),
                    seq_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Copy numeric side-channel - Rule 20: use Tensor .data accessor
    if (!prompt_numeric_values.empty()) {
        cudaMemcpyAsync(training_state_.cached_token_numeric_values.data,
                        prompt_numeric_values.data(),
                        seq_len * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    if (!prompt_numeric_mask.empty()) {
        cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_numeric_mask.data),
                        prompt_numeric_mask.data(),
                        seq_len * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, stream);
    }
    
    // Store sequence length for subsequent steps
    training_state_.kv_cache_len = seq_len;
    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;
    
    // Run forward and return last token logits - Rule 20: use Tensor .data accessor
    return executeInferenceForward(*this, reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data), seq_len, stream);
}

//======================================================//
//  forwardStep - Decode phase (recompute full sequence)
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t numeric_mask) {
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
    
    // Append new token - Rule 20: use Tensor .data accessor with cast for int*
    int* token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
    cudaMemcpyAsync(token_ids_ptr + new_pos,
                    &new_token, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Append numeric side-channel - Rule 20: use Tensor .data accessor
    cudaMemcpyAsync(training_state_.cached_token_numeric_values.data + new_pos,
                    &numeric_value, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_numeric_mask.data) + new_pos,
                    &numeric_mask, sizeof(uint8_t),
                    cudaMemcpyHostToDevice, stream);

    training_state_.kv_cache_len = new_seq_len;
    training_state_.cached_seq_len = new_seq_len;
    
    // Recompute full forward pass - Rule 20: use Tensor .data accessor
    return executeInferenceForward(*this, token_ids_ptr, new_seq_len, stream);
}

//======================================================//
//  forwardStepIncremental - Same as forwardStep (no KV cache optimization)
//======================================================//
Vector LanguageModel::forwardStepIncremental(int new_token, float numeric_value, uint8_t numeric_mask) {
    // Without legacy KV cache system, this is identical to forwardStep
    return forwardStep(new_token, numeric_value, numeric_mask);
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

//======================================================//
//  forwardWithCache - Full sequence forward, return hidden states
//======================================================//
Vector LanguageModel::forwardWithCache(const std::vector<int>& token_ids,
                                       const std::vector<float>& token_numeric_values,
                                       const std::vector<uint8_t>& token_numeric_mask,
                                       bool tokens_on_device,
                                       const std::vector<uint16_t>& token_text_features,
                                       const std::vector<uint8_t>& token_text_mask) {
    if (!training_state_.initialized) {
        throw std::runtime_error("forwardWithCache: training state not initialized");
    }

    const int seq_len = static_cast<int>(token_ids.size());
    if (seq_len <= 0) {
        return Vector();
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Copy tokens to device (if not already there)
    if (!tokens_on_device) {
        cudaMemcpyAsync(reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data),
                        token_ids.data(),
                        seq_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }
    
    // Copy numeric side-channel
    if (!token_numeric_values.empty()) {
        cudaMemcpyAsync(training_state_.cached_token_numeric_values.data,
                        token_numeric_values.data(),
                        seq_len * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    if (!token_numeric_mask.empty()) {
        cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_numeric_mask.data),
                        token_numeric_mask.data(),
                        seq_len * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, stream);
    }
    
    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;
    
    // Initialize autograd context
    Autograd::AutogradContext ctx = Autograd::initAutogradContext(
        &config_,
        &training_state_,
        &getGpuEncoder(),
        getScratchBlockLayer(),
        training_state_.cublas_handle,
        stream,
        1,          // batch_size
        seq_len,
        1.0f,       // grad_scale
        0,          // step
        false       // is_training (disable dropout)
    );
    
    // Run autograd forward
    Autograd::ForwardResult result = Autograd::executeAutogradForward(ctx);
    
    if (!result.success) {
        throw std::runtime_error("forwardWithCache failed: " + result.error_message);
    }
    
    // Return last token's hidden state (encoder output before LM head)
    Vector last_hidden(config_.d_model);
    const size_t last_token_offset = static_cast<size_t>(seq_len - 1) * config_.d_model;
    cudaMemcpyAsync(last_hidden.data.data(),
                    training_state_.cached_encoder_output.data + last_token_offset,
                    config_.d_model * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return last_hidden;
}

} // namespace GRIM
