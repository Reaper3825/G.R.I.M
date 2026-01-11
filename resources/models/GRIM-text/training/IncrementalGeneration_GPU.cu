//======================================================//
//  IncrementalGeneration_GPU.cu
//  Forward orchestration for KV-cached generation
//======================================================//

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/ForwardOps/ForwardOps_Logging.hpp"
#include "../Layers/ForwardOps/ForwardOps_Orchestrator.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"

namespace GRIM {

//======================================================//
//  resetKVCache - Clear KV cache for new generation
//======================================================//
void LanguageModel::resetKVCache() {
    if (!training_state_.initialized) {
        return;
    }
    training_state_.kv_cache_len = 0;
}

//======================================================//
//  getKVCacheLength - Query current cache state
//======================================================//
int LanguageModel::getKVCacheLength() const {
    return training_state_.kv_cache_len;
}

//======================================================//
//  forwardInit - Prefill phase
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
    if (prompt_numeric_values.size() != static_cast<size_t>(seq_len) ||
        prompt_numeric_mask.size() != static_cast<size_t>(seq_len)) {
        throw std::runtime_error("forwardInit: numeric arrays size mismatch with prompt_tokens");
    }
    if (seq_len > training_state_.kv_cache_capacity) {
        throw std::runtime_error("forwardInit: seq_len (" + std::to_string(seq_len) + 
                                 ") exceeds kv_cache_capacity (" + 
                                 std::to_string(training_state_.kv_cache_capacity) + ")");
    }
    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        throw std::runtime_error("forwardInit: numeric buffers not allocated");
    }

    training_state_.kv_cache_len = 0;
    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;
    training_state_.cached_num_layers = getConfig().num_layers;

    cudaMemcpyAsync(training_state_.cached_token_numeric_values,
                    prompt_numeric_values.data(),
                    static_cast<size_t>(seq_len) * sizeof(float),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());
    cudaMemcpyAsync(training_state_.cached_token_numeric_mask,
                    prompt_numeric_mask.data(),
                    static_cast<size_t>(seq_len) * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());

    auto ctx = GRIM::Forward::initForwardContext(
        *this,
        GRIM::Forward::ForwardMode::Prefill,
        1,
        seq_len,
        GRIM::Forward::ForwardLogitsTarget::LastToken,
        prompt_tokens.data(),
        false,
        -1,
        -1,
        false,
        false,
        false);

    const auto status = GRIM::Forward::executeForward(ctx);
    if (status != GRIM::Forward::ForwardStatus::SUCCESS) {
        throw std::runtime_error("forwardInit: forward failed - " + 
                                 std::string(GRIM::Forward::statusToString(status)) + 
                                 ": " + ctx.error_message);
    }

    training_state_.kv_cache_len = seq_len;

    const auto& cfg = getConfig();
    Vector logits(cfg.vocab_size);
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.single_token_logits,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    training_state_.stream_ctrl.getPrimaryStream());
    training_state_.stream_ctrl.syncPrimaryStream();
    
    return logits;
}

//======================================================//
//  forwardStep - Decode phase (full recompute)
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t numeric_mask) {
    if (!training_state_.initialized) {
        throw std::runtime_error("forwardStep: training state not initialized");
    }
    if (training_state_.kv_cache_len == 0) {
        throw std::runtime_error("forwardStep: KV cache empty - call forwardInit first");
    }

    const auto& cfg = getConfig();
    const int new_pos = training_state_.kv_cache_len;
    const int new_seq_len = new_pos + 1;
    
    if (new_seq_len > training_state_.kv_cache_capacity) {
        throw std::runtime_error("forwardStep: sequence length (" + std::to_string(new_seq_len) +
                                 ") exceeds kv_cache_capacity (" + 
                                 std::to_string(training_state_.kv_cache_capacity) + ")");
    }
    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        throw std::runtime_error("forwardStep: numeric buffers not allocated");
    }

    cudaMemcpyAsync(training_state_.cached_token_numeric_values + new_pos,
                    &numeric_value, sizeof(float),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());
    cudaMemcpyAsync(training_state_.cached_token_numeric_mask + new_pos,
                    &numeric_mask, sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());

    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = new_seq_len;
    training_state_.cached_num_layers = cfg.num_layers;

    auto ctx = GRIM::Forward::initForwardContext(
        *this,
        GRIM::Forward::ForwardMode::DecodeFull,
        1,
        new_seq_len,
        GRIM::Forward::ForwardLogitsTarget::LastToken,
        nullptr,
        true,
        new_token,
        new_pos,
        false,
        false,
        false);

    const auto status = GRIM::Forward::executeForward(ctx);
    if (status != GRIM::Forward::ForwardStatus::SUCCESS) {
        throw std::runtime_error("forwardStep: forward failed - " +
                                 std::string(GRIM::Forward::statusToString(status)) +
                                 ": " + ctx.error_message);
    }

    training_state_.kv_cache_len = new_seq_len;

    Vector logits(cfg.vocab_size);
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.single_token_logits,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    training_state_.stream_ctrl.getPrimaryStream());
    training_state_.stream_ctrl.syncPrimaryStream();
    return logits;
}

//======================================================//
//  forwardStepIncremental - Decode phase (incremental)
//======================================================//
Vector LanguageModel::forwardStepIncremental(int new_token, float numeric_value, uint8_t numeric_mask) {
    if (!training_state_.initialized) {
        throw std::runtime_error("forwardStepIncremental: training state not initialized");
    }
    if (training_state_.kv_cache_len == 0) {
        throw std::runtime_error("forwardStepIncremental: KV cache empty - call forwardInit first");
    }

    const auto& cfg = getConfig();
    const int query_pos = training_state_.kv_cache_len;
    const int kv_len = query_pos + 1;
    
    if (kv_len > training_state_.kv_cache_capacity) {
        throw std::runtime_error("forwardStepIncremental: kv_len (" + std::to_string(kv_len) +
                                 ") exceeds kv_cache_capacity (" + 
                                 std::to_string(training_state_.kv_cache_capacity) + ")");
    }
    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        throw std::runtime_error("forwardStepIncremental: numeric buffers not allocated");
    }

    cudaMemcpyAsync(training_state_.cached_token_numeric_values + query_pos,
                    &numeric_value, sizeof(float),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());
    cudaMemcpyAsync(training_state_.cached_token_numeric_mask + query_pos,
                    &numeric_mask, sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());

    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = kv_len;
    training_state_.cached_num_layers = cfg.num_layers;

    auto ctx = GRIM::Forward::initForwardContext(
        *this,
        GRIM::Forward::ForwardMode::DecodeIncremental,
        1,
        kv_len,
        GRIM::Forward::ForwardLogitsTarget::LastToken,
        nullptr,
        true,
        new_token,
        query_pos,
        false,
        false,
        false);

    const auto status = GRIM::Forward::executeForward(ctx);
    if (status != GRIM::Forward::ForwardStatus::SUCCESS) {
        throw std::runtime_error("forwardStepIncremental: forward failed - " +
                                 std::string(GRIM::Forward::statusToString(status)) +
                                 ": " + ctx.error_message);
    }

    training_state_.kv_cache_len = kv_len;

    Vector logits(cfg.vocab_size);
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.single_token_logits,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    training_state_.stream_ctrl.getPrimaryStream());
    training_state_.stream_ctrl.syncPrimaryStream();
    return logits;
}

} // namespace GRIM
