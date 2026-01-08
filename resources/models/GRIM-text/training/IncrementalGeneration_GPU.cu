//======================================================//
//  IncrementalGeneration_GPU.cu
//  Forward orchestration for KV-cached generation
//======================================================//

#define USE_CUDA

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
    {
        std::ostringstream oss;
        oss << "[forwardInit] ENTRY: seq_len=" << seq_len << " initialized=" 
            << training_state_.initialized;
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    if (!training_state_.initialized) {
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, 
                                      "[forwardInit] State not initialized, initializing...");
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, 
                                      "[forwardInit] Initialization complete");
    }

    GRIM::ForwardOps::LogUnexpectedGradState(training_state_, "forwardInit");
    const auto prefix = GRIM::ForwardOps::BuildForwardPrefix(training_state_, "forwardInit");

    const auto& cfg = getConfig();
    {
        std::ostringstream oss;
        oss << "[forwardInit] Config check: max_seq_len=" << cfg.max_seq_len 
            << " kv_cache_capacity=" << training_state_.kv_cache_capacity;
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    if (seq_len <= 0) {
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Inference, 
                                       "[forwardInit] ERROR: seq_len <= 0, returning empty");
        return Vector();
    }
    if (prompt_numeric_values.size() != static_cast<size_t>(seq_len) ||
        prompt_numeric_mask.size() != static_cast<size_t>(seq_len)) {
        std::ostringstream oss;
        oss << "[forwardInit] ERROR: numeric mismatch - values.size=" 
            << prompt_numeric_values.size() << " mask.size=" << prompt_numeric_mask.size() 
            << " expected=" << seq_len;
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Inference, oss.str());
        return Vector();
    }

    {
        std::ostringstream oss;
        oss << "[forwardInit] kv_cache_capacity=" << training_state_.kv_cache_capacity 
            << " kv_cache_len=" << training_state_.kv_cache_len;
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    if (seq_len > training_state_.kv_cache_capacity) {
        std::ostringstream oss;
        oss << "[forwardInit] ERROR: seq_len > kv_cache_capacity (" << seq_len 
            << " > " << training_state_.kv_cache_capacity << "), returning empty";
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Inference, oss.str());
        return Vector();
    }

    GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, 
                                  "[forwardInit] Resetting KV cache state...");
    training_state_.kv_cache_len = 0;
    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;
    training_state_.cached_num_layers = cfg.num_layers;

    {
        std::ostringstream oss;
        oss << "[forwardInit] Checking numeric buffers: values=" << (void*)training_state_.cached_token_numeric_values
            << " mask=" << (void*)training_state_.cached_token_numeric_mask;
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Inference, 
                                       "[forwardInit] ERROR: numeric buffers NULL, returning empty");
        return Vector();
    }
    
    GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, 
                                  "[forwardInit] Copying numeric data to device...");
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

    {
        std::ostringstream oss;
        oss << "[forwardInit] Initializing forward context: mode=Prefill batch=1 seq=" << seq_len
            << " logits_target=LastToken";
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
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

    GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, 
                                  "[forwardInit] Executing forward pass...");
    const auto status = GRIM::Forward::executeForward(ctx);
    
    {
        std::ostringstream oss;
        oss << "[forwardInit] Forward status: " << GRIM::Forward::statusToString(status);
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    if (status != GRIM::Forward::ForwardStatus::SUCCESS) {
        std::ostringstream oss;
        oss << "[forwardInit] ERROR: Forward failed - status=" 
            << GRIM::Forward::statusToString(status) << " error=" << ctx.error_message 
            << ", returning empty";
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Inference, oss.str());
        return Vector();
    }

    training_state_.kv_cache_len = seq_len;

    {
        std::ostringstream oss;
        oss << "[forwardInit] Copying logits from device: single_token_logits=" 
            << (void*)training_state_.single_token_logits 
            << " vocab_size=" << cfg.vocab_size;
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    Vector logits(cfg.vocab_size);
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.single_token_logits,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    training_state_.stream_ctrl.getPrimaryStream());
    training_state_.stream_ctrl.syncPrimaryStream();
    
    {
        std::ostringstream oss;
        oss << "[forwardInit] SUCCESS: returning logits with size=" << logits.data.size();
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Inference, oss.str());
    }
    
    return logits;
}

//======================================================//
//  forwardStep - Decode phase (full recompute)
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t numeric_mask) {
    if (!training_state_.initialized) {
        FWD_ERROR("[forwardStep] training state not initialized; call forwardInit first");
        return Vector();
    }

    GRIM::ForwardOps::LogUnexpectedGradState(training_state_, "forwardStep");
    const auto prefix = GRIM::ForwardOps::BuildForwardPrefix(training_state_, "forwardStep");

    if (training_state_.kv_cache_len == 0) {
        FWD_ERROR(prefix << " KV cache empty; call forwardInit first");
        return Vector();
    }

    const auto& cfg = getConfig();
    const int new_pos = training_state_.kv_cache_len;
    const int new_seq_len = new_pos + 1;
    if (new_seq_len > training_state_.kv_cache_capacity) {
        FWD_ERROR(prefix << " sequence length=" << new_seq_len
                         << " exceeds kv_cache_capacity=" << training_state_.kv_cache_capacity);
        return Vector();
    }

    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        FWD_ERROR(prefix << " numeric side-channel buffers not initialized");
        return Vector();
    }
    cudaMemcpyAsync(training_state_.cached_token_numeric_values + new_pos,
                    &numeric_value,
                    sizeof(float),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());
    cudaMemcpyAsync(training_state_.cached_token_numeric_mask + new_pos,
                    &numeric_mask,
                    sizeof(uint8_t),
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
        FWD_ERROR(prefix << " Forward failed: " << GRIM::Forward::statusToString(status)
                         << " (" << ctx.error_message << ")");
        return Vector();
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
        FWD_ERROR("[forwardStepIncremental] training state not initialized");
        return Vector();
    }

    GRIM::ForwardOps::LogUnexpectedGradState(training_state_, "forwardStepIncremental");
    const auto prefix = GRIM::ForwardOps::BuildForwardPrefix(training_state_, "forwardStepIncremental");

    if (training_state_.kv_cache_len == 0) {
        FWD_ERROR(prefix << " KV cache empty; call forwardInit first");
        return Vector();
    }

    const auto& cfg = getConfig();
    const int query_pos = training_state_.kv_cache_len;
    const int kv_len = query_pos + 1;
    if (kv_len > training_state_.kv_cache_capacity) {
        FWD_ERROR(prefix << " exceeds kv_cache_capacity=" << training_state_.kv_cache_capacity);
        return Vector();
    }

    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        FWD_ERROR(prefix << " numeric side-channel buffers not initialized");
        return Vector();
    }
    cudaMemcpyAsync(training_state_.cached_token_numeric_values + query_pos,
                    &numeric_value,
                    sizeof(float),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());
    cudaMemcpyAsync(training_state_.cached_token_numeric_mask + query_pos,
                    &numeric_mask,
                    sizeof(uint8_t),
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
        FWD_ERROR(prefix << " Forward failed: " << GRIM::Forward::statusToString(status)
                         << " (" << ctx.error_message << ")");
        return Vector();
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
