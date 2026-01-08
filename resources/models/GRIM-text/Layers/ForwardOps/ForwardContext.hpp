/**
 * @file ForwardContext.hpp
 * @brief Shared context struct for forward pass phases
 *
 * This struct carries all state between forward pass phases.
 * It is initialized by the orchestrator and passed to each phase.
 *
 * DESIGN PRINCIPLES:
 * - Immutable config references (const pointers)
 * - Explicit ownership (no hidden allocations)
 * - Fail-loud validation
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <sstream>

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "ForwardOps_Logging.hpp"

// Forward declarations to avoid heavy includes
namespace GRIM {
    class LanguageModel;
    struct TrainingState;
    class GPUGrimEncoder;
    class ScratchBlockLayer;
    class EmbeddingRuntime;
}

namespace GRIM {
namespace Forward {

//======================================================//
//  Forward Status Codes
//======================================================//

enum class ForwardStatus {
    SUCCESS = 0,
    INVALID_STATE = 1,
    CUDA_ERROR = 2,
    CUBLAS_ERROR = 3,
    NULL_POINTER = 4,
    UNSUPPORTED_MODE = 5
};

inline const char* statusToString(ForwardStatus status) {
    switch (status) {
        case ForwardStatus::SUCCESS: return "SUCCESS";
        case ForwardStatus::INVALID_STATE: return "INVALID_STATE";
        case ForwardStatus::CUDA_ERROR: return "CUDA_ERROR";
        case ForwardStatus::CUBLAS_ERROR: return "CUBLAS_ERROR";
        case ForwardStatus::NULL_POINTER: return "NULL_POINTER";
        case ForwardStatus::UNSUPPORTED_MODE: return "UNSUPPORTED_MODE";
        default: return "UNKNOWN";
    }
}

//======================================================//
//  Forward Mode
//======================================================//

enum class ForwardMode {
    TrainingFull = 0,   // forwardWithCache / computeLossBatch
    Prefill,            // forwardInit
    DecodeFull,         // forwardStep (full recompute)
    DecodeIncremental   // forwardStepIncremental (manual attention)
};

inline const char* modeToString(ForwardMode mode) {
    switch (mode) {
        case ForwardMode::TrainingFull: return "TrainingFull";
        case ForwardMode::Prefill: return "Prefill";
        case ForwardMode::DecodeFull: return "DecodeFull";
        case ForwardMode::DecodeIncremental: return "DecodeIncremental";
        default: return "Unknown";
    }
}

enum class ForwardLogitsTarget {
    FullSequence = 0,   // logits for every token
    LastToken           // logits for final token only
};

//======================================================//
//  Forward Context
//======================================================//

struct ForwardContext {
    const LanguageModelConfig* config = nullptr;
    TrainingState* training_state = nullptr;
    LanguageModel* model = nullptr;

    GPUGrimEncoder* gpu_encoder = nullptr;
    EmbeddingRuntime* embedding_runtime = nullptr;
    ScratchBlockLayer* scratch_block = nullptr;
    const float* token_numeric_values = nullptr;
    const uint8_t* token_numeric_mask = nullptr;
    // GRMT v4: text features for ScratchBlock injection
    const uint16_t* token_text_features = nullptr;  // [total_tokens * kTextFeatureDim]
    const uint8_t* token_text_mask = nullptr;       // [total_tokens]
    const ALiBiPositionalBias* alibi = nullptr;

    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;

    ForwardMode mode = ForwardMode::TrainingFull;
    ForwardLogitsTarget logits_target = ForwardLogitsTarget::FullSequence;

    const int* host_tokens = nullptr;
    bool tokens_on_device = false;

    int batch_size = 0;
    int seq_len = 0;
    int total_tokens = 0;

    int new_token = -1;
    int query_pos = -1;

    bool enable_scratch_block = false;
    bool enable_activation_quantization = false;
    bool enable_entropy_output = false;

    EncoderLayerCache* layer_caches = nullptr;
    int layer_cache_count = 0;

    float* encoder_output = nullptr;
    float* logits_output = nullptr;

    ForwardStatus phase1_status = ForwardStatus::SUCCESS;
    ForwardStatus phase2_status = ForwardStatus::SUCCESS;
    ForwardStatus phase3_status = ForwardStatus::SUCCESS;

    std::string error_message;
    int error_layer = -1;

    ForwardStatus validate() const {
        if (!config || !training_state) {
            return ForwardStatus::INVALID_STATE;
        }
        if (!stream) {
            return ForwardStatus::INVALID_STATE;
        }
        if (!cublas_handle) {
            return ForwardStatus::CUBLAS_ERROR;
        }
        return ForwardStatus::SUCCESS;
    }
};

//======================================================//
//  FAIL LOUD Macros
//======================================================//

#ifndef FWD_FAIL_LOUD
#define FWD_FAIL_LOUD(ctx, status, message, layer_idx) \
    do { \
        (ctx).error_message = (message); \
        (ctx).error_layer = (layer_idx); \
        FWD_ERROR("[ForwardFail] " << statusToString(status) << ": " << (message)); \
        FWD_ERROR("  Location: " << __FILE__ << ":" << __LINE__); \
        if ((layer_idx) >= 0) { \
            FWD_ERROR("  Layer: " << (layer_idx)); \
        } \
        return (status); \
    } while (0)
#endif

#ifndef FWD_CHECK_PTR
#define FWD_CHECK_PTR(ctx, ptr, name, layer_idx) \
    do { \
        if ((ptr) == nullptr) { \
            FWD_FAIL_LOUD((ctx), ForwardStatus::NULL_POINTER, std::string("Null pointer: ") + (name), (layer_idx)); \
        } \
    } while (0)
#endif

#ifndef FWD_CHECK_CUDA
#define FWD_CHECK_CUDA(ctx, expr, message, layer_idx) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            std::ostringstream _oss; \
            _oss << (message) << ": " << cudaGetErrorString(_err); \
            FWD_FAIL_LOUD((ctx), ForwardStatus::CUDA_ERROR, _oss.str(), (layer_idx)); \
        } \
    } while (0)
#endif

#ifndef FWD_CHECK_CUBLAS
#define FWD_CHECK_CUBLAS(ctx, expr, message, layer_idx) \
    do { \
        cublasStatus_t _status = (expr); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream _oss; \
            _oss << (message) << ": cublas status " << static_cast<int>(_status); \
            FWD_FAIL_LOUD((ctx), ForwardStatus::CUBLAS_ERROR, _oss.str(), (layer_idx)); \
        } \
    } while (0)
#endif

} // namespace Forward
} // namespace GRIM
