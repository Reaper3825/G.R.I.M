/**
 * @file BackwardContext.hpp
 * @brief Shared context struct for backward pass phases
 *
 * This struct carries all state between backward pass phases.
 * It is initialized by the orchestrator and passed to each phase.
 *
 * DESIGN PRINCIPLES:
 * - Immutable config references (const pointers)
 * - Mutable gradient flow state
 * - Explicit ownership (no hidden allocations)
 * - Fail-loud validation
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <sstream>

// Include LogRecorder for BWD_ERROR macro
#include "../../Shared/LogRecorder/LogRecorder.hpp"

// Include LanguageModelConfig for full type definition
#include "../../GRIM/grim_language_model_cuda.hpp"

// Forward declarations to avoid heavy includes
namespace GRIM {
    // LanguageModelConfig is now included from grim_language_model_cuda.hpp
    struct TrainingState;
    class GPUGrimEncoder;
    class ScratchBlockLayer;
    class EmbeddingRuntime;
}

namespace GRIM {
namespace Backward {

//======================================================//
//  Backward Status Codes
//======================================================//

/**
 * @brief Return status for backward pass operations
 *
 * FAIL LOUD: Any non-SUCCESS status should halt training
 * with detailed error information.
 */
enum class BackwardStatus {
    SUCCESS = 0,              ///< Operation completed successfully
    FATAL_ERROR = 1,          ///< Unrecoverable error, must stop training
    GRADIENT_EXPLOSION = 2,   ///< Detected gradient explosion (>1e6)
    INVALID_STATE = 3,        ///< Training state not initialized
    CUDA_ERROR = 4,           ///< CUDA operation failed
    CUBLAS_ERROR = 5,         ///< cuBLAS operation failed
    TENSOR_CONTRACT_VIOLATION = 6, ///< TensorContract validation failed
    NULL_POINTER = 7,         ///< Required pointer was null
};

/**
 * @brief Convert status to human-readable string
 */
inline const char* statusToString(BackwardStatus status) {
    switch (status) {
        case BackwardStatus::SUCCESS: return "SUCCESS";
        case BackwardStatus::FATAL_ERROR: return "FATAL_ERROR";
        case BackwardStatus::GRADIENT_EXPLOSION: return "GRADIENT_EXPLOSION";
        case BackwardStatus::INVALID_STATE: return "INVALID_STATE";
        case BackwardStatus::CUDA_ERROR: return "CUDA_ERROR";
        case BackwardStatus::CUBLAS_ERROR: return "CUBLAS_ERROR";
        case BackwardStatus::TENSOR_CONTRACT_VIOLATION: return "TENSOR_CONTRACT_VIOLATION";
        case BackwardStatus::NULL_POINTER: return "NULL_POINTER";
        default: return "UNKNOWN";
    }
}

//======================================================//
//  Backward Context
//======================================================//

/**
 * @brief Shared context for backward pass phases
 *
 * This struct is populated by the orchestrator and passed to each phase.
 * Phases may modify gradient pointers and metrics but not config.
 */
struct BackwardContext {
    //--------------------------------------------------//
    // Model Configuration (Immutable)
    //--------------------------------------------------//
    
    const LanguageModelConfig* config = nullptr;  ///< Model architecture config
    TrainingState* training_state = nullptr; ///< Mutable training buffers
    
    //--------------------------------------------------//
    // Batch Information
    //--------------------------------------------------//
    
    int batch_size = 0;       ///< Current batch size (B)
    int seq_len = 0;          ///< Current sequence length (S)
    int total_tokens = 0;     ///< B * S for convenience
    
    //--------------------------------------------------//
    // Gradient Flow State (Mutable)
    //--------------------------------------------------//
    
    float* current_grad = nullptr;  ///< Gradient flowing backward (updates each phase)
    float grad_scale = 1.0f;        ///< Token normalization scale (1.0 / total_tokens)
    bool accumulate = false;        ///< Accumulate gradients (true) or overwrite (false)
    
    //--------------------------------------------------//
    // cuBLAS Constants
    //--------------------------------------------------//
    
    float alpha = 1.0f;             ///< cuBLAS alpha constant
    float beta_zero = 0.0f;         ///< cuBLAS beta for overwrite
    float beta_accum = 0.0f;        ///< cuBLAS beta: 1.0 if accumulate, 0.0 otherwise
    
    //--------------------------------------------------//
    // External Components
    //--------------------------------------------------//
    
    GPUGrimEncoder* gpu_encoder = nullptr;      ///< GPU encoder layers
    ScratchBlockLayer* scratch_block = nullptr; ///< ScratchBlock reasoning layer
    EmbeddingRuntime* embedding_runtime = nullptr; ///< Embedding layer
    cublasHandle_t cublas_handle = nullptr;     ///< cuBLAS handle
    
    //--------------------------------------------------//
    // Diagnostics & Telemetry
    //--------------------------------------------------//
    
    uint64_t backward_call_id = 0;  ///< Monotonic counter for deterministic diagnostics
    uint64_t step = 0;               ///< Global training step (for layer logging)
    bool enable_grad_checks = false;  ///< Enable gradient statistics logging
    bool enable_layer_logging = false; ///< Enable per-layer logging to files
    float explosion_threshold = 1e6f; ///< Gradient norm threshold for explosion detection
    
    //--------------------------------------------------//
    // Phase Results (Written by phases)
    //--------------------------------------------------//
    
    BackwardStatus phase1_status = BackwardStatus::SUCCESS;
    BackwardStatus phase2_status = BackwardStatus::SUCCESS;
    BackwardStatus phase3_status = BackwardStatus::SUCCESS;
    
    std::string error_message;      ///< Detailed error message if any phase fails
    int error_layer = -1;           ///< Layer index where error occurred (-1 if N/A)
    
    //--------------------------------------------------//
    // Validation
    //--------------------------------------------------//
    
    /**
     * @brief Validate context is properly initialized
     * @return SUCCESS if valid, appropriate error code otherwise
     */
    BackwardStatus validate() const {
        if (!config) {
            return BackwardStatus::NULL_POINTER;
        }
        if (!training_state || batch_size <= 0 || seq_len <= 0) {
            return BackwardStatus::INVALID_STATE;
        }
        if (!cublas_handle) {
            return BackwardStatus::CUBLAS_ERROR;
        }
        return BackwardStatus::SUCCESS;
    }
    
    /**
     * @brief Check if context is valid for backward pass
     */
    bool isValid() const {
        return validate() == BackwardStatus::SUCCESS;
    }
};

//======================================================//
//  Gradient Check Result
//======================================================//

/**
 * @brief Result of gradient statistics check
 */
struct GradCheckResult {
    float rms = 0.0f;           ///< Root mean square
    float max_abs = 0.0f;       ///< Maximum absolute value
    float min_val = 0.0f;       ///< Minimum value
    float max_val = 0.0f;       ///< Maximum value
    bool has_nan = false;       ///< Contains NaN values
    bool has_inf = false;       ///< Contains Inf values
    bool is_explosion = false;  ///< RMS exceeds threshold
    
    /**
     * @brief Check if gradient is healthy
     */
    bool isHealthy() const {
        return !has_nan && !has_inf && !is_explosion;
    }
};

//======================================================//
//  FAIL LOUD Macros
//======================================================//

// BWD_ERROR macro for logging - requires LogRecorder to be included
// This must be defined before using BWD_FAIL_LOUD
#ifndef BWD_ERROR
#define BWD_ERROR(msg) do { \
    std::ostringstream _bwd_oss; \
    _bwd_oss << "[BackwardError] " << msg; \
    GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::BackwardPass, _bwd_oss.str()); \
} while (0)
#endif

/**
 * @brief FAIL LOUD macro for backward pass errors
 *
 * Usage:
 *   BWD_FAIL_LOUD(ctx, BackwardStatus::CUDA_ERROR, "cudaMemcpy failed", layer);
 */
#define BWD_FAIL_LOUD(ctx, status, message, layer_idx) \
    do { \
        (ctx).error_message = (message); \
        (ctx).error_layer = (layer_idx); \
        BWD_ERROR("[BackwardFail] " << statusToString(status) << ": " << (message)); \
        BWD_ERROR("  Location: " << __FILE__ << ":" << __LINE__); \
        if ((layer_idx) >= 0) { \
            BWD_ERROR("  Layer: " << (layer_idx)); \
        } \
        return (status); \
    } while (0)

/**
 * @brief Check CUDA error and fail loud if not success
 */
#define BWD_CHECK_CUDA(ctx, expr, message, layer_idx) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            std::ostringstream _oss; \
            _oss << (message) << ": " << cudaGetErrorString(_err); \
            BWD_FAIL_LOUD(ctx, BackwardStatus::CUDA_ERROR, _oss.str(), layer_idx); \
        } \
    } while (0)

/**
 * @brief Check cuBLAS error and fail loud if not success
 */
#define BWD_CHECK_CUBLAS(ctx, expr, message, layer_idx) \
    do { \
        cublasStatus_t _status = (expr); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream _oss; \
            _oss << (message) << ": cuBLAS error " << static_cast<int>(_status); \
            BWD_FAIL_LOUD(ctx, BackwardStatus::CUBLAS_ERROR, _oss.str(), layer_idx); \
        } \
    } while (0)

/**
 * @brief Check pointer is not null
 */
#define BWD_CHECK_PTR(ctx, ptr, name, layer_idx) \
    do { \
        if (!(ptr)) { \
            std::ostringstream _oss; \
            _oss << "NULL pointer: " << (name); \
            BWD_FAIL_LOUD(ctx, BackwardStatus::NULL_POINTER, _oss.str(), layer_idx); \
        } \
    } while (0)

/**
 * @brief Check gradient for explosion
 */
#define BWD_CHECK_GRAD_EXPLOSION(ctx, result, name, layer_idx) \
    do { \
        if ((result).is_explosion) { \
            std::ostringstream _oss; \
            _oss << "Gradient explosion in " << (name) << ": rms=" << (result).rms; \
            BWD_FAIL_LOUD(ctx, BackwardStatus::GRADIENT_EXPLOSION, _oss.str(), layer_idx); \
        } \
        if ((result).has_nan || (result).has_inf) { \
            std::ostringstream _oss; \
            _oss << "NaN/Inf in " << (name) << ": nan=" << (result).has_nan << " inf=" << (result).has_inf; \
            BWD_FAIL_LOUD(ctx, BackwardStatus::GRADIENT_EXPLOSION, _oss.str(), layer_idx); \
        } \
    } while (0)

} // namespace Backward
} // namespace GRIM
