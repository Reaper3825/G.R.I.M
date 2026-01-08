/**
 * @file BackwardOps_Orchestrator.hpp
 * @brief Backward Pass Orchestrator - Main Entry Point
 *
 * This orchestrator coordinates the 3-phase backward pass:
 * - Phase 1: Output Layer (Cross-entropy gradient, LM Head backward)
 * - Phase 2: Encoder (All transformer layers in reverse)
 * - Phase 3: Input Layer (ScratchBlock, Embeddings)
 *
 * USAGE:
 *   BackwardContext ctx = initBackwardContext(model, batch_size, seq_len, ...);
 *   BackwardStatus status = executeBackward(ctx);
 *   if (status != BackwardStatus::SUCCESS) {
 *       // Handle error with detailed ctx.error_message
 *   }
 *
 * FAIL LOUD CONTRACT:
 * - Any error halts training immediately with detailed diagnostics
 * - Gradient explosions detected at every phase boundary
 * - CUDA/cuBLAS errors caught and reported with context
 *
 * @note This replaces the monolithic backward() method in LanguageModel
 */

#pragma once

#include "BackwardContext.hpp"
#include "BackwardPhase1_OutputLayer.hpp"
#include "BackwardPhase2_Encoder.hpp"
#include "BackwardPhase3_InputLayer.hpp"

// Forward declarations
namespace GRIM {
    class LanguageModel;
    // LanguageModelConfig is now included via BackwardContext.hpp
    struct TrainingState;
    class GPUGrimEncoder;
    class ScratchBlockLayer;
    class EmbeddingRuntime;
}

namespace GRIM {
namespace Backward {

//======================================================//
//  Orchestrator Entry Point
//======================================================//

/**
 * @brief Execute complete backward pass through all phases
 *
 * @param ctx Initialized backward context
 * @return BackwardStatus::SUCCESS or appropriate error code
 *
 * @pre Context must be initialized via initBackwardContext()
 * @post All gradients computed and ready for optimizer step
 *
 * This function:
 * 1. Validates context
 * 2. Executes Phase 1 (Output Layer)
 * 3. Executes Phase 2 (Encoder - all layers)
 * 4. Executes Phase 3 (Input Layer)
 * 5. Reports timing metrics
 */
BackwardStatus executeBackward(BackwardContext& ctx);

//======================================================//
//  Context Initialization
//======================================================//

/**
 * @brief Initialize backward context from LanguageModel state
 *
 * @param model LanguageModel instance
 * @param batch_size Current batch size
 * @param seq_len Current sequence length
 * @param loss Loss value for scaling
 * @param accumulate Whether to accumulate gradients
 * @param grad_scale Gradient normalization scale
 * @return Initialized BackwardContext
 *
 * This extracts all necessary state from LanguageModel to create
 * a self-contained context that can be passed through phases.
 */
BackwardContext initBackwardContext(
    LanguageModel& model,
    int batch_size,
    int seq_len,
    float loss,
    bool accumulate,
    float grad_scale);

/**
 * @brief Initialize backward context from raw components
 *
 * @param config Model configuration
 * @param training_state Training state buffers
 * @param gpu_encoder GPU encoder layers
 * @param scratch_block ScratchBlock layer (optional)
 * @param embedding_runtime Embedding layer
 * @param cublas_handle cuBLAS handle
 * @param stream CUDA stream
 * @param batch_size Current batch size
 * @param seq_len Current sequence length
 * @param accumulate Whether to accumulate gradients
 * @param grad_scale Gradient normalization scale
 * @return Initialized BackwardContext
 *
 * This is the low-level initializer for testing or custom usage.
 */
BackwardContext initBackwardContextRaw(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,
    EmbeddingRuntime* embedding_runtime,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    int batch_size,
    int seq_len,
    bool accumulate,
    float grad_scale,
    uint64_t step = 0);

//======================================================//
//  Diagnostic Functions
//======================================================//

/**
 * @brief Get human-readable error report from failed backward pass
 *
 * @param ctx Backward context (after failed backward)
 * @return Formatted error string with full diagnostics
 */
std::string getBackwardErrorReport(const BackwardContext& ctx);

/**
 * @brief Log comprehensive backward pass summary
 *
 * @param ctx Backward context (after backward pass)
 *
 * Logs timing breakdown, gradient statistics, and any warnings.
 */
void logBackwardSummary(const BackwardContext& ctx);

} // namespace Backward
} // namespace GRIM
