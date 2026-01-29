#pragma once
/**
 * @file Embedding_GPU.hpp
 * @brief EmbeddingRuntime memory management for GPU embedding buffers
 *
 * PRODUCTION PATH: autograd::embedding() in TensorContract_GPU.cu
 *
 * COMPONENTS (RETAINED):
 * - EmbeddingConfig, EmbeddingWeights, EmbeddingForwardArgs: Core data structures
 * - EmbeddingRuntime: Struct holding GPU buffers for embedding weights
 * - destroyEmbeddingRuntime(): Frees GPU buffers
 *
 * LEGACY CODE DELETED (Issue #92 / Rule 20: No Backwards Compatibility):
 * Production uses autograd::embedding() which has kernel_embedding_forward/backward
 * in TensorContract_GPU.cu. All legacy kernels and launchers removed.
 */

#include <cuda_runtime.h>
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

//======================================================//
// Core Data Structures
//======================================================//

struct EmbeddingConfig {
    int vocab_size = 0;
    int max_position = 0;
    int d_model = 0;
    bool apply_rms_norm = false;
    float rms_epsilon = 1e-5f;
    float embedding_scale = 1.0f;  // Scale factor applied to embeddings (sqrt(d_model) for AIAYN-style)
    cudaStream_t stream = nullptr;
};

/**
 * @brief Embedding weight tensors
 * 
 * All tensors use BSM layout (2D row-major format).
 * 
 * @note token_embeddings: [vocab_size, d_model] - REQUIRED
 * @note position_embeddings: [max_position, d_model] - Optional (can be null)
 * @note gamma: [1, d_model] - RMSNorm scale parameters - Optional (can be null)
 * 
 * Rule 20 (Fail Loud): Use validate() to check tensors before use.
 */
struct EmbeddingWeights {
    TensorContract::TensorView token_embeddings;      // [vocab_size, d_model]
    TensorContract::TensorView position_embeddings;   // [max_position, d_model] (optional)
    TensorContract::TensorView gamma;                 // [1, d_model] RMSNorm (optional)
    
    /**
     * Validate all tensors (Rule 20: Fail Loud)
     * @throws std::runtime_error if token_embeddings is invalid
     */
    void validate(const char* context) const {
        token_embeddings.require(context);
        
        // position_embeddings and gamma are optional
        if (position_embeddings.ptr) {
            position_embeddings.require(context);
        }
        if (gamma.ptr) {
            gamma.require(context);
        }
    }
    
    // Convenience accessors
    int vocab_size() const { return token_embeddings.shape.as_2d().rows; }
    int d_model() const { return token_embeddings.shape.as_2d().cols; }
    int max_position() const { 
        return position_embeddings.ptr ? position_embeddings.shape.as_2d().rows : 0; 
    }
};

/**
 * @brief Arguments for embedding forward pass
 * 
 * @note token_ids: device int array [batch * seq_len] - index array (not a tensor)
 * @note positions: device int array [batch * seq_len] - optional position indices
 * @note output: BSM layout [batch * seq_len, d_model] - will be written
 * @note weights: embedding tables (TensorView-based)
 * 
 * Rule 20 (Fail Loud): Use validate() to check all tensors before use.
 */
struct EmbeddingForwardArgs {
    const int* token_ids = nullptr;   // [batch * seq_len] - index array
    const int* positions = nullptr;   // [batch * seq_len] - optional
    int batch_size = 0;
    int seq_len = 0;
    TensorContract::TensorView output;           // [batch * seq_len, d_model] BSM
    const EmbeddingWeights* weights = nullptr;
    cudaStream_t stream = nullptr;
    
    /**
     * Validate all tensors (Rule 20: Fail Loud)
     * @throws std::runtime_error if token_ids is null, output is invalid, or weights is null
     */
    void validate(const char* context) const {
        if (!token_ids) {
            throw std::runtime_error(std::string(context) + ": token_ids is NULL");
        }
        output.require(context);
        if (!weights) {
            throw std::runtime_error(std::string(context) + ": weights is NULL");
        }
        weights->validate(context);
        
        // Layout validation
        if (output.layout() != TensorContract::Layout::BSM) {
            throw std::runtime_error(std::string(context) + ": output must have BSM layout");
        }
    }
    
    // Convenience accessors
    int total_tokens() const { return output.shape.as_2d().rows; }
    int d_model() const { return output.shape.as_2d().cols; }
};

//======================================================//
// EmbeddingRuntime: Stateful runtime for inference
//======================================================//

/**
 * @brief Runtime state for embedding operations
 *
 * Holds configuration, weights, and pre-allocated buffers.
 * Destroyed via destroyEmbeddingRuntime().
 */
struct EmbeddingRuntime {
    EmbeddingConfig config{};
    EmbeddingWeights weights{};

    // GPU buffers (may be owned by runtime OR by TrainingTensors)
    float* token_buffer = nullptr;
    float* position_buffer = nullptr;
    float* gamma_buffer = nullptr;

    // Ownership flags: true = runtime owns memory, false = points to external memory (e.g. TrainingTensors)
    // When false, destroyEmbeddingRuntime() will NOT cudaFree these buffers.
    bool owns_token_buffer = true;
    bool owns_position_buffer = true;
    bool owns_gamma_buffer = true;

    // Pre-allocated single-token buffers (avoids malloc per inference call)
    int* single_token_id = nullptr;
    int* single_position = nullptr;

    // Stream management
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
};

/**
 * @brief Destroy embedding runtime and free all resources
 * @param runtime Runtime to destroy (safe to call with nullptr)
 */
void destroyEmbeddingRuntime(EmbeddingRuntime* runtime);

} // namespace GRIM
