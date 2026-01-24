#pragma once
/**
 * @file Embedding_GPU.hpp
 * @brief GPU-accelerated embedding layer with forward, backward, and runtime support
 *
 * CONSOLIDATED from: Embedding_GPU.hpp, EmbeddingBackward.hpp, EmbeddingRuntime.hpp
 *
 * COMPONENTS:
 * - EmbeddingConfig, EmbeddingWeights, EmbeddingForwardArgs: Core data structures
 * - EmbeddingRuntime: Stateful runtime for inference with pre-allocated buffers
 * - EmbeddingLayer: Stateless forward-only class
 * - launchEmbeddingLookup(): Forward kernel launcher
 * - launchEmbeddingBackward(): Backward kernel launcher (used by BackwardPhase3)
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

/**
 * @brief Batched embedding forward pass
 *
 * When positions=nullptr, position IDs are computed as: token_idx % seq_len
 * This gives each sequence in the batch positions [0, 1, ..., seq_len-1].
 *
 * @return true on success, false on error (error logged to stderr)
 */
bool embeddingRuntimeForward(EmbeddingRuntime* runtime,
                             const int* token_ids,
                             const int* positions,
                             int batch_size,
                             int seq_len,
                             float* output);

/**
 * @brief Single-token embedding forward pass (incremental generation)
 *
 * Uses pre-allocated buffers, no CUDA allocation per call.
 *
 * @return true on success, false on error
 */
bool embeddingRuntimeForwardSingle(EmbeddingRuntime* runtime,
                                   int token_id,
                                   int position,
                                   float* output);

//======================================================//
// Kernel Launchers
//======================================================//

/**
 * @brief Validate token IDs and throw if any are invalid (Rule 20: Fail Loud)
 * 
 * Checks that all token IDs are in range [0, vocab_size).
 * Negative token IDs or IDs >= vocab_size will cause a clear exception
 * with the index and value of the first invalid token.
 * 
 * @param token_ids Device pointer to token IDs
 * @param total_tokens Number of tokens to validate
 * @param vocab_size Valid token range is [0, vocab_size)
 * @param stream CUDA stream to use
 * @throws std::runtime_error if any token ID is invalid
 * 
 * @note Call this before launchEmbeddingLookup() or launchEmbeddingBackward()
 *       when debugging data pipeline issues. Enabled automatically in debug builds.
 */
void validateTokenIds(const int* token_ids,
                      int total_tokens,
                      int vocab_size,
                      cudaStream_t stream);

/**
 * @brief Launch embedding lookup kernel (forward pass)
 *
 * Rule 20: Throws on invalid input. Requires non-null stream.
 */
void launchEmbeddingLookup(const EmbeddingForwardArgs& args,
                           const EmbeddingConfig& config);

/**
 * @brief Launch embedding backward kernel (gradient accumulation)
 *
 * Uses atomicAdd to scatter-add gradients to embedding table.
 * Rule 20: Throws on invalid input.
 *
 * @param grad_output Gradient from downstream [batch_size * seq_len, d_model]
 * @param token_ids Token IDs from forward pass [batch_size * seq_len]
 * @param grad_embeddings Gradient accumulator [vocab_size, d_model]
 */
void launchEmbeddingBackward(const float* grad_output,
                             const int* token_ids,
                             float* grad_embeddings,
                             int batch_size,
                             int seq_len,
                             int d_model,
                             int vocab_size,
                             cudaStream_t stream);

/**
 * @brief Launch position embedding backward kernel (gradient accumulation)
 *
 * Issue #36 FIX: Position embeddings MUST be trainable to match PyTorch baseline.
 * Uses atomicAdd to scatter-add gradients to position embedding table.
 * For each position in [0, seq_len), accumulates gradients from all batch elements.
 *
 * @param grad_output Gradient from downstream [batch_size * seq_len, d_model]
 * @param grad_position_embeddings Gradient accumulator [max_seq_len, d_model]
 */
void launchPositionEmbeddingBackward(const float* grad_output,
                                     float* grad_position_embeddings,
                                     int batch_size,
                                     int seq_len,
                                     int d_model,
                                     int max_seq_len,
                                     cudaStream_t stream);

//======================================================//
// EmbeddingLayer: Stateless forward-only class
//======================================================//

/**
 * @brief GPU-accelerated embedding layer
 *
 * Rule 20: Forward-only implementation. Backward pass uses standalone
 * launchEmbeddingBackward() kernel via BackwardPhase3_InputLayer.cu.
 */
class EmbeddingLayer {
public:
    EmbeddingLayer() = default;
    explicit EmbeddingLayer(const EmbeddingConfig& config) : config_(config) {}

    void setConfig(const EmbeddingConfig& cfg) { config_ = cfg; }
    const EmbeddingConfig& config() const noexcept { return config_; }

    void setWeights(const EmbeddingWeights& weights) { weights_ = weights; }
    const EmbeddingWeights& weights() const noexcept { return weights_; }

    void forward(const EmbeddingForwardArgs& args);

private:
    EmbeddingConfig config_{};
    EmbeddingWeights weights_{};
};

} // namespace GRIM
