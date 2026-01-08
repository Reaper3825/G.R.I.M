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

struct EmbeddingWeights {
    const float* token_embeddings = nullptr;      // [vocab_size, d_model]
    const float* position_embeddings = nullptr;   // [max_position, d_model]
    const float* gamma = nullptr;                 // RMSNorm gamma [d_model]
};

struct EmbeddingForwardArgs {
    const int* token_ids = nullptr;   // [batch * seq_len]
    const int* positions = nullptr;   // [batch * seq_len] or nullptr for auto
    int batch_size = 0;
    int seq_len = 0;
    float* output = nullptr;          // [batch * seq_len, d_model]
    const EmbeddingWeights* weights = nullptr;
    cudaStream_t stream = nullptr;
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

    // GPU buffers (owned by runtime)
    float* token_buffer = nullptr;
    float* position_buffer = nullptr;
    float* gamma_buffer = nullptr;

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
