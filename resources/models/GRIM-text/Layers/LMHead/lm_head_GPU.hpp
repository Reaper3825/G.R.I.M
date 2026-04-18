//======================================================//
//  LM Head Layer - GPU (Pattern B: Layer Ownership)
//  Linear projection from hidden states to vocabulary logits
//
//  Owns: weights [vocab_size, d_model], bias [vocab_size] (optional),
//        final_rms_gamma [d_model] (pre-LM-head normalization).
//
//  Architecture: logits = centered(RMSNorm(encoder_output)) @ W^T + bias
//  Where W is either tied to embedding weights or independently allocated.
//
//  Backward is handled automatically by the autograd tape system:
//    grad_W = centered^T @ grad_logits
//    grad_input = grad_logits @ W  (flows back through centering + RMSNorm ops)
//    grad_bias = sum(grad_logits, dim=0)
//    grad_gamma via RMSNormGradFn
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <cstdint>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

//======================================================//
//  Configuration
//======================================================//

struct LMHeadLayerConfig {
    int d_model = 0;           // Hidden dimension (MUST be populated)
    int vocab_size = 0;        // Output vocabulary size (MUST be populated)
    bool use_bias = false;     // Add learnable bias to logits
    bool center_hidden_states = false;  // Apply column+row centering before projection (Issue #125/#132)
    bool project_out_pc1 = false;       // Project out dominant PC1 direction before projection (Issue #149)
    int  pc1_power_iters = 5;           // Number of power iteration steps for PC1 estimation
    bool center_logits = false;         // Row-center logits after projection (numerical stability)
    bool has_final_rms_norm = true;     // Apply RMSNorm before projection (pre-LM-head norm)
    float rms_epsilon = 1e-5f;          // RMSNorm epsilon
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;  // Rule 22: MUST be training_state.cublas_handle
};

//======================================================//
//  LMHeadLayer - Self-Allocating (Pattern B: Layer Ownership)
//
//  Owns its own weights (allocated in constructor) OR aliases
//  embedding weights when tie_embeddings=true (Issue #60).
//  Also owns final_rms_gamma for pre-LM-head normalization.
//
//  forward() applies: RMSNorm → centering → matmul → bias
//  backward() handled by autograd chain (RMSNormGradFn → MatMulGradFn etc.)
//======================================================//

class LMHeadLayer {
public:
    // Rule 20: Default constructor deleted
    LMHeadLayer() = delete;

    /// Self-allocating constructor (Pattern B - Layer Ownership)
    ///
    /// When tied_embedding_weights is non-null, weights alias embedding (Issue #60 weight tying).
    /// When null, weights are independently allocated with Xavier init.
    /// final_rms_gamma is ALWAYS self-allocated (initialized to 1.0).
    ///
    /// @param config               Layer configuration (d_model, vocab_size, etc.)
    /// @param seed                  Xavier init seed for independent weights
    /// @param init_stream           CUDA stream for allocation
    /// @param tied_embedding_weights If non-null, weights alias this tensor (from_ptr + share_grad)
    explicit LMHeadLayer(const LMHeadLayerConfig& config,
                         uint64_t seed,
                         cudaStream_t init_stream,
                         Tensor* tied_embedding_weights = nullptr);

    ~LMHeadLayer() = default;

    // Non-copyable (GPU resource ownership)
    LMHeadLayer(const LMHeadLayer&) = delete;
    LMHeadLayer& operator=(const LMHeadLayer&) = delete;

    // Allow move
    LMHeadLayer(LMHeadLayer&& other) noexcept;
    LMHeadLayer& operator=(LMHeadLayer&& other) noexcept;

    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------
    const LMHeadLayerConfig& config() const noexcept { return config_; }

    //--------------------------------------------------
    // Weight Accessors (for training/serialization)
    //--------------------------------------------------
    Tensor& weights() { return weights_; }
    Tensor& bias() { return bias_; }
    Tensor& finalRmsGamma() { return final_rms_gamma_; }
    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }
    const Tensor& finalRmsGamma() const { return final_rms_gamma_; }

    /// Whether this layer allocated its own weights (false = tied to embedding)
    bool ownsWeights() const { return owns_weights_; }

    /// Whether weights are initialized and ready for forward pass
    bool weightsReady() const { return weights_.data != nullptr; }

    //--------------------------------------------------
    // Runtime Configuration (update before forward)
    //--------------------------------------------------
    void setStream(cudaStream_t s) { config_.stream = s; }
    void setCublasHandle(cublasHandle_t h) { config_.cublas_handle = h; }

    //--------------------------------------------------
    // Forward Pass - Autograd
    //--------------------------------------------------
    /// LM head forward with autograd tracking:
    ///   0. Optional: RMSNorm(input, final_rms_gamma_) — pre-LM-head normalization
    ///   1. Optional: center_columns + center_rows on normalized input (Issue #125/#132)
    ///   2. logits = input @ weights^T  (autograd::matmul, transpose_b=true)
    ///   3. Optional: center_rows on logits (numerical stability)
    ///   4. Optional: logits += bias  (autograd::broadcast_add)
    ///
    /// Builds compute graph for automatic backward().
    ///
    /// @param input                    [total_tokens, d_model] - encoder output (MUST have grad_fn if training)
    /// @param out_centered_hidden      Output: centered hidden states (valid only if centering enabled, for diagnostics)
    /// @return logits [total_tokens, vocab_size] with grad_fn attached
    Tensor forward(const Tensor& input, Tensor& out_centered_hidden);



private:
    LMHeadLayerConfig config_{};

    // Weight Tensors with autograd (requires_grad=true)
    Tensor weights_;          // [vocab_size, d_model] — owned or aliased from embedding
    Tensor bias_;             // [vocab_size] — optional, always owned
    Tensor final_rms_gamma_;  // [d_model] — pre-LM-head RMSNorm gamma, always owned

    // April 2026: Workspace for the row-centered LM head weight matrix
    // (Σ_d W[v,d]=0 constraint that replaces row-centering of hidden states).
    // Held as a member so its data buffer outlives forward() — the matmul GradFn
    // captures W via this tensor, and backward must dereference its .data and
    // grad chain after forward() has returned.
    Tensor centered_weights_;

    bool owns_weights_ = true;  // false when tied to embedding weights
};

} // namespace GRIM

#endif // USE_CUDA
