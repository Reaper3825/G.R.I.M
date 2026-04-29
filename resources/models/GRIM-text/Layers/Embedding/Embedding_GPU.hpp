//======================================================//
//  Embedding Layer - GPU (Pattern B: Layer Ownership)
//  Token embedding lookup
//
//  Owns: token_weights [vocab_size, d_model].
//  Position information is injected inside attention (ALiBi/RoPE); the learned
//  position-embedding path has been removed (Rule 26).
//
//  Architecture: output = token_embed(input_ids)
//
//  Backward is handled automatically by the autograd tape system:
//    grad_W_token[token_id] += grad_output[t] * scale  (scatter accumulate)
//
//  Weight tying: When tie_embeddings=true, LMHeadLayer aliases this layer's
//  token_weights via Tensor::from_ptr + share_grad. EmbeddingLayer always OWNS
//  the underlying GPU buffer.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM {

//======================================================//
//  Configuration
//======================================================//

struct EmbeddingLayerConfig {
    int vocab_size = 0;        // Token vocabulary size (MUST be populated)
    int d_model = 0;           // Hidden dimension (MUST be populated)
    bool requires_grad = true;     // false for inference-only (skip grad allocation)
};

//======================================================//
//  EmbeddingLayer - Self-Allocating (Pattern B: Layer Ownership)
//
//  Owns token embedding weights. Position information is injected inside
//  attention via ALiBi/RoPE (no separate learned position-embedding table).
//
//  forward() performs: output = token_embed(ids)
//  backward() handled by autograd chain (EmbeddingGradFn)
//======================================================//

class EmbeddingLayer {
public:
    // Rule 20: Default constructor deleted
    EmbeddingLayer() = delete;

    /// Self-allocating constructor (Pattern B - Layer Ownership)
    ///
    /// Allocates token embedding weights [vocab_size, d_model] with Xavier init.
    ///
    /// @param config     Layer configuration
    /// @param seed       Xavier init seed
    /// @param stream     CUDA stream for allocation
    explicit EmbeddingLayer(const EmbeddingLayerConfig& config,
                            uint64_t seed,
                            cudaStream_t stream);

    ~EmbeddingLayer() = default;

    // Non-copyable (GPU resource ownership)
    EmbeddingLayer(const EmbeddingLayer&) = delete;
    EmbeddingLayer& operator=(const EmbeddingLayer&) = delete;

    // Allow move
    EmbeddingLayer(EmbeddingLayer&& other) noexcept;
    EmbeddingLayer& operator=(EmbeddingLayer&& other) noexcept;

    //--------------------------------------------------
    // Weight Accessors (for buildParameterGroups, serialization, LM Head tying)
    //--------------------------------------------------
    Tensor& tokenWeights() { return token_weights_; }
    const Tensor& tokenWeights() const { return token_weights_; }

    /// Whether weights are initialized and ready for forward pass
    bool weightsReady() const { return token_weights_.data != nullptr; }

    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------
    const EmbeddingLayerConfig& config() const noexcept { return config_; }

private:
    EmbeddingLayerConfig config_{};

    // Weight Tensors with autograd (requires_grad=true)
    Tensor token_weights_;       // [vocab_size, d_model] — always owned
};

} // namespace GRIM

#endif // USE_CUDA
