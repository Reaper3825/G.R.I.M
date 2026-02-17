//======================================================//
//  Embedding Layer - GPU (Pattern B: Layer Ownership)
//  Token + optional Position embedding lookup
//
//  Owns: token_weights [vocab_size, d_model],
//        position_weights [max_seq_len, d_model] (optional, learned mode only).
//
//  Architecture: output = token_embed(input_ids) [+ pos_embed(pos_ids)]
//  Position embeddings are only allocated for LEARNED mode (PositionalEncodingType::NONE).
//  ALiBi/RoPE modes inject position information inside attention, not here.
//
//  Backward is handled automatically by the autograd tape system:
//    grad_W_token[token_id] += grad_output[t] * scale  (scatter accumulate)
//    grad_W_pos[pos_id] += grad_output[t] * scale       (scatter accumulate)
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
    int max_seq_len = 0;       // Maximum sequence length (MUST be populated)
    HyperParameters::PositionalEncodingType positional_encoding =
        HyperParameters::PositionalEncodingType::ALIBI_ROPE;
    float embedding_scale = 1.0f;  // Issue #140: No scaling (1.0f) for ALiBi/RoPE
    bool requires_grad = true;     // false for inference-only (skip grad allocation)
};

//======================================================//
//  EmbeddingLayer - Self-Allocating (Pattern B: Layer Ownership)
//
//  Owns token embedding weights. Optionally owns position embedding weights
//  (only for LEARNED positional encoding mode).
//
//  forward() performs: output = token_embed(ids) [+ pos_embed(pos_ids)]
//  backward() handled by autograd chain (EmbeddingGradFn)
//======================================================//

class EmbeddingLayer {
public:
    // Rule 20: Default constructor deleted
    EmbeddingLayer() = delete;

    /// Self-allocating constructor (Pattern B - Layer Ownership)
    ///
    /// Allocates token embedding weights [vocab_size, d_model] with Xavier init.
    /// If positional_encoding == NONE (learned mode), also allocates
    /// position embedding weights [max_seq_len, d_model].
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

    Tensor& positionWeights() { return position_weights_; }
    const Tensor& positionWeights() const { return position_weights_; }

    /// Whether position embeddings are allocated (learned mode only)
    bool hasPositionEmbeddings() const { return position_weights_.data != nullptr; }

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
    Tensor position_weights_;    // [max_seq_len, d_model] — optional (learned mode only)
};

} // namespace GRIM

#endif // USE_CUDA
