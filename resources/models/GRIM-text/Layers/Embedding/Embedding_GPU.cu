//======================================================//
//  Embedding_GPU.cu
//  GPU-accelerated Embedding layer using autograd
//  Pattern B: Layer Ownership — self-allocates weights
//
//  Owns: token_weights [vocab_size, d_model],
//        position_weights [max_seq_len, d_model] (optional).
//
//  PyTorch equivalent:
//    class Embedding(nn.Module):
//        def __init__(self, vocab_size, d_model, max_seq_len=None):
//            self.token_embed = nn.Embedding(vocab_size, d_model)
//            self.pos_embed = nn.Embedding(max_seq_len, d_model) if learned else None
//        def forward(self, input_ids, position_ids=None):
//            x = self.token_embed(input_ids)
//            if self.pos_embed: x = x + self.pos_embed(position_ids)
//            return x
//======================================================//

#include "Embedding_GPU.hpp"

#include <stdexcept>
#include <cstdio>
#include <string>

namespace GRIM {

//======================================================//
//  Self-Allocating Constructor (Pattern B)
//======================================================//

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerConfig& config,
                               uint64_t seed,
                               cudaStream_t stream)
    : config_(config)
{
    // Rule 20: Fail loud on invalid configuration
    if (config_.vocab_size <= 0) {
        throw std::runtime_error("EmbeddingLayer: vocab_size must be positive, got " + std::to_string(config_.vocab_size));
    }
    if (config_.d_model <= 0) {
        throw std::runtime_error("EmbeddingLayer: d_model must be positive, got " + std::to_string(config_.d_model));
    }
    if (config_.max_seq_len <= 0) {
        throw std::runtime_error("EmbeddingLayer: max_seq_len must be positive, got " + std::to_string(config_.max_seq_len));
    }
    if (!stream) {
        throw std::runtime_error("EmbeddingLayer: stream is NULL — CUDA stream required for allocation");
    }

    // ══════════════════════════════════════════════════════════════
    //  TOKEN EMBEDDINGS: Always allocated [vocab_size, d_model]
    // ══════════════════════════════════════════════════════════════
    token_weights_ = Tensor::zeros({config_.vocab_size, config_.d_model}, stream, "embedding.token_weights");
    if (config_.requires_grad) {
        token_weights_.requires_grad_();
        token_weights_.ensure_grad();  // Allocate grad NOW so share_grad() works for LM head tying
    }
    // Issue #163: Normal(0,1) init gives embedding RMS ≈ 1.0, matching post-RMSNorm
    // residual stream scale. Xavier gives RMS ≈ sqrt(6/(V+D))/sqrt(3) ≈ 0.014 for
    // V=10000, D=768 — 60× weaker than the sublayer noise accumulated through 12
    // encoder layers. The SNR mismatch causes violent norm divergence at step ~200
    // (positions develop correlated representations → rho spike → weight poisoning).
    // With N(0,1): SNR ≈ 1.0/0.88 instead of 0.014/0.88, eliminating the transient.
    // gamma_final in LMHead is set to 1/sqrt(d_model) to keep initial logit std ≈ 1.0.
    Tensor::normal_(token_weights_, 0.0f, 1.0f, seed, stream);

    fprintf(stdout, "[EmbeddingLayer] Token weights: [%d, %d] Normal(0,1) seed=%llu (RMS≈1.0)\n",
            config_.vocab_size, config_.d_model, static_cast<unsigned long long>(seed));

    // ══════════════════════════════════════════════════════════════
    //  POSITION EMBEDDINGS: Only for LEARNED mode (PositionalEncodingType::NONE)
    //  ALiBi/RoPE inject position info inside attention, not residual stream
    // ══════════════════════════════════════════════════════════════
    if (config_.positional_encoding == HyperParameters::PositionalEncodingType::NONE) {
        position_weights_ = Tensor::zeros({config_.max_seq_len, config_.d_model}, stream, "embedding.position_weights");
        if (config_.requires_grad) {
            position_weights_.requires_grad_();
            position_weights_.ensure_grad();
        }
        Tensor::xavier_uniform_(position_weights_, seed + 200, stream);

        fprintf(stdout, "[EmbeddingLayer] Position weights: ALLOCATED [%d, %d] (learned/additive mode)\n",
                config_.max_seq_len, config_.d_model);
    } else {
        fprintf(stdout, "[EmbeddingLayer] Position weights: SKIPPED (%s uses attention-based encoding)\n",
                HyperParameters::positionalEncodingTypeToString(config_.positional_encoding));
    }
}

//======================================================//
//  Move Operations
//======================================================//

EmbeddingLayer::EmbeddingLayer(EmbeddingLayer&& other) noexcept
    : config_(other.config_)
    , token_weights_(std::move(other.token_weights_))
    , position_weights_(std::move(other.position_weights_))
{
}

EmbeddingLayer& EmbeddingLayer::operator=(EmbeddingLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        token_weights_ = std::move(other.token_weights_);
        position_weights_ = std::move(other.position_weights_);
    }
    return *this;
}

} // namespace GRIM
