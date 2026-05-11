//======================================================//
//  Embedding_GPU.cu
//  GPU-accelerated Embedding layer using autograd
//  Pattern B: Layer Ownership — self-allocates weights
//
//  Owns: token_weights [vocab_size, d_model].
//  Position information is injected inside attention (ALiBi/RoPE);
//  the learned position-embedding path has been removed (Rule 26).
//
//  PyTorch equivalent:
//    class Embedding(nn.Module):
//        def __init__(self, vocab_size, d_model):
//            self.token_embed = nn.Embedding(vocab_size, d_model)
//        def forward(self, input_ids):
//            return self.token_embed(input_ids)
//======================================================//

#include "Embedding_GPU.hpp"

#include <stdexcept>
#include <cstdio>
#include <string>

namespace GRIM {

//======================================================//
//  Self-Allocating Constructor (Pattern B)
//======================================================//

EmbeddingLayer::EmbeddingLayer(const HyperParameters::EmbeddingLayerConstructionHP& hp,
                               uint64_t seed,
                               cudaStream_t stream,
                               bool requires_grad)
    : hp_(hp)
{
    // Rule 20: Fail loud on invalid configuration
    if (hp_.vocab_size <= 0) {
        throw std::runtime_error("EmbeddingLayer: vocab_size must be positive, got " + std::to_string(hp_.vocab_size));
    }
    if (hp_.d_model <= 0) {
        throw std::runtime_error("EmbeddingLayer: d_model must be positive, got " + std::to_string(hp_.d_model));
    }
    if (!stream) {
        throw std::runtime_error("EmbeddingLayer: stream is NULL — CUDA stream required for allocation");
    }

    // ══════════════════════════════════════════════════════════════
    //  TOKEN EMBEDDINGS: Always allocated [vocab_size, d_model]
    // ══════════════════════════════════════════════════════════════
    token_weights_ = Tensor::zeros({hp_.vocab_size, hp_.d_model}, stream, "embedding.token_weights");
    if (requires_grad) {
        token_weights_.requires_grad_();
        token_weights_.ensure_grad();  // Allocate grad NOW so share_grad() works for LM head tying
    }
    Tensor::xavier_uniform_(token_weights_, seed, stream);

    fprintf(stdout, "[EmbeddingLayer] Token weights: [%d, %d] Xavier seed=%llu\n",
            hp_.vocab_size, hp_.d_model, static_cast<unsigned long long>(seed));
}

//======================================================//
//  Move Operations
//======================================================//

EmbeddingLayer::EmbeddingLayer(EmbeddingLayer&& other) noexcept
    : hp_(other.hp_)
    , token_weights_(std::move(other.token_weights_))
{
}

EmbeddingLayer& EmbeddingLayer::operator=(EmbeddingLayer&& other) noexcept {
    if (this != &other) {
        hp_ = other.hp_;
        token_weights_ = std::move(other.token_weights_);
    }
    return *this;
}

} // namespace GRIM
