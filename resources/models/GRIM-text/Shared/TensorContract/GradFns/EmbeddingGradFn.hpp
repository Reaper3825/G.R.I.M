#pragma once
//======================================================//
//  EmbeddingGradFn.hpp
//  Single-owner header for the embedding-lookup autograd node.
//
//  Pattern (mirrors Phase1_Startup discipline):
//    - One concern per .hpp/.cu pair
//    - Header = declaration only (struct shape + member signatures)
//    - .cu owns the kernel(s) and the method bodies
//
//  Owns:
//    - struct EmbeddingGradFn      (backward node for embedding lookup)
//    - kernel_embedding_forward    (defined in EmbeddingGradFn.cu)
//    - kernel_embedding_backward   (defined in EmbeddingGradFn.cu)
//    - autograd::embedding(...)    (forward op; defined in EmbeddingGradFn.cu)
//======================================================//

#include "../TensorContract_GPU.hpp"  // GradFn, Tensor, TensorShape

#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

/**
 * EmbeddingGradFn — Backward node for embedding lookup (ISSUE #48 pattern).
 *
 * STORES STABLE DATA, NOT Tensor*:
 *   - weight_grad         : raw device pointer into the weight tensor's grad
 *   - weight_shape        : captured at forward time (used to build the grad view)
 *   - weight_grad_fn      : shared_ptr to upstream grad_fn (for chain continuation)
 *   - token_ids           : device-resident int IDs (optionally owned/copied)
 *
 * Backward kernel (kernel_embedding_backward) does an atomic scatter-add of
 *   grad_output[token_idx, :] * embedding_scale
 * into weight_grad[token_id, :], with bounds-checking on token_id.
 */
struct EmbeddingGradFn : public GradFn {
    // Captured weight metadata (stable across backward)
    bool weight_requires_grad = false;
    float* weight_grad = nullptr;
    TensorContract::TensorShape weight_shape;
    std::shared_ptr<GradFn> weight_grad_fn;

    // Captured token IDs (device pointer; may be owned)
    int* token_ids = nullptr;
    bool owns_token_ids = false;
    int num_tokens = 0;
    int d_model = 0;
    int vocab_size = 0;            ///< RULE 20: stored for OOB bounds checking in backward kernel
    float embedding_scale = 1.0f;  ///< Issue #140: 1.0f in production (AIAYN sqrt(d_model) removed)

    EmbeddingGradFn();
    ~EmbeddingGradFn() override;

    /// Capture upstream weight metadata (shape, requires_grad, grad pointer, grad_fn).
    void capture_weight(Tensor& w);

    /// Save token IDs needed for the backward scatter. If copy_ids is true,
    /// device memory is allocated and the IDs are copied (owns_token_ids = true).
    void save(const int* ids, int tokens, int d, bool copy_ids, cudaStream_t stream);

    /// Backward: scatter-add grad_output rows into weight_grad rows by token_id.
    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;

    /// Release saved device memory (token IDs if owned) and drop chain refs.
    void release_saved() override;
};

/**
 * Embedding lookup with fixed hard token-type gate:
 *   output[i,d] = weight[token_ids[i],d] * embedding_scale
 *                 if d is in the token-layout type subspace, else 0.
 *
 * Forward gathers from a 2D weight table [vocab_size, d_model], applies the
 * same fixed token-layout type gate used by the LM head, and writes a BSM-shaped
 * [num_tokens, d_model] activation. If weight.requires_grad, an EmbeddingGradFn
 * is attached to the result so that the eventual backward scatter-adds into
 * weight.grad only inside the active type subspace.
 *
 * Public signature is also declared in TensorContract_GPU.hpp under
 * namespace GRIM::autograd; the definition lives in EmbeddingGradFn.cu.
 */
// Tensor embedding(...) is declared in TensorContract_GPU.hpp; defined in EmbeddingGradFn.cu

}  // namespace autograd
}  // namespace GRIM
