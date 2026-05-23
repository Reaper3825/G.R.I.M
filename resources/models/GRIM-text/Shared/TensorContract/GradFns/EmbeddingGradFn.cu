//======================================================//
//  EmbeddingGradFn.cu
//  Implementation of the embedding-lookup autograd node.
//
//  Owns:
//    - kernel_embedding_forward    (anonymous namespace, this TU)
//    - kernel_embedding_backward   (anonymous namespace, this TU)
//    - EmbeddingGradFn methods     (capture_weight, save, apply, release_saved)
//    - autograd::embedding(...)    (forward op)
//
//  Single-file extraction from TensorContract_GPU.cu — same math, same
//  ISSUE #48 stable-data pattern, same Rule 20 OOB bounds checks.
//======================================================//

#include "EmbeddingGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../TokenTypeGate.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

// Mirrors AG_TRACE in TensorContract_GPU.cu — gated on the global verbose flag.
#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

// ─── Forward declarations: defined in TensorContract_GPU.cu at global scope ───
void setCurrentGradFnOp(const char* op_name, void* gradfn_ptr);
void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════════════
// Anonymous-namespace helpers — internal linkage per-TU (no ODR conflict
// with the same names in TensorContract_GPU.cu / AutogradAttention.cu).
// ═══════════════════════════════════════════════════════════════════════════
namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

// Embedding forward: gather from embedding table with optional scaling and a
// hard token-layout type gate. Inactive dimensions are exactly zero.
__global__ void kernel_embedding_forward(
    const int* token_ids,       // [tokens]
    const float* weight,        // [vocab_size, d_model]
    float* output,              // [tokens, d_model]
    int tokens,
    int d_model,
    int vocab_size,             // RULE 20: Bounds check parameter
    float embedding_scale       // Scale factor (1.0 for production — Issue #140 removed sqrt(d_model))
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const int token_id = token_ids[token_idx];
    // RULE 20: Crash loud on OOB token ID — __trap() works in Release (assert compiles out)
    if (token_id < 0 || token_id >= vocab_size) {
        printf("FATAL: OOB token_id=%d (vocab_size=%d) at token_idx=%d in kernel_embedding_forward\n",
               token_id, vocab_size, token_idx);
        __trap();
    }
    float* output_row = output + static_cast<size_t>(token_idx) * d_model;

    if (token_id == GRIM::Tokenizer::PAD_TOKEN_ID) {
        for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
            output_row[i] = 0.0f;
        }
        return;
    }

    const float* weight_row = weight + static_cast<size_t>(token_id) * d_model;

    const auto gate = ::TensorContract::tokenTypeGateRangeForTokenId(token_id, d_model, vocab_size);
    if (gate.width <= 0) {
        printf("FATAL: invalid token type gate for token_id=%d d_model=%d vocab_size=%d in kernel_embedding_forward\n",
               token_id, d_model, vocab_size);
        __trap();
    }

    // Gather with scaling and hard gate:
    // output[token_idx,d] = weight[token_id,d] * scale inside the type subspace, else 0.
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        output_row[i] = (i >= gate.start && i < gate.end)
            ? weight_row[i] * embedding_scale
            : 0.0f;
    }
}

// Embedding backward: scatter-add gradients to embedding table only in the
// active token-layout type subspace.
__global__ void kernel_embedding_backward(
    const float* grad_output,   // [tokens, d_model]
    const int* token_ids,       // [tokens]
    float* grad_weight,         // [vocab_size, d_model]
    int tokens,
    int d_model,
    int vocab_size,             // RULE 20: Bounds check parameter
    float embedding_scale       // Scale factor from forward (for chain rule)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const int token_id = token_ids[token_idx];
    // RULE 20: Crash loud on OOB token ID — __trap() works in Release (assert compiles out)
    if (token_id < 0 || token_id >= vocab_size) {
        printf("FATAL: OOB token_id=%d (vocab_size=%d) at token_idx=%d in kernel_embedding_backward\n",
               token_id, vocab_size, token_idx);
        __trap();
    }

    if (token_id == GRIM::Tokenizer::PAD_TOKEN_ID) {
        return;
    }

    const float* token_grad = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* weight_grad = grad_weight + static_cast<size_t>(token_id) * d_model;

    const auto gate = ::TensorContract::tokenTypeGateRangeForTokenId(token_id, d_model, vocab_size);
    if (gate.width <= 0) {
        printf("FATAL: invalid token type gate for token_id=%d d_model=%d vocab_size=%d in kernel_embedding_backward\n",
               token_id, d_model, vocab_size);
        __trap();
    }

    // Scatter-add: weight_grad[token_id] += grad_output[token_idx] * scale
    // Chain rule: if forward was y = w * scale, then grad_w = grad_y * scale
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        if (i >= gate.start && i < gate.end) {
            atomicAdd(&weight_grad[i], token_grad[i] * embedding_scale);
        }
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

// ═══════════════════════════════════════════════════════════════════════════
// EmbeddingGradFn — method bodies
// ═══════════════════════════════════════════════════════════════════════════

EmbeddingGradFn::EmbeddingGradFn() {
    op_name = "embedding";
}

EmbeddingGradFn::~EmbeddingGradFn() {
    release_saved();
}

void EmbeddingGradFn::capture_weight(Tensor& w) {
    weight_requires_grad = w.requires_grad;
    weight_shape = w.shape;
    vocab_size = w.shape.as_2d().rows;
    if (!vocab_size) {
        throw std::runtime_error("EmbeddingGradFn::capture_weight: vocab_size is 0 — weight shape is invalid");
    }

    // Copy shared_ptr to captured grad_fn
    weight_grad_fn = w.grad_fn;

    if (weight_requires_grad) {
        w.ensure_grad();
        weight_grad = w.grad_data();  // ISSUE #59: Use accessor
    }
}

void EmbeddingGradFn::save(const int* ids, int tokens, int d, bool copy_ids, cudaStream_t stream) {
    num_tokens = tokens;
    d_model = d;

    AG_TRACE("[EmbeddingGradFn::save] ids=%p tokens=%d d=%d copy_ids=%s\n",
            (void*)ids, tokens, d, copy_ids ? "true" : "false");

    if (copy_ids) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&token_ids), tokens * sizeof(int), "EmbeddingGradFn_token_ids");
        owns_token_ids = true;
        const cudaError_t copy_err = cudaMemcpyAsync(
            token_ids, ids, tokens * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("EmbeddingGradFn::save: cudaMemcpyAsync(token_ids) failed: ") +
                                     cudaGetErrorString(copy_err));
        }
    } else {
        token_ids = const_cast<int*>(ids);
        owns_token_ids = false;
    }
}

void EmbeddingGradFn::apply_impl(const Tensor& grad_output, cudaStream_t stream) {
    // RULE 20: Track current operation for error context
    setCurrentGradFnOp("embedding", this);

    AG_TRACE("[EmbeddingGradFn::apply] ENTER - tokens=%d d=%d\n",
            num_tokens, d_model);

    // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
    if (applied) {
        AG_TRACE("[EmbeddingGradFn::apply] SKIP - already applied\n");
        return;
    }

    if (!weight_requires_grad) {
        AG_TRACE("[EmbeddingGradFn::apply] SKIP - weight does not require grad\n");
        applied = true;
        return;
    }
    if (!token_ids) {
        throw std::runtime_error("EmbeddingGradFn::apply: token_ids is NULL - save() must store token IDs for backward scatter");
    }

    if (!weight_grad) {
        throw std::runtime_error("EmbeddingGradFn::apply: weight_grad is NULL - capture_weight() must be called first");
    }

    if (!grad_output.data) {
        throw std::runtime_error("EmbeddingGradFn::apply: grad_output.data is NULL");
    }
    const size_t expected_count = static_cast<size_t>(num_tokens) * static_cast<size_t>(d_model);
    const size_t actual_count = grad_output.numel();
    if (actual_count != expected_count) {
        throw std::runtime_error("EmbeddingGradFn::apply: grad_output element count mismatch. expected=" +
                                 std::to_string(expected_count) + " got=" +
                                 std::to_string(actual_count));
    }

    // PyTorch-style direct accumulation — embedding grad writes
    // to same buffer where LM head grad already lives. Natural ~90% cancellation
    // for frequent tokens acts as frequency-proportional regularization.
    kernel_embedding_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, token_ids, weight_grad, num_tokens, d_model, vocab_size, embedding_scale);
    trackKernelLaunch("kernel_embedding_backward", stream);
    applied = true;

    // CONTINUE AUTOGRAD CHAIN using stored grad_fn
    if (weight_grad_fn) {
        Tensor view;
        view.data = weight_grad; view.shape = weight_shape;
        view.owns_data = false; view.stream = stream;
        weight_grad_fn->apply(view, stream);
        // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
    }
}

void EmbeddingGradFn::release_saved() {
    GradFn::release_saved();
    if (owns_token_ids && token_ids) { cudaFree(token_ids); token_ids = nullptr; }
    weight_grad = nullptr;
    weight_grad_fn.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::embedding — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens, cudaStream_t stream, float embedding_scale) {
    if (!weight.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::embedding: weight must be 2D [vocab_size, d_model]");
    }
    if (!token_ids) {
        throw std::invalid_argument("autograd::embedding: token_ids is NULL");
    }
    if (num_tokens <= 0) {
        throw std::invalid_argument("autograd::embedding: num_tokens must be > 0");
    }
    if (!weight.data) {
        throw std::invalid_argument("autograd::embedding: weight.data is NULL");
    }
    if (embedding_scale <= 0.0f) {
        throw std::invalid_argument("autograd::embedding: embedding_scale must be > 0");
    }

    const int d_model = weight.shape.as_2d().cols;
    if (d_model < 4) {
        throw std::invalid_argument("autograd::embedding: d_model must be >= 4 for hard token-type gate, got " +
                                    std::to_string(d_model));
    }
    auto output_shape = ::TensorContract::TensorShape::make_BSM(num_tokens, d_model);
    Tensor result = Tensor::empty(output_shape, weight.requires_grad, stream, "embedding_result");

    // Forward: gather from weight table with scaling and fixed hard type gate.
    // Issue #140: Scale is 1.0f in production (AIAYN sqrt(d_model) removed for tied weights)
    const int vocab_size = weight.shape.as_2d().rows;
    kernel_embedding_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        token_ids, weight.data, result.data, num_tokens, d_model, vocab_size, embedding_scale);
    trackKernelLaunch("kernel_embedding_forward", stream);

    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (weight.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<EmbeddingGradFn>();
        grad_fn->capture_weight(const_cast<Tensor&>(weight));
        grad_fn->save(token_ids, num_tokens, d_model, true, stream);
        grad_fn->embedding_scale = embedding_scale;   // Store for backward scaling
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
