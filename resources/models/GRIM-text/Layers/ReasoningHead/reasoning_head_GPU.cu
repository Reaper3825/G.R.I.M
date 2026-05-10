//======================================================//
//  reasoning_head_GPU.cu
//  GPU-accelerated Reasoning Head layer
//  Pattern B: Layer Ownership — self-allocates weights
//
//  Owns: W_op [num_ops, d_total], b_op [num_ops],
//        w_arg1 [1, d_total], w_arg2 [1, d_total]
//
//  Forward: gather → concat → mean-pool → matmul projections
//  Backward: ReasoningHeadGradFn handles gather/concat/pool;
//            matmuls use standard MatmulGradFn via autograd::matmul
//======================================================//

#include "reasoning_head_GPU.hpp"

#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {

//======================================================//
//  CUDA Kernels
//======================================================//

static constexpr int kBlockSize = 256;

// Gather encoder hidden states at atom positions
// out[i, j] = encoder[positions[i], j]
__global__ void kernelGatherAtomHidden(
    float* __restrict__ out,
    const float* __restrict__ encoder,
    const int* __restrict__ positions,
    int num_atoms,
    int d_model,
    int total_tokens
) {
    const int i = blockIdx.x;
    if (i >= num_atoms) return;

    const int pos = positions[i];
    // Bounds check — kernel-side guard (host validates before launch too)
    if (pos < 0 || pos >= total_tokens) return;

    const float* src = encoder + static_cast<size_t>(pos) * d_model;
    float* dst = out + static_cast<size_t>(i) * d_model;

    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        dst[j] = src[j];
    }
}

// Concat H_atoms [num_atoms, d_model] and atom_emb [num_atoms, atom_dim]
// into Z [num_atoms, d_total] where d_total = d_model + atom_dim
__global__ void kernelConcatFeatures(
    float* __restrict__ out,
    const float* __restrict__ h_atoms,
    const float* __restrict__ atom_emb,
    int num_atoms,
    int d_model,
    int atom_dim,
    int d_total
) {
    const int i = blockIdx.x;
    if (i >= num_atoms) return;

    float* dst = out + static_cast<size_t>(i) * d_total;
    const float* src_h = h_atoms + static_cast<size_t>(i) * d_model;
    const float* src_e = atom_emb + static_cast<size_t>(i) * atom_dim;

    for (int j = threadIdx.x; j < d_total; j += blockDim.x) {
        if (j < d_model) {
            dst[j] = src_h[j];
        } else {
            dst[j] = src_e[j - d_model];
        }
    }
}

// Mean pool Z [num_atoms, d_total] → z_pool [d_total]
// Each thread handles one output dimension and sums serially over atoms
__global__ void kernelMeanPool(
    float* __restrict__ out,
    const float* __restrict__ Z,
    int num_atoms,
    int d_total
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_total) return;

    float sum = 0.0f;
    for (int i = 0; i < num_atoms; ++i) {
        sum += Z[static_cast<size_t>(i) * d_total + j];
    }
    out[j] = sum / static_cast<float>(num_atoms > 0 ? num_atoms : 1);
}

// Add bias to op_logits: out[j] += bias[j]
__global__ void kernelReasoningBias(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int num_ops
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ops) return;
    output[idx] += bias[idx];
}

// Backward: scatter grad_H_atoms into encoder_grad using atomicAdd
// encoder_grad[positions[i], j] += grad_H_atoms[i, j]
__global__ void kernelScatterAtomHiddenGrad(
    float* __restrict__ encoder_grad,
    const float* __restrict__ grad_h_atoms,
    const int* __restrict__ positions,
    int num_atoms,
    int d_model,
    int total_tokens
) {
    const int i = blockIdx.x;
    if (i >= num_atoms) return;

    const int pos = positions[i];
    if (pos < 0 || pos >= total_tokens) return;

    const float* src = grad_h_atoms + static_cast<size_t>(i) * d_model;
    float* dst = encoder_grad + static_cast<size_t>(pos) * d_model;

    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        atomicAdd(&dst[j], src[j]);
    }
}

// Backward: split grad_Z [num_atoms, d_total] into
//   grad_H_atoms [num_atoms, d_model] and grad_atom_emb [num_atoms, atom_dim]
__global__ void kernelSplitConcatGrad(
    const float* __restrict__ grad_Z,
    float* __restrict__ grad_h_atoms,
    float* __restrict__ grad_atom_emb,
    int num_atoms,
    int d_model,
    int atom_dim,
    int d_total
) {
    const int i = blockIdx.x;
    if (i >= num_atoms) return;

    const float* src = grad_Z + static_cast<size_t>(i) * d_total;

    if (grad_h_atoms) {
        float* dst_h = grad_h_atoms + static_cast<size_t>(i) * d_model;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
            dst_h[j] = src[j];
        }
    }

    if (grad_atom_emb) {
        float* dst_e = grad_atom_emb + static_cast<size_t>(i) * atom_dim;
        for (int j = threadIdx.x; j < atom_dim; j += blockDim.x) {
            dst_e[j] = src[d_model + j];
        }
    }
}

// Backward: mean-pool gradient broadcast
// grad_Z[i, j] += grad_pool[j] / num_atoms
__global__ void kernelMeanPoolBackward(
    float* __restrict__ grad_Z,
    const float* __restrict__ grad_pool,
    int num_atoms,
    int d_total
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_total) return;

    const float scale = 1.0f / static_cast<float>(num_atoms > 0 ? num_atoms : 1);
    const float val = grad_pool[j] * scale;

    for (int i = 0; i < num_atoms; ++i) {
        grad_Z[static_cast<size_t>(i) * d_total + j] += val;
    }
}

//======================================================//
//  Constructor
//======================================================//
ReasoningHeadLayer::ReasoningHeadLayer(const HyperParameters::ReasoningHeadConstructionHP& config,
                                       uint64_t seed,
                                       cudaStream_t init_stream)
    : config_(config)
{
    if (config_.d_model <= 0)
        throw std::runtime_error("ReasoningHeadLayer: d_model must be positive");
    if (config_.atom_embedding_dim <= 0)
        throw std::runtime_error("ReasoningHeadLayer: atom_embedding_dim must be positive");
    if (config_.num_ops <= 0)
        throw std::runtime_error("ReasoningHeadLayer: num_ops must be positive");
    if (!init_stream)
        throw std::runtime_error("ReasoningHeadLayer: init_stream is NULL");

    const int dt = d_total();

    // W_op: [num_ops, d_total]
    w_op_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(config_.num_ops, dt),
                          true, init_stream, "reasoning_head.W_op");
    w_op_.requires_grad_();
    w_op_.ensure_grad();
    Tensor::xavier_uniform_(w_op_, seed, init_stream);

    // b_op: [num_ops]
    b_op_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, config_.num_ops),
                          true, init_stream, "reasoning_head.b_op");
    b_op_.requires_grad_();
    b_op_.ensure_grad();

    // w_arg1: [1, d_total]
    w_arg1_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, dt),
                            true, init_stream, "reasoning_head.w_arg1");
    w_arg1_.requires_grad_();
    w_arg1_.ensure_grad();
    Tensor::xavier_uniform_(w_arg1_, seed + 1, init_stream);

    // w_arg2: [1, d_total]
    w_arg2_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, dt),
                            true, init_stream, "reasoning_head.w_arg2");
    w_arg2_.requires_grad_();
    w_arg2_.ensure_grad();
    Tensor::xavier_uniform_(w_arg2_, seed + 2, init_stream);

    fprintf(stdout, "[ReasoningHeadLayer] Initialized: d_model=%d, atom_dim=%d, d_total=%d, num_ops=%d\n",
            config_.d_model, config_.atom_embedding_dim, dt, config_.num_ops);
}

//======================================================//
//  Move Operations
//======================================================//
ReasoningHeadLayer::ReasoningHeadLayer(ReasoningHeadLayer&& other) noexcept
    : config_(other.config_)
    , w_op_(std::move(other.w_op_))
    , b_op_(std::move(other.b_op_))
    , w_arg1_(std::move(other.w_arg1_))
    , w_arg2_(std::move(other.w_arg2_))
{
}

ReasoningHeadLayer& ReasoningHeadLayer::operator=(ReasoningHeadLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        w_op_ = std::move(other.w_op_);
        b_op_ = std::move(other.b_op_);
        w_arg1_ = std::move(other.w_arg1_);
        w_arg2_ = std::move(other.w_arg2_);
    }
    return *this;
}

//======================================================//
//  Forward
//======================================================//
ReasoningHeadOutput ReasoningHeadLayer::forward(
    Tensor& encoder_output,
    Tensor& atom_embeddings,
    int* atom_positions,
    int num_atoms,
    int total_tokens,
    cudaStream_t stream,
    cublasHandle_t cublas_handle)
{
    // ════════════════════════════════════════════════════
    // Hard-fail validation (Rule 20)
    // ════════════════════════════════════════════════════
    if (!stream)
        throw std::runtime_error("ReasoningHeadLayer::forward: stream is NULL");
    if (!cublas_handle)
        throw std::runtime_error("ReasoningHeadLayer::forward: cublas_handle is NULL");
    autograd::set_autograd_cublas_handle(cublas_handle);
    if (!encoder_output.data)
        throw std::runtime_error("ReasoningHeadLayer::forward: encoder_output.data is NULL");
    if (num_atoms < 0)
        throw std::runtime_error("ReasoningHeadLayer::forward: num_atoms < 0 (" + std::to_string(num_atoms) + ")");
    if (total_tokens <= 0)
        throw std::runtime_error("ReasoningHeadLayer::forward: total_tokens <= 0 (" + std::to_string(total_tokens) + ")");

    if (num_atoms > 0) {
        if (!atom_embeddings.data)
            throw std::runtime_error("ReasoningHeadLayer::forward: atom_embeddings.data is NULL but num_atoms=" + std::to_string(num_atoms));
        if (!atom_positions)
            throw std::runtime_error("ReasoningHeadLayer::forward: atom_positions is NULL but num_atoms=" + std::to_string(num_atoms));
    }

    // Shape validation
    if (!encoder_output.shape.is_2d_layout())
        throw std::runtime_error("ReasoningHeadLayer::forward: encoder_output must be 2D");
    const auto& enc_shape = encoder_output.shape.as_2d();
    if (enc_shape.rows != total_tokens || enc_shape.cols != config_.d_model)
        throw std::runtime_error("ReasoningHeadLayer::forward: encoder_output shape mismatch: got [" +
            std::to_string(enc_shape.rows) + ", " + std::to_string(enc_shape.cols) +
            "], expected [" + std::to_string(total_tokens) + ", " + std::to_string(config_.d_model) + "]");

    if (num_atoms > 0) {
        if (!atom_embeddings.shape.is_2d_layout())
            throw std::runtime_error("ReasoningHeadLayer::forward: atom_embeddings must be 2D");
        const auto& emb_shape = atom_embeddings.shape.as_2d();
        if (emb_shape.rows != num_atoms || emb_shape.cols != config_.atom_embedding_dim)
            throw std::runtime_error("ReasoningHeadLayer::forward: atom_embeddings shape mismatch: got [" +
                std::to_string(emb_shape.rows) + ", " + std::to_string(emb_shape.cols) +
                "], expected [" + std::to_string(num_atoms) + ", " + std::to_string(config_.atom_embedding_dim) + "]");
    }

    // Weight shape validation
    const int dt = d_total();
    {
        const auto& ws = w_op_.shape.as_2d();
        if (ws.rows != config_.num_ops || ws.cols != dt)
            throw std::runtime_error("ReasoningHeadLayer::forward: W_op shape mismatch: got [" +
                std::to_string(ws.rows) + ", " + std::to_string(ws.cols) +
                "], expected [" + std::to_string(config_.num_ops) + ", " + std::to_string(dt) + "]");
    }
    {
        const auto& ws = w_arg1_.shape.as_2d();
        if (ws.rows != 1 || ws.cols != dt)
            throw std::runtime_error("ReasoningHeadLayer::forward: w_arg1 shape mismatch: got [" +
                std::to_string(ws.rows) + ", " + std::to_string(ws.cols) +
                "], expected [1, " + std::to_string(dt) + "]");
    }
    {
        const auto& ws = w_arg2_.shape.as_2d();
        if (ws.rows != 1 || ws.cols != dt)
            throw std::runtime_error("ReasoningHeadLayer::forward: w_arg2 shape mismatch: got [" +
                std::to_string(ws.rows) + ", " + std::to_string(ws.cols) +
                "], expected [1, " + std::to_string(dt) + "]");
    }

    // ════════════════════════════════════════════════════
    // num_atoms == 0: skip head, return null tensors
    // ════════════════════════════════════════════════════
    if (num_atoms == 0) {
        return ReasoningHeadOutput{};
    }

    // ════════════════════════════════════════════════════
    // Step 1: Gather  H_atoms [num_atoms, d_model]
    // ════════════════════════════════════════════════════
    Tensor h_atoms = Tensor::empty(
        TensorContract::TensorShape::make_BSM(num_atoms, config_.d_model),
        false, stream, "reasoning_head.h_atoms");

    kernelGatherAtomHidden<<<num_atoms, kBlockSize, 0, stream>>>(
        h_atoms.data,
        encoder_output.data,
        atom_positions,
        num_atoms,
        config_.d_model,
        total_tokens);

    // ════════════════════════════════════════════════════
    // Step 2: Concat  Z [num_atoms, d_total]
    // ════════════════════════════════════════════════════
    Tensor Z = Tensor::empty(
        TensorContract::TensorShape::make_BSM(num_atoms, dt),
        true, stream, "reasoning_head.Z");

    kernelConcatFeatures<<<num_atoms, kBlockSize, 0, stream>>>(
        Z.data,
        h_atoms.data,
        atom_embeddings.data,
        num_atoms,
        config_.d_model,
        config_.atom_embedding_dim,
        dt);

    // ════════════════════════════════════════════════════
    // Step 3: Mean-pool  z_pool [1, d_total]
    // ════════════════════════════════════════════════════
    Tensor z_pool = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, dt),
        true, stream, "reasoning_head.z_pool");

    {
        const int grid = (dt + kBlockSize - 1) / kBlockSize;
        kernelMeanPool<<<grid, kBlockSize, 0, stream>>>(
            z_pool.data, Z.data, num_atoms, dt);
    }

    // ════════════════════════════════════════════════════
    // Attach ReasoningHeadGradFn for gather/concat/pool backward
    // This sits beneath the matmul grad_fns — the matmul
    // inputs (z_pool, Z) will have this as their grad_fn
    // source for encoder_output and atom_embeddings.
    // ════════════════════════════════════════════════════
    bool needs_grad = encoder_output.requires_grad || atom_embeddings.requires_grad;
    if (needs_grad) {
        auto grad_fn = std::make_shared<ReasoningHeadGradFn>();
        grad_fn->num_atoms = num_atoms;
        grad_fn->total_tokens = total_tokens;
        grad_fn->d_model = config_.d_model;
        grad_fn->atom_dim = config_.atom_embedding_dim;
        grad_fn->d_total = dt;

        grad_fn->capture_encoder(encoder_output, stream);
        grad_fn->capture_atom_emb(atom_embeddings, stream);
        grad_fn->capture_positions(atom_positions, num_atoms, stream);

        // Both Z and z_pool should chain through this grad_fn
        // so that when matmul backward calls apply() on its input grad_fn,
        // it flows through gather/concat/pool backward.
        Z.is_leaf = false;
        Z.requires_grad = true;
        Z.grad_fn = grad_fn;

        z_pool.is_leaf = false;
        z_pool.requires_grad = true;
        z_pool.grad_fn = grad_fn;
    }

    // ════════════════════════════════════════════════════
    // Step 4: Op logits  z_pool @ W_op^T + b_op → [1, num_ops]
    // ════════════════════════════════════════════════════
    w_op_.requires_grad = true;

    Tensor op_logits = autograd::matmul(
        z_pool,
        w_op_,
        stream,
        z_pool.data,   // a_cache
        nullptr,       // b_cache (weights persist)
        true           // transpose_b
    );

    // Add bias
    if (b_op_.data) {
        const int grid = (config_.num_ops + kBlockSize - 1) / kBlockSize;
        kernelReasoningBias<<<grid, kBlockSize, 0, stream>>>(
            op_logits.data, b_op_.data, config_.num_ops);
    }

    // ════════════════════════════════════════════════════
    // Step 5: Arg1 logits  Z @ w_arg1^T → [num_atoms, 1]
    // ════════════════════════════════════════════════════
    w_arg1_.requires_grad = true;

    Tensor arg1_raw = autograd::matmul(
        Z,
        w_arg1_,
        stream,
        Z.data,
        nullptr,
        true
    );
    // Reshape to [1, num_atoms] for API consistency
    arg1_raw.shape = TensorContract::TensorShape::make_BSM(1, num_atoms);

    // ════════════════════════════════════════════════════
    // Step 6: Arg2 logits  Z @ w_arg2^T → [num_atoms, 1]
    // ════════════════════════════════════════════════════
    w_arg2_.requires_grad = true;

    Tensor arg2_raw = autograd::matmul(
        Z,
        w_arg2_,
        stream,
        Z.data,
        nullptr,
        true
    );
    arg2_raw.shape = TensorContract::TensorShape::make_BSM(1, num_atoms);

    // Check kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("ReasoningHeadLayer::forward: CUDA error after kernels: ") +
                                 cudaGetErrorString(err));
    }

    return ReasoningHeadOutput{
        std::move(op_logits),
        std::move(arg1_raw),
        std::move(arg2_raw)
    };
}

//======================================================//
//  ReasoningHeadGradFn Implementation
//======================================================//

ReasoningHeadGradFn::~ReasoningHeadGradFn() {
    if (saved_atom_positions) cudaFree(saved_atom_positions);
    if (owns_encoder_grad && encoder_grad) cudaFree(encoder_grad);
    if (owns_atom_emb_grad && atom_emb_grad) cudaFree(atom_emb_grad);
}

void ReasoningHeadGradFn::capture_encoder(Tensor& enc, cudaStream_t stream) {
    encoder_shape = enc.shape;
    encoder_grad_fn = enc.grad_fn;

    if (enc.requires_grad) {
        enc.ensure_grad();
        const size_t n = enc.numel();
        cudaMallocOrThrow(reinterpret_cast<void**>(&encoder_grad), n * sizeof(float), "ReasoningHeadGradFn_encoder_grad");
        cudaMemsetAsync(encoder_grad, 0, n * sizeof(float), stream);
        owns_encoder_grad = true;
    }
}

void ReasoningHeadGradFn::capture_atom_emb(Tensor& emb, cudaStream_t stream) {
    atom_emb_shape = emb.shape;
    atom_emb_grad_fn = emb.grad_fn;

    if (emb.requires_grad) {
        emb.ensure_grad();
        const size_t n = emb.numel();
        cudaMallocOrThrow(reinterpret_cast<void**>(&atom_emb_grad), n * sizeof(float), "ReasoningHeadGradFn_atom_emb_grad");
        cudaMemsetAsync(atom_emb_grad, 0, n * sizeof(float), stream);
        owns_atom_emb_grad = true;
    }
}

void ReasoningHeadGradFn::capture_positions(const int* d_positions, int n_atoms, cudaStream_t stream) {
    num_atoms = n_atoms;
    if (n_atoms > 0 && d_positions) {
        const size_t bytes = static_cast<size_t>(n_atoms) * sizeof(int);
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_atom_positions), bytes, "ReasoningHeadGradFn_saved_positions");
        cudaMemcpyAsync(saved_atom_positions, d_positions, bytes,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

void ReasoningHeadGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("reasoning_head", this);

    if (applied) return;
    applied = true;

    if (!grad_output.data)
        throw std::runtime_error("ReasoningHeadGradFn::apply: grad_output.data is NULL");

    // grad_output is the gradient of Z [num_atoms, d_total] from concat
    // (the matmul grad_fns have already backpropped through pool→Z and Z→arg)
    //
    // We need to:
    //  1. Split grad_Z into grad_H_atoms [num_atoms, d_model] and grad_atom_emb [num_atoms, atom_dim]
    //  2. Scatter grad_H_atoms into encoder_grad using atomicAdd
    //  3. Accumulate grad_atom_emb into atom_embeddings grad

    if (num_atoms <= 0) {
        clearCurrentGradFnOp();
        return;
    }

    // Allocate temporaries for split
    float* grad_h_atoms = nullptr;
    float* grad_atom_emb_local = nullptr;

    if (encoder_grad) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_h_atoms), static_cast<size_t>(num_atoms) * d_model * sizeof(float), "ReasoningHeadGradFn_grad_h_atoms");
    }
    if (atom_emb_grad) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_atom_emb_local), static_cast<size_t>(num_atoms) * atom_dim * sizeof(float), "ReasoningHeadGradFn_grad_atom_emb_local");
    }

    // Split concat gradient
    kernelSplitConcatGrad<<<num_atoms, kBlockSize, 0, stream>>>(
        grad_output.data,
        grad_h_atoms,
        grad_atom_emb_local,
        num_atoms,
        d_model,
        atom_dim,
        d_total);

    // Scatter grad_H_atoms → encoder_grad with atomicAdd
    if (grad_h_atoms && encoder_grad && saved_atom_positions) {
        kernelScatterAtomHiddenGrad<<<num_atoms, kBlockSize, 0, stream>>>(
            encoder_grad,
            grad_h_atoms,
            saved_atom_positions,
            num_atoms,
            d_model,
            total_tokens);

        // Chain into encoder's grad_fn
        if (encoder_grad_fn) {
            Tensor enc_grad_tensor = Tensor::from_ptr(
                encoder_grad, encoder_shape, false, false, "reasoning_head_enc_grad");
            enc_grad_tensor.stream = stream;
            encoder_grad_fn->apply(enc_grad_tensor, stream);
        }
    }

    // Accumulate grad_atom_emb into atom_embeddings grad
    if (grad_atom_emb_local && atom_emb_grad) {
        const size_t n = static_cast<size_t>(num_atoms) * atom_dim;
        // Simple element-wise add into the grad buffer
        // (kernel_accumulate_grad pattern from the codebase)
        cudaMemcpyAsync(atom_emb_grad, grad_atom_emb_local,
                        n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        if (atom_emb_grad_fn) {
            Tensor emb_grad_tensor = Tensor::from_ptr(
                atom_emb_grad, atom_emb_shape, false, false, "reasoning_head_emb_grad");
            emb_grad_tensor.stream = stream;
            atom_emb_grad_fn->apply(emb_grad_tensor, stream);
        }
    }

    // Free temporaries
    if (grad_h_atoms) cudaFreeAsync(grad_h_atoms, stream);
    if (grad_atom_emb_local) cudaFreeAsync(grad_atom_emb_local, stream);

    clearCurrentGradFnOp();
}

void ReasoningHeadGradFn::release_saved() {
    if (saved_atom_positions) { cudaFree(saved_atom_positions); saved_atom_positions = nullptr; }
    if (owns_encoder_grad && encoder_grad) { cudaFree(encoder_grad); encoder_grad = nullptr; }
    if (owns_atom_emb_grad && atom_emb_grad) { cudaFree(atom_emb_grad); atom_emb_grad = nullptr; }
    encoder_grad_fn.reset();
    atom_emb_grad_fn.reset();
}

}  // namespace GRIM
