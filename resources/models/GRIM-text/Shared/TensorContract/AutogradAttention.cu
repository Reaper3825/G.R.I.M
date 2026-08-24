//======================================================//
//  AutogradAttention.cu
//  Autograd attention operations extracted from TensorContract_GPU.cu
//  Contains: MatMul, ScaledDotProductAttention, ReshapeBHSD,
//            SplitAndReshapeQKV, RoPE rotation
//======================================================//
#include "TensorContract_GPU.hpp"
#include "AttentionEpilogue.hpp"
#include "AutogradQKVDiagnostics.hpp"
#include "LMHeadGemmDiagnostics.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../CudaAllocUtils.hpp"
#include "GradientAccumulation.hpp"
#include "../TensorConversion/TensorConversion.hpp"
#include "../../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Layers/Encoding/AblationFlags.hpp"
#include "../PBM/PositionalBiasMethod.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <atomic>

// Mirror of AG_TRACE from TensorContract_GPU.cu
#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

// ─── Forward declaration: defined in TensorContract_GPU.cu at global scope ───
void trackCublasCall(const char* op_name, cublasHandle_t handle, cudaStream_t stream, cublasStatus_t status);

// ═══════════════════════════════════════════════════════════════════════════
// Attention off-by-one (softmax1 / zero-value sink) epilogue
//
// Converts the standard-softmax FlashAttention result into softmax-off-by-one
// (Miller 2023, "Attention Is Off By One") WITHOUT touching the vendored kernel.
// softmax1(s)_i = exp(s_i) / (1 + Σ_j exp(s_j)) adds a phantom logit at 0 with a
// ZERO value vector — a free "attend to nothing" slot. A head that wants to no-op
// parks its mass there and emits ~0 instead of mean-pooling prior tokens, which is
// the residual-stream common-mode (rho) injector at the first attention layer.
//
// Exact post-process of the standard output O_std and per-row logsumexp lse:
//     O_obo   = O_std · σ(lse)            (σ = logistic sigmoid)
//     lse_obo = softplus(lse) = log(1 + e^{lse})
// because the softmax1 denominator 1 + Σe^{s_j} = 1 + e^{lse}, so every weight
// scales by e^{lse}/(1 + e^{lse}) = σ(lse) and the saved logsumexp becomes
// log(1 + e^{lse}). Feeding (O_obo, lse_obo) to the UNCHANGED FlashAttention
// backward yields exact softmax1 gradients: the phantom slot's value is zero, so
// it contributes no dP term and the Jacobian form P_i(δ_ij − P_j) is unchanged
// (P is recomputed from lse_obo).
//
// out layout: [B, S, H, D] (BSHD). lse layout: [B, H, S]. Grid is laid out
// b-major→h→s, so blockIdx.x indexes lse directly.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

__global__ void kernelAttentionOffByOneEpilogue(
    __nv_bfloat16* __restrict__ out_bshd,   // [B, S, H, D] — scaled in place
    float* __restrict__ lse_bhs,            // [B, H, S]    — softplus'd in place
    int seq_len, int num_heads, int head_dim) {
    const int bid = blockIdx.x;             // == (b*H + h)*S + s  == lse index
    const int s = bid % seq_len;
    const int h = (bid / seq_len) % num_heads;
    const int b = bid / (seq_len * num_heads);

    const float lse = lse_bhs[bid];
    // Numerically stable sigmoid: σ(lse) = O_obo / O_std scale factor.
    const float sig = (lse >= 0.0f)
        ? 1.0f / (1.0f + __expf(-lse))
        : __expf(lse) / (1.0f + __expf(lse));
    const size_t base =
        ((static_cast<size_t>(b) * seq_len + s) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float o = __bfloat162float(out_bshd[base + d]);
        out_bshd[base + d] = __float2bfloat16(o * sig);
    }
    // Numerically stable softplus: lse_obo = log(1 + e^{lse}).
    if (threadIdx.x == 0) {
        lse_bhs[bid] = fmaxf(lse, 0.0f) + log1pf(__expf(-fabsf(lse)));
    }
}

// launchAttentionOffByOneEpilogue() is defined below in GRIM::autograd (it has a
// header declaration in TensorContract_GPU.hpp) so the inference KV-cache decode
// path can reapply the exact same softmax1 post-process. The kernel above stays
// in this anonymous namespace; the launcher reaches it by unqualified TU lookup.

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Open GRIM::autograd namespace — all functions below are in this namespace
// (matching TensorContract_GPU.cu structure)
// ═══════════════════════════════════════════════════════════════════════════
namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

using Batching::BatchPayload;

// Attention off-by-one (softmax1) epilogue launcher. Exposed via
// TensorContract_GPU.hpp so both the training SDPA path and the inference
// KV-cache decode facade apply the identical post-process. Reaches the kernel in
// the anonymous namespace above by unqualified translation-unit lookup.
void launchAttentionOffByOneEpilogue(
    __nv_bfloat16* out_bshd, float* lse_bhs,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream) {
    const int blocks = batch_size * num_heads * seq_len;
    if (blocks <= 0) {
        return;
    }
    const int threads = head_dim < 256 ? head_dim : 256;
    kernelAttentionOffByOneEpilogue<<<blocks, threads, 0, stream>>>(
        out_bshd, lse_bhs, seq_len, num_heads, head_dim);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launchAttentionOffByOneEpilogue: kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

inline void throwIfCudaFailed(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

static void requireEncoderAttentionHP(const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
                                      const char* caller) {
    if (hp.d_model <= 0 || hp.num_heads <= 0 || hp.num_kv_heads <= 0 || hp.head_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": EncoderSelfAttentionHP has invalid dimensions");
    }
}

// Issue #142: applyLmHeadGradCorrections DELETED.
// Centering is now INSIDE autograd graph (Issues #125/#132):
//   CenterRowsGradFn::apply() row-centers grad_A in backward
//   CenterColumnsGradFn::apply() column-centers grad_A in backward
// The old external centerGradientsKernel was redundant (row centering is idempotent)
// and wasted GPU time (kernel launch + cudaStreamSynchronize + 6x fprintf per call).

/**
 * MatMulGradFn - Backward for matrix multiplication (TAPE-BASED)
 * Forward: C = A @ B  [M,K] @ [K,N] = [M,N]
 * Backward:
 *   grad_A = grad_C @ B^T  [M,N] @ [N,K] = [M,K]
 *   grad_B = A^T @ grad_C  [K,M] @ [M,N] = [K,N]
 *
 * TAPE-BASED: Saves owned forward copies of A/B for backward and writes directly to grad buffers.
 * 
 * ISSUE #48 FIX: Stores stable data (shapes, grad pointers, grad_fn) instead of Tensor* 
 * which may become dangling after the forward function returns.
 * 
 * ISSUE #55 FIX: For non-leaf (activation) tensors, owns grad buffers.
 * For leaf (weight) tensors, uses their persistent grad buffers directly.
 */
struct MatMulGradFn : public GradFn {
    // ISSUE #48: Don't store Tensor* - they become dangling after forward returns
    // Instead, store what we actually need for backward:
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    
    std::shared_ptr<Tensor> a_gradient;
    std::shared_ptr<Tensor> b_gradient;
    
    // ISSUE #51 FIX: Own copies of cached activations instead of dangling external pointers.
    // Same pattern as GeluGradFn/RMSNormGradFn — allocate, copy, wrap in shared_ptr.
    std::shared_ptr<float> owned_cached_a;  // Owned GPU copy of A activations (for grad_B)
    std::shared_ptr<float> owned_cached_b;  // Owned GPU copy of B activations (for grad_A)
    const float* cached_a = nullptr;  // Points to owned_cached_a.get()
    const float* cached_b = nullptr;  // Points to owned_cached_b.get()
    int M = 0, K = 0, N = 0;   // Dimensions
    cublasHandle_t cublas_handle = nullptr;
    bool transpose_b = false;  // Was B transposed in forward?
    cudaStream_t cache_stream = nullptr;  // Stream for cache copy operations
    const char* a_name = nullptr;
    const char* b_name = nullptr;
    
    MatMulGradFn() { op_name = "matmul"; }
    
    // shared_ptr members destruct automatically
    

    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_name = a.name;
        b_name = b.name;

        if (a_requires_grad) {
            a_gradient = capture_input_gradient(
                a, stream, "MatMulGradFn::capture_inputs A");
            AG_TRACE("[MatMulGradFn] Captured grad_A Tensor data: %p\n", (void*)a_gradient->data);
        }
        if (b_requires_grad) {
            b_gradient = capture_input_gradient(
                b, stream, "MatMulGradFn::capture_inputs B");
            AG_TRACE("[MatMulGradFn] Captured grad_B Tensor data: %p\n", (void*)b_gradient->data);
        }
    }

    // ISSUE #51 FIX: Copy forward data to owned buffers instead of storing dangling pointers.
    // Same owned-save pattern as GeluGradFn::set_cache_copy / RMSNormGradFn::set_cache_copy.
    void set_cache_copy(const float* a_forward, const float* b_forward, int m, int k, int n,
                        cublasHandle_t handle, cudaStream_t stream, bool transB = false) {
        transpose_b = transB;
        M = m; K = k; N = n;
        cublas_handle = handle;
        cache_stream = stream;
        
        // Validate that required forward tensors are available for owned copies.
        if (a_requires_grad && !b_forward) {
            throw std::runtime_error(
                "MatMulGradFn::set_cache_copy: b_forward is NULL but input_a requires grad "
                "(A.grad=true requires saved B for grad_A)");
        }
        if (b_requires_grad && !a_forward) {
            throw std::runtime_error(
                "MatMulGradFn::set_cache_copy: a_forward is NULL but input_b requires grad "
                "(B.grad=true requires saved A for grad_B)");
        }
        
        // Allocate and copy A forward tensor (needed for grad_B = A^T @ grad_C)
        if (b_requires_grad && a_forward) {
            const size_t a_size = static_cast<size_t>(m) * k;
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), a_size * sizeof(float), "MatMulGradFn_cache_a");
            cudaMemcpyAsync(buffer, a_forward, a_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            owned_cached_a = std::shared_ptr<float>(buffer, [](float* p) {
                queueForDeferredCleanup(p);
            });
            cached_a = owned_cached_a.get();
            AG_TRACE("[MatMulGradFn] Copied cache_a: %zu floats to %p\n", a_size, (void*)cached_a);
        }
        
        // Allocate and copy B forward tensor (needed for grad_A = grad_C @ B^T)
        if (a_requires_grad && b_forward) {
            // B shape: [K,N] normal or [N,K] if transposed
            const size_t b_size = static_cast<size_t>(k) * n;
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), b_size * sizeof(float), "MatMulGradFn_cache_b");
            cudaMemcpyAsync(buffer, b_forward, b_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            owned_cached_b = std::shared_ptr<float>(buffer, [](float* p) {
                queueForDeferredCleanup(p);
            });
            cached_b = owned_cached_b.get();
            AG_TRACE("[MatMulGradFn] Copied cache_b: %zu floats to %p\n", b_size, (void*)cached_b);
        }
    }
    
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        (void)backward_bindings;
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("matmul", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        if (!cublas_handle) {
            throw std::runtime_error("MatMulGradFn::apply: cublas_handle is NULL");
        }
        if (a_requires_grad && !a_gradient) {
            throw std::runtime_error("MatMulGradFn::apply: A gradient Tensor is NULL - capture_inputs() must be called");
        }
        if (b_requires_grad && !b_gradient) {
            throw std::runtime_error("MatMulGradFn::apply: B gradient Tensor is NULL - capture_inputs() must be called");
        }

        float* grad_a_data = nullptr;
        float* grad_b_data = nullptr;
        if (a_requires_grad) grad_a_data = a_gradient->data;
        if (b_requires_grad) grad_b_data = b_gradient->data;
        
        const float alpha = 1.0f;
        const float beta_accum = 1.0f;  // Accumulate to existing gradient
        
        cublasSetStream(cublas_handle, stream);;

        // Trace-mode stage barriers make asynchronous cuBLAS failures attributable to
        // the exact half of matmul backward. They compile out with the shared autograd
        // trace gate, so normal training remains asynchronous.
        auto trace_stage_or_throw = [&](const char* stage) {
            if constexpr (!GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) {
                return;
            }
            const cudaError_t status = cudaStreamSynchronize(stream);
            AG_TRACE(
                "[MatMulGradFn][STAGE] gradfn=%p stage=%s status=%s(%d) "
                "A=%s B=%s M=%d K=%d N=%d transB=%d "
                "grad_output=%p grad_count=%zu grad_a=%p grad_b=%p "
                "cached_a=%p cached_b=%p\n",
                static_cast<void*>(this),
                stage,
                cudaGetErrorString(status),
                static_cast<int>(status),
                a_name ? a_name : "<unnamed>",
                b_name ? b_name : "<unnamed>",
                M,
                K,
                N,
                transpose_b ? 1 : 0,
                static_cast<const void*>(grad_output.data),
                grad_output.numel(),
                static_cast<void*>(grad_a_data),
                static_cast<void*>(grad_b_data),
                static_cast<const void*>(cached_a),
                static_cast<const void*>(cached_b));
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("MatMulGradFn::apply: CUDA failure after ") + stage +
                    " for A=" + (a_name ? a_name : "<unnamed>") +
                    " B=" + (b_name ? b_name : "<unnamed>") +
                    " M=" + std::to_string(M) +
                    " K=" + std::to_string(K) +
                    " N=" + std::to_string(N) +
                    ": " + cudaGetErrorString(status));
            }
        };

        AG_TRACE(
            "[MatMulGradFn][STAGE] gradfn=%p stage=begin "
            "A=%s B=%s M=%d K=%d N=%d transB=%d "
            "grad_output=%p grad_count=%zu grad_a=%p grad_b=%p "
            "cached_a=%p cached_b=%p\n",
            static_cast<void*>(this),
            a_name ? a_name : "<unnamed>",
            b_name ? b_name : "<unnamed>",
            M,
            K,
            N,
            transpose_b ? 1 : 0,
            static_cast<const void*>(grad_output.data),
            grad_output.numel(),
            static_cast<void*>(grad_a_data),
            static_cast<void*>(grad_b_data),
            static_cast<const void*>(cached_a),
            static_cast<const void*>(cached_b));
        
        // Without transpose_b: Forward was C = A @ B, where A[M,K], B[K,N], C[M,N]
        //   grad_A = grad_C @ B^T   [M,N] @ [N,K] = [M,K]
        //   grad_B = A^T @ grad_C   [K,M] @ [M,N] = [K,N]
        //
        // With transpose_b: Forward was C = A @ B^T, where A[M,K], B[N,K], C[M,N]
        //   grad_A = grad_C @ B     [M,N] @ [N,K] = [M,K]  (B, not B^T)
        //   grad_B = grad_C^T @ A   [N,M] @ [M,K] = [N,K]  (gradient w.r.t. B before transpose)
        
        // ISSUE #48 FIX: Use stored grad pointers instead of Tensor* (which may be dangling)
        if (a_requires_grad) {
            AG_TRACE("[MatMulGradFn] Computing grad_A, grad_a=%p, cached_b=%p\n",
                   static_cast<void*>(a_gradient->data), static_cast<const void*>(cached_b));
            if (!cached_b) {
                throw std::runtime_error("MatMulGradFn::apply: cached_b is NULL but input_a requires grad");
            }
            if (M <= 0 || K <= 0 || N <= 0) {
                throw std::runtime_error(
                    std::string("MatMulGradFn::apply: invalid dimensions M=") + std::to_string(M) +
                    " K=" + std::to_string(K) + " N=" + std::to_string(N));
            }


            if (transpose_b) {
                // grad_A = grad_C @ B  where B is [N, K] (original weights, NOT transposed now)
                // Row-major: grad_A[M,K] = grad_C[M,N] @ B[N,K]
                //
                // Using row-major GEMM trick: C = A @ B row-major ≡ C^T = B^T @ A^T col-major
                // So: grad_A^T[K,M] = B^T[K,N] @ grad_C^T[N,M] col-major
                //
                // Our storage (row-major → col-major conversion):
                //   cached_b is B[N,K] row-major = [K,N] col-major (K=768 rows, N=50377 cols)
                //   grad_C is [M,N] row-major = [N,M] col-major (N=50377 rows, M=3598 cols)
                //   grad_A is [M,K] row-major = [K,M] col-major (K=768 rows, M=3598 cols)
                //
                // cuBLAS: C = op(A) @ op(B)
                //   We want: grad_A^T[K,M] = [K,N] @ [N,M]
                //   A = cached_b[K,N] col-major, op=N gives [K,N] logical ✓
                //   B = grad_C[N,M] col-major, op=N gives [N,M] logical ✓
                //   Result: [K,M] col-major = grad_A^T ✓
                cublasStatus_t sgemm_status_1 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,    // cached_b is [K,N] col-major, use as [K,N]
                    CUBLAS_OP_N,    // grad_C is [N,M] col-major, use as [N,M]
                    K, M, N,        // m=K=768, n=M=3598, k=N=50377
                    &alpha,
                    cached_b, K,          // lda=K=768 (leading dim of [K,N])
                    grad_output.data, N,  // ldb=N=50377 (leading dim of [N,M])
                    &beta_accum,
                    a_gradient->data, K   // ldc=K=768 (leading dim of [K,M])
                );
                trackCublasCall("cublasSgemm_grad_A_transB", cublas_handle, stream, sgemm_status_1);
            } else {
                // grad_A = grad_C @ B^T  where B is [K, N]
                // Goal: grad_A[M,K] = grad_C[M,N] @ B^T[N,K]  (row-major)
                //
                // Row-major GEMM trick: R = X @ Y row-major ≡ R^T = Y^T @ X^T col-major
                // So: grad_A^T[K,M] = B[K,N] @ grad_C^T[N,M] col-major
                //
                // Storage conversions (row-major → col-major = transpose):
                //   cached_b is B[K,N] row-major = [N,K] col-major  
                //   grad_C is [M,N] row-major = [N,M] col-major
                //   grad_A is [M,K] row-major = [K,M] col-major (output)
                //
                // We want: [K,M] = [K,N] @ [N,M] col-major
                //   cached_b is [N,K] col-major, need OP_T to get [K,N]
                //   grad_C is [N,M] col-major, use OP_N to get [N,M]
                //
                // cuBLAS: C[K,M] = OP_T(cached_b[N,K]) @ OP_N(grad_C[N,M])
                cublasStatus_t sgemm_status_2 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_T,    // cached_b is [N,K] col-major, transpose to [K,N]
                    CUBLAS_OP_N,    // grad_C is [N,M] col-major, use as [N,M]
                    K, M, N,        // m=K, n=M, k=N: result [K,M] col-major = grad_A[M,K] row-major
                    &alpha,
                    cached_b, N,          // lda=N (leading dim of [N,K] col-major, transposed)
                    grad_output.data, N,  // ldb=N (leading dim of [N,M] col-major)
                    &beta_accum,
                    a_gradient->data, K   // ldc=K (leading dim of [K,M] col-major)
                );
                trackCublasCall("cublasSgemm_grad_A", cublas_handle, stream, sgemm_status_2);
            }
            trace_stage_or_throw("grad_A");
        }
        
        // ISSUE #48 FIX: Use stored grad pointers instead of Tensor*
        if (b_requires_grad) {
            AG_TRACE("[MatMulGradFn] Computing grad_B, grad_b=%p, cached_a=%p\n",
                   static_cast<void*>(b_gradient->data), static_cast<const void*>(cached_a));
            if (!cached_a) {
                throw std::runtime_error("MatMulGradFn::apply: cached_a is NULL but input_b requires grad");
            }
            
            if (transpose_b) {
                // grad_B = grad_C^T @ A  where A is [M, K], result is [N, K]
                // This is the gradient w.r.t. B BEFORE the transpose in forward.
                //
                // Row-major: grad_B[N,K] = grad_C^T[N,M] @ A[M,K]
                // Using row-major trick: R = X @ Y row-major ≡ R^T = Y^T @ X^T col-major
                // So: grad_B^T[K,N] = A^T[K,M] @ grad_C[M,N] col-major
                //
                // Storage conversions:
                //   cached_a is A[M,K] row-major = A^T[K,M] col-major
                //   grad_C is [M,N] row-major = [N,M] col-major
                //   grad_B is [N,K] row-major = [K,N] col-major
                //
                // cuBLAS: C = op(A) @ op(B)
                //   We want: grad_B^T[K,N] = A^T[K,M] @ grad_C[M,N]
                //   A = cached_a[K,M] col-major, op=N gives [K,M] = A^T ✓
                //   B = grad_C[N,M] col-major, op=T gives [M,N] = grad_C ✓
                //   Result: [K,N] col-major = grad_B^T = grad_B[N,K] row-major ✓
                cublasStatus_t sgemm_status_3 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,    // cached_a is [K,M] col-major, use as [K,M] = A^T
                    CUBLAS_OP_T,    // grad_output is [N,M] col-major, transpose to [M,N]
                    K, N, M,        // m=K=768, n=N=50377, k=M=3598
                    &alpha,
                    cached_a, K,          // lda=K (leading dim of [K,M])
                    grad_output.data, N,  // ldb=N (leading dim of [N,M])
                    &beta_accum,
                    b_gradient->data, K   // ldc=K (leading dim of [K,N])
                );
                trackCublasCall("cublasSgemm_grad_B_transB", cublas_handle, stream, sgemm_status_3);
            } else {
                // grad_B = A^T @ grad_C  where A is [M, K], result is [K, N]
                // Goal: grad_B[K,N] = A^T[K,M] @ grad_C[M,N]  (row-major)
                //
                // Row-major GEMM trick: R = X @ Y row-major ≡ R^T = Y^T @ X^T col-major
                // So: grad_B^T[N,K] = grad_C^T[N,M] @ A[M,K] col-major
                //
                // Storage conversions (row-major → col-major = transpose):
                //   cached_a is A[M,K] row-major = [K,M] col-major
                //   grad_C is [M,N] row-major = [N,M] col-major
                //   grad_B is [K,N] row-major = [N,K] col-major (output)
                //
                // We want: [N,K] = [N,M] @ [M,K] col-major
                //   grad_C is [N,M] col-major, use OP_N to get [N,M]
                //   cached_a is [K,M] col-major, need OP_T to get [M,K]
                //
                // cuBLAS: C[N,K] = OP_N(grad_C[N,M]) @ OP_T(cached_a[K,M])
                cublasStatus_t sgemm_status_4 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,    // grad_C is [N,M] col-major, use as [N,M]
                    CUBLAS_OP_T,    // cached_a is [K,M] col-major, transpose to [M,K]
                    N, K, M,        // m=N, n=K, k=M: result [N,K] col-major = grad_B[K,N] row-major
                    &alpha,
                    grad_output.data, N,  // lda=N (leading dim of [N,M] col-major)
                    cached_a, K,          // ldb=K (leading dim of [K,M] col-major, transposed)
                    &beta_accum,
                    b_gradient->data, N   // ldc=N (leading dim of [N,K] col-major)
                );
                trackCublasCall("cublasSgemm_grad_B", cublas_handle, stream, sgemm_status_4);
            }
            trace_stage_or_throw("grad_B");
        }

        logLmHeadGemmBackwardEquation(grad_output,
                          grad_a_data,
                          grad_b_data,
                          a_requires_grad,
                          b_requires_grad,
                          a_name,
                          b_name,
                          M,
                          K,
                          N,
                          transpose_b,
                          stream);


        if (a_requires_grad) {
            propagate_input_gradient(
                a_gradient, stream, backward_payload, backward_bindings,
                "MatMulGradFn::apply A");
        }
        if (b_requires_grad &&
            (b_gradient->is_leaf || !a_requires_grad ||
             b_gradient->grad_fn != a_gradient->grad_fn)) {
            propagate_input_gradient(
                b_gradient, stream, backward_payload, backward_bindings,
                "MatMulGradFn::apply B");
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        // ISSUE #51 FIX: Release owned cache copies (shared_ptr → deferred cleanup)
        owned_cached_a.reset();
        owned_cached_b.reset();
        cached_a = nullptr;
        cached_b = nullptr;
        a_gradient.reset();
        b_gradient.reset();
    }
};

// Thread-local cuBLAS handle for autograd operations
static thread_local cublasHandle_t s_autograd_cublas_handle = nullptr;

void set_autograd_cublas_handle(cublasHandle_t handle) {
    s_autograd_cublas_handle = handle;
}

cublasHandle_t get_autograd_cublas_handle() {
    return s_autograd_cublas_handle;
}

/**
 * autograd::matmul - Matrix multiplication with automatic differentiation
 * 
 * TAPE-BASED: Saves owned forward copies of A and B for backward.
 *
 * @param a Input tensor A [M, K]
 * @param b Input tensor B [K, N]
 * @param stream CUDA stream
 * @return Output tensor C [M, N]
 */
Tensor matmul(const Tensor& a, const Tensor& b, cudaStream_t stream,
              bool transpose_b) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::matmul: stream is NULL - caller MUST provide valid stream");
    }
    
    // Validate inputs are 2D
    if (!a.shape.is_2d_layout() || !b.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::matmul: inputs must be 2D (BSM layout)");
    }
    
    const auto& a_shape = a.shape.as_2d();
    const auto& b_shape = b.shape.as_2d();
    
    // A is [M, K]
    // B is [K, N] if !transpose_b, or [N, K] if transpose_b (so B^T is [K, N])
    const int M = a_shape.rows;
    const int K = a_shape.cols;
    const int K2 = transpose_b ? b_shape.cols : b_shape.rows;
    const int N = transpose_b ? b_shape.rows : b_shape.cols;
    
    if (K != K2) {
        throw std::invalid_argument("autograd::matmul: inner dimensions must match (K=" + std::to_string(K) + " vs K2=" + std::to_string(K2) + ")");
    }
    
    // Get cuBLAS handle
    cublasHandle_t handle = get_autograd_cublas_handle();
    
    if (!handle) {
        throw std::runtime_error("autograd::matmul: cuBLAS handle not set. Call set_autograd_cublas_handle() first.");
    }
    
    // Output shape: [M, N]
    auto output_shape = TensorContract::TensorShape::make_BSM(M, N);
    Tensor result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad, stream, "matmul_result");
    
    // Forward: C = A @ B  (or C = A @ B^T if transpose_b)
    // Row-major storage: for cuBLAS we compute C^T = B^T @ A^T (or C^T = B @ A^T if transpose_b)
    // 
    // Without transpose_b: B is [K, N], we want B^T for cuBLAS
    //   cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    //
    // With transpose_b: B is [N, K], we want B (no transpose) for cuBLAS since B^T is [K, N]
    //   cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, B, K, A, K, beta, C, N)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSetStream(handle, stream);
    
    cublasStatus_t status;
    if (transpose_b) {
        // B is [N, K] row-major, we want A @ B^T = C[M,N]
        // Row-major A[M,K] @ (B[N,K])^T = C[M,N]
        // 
        // For cuBLAS (column-major), we compute C^T = (A @ B^T)^T = B @ A^T
        // C^T[N,M] = B[N,K] @ A[M,K]^T = B[N,K] @ A^T[K,M]
        //
        // Row-major to col-major mapping:
        // - B[N,K] row-major: element [i,j] at B + i*K + j, row stride = K
        //   cuBLAS sees as [K,N] col-major (transpose). To get [N,K], use CUBLAS_OP_T.
        //   lda = K (leading dim of stored [K,N] matrix = K)
        //
        // - A[M,K] row-major: element [i,j] at A + i*K + j, row stride = K  
        //   cuBLAS sees as [K,M] col-major (= A^T). We want A^T, so use CUBLAS_OP_N.
        //   ldb = K (leading dim of stored [K,M] matrix = K)
        //
        // - C[M,N] row-major: element [i,j] at C + i*N + j, row stride = N
        //   cuBLAS sees as [N,M] col-major (= C^T). ldc = N.
        //
        // cublasSgemm(op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        // computes C[m,n] = op(A)[m,k] @ op(B)[k,n]
        // We want C^T[N,M] = B[N,K] @ A^T[K,M]
        // So m=N, n=M, k=K, op(A)=CUBLAS_OP_T(B_stored), op(B)=CUBLAS_OP_N(A_stored)
        
        status = cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            b.data, K,    // B: [N,K] row-major stored as [K,N] col-major, lda=K
            a.data, K,    // A: [M,K] row-major stored as [K,M] col-major, ldb=K
            &beta,
            result.data, N  // C: [M,N] row-major stored as [N,M] col-major, ldc=N
        );
    } else {
        // B is [K, N] row-major
        status = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            b.data, N,    // B: [K, N] row-major
            a.data, K,    // A: [M, K] row-major
            &beta,
            result.data, N  // C: [M, N] row-major
        );
    }
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("autograd::matmul: cuBLAS sgemm failed");
    }
    
    // Set up backward (TAPE-BASED)
    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<MatMulGradFn>();
        
        // ISSUE #48 FIX: Capture stable data from tensors NOW, before they go out of scope
        // Don't store Tensor* - the tensors may be stack variables that become dangling
        // ISSUE #55 FIX: Pass stream for async allocation of owned grad buffers
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        
        // Save the actual forward inputs owned by this matmul node.
        const float* effective_a_cache = a.data;
        const float* effective_b_cache = b.data;
        
        // Null check: grad_B = A^T @ grad_C requires saved A; grad_A = grad_C @ B^T requires saved B
        if (grad_fn->b_requires_grad && !effective_a_cache) {
            throw std::runtime_error(
            "autograd::matmul: input_a data is NULL while input_b requires grad. "
            "matmul must save forward A for grad_B. "
                "Context: A.name=" + std::string(a.name ? a.name : "<unnamed>") +
                " A.data=" + std::to_string(reinterpret_cast<uintptr_t>(a.data)) +
                " B.name=" + std::string(b.name ? b.name : "<unnamed>") +
                " B.data=" + std::to_string(reinterpret_cast<uintptr_t>(b.data)) +
                " shape(A)=[" + std::to_string(M) + "," + std::to_string(K) + "]"
                " shape(B)=[" + std::to_string(b_shape.rows) + "," + std::to_string(b_shape.cols) + "]"
                " transpose_b=" + std::to_string(transpose_b ? 1 : 0));
        }
        if (grad_fn->a_requires_grad && !effective_b_cache) {
            throw std::runtime_error(
                "autograd::matmul: Cannot compute grad for input_a - input_b data is NULL. "
                "Second input tensor has null data.");
        }
        
        grad_fn->set_cache_copy(effective_a_cache, effective_b_cache, M, K, N, handle, stream, transpose_b);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

//======================================================//
//  BF16 Conversion Kernels (for FlashAttention integration)
//  NOTE: Layout conversion kernels (BHSD<->BSHD bf16) live in TensorConversion.cu
//  Only GQA-specific reduction kernel remains here.
//======================================================

// ISSUE #72 FIX: Reduce GQA gradients from num_heads to num_kv_heads
// FlashAttention backward writes dK/dV for each query head separately (12 heads).
// For GQA with 4 KV heads, we need to SUM the gradients from grouped Q heads.
// E.g., Q heads 0,1,2 all use KV head 0, so dK[kv_head=0] = dK[q_head=0] + dK[q_head=1] + dK[q_head=2]
//
// heads_per_kv_group is a finalized HyperParameters-derived GQA dimension and
// must be threaded in explicitly. Do not recompute grouped-head geometry inside
// the kernel boundary.
//
// Input layout:  src [B, S, num_heads, D] bf16 - gradients per query head
// Output layout: dst [B, num_kv_heads, S, D] fp32 - reduced gradients per KV head
__global__ void kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32(
    const __nv_bfloat16* __restrict__ src,  // [B, S, num_heads, D]
    float* __restrict__ dst,                // [B, num_kv_heads, S, D]
    int batch, int num_heads, int num_kv_heads, int heads_per_kv_group, int seq_len, int head_dim
) {
    // Each thread handles one element in the output [B, num_kv_heads, S, D]
    const size_t total = static_cast<size_t>(batch) * num_kv_heads * seq_len * head_dim;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Decode destination BHSD index where H = num_kv_heads
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq_len;
    const int kv_h = (idx / (head_dim * seq_len)) % num_kv_heads;
    const int b = idx / (head_dim * seq_len * num_kv_heads);
    
    // ISSUE #85 FIX: The correct GQA gradient is the plain SUM of per-query-head contributions
    // (chain rule: dL/dK = sum over grouped Q heads). The previous 1/sqrt scaling halved the
    // effective K/V learning rate. Upstream FA2 uses at::sum_out without any scaling.
    float sum = 0.0f;
    for (int g = 0; g < heads_per_kv_group; ++g) {
        const int q_head = kv_h * heads_per_kv_group + g;
        const size_t src_idx = (static_cast<size_t>(b) * seq_len * num_heads * head_dim) +
                               (static_cast<size_t>(s) * num_heads * head_dim) +
                               (static_cast<size_t>(q_head) * head_dim) + d;
        sum += __bfloat162float(src[src_idx]);
    }

    dst[idx] = sum;
}

/**
 * ScaledDotProductAttentionGradFn - Backward for attention (ISSUE #48 FIX)
 * Forward: O = softmax(Q @ K^T / sqrt(d)) @ V
 * Uses FlashAttention v2 for memory-efficient backward.
 *
 * DOES NOT store Tensor* - stores stable data instead.
 * Requires saving: Q, K, V, O, and softmax_lse from forward.
 * Backward computes: grad_Q, grad_K, grad_V
 */
struct ScaledDotProductAttentionGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool q_requires_grad = false;
    bool k_requires_grad = false;
    bool v_requires_grad = false;
    float* q_grad = nullptr;
    float* k_grad = nullptr;
    float* v_grad = nullptr;
    // Q/K/V are non-leaf activations (outputs of split_and_reshape_qkv / rope_rotation),
    // so dQ/dK/dV are carried in GradFn-owned scratch buffers — the same Issue #55/#56
    // ownership model every other node uses (MatMulGradFn, AddGradFn, …). We must NOT
    // repurpose the input activation tensors' grad_ buffers as backward scratch: a
    // non-leaf's grad_ is transient, is never zeroed by the leaf zero_grad sweep, and
    // is not part of the gradient carrier contract. The leaf branch (unused in this
    // pipeline) falls back to the persistent grad buffer like the other nodes.
    bool q_is_leaf = false;
    bool k_is_leaf = false;
    bool v_is_leaf = false;
    std::shared_ptr<float> owned_q_grad;  // dQ scratch (non-leaf carrier)
    std::shared_ptr<float> owned_k_grad;  // dK scratch (non-leaf carrier)
    std::shared_ptr<float> owned_v_grad;  // dV scratch (non-leaf carrier)
    TensorContract::TensorShape q_shape, k_shape, v_shape;
    std::shared_ptr<GradFn> q_grad_fn;
    std::shared_ptr<GradFn> k_grad_fn;
    std::shared_ptr<GradFn> v_grad_fn;
    
    // Saved tensors for backward (in bf16 for FlashAttention)
    __nv_bfloat16* saved_q_bf16 = nullptr;
    __nv_bfloat16* saved_k_bf16 = nullptr;
    __nv_bfloat16* saved_v_bf16 = nullptr;
    __nv_bfloat16* saved_out_bf16 = nullptr;
    float* saved_softmax_lse = nullptr;
    
    // Workspace buffers for backward
    void* dq_accum = nullptr;
    void* dsoftmax_sum = nullptr;
    __nv_bfloat16* dq_bf16 = nullptr;
    __nv_bfloat16* dk_bf16 = nullptr;
    __nv_bfloat16* dv_bf16 = nullptr;
    __nv_bfloat16* dout_bf16 = nullptr;
    
    // Dimensions
    int batch_size = 0;
    int seq_len = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int heads_per_kv_group = 0;
    float softmax_scale = 0.0f;
    bool causal = true;
    bool is_bf16 = false;
    // Resolved from scheduler-provided backward bindings; never owned here.
    bool uses_sequence_lengths = false;
    
    // ALiBi slopes (pointer to device memory, not owned - do NOT free)
    const float* alibi_slopes = nullptr;
    
    // Attention dropout (saved for backward to reproduce same mask)
    float attention_dropout_p = 0.0f;
    uint64_t dropout_seed = 0;
    
    ScaledDotProductAttentionGradFn() { op_name = "scaled_dot_product_attention"; }
    
    ~ScaledDotProductAttentionGradFn() override {
        release_saved();
    }
    
    void capture_inputs(Tensor& q, Tensor& k, Tensor& v) {
        q_requires_grad = q.requires_grad;
        k_requires_grad = k.requires_grad;
        v_requires_grad = v.requires_grad;
        q_shape = q.shape;
        k_shape = k.shape;
        v_shape = v.shape;
        
        // Copy shared_ptrs to captured grad_fns
        q_grad_fn = q.grad_fn;
        k_grad_fn = k.grad_fn;
        v_grad_fn = v.grad_fn;
        register_input(q.grad_fn);
        register_input(k.grad_fn);
        register_input(v.grad_fn);
        
        q_is_leaf = q.is_leaf;
        k_is_leaf = k.is_leaf;
        v_is_leaf = v.is_leaf;

        // Leaf inputs (weights) own a persistent grad buffer — accumulate into it
        // directly, exactly like AddGradFn/MatMulGradFn. Non-leaf activations (the
        // real path here) get a GradFn-owned scratch carrier allocated in apply()
        // once dimensions are known; we never call alloc_grad() on a transient.
        if (q_requires_grad && q_is_leaf) { q.ensure_grad(); q_grad = q.grad_data(); }
        if (k_requires_grad && k_is_leaf) { k.ensure_grad(); k_grad = k.grad_data(); }
        if (v_requires_grad && v_is_leaf) { v.ensure_grad(); v_grad = v.grad_data(); }
    }
    
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("scaled_dot_product_attention", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        if (!saved_q_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_q_bf16 is NULL - save() must store Q for backward");
        if (!saved_k_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_k_bf16 is NULL - save() must store K for backward");
        if (!saved_v_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_v_bf16 is NULL - save() must store V for backward");
        if (!saved_out_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_out_bf16 is NULL - save() must store output for backward");
        if (heads_per_kv_group <= 0) {
            throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: heads_per_kv_group must be > 0");
        }
        const int* sequence_lengths = nullptr;
        if (uses_sequence_lengths) {
            if (!backward_payload || !backward_bindings) {
                throw std::runtime_error(
                    "ScaledDotProductAttentionGradFn::apply: scheduler must provide "
                    "BatchPayload and BatchDeviceBindings for non-causal padding bounds");
            }
            if (backward_payload->batch_size != batch_size ||
                backward_payload->max_seq_len != seq_len) {
                throw std::runtime_error(
                    "ScaledDotProductAttentionGradFn::apply: backward batch geometry "
                    "differs from forward");
            }
            if (!backward_bindings->d_sequence_lengths) {
                throw std::runtime_error(
                    "ScaledDotProductAttentionGradFn::apply: sequence lengths were not "
                    "uploaded for this step");
            }
            sequence_lengths = backward_bindings->d_sequence_lengths;
        }
        
        const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
        const size_t kv_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
        const size_t dk_dv_alloc_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
        const int block_size = 256;
        const int kv_blocks = static_cast<int>((kv_elems + block_size - 1) / block_size);
        
        // Convert grad_output (FP32 BHSD) to BF16 BSHD
        logGradFlowTensorStats("SDPA.apply grad_output(BHSD)", grad_output.data, grad_output.numel(), stream);
        TensorConversion::convert_BHSD_to_BSHD_bf16(
            grad_output.data, dout_bf16, batch_size, num_heads, seq_len, head_dim, stream);
        logGradFlowBf16TensorStats("SDPA.apply dout_bf16(BSHD)", dout_bf16, q_elems, stream);

        // These buffers are backward outputs, not forward state. Keep their hygiene
        // inside apply() so stale or stray writes between forward and backward cannot
        // be mistaken for FlashAttention gradients.
        throwIfCudaFailed(cudaMemsetAsync(dq_bf16, 0, q_elems * sizeof(__nv_bfloat16), stream),
                          "ScaledDotProductAttentionGradFn::apply: cudaMemsetAsync(dq_bf16) failed");
        throwIfCudaFailed(cudaMemsetAsync(dk_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16), stream),
                          "ScaledDotProductAttentionGradFn::apply: cudaMemsetAsync(dk_bf16) failed");
        throwIfCudaFailed(cudaMemsetAsync(dv_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16), stream),
                          "ScaledDotProductAttentionGradFn::apply: cudaMemsetAsync(dv_bf16) failed");
        logGradFlowBf16TensorStats("SDPA.apply dv_bf16_pre_bwd", dv_bf16, dk_dv_alloc_elems, stream);
        
        // Call FlashAttention backward
        flash_attn_bwd_ex(
            saved_q_bf16,      // Q  [B, S, H, D] bf16
            saved_k_bf16,      // K  [B, S, Hkv, D] bf16
            saved_v_bf16,      // V  [B, S, Hkv, D] bf16
            saved_out_bf16,    // O  [B, S, H, D] bf16
            dout_bf16,         // dO [B, S, H, D] bf16
            saved_softmax_lse, // LSE [B, H, S] fp32
            alibi_slopes,      // ALiBi slopes [num_heads] (saved from forward)
            dq_bf16,           // dQ output
            dk_bf16,           // dK output
            dv_bf16,           // dV output
            dq_accum,          // workspace
            dsoftmax_sum,      // workspace
            batch_size,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            causal,
            is_bf16,
            attention_dropout_p,  // Same dropout rate as forward
            dropout_seed,         // Same seed as forward (reproduces identical mask)
            stream,
            sequence_lengths      // Same non-causal key bounds as forward
        );
        logGradFlowBf16TensorStats("SDPA.apply dq_bf16_post_bwd", dq_bf16, q_elems, stream);
        logGradFlowBf16TensorStats("SDPA.apply dk_bf16_post_bwd", dk_bf16, dk_dv_alloc_elems, stream);
        logGradFlowBf16TensorStats("SDPA.apply dv_bf16_post_bwd", dv_bf16, dk_dv_alloc_elems, stream);
        
        // =========================================================================
        // ISSUE #83 REMOVAL: Issue #84 (missing preprocessing kernel)
        // =========================================================================
        // The dQ/dK normalization below was a BANDAID for the gradient explosion bug.
        // With Issue #84's preprocessing kernel fix, dQ/dK are now at proper magnitude.
        // Keeping this normalization would CRUSH attention gradients, causing vanishing.
        // 
        // Evidence from training_17696307607301724.log:
        //   - attn gradients: 1.96 → 0.08 (24x DECREASE - vanishing!)
        //   - ffn gradients:  1.83 → 0.07 (26x DECREASE - vanishing!)
        //   - rms gradients:  0.02 → 0.60 (30x INCREASE - still has signal)
        // This pattern shows encoder layers are frozen while RMSNorm (closer to output) learns.
        // 
        // DISABLED Issue #83 normalization - use scale=1.0 for all gradients.
        // =========================================================================
        
        // Convert gradients back to FP32 BHSD and accumulate WITHOUT normalization.
        // Non-leaf inputs receive a freshly zeroed GradFn-owned carrier here (Issue
        // #55/#56 model); leaf inputs accumulate into their persistent grad buffer.
        if (q_requires_grad) {
            if (!q_is_leaf && !owned_q_grad) {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), q_elems * sizeof(float), "sdpa_owned_q_grad");
                cudaMemsetAsync(buf, 0, q_elems * sizeof(float), stream);
                owned_q_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
                q_grad = buf;
            }
            float* grad_q_fp32 = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&grad_q_fp32), q_elems * sizeof(float), "sdpa_grad_q_fp32");
            cudaMemsetAsync(grad_q_fp32, 0, q_elems * sizeof(float), stream);
            
            TensorConversion::convert_BSHD_bf16_to_BHSD(
                dq_bf16, grad_q_fp32, batch_size, seq_len, num_heads, head_dim, stream);
            logGradFlowTensorStats("SDPA.apply grad_q_fp32", grad_q_fp32, q_elems, stream);
            
            // Scale = 1.0 (no normalization - Issue #84 fixed root cause)
            accumulate_grad(q_grad, grad_q_fp32, q_elems, 1.0f, stream, "ScaledDotProductAttentionGradFn::apply q_grad");
            logGradFlowTensorStats("SDPA.apply q_grad_accum", q_grad, q_elems, stream);
            cudaFreeAsync(grad_q_fp32, stream);
        }
        
        if (k_requires_grad) {
            if (!k_is_leaf && !owned_k_grad) {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), kv_elems * sizeof(float), "sdpa_owned_k_grad");
                cudaMemsetAsync(buf, 0, kv_elems * sizeof(float), stream);
                owned_k_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
                k_grad = buf;
            }
            float* grad_k_fp32 = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&grad_k_fp32), kv_elems * sizeof(float), "sdpa_grad_k_fp32");
            cudaMemsetAsync(grad_k_fp32, 0, kv_elems * sizeof(float), stream);
            // ISSUE #72 FIX: Use GQA reduction kernel to sum gradients from grouped Q heads
            // dk_bf16 is [B, S, num_heads, D], we reduce to [B, num_kv_heads, S, D]
            kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dk_bf16, grad_k_fp32, batch_size, num_heads, num_kv_heads, heads_per_kv_group, seq_len, head_dim);
            logGradFlowTensorStats("SDPA.apply grad_k_fp32", grad_k_fp32, kv_elems, stream);
            // Scale = 1.0 (no normalization - Issue #84 fixed root cause)
            accumulate_grad(k_grad, grad_k_fp32, kv_elems, 1.0f, stream, "ScaledDotProductAttentionGradFn::apply k_grad");
            logGradFlowTensorStats("SDPA.apply k_grad_accum", k_grad, kv_elems, stream);
            cudaFreeAsync(grad_k_fp32, stream);
        }
        
        if (v_requires_grad) {
            if (!v_is_leaf && !owned_v_grad) {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), kv_elems * sizeof(float), "sdpa_owned_v_grad");
                cudaMemsetAsync(buf, 0, kv_elems * sizeof(float), stream);
                owned_v_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
                v_grad = buf;
            }
            float* grad_v_fp32 = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&grad_v_fp32), kv_elems * sizeof(float), "sdpa_grad_v_fp32");
            cudaMemsetAsync(grad_v_fp32, 0, kv_elems * sizeof(float), stream);
            // ISSUE #72 FIX: Use GQA reduction kernel to sum gradients from grouped Q heads
            // dv_bf16 is [B, S, num_heads, D], we reduce to [B, num_kv_heads, S, D]
            kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dv_bf16, grad_v_fp32, batch_size, num_heads, num_kv_heads, heads_per_kv_group, seq_len, head_dim);
            logGradFlowTensorStats("SDPA.apply grad_v_fp32", grad_v_fp32, kv_elems, stream);
            // Scale = 1.0 (no normalization needed)
            accumulate_grad(v_grad, grad_v_fp32, kv_elems, 1.0f, stream, "ScaledDotProductAttentionGradFn::apply v_grad");
            logGradFlowTensorStats("SDPA.apply v_grad_accum", v_grad, kv_elems, stream);
            cudaFreeAsync(grad_v_fp32, stream);
        }
        
        // CONTINUE AUTOGRAD CHAIN - call grad_fns for Q, K, V
        if (q_requires_grad && q_grad_fn) {
            Tensor q_view;
            q_view.data = q_grad; q_view.shape = q_shape;
            q_view.owns_data = false; q_view.stream = stream;
            q_grad_fn->apply(q_view, stream, backward_payload, backward_bindings);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
        }
        if (k_requires_grad && k_grad_fn) {
            Tensor k_view;
            k_view.data = k_grad; k_view.shape = k_shape;
            k_view.owns_data = false; k_view.stream = stream;
            k_grad_fn->apply(k_view, stream, backward_payload, backward_bindings);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
        }
        if (v_requires_grad && v_grad_fn) {
            Tensor v_view;
            v_view.data = v_grad; v_view.shape = v_shape;
            v_view.owns_data = false; v_view.stream = stream;
            v_grad_fn->apply(v_view, stream, backward_payload, backward_bindings);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_q_bf16) { cudaFree(saved_q_bf16); saved_q_bf16 = nullptr; }
        if (saved_k_bf16) { cudaFree(saved_k_bf16); saved_k_bf16 = nullptr; }
        if (saved_v_bf16) { cudaFree(saved_v_bf16); saved_v_bf16 = nullptr; }
        if (saved_out_bf16) { cudaFree(saved_out_bf16); saved_out_bf16 = nullptr; }
        if (saved_softmax_lse) { cudaFree(saved_softmax_lse); saved_softmax_lse = nullptr; }
        if (dq_accum) { cudaFree(dq_accum); dq_accum = nullptr; }
        if (dsoftmax_sum) { cudaFree(dsoftmax_sum); dsoftmax_sum = nullptr; }
        if (dq_bf16) { cudaFree(dq_bf16); dq_bf16 = nullptr; }
        if (dk_bf16) { cudaFree(dk_bf16); dk_bf16 = nullptr; }
        if (dv_bf16) { cudaFree(dv_bf16); dv_bf16 = nullptr; }
        if (dout_bf16) { cudaFree(dout_bf16); dout_bf16 = nullptr; }
        q_grad = nullptr; k_grad = nullptr; v_grad = nullptr;
        owned_q_grad.reset();
        owned_k_grad.reset();
        owned_v_grad.reset();
        q_grad_fn.reset();
        k_grad_fn.reset();
        v_grad_fn.reset();
    }
};

Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const float* alibi_slopes,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& attention_hp,
    const Batching::BatchDeviceBindings& bindings,
    float scale, cudaStream_t stream,
    float attention_dropout_p, uint64_t dropout_seed
) {
    if (!alibi_slopes) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: alibi_slopes is NULL - GRIM attention requires ALiBi slopes");
    }

    // Validate inputs are 4D BHSD layout
    if (!q.shape.is_4d() || !k.shape.is_4d() || !v.shape.is_4d()) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: Q/K/V must be BHSD layout");
    }
    
    const auto& q_4d = q.shape.as_4d();
    const auto& k_4d = k.shape.as_4d();
    const auto& v_4d = v.shape.as_4d();
    
    const int batch_size = q_4d.batch;
    const int num_heads = q_4d.heads;
    const int seq_len = q_4d.seq;
    const int head_dim = q_4d.head_dim;
    const int num_kv_heads = k_4d.heads;
    
    // Validate shapes
    if (k_4d.batch != batch_size || v_4d.batch != batch_size) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: batch size mismatch");
    }
    if (k_4d.seq != seq_len || v_4d.seq != seq_len) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: sequence length mismatch");
    }
    if (k_4d.head_dim != head_dim || v_4d.head_dim != head_dim) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: head_dim mismatch");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: num_heads must be divisible by num_kv_heads");
    }
    if (num_heads != attention_hp.num_heads) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: Q heads=" +
                                    std::to_string(num_heads) +
                                    " does not match EncoderSelfAttentionHP num_heads=" +
                                    std::to_string(attention_hp.num_heads));
    }
    if (num_kv_heads != attention_hp.num_kv_heads) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: K heads=" +
                                    std::to_string(num_kv_heads) +
                                    " does not match EncoderSelfAttentionHP num_kv_heads=" +
                                    std::to_string(attention_hp.num_kv_heads));
    }
    if (head_dim != attention_hp.head_dim) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: head_dim=" +
                                    std::to_string(head_dim) +
                                    " does not match EncoderSelfAttentionHP head_dim=" +
                                    std::to_string(attention_hp.head_dim));
    }
    if (attention_hp.heads_per_kv_group <= 0) {
        throw std::invalid_argument("autograd::scaled_dot_product_attention: EncoderSelfAttentionHP heads_per_kv_group must be > 0");
    }
    
    // Output shape: same as Q [B, H, S, D]
    auto output_shape = TensorContract::TensorShape::make_BHSD(batch_size, num_heads, seq_len, head_dim);
    bool requires_grad = q.requires_grad || k.requires_grad || v.requires_grad;
    Tensor result = Tensor::zeros(output_shape, requires_grad, stream, "sdpa_result");
    
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument(
            "autograd::scaled_dot_product_attention: scale must be finite and > 0");
    }
    
    // Allocate bf16 buffers for FlashAttention
    const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
    const size_t kv_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
    const size_t lse_elems = static_cast<size_t>(batch_size) * num_heads * seq_len;
    
    __nv_bfloat16* q_bf16 = nullptr;
    __nv_bfloat16* k_bf16 = nullptr;
    __nv_bfloat16* v_bf16 = nullptr;
    __nv_bfloat16* out_bf16 = nullptr;
    float* softmax_lse = nullptr;
    
    cudaMallocOrThrow(reinterpret_cast<void**>(&q_bf16), q_elems * sizeof(__nv_bfloat16), "sdpa_q_bf16");
    cudaMallocOrThrow(reinterpret_cast<void**>(&k_bf16), kv_elems * sizeof(__nv_bfloat16), "sdpa_k_bf16");
    cudaMallocOrThrow(reinterpret_cast<void**>(&v_bf16), kv_elems * sizeof(__nv_bfloat16), "sdpa_v_bf16");
    cudaMallocOrThrow(reinterpret_cast<void**>(&out_bf16), q_elems * sizeof(__nv_bfloat16), "sdpa_out_bf16");
    cudaMallocOrThrow(reinterpret_cast<void**>(&softmax_lse), lse_elems * sizeof(float), "sdpa_softmax_lse");
    // Sentinel fill: 0xFF bytes → float NaN. If LSE shows NaN after kernel, kernel didn't write.
    // Valid LSE is always finite (LSE = max_score * scale + log(sum_exp)), never NaN.
    cudaMemsetAsync(softmax_lse, 0xFF, lse_elems * sizeof(float), stream);
    
    // Convert FP32 BHSD -> BF16 BSHD
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        q.data, q_bf16, batch_size, num_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        k.data, k_bf16, batch_size, num_kv_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        v.data, v_bf16, batch_size, num_kv_heads, seq_len, head_dim, stream);
    
    // Ablation (AblationFlags.hpp): kZeroAlibiBias disables the ALiBi positional
    // bias by handing the kernel a NULL slope pointer. The vendored Dao kernel's
    // ALIBI_SWITCH(alibi_slopes_ptr != nullptr) then selects the no-bias path for
    // both forward and backward, so the saved grad_fn pointer must match.
    const float* effective_alibi_slopes =
        GRIM::Ablation::kZeroAlibiBias ? nullptr : alibi_slopes;

    // Ablation (AblationFlags.hpp): kDisableCausalMask drops the causal mask so
    // the kernel runs full (bidirectional) self-attention. This requires a build
    // with -DGRIM_ABLATE_DISABLE_CAUSAL_MASK=ON, which forces non-causal behavior
    // (compiling the non-causal kernel templates and removing the runtime guard)
    // and defines the macro that flips the constexpr in lockstep. The forward and
    // saved-for-backward causal flags must match.
    const bool effective_causal =
        attention_hp.causal_mask && !GRIM::Ablation::kDisableCausalMask;
    const int* sequence_lengths = nullptr;
    if (!effective_causal) {
        if (!bindings.d_sequence_lengths) {
            throw std::invalid_argument(
                "autograd::scaled_dot_product_attention: non-causal padded attention "
                "requires uploaded sequence lengths");
        }
        sequence_lengths = bindings.d_sequence_lengths;
    }

    // Forward pass with FlashAttention. The resolved scale is passed explicitly;
    // ignoring it silently changes the attention equation.
    flash_attn_fwd_ex(
        q_bf16,      // Q  [B, S, H, D] bf16
        k_bf16,      // K  [B, S, Hkv, D] bf16
        v_bf16,      // V  [B, S, Hkv, D] bf16
        out_bf16,    // O  [B, S, H, D] bf16
        softmax_lse, // LSE [B, H, S] fp32
        effective_alibi_slopes, // ALiBi slopes [num_heads] (NULL when kZeroAlibiBias)
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        effective_causal,
        true,
        attention_dropout_p, // Attention dropout rate (0.0 = disabled)
        dropout_seed,        // Per-step Philox seed for reproducible masks
        stream,
        sequence_lengths     // Per-row key bounds for non-causal fixed rectangles
    );
    
    // Attention off-by-one (softmax1 / zero-value sink): guarded exact post-process
    // of the standard FlashAttention result. Applied in place to out_bf16 + softmax_lse
    // BEFORE the BHSD convert and BEFORE the GradFn captures these buffers, so the
    // forward output AND the unchanged FlashAttention backward both operate on the
    // softmax1 result (the phantom zero-value slot contributes no gradient term).
    if (attention_hp.attention_off_by_one) {
        launchAttentionOffByOneEpilogue(
            out_bf16, softmax_lse, batch_size, seq_len, num_heads, head_dim, stream);
    }

    // Convert BF16 BSHD -> FP32 BHSD for output
    TensorConversion::convert_BSHD_bf16_to_BHSD(
        out_bf16, result.data, batch_size, seq_len, num_heads, head_dim, stream);
    
    // Set up backward if needed - ISSUE #48: capture stable data, not Tensor*
    if (requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ScaledDotProductAttentionGradFn>();
        
        // ISSUE #48 FIX: Capture stable data instead of storing dangling Tensor*
        grad_fn->capture_inputs(const_cast<Tensor&>(q), const_cast<Tensor&>(k), const_cast<Tensor&>(v));
        
        // Transfer ownership of bf16 buffers to grad_fn for backward
        grad_fn->saved_q_bf16 = q_bf16;
        grad_fn->saved_k_bf16 = k_bf16;
        grad_fn->saved_v_bf16 = v_bf16;
        grad_fn->saved_out_bf16 = out_bf16;
        grad_fn->saved_softmax_lse = softmax_lse;
        grad_fn->batch_size = batch_size;
        grad_fn->seq_len = seq_len;
        grad_fn->num_heads = num_heads;
        grad_fn->num_kv_heads = num_kv_heads;
        grad_fn->head_dim = head_dim;
        grad_fn->heads_per_kv_group = attention_hp.heads_per_kv_group;
        grad_fn->softmax_scale = scale;
        grad_fn->causal = effective_causal;
        grad_fn->is_bf16 = true;
        grad_fn->uses_sequence_lengths = sequence_lengths != nullptr;
        grad_fn->alibi_slopes = effective_alibi_slopes;  // Save for backward pass (not owned); NULL when kZeroAlibiBias
        grad_fn->attention_dropout_p = attention_dropout_p;  // Same dropout for backward mask reproduction
        grad_fn->dropout_seed = dropout_seed;                // Same seed reproduces identical Philox mask
        
        // Allocate backward workspace
        const size_t dq_accum_bytes = flash_attn_dq_accum_bytes(batch_size, seq_len, num_heads, head_dim);
        const size_t dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(batch_size, seq_len, num_heads);
        cudaMallocOrThrow(&grad_fn->dq_accum, dq_accum_bytes, "sdpa_gradfn_dq_accum_workspace");
        cudaMallocOrThrow(&grad_fn->dsoftmax_sum, dsoftmax_sum_bytes, "sdpa_gradfn_dsoftmax_sum_workspace");
        
        // ISSUE #72 FIX: FlashAttention backward kernel writes dK/dV using query head index (bidh=0..num_heads-1),
        // NOT the KV head index (bidh / h_h_k_ratio). With GQA (12 Q heads, 4 KV heads), the library writes
        // to positions 0-11 * head_stride, but if we only allocate for 4 KV heads, heads 4-11 write out-of-bounds!
        // This causes STATUS_STACK_BUFFER_OVERRUN crashes.
        //
        // Solution: Allocate dk_bf16/dv_bf16 for num_heads (not num_kv_heads), let FlashAttention write to all,
        // then reduce the 12-head gradients down to 4 KV heads by summing grouped heads in apply().
        const size_t dk_dv_alloc_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;  // Use num_heads!
        
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_fn->dq_bf16), q_elems * sizeof(__nv_bfloat16), "sdpa_gradfn_dq_bf16");
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_fn->dk_bf16), dk_dv_alloc_elems * sizeof(__nv_bfloat16), "sdpa_gradfn_dk_bf16");  // ISSUE #72: Sized for num_heads
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_fn->dv_bf16), dk_dv_alloc_elems * sizeof(__nv_bfloat16), "sdpa_gradfn_dv_bf16");  // ISSUE #72: Sized for num_heads
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_fn->dout_bf16), q_elems * sizeof(__nv_bfloat16), "sdpa_gradfn_dout_bf16");
        throwIfCudaFailed(
            cudaMemsetAsync(grad_fn->dq_bf16, 0, q_elems * sizeof(__nv_bfloat16), stream),
            "scaled_dot_product_attention: cudaMemsetAsync(grad_fn->dq_bf16) failed");
        throwIfCudaFailed(
            cudaMemsetAsync(grad_fn->dk_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16), stream),
            "scaled_dot_product_attention: cudaMemsetAsync(grad_fn->dk_bf16) failed");
        throwIfCudaFailed(
            cudaMemsetAsync(grad_fn->dv_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16), stream),
            "scaled_dot_product_attention: cudaMemsetAsync(grad_fn->dv_bf16) failed");
        
        result.grad_fn = grad_fn;
        
        // Ownership of bf16 buffers transferred to grad_fn for backward pass.
        // DO NOT nullify pointers - if someone accidentally frees them after transfer,
        // let the double-free crash loudly (Rule 20: fail loud).
        // Any accidental cudaFree(q_bf16) here will produce a clear double-free error
        // that immediately reveals the bug instead of silently succeeding. 
    } else {
        // Free bf16 buffers if no backward needed
        cudaFree(q_bf16);
        cudaFree(k_bf16);
        cudaFree(v_bf16);
        cudaFree(out_bf16);
        cudaFree(softmax_lse);
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// ReshapeFromBHSDGradFn - ISSUE #62 FIX: Autograd-tracked BHSD->flat adapter
//
// ROOT CAUSE OF W_o/QKV GRADIENT BUG:
// Encoding_GPU.cu step 5 used Tensor::empty() + raw BHSD->BSM conversion
// which broke the autograd chain (attn_out had no grad_fn).
// When W_o matmul backward called input_grad_fn->apply(), it got nullptr
// because attn_out.grad_fn was never set.
//
// OWNERSHIP BOUNDARY:
// - TensorConversion owns the raw geometry kernels (`convert_BHSD_to_BSM`
//   and the inverse index mapping used below).
// - TensorContract/autograd owns the TAPE NODE that bridges attention output
//   into the output projection matmul.
//
// FIX: This wrapper takes attn_out_bhsd with its grad_fn and produces attn_out
// (flat) that has THIS GradFn. When W_o backward calls input_grad_fn->apply(),
// it calls THIS apply() which:
// 1. Reshapes the gradient from flat [tokens, d_model] to BHSD [B, H, S, D]
// 2. Continues chain to input's grad_fn (ScaledDotProductAttentionGradFn)
// ═══════════════════════════════════════════════════════════════════════════

struct ReshapeFromBHSDGradFn : public GradFn {
    // Input tensor info
    bool input_requires_grad = false;
    std::shared_ptr<Tensor> input_gradient;
    
    // Dimensions for reshape
    int batch_size = 0;
    int seq_len = 0;
    int num_heads = 0;
    int head_dim = 0;
    
    ReshapeFromBHSDGradFn() { op_name = "reshape_bhsd_to_flat"; }
    
    ~ReshapeFromBHSDGradFn() override {
        release_saved();
    }
    
    void capture_input(Tensor& bhsd_input, cudaStream_t stream) {
        input_requires_grad = bhsd_input.requires_grad;
        if (input_requires_grad) {
            input_gradient = capture_input_gradient(
                bhsd_input, stream, "ReshapeFromBHSDGradFn::capture_input");
        }
    }
    
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        setCurrentGradFnOp("reshape_bhsd_to_flat", this);
        
        if (applied) return;
        applied = true;
        
        if (!input_requires_grad) {
            return;
        }
        if (!input_gradient) {
            throw std::runtime_error("ReshapeFromBHSDGradFn::apply: input gradient Tensor is NULL");
        }
        
        // Reshape gradient from flat [tokens, d_model] to BHSD [B, H, S, D]
        // via TensorConversion's single source of truth geometry kernel.
        TensorConversion::convert_BSM_to_BHSD(
            grad_output.data, input_gradient->data, batch_size, seq_len, num_heads, head_dim, stream);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("ReshapeFromBHSDGradFn::apply: convert_BSM_to_BHSD launch failed: " +
                                         std::string(cudaGetErrorString(err)));
            }
        }
        
        propagate_input_gradient(
            input_gradient,
            stream,
            backward_payload,
            backward_bindings,
            "ReshapeFromBHSDGradFn::apply");
    }
    
    void release_saved() override {
        GradFn::release_saved();
        input_gradient.reset();
    }
};

/**
 * Autograd wrapper over TensorConversion's BHSD->BSM geometry kernel.
 * 
 * ISSUE #62 FIX: This replaces Tensor::empty() + raw BHSD->BSM conversion
 * that broke the autograd chain (output had no grad_fn).
 *
 * This function does NOT create a second geometry owner. TensorConversion still
 * owns the raw BHSD->BSM movement; this wrapper exists because the attention
 * boundary needs a GradFn that can invert that flattening in backward and then
 * continue into `ScaledDotProductAttentionGradFn`.
 */
Tensor reshape_bhsd_to_flat(
    Tensor& bhsd_input,
    const BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
    cudaStream_t stream
) {
    requireEncoderAttentionHP(hp, "reshape_bhsd_to_flat");
    const int batch_size = payload.batch_size;
    const int seq_len = payload.max_seq_len;
    const int num_heads = hp.num_heads;
    const int head_dim = hp.head_dim;
    const int tokens = batch_size * seq_len;
    const int d_model = hp.d_model;

    if (!bhsd_input.data) {
        throw std::runtime_error("reshape_bhsd_to_flat: bhsd_input.data is NULL");
    }
    bhsd_input.shape.require("reshape_bhsd_to_flat bhsd_input");
    if (!bhsd_input.shape.is_4d()) {
        throw std::runtime_error("reshape_bhsd_to_flat: bhsd_input must be BHSD/4D layout");
    }
    
    // Allocate output tensor in flat layout
    Tensor result = Tensor::empty(TensorContract::TensorShape::make_BSM(tokens, d_model), bhsd_input.requires_grad, stream, "reshape_bhsd_to_flat_result");
    
    // TensorContract owns the autograd wrapper; TensorConversion owns raw geometry movement.
    // Do not route this through Layers/Attention/QKV_Projector (deleted stale wrapper).
    TensorConversion::convert_BHSD_to_BSM(
        bhsd_input.data, result.data, batch_size, num_heads, seq_len, head_dim, stream);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("reshape_bhsd_to_flat: convert_BHSD_to_BSM launch failed: " +
                std::string(cudaGetErrorString(err)));
        }
    }
    
    // Set up backward if needed
    if (bhsd_input.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ReshapeFromBHSDGradFn>();
        
        // Capture input for backward
        grad_fn->capture_input(bhsd_input, stream);
        grad_fn->batch_size = batch_size;
        grad_fn->seq_len = seq_len;
        grad_fn->num_heads = num_heads;
        grad_fn->head_dim = head_dim;
        
        result.grad_fn = grad_fn;
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// SplitAndReshapeQKVGradFn - ISSUE #61 FIX: Autograd-tracked QKV split
//
// ROOT CAUSE OF QKV GRADIENT BUG:
// Previously, Encoding_GPU.cu split QKV using cudaMemcpy2D and created new
// Tensor::empty() objects. These have NO grad_fn, breaking the autograd chain!
// When attention backward calls q_grad_fn->apply(), it gets nullptr because
// Q.grad_fn was never set.
//
// FIX: This operation takes qkv_out with its grad_fn and produces Q, K, V
// tensors that have THIS GradFn as their grad_fn. When attention backward
// calls q_grad_fn->apply(), it calls THIS apply() which:
// 1. Combines grad_Q, grad_K, grad_V into an owned grad_qkv scratch buffer
// 2. Calls qkv_out->grad_fn->apply() to continue the chain to W_qkv / b_qkv
// ═══════════════════════════════════════════════════════════════════════════

// NOTE: QKV split/merge kernels live in TensorConversion.cu - single source of truth.

/**
 * GradFn for split_and_reshape_qkv operation
 * 
 * This bridges the gap between qkv_out (from matmul) and Q_bhsd/K_bhsd/V_bhsd (for attention).
 * 
 * Forward: qkv_out [tokens, qkv_dim] -> Q_bhsd, K_bhsd, V_bhsd [batch, heads, seq, head_dim]
 * Backward: grad_Q_bhsd, grad_K_bhsd, grad_V_bhsd -> grad_qkv_out -> W_qkv gradients
 */
struct SplitAndReshapeQKVGradFn : public GradFn {
    // Which output this GradFn is attached to (Q, K, or V)
    enum class OutputType { Q, K, V };
    OutputType output_type = OutputType::Q;
    
    // Shared state for all three outputs (only one instance owns the upstream chain)
    struct SharedState {
        std::shared_ptr<GradFn> qkv_grad_fn;       // Grad fn of qkv_out (the matmul result)
        TensorContract::TensorShape qkv_shape;
        bool qkv_requires_grad = false;
        bool qkv_input_is_leaf = false;
        size_t qkv_numel = 0;
        float* qkv_leaf_grad = nullptr;            // Persistent leaf grad buffer, if qkv_out is a leaf
        float* merged_qkv_grad = nullptr;          // Owned local Jacobian result [tokens, qkv_dim]
        std::shared_ptr<float> owned_merged_qkv_grad;
        
        // BHSD gradient pointers from Q, K, V backward passes
        // Stored from owned copies of grad_output.data — no intermediate BSM buffers needed.
        // TensorConversion reads these in BHSD layout and writes flat QKV grad.
        const float* grad_Q_bhsd = nullptr;  // [batch, num_heads, seq, head_dim]
        const float* grad_K_bhsd = nullptr;  // [batch, num_kv_heads, seq, head_dim]
        const float* grad_V_bhsd = nullptr;  // [batch, num_kv_heads, seq, head_dim]
        
        // Owned references that keep the gradient buffers alive until the merge
        // kernel has been submitted. Without these, an upstream GradFn's
        // release_saved() can free the buffer before the 3rd apply() fires the
        // merge that reads all three pointers.
        std::shared_ptr<float> owned_grad_Q;
        std::shared_ptr<float> owned_grad_K;
        std::shared_ptr<float> owned_grad_V;
        
        // Dimensions
        int tokens = 0;
        int batch = 0;
        int seq = 0;
        int num_heads = 0;
        int num_kv_heads = 0;
        int head_dim = 0;
        
        // Count of how many outputs have been processed
        std::atomic<int> apply_count{0};
        
        ~SharedState() {
            // shared_ptr members destruct automatically
        }
    };
    
    std::shared_ptr<SharedState> shared;
    
    SplitAndReshapeQKVGradFn() { op_name = "split_and_reshape_qkv"; }

    // Engine topology: the three Q/K/V instances share one upstream (qkv_out).
    // The merge that calls qkv_grad_fn->apply() fires exactly once (on whichever
    // instance runs third), so exactly ONE instance must report the edge for the
    // in-degree count to match. We designate the Q instance. When qkv_out is a
    // leaf, the merge accumulates terminally into its registry grad buffer and
    // there is no upstream edge.
    void collect_input_edges(std::vector<GradFn*>& out) const override {
        out.clear();
        if (output_type == OutputType::Q && shared && shared->qkv_grad_fn) {
            out.push_back(shared->qkv_grad_fn.get());
        }
    }
    
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        const char* type_str = (output_type == OutputType::Q) ? "Q" : 
                               (output_type == OutputType::K) ? "K" : "V";
        
        if (!shared) {
            throw std::runtime_error(std::string("[SplitQKV-") + type_str + "] shared is null");
        }
        if (applied) {
            return;
        }
        applied = true;
        
        auto& state = *shared;
        
        // Copy grad_output into an owned buffer.  The upstream GradFn (e.g.
        // RoPEGradFn) may call release_saved() which frees its buffer BEFORE
        // the 3rd apply() fires the fused kernel that reads all three pointers.
        // Owning a copy here guarantees the data survives.
        const std::size_t n_bytes = grad_output.numel() * sizeof(float);
        float* owned_buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&owned_buf), n_bytes, "SplitQKV_owned_grad");
        cudaMemcpyAsync(owned_buf, grad_output.data, n_bytes, cudaMemcpyDeviceToDevice, stream);
        
        if (output_type == OutputType::Q) {
            state.owned_grad_Q.reset(owned_buf, [](float* p) { queueForDeferredCleanup(p); });
            state.grad_Q_bhsd = owned_buf;
        } else if (output_type == OutputType::K) {
            state.owned_grad_K.reset(owned_buf, [](float* p) { queueForDeferredCleanup(p); });
            state.grad_K_bhsd = owned_buf;
        } else {  // V
            state.owned_grad_V.reset(owned_buf, [](float* p) { queueForDeferredCleanup(p); });
            state.grad_V_bhsd = owned_buf;
        }
        
        // Check if all three outputs have been processed
        const int count = state.apply_count.fetch_add(1) + 1;
        
        if (count == 3) {
            if (!state.qkv_requires_grad) {
                throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: qkv_out does not require grad but SplitQKV GradFn was invoked");
            }
            if (!state.grad_Q_bhsd || !state.grad_K_bhsd || !state.grad_V_bhsd) {
                throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: missing Q/K/V gradient before merge");
            }
            if (!state.merged_qkv_grad) {
                throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: merged_qkv_grad is NULL - forward capture failed");
            }

            // All three BHSD gradient pointers collected. Delegate the BHSD -> flat
            // merge to TensorConversion, the single source of truth for QKV layout.
            TensorConversion::merge_qkv_grads_gqa(
                state.grad_Q_bhsd, state.grad_K_bhsd, state.grad_V_bhsd,
                state.merged_qkv_grad,
                state.batch, state.num_heads, state.num_kv_heads, state.seq, state.head_dim,
                stream);
            {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: merge_qkv_grads_gqa launch failed: " +
                        std::string(cudaGetErrorString(err)));
                }
            }

            logGradFlowTensorStats("SplitQKV.merge grad_Q_bhsd",
                                   state.grad_Q_bhsd,
                                   static_cast<std::size_t>(state.batch) * state.num_heads * state.seq * state.head_dim,
                                   stream);
            logGradFlowTensorStats("SplitQKV.merge grad_K_bhsd",
                                   state.grad_K_bhsd,
                                   static_cast<std::size_t>(state.batch) * state.num_kv_heads * state.seq * state.head_dim,
                                   stream);
            logGradFlowTensorStats("SplitQKV.merge grad_V_bhsd",
                                   state.grad_V_bhsd,
                                   static_cast<std::size_t>(state.batch) * state.num_kv_heads * state.seq * state.head_dim,
                                   stream);
            logGradFlowTensorStats("SplitQKV.merge merged_qkv_grad",
                                   state.merged_qkv_grad,
                                   state.qkv_numel,
                                   stream);

            if (state.qkv_input_is_leaf) {
                if (!state.qkv_leaf_grad) {
                    throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: qkv_out is leaf but qkv_leaf_grad is NULL");
                }
                accumulate_grad(state.qkv_leaf_grad,
                                state.merged_qkv_grad,
                                state.qkv_numel,
                                1.0f,
                                stream,
                                "SplitAndReshapeQKVGradFn::apply qkv_leaf_grad");
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: leaf qkv gradient accumulation failed: " +
                        std::string(cudaGetErrorString(err)));
                }
                return;
            }

            // Continue the chain to qkv_out -> W_qkv / b_qkv. Non-leaf qkv_out
            // gradients are local scratch owned by this GradFn, not qkv_out.grad().
            if (state.qkv_grad_fn) {
                Tensor qkv_grad_tensor;
                qkv_grad_tensor.data = state.merged_qkv_grad;
                qkv_grad_tensor.shape = state.qkv_shape;
                qkv_grad_tensor.owns_data = false;
                qkv_grad_tensor.stream = stream;
                
                state.qkv_grad_fn->apply(qkv_grad_tensor, stream, backward_payload, backward_bindings);
                // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
            } else {
                throw std::runtime_error("SplitAndReshapeQKVGradFn::apply: non-leaf qkv_out has NULL upstream grad_fn");
            }
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        // SharedState cleanup happens via shared_ptr destructor
    }
};

/**
 * Split qkv_out [tokens, qkv_dim] into Q, K, V and reshape to BHSD layout.
 * 
 * This replaces the manual cudaMemcpy2D + launchQKVReshapeToBHSD in Encoding_GPU.cu
 * with a properly autograd-tracked operation.
 * 
 * @param qkv_out  Input tensor [tokens, d_model + 2*kv_dim] from matmul(ln1_out, W_qkv)
 * @param payload  BatchPayload source of truth for batch/sequence geometry
 * @param hp       Grouped attention HP source of truth for GQA/head geometry
 * @param stream   CUDA stream
 * @return Tuple of (Q_bhsd, K_bhsd, V_bhsd) with autograd tracking
 */
std::tuple<Tensor, Tensor, Tensor> split_and_reshape_qkv(
    Tensor& qkv_out,
    const BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
    cudaStream_t stream
) {
    requireEncoderAttentionHP(hp, "split_and_reshape_qkv");
    const int batch = payload.batch_size;
    const int seq = payload.max_seq_len;
    const int num_heads = hp.num_heads;
    const int num_kv_heads = hp.num_kv_heads;
    const int head_dim = hp.head_dim;
    const int tokens = batch * seq;
    const int qkv_dim = hp.qkv_dim;
    
    if (!qkv_out.data) {
        throw std::runtime_error("split_and_reshape_qkv: qkv_out.data is NULL");
    }
    qkv_out.shape.require("split_and_reshape_qkv qkv_out");
    if (!qkv_out.shape.is_2d_layout()) {
        throw std::invalid_argument("split_and_reshape_qkv: qkv_out must be a 2D flat/QKV tensor");
    }

    // Validate input shape
    if (qkv_out.shape.flat.rows != tokens || qkv_out.shape.flat.cols != qkv_dim) {
        throw std::invalid_argument(
            "split_and_reshape_qkv: qkv_out shape mismatch. Expected [" + 
            std::to_string(tokens) + ", " + std::to_string(qkv_dim) + 
            "], got [" + std::to_string(qkv_out.shape.flat.rows) + ", " + 
            std::to_string(qkv_out.shape.flat.cols) + "]");
    }
    
    // Allocate output tensors in BHSD layout
    auto q_shape = TensorContract::TensorShape::make_BHSD(batch, num_heads, seq, head_dim);
    auto k_shape = TensorContract::TensorShape::make_BHSD(batch, num_kv_heads, seq, head_dim);
    auto v_shape = TensorContract::TensorShape::make_BHSD(batch, num_kv_heads, seq, head_dim);
    
    bool requires_grad = qkv_out.requires_grad;
    Tensor Q_bhsd = Tensor::zeros(q_shape, requires_grad, stream, "qkv_split_Q");
    Tensor K_bhsd = Tensor::zeros(k_shape, requires_grad, stream, "qkv_split_K");
    Tensor V_bhsd = Tensor::zeros(v_shape, requires_grad, stream, "qkv_split_V");
    
    // Inference-only: drain stale CUDA error from earlier in this layer (rms_norm,
    // matmul, broadcast_add). Without this, cudaGetLastError() after the launch
    // reports that stale error and we throw incorrectly.
    (void)cudaGetLastError();
    
    // Forward: split fused qkv_out directly to BHSD using TensorConversion's
    // canonical GQA-aware split kernel. Do not create a second split/reshape truth here.
    TensorConversion::split_qkv_gqa(
        qkv_out.data, Q_bhsd.data, K_bhsd.data, V_bhsd.data,
        batch, num_heads, num_kv_heads, seq, head_dim, stream);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("split_and_reshape_qkv: split_qkv_gqa launch failed: " +
                std::string(cudaGetErrorString(err)));
        }
    }
    
    // Set up backward if needed
    if (requires_grad) {
        // Create shared state for all three outputs
        auto shared = std::make_shared<SplitAndReshapeQKVGradFn::SharedState>();
        shared->tokens = tokens;
        shared->batch = batch;
        shared->seq = seq;
        shared->num_heads = num_heads;
        shared->num_kv_heads = num_kv_heads;
        shared->head_dim = head_dim;
        shared->qkv_shape = qkv_out.shape;
        shared->qkv_requires_grad = qkv_out.requires_grad;
        shared->qkv_input_is_leaf = qkv_out.is_leaf;
        shared->qkv_numel = qkv_out.numel();
        
        float* merged_qkv_grad = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&merged_qkv_grad),
                          shared->qkv_numel * sizeof(float),
                          "SplitQKV_merged_qkv_grad");
        cudaMemsetAsync(merged_qkv_grad, 0, shared->qkv_numel * sizeof(float), stream);
        shared->owned_merged_qkv_grad.reset(merged_qkv_grad, [](float* p) {
            queueForDeferredCleanup(p);
        });
        shared->merged_qkv_grad = merged_qkv_grad;

        if (qkv_out.is_leaf) {
            qkv_out.ensure_grad();
            shared->qkv_leaf_grad = qkv_out.grad_data();
            if (!shared->qkv_leaf_grad) {
                throw std::runtime_error("split_and_reshape_qkv: qkv_out leaf grad_data is NULL after ensure_grad");
            }
        } else {
            shared->qkv_grad_fn = qkv_out.grad_fn;
            if (!shared->qkv_grad_fn) {
                throw std::runtime_error("split_and_reshape_qkv: qkv_out is non-leaf and requires grad, but qkv_out.grad_fn is NULL");
            }
        }
        
        // Create GradFns for each output
        auto q_grad_fn = std::make_shared<SplitAndReshapeQKVGradFn>();
        q_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::Q;
        q_grad_fn->shared = shared;
        
        auto k_grad_fn = std::make_shared<SplitAndReshapeQKVGradFn>();
        k_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::K;
        k_grad_fn->shared = shared;
        
        auto v_grad_fn = std::make_shared<SplitAndReshapeQKVGradFn>();
        v_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::V;
        v_grad_fn->shared = shared;
        
        Q_bhsd.is_leaf = false;
        Q_bhsd.grad_fn = q_grad_fn;
        
        K_bhsd.is_leaf = false;
        K_bhsd.grad_fn = k_grad_fn;
        
        V_bhsd.is_leaf = false;
        V_bhsd.grad_fn = v_grad_fn;
    }
    
    return {std::move(Q_bhsd), std::move(K_bhsd), std::move(V_bhsd)};
}




struct RoPEGradFn : public GradFn {
    /**
     * OutputType identifies which tensor (Q or K) this GradFn is attached to.
     */
    enum class OutputType { Q, K };
    OutputType output_type = OutputType::Q;
    
    /**
     * SharedState holds the common data for coordinating Q and K backward passes.
     * - Stores upstream grad_fns for Q and K (from split_and_reshape_qkv)
     * - Stores RoPE parameters (inv_freq, dimensions)
     * - Uses atomic counter to detect when both Q and K backward are complete
     */
    struct SharedState {
        // Upstream grad_fns for Q and K (from split_and_reshape_qkv output)
        std::shared_ptr<GradFn> q_upstream_grad_fn;
        std::shared_ptr<GradFn> k_upstream_grad_fn;
        
        // ISSUE #48 FIX: Don't store Tensor by value - operator= is deleted
        // Instead, store only what we need for backward: requires_grad flags
        bool q_requires_grad = false;
        bool k_requires_grad = false;
        
        // RoPE parameters (captured at forward time)
        GRIM::HyperParameters::EncoderSelfAttentionHP hp{};
        const float* inv_freq = nullptr;
        int rotary_dim = 0;
        int pos_offset = 0;
        
        // Atomic counter: when reaches 2, both Q and K backward complete
        std::atomic<int> apply_count{0};
        
        ~SharedState() {
            // shared_ptr members destruct automatically
        }
    };
    
    std::shared_ptr<SharedState> shared;
    
    // Owned buffer for the inverse-rotated gradient.
    // Keeps the data alive until release_saved(), which prevents
    // SplitAndReshapeQKVGradFn from reading a dangling pointer.
    std::shared_ptr<float> owned_grad_buf;

    RoPEGradFn() { op_name = "rope_rotation"; }

    // Engine topology: Q and K are independent 1-in/1-out instances (the atomic
    // apply_count is vestigial). Each reports only its own upstream edge.
    void collect_input_edges(std::vector<GradFn*>& out) const override {
        out.clear();
        if (!shared) {
            return;
        }
        if (output_type == OutputType::Q) {
            if (shared->q_requires_grad && shared->q_upstream_grad_fn) {
                out.push_back(shared->q_upstream_grad_fn.get());
            }
        } else {
            if (shared->k_requires_grad && shared->k_upstream_grad_fn) {
                out.push_back(shared->k_upstream_grad_fn.get());
            }
        }
    }

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        (void)backward_bindings;
        if (!shared) {
            throw std::runtime_error("RoPEGradFn::apply: shared state is NULL - RoPE forward must initialize shared state");
        }
        auto& state = *shared;
        
        const char* type_str = (output_type == OutputType::Q) ? "Q" : "K";
        // Shape is 4D (BHSD layout) - use as_4d() accessor
        const auto& s4d = grad_output.shape.as_4d();
        AG_TRACE("[RoPEGradFn-%s] apply() ENTER grad_output.data=%p shape=[%d,%d,%d,%d]\n",
                 type_str, grad_output.data, 
                 s4d.batch, s4d.heads, s4d.seq, s4d.head_dim);
        
        // Allocate a separate buffer and copy grad_output into it.
        // Inverse RoPE rotation is applied to this copy — grad_output is NOT mutated.
        const std::size_t n_elems = grad_output.numel();
        float* grad_buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&grad_buf), n_elems * sizeof(float), "RoPEGradFn_grad_buf");
        cudaMemcpyAsync(grad_buf, grad_output.data, n_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        owned_grad_buf.reset(grad_buf, [](float* p) { queueForDeferredCleanup(p); });

        if (!backward_payload) {
            throw std::runtime_error("RoPEGradFn::apply: backward_payload is NULL - orchestration MUST pass the active BatchPayload into backward()");
        }
        backward_payload->validate("RoPEGradFn::apply");
        
        if (output_type == OutputType::Q) {
            // Inverse-rotate dQ only - K gradients handled by K's GradFn
            PBM::launchRoPERotationGQA_backward(
                grad_buf,         // dQ copy - modified in-place
                nullptr,          // dK = nullptr, handle separately
                state.inv_freq,
                *backward_payload,
                state.hp,
                state.rotary_dim,
                stream,
                state.pos_offset
            );
            AG_TRACE("[RoPEGradFn-Q] Inverse RoPE applied to dQ copy\n");
        } else {
            // Inverse-rotate dK only - Q gradients handled by Q's GradFn
            PBM::launchRoPERotationGQA_backward(
                nullptr,          // dQ = nullptr, handle separately
                grad_buf,         // dK copy - modified in-place
                state.inv_freq,
                *backward_payload,
                state.hp,
                state.rotary_dim,
                stream,
                state.pos_offset
            );
            AG_TRACE("[RoPEGradFn-K] Inverse RoPE applied to dK copy\n");
        }
        
        // Build a tensor view over the owned copy for downstream propagation
        Tensor rotated_grad;
        rotated_grad.data = grad_buf;
        rotated_grad.shape = grad_output.shape;
        rotated_grad.owns_data = false;  // owned_grad_buf controls lifetime
        rotated_grad.stream = stream;
        
        // Increment counter to track completion
        const int count = state.apply_count.fetch_add(1) + 1;
        AG_TRACE("[RoPEGradFn-%s] apply_count = %d/2\n", type_str, count);
        
        // Continue upstream chain for THIS output immediately
        // (Unlike SplitAndReshapeQKV, we don't need to wait for both because
        //  Q and K have independent upstream paths after split_and_reshape_qkv)
        if (output_type == OutputType::Q) {
            if (state.q_requires_grad && state.q_upstream_grad_fn) {
                AG_TRACE("[RoPEGradFn-Q] Continuing to q_upstream_grad_fn...\n");
                state.q_upstream_grad_fn->apply(rotated_grad, stream, backward_payload, backward_bindings);
                // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
            }
        } else {
            if (state.k_requires_grad && state.k_upstream_grad_fn) {
                AG_TRACE("[RoPEGradFn-K] Continuing to k_upstream_grad_fn...\n");
                state.k_upstream_grad_fn->apply(rotated_grad, stream, backward_payload, backward_bindings);
                // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
            }
        }
        
        AG_TRACE("[RoPEGradFn-%s] apply() EXIT\n", type_str);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        owned_grad_buf.reset();
        // SharedState cleanup happens via shared_ptr destructor
    }
};


/**
 * Apply RoPE rotation to Q and K tensors OUT-OF-PLACE with autograd tracking.
 * 
 * Allocates new buffers for Q_rot and K_rot, copies the inputs, applies RoPE
 * to the copies, and returns the rotated tensors. The input tensors are never
 * mutated — their data and autograd metadata remain untouched.
 * 
 * RoPE Math:
 *   Forward:  Q' = R(θ) * Q,  K' = R(θ) * K   (rotation by position-dependent angle)
 *   Backward: dQ = R(-θ) * dQ', dK = R(-θ) * dK'  (inverse rotation)
 * 
 * Since R(-θ) = R(θ)^T and R is orthogonal, this is mathematically correct.
 * 
 * @return {Q_rot, K_rot} — new tensors with owns_data=true
 */
std::pair<Tensor, Tensor> rope_rotation(
    const Tensor& Q,
    const Tensor& K,
    const float* inv_freq,
    const BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
    int rotary_dim,
    cudaStream_t stream,
    int pos_offset
) {
    requireEncoderAttentionHP(hp, "rope_rotation");
    payload.validate("rope_rotation");
    // RULE 20: Fail loud validation
    if (!Q.data) {
        throw std::runtime_error("rope_rotation: Q.data is NULL");
    }
    if (!K.data) {
        throw std::runtime_error("rope_rotation: K.data is NULL");
    }
    if (!inv_freq) {
        throw std::runtime_error("rope_rotation: inv_freq is NULL");
    }
    if (rotary_dim <= 0 || rotary_dim > hp.head_dim) {
        throw std::runtime_error("rope_rotation: invalid rotary_dim=" + std::to_string(rotary_dim) +
                                 " (head_dim=" + std::to_string(hp.head_dim) + ")");
    }
    
    AG_TRACE("[rope_rotation] ENTER Q.data=%p K.data=%p batch=%d seq=%d heads=%d/%d dim=%d rotary=%d\n",
             Q.data, K.data, payload.batch_size, payload.max_seq_len, hp.num_heads, hp.num_kv_heads, hp.head_dim, rotary_dim);
    
    // Allocate separate output buffers — input tensors are never mutated.
    const std::size_t q_bytes = Q.numel() * sizeof(float);
    const std::size_t k_bytes = K.numel() * sizeof(float);
    
    float* q_rot_data = nullptr;
    float* k_rot_data = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&q_rot_data), q_bytes, "rope_Q_rot");
    cudaMallocOrThrow(reinterpret_cast<void**>(&k_rot_data), k_bytes, "rope_K_rot");
    
    cudaMemcpyAsync(q_rot_data, Q.data, q_bytes, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(k_rot_data, K.data, k_bytes, cudaMemcpyDeviceToDevice, stream);
    
    // Forward pass: Apply RoPE rotation to the COPIES
    PBM::launchRoPERotationGQA(
        q_rot_data, k_rot_data, inv_freq,
        payload, hp, rotary_dim,
        stream, pos_offset
    );
    
    // Build output tensors that own their buffers
    Tensor Q_rot;
    Q_rot.data = q_rot_data;
    Q_rot.shape = Q.shape;
    Q_rot.owns_data = true;
    Q_rot.requires_grad = Q.requires_grad;
    Q_rot.is_leaf = false;
    Q_rot.stream = stream;
    
    Tensor K_rot;
    K_rot.data = k_rot_data;
    K_rot.shape = K.shape;
    K_rot.owns_data = true;
    K_rot.requires_grad = K.requires_grad;
    K_rot.is_leaf = false;
    K_rot.stream = stream;
    
    // Setup backward pass if either tensor requires gradients
    const bool requires_grad = Q.requires_grad || K.requires_grad;
    
    if (requires_grad) {
        AG_TRACE("[rope_rotation] Setting up RoPEGradFn for backward...\n");
        
        // Create shared state for coordinating Q and K backward
        auto shared = std::make_shared<RoPEGradFn::SharedState>();
        
        // Chain to the INPUT tensors' grad_fns (not the outputs')
        if (Q.grad_fn) {
            shared->q_upstream_grad_fn = Q.grad_fn;
        }
        if (K.grad_fn) {
            shared->k_upstream_grad_fn = K.grad_fn;
        }
        
        shared->q_requires_grad = Q.requires_grad;
        shared->k_requires_grad = K.requires_grad;
        
        // Capture RoPE parameters
        shared->hp = hp;
        shared->inv_freq = inv_freq;
        shared->rotary_dim = rotary_dim;
        shared->pos_offset = pos_offset;
        
        // Attach GradFn to the OUTPUT tensors
        auto q_grad_fn = std::make_shared<RoPEGradFn>();
        q_grad_fn->output_type = RoPEGradFn::OutputType::Q;
        q_grad_fn->shared = shared;
        Q_rot.grad_fn = q_grad_fn;
        
        auto k_grad_fn = std::make_shared<RoPEGradFn>();
        k_grad_fn->output_type = RoPEGradFn::OutputType::K;
        k_grad_fn->shared = shared;
        K_rot.grad_fn = k_grad_fn;
        
        AG_TRACE("[rope_rotation] RoPEGradFn attached: Q_rot.grad_fn=%p K_rot.grad_fn=%p\n",
                 (void*)Q_rot.grad_fn.get(), (void*)K_rot.grad_fn.get());
    }
    
    AG_TRACE("[rope_rotation] EXIT\n");
    return {std::move(Q_rot), std::move(K_rot)};
}

}  // namespace autograd
}  // namespace GRIM
