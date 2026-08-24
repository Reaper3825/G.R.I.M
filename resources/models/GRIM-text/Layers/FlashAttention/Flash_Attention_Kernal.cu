// Flash_Attention_Kernal.cu
// FlashAttention v2 forward/backward kernels (SM86, BF16/FP16, head_dim 32/64).
//
// Portions adapted from https://github.com/Dao-AILab/flash-attention
// Copyright (c) 2022-2024, Tri Dao and contributors.
// Licensed under BSD 3-Clause (see external/flash-attention/LICENSE).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "AttentionDiagnostics.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/VerboseLogging.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

// Inline alias replacing module_logger.hpp template
namespace {
struct FlashAttentionLog {
    static void info(std::string_view msg, std::uint64_t step = 0) {
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Activations, msg, step);
    }
    static void warn(std::string_view msg, std::uint64_t step = 0) {
        GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Activations, msg, step);
    }
    static void error(std::string_view msg, std::uint64_t step = 0) {
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Activations, msg, step);
    }
};
}

#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_SOFTCAP
// FLASHATTENTION_DISABLE_UNEVEN_K defined via CMake target_compile_definitions
// NOTE: FLASHATTENTION_DISABLE_DROPOUT removed per Rule 20 - feature dropout is now enabled
// (was silently ignored before despite config having attention_dropout parameter)

// ============================================================================
// IMPORTANT: FlashAttention pinned API contract.
//
// The Bridges-2 sync path pins external/flash-attention to the superproject
// gitlink and pins nested Cutlass separately. That FlashAttention revision uses
// namespace FLASH_NAMESPACE and a forward signature shaped like:
//   compute_attn<..., Is_even_K, Is_softcap, Return_softmax>(params)
//
// This wrapper defines FLASH_NAMESPACE as grim_flash before including the
// vendored headers, so upstream compute_* functions and GRIM-owned params must
// live in grim_flash on both Windows and Linux. If `compute_attn` fails again,
// verify the Bridges-2 FAS sync path and the pinned header API before editing
// vendored flash-attention source.
// ============================================================================

// ATen stub required before flash-attention headers (flash_fwd_kernel.h uses at::cuda::philox::unpack).
#include <ATen/cuda/detail/UnpackRaw.cuh>

#define FLASH_NAMESPACE grim_flash

#define FLASH_UPSTREAM_NS grim_flash

#include "namespace_config.h"
#include "static_switch.h"
#include "hardware_info.h"
#include "kernel_traits.h"
#include "block_info.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "flash_fwd_kernel.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

// ============================================================================
// Reconcile flash-attention namespace:
//
// The pinned FlashAttention headers respect FLASH_NAMESPACE and therefore emit
// compute_* functions into grim_flash. Keep GRIM's params in the same namespace
// so every downstream caller uses grim_flash:: uniformly.
// ============================================================================

#define FLASH_PARAMS_NS grim_flash

namespace FLASH_PARAMS_NS {
// Copy of flash.h parameter structs (ATen removed via philox_unpack.cuh stub).
struct Qkv_params {
    using index_t = int64_t;
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    int h;
    int h_k;
    int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
    void* __restrict__ o_ptr;
    void* __restrict__ oaccum_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void* __restrict__ p_ptr;

    void* __restrict__ softmax_lse_ptr;
    void* __restrict__ softmax_lseaccum_ptr;

    int b;
    int seqlen_q;
    int seqlen_k;
    int seqlen_knew;
    int d;
    int seqlen_q_rounded;
    int seqlen_k_rounded;
    int d_rounded;
    int rotary_dim;
    int total_q;

    float scale_softmax;
    float scale_softmax_log2;

    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;
    int* __restrict__ leftpad_k;
    int* __restrict__ seqused_k;

    int* __restrict__ blockmask;

    void* __restrict__ knew_ptr;
    void* __restrict__ vnew_ptr;

    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    void* __restrict__ rotary_cos_ptr;
    void* __restrict__ rotary_sin_ptr;

    int* __restrict__ cache_batch_idx;

    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    float p_dropout;
    uint8_t p_dropout_in_uint8_t;

    float rp_dropout;
    float scale_softmax_rp_dropout;

    int window_size_left;
    int window_size_right;
    float softcap;

    at::PhiloxCudaState philox_args;

    uint64_t* rng_state;

    bool is_bf16;
    bool is_causal;
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;

    void* __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;
};

struct Flash_bwd_params : public Flash_fwd_params {
    void* __restrict__ do_ptr;
    void* __restrict__ dq_ptr;
    void* __restrict__ dk_ptr;
    void* __restrict__ dv_ptr;

    void* __restrict__ dq_accum_ptr;
    void* __restrict__ dk_accum_ptr;
    void* __restrict__ dv_accum_ptr;

    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    void* __restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};
}  // namespace FLASH_PARAMS_NS

namespace grim_flash {
namespace detail {

constexpr float kLog2e = 1.4426950408889634f;

inline int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

inline void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        std::string error_msg = std::string("[FlashAttention] CUDA error at ") + where + ": " + cudaGetErrorString(status);
        FlashAttentionLog::error(error_msg);
        throw std::runtime_error(error_msg);
    }
}

inline bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

inline bool check_no_alias(const char* name_a, const void* a, const char* name_b, const void* b) {
    if (a && b && a == b) {
        std::string error_msg = std::string("[FlashAttention] ") + name_a + " must not alias " + name_b;
        FlashAttentionLog::error(error_msg);
        return false;
    }
    return true;
}

inline bool validate_bwd_workspace_pointers(const void* dq_accum,
                                            const void* dsoftmax_sum,
                                            const void* q,
                                            const void* k,
                                            const void* v,
                                            const void* out,
                                            const void* dout,
                                            const void* dq,
                                            const void* dk,
                                            const void* dv,
                                            const void* softmax_lse) {
    bool ok = true;
    ok &= check_no_alias("dq_accum", dq_accum, "dsoftmax_sum", dsoftmax_sum);
    ok &= check_no_alias("dq_accum", dq_accum, "dq", dq);
    ok &= check_no_alias("dq_accum", dq_accum, "dk", dk);
    ok &= check_no_alias("dq_accum", dq_accum, "dv", dv);
    ok &= check_no_alias("dq_accum", dq_accum, "out", out);
    ok &= check_no_alias("dq_accum", dq_accum, "dout", dout);
    ok &= check_no_alias("dq_accum", dq_accum, "q", q);
    ok &= check_no_alias("dq_accum", dq_accum, "k", k);
    ok &= check_no_alias("dq_accum", dq_accum, "v", v);
    ok &= check_no_alias("dq_accum", dq_accum, "softmax_lse", softmax_lse);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "dq", dq);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "dk", dk);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "dv", dv);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "out", out);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "dout", dout);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "q", q);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "k", k);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "v", v);
    ok &= check_no_alias("dsoftmax_sum", dsoftmax_sum, "softmax_lse", softmax_lse);
    return ok;
}

inline size_t dq_accum_bytes(int batch, int seqlen, int n_heads, int head_dim) {
    const int seqlen_rounded = round_multiple(seqlen, 128);
    const int d_rounded = round_multiple(head_dim, head_dim <= 128 ? 32 : 64);
    return static_cast<size_t>(batch) * seqlen_rounded * n_heads * d_rounded * sizeof(float);
}

inline size_t dsoftmax_sum_bytes(int batch, int seqlen, int n_heads) {
    const int seqlen_rounded = round_multiple(seqlen, 128);
    return static_cast<size_t>(batch) * n_heads * seqlen_rounded * sizeof(float);
}

inline float require_softmax_scale(float requested_scale, int head_dim, const char* caller) {
    if (head_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": head_dim must be > 0, got " + std::to_string(head_dim));
    }
    if (!std::isfinite(requested_scale) || requested_scale <= 0.0f) {
        throw std::runtime_error(std::string(caller) + ": softmax_scale must be finite and > 0, got " + std::to_string(requested_scale));
    }
    return requested_scale;
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN,
         bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_kernel(const FLASH_PARAMS_NS::Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_UPSTREAM_NS::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                       Is_even_MN, Is_even_K, /*Is_softcap=*/false, Return_softmax>(params);
#else
    if (threadIdx.x == 0) {
        printf("FATAL: FlashAttention requires SM80+.\n");
    }
#endif
}

// Split-KV forward kernel wrapper (inference KV-cache decode path).
//
// This is the ONLY FlashAttention kernel that contains the fused-rotary
// Append_KV code path (vendored rotary.h copy_rotary_interleaved /
// copy_rotary_contiguous). GRIM's training forward uses the non-split
// compute_attn kernel above and never touches rotary; the decode primitive
// (flash_attn_fwd_kvcache_rotary) routes here so the new K token is rotated and
// appended to the cache inside the kernel and Q is rotated on load.
//
// GRIM only ever launches this with Split=false (num_splits==1): the epilogue
// then writes the final output straight to o_ptr/softmax_lse_ptr, so no split
// accumulation workspace and no combine kernel are required.
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV>
__global__ void flash_fwd_splitkv_kernel(const FLASH_PARAMS_NS::Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_UPSTREAM_NS::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi,
                       Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params);
#else
    if (threadIdx.x == 0) {
        printf("FATAL: FlashAttention requires SM80+.\n");
    }
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(const FLASH_PARAMS_NS::Flash_bwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_UPSTREAM_NS::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_local,
                                                      Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
#else
    if (threadIdx.x == 0) {
        printf("FATAL: FlashAttention requires SM80+.\n");
    }
#endif
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const FLASH_PARAMS_NS::Flash_bwd_params params, const int nsplits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_UPSTREAM_NS::convert_dQ<Kernel_traits>(params, nsplits);
#else
    if (threadIdx.x == 0) {
        printf("FATAL: FlashAttention requires SM80+.\n");
    }
#endif
}

// ============================================================================
// ISSUE #84 FIX: CRITICAL - Preprocessing kernel for FlashAttention backward
// ============================================================================
// This kernel computes dP_sum = dot(dO, O) for ALL query positions BEFORE the
// main backward kernel runs. Without this, the dsoftmax_sum buffer contains
// garbage for most positions, causing dQ/dK gradient explosion.
//
// The main backward kernel reads gdPsum for ALL m_blocks, expecting valid
// pre-computed dP_sum values from this preprocessing pass.
//
// ROOT CAUSE: GRIM was missing this preprocessing kernel, so gdPsum contained
// uninitialized memory (cudaMalloc doesn't zero). This caused:
// - dQ/dK to explode by 100,000-500,000x despite dO being near-zero
// - Layers 9-0 showed exploding gradients while layers 11-10 were correct
//   (first 2 layers' starting m_blocks got valid data from inline dot_do_o)
// ============================================================================
template<bool Clear_dQaccum, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const FLASH_PARAMS_NS::Flash_bwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    FLASH_UPSTREAM_NS::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
#else
    if (threadIdx.x == 0) {
        printf("FATAL: FlashAttention requires SM80+.\n");
    }
#endif
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            params.seqused_k == nullptr &&
                            (params.seqlen_k % Kernel_traits::kBlockN == 0) &&
                            (params.seqlen_q % Kernel_traits::kBlockM == 0);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, HasAlibi, [&] {
                // Attention dropout: switch template based on p_dropout
                const bool use_dropout = params.p_dropout < 1.0f;
                BOOL_SWITCH(use_dropout, IsDropout, [&] {
                auto kernel = &flash_fwd_kernel<Kernel_traits,
                                                IsDropout,
                                                Is_causal,
                                                /*Is_local=*/false,
                                                HasAlibi,
                                                IsEvenMNConst && IsEvenKConst,
                                                IsEvenKConst,
                                                /*Return_softmax=*/false>;
                if (smem_size >= 48 * 1024) {
                    check_cuda(cudaFuncSetAttribute(kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    smem_size),
                               "cudaFuncSetAttribute(fwd)");
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                check_cuda(cudaGetLastError(), "flash_fwd_kernel launch");
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(Flash_bwd_params& params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize1colblock;
    if (params.deterministic) {
        throw std::runtime_error("run_flash_bwd: deterministic=true requires split workspace and nsplits-aware dQ conversion; GRIM only supports deterministic=false here");
    }
    
    // ===========================================================================
    // FlashAttention backward launch contract
    // ===========================================================================
    // Preprocessing kernel: one block per query tile (num_m_block, batch, heads).
    // Main kernel: pinned Dao launcher uses seqK-parallel grid
    // (num_n_block, batch, heads), then converts accumulated dQ from fp32 workspace.
    //
    // The direct compute_dq_dk_dv launch (grid=(batch, heads, 1)) is not the
    // active upstream path for this pinned revision and produced corrupt dK/dV under
    // GQA+dropout while dQ looked sane. Keep this wrapper aligned with the vendored
    // launcher instead of maintaining a second launch contract.
    // ===========================================================================
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_preprocess(num_m_block, params.b, params.h);  // For preprocessing
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    dim3 grid_n(num_n_block, params.b, params.h);  // For seqK-parallel main kernel
    
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            params.seqused_k == nullptr &&
                            (params.seqlen_q % Kernel_traits::kBlockM == 0) &&
                            (params.seqlen_k % Kernel_traits::kBlockN == 0);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    // ===========================================================================
    // ISSUE #84 FIX: Launch preprocessing kernel FIRST
    // ===========================================================================
    // This computes dP_sum = dot(dO, O) for ALL query positions and stores in
    // dsoftmax_sum buffer. Without this, the main backward kernel reads garbage
    // from dsoftmax_sum for most positions, causing gradient explosion.
    //
    // Clear_dQaccum=true means this kernel will also zero dq_accum, which is
    // the correct behavior for non-deterministic mode.
    // ===========================================================================
    {
        auto preprocess_kernel = &flash_bwd_dot_do_o_kernel</*Clear_dQaccum=*/true, Kernel_traits>;
        preprocess_kernel<<<grid_preprocess, Kernel_traits::kNThreads, 0, stream>>>(params);
        check_cuda(cudaGetLastError(), "flash_bwd_dot_do_o_kernel launch");
    }

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, HasAlibi, [&] {
                // Attention dropout: switch template based on p_dropout
                const bool use_dropout = params.p_dropout < 1.0f;
                BOOL_SWITCH(use_dropout, IsDropout, [&] {
                auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits,
                                                                            IsDropout,
                                                                            Is_causal,
                                                                            /*Is_local=*/false,
                                                                            HasAlibi,
                                                                            IsEvenMNConst && IsEvenKConst && !HasAlibi,
                                                                            IsEvenKConst && !HasAlibi,
                                                                            /*Is_softcap=*/false>;
                if (smem_size >= 48 * 1024) {
                    check_cuda(cudaFuncSetAttribute(kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    smem_size),
                               "cudaFuncSetAttribute(bwd seqk)");
                }
                kernel<<<grid_n, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                check_cuda(cudaGetLastError(), "flash_bwd_seqk_kernel launch");
                });
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024) {
        check_cuda(cudaFuncSetAttribute(kernel_dq,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Kernel_traits::kSmemdQSize),
                   "cudaFuncSetAttribute(bwd convert_dq)");
    }
    kernel_dq<<<grid_preprocess, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, 1);
    check_cuda(cudaGetLastError(), "flash_bwd_convert_dq_kernel launch");
}

// Split-KV forward launcher, fixed to num_splits==1 (Split=false).
//
// Mirrors the vendored run_flash_splitkv_fwd template-arg selection but drops
// the >1 split branch (no combine kernel / no oaccum workspace) and the
// disabled feature dims (local attention, softcap). Append_KV is switched on
// the presence of knew_ptr so the same launcher serves prefill (empty cache)
// and decode (incremental append) — both with fused rotary when rotary_dim>0.
template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_no_split(Flash_fwd_params& params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    // num_splits == 1: grid is (m_blocks, batch, heads).
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            params.seqused_k == nullptr &&
                            (params.seqlen_k % Kernel_traits::kBlockN == 0) &&
                            (params.seqlen_q % Kernel_traits::kBlockM == 0);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, HasAlibi, [&] {
                    auto kernel = &flash_fwd_splitkv_kernel<
                        Kernel_traits,
                        Is_causal,
                        /*Is_local=*/false,
                        HasAlibi,
                        IsEvenMNConst && !Append_KV && IsEvenKConst && !HasAlibi && Kernel_traits::kHeadDim <= 128,
                        IsEvenKConst && !HasAlibi,
                        /*Is_softcap=*/false,
                        /*Split=*/false,
                        Append_KV>;
                    if (smem_size >= 48 * 1024) {
                        check_cuda(cudaFuncSetAttribute(kernel,
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                        smem_size),
                                   "cudaFuncSetAttribute(splitkv)");
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    check_cuda(cudaGetLastError(), "flash_fwd_splitkv_kernel launch");
                });
            });
        });
    });
}

template<typename T, int Headdim, bool Is_causal>
void run_kvcache_rotary_dispatch(Flash_fwd_params& params, cudaStream_t stream) {
    // Same block-size heuristic as the vendored run_mha_fwd_splitkv_dispatch.
    constexpr int kBlockM = 64;
    constexpr int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    run_flash_splitkv_fwd_no_split<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_flash_attn_fwd_hdim32(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 32;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_flash_attn_fwd_hdim64(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 64;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_flash_attn_bwd_hdim32(Flash_bwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 32;
    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_flash_attn_bwd_hdim64(Flash_bwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 64;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    int max_smem = 0;
    check_cuda(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
               "cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");

    if (max_smem >= 144 * 1024) {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_causal>(params, stream);
    } else {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_causal>(params, stream);
    }
}

void init_fwd_params_contiguous(Flash_fwd_params& params,
                                const void* q, const void* k, const void* v,
                                void* out, void* softmax_lse,
                                const float* alibi_slopes,
                                int batch, int seqlen, int n_heads, int n_kv_heads, int head_dim,
                                float softmax_scale, bool is_bf16, bool is_causal,
                                float attention_dropout_p, uint64_t dropout_seed,
                                const int* sequence_lengths) {
    params = {};
    params.is_bf16 = is_bf16;
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k);
    params.v_ptr = const_cast<void*>(v);
    params.o_ptr = out;
    params.softmax_lse_ptr = softmax_lse;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.q_row_stride = n_heads * head_dim;
    params.k_row_stride = n_kv_heads * head_dim;
    params.v_row_stride = n_kv_heads * head_dim;
    params.o_row_stride = n_heads * head_dim;

    params.q_batch_stride = seqlen * params.q_row_stride;
    params.k_batch_stride = seqlen * params.k_row_stride;
    params.v_batch_stride = seqlen * params.v_row_stride;
    params.o_batch_stride = seqlen * params.o_row_stride;

    params.b = batch;
    params.h = n_heads;
    params.h_k = n_kv_heads;
    params.h_h_k_ratio = n_heads / n_kv_heads;
    params.seqlen_q = seqlen;
    params.seqlen_k = seqlen;
    params.seqlen_knew = 0;
    params.d = head_dim;
    params.seqlen_q_rounded = round_multiple(seqlen, 128);
    params.seqlen_k_rounded = round_multiple(seqlen, 128);
    params.d_rounded = round_multiple(head_dim, head_dim <= 128 ? 32 : 64);
    params.rotary_dim = 0;
    params.total_q = batch * seqlen;

    const float scale = require_softmax_scale(softmax_scale, head_dim, "init_fwd_params_contiguous");
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * kLog2e;
    params.softcap = 0.0f;

    params.p_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = const_cast<int*>(sequence_lengths);
    params.blockmask = nullptr;

    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;

    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;

    // Dropout setup: p_dropout in the struct is KEEP probability (1.0 = no dropout)
    // attention_dropout_p is DROP probability from config (e.g., 0.15 = drop 15%)
    if (attention_dropout_p > 0.0f && attention_dropout_p < 1.0f) {
        const float keep_prob = 1.0f - attention_dropout_p;
        params.p_dropout = keep_prob;
        params.p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(keep_prob * 255.0f));
        params.rp_dropout = 1.0f / keep_prob;
        params.scale_softmax_rp_dropout = scale * params.rp_dropout;
        params.philox_args = {dropout_seed, 0ull};
    } else {
        // No dropout (p=0 or invalid)
        params.p_dropout = 1.0f;
        params.p_dropout_in_uint8_t = 255;
        params.rp_dropout = 1.0f;
        params.scale_softmax_rp_dropout = scale;
        params.philox_args = {0ull, 0ull};
    }
    params.rng_state = nullptr;

    params.window_size_left = is_causal ? -1 : -1;
    params.window_size_right = is_causal ? 0 : -1;
    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = false;
    params.is_rotary_interleaved = false;

    params.num_splits = 1;
    params.alibi_slopes_ptr = const_cast<float*>(alibi_slopes);
    params.alibi_slopes_batch_stride = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
}

// KV-cache variant: Q has seqlen_q tokens, K/V cache has seqlen_k tokens.
// No dropout (inference only). No knew/vnew (caller pre-appends to cache).
void init_fwd_params_kvcache(Flash_fwd_params& params,
                             const void* q, const void* k_cache, const void* v_cache,
                             void* out, void* softmax_lse,
                             const float* alibi_slopes,
                             int batch, int seqlen_q, int seqlen_k,
                             int n_heads, int n_kv_heads, int head_dim,
                             float softmax_scale,
                             bool is_bf16, bool is_causal) {
    params = {};
    params.is_bf16 = is_bf16;
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k_cache);
    params.v_ptr = const_cast<void*>(v_cache);
    params.o_ptr = out;
    params.softmax_lse_ptr = softmax_lse;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.q_row_stride = n_heads * head_dim;
    params.k_row_stride = n_kv_heads * head_dim;
    params.v_row_stride = n_kv_heads * head_dim;
    params.o_row_stride = n_heads * head_dim;

    // Q uses seqlen_q, K/V use seqlen_k for batch strides
    params.q_batch_stride = seqlen_q * params.q_row_stride;
    params.k_batch_stride = seqlen_k * params.k_row_stride;
    params.v_batch_stride = seqlen_k * params.v_row_stride;
    params.o_batch_stride = seqlen_q * params.o_row_stride;

    params.b = batch;
    params.h = n_heads;
    params.h_k = n_kv_heads;
    params.h_h_k_ratio = n_heads / n_kv_heads;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_knew = 0;
    params.d = head_dim;
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(seqlen_k, 128);
    params.d_rounded = round_multiple(head_dim, head_dim <= 128 ? 32 : 64);
    params.rotary_dim = 0;
    params.total_q = batch * seqlen_q;

    const float scale = require_softmax_scale(softmax_scale, head_dim, "init_fwd_params_kvcache");
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * kLog2e;
    params.softcap = 0.0f;

    params.p_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;

    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;

    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;

    // No dropout for inference
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;
    params.philox_args = {0ull, 0ull};
    params.rng_state = nullptr;

    params.window_size_left = is_causal ? -1 : -1;
    params.window_size_right = is_causal ? 0 : -1;
    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = false;
    params.is_rotary_interleaved = false;

    params.num_splits = 1;
    params.alibi_slopes_ptr = const_cast<float*>(alibi_slopes);
    params.alibi_slopes_batch_stride = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
}

// KV-cache + fused-rotary variant (inference decode/prefill via split-KV kernel).
//
// Contract (upstream FA2 mha_fwd_kvcache, non-paged, non-cumulative cu_seqlens_k):
//   q     : [batch, seqlen_q, n_heads,    head_dim]  new query tokens, UNROTATED
//   knew  : [batch, seqlen_q, n_kv_heads, head_dim]  new key tokens,   UNROTATED
//   vnew  : [batch, seqlen_q, n_kv_heads, head_dim]  new value tokens
//   k/v_cache : [batch, cache_max_seq, n_kv_heads, head_dim]  capacity buffers;
//               positions [0, fill) already hold rotated K / raw V from prior steps.
//   cache_seqlens : device int[batch] = per-batch current fill level (seqlen_k_cache).
//   rotary_cos/sin: [cache_max_seq, rotary_dim/2] in element dtype (see PBM builder).
//
// The kernel rotates Q at absolute position = fill (+causal offset), rotates and
// appends knew into k_cache at [fill, fill+seqlen_q), copies vnew into v_cache,
// then attends Q over the whole cache. params.seqlen_k is the CAPACITY because
// n_block_max is clamped by ceil(seqlen_k/kBlockN); the fill comes via cu_seqlens_k.
void init_fwd_params_kvcache_rotary(Flash_fwd_params& params,
                                    const void* q, const void* knew, const void* vnew,
                                    void* k_cache, void* v_cache,
                                    void* out, void* softmax_lse,
                                    const void* rotary_cos, const void* rotary_sin,
                                    const int* cache_seqlens,
                                    const float* alibi_slopes,
                                    int batch, int seqlen_q, int cache_max_seq,
                                    int n_heads, int n_kv_heads, int head_dim, int rotary_dim,
                                    float softmax_scale, bool is_bf16, bool is_causal) {
    params = {};
    params.is_bf16 = is_bf16;
    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = k_cache;
    params.v_ptr = v_cache;
    params.o_ptr = out;
    params.softmax_lse_ptr = softmax_lse;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;
    params.o_head_stride = head_dim;

    params.q_row_stride = n_heads * head_dim;
    params.k_row_stride = n_kv_heads * head_dim;
    params.v_row_stride = n_kv_heads * head_dim;
    params.o_row_stride = n_heads * head_dim;

    // Q/O batch stride spans seqlen_q; K/V cache batch stride spans the full
    // capacity (cache_max_seq), since the cache buffer holds every position.
    params.q_batch_stride = static_cast<int64_t>(seqlen_q) * params.q_row_stride;
    params.k_batch_stride = static_cast<int64_t>(cache_max_seq) * params.k_row_stride;
    params.v_batch_stride = static_cast<int64_t>(cache_max_seq) * params.v_row_stride;
    params.o_batch_stride = static_cast<int64_t>(seqlen_q) * params.o_row_stride;

    params.b = batch;
    params.h = n_heads;
    params.h_k = n_kv_heads;
    params.h_h_k_ratio = n_heads / n_kv_heads;
    params.seqlen_q = seqlen_q;
    // CAPACITY, not fill (see header note); fill arrives via cu_seqlens_k.
    params.seqlen_k = cache_max_seq;
    params.seqlen_knew = seqlen_q;
    params.d = head_dim;
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(cache_max_seq, 128);
    params.d_rounded = round_multiple(head_dim, head_dim <= 128 ? 32 : 64);
    params.rotary_dim = rotary_dim;
    params.total_q = batch * seqlen_q;

    const float scale = require_softmax_scale(softmax_scale, head_dim, "init_fwd_params_kvcache_rotary");
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * kLog2e;
    params.softcap = 0.0f;

    params.p_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.cu_seqlens_q = nullptr;
    // Non-cumulative: cu_seqlens_k[b] stores the per-batch cache fill level.
    params.cu_seqlens_k = const_cast<int*>(cache_seqlens);
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;

    // New tokens the kernel appends into the cache (rotary applied to K-new).
    params.knew_ptr = const_cast<void*>(knew);
    params.vnew_ptr = const_cast<void*>(vnew);
    params.knew_batch_stride = static_cast<int64_t>(seqlen_q) * params.k_row_stride;
    params.vnew_batch_stride = static_cast<int64_t>(seqlen_q) * params.v_row_stride;
    params.knew_row_stride = params.k_row_stride;
    params.vnew_row_stride = params.v_row_stride;
    params.knew_head_stride = head_dim;
    params.vnew_head_stride = head_dim;

    params.rotary_cos_ptr = const_cast<void*>(rotary_cos);
    params.rotary_sin_ptr = const_cast<void*>(rotary_sin);
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;

    // Inference: no dropout.
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;
    params.philox_args = {0ull, 0ull};
    params.rng_state = nullptr;

    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;
    params.is_causal = is_causal;
    // cu_seqlens_k stores absolute fill lengths, not cumulative offsets.
    params.is_seqlens_k_cumulative = false;
    // GRIM training RoPE is GPT-J interleaved (pairs 2i, 2i+1) — match it so the
    // fused decode rotation is numerically identical to the trained model.
    params.is_rotary_interleaved = true;

    params.num_splits = 1;
    params.alibi_slopes_ptr = const_cast<float*>(alibi_slopes);
    params.alibi_slopes_batch_stride = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
}

void init_bwd_params_contiguous(Flash_bwd_params& params,
                                const void* q, const void* k, const void* v,
                                const void* out, const void* dout,
                                const void* softmax_lse,
                                const float* alibi_slopes,
                                void* dq, void* dk, void* dv,
                                void* dq_accum, void* dsoftmax_sum,
                                int batch, int seqlen, int n_heads, int n_kv_heads, int head_dim,
                                float softmax_scale, bool is_bf16, bool is_causal,
                                float attention_dropout_p, uint64_t dropout_seed,
                                const int* sequence_lengths) {
    init_fwd_params_contiguous(params, q, k, v,
                               const_cast<void*>(out), const_cast<void*>(softmax_lse),
                               alibi_slopes,
                               batch, seqlen, n_heads, n_kv_heads, head_dim, softmax_scale, is_bf16, is_causal,
                               attention_dropout_p, dropout_seed, sequence_lengths);
    params.do_ptr = const_cast<void*>(dout);
    params.dq_ptr = dq;
    params.dk_ptr = dk;
    params.dv_ptr = dv;

    params.do_head_stride = head_dim;
    params.dq_head_stride = head_dim;
    params.dk_head_stride = head_dim;
    params.dv_head_stride = head_dim;

    params.do_row_stride = n_heads * head_dim;
    params.dq_row_stride = n_heads * head_dim;
    // ISSUE #85 FIX: dK/dV buffers are allocated for num_heads (not num_kv_heads) so the
    // kernel can write per-query-head contributions at bidh * dk_head_stride without overlap.
    // Row stride must match the allocation layout; using n_kv_heads caused head writes to
    // alias across sequence positions, silently corrupting dK/dV gradients.
    params.dk_row_stride = n_heads * head_dim;
    params.dv_row_stride = n_heads * head_dim;

    params.do_batch_stride = seqlen * params.do_row_stride;
    params.dq_batch_stride = seqlen * params.dq_row_stride;
    params.dk_batch_stride = seqlen * params.dk_row_stride;
    params.dv_batch_stride = seqlen * params.dv_row_stride;

    params.dq_accum_ptr = dq_accum;
    // Non-split kernels only: dk/dv accum buffers are unused and must stay null.
    // If we enable split-K or change kernel traits, this must be revisited.
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;
    params.dsoftmax_sum = dsoftmax_sum;

    params.deterministic = false;
    params.dq_accum_split_stride = 0;
}

}  // namespace detail

}  // namespace grim_flash

extern "C" size_t flash_attn_bwd_workspace_size_bytes(int batch, int seqlen, int n_heads, int head_dim) {
    return grim_flash::detail::dq_accum_bytes(batch, seqlen, n_heads, head_dim) +
           grim_flash::detail::dsoftmax_sum_bytes(batch, seqlen, n_heads);
}

extern "C" size_t flash_attn_dq_accum_bytes(int batch, int seqlen, int n_heads, int head_dim) {
    return grim_flash::detail::dq_accum_bytes(batch, seqlen, n_heads, head_dim);
}

extern "C" size_t flash_attn_dsoftmax_sum_bytes(int batch, int seqlen, int n_heads) {
    return grim_flash::detail::dsoftmax_sum_bytes(batch, seqlen, n_heads);
}

extern "C" void flash_attn_fwd_ex(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    void* softmax_lse,
    const float* alibi_slopes,
    int batch,
    int seqlen,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float softmax_scale,
    bool causal,
    bool is_bf16,
    float attention_dropout_p,
    uint64_t dropout_seed,
    cudaStream_t stream,
    const int* sequence_lengths) {
    if (!q || !k || !v || !out || !softmax_lse) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd received null pointer input");
        throw std::runtime_error("flash_attn_fwd: null pointer input");
    }
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
    if (head_dim != 64) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd head_dim must be 64 (GRIM_FLASHATTN_HDIM64_ONLY)");
        throw std::runtime_error("flash_attn_fwd: head_dim must be 64 (GRIM_FLASHATTN_HDIM64_ONLY)");
    }
#else
    if (head_dim != 32 && head_dim != 64) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd head_dim must be 32 or 64 for FA2 kernels");
        throw std::runtime_error("flash_attn_fwd: head_dim must be 32 or 64 for FA2 kernels");
    }
#endif
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd invalid head configuration");
        throw std::runtime_error("flash_attn_fwd: invalid head configuration");
    }
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
    if (!causal) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd non-causal disabled (GRIM_FLASHATTN_CAUSAL_ONLY)");
        throw std::runtime_error("flash_attn_fwd: non-causal disabled (GRIM_FLASHATTN_CAUSAL_ONLY)");
    }
#endif
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    if (!is_bf16) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd FP16 disabled (GRIM_FLASHATTN_BF16_ONLY)");
        throw std::runtime_error("flash_attn_fwd: FP16 disabled (GRIM_FLASHATTN_BF16_ONLY)");
    }
#endif

    grim_flash::Flash_fwd_params params;
    grim_flash::detail::init_fwd_params_contiguous(params, q, k, v, out, softmax_lse,
                                                   alibi_slopes,
                                                   batch, seqlen, n_heads, n_kv_heads, head_dim,
                                                   softmax_scale, is_bf16, causal,
                                                   attention_dropout_p, dropout_seed,
                                                   sequence_lengths);

    // FIX: Allocate rng_state when dropout is enabled.
    // The Dao FA kernel writes seed/offset to rng_state[0]/[1] when Is_dropout=true (flash_fwd_kernel.h:76-77).
    // Without this, params.rng_state=nullptr causes illegal memory access (CUDA error 700).
    uint64_t* rng_state_buf = nullptr;
    if (attention_dropout_p > 0.0f && attention_dropout_p < 1.0f) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&rng_state_buf), 2 * sizeof(uint64_t), "flash_attn_fwd_rng_state");
        params.rng_state = rng_state_buf;
    }

    GRIM::FlashAttentionDiagnostics::emitForwardPreKernelDiagnostics({
        q,
        k,
        v,
        out,
        softmax_lse,
        alibi_slopes,
        batch,
        seqlen,
        n_heads,
        n_kv_heads,
        head_dim,
        params.scale_softmax,
        is_bf16,
        stream,
    });

#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
    if (is_bf16) {
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
    } else {
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
    }
#endif
#else
    if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, false>(params, stream);
        }
#endif
    }
#endif
#else
    if (head_dim == 32) {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    }
#endif

    // CRITICAL: Check for async CUDA errors from the FlashAttention kernel
    {
        cudaError_t kernel_err = cudaStreamSynchronize(stream);
        if (rng_state_buf) { cudaFree(rng_state_buf); rng_state_buf = nullptr; }
        if (kernel_err != cudaSuccess) {
            fprintf(stderr, "[FA-FWD-KERNEL-ERROR] CUDA error after flash_fwd_kernel: %s (%d)\n",
                    cudaGetErrorString(kernel_err), static_cast<int>(kernel_err));
            fprintf(stderr, "  This means the FlashAttention kernel FAILED - output and LSE are INVALID!\n");
            fprintf(stderr, "  batch=%d seqlen=%d n_heads=%d n_kv_heads=%d head_dim=%d\n",
                    batch, seqlen, n_heads, n_kv_heads, head_dim);
            // Rule 20: Crash on kernel failure - don't silently produce garbage
            throw std::runtime_error("flash_attn_fwd: kernel execution failed: " + std::string(cudaGetErrorString(kernel_err)));
        }
    }

    GRIM::FlashAttentionDiagnostics::emitForwardPostKernelDiagnostics({
        q,
        k,
        v,
        out,
        softmax_lse,
        alibi_slopes,
        batch,
        seqlen,
        n_heads,
        n_kv_heads,
        head_dim,
        params.scale_softmax,
        is_bf16,
        stream,
    });
}

// ============================================================================
// KV-cache forward: inference-only, supports seqlen_q != seqlen_k.
// Caller is responsible for pre-appending new K/V to cache before calling.
// No dropout, no RNG state, no backward.
// ============================================================================
extern "C" void flash_attn_fwd_kvcache(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    void* softmax_lse,
    const float* alibi_slopes,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float softmax_scale,
    bool causal,
    bool is_bf16,
    cudaStream_t stream)
{
    // --- Validation (Rule 20: crash on bad input) ---
    if (!q) throw std::runtime_error("flash_attn_fwd_kvcache: q is NULL");
    if (!k_cache) throw std::runtime_error("flash_attn_fwd_kvcache: k_cache is NULL");
    if (!v_cache) throw std::runtime_error("flash_attn_fwd_kvcache: v_cache is NULL");
    if (!out) throw std::runtime_error("flash_attn_fwd_kvcache: out is NULL");
    if (!softmax_lse) throw std::runtime_error("flash_attn_fwd_kvcache: softmax_lse is NULL");
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
    if (head_dim != 64) {
        throw std::runtime_error("flash_attn_fwd_kvcache: head_dim must be 64 (GRIM_FLASHATTN_HDIM64_ONLY), got " + std::to_string(head_dim));
    }
#else
    if (head_dim != 32 && head_dim != 64) {
        throw std::runtime_error("flash_attn_fwd_kvcache: head_dim must be 32 or 64, got " + std::to_string(head_dim));
    }
#endif
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        throw std::runtime_error("flash_attn_fwd_kvcache: invalid head configuration n_heads=" + std::to_string(n_heads) +
                                 " n_kv_heads=" + std::to_string(n_kv_heads));
    }
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
    if (!causal) {
        throw std::runtime_error("flash_attn_fwd_kvcache: non-causal disabled (GRIM_FLASHATTN_CAUSAL_ONLY)");
    }
#endif
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    if (!is_bf16) {
        throw std::runtime_error("flash_attn_fwd_kvcache: FP16 disabled (GRIM_FLASHATTN_BF16_ONLY)");
    }
#endif
    if (seqlen_q > seqlen_k)
        throw std::runtime_error("flash_attn_fwd_kvcache: seqlen_q (" + std::to_string(seqlen_q) +
                                 ") > seqlen_k (" + std::to_string(seqlen_k) + ")");
    if (batch <= 0)
        throw std::runtime_error("flash_attn_fwd_kvcache: batch must be > 0, got " + std::to_string(batch));
    if (seqlen_q <= 0)
        throw std::runtime_error("flash_attn_fwd_kvcache: seqlen_q must be > 0, got " + std::to_string(seqlen_q));
    if (seqlen_k <= 0)
        throw std::runtime_error("flash_attn_fwd_kvcache: seqlen_k must be > 0, got " + std::to_string(seqlen_k));

    // --- Init params ---
    grim_flash::Flash_fwd_params params;
    grim_flash::detail::init_fwd_params_kvcache(
        params, q, k_cache, v_cache, out, softmax_lse, alibi_slopes,
        batch, seqlen_q, seqlen_k, n_heads, n_kv_heads, head_dim, softmax_scale,
        is_bf16, causal);

    // --- Kernel dispatch (same template dispatch as flash_attn_fwd_ex) ---
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
    if (is_bf16) {
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
    } else {
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
    }
#endif
#else
    if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, false>(params, stream);
        }
#endif
    }
#endif
#else
    if (head_dim == 32) {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY) && defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#elif defined(GRIM_FLASHATTN_CAUSAL_ONLY)
        if (is_bf16) grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
        else         grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::half_t, true>(params, stream);
#elif defined(GRIM_FLASHATTN_BF16_ONLY)
        if (causal) grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
        else        grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
#else
        if (causal) {
            if (is_bf16) grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
            else         grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::half_t, true>(params, stream);
        } else {
            if (is_bf16) grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
            else         grim_flash::detail::run_flash_attn_fwd_hdim32<cutlass::half_t, false>(params, stream);
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY) && defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#elif defined(GRIM_FLASHATTN_CAUSAL_ONLY)
        if (is_bf16) grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        else         grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
#elif defined(GRIM_FLASHATTN_BF16_ONLY)
        if (causal) grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        else        grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
        if (causal) {
            if (is_bf16) grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
            else         grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, true>(params, stream);
        } else {
            if (is_bf16) grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
            else         grim_flash::detail::run_flash_attn_fwd_hdim64<cutlass::half_t, false>(params, stream);
        }
#endif
    }
#endif

    // --- Error check (Rule 20: crash on failure) ---
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("flash_attn_fwd_kvcache: kernel failed: " +
                                 std::string(cudaGetErrorString(err)) +
                                 " (batch=" + std::to_string(batch) +
                                 " seqlen_q=" + std::to_string(seqlen_q) +
                                 " seqlen_k=" + std::to_string(seqlen_k) +
                                 " n_heads=" + std::to_string(n_heads) +
                                 " n_kv_heads=" + std::to_string(n_kv_heads) +
                                 " head_dim=" + std::to_string(head_dim) + ")");
    }
}

extern "C" void flash_attn_fwd_kvcache_rotary(
    const void* q,
    const void* knew,
    const void* vnew,
    void* k_cache,
    void* v_cache,
    void* out,
    void* softmax_lse,
    const void* rotary_cos,
    const void* rotary_sin,
    const int* cache_seqlens,
    const float* alibi_slopes,
    int batch,
    int seqlen_q,
    int cache_max_seq,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim,
    float softmax_scale,
    bool causal,
    bool is_bf16,
    cudaStream_t stream)
{
    // --- Validation (Rule 20: crash on bad input) ---
    if (!q) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: q is NULL");
    if (!knew) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: knew is NULL");
    if (!vnew) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: vnew is NULL");
    if (!k_cache) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: k_cache is NULL");
    if (!v_cache) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: v_cache is NULL");
    if (!out) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: out is NULL");
    if (!softmax_lse) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: softmax_lse is NULL");
    if (!rotary_cos) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: rotary_cos is NULL");
    if (!rotary_sin) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: rotary_sin is NULL");
    if (!cache_seqlens) throw std::runtime_error("flash_attn_fwd_kvcache_rotary: cache_seqlens is NULL");
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
    if (head_dim != 64) {
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: head_dim must be 64 (GRIM_FLASHATTN_HDIM64_ONLY), got " + std::to_string(head_dim));
    }
#else
    if (head_dim != 32 && head_dim != 64) {
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: head_dim must be 32 or 64, got " + std::to_string(head_dim));
    }
#endif
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: invalid head configuration n_heads=" + std::to_string(n_heads) +
                                 " n_kv_heads=" + std::to_string(n_kv_heads));
    }
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
    if (!causal) {
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: non-causal disabled (GRIM_FLASHATTN_CAUSAL_ONLY)");
    }
#endif
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    if (!is_bf16) {
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: FP16 disabled (GRIM_FLASHATTN_BF16_ONLY)");
    }
#endif
    if (batch <= 0)
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: batch must be > 0, got " + std::to_string(batch));
    if (seqlen_q <= 0)
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: seqlen_q must be > 0, got " + std::to_string(seqlen_q));
    if (cache_max_seq < seqlen_q)
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: cache_max_seq (" + std::to_string(cache_max_seq) +
                                 ") < seqlen_q (" + std::to_string(seqlen_q) + ")");
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim)
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: invalid rotary_dim=" + std::to_string(rotary_dim) +
                                 " for head_dim=" + std::to_string(head_dim));

    // --- Init params ---
    grim_flash::Flash_fwd_params params;
    grim_flash::detail::init_fwd_params_kvcache_rotary(
        params, q, knew, vnew, k_cache, v_cache, out, softmax_lse,
        rotary_cos, rotary_sin, cache_seqlens, alibi_slopes,
        batch, seqlen_q, cache_max_seq, n_heads, n_kv_heads, head_dim, rotary_dim,
        softmax_scale, is_bf16, causal);

    // --- Kernel dispatch (split-KV kernel, num_splits==1, fused rotary) ---
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
    grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::bfloat16_t, 64, true>(params, stream);
#else
    if (causal) {
        if (head_dim == 32) {
            if (is_bf16) grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::bfloat16_t, 32, true>(params, stream);
            else         grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::half_t, 32, true>(params, stream);
        } else {
            if (is_bf16) grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::bfloat16_t, 64, true>(params, stream);
            else         grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::half_t, 64, true>(params, stream);
        }
    } else {
        if (head_dim == 32) {
            if (is_bf16) grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::bfloat16_t, 32, false>(params, stream);
            else         grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::half_t, 32, false>(params, stream);
        } else {
            if (is_bf16) grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::bfloat16_t, 64, false>(params, stream);
            else         grim_flash::detail::run_kvcache_rotary_dispatch<cutlass::half_t, 64, false>(params, stream);
        }
    }
#endif

    // --- Error check (Rule 20: crash on failure) ---
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("flash_attn_fwd_kvcache_rotary: kernel failed: " +
                                 std::string(cudaGetErrorString(err)) +
                                 " (batch=" + std::to_string(batch) +
                                 " seqlen_q=" + std::to_string(seqlen_q) +
                                 " cache_max_seq=" + std::to_string(cache_max_seq) +
                                 " n_heads=" + std::to_string(n_heads) +
                                 " n_kv_heads=" + std::to_string(n_kv_heads) +
                                 " head_dim=" + std::to_string(head_dim) +
                                 " rotary_dim=" + std::to_string(rotary_dim) + ")");
    }
}

extern "C" void flash_attn_bwd_ex(
    const void* q,
    const void* k,
    const void* v,
    const void* out,
    const void* dout,
    const void* softmax_lse, // must be from the same forward as forward
    const float* alibi_slopes,
    void* dq,
    void* dk,
    void* dv,
    void* dq_accum,
    void* dsoftmax_sum,
    int batch,
    int seqlen,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float softmax_scale,
    bool causal,
    bool is_bf16,
    float attention_dropout_p,
    uint64_t dropout_seed,
    cudaStream_t stream,
    const int* sequence_lengths) {
    if (!q || !k || !v || !out || !dout || !softmax_lse || !dq || !dk || !dv) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd received null pointer input");
        throw std::runtime_error("flash_attn_bwd: null pointer input");
    }
    else {
        char input_msg[512];
        snprintf(input_msg, sizeof(input_msg),
                 "[FlashAttention] flash_attn_bwd_ex called with q=%p, k=%p, v=%p, out=%p, dout=%p, "
                 "softmax_lse=%p, dq=%p, dk=%p, dv=%p (batch=%d, seqlen=%d, n_heads=%d, n_kv_heads=%d, "
                 "head_dim=%d, softmax_scale=%f, causal=%s, is_bf16=%s)",
                 q, k, v, out, dout, softmax_lse, dq, dk, dv,
                 batch, seqlen, n_heads, n_kv_heads, head_dim,
                 grim_flash::detail::require_softmax_scale(softmax_scale, head_dim, "flash_attn_bwd_ex"),
                 causal ? "true" : "false", is_bf16 ? "true" : "false");
        if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
            FlashAttentionLog::info(input_msg);
        }
    }
    if (!dq_accum || !dsoftmax_sum) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd dq_accum and dsoftmax_sum workspace required");
        throw std::runtime_error("flash_attn_bwd: dq_accum and dsoftmax_sum workspace required");
    }
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
    if (head_dim != 64) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd head_dim must be 64 (GRIM_FLASHATTN_HDIM64_ONLY)");
        throw std::runtime_error("flash_attn_bwd: head_dim must be 64 (GRIM_FLASHATTN_HDIM64_ONLY)");
    }
#else
    if (head_dim != 32 && head_dim != 64) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd head_dim must be 32 or 64 for FA2 kernels");
        throw std::runtime_error("flash_attn_bwd: head_dim must be 32 or 64 for FA2 kernels");
    }
#endif
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd invalid head configuration");
        throw std::runtime_error("flash_attn_bwd: invalid head configuration");
    }
    if (!grim_flash::detail::validate_bwd_workspace_pointers(dq_accum, dsoftmax_sum,
                                                             q, k, v, out, dout, dq, dk, dv, softmax_lse)) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd workspace pointer aliasing detected");
        throw std::runtime_error("flash_attn_bwd: workspace pointer aliasing detected");
    }
    if (!grim_flash::detail::is_aligned(dq_accum, 16) || !grim_flash::detail::is_aligned(dsoftmax_sum, 16)) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd workspace pointers must be >=16-byte aligned");
        throw std::runtime_error("flash_attn_bwd: workspace pointers must be >=16-byte aligned");
    }
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
    if (!causal) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd non-causal disabled (GRIM_FLASHATTN_CAUSAL_ONLY)");
        throw std::runtime_error("flash_attn_bwd: non-causal disabled (GRIM_FLASHATTN_CAUSAL_ONLY)");
    }
#endif
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    if (!is_bf16) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd FP16 disabled (GRIM_FLASHATTN_BF16_ONLY)");
        throw std::runtime_error("flash_attn_bwd: FP16 disabled (GRIM_FLASHATTN_BF16_ONLY)");
    }
#endif

    grim_flash::Flash_bwd_params params;
    grim_flash::detail::init_bwd_params_contiguous(params, q, k, v, out, dout, softmax_lse,
                                                   alibi_slopes,
                                                   dq, dk, dv,
                                                   dq_accum, dsoftmax_sum,
                                                   batch, seqlen, n_heads, n_kv_heads, head_dim,
                                                   softmax_scale, is_bf16, causal,
                                                   attention_dropout_p, dropout_seed,
                                                   sequence_lengths);

    // FIX: Allocate rng_state when dropout is enabled.
    // The Dao FA backward kernel reads rng_state[0]/[1] to reproduce the dropout mask (flash_bwd_kernel.h:446).
    // Pre-fill with the same seed/offset that the forward kernel wrote.
    uint64_t* rng_state_buf_bwd = nullptr;
    if (attention_dropout_p > 0.0f && attention_dropout_p < 1.0f) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&rng_state_buf_bwd), 2 * sizeof(uint64_t), "flash_attn_bwd_rng_state");
        uint64_t host_rng[2] = {dropout_seed, 0ull};
        cudaMemcpyAsync(rng_state_buf_bwd, host_rng, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        params.rng_state = rng_state_buf_bwd;
    }

    {
        char shape_msg[256];
        snprintf(shape_msg, sizeof(shape_msg),
                 "[FlashAttention] bwd rounded: seqlen_q_rounded=%d seqlen_k_rounded=%d d_rounded=%d total_q=%d unpadded_lse=%s",
                 params.seqlen_q_rounded,
                 params.seqlen_k_rounded,
                 params.d_rounded,
                 params.total_q,
                 params.unpadded_lse ? "true" : "false");
        if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
            FlashAttentionLog::info(shape_msg);
        }
    }
    if (params.dk_accum_ptr != nullptr || params.dv_accum_ptr != nullptr) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd dk_accum_ptr/dv_accum_ptr must be null for non-split kernels");
        throw std::runtime_error("flash_attn_bwd: dk_accum_ptr/dv_accum_ptr must be null for non-split kernels");
    }

    const size_t dq_bytes = grim_flash::detail::dq_accum_bytes(batch, seqlen, n_heads, head_dim);
    const size_t dsoftmax_bytes = grim_flash::detail::dsoftmax_sum_bytes(batch, seqlen, n_heads);
    {
        char ws_msg[192];
        snprintf(ws_msg, sizeof(ws_msg),
                 "[FlashAttention] bwd workspace bytes: dq_accum=%zu dsoftmax_sum=%zu",
                 dq_bytes, dsoftmax_bytes);
        if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
            FlashAttentionLog::info(ws_msg);
        }
    }
    // dq_accum zeroing is handled by the preprocessing kernel (Clear_dQaccum=true in
    // flash_bwd_dot_do_o_kernel) and the main kernel's Is_first path. No need to memset here.

    const GRIM::FlashAttentionDiagnostics::BackwardStrideLayout diagnostic_strides{
        { static_cast<long long>(params.q_batch_stride),  static_cast<long long>(params.q_row_stride),  static_cast<long long>(params.q_head_stride)  },
        { static_cast<long long>(params.k_batch_stride),  static_cast<long long>(params.k_row_stride),  static_cast<long long>(params.k_head_stride)  },
        { static_cast<long long>(params.v_batch_stride),  static_cast<long long>(params.v_row_stride),  static_cast<long long>(params.v_head_stride)  },
        { static_cast<long long>(params.o_batch_stride),  static_cast<long long>(params.o_row_stride),  static_cast<long long>(params.o_head_stride)  },
        { static_cast<long long>(params.do_batch_stride), static_cast<long long>(params.do_row_stride), static_cast<long long>(params.do_head_stride) },
        { static_cast<long long>(params.dq_batch_stride), static_cast<long long>(params.dq_row_stride), static_cast<long long>(params.dq_head_stride) },
        { static_cast<long long>(params.dk_batch_stride), static_cast<long long>(params.dk_row_stride), static_cast<long long>(params.dk_head_stride) },
        { static_cast<long long>(params.dv_batch_stride), static_cast<long long>(params.dv_row_stride), static_cast<long long>(params.dv_head_stride) },
    };

    GRIM::FlashAttentionDiagnostics::emitBackwardPreKernelDiagnostics({
        dout,
        softmax_lse,
        alibi_slopes,
        dq,
        dk,
        dv,
        diagnostic_strides,
        batch,
        seqlen,
        n_heads,
        n_kv_heads,
        head_dim,
        is_bf16,
        stream,
    });

    if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
        FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex launching kernels");
    }
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
    if (is_bf16) {
        grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
    } else {
        grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::half_t, true>(params, stream);
    }
#endif
#else
    if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::half_t, false>(params, stream);
        }
#endif
    }
#endif
#else
    if (head_dim == 32) {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_bwd_hdim32<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_flash_attn_bwd_hdim64<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    }
#endif
    if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
        FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex kernel launch complete, checking CUDA status");
    }
    grim_flash::detail::check_cuda(cudaGetLastError(), "flash_attn_bwd_ex launch");
    if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
        FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex synchronizing stream for error check");
    }
    grim_flash::detail::check_cuda(cudaStreamSynchronize(stream), "flash_attn_bwd_ex sync");
    if (rng_state_buf_bwd) { cudaFree(rng_state_buf_bwd); rng_state_buf_bwd = nullptr; }
    if constexpr (GRIM::VerboseLogging::ENABLE_BACKWARD_FLASH_ATTN_LOGS) {
        FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex stream synchronized");
    }

    GRIM::FlashAttentionDiagnostics::emitBackwardPostKernelDiagnostics({
        dout,
        softmax_lse,
        alibi_slopes,
        dq,
        dk,
        dv,
        diagnostic_strides,
        batch,
        seqlen,
        n_heads,
        n_kv_heads,
        head_dim,
        is_bf16,
        stream,
    });
}
