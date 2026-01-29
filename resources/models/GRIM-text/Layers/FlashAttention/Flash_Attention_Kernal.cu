// Flash_Attention_Kernal.cu
// FlashAttention v2 forward/backward kernels (SM86, BF16/FP16, head_dim 32/64).
//
// Portions adapted from https://github.com/Dao-AILab/flash-attention
// Copyright (c) 2022-2024, Tri Dao and contributors.
// Licensed under BSD 3-Clause (see external/flash-attention/LICENSE).

#include <cuda_runtime.h>
#include <cuda_bf16.h>  // ISSUE #79: For __nv_bfloat16 in diagnostic logging

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../training/module_logger.hpp"

// Module logger for Flash Attention diagnostics
namespace {
using FlashAttentionLog = ModuleLogger<GRIM::Logging::ModuleId::Activations>;
}

#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_UNEVEN_K

#define FLASH_NAMESPACE grim_flash
#include "../../../../../external/flash-attention/csrc/flash_attn/src/namespace_config.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/static_switch.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/hardware_info.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/kernel_traits.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/block_info.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/utils.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/softmax.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/mask.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/dropout.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/flash_bwd_preprocess_kernel.h"
#include "../../../../../external/flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h"

namespace grim_flash {

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

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN,
         bool Is_even_K, bool Is_softcap, bool Return_softmax>
__global__ void flash_fwd_kernel(const Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    grim_flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                             Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
#else
    if (threadIdx.x == 0) {
        printf("FATAL: FlashAttention requires SM80+.\n");
    }
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_M, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_kernel(const Flash_bwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    grim_flash::compute_dq_dk_dv<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K>(params);
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
// The main backward kernel (flash_bwd_dq_dk_dv_loop_kernel) uses the Is_first
// template parameter to conditionally call dot_do_o() - but Is_first is only
// true for the FIRST column block. The loop then reads from gdPsum for ALL
// m_blocks, expecting valid pre-computed dP_sum values.
//
// ROOT CAUSE: GRIM was missing this preprocessing kernel, so gdPsum contained
// uninitialized memory (cudaMalloc doesn't zero). This caused:
// - dQ/dK to explode by 100,000-500,000x despite dO being near-zero
// - Layers 9-0 showed exploding gradients while layers 11-10 were correct
//   (first 2 layers' starting m_blocks got valid data from inline dot_do_o)
// ============================================================================
template<bool Clear_dQaccum, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    grim_flash::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
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
                            (params.seqlen_k % Kernel_traits::kBlockN == 0) &&
                            (params.seqlen_q % Kernel_traits::kBlockM == 0);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, HasAlibi, [&] {
                auto kernel = &flash_fwd_kernel<Kernel_traits,
                                                /*Is_dropout=*/false,
                                                Is_causal,
                                                /*Is_local=*/false,
                                                HasAlibi,
                                                IsEvenMNConst && IsEvenKConst,
                                                IsEvenKConst,
                                                /*Is_softcap=*/false,
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
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(Flash_bwd_params& params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize1colblock;
    
    // ===========================================================================
    // ISSUE #84 FIX: Grid dimensions for preprocessing vs main kernel
    // ===========================================================================
    // Preprocessing kernel: one block per query tile (num_m_block, batch, heads)
    // Main kernel: one block per (batch, heads) - loops over all m_blocks internally
    // ===========================================================================
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_preprocess(num_m_block, params.b, params.h);  // For preprocessing
    dim3 grid(params.b, params.h, 1);  // For main backward kernel
    
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
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
                auto kernel = &flash_bwd_dq_dk_dv_loop_kernel<Kernel_traits,
                                                              /*Is_dropout=*/false,
                                                              Is_causal,
                                                              HasAlibi,
                                                              IsEvenMNConst,
                                                              IsEvenKConst>;
                if (smem_size >= 48 * 1024) {
                    check_cuda(cudaFuncSetAttribute(kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    smem_size),
                               "cudaFuncSetAttribute(bwd)");
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                check_cuda(cudaGetLastError(), "flash_bwd_kernel launch");
            });
        });
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 32;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 64;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim32(Flash_bwd_params& params, cudaStream_t stream) {
    constexpr int Headdim = 32;
    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_bwd_hdim64(Flash_bwd_params& params, cudaStream_t stream) {
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
                                bool is_bf16, bool is_causal) {
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

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
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

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;

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

    params.philox_args = {0ull, 0ull};
    params.rng_state = nullptr;
}

void init_bwd_params_contiguous(Flash_bwd_params& params,
                                const void* q, const void* k, const void* v,
                                const void* out, const void* dout,
                                const void* softmax_lse,
                                const float* alibi_slopes,
                                void* dq, void* dk, void* dv,
                                void* dq_accum, void* dsoftmax_sum,
                                int batch, int seqlen, int n_heads, int n_kv_heads, int head_dim,
                                bool is_bf16, bool is_causal) {
    init_fwd_params_contiguous(params, q, k, v,
                               const_cast<void*>(out), const_cast<void*>(softmax_lse),
                               alibi_slopes,
                               batch, seqlen, n_heads, n_kv_heads, head_dim, is_bf16, is_causal);
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
    params.dk_row_stride = n_kv_heads * head_dim;
    params.dv_row_stride = n_kv_heads * head_dim;

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
    bool causal,
    bool is_bf16,
    cudaStream_t stream) {
    if (!q || !k || !v || !out || !softmax_lse) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_fwd received null pointer input");
        throw std::runtime_error("flash_attn_fwd: null pointer input");
    }
    // CRITICAL: ALiBi slopes must be provided for GRIM hybrid PBM (ALiBi+RoPE)
    if (!alibi_slopes) {
        FlashAttentionLog::error("[FlashAttention] FATAL: alibi_slopes is NULL - GRIM requires ALiBi for hybrid PBM");
        throw std::runtime_error("flash_attn_fwd: alibi_slopes is NULL - GRIM requires ALiBi for hybrid PBM");
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
                                                   is_bf16, causal);

    // ISSUE #76: Log ALiBi slopes for this sequence - CRITICAL for max_seq_len boundary debugging
    static int s_fwd_call_count = 0;
    ++s_fwd_call_count;
    {
        std::vector<float> h_slopes(static_cast<size_t>(n_heads));
        cudaMemcpyAsync(h_slopes.data(), alibi_slopes, n_heads * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Log slopes range and all values if at or near max_seq_len
        float min_slope = h_slopes[0], max_slope = h_slopes[0];
        for (int h = 1; h < n_heads; ++h) {
            min_slope = std::min(min_slope, h_slopes[h]);
            max_slope = std::max(max_slope, h_slopes[h]);
        }
        
        // Always log sequence dimensions and slope range - CRITICAL for Issue #76
        fprintf(stderr, "[FA-FWD-ALIBI] call=%d seqlen=%d batch=%d n_heads=%d | slope_range=[%.6f, %.6f]",
                s_fwd_call_count, seqlen, batch, n_heads, min_slope, max_slope);
        
        // Log all slopes for sequences near max_seq_len (potential boundary issue)
        // Standard max_seq_len is 1024 or 2048, flag when within 10% of that
        const bool is_boundary_seq = (seqlen >= 920) || (seqlen >= 1840);  // 90% of 1024 or 2048
        if (is_boundary_seq) {
            fprintf(stderr, " [*** BOUNDARY_SEQ seqlen=%d ***] slopes=[", seqlen);
            for (int h = 0; h < n_heads; ++h) {
                fprintf(stderr, "%.6f%s", h_slopes[h], h < n_heads - 1 ? "," : "");
            }
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }

    // ISSUE #67: Log Flash Attention FORWARD INPUTS
    {
        const size_t q_elems = static_cast<size_t>(batch) * n_heads * seqlen * head_dim;
        const size_t kv_elems = static_cast<size_t>(batch) * n_kv_heads * seqlen * head_dim;
        const size_t sample_size = 100;
        std::vector<float> h_q(std::min(q_elems, sample_size));
        std::vector<float> h_k(std::min(kv_elems, sample_size));
        std::vector<float> h_v(std::min(kv_elems, sample_size));
        cudaMemcpyAsync(h_q.data(), q, h_q.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_k.data(), k, h_k.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_v.data(), v, h_v.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int nan_q = 0, inf_q = 0, nan_k = 0, inf_k = 0, nan_v = 0, inf_v = 0;
        for (float val : h_q) { if (std::isnan(val)) nan_q++; if (std::isinf(val)) inf_q++; }
        for (float val : h_k) { if (std::isnan(val)) nan_k++; if (std::isinf(val)) inf_k++; }
        for (float val : h_v) { if (std::isnan(val)) nan_v++; if (std::isinf(val)) inf_v++; }
        fprintf(stderr, "[FA-FWD-IN] Q: nan=%d inf=%d first=%.6f | K: nan=%d inf=%d first=%.6f | V: nan=%d inf=%d first=%.6f\n",
                nan_q, inf_q, h_q.empty() ? 0.0f : h_q[0],
                nan_k, inf_k, h_k.empty() ? 0.0f : h_k[0],
                nan_v, inf_v, h_v.empty() ? 0.0f : h_v[0]);
    }

#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
    if (is_bf16) {
        grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
    } else {
        grim_flash::detail::run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
    }
#endif
#else
    if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
        } else {
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
        }
#endif
    }
#endif
#else
    if (head_dim == 32) {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_mha_fwd_hdim32<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_fwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_mha_fwd_hdim32<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_fwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_fwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_mha_fwd_hdim32<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    }
#endif

    // ISSUE #67: Log Flash Attention FORWARD OUTPUT
    {
        const size_t out_elems = static_cast<size_t>(batch) * n_heads * seqlen * head_dim;
        const size_t sample_size = 100;
        std::vector<float> h_out(std::min(out_elems, sample_size));
        cudaMemcpyAsync(h_out.data(), out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int nan_out = 0, inf_out = 0;
        for (float val : h_out) { if (std::isnan(val)) nan_out++; if (std::isinf(val)) inf_out++; }
        fprintf(stderr, "[FA-FWD-OUT] output: nan=%d inf=%d first=%.6f\n",
                nan_out, inf_out, h_out.empty() ? 0.0f : h_out[0]);
    }
    
    // ISSUE #76: Log softmax_lse stats - CRITICAL for max_seq_len boundary debugging
    // softmax_lse has shape [batch, num_heads, seqlen] (FP32 dense)
    {
        const size_t lse_elems = static_cast<size_t>(batch) * n_heads * seqlen;
        std::vector<float> h_lse(lse_elems);
        cudaMemcpyAsync(h_lse.data(), softmax_lse, lse_elems * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Compute statistics per head to identify which head explodes first
        fprintf(stderr, "[FA-FWD-LSE] seqlen=%d batch=%d n_heads=%d total_elems=%zu\n", seqlen, batch, n_heads, lse_elems);
        
        // Overall stats
        float global_min = h_lse[0], global_max = h_lse[0];
        int nan_count = 0, inf_count = 0;
        double sum = 0.0;
        for (size_t i = 0; i < lse_elems; ++i) {
            float val = h_lse[i];
            if (std::isnan(val)) { nan_count++; continue; }
            if (std::isinf(val)) { inf_count++; continue; }
            if (val < global_min) global_min = val;
            if (val > global_max) global_max = val;
            sum += val;
        }
        float mean = lse_elems > 0 ? static_cast<float>(sum / lse_elems) : 0.0f;
        
        fprintf(stderr, "[FA-FWD-LSE-SUMMARY] nan=%d inf=%d range=[%.4f, %.4f] mean=%.4f\n",
                nan_count, inf_count, global_min, global_max, mean);
        
        // Per-head stats (only if at boundary or anomalous)
        const bool is_boundary_seq = (seqlen >= 920) || (seqlen >= 1840);
        const bool has_anomaly = (global_max > 50.0f) || (nan_count > 0) || (inf_count > 0);
        if (is_boundary_seq || has_anomaly) {
            fprintf(stderr, "[FA-FWD-LSE-PERHEAD] %s head_stats:\n", 
                    has_anomaly ? "*** ANOMALY ***" : "(boundary sequence)");
            for (int h = 0; h < n_heads; ++h) {
                // Head slice: [batch, h, seqlen] -> starts at h * seqlen for batch 0
                float head_min = std::numeric_limits<float>::max();
                float head_max = std::numeric_limits<float>::lowest();
                double head_sum = 0.0;
                int head_nan = 0, head_inf = 0;
                size_t head_count = 0;
                
                for (int b = 0; b < batch; ++b) {
                    for (int s = 0; s < seqlen; ++s) {
                        size_t idx = static_cast<size_t>(b) * n_heads * seqlen + h * seqlen + s;
                        float val = h_lse[idx];
                        if (std::isnan(val)) { head_nan++; continue; }
                        if (std::isinf(val)) { head_inf++; continue; }
                        if (val < head_min) head_min = val;
                        if (val > head_max) head_max = val;
                        head_sum += val;
                        head_count++;
                    }
                }
                float head_mean = head_count > 0 ? static_cast<float>(head_sum / head_count) : 0.0f;
                fprintf(stderr, "    head[%d]: range=[%.4f, %.4f] mean=%.4f nan=%d inf=%d\n",
                        h, head_min, head_max, head_mean, head_nan, head_inf);
            }
        }
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
    bool causal,
    bool is_bf16,
    cudaStream_t stream) {
    if (!q || !k || !v || !out || !dout || !softmax_lse || !dq || !dk || !dv) {
        FlashAttentionLog::error("[FlashAttention] FATAL: flash_attn_bwd received null pointer input");
        throw std::runtime_error("flash_attn_bwd: null pointer input");
    }
    else {
        char input_msg[512];
        snprintf(input_msg, sizeof(input_msg),
                 "[FlashAttention] flash_attn_bwd_ex called with q=%p, k=%p, v=%p, out=%p, dout=%p, "
                 "softmax_lse=%p, dq=%p, dk=%p, dv=%p (batch=%d, seqlen=%d, n_heads=%d, n_kv_heads=%d, "
                 "head_dim=%d, causal=%s, is_bf16=%s)",
                 q, k, v, out, dout, softmax_lse, dq, dk, dv,
                 batch, seqlen, n_heads, n_kv_heads, head_dim,
                 causal ? "true" : "false", is_bf16 ? "true" : "false");
        FlashAttentionLog::info(input_msg);
    }
    // CRITICAL: ALiBi slopes must be provided for GRIM hybrid PBM (ALiBi+RoPE)
    if (!alibi_slopes) {
        FlashAttentionLog::error("[FlashAttention] FATAL: alibi_slopes is NULL - GRIM requires ALiBi for hybrid PBM");
        throw std::runtime_error("flash_attn_bwd: alibi_slopes is NULL - GRIM requires ALiBi for hybrid PBM");
    }
    else {
        float alibi0 = 0.0f;
        cudaError_t copy_err = cudaMemcpyAsync(&alibi0, alibi_slopes, sizeof(float),
                                               cudaMemcpyDeviceToHost, stream);
        if (copy_err != cudaSuccess) {
            std::string error_msg = std::string("[FlashAttention] Failed to copy alibi_slopes[0]: ") +
                                    cudaGetErrorString(copy_err);
            FlashAttentionLog::error(error_msg);
        } else {
            cudaError_t sync_err = cudaStreamSynchronize(stream);
            if (sync_err != cudaSuccess) {
                std::string error_msg = std::string("[FlashAttention] Failed to sync after alibi_slopes[0] copy: ") +
                                        cudaGetErrorString(sync_err);
                FlashAttentionLog::error(error_msg);
            } else {
                char slope_msg[256];
                snprintf(slope_msg, sizeof(slope_msg),
                         "[FlashAttention] flash_attn_bwd_ex called with alibi_slopes[0]=%f "
                         "(n_heads=%d, n_kv_heads=%d)",
                         alibi0, n_heads, n_kv_heads);
                FlashAttentionLog::info(slope_msg);
            }
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
                                                   is_bf16, causal);
    {
        char stride_msg[512];
        snprintf(stride_msg, sizeof(stride_msg),
                 "[FlashAttention] bwd strides: q(b=%lld r=%lld h=%lld) k(b=%lld r=%lld h=%lld) v(b=%lld r=%lld h=%lld) "
                 "o(b=%lld r=%lld h=%lld) do(b=%lld r=%lld h=%lld) dq(b=%lld r=%lld h=%lld) dk(b=%lld r=%lld h=%lld) dv(b=%lld r=%lld h=%lld)",
                 static_cast<long long>(params.q_batch_stride),
                 static_cast<long long>(params.q_row_stride),
                 static_cast<long long>(params.q_head_stride),
                 static_cast<long long>(params.k_batch_stride),
                 static_cast<long long>(params.k_row_stride),
                 static_cast<long long>(params.k_head_stride),
                 static_cast<long long>(params.v_batch_stride),
                 static_cast<long long>(params.v_row_stride),
                 static_cast<long long>(params.v_head_stride),
                 static_cast<long long>(params.o_batch_stride),
                 static_cast<long long>(params.o_row_stride),
                 static_cast<long long>(params.o_head_stride),
                 static_cast<long long>(params.do_batch_stride),
                 static_cast<long long>(params.do_row_stride),
                 static_cast<long long>(params.do_head_stride),
                 static_cast<long long>(params.dq_batch_stride),
                 static_cast<long long>(params.dq_row_stride),
                 static_cast<long long>(params.dq_head_stride),
                 static_cast<long long>(params.dk_batch_stride),
                 static_cast<long long>(params.dk_row_stride),
                 static_cast<long long>(params.dk_head_stride),
                 static_cast<long long>(params.dv_batch_stride),
                 static_cast<long long>(params.dv_row_stride),
                 static_cast<long long>(params.dv_head_stride));
        FlashAttentionLog::info(stride_msg);
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
        FlashAttentionLog::info(shape_msg);
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
        FlashAttentionLog::info(ws_msg);
    }
    grim_flash::detail::check_cuda(cudaMemsetAsync(dq_accum, 0, dq_bytes, stream),
                                   "cudaMemsetAsync(dq_accum)");

    // ISSUE #76: Log saved softmax_lse statistics BEFORE backward pass - CRITICAL for boundary debugging
    // This is the LSE from forward pass, used to recompute softmax in backward
    static int s_bwd_call_count = 0;
    ++s_bwd_call_count;
    {
        const size_t lse_elems = static_cast<size_t>(batch) * n_heads * seqlen;
        std::vector<float> h_lse(lse_elems);
        cudaMemcpyAsync(h_lse.data(), softmax_lse, lse_elems * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        float global_min = h_lse[0], global_max = h_lse[0];
        int nan_count = 0, inf_count = 0;
        double sum = 0.0;
        for (size_t i = 0; i < lse_elems; ++i) {
            float val = h_lse[i];
            if (std::isnan(val)) { nan_count++; continue; }
            if (std::isinf(val)) { inf_count++; continue; }
            if (val < global_min) global_min = val;
            if (val > global_max) global_max = val;
            sum += val;
        }
        float mean = lse_elems > 0 ? static_cast<float>(sum / lse_elems) : 0.0f;
        
        fprintf(stderr, "[FA-BWD-SAVED-LSE] call=%d seqlen=%d | nan=%d inf=%d range=[%.4f, %.4f] mean=%.4f\n",
                s_bwd_call_count, seqlen, nan_count, inf_count, global_min, global_max, mean);
        
        // Per-head stats if anomalous or at boundary
        const bool is_boundary_seq = (seqlen >= 920) || (seqlen >= 1840);
        const bool has_anomaly = (global_max > 50.0f) || (nan_count > 0) || (inf_count > 0);
        if (has_anomaly || is_boundary_seq) {
            fprintf(stderr, "[FA-BWD-SAVED-LSE-PERHEAD] %s%s per-head stats:\n",
                    has_anomaly ? "*** LSE ANOMALY *** " : "",
                    is_boundary_seq ? "(BOUNDARY_SEQ)" : "");
            for (int h = 0; h < n_heads; ++h) {
                float head_min = std::numeric_limits<float>::max();
                float head_max = std::numeric_limits<float>::lowest();
                int head_nan = 0, head_inf = 0;
                double head_sum = 0.0;
                size_t head_count = 0;
                
                for (int b = 0; b < batch; ++b) {
                    for (int s = 0; s < seqlen; ++s) {
                        size_t idx = static_cast<size_t>(b) * n_heads * seqlen + h * seqlen + s;
                        float val = h_lse[idx];
                        if (std::isnan(val)) { head_nan++; continue; }
                        if (std::isinf(val)) { head_inf++; continue; }
                        if (val < head_min) head_min = val;
                        if (val > head_max) head_max = val;
                        head_sum += val;
                        head_count++;
                    }
                }
                float head_mean = head_count > 0 ? static_cast<float>(head_sum / head_count) : 0.0f;
                
                // Flag heads with unusually high LSE (potential ALiBi slope issue)
                const char* flag = (head_max > 50.0f) ? " *** EXPLOSION ***" : "";
                fprintf(stderr, "    head[%d]: range=[%.4f, %.4f] mean=%.4f nan=%d inf=%d%s\n",
                        h, head_min, head_max, head_mean, head_nan, head_inf, flag);
            }
        }
    }

    // ISSUE #67: Log Flash Attention BACKWARD INPUT (grad_output)
    // ISSUE #79 FIX: dout is BF16 (not FP32)! Must read as __nv_bfloat16 and convert.
    {
        const size_t grad_elems = static_cast<size_t>(batch) * n_heads * seqlen * head_dim;
        const size_t sample_size = std::min(grad_elems, static_cast<size_t>(10000));
        std::vector<__nv_bfloat16> h_dout_bf16(sample_size);
        cudaMemcpyAsync(h_dout_bf16.data(), dout, sample_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        int nan_dout = 0, inf_dout = 0;
        float dout_max = 0.0f;
        double dout_sq_sum = 0.0;
        for (const auto& val_bf16 : h_dout_bf16) {
            float val = __bfloat162float(val_bf16);
            if (std::isnan(val)) { nan_dout++; continue; }
            if (std::isinf(val)) { inf_dout++; continue; }
            dout_max = std::max(dout_max, std::abs(val));
            dout_sq_sum += static_cast<double>(val) * val;
        }
        float dout_rms = std::sqrt(static_cast<float>(dout_sq_sum / sample_size));
        float first_val = h_dout_bf16.empty() ? 0.0f : __bfloat162float(h_dout_bf16[0]);
        
        fprintf(stderr, "[FA-BWD-IN] grad_output (BF16): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f\n",
                nan_dout, inf_dout, dout_max, dout_rms, first_val);
    }

    FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex launching kernels");
#if defined(GRIM_FLASHATTN_HDIM64_ONLY)
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
    grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
    if (is_bf16) {
        grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
    } else {
        grim_flash::detail::run_mha_bwd_hdim64<cutlass::half_t, true>(params, stream);
    }
#endif
#else
    if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
        } else {
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::half_t, false>(params, stream);
        }
#endif
    }
#endif
#else
    if (head_dim == 32) {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_mha_bwd_hdim32<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_bwd_hdim32<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_mha_bwd_hdim32<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_bwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_bwd_hdim32<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_mha_bwd_hdim32<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    } else {
#if defined(GRIM_FLASHATTN_CAUSAL_ONLY)
#if defined(GRIM_FLASHATTN_BF16_ONLY)
        grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
        if (is_bf16) {
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
        } else {
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::half_t, true>(params, stream);
        }
#endif
#else
        if (causal) {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
            } else {
                grim_flash::detail::run_mha_bwd_hdim64<cutlass::half_t, true>(params, stream);
            }
#endif
        } else {
#if defined(GRIM_FLASHATTN_BF16_ONLY)
            grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
#else
            if (is_bf16) {
                grim_flash::detail::run_mha_bwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
            } else {
                grim_flash::detail::run_mha_bwd_hdim64<cutlass::half_t, false>(params, stream);
            }
#endif
        }
#endif
    }
#endif
    FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex kernel launch complete, checking CUDA status");
    grim_flash::detail::check_cuda(cudaGetLastError(), "flash_attn_bwd_ex launch");
    FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex synchronizing stream for error check");
    grim_flash::detail::check_cuda(cudaStreamSynchronize(stream), "flash_attn_bwd_ex sync");
    FlashAttentionLog::info("[FlashAttention] flash_attn_bwd_ex stream synchronized");

    // ISSUE #75: Log Flash Attention BACKWARD OUTPUTS (dQ, dK, dV) to isolate gradient explosion source
    // NOTE: dK and dV are written using QUERY head indices (0..n_heads-1), NOT KV head indices!
    // The Dao-AILab library writes to bidh * dk_head_stride where bidh goes from 0 to n_heads-1.
    // So these buffers must be sized for n_heads (12), not n_kv_heads (4).
    // ISSUE #79 FIX: dQ/dK/dV are BF16 (not FP32)! Must read as __nv_bfloat16 and convert.
    {
        // dQ is sized for n_heads (12), layout BSHD
        const size_t dq_elems = static_cast<size_t>(batch) * seqlen * n_heads * head_dim;
        const size_t dk_elems = static_cast<size_t>(batch) * seqlen * n_heads * head_dim;
        const size_t dv_elems = static_cast<size_t>(batch) * seqlen * n_heads * head_dim;
        
        const size_t sample_size = 10000;
        std::vector<__nv_bfloat16> h_dq(std::min(dq_elems, sample_size));
        std::vector<__nv_bfloat16> h_dk(std::min(dk_elems, sample_size));
        std::vector<__nv_bfloat16> h_dv(std::min(dv_elems, sample_size));
        
        cudaMemcpyAsync(h_dq.data(), dq, h_dq.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_dk.data(), dk, h_dk.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_dv.data(), dv, h_dv.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        int nan_dq = 0, inf_dq = 0, nan_dk = 0, inf_dk = 0, nan_dv = 0, inf_dv = 0;
        float max_dq = 0, max_dk = 0, max_dv = 0;
        double sum_sq_dq = 0, sum_sq_dk = 0, sum_sq_dv = 0;
        float first_dq = 0, first_dk = 0, first_dv = 0;
        
        for (size_t i = 0; i < h_dq.size(); ++i) {
            float val = __bfloat162float(h_dq[i]);
            if (i == 0) first_dq = val;
            if (std::isnan(val)) { nan_dq++; continue; }
            if (std::isinf(val)) { inf_dq++; continue; }
            float abs_val = std::fabs(val);
            if (abs_val > max_dq) max_dq = abs_val;
            sum_sq_dq += static_cast<double>(val) * val;
        }
        for (size_t i = 0; i < h_dk.size(); ++i) {
            float val = __bfloat162float(h_dk[i]);
            if (i == 0) first_dk = val;
            if (std::isnan(val)) { nan_dk++; continue; }
            if (std::isinf(val)) { inf_dk++; continue; }
            float abs_val = std::fabs(val);
            if (abs_val > max_dk) max_dk = abs_val;
            sum_sq_dk += static_cast<double>(val) * val;
        }
        for (size_t i = 0; i < h_dv.size(); ++i) {
            float val = __bfloat162float(h_dv[i]);
            if (i == 0) first_dv = val;
            if (std::isnan(val)) { nan_dv++; continue; }
            if (std::isinf(val)) { inf_dv++; continue; }
            float abs_val = std::fabs(val);
            if (abs_val > max_dv) max_dv = abs_val;
            sum_sq_dv += static_cast<double>(val) * val;
        }
        
        float rms_dq = std::sqrt(static_cast<float>(sum_sq_dq / std::max(h_dq.size(), size_t(1))));
        float rms_dk = std::sqrt(static_cast<float>(sum_sq_dk / std::max(h_dk.size(), size_t(1))));
        float rms_dv = std::sqrt(static_cast<float>(sum_sq_dv / std::max(h_dv.size(), size_t(1))));
        
        fprintf(stderr, "[FA-BWD-OUT] dQ (BF16): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f\n",
                nan_dq, inf_dq, max_dq, rms_dq, first_dq);
        fprintf(stderr, "[FA-BWD-OUT] actual tensor elems: dq=%zu dk=%zu dv=%zu (sampled %zu each) head_dim=%d\n",
                dq_elems, dk_elems, dv_elems, h_dq.size(), head_dim);
        fprintf(stderr, "[FA-BWD-OUT] dK (BF16): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f (n_heads=%d buffer)\n",
                nan_dk, inf_dk, max_dk, rms_dk, first_dk, n_heads);
        fprintf(stderr, "[FA-BWD-OUT] dV (BF16): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f (n_heads=%d buffer)\n",
                nan_dv, inf_dv, max_dv, rms_dv, first_dv, n_heads);
    }
}
