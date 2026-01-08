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
#include <stdexcept>
#include <string>

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
    dim3 grid(params.b, params.h, 1);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            (params.seqlen_q % Kernel_traits::kBlockM == 0) &&
                            (params.seqlen_k % Kernel_traits::kBlockN == 0);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

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
}
