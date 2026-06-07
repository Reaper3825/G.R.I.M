#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Returns total workspace bytes required for backward (dq_accum + dsoftmax_sum).
std::size_t flash_attn_bwd_workspace_size_bytes(int batch,
                                                int seqlen,
                                                int n_heads,
                                                int head_dim);

// Returns bytes required for dq_accum workspace (fp32, padded as kernel expects).
std::size_t flash_attn_dq_accum_bytes(int batch,
                                      int seqlen,
                                      int n_heads,
                                      int head_dim);

// Returns bytes required for dsoftmax_sum workspace (fp32, padded as kernel expects).
std::size_t flash_attn_dsoftmax_sum_bytes(int batch,
                                          int seqlen,
                                          int n_heads);

// FlashAttention v2 forward.
// Q:  [batch, seqlen, n_heads, head_dim]    (FP16 or BF16)
// K,V:[batch, seqlen, n_kv_heads, head_dim] (FP16 or BF16)
// O:  [batch, seqlen, n_heads, head_dim]    (FP16 or BF16)
// LSE:[batch, n_heads, seqlen]              (FP32, dense, no padding)
// softmax_scale must be finite and > 0 and must match the caller's attention equation.
void flash_attn_fwd_ex(const void* q,
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
                       cudaStream_t stream);

// FlashAttention v2 forward with KV cache (inference-only, no backward).
//
// For autoregressive decoding: Q has seqlen_q new tokens, K/V cache has
// seqlen_k total tokens (including the new ones already appended by caller).
//
// Q:       [batch, seqlen_q, n_heads, head_dim]      (FP16 or BF16)
// K_cache: [batch, seqlen_k, n_kv_heads, head_dim]   (FP16 or BF16) — full cache including new tokens
// V_cache: [batch, seqlen_k, n_kv_heads, head_dim]   (FP16 or BF16) — full cache including new tokens
// O:       [batch, seqlen_q, n_heads, head_dim]       (FP16 or BF16)
// LSE:     [batch, n_heads, seqlen_q]                 (FP32)
//
// Typical decode usage: seqlen_q=1, seqlen_k=total_cached_tokens
// Prefill usage: seqlen_q=seqlen_k=prompt_length (equivalent to flash_attn_fwd_ex)
void flash_attn_fwd_kvcache(const void* q,
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
                            cudaStream_t stream);

// FlashAttention v2 backward.
// Requires softmax_lse from the matching forward pass.
// softmax_scale must match the forward pass and must be finite and > 0.
void flash_attn_bwd_ex(const void* q,
                       const void* k,
                       const void* v,
                       const void* out,
                       const void* dout,
                       const void* softmax_lse,
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
                       cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
