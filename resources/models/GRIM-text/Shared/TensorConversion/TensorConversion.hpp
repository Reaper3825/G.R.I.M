#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <stdexcept>

/**
 * TensorConversion - CUDA kernels for tensor format conversions
 * 
 * Common tensor dimension conventions:
 *   B = Batch size
 *   H = Number of attention heads (num_heads for Q, num_kv_heads for K/V with GQA)
 *   S = Sequence length
 *   D = Head dimension (d_head = d_model / num_heads)
 *   M = Model dimension (d_model)
 * 
 * Common layout formats in transformers:
 *   BHSD = [Batch, Heads, Seq, Dim]   - Standard attention format
 *   BHDS = [Batch, Heads, Dim, Seq]   - Transposed for matmul
 *   BSHD = [Batch, Seq, Heads, Dim]   - Merged head format
 *   BSM  = [Batch, Seq, Model]        - Input/output embedding format
 */

namespace TensorConversion {

// ============================================================================
// Generic Precision Conversions
// ============================================================================

/**
 * Convert a contiguous FP32 buffer to BF16.
 * TensorContract owns the tensor metadata and calls this owner when an operation
 * boundary requires an implicit precision conversion.
 *
 * @throws std::runtime_error if src/dst/stream is NULL or count is zero
 */
void convert_fp32_to_bf16(const float* src, __nv_bfloat16* dst,
                          std::size_t count,
                          cudaStream_t stream);

/**
 * Convert a contiguous BF16 buffer to FP32.
 * TensorContract owns the tensor metadata and calls this owner when an operation
 * boundary requires an implicit precision conversion.
 *
 * @throws std::runtime_error if src/dst/stream is NULL or count is zero
 */
void convert_bf16_to_fp32(const __nv_bfloat16* src, float* dst,
                          std::size_t count,
                          cudaStream_t stream);

// ============================================================================
// Attention Tensor Conversions (4D tensors)
// ============================================================================

/**
 * Convert BHSD (float) to BSHD (bf16) with layout change.
 * Used for: FlashAttention v2 inputs (expects BSHD).
 * src: [B, H, S, D] float -> dst: [B, S, H, D] bf16
 */
void convert_BHSD_to_BSHD_bf16(const float* src, __nv_bfloat16* dst,
                               int B, int H, int S, int D,
                               cudaStream_t stream = nullptr);

/**
 * Convert BSHD (bf16) to BHSD (float) with layout change.
 * Used for: FlashAttention v2 outputs (convert back to BHSD float).
 * src: [B, S, H, D] bf16 -> dst: [B, H, S, D] float
 */
void convert_BSHD_bf16_to_BHSD(const __nv_bfloat16* src, float* dst,
                               int B, int S, int H, int D,
                               cudaStream_t stream = nullptr);

/**
 * Convert BHSD to BSM (separate heads to merged embedding)
 * Used for: Converting multi-head attention output to embedding format
 * src: [B, H, S, D] -> dst: [B, S, M] where M = H * D
 */
void convert_BHSD_to_BSM(const float* src, float* dst,
                         int B, int H, int S, int D,
                         cudaStream_t stream = nullptr);

/**
 * Convert BSM to BHSD (merged embedding to separate heads)
 * Used for: Converting embedding format to multi-head attention format
 * src: [B, S, M] -> dst: [B, H, S, D] where M = H * D
 */
void convert_BSM_to_BHSD(const float* src, float* dst,
                         int B, int S, int H, int D,
                         cudaStream_t stream = nullptr);

// ============================================================================
// QKV Split/Merge Operations (GQA-Aware)
// Supports Grouped Query Attention where num_heads != num_kv_heads.
// For standard MHA: pass num_heads == num_kv_heads.
// ============================================================================

/**
 * Split fused QKV projection output into separate Q, K, V tensors.
 * GQA-aware: Q has num_heads, K/V have num_kv_heads.
 * Uses float4 vectorization when head_dim % 4 == 0 (scalar fallback otherwise).
 *
 * @param qkv_fused  [batch*seq, (num_heads + 2*num_kv_heads) * head_dim]
 * @param Q          [batch, num_heads, seq, head_dim]
 * @param K          [batch, num_kv_heads, seq, head_dim]
 * @param V          [batch, num_kv_heads, seq, head_dim]
 *
 * @throws std::runtime_error if any pointer is NULL or any dimension <= 0
 */
void split_qkv_gqa(
    const float* qkv_fused, float* Q, float* K, float* V,
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim,
    cudaStream_t stream = nullptr);

/**
 * Merge separate Q, K, V gradient tensors back into fused QKV gradient.
 * GQA-aware: grad_Q has num_heads, grad_K/V have num_kv_heads.
 * Uses float4 vectorization when head_dim % 4 == 0 (scalar fallback otherwise).
 *
 * @param grad_Q     [batch, num_heads, seq, head_dim]
 * @param grad_K     [batch, num_kv_heads, seq, head_dim]
 * @param grad_V     [batch, num_kv_heads, seq, head_dim]
 * @param grad_qkv   [batch*seq, (num_heads + 2*num_kv_heads) * head_dim]
 *
 * @throws std::runtime_error if any pointer is NULL or any dimension <= 0
 */
void merge_qkv_grads_gqa(
    const float* grad_Q, const float* grad_K, const float* grad_V,
    float* grad_qkv,
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim,
    cudaStream_t stream = nullptr);

} // namespace TensorConversion
