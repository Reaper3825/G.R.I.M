#pragma once

#include <cuda_runtime.h>
#ifdef USE_CUDA
#include <cuda_bf16.h>
#endif
#include <cstddef>
#include <stdexcept>

/**
 * TensorConversion - CUDA kernels for tensor format conversions
 * 
 * Common tensor dimension conventions:
 *   B = Batch size
 *   H = Number of attention heads (num_heads)
 *   S = Sequence length
 *   D = Head dimension (d_head = d_model / num_heads)
 *   M = Model dimension (d_model)
 *   F = Feed-forward dimension (d_ff)
 *   V = Vocabulary size
 * 
 * Common layout formats in transformers:
 *   BHSD = [Batch, Heads, Seq, Dim]   - Standard attention format
 *   BHDS = [Batch, Heads, Dim, Seq]   - Transposed for matmul
 *   BSHD = [Batch, Seq, Heads, Dim]   - Merged head format
 *   BSM  = [Batch, Seq, Model]        - Input/output embedding format
 *   SHD  = [Seq, Heads, Dim]          - KV cache format (single batch)
 */

namespace TensorConversion {

// ============================================================================
// Tensor Format Enum
// ============================================================================

/**
 * Tensor layout format identifiers for unified conversion API
 */
enum class TensorFormat {
    // 4D formats (attention tensors)
    BHSD,              // [Batch, Heads, Seq, Dim] - Standard attention format
    BHDS,              // [Batch, Heads, Dim, Seq] - Key transposed for Q @ K^T
    BSHD,              // [Batch, Seq, Heads, Dim] - Merged head format
    
    // 3D formats (KV cache, embeddings)
    SHD,               // [Seq, Heads, Dim] - KV cache format (single batch)
    HSD,               // [Heads, Seq, Dim] - Reordered KV cache
    BSM,               // [Batch, Seq, Model] - Input/output embeddings
    SMB,               // [Seq, Model, Batch] - Transposed embeddings
    
    // QKV special formats
    QKV_FUSED,         // [B, S, 3*M] - Fused QKV projection output
    QKV_SPLIT,         // Three separate [B, S, M] tensors
    QKV_HEAD_SPLIT,    // Three separate [B, H, S, D] tensors
};

// ============================================================================
// Attention Tensor Conversions (4D tensors)
// ============================================================================

/**
 * Convert BHSD to BHDS (transpose last two dims)
 * Used for: Key transpose before Q @ K^T
 * src: [B, H, S, D] -> dst: [B, H, D, S]
 */
void convert_BHSD_to_BHDS(const float* src, float* dst,
                          int B, int H, int S, int D,
                          cudaStream_t stream = nullptr);

/**
 * Convert BHDS to BHSD (transpose last two dims)
 * Used for: Inverse of key transpose
 * src: [B, H, D, S] -> dst: [B, H, S, D]
 */
void convert_BHDS_to_BHSD(const float* src, float* dst,
                          int B, int H, int D, int S,
                          cudaStream_t stream = nullptr);

/**
 * Convert BSHD to BHSD (swap Seq and Heads)
 * Used for: Converting merged head format to separate heads
 * src: [B, S, H, D] -> dst: [B, H, S, D]
 */
void convert_BSHD_to_BHSD(const float* src, float* dst,
                          int B, int S, int H, int D,
                          cudaStream_t stream = nullptr);

/**
 * Convert BHSD to BSHD (swap Heads and Seq)
 * Used for: Converting separate heads back to merged format
 * src: [B, H, S, D] -> dst: [B, S, H, D]
 */
void convert_BHSD_to_BSHD(const float* src, float* dst,
                          int B, int H, int S, int D,
                          cudaStream_t stream = nullptr);

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
// QKV Split/Merge Operations
// ============================================================================

/**
 * Split fused QKV into separate Q, K, V tensors
 * src: [B, S, 3*M] -> Q: [B, S, M], K: [B, S, M], V: [B, S, M]
 * where M = d_model = H * D
 */
void split_QKV(const float* src, float* Q, float* K, float* V,
               int B, int S, int M,
               cudaStream_t stream = nullptr);

/**
 * Split fused QKV and reshape to multi-head format
 * src: [B, S, 3*M] -> Q: [B, H, S, D], K: [B, H, S, D], V: [B, H, S, D]
 */
void split_QKV_to_heads(const float* src, float* Q, float* K, float* V,
                        int B, int S, int H, int D,
                        cudaStream_t stream = nullptr);

/**
 * Merge separate Q, K, V back into fused tensor
 * Q: [B, S, M], K: [B, S, M], V: [B, S, M] -> dst: [B, S, 3*M]
 */
void merge_QKV(const float* Q, const float* K, const float* V, float* dst,
               int B, int S, int M,
               cudaStream_t stream = nullptr);

// ============================================================================
// KV Cache Conversions (3D tensors, single batch)
// ============================================================================

/**
 * Convert SHD to HSD (KV cache reorder for attention)
 * src: [S, H, D] -> dst: [H, S, D]
 */
void convert_SHD_to_HSD(const float* src, float* dst,
                        int S, int H, int D,
                        cudaStream_t stream = nullptr);

/**
 * Convert HSD to SHD
 * src: [H, S, D] -> dst: [S, H, D]
 */
void convert_HSD_to_SHD(const float* src, float* dst,
                        int H, int S, int D,
                        cudaStream_t stream = nullptr);

// ============================================================================
// Embedding/Output Conversions (3D tensors)
// ============================================================================

/**
 * Convert BSM to SMB (batch-first to seq-first)
 * src: [B, S, M] -> dst: [S, M, B]
 */
void convert_BSM_to_SMB(const float* src, float* dst,
                        int B, int S, int M,
                        cudaStream_t stream = nullptr);

/**
 * Convert SMB to BSM (seq-first to batch-first)
 * src: [S, M, B] -> dst: [B, S, M]
 */
void convert_SMB_to_BSM(const float* src, float* dst,
                        int S, int M, int B,
                        cudaStream_t stream = nullptr);

// ============================================================================
// Weight Matrix Transpositions (2D tensors)
// ============================================================================

/**
 * Transpose 2D matrix
 * src: [rows, cols] -> dst: [cols, rows]
 */
void transpose_2D(const float* src, float* dst,
                  int rows, int cols,
                  cudaStream_t stream = nullptr);

/**
 * Transpose weight matrix for linear layer
 * Specialized for large vocabulary/embedding matrices
 * src: [V, M] -> dst: [M, V]
 */
void transpose_embedding(const float* src, float* dst,
                         int V, int M,
                         cudaStream_t stream = nullptr);

// ============================================================================
// Batch Operations
// ============================================================================

/**
 * Permute batched attention scores for softmax
 * src: [B, H, S_q, S_k] -> dst: [B * H, S_q, S_k]
 */
void flatten_batch_heads(const float* src, float* dst,
                         int B, int H, int S_q, int S_k,
                         cudaStream_t stream = nullptr);

/**
 * Unflatten batched attention back
 * src: [B * H, S_q, S_k] -> dst: [B, H, S_q, S_k]
 */
void unflatten_batch_heads(const float* src, float* dst,
                           int B, int H, int S_q, int S_k,
                           cudaStream_t stream = nullptr);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Copy tensor with optional stream
 */
void copy_tensor(const float* src, float* dst,
                 size_t num_elements,
                 cudaStream_t stream = nullptr);

/**
 * Zero tensor
 */
void zero_tensor(float* dst, size_t num_elements,
                 cudaStream_t stream = nullptr);

#ifdef USE_CUDA
/**
 * Convert float array to BF16.
 */
void convertFloatToBF16(const float* src, __nv_bfloat16* dst,
                        size_t num_elements,
                        cudaStream_t stream = nullptr);

/**
 * Convert BF16 array to float.
 */
void convertBF16ToFloat(const __nv_bfloat16* src, float* dst,
                        size_t num_elements,
                        cudaStream_t stream = nullptr);
#endif

// ============================================================================
// Unified Conversion API
// ============================================================================

/**
 * Unified tensor format conversion function
 * 
 * Converts tensor data between different memory layouts.
 * Supports all format combinations where conversion is meaningful.
 * 
 * @param src      Source tensor (device memory)
 * @param dst      Destination tensor (device memory)
 * @param srcFmt   Source tensor format
 * @param dstFmt   Destination tensor format
 * @param B        Batch size
 * @param H        Number of heads (or 1 for 3D tensors)
 * @param S        Sequence length
 * @param D        Head dimension (d_head) or model dimension for BSM/SMB
 * @param stream   CUDA stream for async execution
 * 
 * @throws std::runtime_error if conversion is not supported
 * 
 * Usage examples:
 *   // Transpose keys for attention
 *   convertTensor(K, K_t, TensorFormat::BHSD, TensorFormat::BHDS, B, H, S, D, stream);
 *   
 *   // Convert merged heads to separate heads
 *   convertTensor(x, x_heads, TensorFormat::BSHD, TensorFormat::BHSD, B, H, S, D, stream);
 */
void convertTensor(const float* src,
                   float* dst,
                   TensorFormat srcFmt,
                   TensorFormat dstFmt,
                   int B, int H, int S, int D,
                   cudaStream_t stream = nullptr);

/**
 * Get string name for tensor format (for debugging/logging)
 */
const char* getTensorFormatName(TensorFormat fmt);

/**
 * Calculate total number of elements for a given format
 */
size_t getTensorSize(TensorFormat fmt, int B, int H, int S, int D);

} // namespace TensorConversion
