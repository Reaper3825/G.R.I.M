#include "TensorConversion.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>

namespace TensorConversion {

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

// Use HyperParameters for kernel configuration
using GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
using GRIM::HyperParameters::CUDA_TILE_DIM_TRANSPOSE;

constexpr int BLOCK_SIZE = CUDA_BLOCK_SIZE_STANDARD;
constexpr int TILE_DIM = CUDA_TILE_DIM_TRANSPOSE;

// ----------------------------------------------------------------------------
// BHSD <-> BHDS Conversions (transpose last two dims)
// ----------------------------------------------------------------------------

__global__ void kernel_BHSD_to_BHDS(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    
    if (idx >= total) return;
    
    // Decode linear index to BHSD coordinates
    int d = idx % D;
    int s = (idx / D) % S;
    int h = (idx / (D * S)) % H;
    int b = idx / (D * S * H);
    
    // Source: [b,h,s,d] at index (((b*H + h)*S + s)*D + d)
    // Dest:   [b,h,d,s] at index (((b*H + h)*D + d)*S + s)
    int dstIdx = ((b * H + h) * D + d) * S + s;
    
    dst[dstIdx] = src[idx];
}

__global__ void kernel_BHDS_to_BHSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H, int D, int S)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * D * S;
    
    if (idx >= total) return;
    
    // Decode linear index to BHDS coordinates
    int s = idx % S;
    int d = (idx / S) % D;
    int h = (idx / (S * D)) % H;
    int b = idx / (S * D * H);
    
    // Source: [b,h,d,s] at index (((b*H + h)*D + d)*S + s)
    // Dest:   [b,h,s,d] at index (((b*H + h)*S + s)*D + d)
    int dstIdx = ((b * H + h) * S + s) * D + d;
    
    dst[dstIdx] = src[idx];
}

// ----------------------------------------------------------------------------
// BSHD <-> BHSD Conversions (swap Seq and Heads dims)
// ----------------------------------------------------------------------------

__global__ void kernel_BSHD_to_BHSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    
    if (idx >= total) return;
    
    // Decode linear index to BSHD coordinates
    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (D * H * S);
    
    // Source: [b,s,h,d] at index (((b*S + s)*H + h)*D + d)
    // Dest:   [b,h,s,d] at index (((b*H + h)*S + s)*D + d)
    int dstIdx = ((b * H + h) * S + s) * D + d;
    
    dst[dstIdx] = src[idx];
}

__global__ void kernel_BHSD_to_BSHD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    
    if (idx >= total) return;
    
    // Decode linear index to BHSD coordinates
    int d = idx % D;
    int s = (idx / D) % S;
    int h = (idx / (D * S)) % H;
    int b = idx / (D * S * H);
    
    // Source: [b,h,s,d] at index (((b*H + h)*S + s)*D + d)
    // Dest:   [b,s,h,d] at index (((b*S + s)*H + h)*D + d)
    int dstIdx = ((b * S + s) * H + h) * D + d;
    
    dst[dstIdx] = src[idx];
}

// ----------------------------------------------------------------------------
// BHSD float <-> BSHD bf16 Conversions (FlashAttention v2 input/output)
// ----------------------------------------------------------------------------

__global__ void kernel_BHSD_to_BSHD_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int B, int H, int S, int D)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(B) * H * S * D;
    
    if (idx >= total) return;
    
    int d = static_cast<int>(idx % D);
    int s = static_cast<int>((idx / D) % S);
    int h = static_cast<int>((idx / (static_cast<size_t>(D) * S)) % H);
    int b = static_cast<int>(idx / (static_cast<size_t>(D) * S * H));
    
    size_t dstIdx = (static_cast<size_t>(b) * S + s) * H * D + h * D + d;
    dst[dstIdx] = __float2bfloat16_rn(src[idx]);
}

__global__ void kernel_BSHD_bf16_to_BHSD(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int B, int S, int H, int D)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(B) * S * H * D;
    
    if (idx >= total) return;
    
    int d = static_cast<int>(idx % D);
    int h = static_cast<int>((idx / D) % H);
    int s = static_cast<int>((idx / (static_cast<size_t>(D) * H)) % S);
    int b = static_cast<int>(idx / (static_cast<size_t>(D) * H * S));
    
    size_t dstIdx = (static_cast<size_t>(b) * H + h) * S * D + s * D + d;
    dst[dstIdx] = __bfloat162float(src[idx]);
}

// ----------------------------------------------------------------------------
// BHSD <-> BSM Conversions (attention heads to embedding format)
// ----------------------------------------------------------------------------

__global__ void kernel_BHSD_to_BSM(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    
    if (idx >= total) return;
    
    // Decode linear index to BHSD coordinates
    int d = idx % D;
    int s = (idx / D) % S;
    int h = (idx / (D * S)) % H;
    int b = idx / (D * S * H);
    
    // Source: [b,h,s,d] at index (((b*H + h)*S + s)*D + d)
    // Dest:   [b,s,M] where M=H*D at index ((b*S + s)*M + h*D + d)
    int M = H * D;
    int dstIdx = (b * S + s) * M + h * D + d;
    
    dst[dstIdx] = src[idx];
}

__global__ void kernel_BSM_to_BHSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = H * D;
    int total = B * S * M;
    
    if (idx >= total) return;
    
    // Decode linear index to BSM coordinates
    int m = idx % M;
    int s = (idx / M) % S;
    int b = idx / (M * S);
    
    // Convert M to (h,d)
    int h = m / D;
    int d = m % D;
    
    // Source: [b,s,M] at index ((b*S + s)*M + m)
    // Dest:   [b,h,s,d] at index (((b*H + h)*S + s)*D + d)
    int dstIdx = ((b * H + h) * S + s) * D + d;
    
    dst[dstIdx] = src[idx];
}

// ----------------------------------------------------------------------------
// QKV Split/Merge Operations
// ----------------------------------------------------------------------------

__global__ void kernel_split_QKV(
    const float* __restrict__ src,
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    int B, int S, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * M;
    
    if (idx >= total) return;
    
    // Decode to B, S, M coordinates
    int m = idx % M;
    int s = (idx / M) % S;
    int b = idx / (M * S);
    
    // Source has 3*M in last dim: [b, s, 0..M] = Q, [b, s, M..2M] = K, [b, s, 2M..3M] = V
    int srcBase = (b * S + s) * (3 * M);
    int dstIdx = (b * S + s) * M + m;
    
    Q[dstIdx] = src[srcBase + m];
    K[dstIdx] = src[srcBase + M + m];
    V[dstIdx] = src[srcBase + 2 * M + m];
}

__global__ void kernel_split_QKV_to_heads(
    const float* __restrict__ src,
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = H * D;
    int total = B * S * M;
    
    if (idx >= total) return;
    
    // Decode to B, S, H, D coordinates
    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (D * H * S);
    
    // Source: [b, s, 3*M] where M = H*D
    // Layout within 3*M: Q[h,d], K[h,d], V[h,d] (interleaved by head)
    int srcBase = (b * S + s) * (3 * M);
    int srcQ = srcBase + h * D + d;
    int srcK = srcBase + M + h * D + d;
    int srcV = srcBase + 2 * M + h * D + d;
    
    // Dest: [b, h, s, d]
    int dstIdx = ((b * H + h) * S + s) * D + d;
    
    Q[dstIdx] = src[srcQ];
    K[dstIdx] = src[srcK];
    V[dstIdx] = src[srcV];
}

__global__ void kernel_merge_QKV(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ dst,
    int B, int S, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * M;
    
    if (idx >= total) return;
    
    int m = idx % M;
    int s = (idx / M) % S;
    int b = idx / (M * S);
    
    int srcIdx = (b * S + s) * M + m;
    int dstBase = (b * S + s) * (3 * M);
    
    dst[dstBase + m] = Q[srcIdx];
    dst[dstBase + M + m] = K[srcIdx];
    dst[dstBase + 2 * M + m] = V[srcIdx];
}

// ----------------------------------------------------------------------------
// KV Cache Conversions (3D: SHD <-> HSD)
// ----------------------------------------------------------------------------

__global__ void kernel_SHD_to_HSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * D;
    
    if (idx >= total) return;
    
    // Decode to SHD coordinates
    int d = idx % D;
    int h = (idx / D) % H;
    int s = idx / (D * H);
    
    // Source: [s,h,d] at index ((s*H + h)*D + d)
    // Dest:   [h,s,d] at index ((h*S + s)*D + d)
    int dstIdx = (h * S + s) * D + d;
    
    dst[dstIdx] = src[idx];
}

__global__ void kernel_HSD_to_SHD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int H, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * S * D;
    
    if (idx >= total) return;
    
    // Decode to HSD coordinates
    int d = idx % D;
    int s = (idx / D) % S;
    int h = idx / (D * S);
    
    // Source: [h,s,d] at index ((h*S + s)*D + d)
    // Dest:   [s,h,d] at index ((s*H + h)*D + d)
    int dstIdx = (s * H + h) * D + d;
    
    dst[dstIdx] = src[idx];
}

// ----------------------------------------------------------------------------
// Embedding/Output Conversions (BSM <-> SMB)
// ----------------------------------------------------------------------------

__global__ void kernel_BSM_to_SMB(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int S, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * M;
    
    if (idx >= total) return;
    
    // Decode to BSM coordinates
    int m = idx % M;
    int s = (idx / M) % S;
    int b = idx / (M * S);
    
    // Source: [b,s,m] at index ((b*S + s)*M + m)
    // Dest:   [s,m,b] at index ((s*M + m)*B + b)
    int dstIdx = (s * M + m) * B + b;
    
    dst[dstIdx] = src[idx];
}

__global__ void kernel_SMB_to_BSM(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int S, int M, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * M * B;
    
    if (idx >= total) return;
    
    // Decode to SMB coordinates
    int b = idx % B;
    int m = (idx / B) % M;
    int s = idx / (B * M);
    
    // Source: [s,m,b] at index ((s*M + m)*B + b)
    // Dest:   [b,s,m] at index ((b*S + s)*M + m)
    int dstIdx = (b * S + s) * M + m;
    
    dst[dstIdx] = src[idx];
}

// ----------------------------------------------------------------------------
// 2D Transpose (tiled for better memory coalescing)
// ----------------------------------------------------------------------------

__global__ void kernel_transpose_2D(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows, int cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile from source (coalesced read)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = src[y * cols + x];
    }
    
    __syncthreads();
    
    // Transposed coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write transposed tile to destination (coalesced write)
    if (x < rows && y < cols) {
        dst[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Simple transpose for non-tile-aligned cases
__global__ void kernel_transpose_2D_simple(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx >= total) return;
    
    int c = idx % cols;
    int r = idx / cols;
    
    // src[r,c] -> dst[c,r]
    dst[c * rows + r] = src[idx];
}

// ----------------------------------------------------------------------------
// Batch Operations (flatten/unflatten batch and heads)
// ----------------------------------------------------------------------------

__global__ void kernel_flatten_batch_heads(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H, int S_q, int S_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S_q * S_k;
    
    if (idx >= total) return;
    
    // For flatten, the data layout is actually the same!
    // [B, H, S_q, S_k] is contiguous and equals [B*H, S_q, S_k]
    // This is effectively a no-op copy, but we include it for completeness
    dst[idx] = src[idx];
}

__global__ void kernel_unflatten_batch_heads(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H, int S_q, int S_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S_q * S_k;
    
    if (idx >= total) return;
    
    // Same as flatten - contiguous layouts are identical
    dst[idx] = src[idx];
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

void convert_BHSD_to_BHDS(const float* src, float* dst,
                          int B, int H, int S, int D,
                          cudaStream_t stream)
{
    int total = B * H * S * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BHSD_to_BHDS<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, S, D);
}

void convert_BHDS_to_BHSD(const float* src, float* dst,
                          int B, int H, int D, int S,
                          cudaStream_t stream)
{
    int total = B * H * D * S;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BHDS_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, D, S);
}

void convert_BSHD_to_BHSD(const float* src, float* dst,
                          int B, int S, int H, int D,
                          cudaStream_t stream)
{
    int total = B * S * H * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BSHD_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, S, H, D);
}

void convert_BHSD_to_BSHD(const float* src, float* dst,
                          int B, int H, int S, int D,
                          cudaStream_t stream)
{
    int total = B * H * S * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BHSD_to_BSHD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, S, D);
}

void convert_BHSD_to_BSHD_bf16(const float* src, __nv_bfloat16* dst,
                               int B, int H, int S, int D,
                               cudaStream_t stream)
{
    size_t total = static_cast<size_t>(B) * H * S * D;
    int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_BHSD_to_BSHD_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, S, D);
}

void convert_BSHD_bf16_to_BHSD(const __nv_bfloat16* src, float* dst,
                               int B, int S, int H, int D,
                               cudaStream_t stream)
{
    size_t total = static_cast<size_t>(B) * S * H * D;
    int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_BSHD_bf16_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, S, H, D);
}

void convert_BHSD_to_BSM(const float* src, float* dst,
                         int B, int H, int S, int D,
                         cudaStream_t stream)
{
    int total = B * H * S * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BHSD_to_BSM<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, S, D);
}

void convert_BSM_to_BHSD(const float* src, float* dst,
                         int B, int S, int H, int D,
                         cudaStream_t stream)
{
    int M = H * D;
    int total = B * S * M;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BSM_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, S, H, D);
}

void split_QKV(const float* src, float* Q, float* K, float* V,
               int B, int S, int M,
               cudaStream_t stream)
{
    int total = B * S * M;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_split_QKV<<<blocks, BLOCK_SIZE, 0, stream>>>(src, Q, K, V, B, S, M);
}

void split_QKV_to_heads(const float* src, float* Q, float* K, float* V,
                        int B, int S, int H, int D,
                        cudaStream_t stream)
{
    int M = H * D;
    int total = B * S * M;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_split_QKV_to_heads<<<blocks, BLOCK_SIZE, 0, stream>>>(src, Q, K, V, B, S, H, D);
}

void merge_QKV(const float* Q, const float* K, const float* V, float* dst,
               int B, int S, int M,
               cudaStream_t stream)
{
    int total = B * S * M;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_merge_QKV<<<blocks, BLOCK_SIZE, 0, stream>>>(Q, K, V, dst, B, S, M);
}

void convert_SHD_to_HSD(const float* src, float* dst,
                        int S, int H, int D,
                        cudaStream_t stream)
{
    int total = S * H * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_SHD_to_HSD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, S, H, D);
}

void convert_HSD_to_SHD(const float* src, float* dst,
                        int H, int S, int D,
                        cudaStream_t stream)
{
    int total = H * S * D;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_HSD_to_SHD<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, H, S, D);
}

void convert_BSM_to_SMB(const float* src, float* dst,
                        int B, int S, int M,
                        cudaStream_t stream)
{
    int total = B * S * M;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_BSM_to_SMB<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, S, M);
}

void convert_SMB_to_BSM(const float* src, float* dst,
                        int S, int M, int B,
                        cudaStream_t stream)
{
    int total = S * M * B;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_SMB_to_BSM<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, S, M, B);
}

void transpose_2D(const float* src, float* dst,
                  int rows, int cols,
                  cudaStream_t stream)
{
    // Use tiled version for better performance on large matrices
    if (rows >= TILE_DIM && cols >= TILE_DIM) {
        dim3 blocks((cols + TILE_DIM - 1) / TILE_DIM,
                    (rows + TILE_DIM - 1) / TILE_DIM);
        dim3 threads(TILE_DIM, TILE_DIM);
        kernel_transpose_2D<<<blocks, threads, 0, stream>>>(src, dst, rows, cols);
    } else {
        // Fall back to simple version for small matrices
        int total = rows * cols;
        int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_transpose_2D_simple<<<numBlocks, BLOCK_SIZE, 0, stream>>>(src, dst, rows, cols);
    }
}

void transpose_embedding(const float* src, float* dst,
                         int V, int M,
                         cudaStream_t stream)
{
    // Wrapper around transpose_2D for embedding matrices
    transpose_2D(src, dst, V, M, stream);
}

void flatten_batch_heads(const float* src, float* dst,
                         int B, int H, int S_q, int S_k,
                         cudaStream_t stream)
{
    int total = B * H * S_q * S_k;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_flatten_batch_heads<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, S_q, S_k);
}

void unflatten_batch_heads(const float* src, float* dst,
                           int B, int H, int S_q, int S_k,
                           cudaStream_t stream)
{
    int total = B * H * S_q * S_k;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_unflatten_batch_heads<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, B, H, S_q, S_k);
}

void copy_tensor(const float* src, float* dst,
                 size_t num_elements,
                 cudaStream_t stream)
{
    cudaMemcpyAsync(dst, src, num_elements * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}

void zero_tensor(float* dst, size_t num_elements,
                 cudaStream_t stream)
{
    cudaMemsetAsync(dst, 0, num_elements * sizeof(float), stream);
}

#ifdef USE_CUDA
// ----------------------------------------------------------------------------
// Float <-> BF16 Conversions
// ----------------------------------------------------------------------------

__global__ void kernel_float_to_bf16(const float* __restrict__ src,
                                     __nv_bfloat16* __restrict__ dst,
                                     size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ void kernel_bf16_to_float(const __nv_bfloat16* __restrict__ src,
                                     float* __restrict__ dst,
                                     size_t count)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

void convertFloatToBF16(const float* src, __nv_bfloat16* dst,
                        size_t num_elements,
                        cudaStream_t stream)
{
    if (!src || !dst || num_elements == 0) {
        return;
    }
    const int blocks = static_cast<int>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_float_to_bf16<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, num_elements);
}

void convertBF16ToFloat(const __nv_bfloat16* src, float* dst,
                        size_t num_elements,
                        cudaStream_t stream)
{
    if (!src || !dst || num_elements == 0) {
        return;
    }
    const int blocks = static_cast<int>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_bf16_to_float<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, num_elements);
}
#endif

// ============================================================================
// Unified Conversion API Implementation
// ============================================================================

const char* getTensorFormatName(TensorFormat fmt)
{
    switch (fmt) {
        case TensorFormat::BHSD:          return "BHSD";
        case TensorFormat::BHDS:          return "BHDS";
        case TensorFormat::BSHD:          return "BSHD";
        case TensorFormat::SHD:           return "SHD";
        case TensorFormat::HSD:           return "HSD";
        case TensorFormat::BSM:           return "BSM";
        case TensorFormat::SMB:           return "SMB";
        case TensorFormat::QKV_FUSED:     return "QKV_FUSED";
        case TensorFormat::QKV_SPLIT:     return "QKV_SPLIT";
        case TensorFormat::QKV_HEAD_SPLIT: return "QKV_HEAD_SPLIT";
        default:                          return "UNKNOWN";
    }
}

size_t getTensorSize(TensorFormat fmt, int B, int H, int S, int D)
{
    switch (fmt) {
        // 4D formats: B * H * S * D
        case TensorFormat::BHSD:
        case TensorFormat::BHDS:
        case TensorFormat::BSHD:
            return static_cast<size_t>(B) * H * S * D;
        
        // 3D KV cache formats: S * H * D (B assumed = 1)
        case TensorFormat::SHD:
        case TensorFormat::HSD:
            return static_cast<size_t>(S) * H * D;
        
        // 3D embedding formats: B * S * M where M = H * D
        case TensorFormat::BSM:
        case TensorFormat::SMB:
            return static_cast<size_t>(B) * S * H * D;
        
        // QKV formats: 3 * B * S * M where M = H * D
        case TensorFormat::QKV_FUSED:
            return static_cast<size_t>(3) * B * S * H * D;
        
        // Split QKV: each tensor is B * S * M
        case TensorFormat::QKV_SPLIT:
        case TensorFormat::QKV_HEAD_SPLIT:
            return static_cast<size_t>(B) * S * H * D;  // Per-tensor size
        
        default:
            return 0;
    }
}

void convertTensor(const float* src,
                   float* dst,
                   TensorFormat srcFmt,
                   TensorFormat dstFmt,
                   int B, int H, int S, int D,
                   cudaStream_t stream)
{
    // Same format - just copy
    if (srcFmt == dstFmt) {
        size_t numElements = getTensorSize(srcFmt, B, H, S, D);
        cudaMemcpyAsync(dst, src, numElements * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // =========================================================================
    // BHSD conversions (standard attention format)
    // =========================================================================
    
    // BHSD -> BHDS (transpose last two dims for key transpose)
    if (srcFmt == TensorFormat::BHSD && dstFmt == TensorFormat::BHDS) {
        convert_BHSD_to_BHDS(src, dst, B, H, S, D, stream);
        return;
    }
    
    // BHSD -> BSHD (separate heads to merged heads)
    if (srcFmt == TensorFormat::BHSD && dstFmt == TensorFormat::BSHD) {
        convert_BHSD_to_BSHD(src, dst, B, H, S, D, stream);
        return;
    }

    // =========================================================================
    // BHDS conversions (transposed attention format)
    // =========================================================================
    
    // BHDS -> BHSD (inverse of key transpose)
    if (srcFmt == TensorFormat::BHDS && dstFmt == TensorFormat::BHSD) {
        convert_BHDS_to_BHSD(src, dst, B, H, D, S, stream);
        return;
    }

    // =========================================================================
    // BSHD conversions (merged head format)
    // =========================================================================
    
    // BSHD -> BHSD (merged heads to separate heads)
    if (srcFmt == TensorFormat::BSHD && dstFmt == TensorFormat::BHSD) {
        convert_BSHD_to_BHSD(src, dst, B, S, H, D, stream);
        return;
    }

    // =========================================================================
    // BHSD <-> BSM conversions (attention heads to/from embedding format)
    // =========================================================================
    
    // BHSD -> BSM (separate heads to merged embedding)
    if (srcFmt == TensorFormat::BHSD && dstFmt == TensorFormat::BSM) {
        convert_BHSD_to_BSM(src, dst, B, H, S, D, stream);
        return;
    }
    
    // BSM -> BHSD (merged embedding to separate heads)
    if (srcFmt == TensorFormat::BSM && dstFmt == TensorFormat::BHSD) {
        convert_BSM_to_BHSD(src, dst, B, S, H, D, stream);
        return;
    }

    // =========================================================================
    // SHD/HSD conversions (KV cache formats, single batch)
    // =========================================================================
    
    // SHD -> HSD (reorder KV cache for attention)
    if (srcFmt == TensorFormat::SHD && dstFmt == TensorFormat::HSD) {
        convert_SHD_to_HSD(src, dst, S, H, D, stream);
        return;
    }
    
    // HSD -> SHD (inverse KV cache reorder)
    if (srcFmt == TensorFormat::HSD && dstFmt == TensorFormat::SHD) {
        convert_HSD_to_SHD(src, dst, H, S, D, stream);
        return;
    }

    // =========================================================================
    // BSM/SMB conversions (embedding formats)
    // =========================================================================
    
    // BSM -> SMB (batch-first to seq-first)
    // Note: M = H * D in this context
    if (srcFmt == TensorFormat::BSM && dstFmt == TensorFormat::SMB) {
        int M = H * D;
        convert_BSM_to_SMB(src, dst, B, S, M, stream);
        return;
    }
    
    // SMB -> BSM (seq-first to batch-first)
    if (srcFmt == TensorFormat::SMB && dstFmt == TensorFormat::BSM) {
        int M = H * D;
        convert_SMB_to_BSM(src, dst, S, M, B, stream);
        return;
    }

    // =========================================================================
    // Multi-hop conversions (using intermediate buffer would be needed)
    // For now, throw error for unsupported direct conversions
    // =========================================================================
    
    // Throw error for unsupported conversion
    std::string errorMsg = "Unsupported tensor format conversion: ";
    errorMsg += getTensorFormatName(srcFmt);
    errorMsg += " -> ";
    errorMsg += getTensorFormatName(dstFmt);
    throw std::runtime_error(errorMsg);
}

} // namespace TensorConversion
