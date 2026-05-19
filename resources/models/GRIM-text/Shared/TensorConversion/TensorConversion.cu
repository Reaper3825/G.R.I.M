#include "TensorConversion.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <algorithm>  // for std::max in split/merge GQA
#include <limits>

namespace TensorConversion {

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

// Use HyperParameters for kernel configuration
using GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

constexpr int BLOCK_SIZE = CUDA_BLOCK_SIZE_STANDARD;

constexpr int kMaxGridDimY = 65535;

dim3 gridForCount(std::size_t count) {
    if (count == 0) {
        throw std::runtime_error("TensorConversion::gridForCount: count is zero");
    }
    const std::size_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return dim3(static_cast<unsigned int>(blocks), 1, 1);
    }
    const std::size_t grid_x = (blocks + kMaxGridDimY - 1) / kMaxGridDimY;
    if (grid_x > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error("TensorConversion::gridForCount: tensor is too large for CUDA grid dimensions");
    }
    return dim3(static_cast<unsigned int>(grid_x), kMaxGridDimY, 1);
}

__device__ std::size_t globalLinearIndex() {
    return (static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
}

// ----------------------------------------------------------------------------
// Generic FP32 <-> BF16 Conversions
// ----------------------------------------------------------------------------

__global__ void kernel_fp32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    std::size_t count)
{
    const std::size_t idx = globalLinearIndex();
    if (idx >= count) return;
    dst[idx] = __float2bfloat16_rn(src[idx]);
}

__global__ void kernel_bf16_to_fp32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    std::size_t count)
{
    const std::size_t idx = globalLinearIndex();
    if (idx >= count) return;
    dst[idx] = __bfloat162float(src[idx]);
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
// QKV Split/Merge Operations (GQA-aware)
// These are the production kernels - support different Q/KV head counts.
// For MHA: set num_heads == num_kv_heads.
// ----------------------------------------------------------------------------

// Vectorized (float4) GQA split - requires head_dim % 4 == 0
__global__ void kernel_split_qkv_gqa_f4(
    const float* __restrict__ qkv_fused,    // [batch*seq, total_qkv_dim]
    float* __restrict__ Q,                   // [batch, num_heads, seq, head_dim]
    float* __restrict__ K,                   // [batch, num_kv_heads, seq, head_dim]
    float* __restrict__ V,                   // [batch, num_kv_heads, seq, head_dim]
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim)
{
    // Grid: (tokens, max(num_heads, num_kv_heads))
    // Block: (head_dim / 4)
    const int token = blockIdx.x;
    const int h = blockIdx.y;
    const int d4 = threadIdx.x;
    
    const int total_tokens = batch * seq;
    const int D4 = head_dim >> 2;
    
    if (token >= total_tokens || d4 >= D4) return;
    
    const int b = token / seq;
    const int s = token % seq;
    
    const int q_dim4 = num_heads * D4;
    const int kv_dim4 = num_kv_heads * D4;
    const int total_dim4 = q_dim4 + 2 * kv_dim4;
    
    const float4* in4 = reinterpret_cast<const float4*>(qkv_fused) + token * total_dim4;
    
    // Extract Q (h < num_heads)
    if (h < num_heads) {
        const int q_feature4 = h * D4 + d4;
        const float4 qv = in4[q_feature4];
        const int q_out_idx4 = ((b * num_heads + h) * seq + s) * D4 + d4;
        reinterpret_cast<float4*>(Q)[q_out_idx4] = qv;
    }
    
    // Extract K, V (h < num_kv_heads)
    if (h < num_kv_heads) {
        const int kv_feature4 = h * D4 + d4;
        const float4 kv = in4[q_dim4 + kv_feature4];
        const float4 vv = in4[q_dim4 + kv_dim4 + kv_feature4];
        const int kv_out_idx4 = ((b * num_kv_heads + h) * seq + s) * D4 + d4;
        reinterpret_cast<float4*>(K)[kv_out_idx4] = kv;
        reinterpret_cast<float4*>(V)[kv_out_idx4] = vv;
    }
}

// Scalar fallback GQA split - works for any head_dim
__global__ void kernel_split_qkv_gqa(
    const float* __restrict__ qkv_fused,
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch * seq;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int total_qkv_dim = q_dim + 2 * kv_dim;
    int total = total_tokens * total_qkv_dim;
    
    if (idx >= total) return;
    
    int feature = idx % total_qkv_dim;
    int token = idx / total_qkv_dim;
    int b = token / seq;
    int s = token % seq;
    
    float value = qkv_fused[idx];
    
    if (feature < q_dim) {
        // Q region -> output to Q[b, h, s, d]
        int h = feature / head_dim;
        int d = feature % head_dim;
        int out_idx = ((b * num_heads + h) * seq + s) * head_dim + d;
        Q[out_idx] = value;
    } else if (feature < q_dim + kv_dim) {
        // K region -> output to K[b, h, s, d]
        int kv_feature = feature - q_dim;
        int h = kv_feature / head_dim;
        int d = kv_feature % head_dim;
        int out_idx = ((b * num_kv_heads + h) * seq + s) * head_dim + d;
        K[out_idx] = value;
    } else {
        // V region -> output to V[b, h, s, d]
        int kv_feature = feature - q_dim - kv_dim;
        int h = kv_feature / head_dim;
        int d = kv_feature % head_dim;
        int out_idx = ((b * num_kv_heads + h) * seq + s) * head_dim + d;
        V[out_idx] = value;
    }
}

// Vectorized (float4) GQA merge gradients - requires head_dim % 4 == 0
__global__ void kernel_merge_qkv_grads_gqa_f4(
    const float* __restrict__ grad_Q,       // [batch, num_heads, seq, head_dim]
    const float* __restrict__ grad_K,       // [batch, num_kv_heads, seq, head_dim]
    const float* __restrict__ grad_V,       // [batch, num_kv_heads, seq, head_dim]
    float* __restrict__ grad_qkv,           // [batch*seq, q_dim + 2*kv_dim]
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim)
{
    // Grid: (tokens, max(num_heads, num_kv_heads))
    // Block: (head_dim / 4)
    const int token = blockIdx.x;
    const int h = blockIdx.y;
    const int d4 = threadIdx.x;
    
    const int total_tokens = batch * seq;
    const int D4 = head_dim >> 2;
    
    if (token >= total_tokens || d4 >= D4) return;
    
    const int b = token / seq;
    const int s = token % seq;
    
    const int q_dim4 = num_heads * D4;
    const int kv_dim4 = num_kv_heads * D4;
    const int total_dim4 = q_dim4 + 2 * kv_dim4;
    
    float4* out4 = reinterpret_cast<float4*>(grad_qkv) + token * total_dim4;
    
    // Handle Q gradient (h < num_heads)
    if (h < num_heads) {
        const int q_idx4 = ((b * num_heads + h) * seq + s) * D4 + d4;
        const float4 qg = reinterpret_cast<const float4*>(grad_Q)[q_idx4];
        const int q_feature4 = h * D4 + d4;
        out4[q_feature4] = qg;
    }
    
    // Handle K, V gradients (h < num_kv_heads)
    if (h < num_kv_heads) {
        const int kv_idx4 = ((b * num_kv_heads + h) * seq + s) * D4 + d4;
        const float4 kg = reinterpret_cast<const float4*>(grad_K)[kv_idx4];
        const float4 vg = reinterpret_cast<const float4*>(grad_V)[kv_idx4];
        const int kv_feature4 = h * D4 + d4;
        out4[q_dim4 + kv_feature4] = kg;
        out4[q_dim4 + kv_dim4 + kv_feature4] = vg;
    }
}

// Scalar fallback GQA merge gradients - works for any head_dim
__global__ void kernel_merge_qkv_grads_gqa(
    const float* __restrict__ grad_Q,
    const float* __restrict__ grad_K,
    const float* __restrict__ grad_V,
    float* __restrict__ grad_qkv,
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch * seq;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int total_qkv_dim = q_dim + 2 * kv_dim;
    int total = total_tokens * total_qkv_dim;
    
    if (idx >= total) return;
    
    int feature = idx % total_qkv_dim;
    int token = idx / total_qkv_dim;
    int b = token / seq;
    int s = token % seq;
    
    float value;
    
    if (feature < q_dim) {
        // Q region
        int h = feature / head_dim;
        int d = feature % head_dim;
        int src_idx = ((b * num_heads + h) * seq + s) * head_dim + d;
        value = grad_Q[src_idx];
    } else if (feature < q_dim + kv_dim) {
        // K region
        int kv_feature = feature - q_dim;
        int h = kv_feature / head_dim;
        int d = kv_feature % head_dim;
        int src_idx = ((b * num_kv_heads + h) * seq + s) * head_dim + d;
        value = grad_K[src_idx];
    } else {
        // V region
        int kv_feature = feature - q_dim - kv_dim;
        int h = kv_feature / head_dim;
        int d = kv_feature % head_dim;
        int src_idx = ((b * num_kv_heads + h) * seq + s) * head_dim + d;
        value = grad_V[src_idx];
    }
    
    grad_qkv[idx] = value;
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

void convert_fp32_to_bf16(const float* src, __nv_bfloat16* dst,
                          std::size_t count,
                          cudaStream_t stream)
{
    if (!src) throw std::runtime_error("convert_fp32_to_bf16: src is NULL");
    if (!dst) throw std::runtime_error("convert_fp32_to_bf16: dst is NULL");
    if (count == 0) throw std::runtime_error("convert_fp32_to_bf16: count is zero");
    if (stream == nullptr) throw std::runtime_error("convert_fp32_to_bf16: stream is NULL - caller MUST provide valid CUDA stream");
    kernel_fp32_to_bf16<<<gridForCount(count), BLOCK_SIZE, 0, stream>>>(src, dst, count);
}

void convert_bf16_to_fp32(const __nv_bfloat16* src, float* dst,
                          std::size_t count,
                          cudaStream_t stream)
{
    if (!src) throw std::runtime_error("convert_bf16_to_fp32: src is NULL");
    if (!dst) throw std::runtime_error("convert_bf16_to_fp32: dst is NULL");
    if (count == 0) throw std::runtime_error("convert_bf16_to_fp32: count is zero");
    if (stream == nullptr) throw std::runtime_error("convert_bf16_to_fp32: stream is NULL - caller MUST provide valid CUDA stream");
    kernel_bf16_to_fp32<<<gridForCount(count), BLOCK_SIZE, 0, stream>>>(src, dst, count);
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

// ----------------------------------------------------------------------------
// GQA-Aware Split/Merge Host Wrappers
// These are the ONLY split/merge functions. For MHA: num_heads == num_kv_heads.
// ----------------------------------------------------------------------------

void split_qkv_gqa(
    const float* qkv_fused,    // [batch*seq, (num_heads + 2*num_kv_heads) * head_dim]
    float* Q,                   // [batch, num_heads, seq, head_dim]
    float* K,                   // [batch, num_kv_heads, seq, head_dim]
    float* V,                   // [batch, num_kv_heads, seq, head_dim]
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim,
    cudaStream_t stream)
{
    if (!qkv_fused) throw std::runtime_error("split_qkv_gqa: qkv_fused is NULL");
    if (!Q) throw std::runtime_error("split_qkv_gqa: Q is NULL");
    if (!K) throw std::runtime_error("split_qkv_gqa: K is NULL");
    if (!V) throw std::runtime_error("split_qkv_gqa: V is NULL");
    if (batch <= 0 || num_heads <= 0 || num_kv_heads <= 0 || seq <= 0 || head_dim <= 0)
        throw std::runtime_error("split_qkv_gqa: all dimensions must be > 0");

    if (head_dim % 4 == 0) {
        // Float4 vectorized path
        int max_heads = std::max(num_heads, num_kv_heads);
        dim3 grid(batch * seq, max_heads);
        dim3 block(head_dim / 4);
        kernel_split_qkv_gqa_f4<<<grid, block, 0, stream>>>(
            qkv_fused, Q, K, V, batch, num_heads, num_kv_heads, seq, head_dim);
    } else {
        // Scalar fallback
        int total_tokens = batch * seq;
        int q_dim = num_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;
        int total = total_tokens * (q_dim + 2 * kv_dim);
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_split_qkv_gqa<<<blocks, BLOCK_SIZE, 0, stream>>>(
            qkv_fused, Q, K, V, batch, num_heads, num_kv_heads, seq, head_dim);
    }
}

void merge_qkv_grads_gqa(
    const float* grad_Q,       // [batch, num_heads, seq, head_dim]
    const float* grad_K,       // [batch, num_kv_heads, seq, head_dim]
    const float* grad_V,       // [batch, num_kv_heads, seq, head_dim]
    float* grad_qkv,           // [batch*seq, (num_heads + 2*num_kv_heads) * head_dim]
    int batch, int num_heads, int num_kv_heads, int seq, int head_dim,
    cudaStream_t stream)
{
    if (!grad_Q) throw std::runtime_error("merge_qkv_grads_gqa: grad_Q is NULL");
    if (!grad_K) throw std::runtime_error("merge_qkv_grads_gqa: grad_K is NULL");
    if (!grad_V) throw std::runtime_error("merge_qkv_grads_gqa: grad_V is NULL");
    if (!grad_qkv) throw std::runtime_error("merge_qkv_grads_gqa: grad_qkv is NULL");
    if (batch <= 0 || num_heads <= 0 || num_kv_heads <= 0 || seq <= 0 || head_dim <= 0)
        throw std::runtime_error("merge_qkv_grads_gqa: all dimensions must be > 0");

    if (head_dim % 4 == 0) {
        // Float4 vectorized path
        int max_heads = std::max(num_heads, num_kv_heads);
        dim3 grid(batch * seq, max_heads);
        dim3 block(head_dim / 4);
        kernel_merge_qkv_grads_gqa_f4<<<grid, block, 0, stream>>>(
            grad_Q, grad_K, grad_V, grad_qkv, batch, num_heads, num_kv_heads, seq, head_dim);
    } else {
        // Scalar fallback
        int total_tokens = batch * seq;
        int q_dim = num_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;
        int total = total_tokens * (q_dim + 2 * kv_dim);
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_merge_qkv_grads_gqa<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_Q, grad_K, grad_V, grad_qkv, batch, num_heads, num_kv_heads, seq, head_dim);
    }
}

} // namespace TensorConversion
