//======================================================//
//  TensorContract_GPU.cu
//  CUDA implementation of type-safe tensor operations
//======================================================//

#include "TensorContract_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace TensorContract {

//======================================================//
//  Constants (from HyperParameters)
//======================================================//

constexpr int BLOCK_SIZE = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr int WARP_SIZE = GRIM::HyperParameters::CUDA_WARP_SIZE;

//======================================================//
//  CUDA Error Checking
//======================================================//

#define TC_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw ContractViolation(std::string("CUDA error in TensorContract: ") + cudaGetErrorString(err)); \
    } \
} while(0)

//======================================================//
//  TensorView Implementation
//======================================================//

std::string TensorView::to_string() const {
    std::ostringstream oss;
    oss << (name ? name : "tensor") << " [";
    
    if (shape.is_flat()) {
        const auto& s = shape.as_2d();
        oss << s.rows << ", " << s.cols;
    } else if (shape.is_4d()) {
        const auto& s = shape.as_4d();
        oss << s.batch << ", " << s.heads << ", " << s.seq << ", " << s.head_dim;
    } else {
        oss << "unknown";
    }
    
    oss << "] " << layout_name(shape.layout);
    oss << " @ " << static_cast<void*>(ptr);
    return oss.str();
}

//======================================================//
//  TensorBuffer Implementation
//======================================================//

TensorBuffer::~TensorBuffer() {
    free();
}

TensorBuffer::TensorBuffer(TensorBuffer&& other) noexcept
    : view_(other.view_), owns_memory_(other.owns_memory_) {
    other.view_ = TensorView();
    other.owns_memory_ = false;
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
    if (this != &other) {
        free();
        view_ = other.view_;
        owns_memory_ = other.owns_memory_;
        other.view_ = TensorView();
        other.owns_memory_ = false;
    }
    return *this;
}

bool TensorBuffer::allocate(TensorShape shape, const char* name) {
    free();
    
    if (!shape.is_valid()) {
        return false;
    }
    
    size_t bytes = shape.total_elements() * sizeof(float);
    float* ptr = nullptr;
    
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        return false;
    }
    
    view_ = TensorView(ptr, shape, name);
    owns_memory_ = true;
    return true;
}

void TensorBuffer::free() {
    if (owns_memory_ && view_.ptr) {
        cudaFree(view_.ptr);
    }
    view_ = TensorView();
    owns_memory_ = false;
}

//======================================================//
//  Validation Functions
//======================================================//

bool tensors_alias(const TensorView& a, const TensorView& b) {
    if (!a.ptr || !b.ptr) return false;
    
    const char* a_start = reinterpret_cast<const char*>(a.ptr);
    const char* a_end = a_start + a.size_bytes();
    const char* b_start = reinterpret_cast<const char*>(b.ptr);
    const char* b_end = b_start + b.size_bytes();
    
    // Check for overlap
    return !(a_end <= b_start || b_end <= a_start);
}

void validate_binary_op(const TensorView& a, const TensorView& b, const char* op_name) {
    if (!a.is_valid()) {
        throw ContractViolation(std::string(op_name) + ": first tensor is invalid");
    }
    if (!b.is_valid()) {
        throw ContractViolation(std::string(op_name) + ": second tensor is invalid");
    }
    if (a.size_elements() != b.size_elements()) {
        std::ostringstream oss;
        oss << op_name << ": tensor size mismatch (" 
            << a.size_elements() << " vs " << b.size_elements() << ")";
        throw ContractViolation(oss.str());
    }
}

void validate_conversion(const TensorView& src, const TensorView& dst, const char* op_name) {
    if (!src.is_valid()) {
        throw ContractViolation(std::string(op_name) + ": source tensor is invalid");
    }
    if (!dst.ptr) {
        throw ContractViolation(std::string(op_name) + ": destination pointer is null");
    }
    if (src.size_elements() != dst.size_elements()) {
        std::ostringstream oss;
        oss << op_name << ": element count mismatch (src=" 
            << src.size_elements() << ", dst=" << dst.size_elements() << ")";
        throw ContractViolation(oss.str());
    }
}

//======================================================//
//  Utility Functions
//======================================================//

const char* layout_name(Layout layout) {
    switch (layout) {
        case Layout::BSM:       return "BSM";
        case Layout::BHSD:      return "BHSD";
        case Layout::BHDS:      return "BHDS";
        case Layout::BSHD:      return "BSHD";
        case Layout::QKV_FUSED: return "QKV_FUSED";
        case Layout::LOGITS:    return "LOGITS";
        case Layout::UNKNOWN:   return "UNKNOWN";
        default:                return "INVALID";
    }
}

size_t compute_buffer_size(const TensorShape& shape) {
    return shape.total_elements() * sizeof(float);
}

bool is_conversion_supported(Layout from, Layout to) {
    if (from == to) return true;
    
    // Supported conversion pairs
    if ((from == Layout::BSM && to == Layout::BHSD) ||
        (from == Layout::BHSD && to == Layout::BSM) ||
        (from == Layout::BHSD && to == Layout::BHDS) ||
        (from == Layout::BHDS && to == Layout::BHSD) ||
        (from == Layout::BHSD && to == Layout::BSHD) ||
        (from == Layout::BSHD && to == Layout::BHSD)) {
        return true;
    }
    return false;
}

//======================================================//
//  CUDA Kernels - Basic Operations
//======================================================//

__global__ void kernel_zero(float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = 0.0f;
    }
}

__global__ void kernel_add(const float* __restrict__ a, 
                           const float* __restrict__ b,
                           float* __restrict__ dst,
                           size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_scale(const float* __restrict__ src,
                             float alpha,
                             float* __restrict__ dst,
                             size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = alpha * src[idx];
    }
}

//======================================================//
//  CUDA Kernels - Layout Conversions
//======================================================//

// BSM to BHSD: [tokens, d_model] -> [batch, heads, seq, head_dim]
__global__ void kernel_BSM_to_BHSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch, int seq, int heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = heads * head_dim;
    int tokens = batch * seq;
    int total = tokens * d_model;
    
    if (idx >= total) return;
    
    // Decode from BSM: [token, feature]
    int feature = idx % d_model;
    int token = idx / d_model;
    
    // Convert token to (b, s)
    int b = token / seq;
    int s = token % seq;
    
    // Convert feature to (h, d)
    int h = feature / head_dim;
    int d = feature % head_dim;
    
    // Write to BHSD: [b, h, s, d]
    int dst_idx = ((b * heads + h) * seq + s) * head_dim + d;
    dst[dst_idx] = src[idx];
}

// BHSD to BSM: [batch, heads, seq, head_dim] -> [tokens, d_model]
__global__ void kernel_BHSD_to_BSM(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch, int heads, int seq, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * head_dim;
    
    if (idx >= total) return;
    
    // Decode from BHSD
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq;
    int h = (idx / (head_dim * seq)) % heads;
    int b = idx / (head_dim * seq * heads);
    
    // Write to BSM: [b*seq + s, h*head_dim + d]
    int d_model = heads * head_dim;
    int token = b * seq + s;
    int feature = h * head_dim + d;
    int dst_idx = token * d_model + feature;
    
    dst[dst_idx] = src[idx];
}

// BHSD to BHDS: Transpose last two dimensions
__global__ void kernel_BHSD_to_BHDS(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch, int heads, int seq, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * head_dim;
    
    if (idx >= total) return;
    
    // Decode from BHSD
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq;
    int h = (idx / (head_dim * seq)) % heads;
    int b = idx / (head_dim * seq * heads);
    
    // Write to BHDS: [b, h, d, s]
    int dst_idx = ((b * heads + h) * head_dim + d) * seq + s;
    dst[dst_idx] = src[idx];
}

// BHDS to BHSD: Transpose last two dimensions
__global__ void kernel_BHDS_to_BHSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch, int heads, int head_dim, int seq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * head_dim * seq;
    
    if (idx >= total) return;
    
    // Decode from BHDS
    int s = idx % seq;
    int d = (idx / seq) % head_dim;
    int h = (idx / (seq * head_dim)) % heads;
    int b = idx / (seq * head_dim * heads);
    
    // Write to BHSD: [b, h, s, d]
    int dst_idx = ((b * heads + h) * seq + s) * head_dim + d;
    dst[dst_idx] = src[idx];
}

// BHSD to BSHD: Swap heads and seq dimensions
__global__ void kernel_BHSD_to_BSHD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch, int heads, int seq, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * head_dim;
    
    if (idx >= total) return;
    
    // Decode from BHSD
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq;
    int h = (idx / (head_dim * seq)) % heads;
    int b = idx / (head_dim * seq * heads);
    
    // Write to BSHD: [b, s, h, d]
    int dst_idx = ((b * seq + s) * heads + h) * head_dim + d;
    dst[dst_idx] = src[idx];
}

// BSHD to BHSD: Swap seq and heads dimensions
__global__ void kernel_BSHD_to_BHSD(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch, int seq, int heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * heads * head_dim;
    
    if (idx >= total) return;
    
    // Decode from BSHD
    int d = idx % head_dim;
    int h = (idx / head_dim) % heads;
    int s = (idx / (head_dim * heads)) % seq;
    int b = idx / (head_dim * heads * seq);
    
    // Write to BHSD: [b, h, s, d]
    int dst_idx = ((b * heads + h) * seq + s) * head_dim + d;
    dst[dst_idx] = src[idx];
}

//======================================================//
//  CUDA Kernels - GQA Operations
//======================================================//

// Split fused QKV into separate Q, K, V tensors (GQA-aware)
// Input:  [batch*seq, q_dim + 2*kv_dim] where q_dim = num_heads * head_dim, kv_dim = num_kv_heads * head_dim
// Output: Q [batch, num_heads, seq, head_dim], K [batch, num_kv_heads, seq, head_dim], V [batch, num_kv_heads, seq, head_dim]
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

// Scalar fallback for non-vectorizable dimensions
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

// Pack Q (num_heads), K, V (num_kv_heads) gradients into fused format
// Uses float4 vectorized access for better memory bandwidth
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

// Scalar fallback for non-vectorizable dimensions
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

//======================================================//
//  Host API Implementation - Basic Operations
//======================================================//

void zero(TensorView& tensor, cudaStream_t stream) {
    if (!tensor.is_valid()) {
        throw ContractViolation("zero: tensor is invalid");
    }
    
    size_t n = tensor.size_elements();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_zero<<<blocks, BLOCK_SIZE, 0, stream>>>(tensor.ptr, n);
    TC_CUDA_CHECK(cudaGetLastError());
}

void copy(const TensorView& src, TensorView& dst, cudaStream_t stream) {
    validate_conversion(src, dst, "copy");
    
    if (src.layout() != dst.layout()) {
        throw ContractViolation("copy: layout mismatch (use convert for layout changes)");
    }
    
    TC_CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr, src.size_bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
}

void add(const TensorView& a, const TensorView& b, TensorView& dst, cudaStream_t stream) {
    validate_binary_op(a, b, "add");
    
    if (dst.size_elements() != a.size_elements()) {
        throw ContractViolation("add: output size mismatch");
    }
    
    size_t n = a.size_elements();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_add<<<blocks, BLOCK_SIZE, 0, stream>>>(a.ptr, b.ptr, dst.ptr, n);
    TC_CUDA_CHECK(cudaGetLastError());
}

void scale(const TensorView& src, float alpha, TensorView& dst, cudaStream_t stream) {
    validate_conversion(src, dst, "scale");
    
    size_t n = src.size_elements();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_scale<<<blocks, BLOCK_SIZE, 0, stream>>>(src.ptr, alpha, dst.ptr, n);
    TC_CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  Host API Implementation - Layout Conversions
//======================================================//

void convert(const TensorView& src, TensorView& dst, cudaStream_t stream) {
    validate_conversion(src, dst, "convert");
    
    const Layout src_layout = src.layout();
    const Layout dst_layout = dst.layout();
    
    if (!is_conversion_supported(src_layout, dst_layout)) {
        std::ostringstream oss;
        oss << "convert: unsupported conversion from " << layout_name(src_layout)
            << " to " << layout_name(dst_layout);
        throw ContractViolation(oss.str());
    }
    
    // Same layout = just copy
    if (src_layout == dst_layout) {
        TC_CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr, src.size_bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
        return;
    }
    
    // BSM <-> BHSD requires the destination to provide 4D shape info
    // since BSM is 2D and doesn't carry batch/heads/seq/head_dim info
    int batch = 0, heads = 0, seq = 0, head_dim = 0;
    
    if (src.is_4d()) {
        const auto& s = src.shape.as_4d();
        batch = s.batch; heads = s.heads; seq = s.seq; head_dim = s.head_dim;
    } else if (dst.is_4d()) {
        const auto& s = dst.shape.as_4d();
        batch = s.batch; heads = s.heads; seq = s.seq; head_dim = s.head_dim;
    } else {
        // Both 2D - this shouldn't happen for valid conversions
        throw ContractViolation("convert: cannot determine 4D dimensions");
    }
    
    const size_t total = src.size_elements();
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Dispatch to appropriate kernel
    if (src_layout == Layout::BSM && dst_layout == Layout::BHSD) {
        kernel_BSM_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(
            src.ptr, dst.ptr, batch, seq, heads, head_dim);
    }
    else if (src_layout == Layout::BHSD && dst_layout == Layout::BSM) {
        kernel_BHSD_to_BSM<<<blocks, BLOCK_SIZE, 0, stream>>>(
            src.ptr, dst.ptr, batch, heads, seq, head_dim);
    }
    else if (src_layout == Layout::BHSD && dst_layout == Layout::BHDS) {
        kernel_BHSD_to_BHDS<<<blocks, BLOCK_SIZE, 0, stream>>>(
            src.ptr, dst.ptr, batch, heads, seq, head_dim);
    }
    else if (src_layout == Layout::BHDS && dst_layout == Layout::BHSD) {
        kernel_BHDS_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(
            src.ptr, dst.ptr, batch, heads, head_dim, seq);
    }
    else if (src_layout == Layout::BHSD && dst_layout == Layout::BSHD) {
        kernel_BHSD_to_BSHD<<<blocks, BLOCK_SIZE, 0, stream>>>(
            src.ptr, dst.ptr, batch, heads, seq, head_dim);
    }
    else if (src_layout == Layout::BSHD && dst_layout == Layout::BHSD) {
        kernel_BSHD_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>(
            src.ptr, dst.ptr, batch, seq, heads, head_dim);
    }
    else {
        throw ContractViolation("convert: internal error - unhandled conversion");
    }
    
    TC_CUDA_CHECK(cudaGetLastError());
}

bool can_convert_inplace(const TensorView& tensor, Layout target_layout) {
    // In-place conversion is NOT safe for layout changes that require element reordering.
    // Only same-layout is safe (trivially - no conversion needed).
    return tensor.layout() == target_layout;
}

bool convert_inplace(TensorView& tensor, Layout target_layout, cudaStream_t /*stream*/) {
    // Only "conversion" that works in-place is no conversion at all
    return tensor.layout() == target_layout;
}

//======================================================//
//  Host API Implementation - GQA Operations
//======================================================//

void split_qkv_gqa(const TensorView& qkv_fused,
                   TensorView& Q, TensorView& K, TensorView& V,
                   const GQADims& gqa, Layout target_layout,
                   cudaStream_t stream) {
    if (!gqa.is_valid()) {
        throw ContractViolation("split_qkv_gqa: invalid GQA dimensions");
    }
    if (!qkv_fused.is_valid()) {
        throw ContractViolation("split_qkv_gqa: invalid input tensor");
    }
    
    // Verify input is QKV_FUSED layout
    if (qkv_fused.layout() != Layout::QKV_FUSED && qkv_fused.layout() != Layout::BSM) {
        throw ContractViolation("split_qkv_gqa: input must be QKV_FUSED or BSM layout");
    }
    
    // Get dimensions from input
    if (!qkv_fused.is_flat()) {
        throw ContractViolation("split_qkv_gqa: input must be 2D (tokens, total_qkv_dim)");
    }
    
    const auto& in_shape = qkv_fused.shape.as_2d();
    const int total_tokens = in_shape.rows;
    const int expected_qkv_dim = gqa.total_qkv_dim();
    
    if (in_shape.cols != expected_qkv_dim) {
        std::ostringstream oss;
        oss << "split_qkv_gqa: input dim mismatch (got " << in_shape.cols 
            << ", expected " << expected_qkv_dim << ")";
        throw ContractViolation(oss.str());
    }
    
    // Verify output pointers
    if (!Q.ptr || !K.ptr || !V.ptr) {
        throw ContractViolation("split_qkv_gqa: output pointer(s) are null");
    }
    
    // REQUIRE output Q to have valid 4D shape - we use it to infer batch/seq
    // This is safer than assuming batch=1 which is often wrong
    if (!Q.is_4d()) {
        throw ContractViolation("split_qkv_gqa: output Q must have 4D shape (BHSD) to infer batch/seq");
    }
    
    const auto& q_shape = Q.shape.as_4d();
    const int batch = q_shape.batch;
    const int seq = q_shape.seq;
    
    if (batch * seq != total_tokens) {
        std::ostringstream oss;
        oss << "split_qkv_gqa: Q shape (batch=" << batch << " * seq=" << seq 
            << " = " << (batch * seq) << ") doesn't match input tokens (" << total_tokens << ")";
        throw ContractViolation(oss.str());
    }
    
    const int head_dim = gqa.head_dim;
    const int num_heads = gqa.num_heads;
    const int num_kv_heads = gqa.num_kv_heads;
    
    // Use vectorized kernel if head_dim is multiple of 4
    if (head_dim % 4 == 0) {
        const int max_heads = std::max(num_heads, num_kv_heads);
        dim3 grid(total_tokens, max_heads);
        dim3 block(head_dim >> 2);
        
        kernel_split_qkv_gqa_f4<<<grid, block, 0, stream>>>(
            qkv_fused.ptr, Q.ptr, K.ptr, V.ptr,
            batch, num_heads, num_kv_heads, seq, head_dim);
    } else {
        // Scalar fallback
        const size_t total = static_cast<size_t>(total_tokens) * expected_qkv_dim;
        const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        kernel_split_qkv_gqa<<<blocks, BLOCK_SIZE, 0, stream>>>(
            qkv_fused.ptr, Q.ptr, K.ptr, V.ptr,
            batch, num_heads, num_kv_heads, seq, head_dim);
    }
    
    TC_CUDA_CHECK(cudaGetLastError());
}

void merge_qkv_grads_gqa(const TensorView& grad_Q, const TensorView& grad_K, const TensorView& grad_V,
                         TensorView& grad_qkv, const GQADims& gqa,
                         cudaStream_t stream) {
    if (!gqa.is_valid()) {
        throw ContractViolation("merge_qkv_grads_gqa: invalid GQA dimensions");
    }
    if (!grad_Q.is_valid() || !grad_K.is_valid() || !grad_V.is_valid()) {
        throw ContractViolation("merge_qkv_grads_gqa: invalid input tensor(s)");
    }
    if (!grad_qkv.ptr) {
        throw ContractViolation("merge_qkv_grads_gqa: output pointer is null");
    }
    
    // grad_Q must be 4D (BHSD)
    if (!grad_Q.is_4d()) {
        throw ContractViolation("merge_qkv_grads_gqa: grad_Q must be 4D (BHSD)");
    }
    
    const auto& q_shape = grad_Q.shape.as_4d();
    const int batch = q_shape.batch;
    const int seq = q_shape.seq;
    const int head_dim = gqa.head_dim;
    const int num_heads = gqa.num_heads;
    const int num_kv_heads = gqa.num_kv_heads;
    
    // Use vectorized kernel if head_dim is multiple of 4
    if (head_dim % 4 == 0) {
        const int max_heads = std::max(num_heads, num_kv_heads);
        dim3 grid(batch * seq, max_heads);
        dim3 block(head_dim >> 2);
        
        kernel_merge_qkv_grads_gqa_f4<<<grid, block, 0, stream>>>(
            grad_Q.ptr, grad_K.ptr, grad_V.ptr, grad_qkv.ptr,
            batch, num_heads, num_kv_heads, seq, head_dim);
    } else {
        // Scalar fallback
        const int total_tokens = batch * seq;
        const int total_qkv_dim = gqa.total_qkv_dim();
        const size_t total = total_tokens * total_qkv_dim;
        const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        kernel_merge_qkv_grads_gqa<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_Q.ptr, grad_K.ptr, grad_V.ptr, grad_qkv.ptr,
            batch, num_heads, num_kv_heads, seq, head_dim);
    }
    
    TC_CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  Debug Utilities
//======================================================//

#ifndef NDEBUG

void debug_print_stats(const TensorView& tensor, const char* label, cudaStream_t stream) {
    if (!tensor.is_valid()) {
        printf("[TensorContract] %s: INVALID TENSOR\n", label);
        return;
    }
    
    // Sample a subset for performance (CPU-side for simplicity)
    const size_t sample_size = std::min<size_t>(tensor.size_elements(), 1024);
    std::vector<float> samples(sample_size);
    
    cudaMemcpyAsync(samples.data(), tensor.ptr, sample_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    float sum_sq = 0.0f;
    float max_abs = 0.0f;
    int nan_count = 0, inf_count = 0;
    
    for (float v : samples) {
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else {
            sum_sq += v * v;
            max_abs = std::max(max_abs, std::abs(v));
        }
    }
    
    float rms = std::sqrt(sum_sq / sample_size);
    
    printf("[TensorContract] %s: rms=%.6f max_abs=%.6f", label, rms, max_abs);
    if (nan_count > 0 || inf_count > 0) {
        printf(" ⚠ NaN=%d Inf=%d", nan_count, inf_count);
    }
    printf(" (%s)\n", tensor.to_string().c_str());
}

bool debug_check_finite(const TensorView& tensor, cudaStream_t stream) {
    if (!tensor.is_valid()) return false;
    
    // Quick sample check
    const size_t sample_size = std::min<size_t>(tensor.size_elements(), 256);
    std::vector<float> samples(sample_size);
    
    cudaMemcpyAsync(samples.data(), tensor.ptr, sample_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    for (float v : samples) {
        if (std::isnan(v) || std::isinf(v)) return false;
    }
    return true;
}

#endif  // NDEBUG

}  // namespace TensorContract

//======================================================//
//======================================================//
//  GRIM NATIVE AUTOGRAD IMPLEMENTATION
//======================================================//
//======================================================//

namespace GRIM {

//======================================================//
//  CUDA Kernels for Tensor Operations
//======================================================//

namespace {

// Kernel: Zero-initialize tensor
__global__ void kernel_zero(float* data, size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = 0.0f;
    }
}

// Kernel: Xavier uniform initialization (uses curand-free LCG for reproducibility)
__global__ void kernel_xavier_uniform(float* data, size_t count, float scale, uint64_t seed) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        // LCG PRNG: x_{n+1} = (a * x_n + c) mod m
        // Parameters from Numerical Recipes
        uint64_t state = seed + idx * 6364136223846793005ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        
        // Convert to uniform [0, 1)
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        
        // Transform to [-scale, +scale]
        data[idx] = (2.0f * u - 1.0f) * scale;
    }
}

// Kernel: Accumulate gradient (dst += src * scale)
__global__ void kernel_accumulate_grad(float* dst, const float* src, size_t count, float scale) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] += src[idx] * scale;
    }
}

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

}  // anonymous namespace

//======================================================//
//  Tensor Move Constructor/Assignment
//======================================================//

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data)
    , shape(other.shape)
    , owns_data(other.owns_data)
    , grad(other.grad)
    , owns_grad(other.owns_grad)
    , grad_fn(other.grad_fn)
    , requires_grad(other.requires_grad)
    , is_leaf(other.is_leaf)
    , retain_grad(other.retain_grad)
    , stream(other.stream)
    , device_id(other.device_id)
    , name(other.name)
    , version(other.version)
{
    // Null out other's pointers to prevent double-free
    other.data = nullptr;
    other.grad = nullptr;
    other.grad_fn = nullptr;
    other.owns_data = false;
    other.owns_grad = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Release current resources
        release();
        
        // Move from other
        data = other.data;
        shape = other.shape;
        owns_data = other.owns_data;
        grad = other.grad;
        owns_grad = other.owns_grad;
        grad_fn = other.grad_fn;
        requires_grad = other.requires_grad;
        is_leaf = other.is_leaf;
        retain_grad = other.retain_grad;
        stream = other.stream;
        device_id = other.device_id;
        name = other.name;
        version = other.version;
        
        // Null out other's pointers
        other.data = nullptr;
        other.grad = nullptr;
        other.grad_fn = nullptr;
        other.owns_data = false;
        other.owns_grad = false;
    }
    return *this;
}

//======================================================//
//  Tensor Factory Methods
//======================================================//

Tensor Tensor::zeros(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream) {
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::zeros: invalid shape");
    }
    
    Tensor t;
    t.shape = shape;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    t.stream = stream;
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    cudaError_t err = cudaMalloc(&t.data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::zeros cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    t.owns_data = true;
    
    // Zero-initialize
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_zero<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(t.data, count);
    
    return t;
}

Tensor Tensor::zeros(std::initializer_list<int> dims, cudaStream_t stream) {
    // Convert raw dimensions to TensorShape with default layouts
    std::vector<int> d(dims);
    TensorContract::TensorShape shape;
    
    if (d.size() == 1) {
        // 1D -> BSM [1, dim]
        shape = TensorContract::TensorShape::make_BSM(1, d[0]);
    } else if (d.size() == 2) {
        // 2D -> BSM [tokens, features]
        shape = TensorContract::TensorShape::make_BSM(d[0], d[1]);
    } else if (d.size() == 3) {
        // 3D -> BSM with flattened first two dims [d0*d1, d2]
        shape = TensorContract::TensorShape::make_BSM(d[0] * d[1], d[2]);
    } else if (d.size() == 4) {
        // 4D -> BHSD [batch, heads, seq, head_dim]
        shape = TensorContract::TensorShape::make_BHSD(d[0], d[1], d[2], d[3]);
    } else {
        throw std::invalid_argument("Tensor::zeros: unsupported dimension count " + std::to_string(d.size()));
    }
    
    // requires_grad defaults to false for this convenience overload
    return zeros(shape, false, stream);
}

Tensor Tensor::empty(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream) {
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::empty: invalid shape");
    }
    
    Tensor t;
    t.shape = shape;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    t.stream = stream;
    
    const size_t bytes = shape.total_elements() * sizeof(float);
    
    cudaError_t err = cudaMalloc(&t.data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::empty cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    t.owns_data = true;
    
    // NOTE: Memory is NOT initialized (undefined values)
    return t;
}

Tensor Tensor::from_ptr(float* ptr, TensorContract::TensorShape shape, bool takes_ownership, bool requires_grad) {
    if (!ptr) {
        throw std::invalid_argument("Tensor::from_ptr: null pointer");
    }
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::from_ptr: invalid shape");
    }
    
    Tensor t;
    t.data = ptr;
    t.shape = shape;
    t.owns_data = takes_ownership;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    
    return t;
}

Tensor Tensor::from_ptr(float* ptr, std::initializer_list<int> dims, cudaStream_t stream) {
    std::vector<int> d(dims);
    TensorContract::TensorShape shape;
    
    if (d.size() == 1) {
        shape = TensorContract::TensorShape::make_BSM(1, d[0]);
    } else if (d.size() == 2) {
        shape = TensorContract::TensorShape::make_BSM(d[0], d[1]);
    } else if (d.size() == 3) {
        shape = TensorContract::TensorShape::make_BSM(d[0] * d[1], d[2]);
    } else if (d.size() == 4) {
        shape = TensorContract::TensorShape::make_BHSD(d[0], d[1], d[2], d[3]);
    } else {
        throw std::invalid_argument("Tensor::from_ptr: unsupported dimension count " + std::to_string(d.size()));
    }
    
    Tensor t = from_ptr(ptr, shape, false, false);
    t.stream = stream;
    return t;
}

Tensor Tensor::xavier_uniform(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream) {
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::xavier_uniform: invalid shape");
    }
    
    Tensor t;
    t.shape = shape;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    t.stream = stream;
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    cudaError_t err = cudaMalloc(&t.data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::xavier_uniform cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    t.owns_data = true;
    
    // Xavier uniform: U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]
    // For 2D [rows, cols]: fan_in=cols, fan_out=rows
    // For 4D [B,H,S,D]: treat as [B*H*S, D] => fan_in=D, fan_out=B*H*S
    float fan_in = 1.0f, fan_out = 1.0f;
    if (shape.is_flat()) {
        const auto& s = shape.as_2d();
        fan_in = static_cast<float>(s.cols);
        fan_out = static_cast<float>(s.rows);
    } else if (shape.is_4d()) {
        const auto& s = shape.as_4d();
        fan_in = static_cast<float>(s.head_dim);
        fan_out = static_cast<float>(s.batch * s.heads * s.seq);
    }
    
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    
    // Use current time as seed for reproducibility across runs
    uint64_t seed = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.data));
    
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_xavier_uniform<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(t.data, count, scale, seed);
    
    return t;
}

void Tensor::xavier_uniform_(Tensor& t, cudaStream_t stream) {
    if (!t.data) {
        throw std::invalid_argument("Tensor::xavier_uniform_: tensor has no data");
    }
    if (!t.shape.is_valid()) {
        throw std::invalid_argument("Tensor::xavier_uniform_: tensor has invalid shape");
    }
    
    const size_t count = t.shape.total_elements();
    
    // Xavier uniform calculation
    float fan_in = 1.0f, fan_out = 1.0f;
    if (t.shape.is_flat()) {
        const auto& s = t.shape.as_2d();
        fan_in = static_cast<float>(s.cols);
        fan_out = static_cast<float>(s.rows);
    } else if (t.shape.is_4d()) {
        const auto& s = t.shape.as_4d();
        fan_in = static_cast<float>(s.head_dim);
        fan_out = static_cast<float>(s.batch * s.heads * s.seq);
    }
    
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    uint64_t seed = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.data));
    
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_xavier_uniform<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(t.data, count, scale, seed);
    
    t.version++;
}

//======================================================//
//  Gradient Management
//======================================================//

void Tensor::ensure_grad() {
    if (!requires_grad) {
        return;  // No gradient tracking needed
    }
    
    if (grad != nullptr) {
        return;  // Already allocated
    }
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    cudaError_t err = cudaMalloc(&grad, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::ensure_grad cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    owns_grad = true;
    
    // Zero-initialize gradient
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_zero<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(grad, count);
}

void Tensor::zero_grad(cudaStream_t exec_stream) {
    if (grad == nullptr) {
        return;  // Nothing to zero
    }
    
    cudaStream_t s = exec_stream ? exec_stream : stream;
    const size_t count = shape.total_elements();
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_zero<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, s>>>(grad, count);
}

void Tensor::accumulate_grad(const float* incoming_grad, size_t count, float scale, cudaStream_t exec_stream) {
    if (!requires_grad) {
        return;  // Not tracking gradients
    }
    
    // Ensure gradient buffer exists
    ensure_grad();
    
    // Validate size matches
    if (count != shape.total_elements()) {
        throw std::invalid_argument("Tensor::accumulate_grad: size mismatch");
    }
    
    cudaStream_t s = exec_stream ? exec_stream : stream;
    
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, s>>>(grad, incoming_grad, count, scale);
}

Tensor Tensor::detach() const {
    Tensor t;
    t.data = data;
    t.shape = shape;
    t.owns_data = false;  // Non-owning view
    t.grad = nullptr;
    t.owns_grad = false;
    t.grad_fn = nullptr;
    t.requires_grad = false;  // Detached = no gradient tracking
    t.is_leaf = true;
    t.retain_grad = false;
    t.stream = stream;
    t.device_id = device_id;
    t.name = name;
    t.version = version;
    
    return t;
}

//======================================================//
//  Backward Pass
//======================================================//

void Tensor::backward(const Tensor* grad_output) {
    if (!requires_grad) {
        throw std::runtime_error("Tensor::backward called on tensor that doesn't require grad");
    }
    
    // Initialize gradient if this is the starting point (loss tensor)
    if (grad_output == nullptr) {
        // Default: scalar 1.0 gradient
        ensure_grad();
        const size_t count = shape.total_elements();
        
        // For scalar loss, set grad to 1.0
        // For non-scalar, this should be provided explicitly
        if (count == 1) {
            float one = 1.0f;
            cudaMemcpyAsync(grad, &one, sizeof(float), cudaMemcpyHostToDevice, stream);
        } else {
            // Fill with ones (implicit broadcast for reduction operations)
            std::vector<float> ones(count, 1.0f);
            cudaMemcpyAsync(grad, ones.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    } else {
        // Accumulate provided gradient
        accumulate_grad(grad_output->data, grad_output->numel(), 1.0f, stream);
    }
    
    // Traverse backward through computation graph
    if (grad_fn != nullptr) {
        // Execute backward function
        Tensor grad_tensor;
        grad_tensor.data = grad;
        grad_tensor.shape = shape;
        grad_tensor.owns_data = false;  // grad_tensor is a view
        
        grad_fn->apply(grad_tensor, stream);
        
        // Release saved tensors after backward
        grad_fn->release_saved();
    }
    
    // If non-leaf and not retaining grad, free gradient memory
    if (!is_leaf && !retain_grad && owns_grad && grad) {
        cudaFree(grad);
        grad = nullptr;
        owns_grad = false;
    }
}

//======================================================//
//======================================================//
//  GradFn SUBCLASSES - Backward Function Nodes
//======================================================//
//======================================================//

//======================================================//
//  CUDA Kernels for Forward + Backward Operations
//======================================================//

namespace {

// GELU forward: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void kernel_gelu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float x = input[idx];
        const float c = 0.7978845608f;  // sqrt(2/pi)
        const float k = 0.044715f;
        
        const float x3 = x * x * x;
        const float inner = c * (x + k * x3);
        const float tanh_inner = tanhf(inner);
        
        output[idx] = 0.5f * x * (1.0f + tanh_inner);
    }
}

// RMSNorm forward: y = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
// Each block processes one token (row)
__global__ void kernel_rmsnorm_forward(
    const float* __restrict__ input,    // [tokens, d_model]
    const float* __restrict__ gamma,    // [d_model]
    float* __restrict__ output,         // [tokens, d_model]
    int tokens,
    int d_model,
    float eps
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    extern __shared__ float shared[];
    
    const float* x = input + static_cast<size_t>(token_idx) * d_model;
    float* y = output + static_cast<size_t>(token_idx) * d_model;
    
    // Step 1: Compute sum of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum_sq += x[i] * x[i];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    // Block reduction via shared memory
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = local_sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_sum_sq = shared[threadIdx.x];
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
    }
    
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float rms_sq = local_sum_sq / d_model + eps;
        s_inv_rms = rsqrtf(rms_sq);
    }
    __syncthreads();
    
    const float inv_rms = s_inv_rms;
    
    // Step 2: Normalize and scale
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

// GELU backward: grad_x = grad_y * gelu'(x)
// gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// gelu'(x) = 0.5 * (1 + tanh(c*(x+0.044715*x^3))) + 0.5 * x * sech^2(c*(x+0.044715*x^3)) * c * (1 + 3*0.044715*x^2)
// where c = sqrt(2/pi) ≈ 0.7978845608
__global__ void kernel_gelu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    size_t count
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float x = input[idx];
        const float c = 0.7978845608f;  // sqrt(2/pi)
        const float k = 0.044715f;
        
        const float x3 = x * x * x;
        const float inner = c * (x + k * x3);
        const float tanh_inner = tanhf(inner);
        const float sech2 = 1.0f - tanh_inner * tanh_inner;  // sech^2 = 1 - tanh^2
        
        // gelu'(x) = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * c * (1 + 3*k*x^2)
        const float dgelu = 0.5f * (1.0f + tanh_inner) + 
                           0.5f * x * sech2 * c * (1.0f + 3.0f * k * x * x);
        
        grad_input[idx] = grad_output[idx] * dgelu;
    }
}

// Element-wise add backward: grad_a = grad_out, grad_b = grad_out
// (Gradient just passes through to both inputs)
__global__ void kernel_add_backward(
    const float* grad_output,
    float* grad_a,
    float* grad_b,
    size_t count
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float g = grad_output[idx];
        grad_a[idx] += g;  // Accumulate (not overwrite)
        grad_b[idx] += g;
    }
}

// RMSNorm backward kernel
// Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
// Backward: Complex due to normalization across dimension
__global__ void kernel_rmsnorm_backward(
    const float* grad_output,   // [tokens, d_model]
    const float* input,         // [tokens, d_model]
    const float* gamma,         // [d_model]
    float* grad_input,          // [tokens, d_model]
    float* grad_gamma,          // [d_model] - accumulated
    int tokens,
    int d_model,
    float eps
) {
    // Each block handles one token
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    extern __shared__ float shared[];
    float* s_sum_sq = shared;  // For reduction
    
    const float* x = input + static_cast<size_t>(token_idx) * d_model;
    const float* dy = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* dx = grad_input + static_cast<size_t>(token_idx) * d_model;
    
    // Step 1: Compute mean(x^2)
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum_sq += x[i] * x[i];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    // Block reduction via shared memory
    if (threadIdx.x % 32 == 0) {
        s_sum_sq[threadIdx.x / 32] = local_sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_sum_sq = s_sum_sq[threadIdx.x];
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
    }
    
    __shared__ float s_rms_sq, s_inv_rms;
    if (threadIdx.x == 0) {
        s_rms_sq = local_sum_sq / d_model + eps;
        s_inv_rms = rsqrtf(s_rms_sq);
    }
    __syncthreads();
    
    const float inv_rms = s_inv_rms;
    const float rms_sq = s_rms_sq;
    
    // Step 2: Compute sum(dy * gamma * x) for the chain rule term
    float local_dgamma_x = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_dgamma_x += dy[i] * gamma[i] * x[i];
    }
    
    // Reduce dgamma_x
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dgamma_x += __shfl_down_sync(0xffffffff, local_dgamma_x, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        s_sum_sq[threadIdx.x / 32] = local_dgamma_x;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_dgamma_x = s_sum_sq[threadIdx.x];
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            local_dgamma_x += __shfl_down_sync(0xffffffff, local_dgamma_x, offset);
        }
    }
    
    __shared__ float s_dgamma_x;
    if (threadIdx.x == 0) {
        s_dgamma_x = local_dgamma_x;
    }
    __syncthreads();
    
    const float dgamma_x_sum = s_dgamma_x;
    
    // Step 3: Compute grad_input and accumulate grad_gamma
    // dx = (dy * gamma - x * dgamma_x_sum / (d_model * rms_sq)) * inv_rms
    const float scale = dgamma_x_sum / (d_model * rms_sq);
    
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        dx[i] = (dy[i] * gamma[i] - x[i] * scale) * inv_rms;
        
        // Accumulate grad_gamma: dgamma[i] += dy[i] * x[i] * inv_rms
        atomicAdd(&grad_gamma[i], dy[i] * x[i] * inv_rms);
    }
}

// Cross-entropy backward: grad_logits = softmax(logits) - one_hot(targets)
__global__ void kernel_cross_entropy_backward(
    const float* logits,        // [tokens, vocab_size]
    const int* targets,         // [tokens]
    float* grad_logits,         // [tokens, vocab_size]
    int tokens,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const float* token_logits = logits + static_cast<size_t>(token_idx) * vocab_size;
    float* token_grad = grad_logits + static_cast<size_t>(token_idx) * vocab_size;
    const int target = targets[token_idx];
    
    extern __shared__ float shared[];
    
    // Step 1: Find max for numerical stability
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, token_logits[i]);
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = local_max;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_max = shared[threadIdx.x];
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
    }
    
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    
    // Step 2: Compute sum(exp)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf(token_logits[i] - s_max);
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = local_sum;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_sum = shared[threadIdx.x];
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    
    // Step 3: Compute grad = softmax - one_hot
    const float inv_sum = 1.0f / s_sum;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float softmax_i = expf(token_logits[i] - s_max) * inv_sum;
        float one_hot_i = (i == target) ? 1.0f : 0.0f;
        token_grad[i] = softmax_i - one_hot_i;
    }
}

// Embedding backward: scatter-add gradients to embedding table
__global__ void kernel_embedding_backward(
    const float* grad_output,   // [tokens, d_model]
    const int* token_ids,       // [tokens]
    float* grad_weight,         // [vocab_size, d_model]
    int tokens,
    int d_model
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const int token_id = token_ids[token_idx];
    const float* token_grad = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* weight_grad = grad_weight + static_cast<size_t>(token_id) * d_model;
    
    // Scatter-add: weight_grad[token_id] += grad_output[token_idx]
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        atomicAdd(&weight_grad[i], token_grad[i]);
    }
}

// Softmax backward: grad_x = softmax(x) * (grad_y - sum(grad_y * softmax(x)))
// Input: saved softmax output (not original input!)
// This is the efficient form that reuses softmax output
__global__ void kernel_softmax_backward(
    const float* grad_output,   // [tokens, dim]
    const float* softmax_out,   // [tokens, dim] - saved from forward
    float* grad_input,          // [tokens, dim]
    int tokens,
    int dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const float* dy = grad_output + static_cast<size_t>(token_idx) * dim;
    const float* y = softmax_out + static_cast<size_t>(token_idx) * dim;
    float* dx = grad_input + static_cast<size_t>(token_idx) * dim;
    
    extern __shared__ float shared[];
    
    // Step 1: Compute dot(grad_y, softmax_y)
    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_dot += dy[i] * y[i];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
    }
    
    // Block reduction
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = local_dot;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_dot = shared[threadIdx.x];
        for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
            local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
        }
    }
    
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = local_dot;
    __syncthreads();
    
    // Step 2: grad_x = softmax * (grad_y - dot)
    const float dot_sum = s_dot;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dx[i] = y[i] * (dy[i] - dot_sum);
    }
}

// Dropout backward: grad_x = grad_y * mask / (1 - p)
// mask is 0 where dropped, 1 where kept
__global__ void kernel_dropout_backward(
    const float* grad_output,
    const uint8_t* mask,        // Binary mask from forward
    float* grad_input,
    float scale,                // 1.0 / (1.0 - dropout_prob)
    size_t count
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] = grad_output[idx] * (mask[idx] ? scale : 0.0f);
    }
}

}  // anonymous namespace

//======================================================//
//  GradFn Subclasses
//======================================================//

/**
 * AddGradFn - Backward for element-wise addition
 * Forward: c = a + b
 * Backward: grad_a += grad_c, grad_b += grad_c
 */
struct AddGradFn : public GradFn {
    Tensor* input_a = nullptr;  // Non-owning pointers to inputs
    Tensor* input_b = nullptr;
    
    AddGradFn() { op_name = "add"; }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        const size_t count = grad_output.numel();
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        // Accumulate gradients to both inputs
        if (input_a && input_a->requires_grad) {
            input_a->accumulate_grad(grad_output.data, count, 1.0f, stream);
        }
        if (input_b && input_b->requires_grad) {
            input_b->accumulate_grad(grad_output.data, count, 1.0f, stream);
        }
    }
};

/**
 * GeluGradFn - Backward for GELU activation
 * Forward: y = gelu(x)
 * Backward: grad_x = grad_y * gelu'(x)
 */
struct GeluGradFn : public GradFn {
    Tensor* input = nullptr;    // Non-owning pointer
    float* saved_input = nullptr;  // Owned copy of input data
    size_t saved_size = 0;
    
    GeluGradFn() { op_name = "gelu"; }
    
    ~GeluGradFn() override {
        if (saved_input) {
            cudaFree(saved_input);
            saved_input = nullptr;
        }
    }
    
    void save_input(const Tensor& x, cudaStream_t stream) {
        saved_size = x.numel();
        cudaMalloc(&saved_input, saved_size * sizeof(float));
        cudaMemcpyAsync(saved_input, x.data, saved_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!input || !input->requires_grad || !saved_input) return;
        
        input->ensure_grad();
        
        const size_t count = grad_output.numel();
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        // Allocate temp buffer for local grad computation
        float* grad_temp = nullptr;
        cudaMalloc(&grad_temp, count * sizeof(float));
        
        kernel_gelu_backward<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, saved_input, grad_temp, count);
        
        // Accumulate to input's gradient
        input->accumulate_grad(grad_temp, count, 1.0f, stream);
        
        // Must sync before freeing - kernel runs async on stream
        cudaStreamSynchronize(stream);
        cudaFree(grad_temp);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_input) {
            cudaFree(saved_input);
            saved_input = nullptr;
        }
    }
};

/**
 * RMSNormGradFn - Backward for RMSNorm
 * Forward: y = x / rms(x) * gamma
 * Backward: Complex chain rule through normalization
 */
struct RMSNormGradFn : public GradFn {
    Tensor* input = nullptr;
    Tensor* gamma = nullptr;
    float* saved_input = nullptr;
    size_t saved_size = 0;
    int d_model = 0;
    float eps = 1e-5f;
    
    RMSNormGradFn() { op_name = "rms_norm"; }
    
    ~RMSNormGradFn() override {
        if (saved_input) {
            cudaFree(saved_input);
            saved_input = nullptr;
        }
    }
    
    void save_input(const Tensor& x, int d, float e, cudaStream_t stream) {
        d_model = d;
        eps = e;
        saved_size = x.numel();
        cudaMalloc(&saved_input, saved_size * sizeof(float));
        cudaMemcpyAsync(saved_input, x.data, saved_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!saved_input || d_model <= 0) return;
        
        const int tokens = static_cast<int>(saved_size / d_model);
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
        
        if (input && input->requires_grad) {
            input->ensure_grad();
            
            // Note: grad_gamma accumulation happens in kernel via atomicAdd
            float* gamma_grad = (gamma && gamma->requires_grad) ? gamma->grad : nullptr;
            if (gamma && gamma->requires_grad) gamma->ensure_grad();
            
            kernel_rmsnorm_backward<<<tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
                grad_output.data, saved_input, gamma ? gamma->data : nullptr,
                input->grad, gamma_grad,
                tokens, d_model, eps);
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_input) {
            cudaFree(saved_input);
            saved_input = nullptr;
        }
    }
};

/**
 * CrossEntropyGradFn - Backward for softmax cross-entropy loss
 * Forward: loss = -log(softmax(logits)[target])
 * Backward: grad_logits = softmax(logits) - one_hot(target)
 */
struct CrossEntropyGradFn : public GradFn {
    Tensor* logits = nullptr;
    float* saved_logits = nullptr;
    int* targets = nullptr;
    bool owns_targets = false;
    int num_tokens = 0;
    int vocab_size = 0;
    
    CrossEntropyGradFn() { op_name = "cross_entropy"; }
    
    ~CrossEntropyGradFn() override {
        if (saved_logits) cudaFree(saved_logits);
        if (owns_targets && targets) cudaFree(targets);
    }
    
    void save(const Tensor& l, const int* t, int tokens, int vocab, bool copy_targets, cudaStream_t stream) {
        num_tokens = tokens;
        vocab_size = vocab;
        
        // Save logits
        size_t logit_bytes = static_cast<size_t>(tokens) * vocab * sizeof(float);
        cudaMalloc(&saved_logits, logit_bytes);
        cudaMemcpyAsync(saved_logits, l.data, logit_bytes, cudaMemcpyDeviceToDevice, stream);
        
        // Copy targets if needed
        if (copy_targets) {
            size_t target_bytes = tokens * sizeof(int);
            cudaMalloc(&targets, target_bytes);
            cudaMemcpyAsync(targets, t, target_bytes, cudaMemcpyDeviceToDevice, stream);
            owns_targets = true;
        } else {
            targets = const_cast<int*>(t);
            owns_targets = false;
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!logits || !logits->requires_grad || !saved_logits || !targets) return;
        
        logits->ensure_grad();
        
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
        
        kernel_cross_entropy_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
            saved_logits, targets, logits->grad, num_tokens, vocab_size);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_logits) { cudaFree(saved_logits); saved_logits = nullptr; }
        if (owns_targets && targets) { cudaFree(targets); targets = nullptr; }
    }
};

/**
 * EmbeddingGradFn - Backward for embedding lookup
 * Forward: output[i] = weight[token_ids[i]]
 * Backward: grad_weight[token_id] += grad_output (scatter-add)
 */
struct EmbeddingGradFn : public GradFn {
    Tensor* weight = nullptr;
    int* token_ids = nullptr;
    bool owns_token_ids = false;
    int num_tokens = 0;
    int d_model = 0;
    
    EmbeddingGradFn() { op_name = "embedding"; }
    
    ~EmbeddingGradFn() override {
        if (owns_token_ids && token_ids) cudaFree(token_ids);
    }
    
    void save(const int* ids, int tokens, int d, bool copy_ids, cudaStream_t stream) {
        num_tokens = tokens;
        d_model = d;
        
        if (copy_ids) {
            cudaMalloc(&token_ids, tokens * sizeof(int));
            cudaMemcpyAsync(token_ids, ids, tokens * sizeof(int), cudaMemcpyDeviceToDevice, stream);
            owns_token_ids = true;
        } else {
            token_ids = const_cast<int*>(ids);
            owns_token_ids = false;
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!weight || !weight->requires_grad || !token_ids) return;
        
        weight->ensure_grad();
        
        kernel_embedding_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, token_ids, weight->grad, num_tokens, d_model);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (owns_token_ids && token_ids) { cudaFree(token_ids); token_ids = nullptr; }
    }
};

/**
 * SoftmaxGradFn - Backward for softmax operation
 * Forward: y = softmax(x) = exp(x - max) / sum(exp(x - max))
 * Backward: grad_x = y * (grad_y - sum(grad_y * y))
 * Note: We save softmax OUTPUT (not input) for efficient backward
 */
struct SoftmaxGradFn : public GradFn {
    Tensor* input = nullptr;
    float* saved_softmax = nullptr;  // Saved softmax output
    int num_tokens = 0;
    int dim = 0;
    
    SoftmaxGradFn() { op_name = "softmax"; }
    
    ~SoftmaxGradFn() override {
        if (saved_softmax) cudaFree(saved_softmax);
    }
    
    void save(const float* softmax_output, int tokens, int d, cudaStream_t stream) {
        num_tokens = tokens;
        dim = d;
        
        size_t bytes = static_cast<size_t>(tokens) * d * sizeof(float);
        cudaMalloc(&saved_softmax, bytes);
        cudaMemcpyAsync(saved_softmax, softmax_output, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!input || !input->requires_grad || !saved_softmax) return;
        
        input->ensure_grad();
        
        // Shared memory for dot product reduction: one float per warp
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
        
        kernel_softmax_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
            grad_output.data, saved_softmax, input->grad, num_tokens, dim);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_softmax) { cudaFree(saved_softmax); saved_softmax = nullptr; }
    }
};

/**
 * DropoutGradFn - Backward for dropout operation
 * Forward: y = x * mask / (1 - p), where mask is 1 with prob (1-p), 0 with prob p
 * Backward: grad_x = grad_y * mask / (1 - p)
 * Same as forward but with gradient instead of input
 */
struct DropoutGradFn : public GradFn {
    Tensor* input = nullptr;
    uint8_t* saved_mask = nullptr;  // Binary mask from forward
    float scale = 1.0f;             // 1.0 / (1.0 - dropout_prob)
    size_t count = 0;
    
    DropoutGradFn() { op_name = "dropout"; }
    
    ~DropoutGradFn() override {
        if (saved_mask) cudaFree(saved_mask);
    }
    
    void save(const uint8_t* mask, float dropout_prob, size_t n, cudaStream_t stream) {
        count = n;
        scale = (dropout_prob < 1.0f) ? 1.0f / (1.0f - dropout_prob) : 0.0f;
        
        cudaMalloc(&saved_mask, n * sizeof(uint8_t));
        cudaMemcpyAsync(saved_mask, mask, n * sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!input || !input->requires_grad || !saved_mask) return;
        
        input->ensure_grad();
        
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        kernel_dropout_backward<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, saved_mask, input->grad, scale, count);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_mask) { cudaFree(saved_mask); saved_mask = nullptr; }
    }
};

/**
 * ResidualAddGradFn - Optimized backward for residual/skip connections
 * Forward: y = x + residual
 * Backward: grad_x = grad_y, grad_residual = grad_y (both get full gradient)
 * 
 * This is a specialized AddGradFn that avoids unnecessary copies when
 * one input is from a skip connection and should receive the unmodified gradient.
 */
struct ResidualAddGradFn : public GradFn {
    Tensor* input = nullptr;        // Main branch input
    Tensor* residual = nullptr;     // Skip connection input
    
    ResidualAddGradFn() { op_name = "residual_add"; }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // Both inputs receive the same gradient (d(x+r)/dx = 1, d(x+r)/dr = 1)
        const size_t count = grad_output.numel();
        
        if (input && input->requires_grad) {
            input->accumulate_grad(grad_output.data, count, 1.0f, stream);
        }
        
        if (residual && residual->requires_grad) {
            residual->accumulate_grad(grad_output.data, count, 1.0f, stream);
        }
    }
};

//======================================================//
//  Autograd Operations (namespace GRIM::autograd)
//======================================================//

namespace autograd {

Tensor add(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("autograd::add: tensor size mismatch");
    }
    
    Tensor result = Tensor::empty(a.shape, a.requires_grad || b.requires_grad, stream);
    
    // c = a + b
    const size_t count = a.numel();
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    
    // Simple element-wise add kernel
    auto kernel = [](float* c, const float* a, const float* b, size_t n) {
        // Launched via lambda won't work - need explicit kernel
    };
    
    // Use TensorContract::add for the forward
    TensorContract::TensorView a_view(const_cast<float*>(a.data), a.shape);
    TensorContract::TensorView b_view(const_cast<float*>(b.data), b.shape);
    TensorContract::TensorView r_view(result.data, result.shape);
    TensorContract::add(a_view, b_view, r_view, stream);
    
    // Set up backward
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new AddGradFn();
        grad_fn->input_a = const_cast<Tensor*>(&a);
        grad_fn->input_b = const_cast<Tensor*>(&b);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor gelu(const Tensor& x, cudaStream_t stream) {
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream);
    
    // Forward: y = gelu(x)
    // gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const size_t count = x.numel();
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    
    // Launch GELU forward kernel
    kernel_gelu_forward<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);
    
    // Set up backward
    if (x.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new GeluGradFn();
        grad_fn->input = const_cast<Tensor*>(&x);
        grad_fn->save_input(x, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps, cudaStream_t stream) {
    if (!x.shape.is_flat()) {
        throw std::invalid_argument("autograd::rms_norm: input must be 2D (BSM)");
    }
    
    const auto& dims = x.shape.as_2d();
    const int tokens = dims.rows;
    const int d_model = dims.cols;
    Tensor result = Tensor::empty(x.shape, x.requires_grad || gamma.requires_grad, stream);
    
    // Forward: y = x / rms(x) * gamma
    // Shared memory: one float per warp for reduction
    const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
    kernel_rmsnorm_forward<<<tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
        x.data, gamma.data, result.data, tokens, d_model, eps);
    
    // Set up backward
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new RMSNormGradFn();
        grad_fn->input = const_cast<Tensor*>(&x);
        grad_fn->gamma = const_cast<Tensor*>(&gamma);
        grad_fn->save_input(x, d_model, eps, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor cross_entropy(const Tensor& logits, const int* targets, int num_tokens, int vocab_size, cudaStream_t stream) {
    // Output is a scalar loss
    auto loss_shape = TensorContract::TensorShape::make_BSM(1, 1);
    Tensor result = Tensor::zeros(loss_shape, logits.requires_grad, stream);
    
    // Forward: compute cross-entropy loss (would use existing loss kernel)
    // Placeholder: set to 0
    
    // Set up backward
    if (logits.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new CrossEntropyGradFn();
        grad_fn->logits = const_cast<Tensor*>(&logits);
        grad_fn->save(logits, targets, num_tokens, vocab_size, true, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens, cudaStream_t stream) {
    if (!weight.shape.is_flat()) {
        throw std::invalid_argument("autograd::embedding: weight must be 2D [vocab_size, d_model]");
    }
    
    const int d_model = weight.shape.as_2d().cols;
    auto output_shape = TensorContract::TensorShape::make_BSM(num_tokens, d_model);
    Tensor result = Tensor::empty(output_shape, weight.requires_grad, stream);
    
    // Forward: gather from weight table (would use existing embedding kernel)
    // Placeholder
    
    // Set up backward
    if (weight.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new EmbeddingGradFn();
        grad_fn->weight = const_cast<Tensor*>(&weight);
        grad_fn->save(token_ids, num_tokens, d_model, true, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor softmax(const Tensor& x, cudaStream_t stream) {
    if (!x.shape.is_flat()) {
        throw std::invalid_argument("autograd::softmax: input must be 2D [tokens, dim]");
    }
    
    const auto dims = x.shape.as_2d();
    const int num_tokens = dims.rows;
    const int dim = dims.cols;
    
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream);
    
    // Forward: compute softmax
    // For each row: y = exp(x - max(x)) / sum(exp(x - max(x)))
    // Would use existing softmax kernel here
    // Placeholder: copy input
    cudaMemcpyAsync(result.data, x.data, x.size_bytes(), cudaMemcpyDeviceToDevice, stream);
    
    // Set up backward - save the softmax OUTPUT for efficient backward
    if (x.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new SoftmaxGradFn();
        grad_fn->input = const_cast<Tensor*>(&x);
        grad_fn->save(result.data, num_tokens, dim, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor dropout(const Tensor& x, float p, bool training, const uint8_t* mask, cudaStream_t stream) {
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream);
    
    if (!training || p == 0.0f) {
        // No dropout during inference or when p=0
        cudaMemcpyAsync(result.data, x.data, x.size_bytes(), cudaMemcpyDeviceToDevice, stream);
        
        if (x.requires_grad) {
            result.is_leaf = false;
            // Identity backward - just pass gradient through
            auto* grad_fn = new AddGradFn();  // Reuse AddGradFn as identity
            grad_fn->input_a = const_cast<Tensor*>(&x);
            grad_fn->input_b = nullptr;  // Mark as identity (no second input)
            result.grad_fn = grad_fn;
        }
        return result;
    }
    
    const size_t count = x.numel();
    
    // Forward: y = x * mask / (1 - p) where mask is 0/1
    // Would call dropout forward kernel with mask
    // Placeholder: copy input
    cudaMemcpyAsync(result.data, x.data, x.size_bytes(), cudaMemcpyDeviceToDevice, stream);
    
    // Set up backward
    if (x.requires_grad && mask) {
        result.is_leaf = false;
        auto* grad_fn = new DropoutGradFn();
        grad_fn->input = const_cast<Tensor*>(&x);
        grad_fn->save(mask, p, count, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor residual_add(const Tensor& x, const Tensor& residual, cudaStream_t stream) {
    if (x.numel() != residual.numel()) {
        throw std::invalid_argument("autograd::residual_add: tensor size mismatch");
    }
    
    Tensor result = Tensor::empty(x.shape, x.requires_grad || residual.requires_grad, stream);
    
    // Forward: y = x + residual
    TensorContract::TensorView x_view(const_cast<float*>(x.data), x.shape);
    TensorContract::TensorView r_view(const_cast<float*>(residual.data), residual.shape);
    TensorContract::TensorView out_view(result.data, result.shape);
    TensorContract::add(x_view, r_view, out_view, stream);
    
    // Set up backward
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new ResidualAddGradFn();
        grad_fn->input = const_cast<Tensor*>(&x);
        grad_fn->residual = const_cast<Tensor*>(&residual);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

/**
 * MatMulGradFn - Backward for matrix multiplication
 * Forward: C = A @ B  where A is [M, K], B is [K, N], C is [M, N]
 * Backward:
 *   grad_A = grad_C @ B^T   [M, N] @ [N, K] = [M, K]
 *   grad_B = A^T @ grad_C   [K, M] @ [M, N] = [K, N]
 *
 * cuBLAS convention (column-major):
 * For row-major C = A @ B, we compute as C^T = B^T @ A^T
 * So cublasSgemm(B^T, A^T) gives us C in row-major
 */
struct MatMulGradFn : public GradFn {
    Tensor* input_a = nullptr;
    Tensor* input_b = nullptr;
    float* saved_a = nullptr;  // Saved copy of A
    float* saved_b = nullptr;  // Saved copy of B
    int M = 0, K = 0, N = 0;   // Dimensions
    cublasHandle_t cublas_handle = nullptr;
    
    MatMulGradFn() { op_name = "matmul"; }
    
    ~MatMulGradFn() override {
        if (saved_a) { cudaFree(saved_a); saved_a = nullptr; }
        if (saved_b) { cudaFree(saved_b); saved_b = nullptr; }
    }
    
    void save(const Tensor& a, const Tensor& b, int m, int k, int n, 
              cublasHandle_t handle, cudaStream_t stream) {
        M = m; K = k; N = n;
        cublas_handle = handle;
        
        // Save A if B requires grad (needed for grad_B = A^T @ grad_C)
        if (input_b && input_b->requires_grad) {
            size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
            cudaMalloc(&saved_a, a_bytes);
            cudaMemcpyAsync(saved_a, a.data, a_bytes, cudaMemcpyDeviceToDevice, stream);
        }
        
        // Save B if A requires grad (needed for grad_A = grad_C @ B^T)
        if (input_a && input_a->requires_grad) {
            size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
            cudaMalloc(&saved_b, b_bytes);
            cudaMemcpyAsync(saved_b, b.data, b_bytes, cudaMemcpyDeviceToDevice, stream);
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!cublas_handle) return;
        
        const float alpha = 1.0f;
        const float beta_accum = 1.0f;  // Accumulate to existing gradient
        
        cublasSetStream(cublas_handle, stream);
        
        // grad_A = grad_C @ B^T  [M, N] @ [N, K] = [M, K]
        // Row-major: grad_A = grad_C @ B^T
        // cuBLAS (col-major): grad_A^T = B @ grad_C^T
        // cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, alpha, B, K, grad_C, N, beta, grad_A, K)
        if (input_a && input_a->requires_grad && saved_b) {
            input_a->ensure_grad();
            
            // Allocate temp buffer for this gradient computation
            float* grad_a_temp = nullptr;
            cudaMalloc(&grad_a_temp, static_cast<size_t>(M) * K * sizeof(float));
            cudaMemsetAsync(grad_a_temp, 0, static_cast<size_t>(M) * K * sizeof(float), stream);
            
            // C = grad_C [M, N] row-major
            // B = saved_b [K, N] row-major  -> B^T is [N, K]
            // We want grad_A [M, K] = grad_C @ B^T
            // In cuBLAS col-major: C_col = B^T_col @ A_col
            // grad_A_col [K, M] = B_col [N, K]^T @ grad_C_col [N, M]
            // = B_col^T [K, N] @ grad_C_col [N, M] -- but B is row-major [K,N] so B_col is [N,K]
            // Simpler: use CUBLAS_OP_T on appropriate matrices
            
            // For row-major: C[M,K] = A[M,N] @ B[N,K]
            // cuBLAS: C^T[K,M] = B^T[K,N] @ A^T[N,M]
            // Here A=grad_C[M,N], B=saved_b^T[N,K], C=grad_A[M,K]
            // So: grad_A^T[K,M] = saved_b[K,N] @ grad_C^T[N,M]
            cublasSgemm(cublas_handle,
                CUBLAS_OP_N,    // saved_b is [K,N] row-major = [N,K] col-major, no transpose
                CUBLAS_OP_T,    // grad_C is [M,N] row-major, we want grad_C^T
                K, M, N,        // C is [K,M] col-major = [M,K] row-major (grad_A)
                &alpha,
                saved_b, N,     // B: [K,N] row-major -> ldb = N
                grad_output.data, N,  // grad_C: [M,N] row-major -> lda = N
                &beta_accum,
                grad_a_temp, K  // grad_A: [M,K] row-major -> ldc = K
            );
            
            input_a->accumulate_grad(grad_a_temp, static_cast<size_t>(M) * K, 1.0f, stream);
            // Must sync before freeing - GEMM runs async on stream
            cudaStreamSynchronize(stream);
            cudaFree(grad_a_temp);
        }
        
        // grad_B = A^T @ grad_C  [K, M] @ [M, N] = [K, N]
        // Row-major: grad_B = A^T @ grad_C
        // cuBLAS (col-major): grad_B^T = grad_C^T @ A
        if (input_b && input_b->requires_grad && saved_a) {
            input_b->ensure_grad();
            
            float* grad_b_temp = nullptr;
            cudaMalloc(&grad_b_temp, static_cast<size_t>(K) * N * sizeof(float));
            cudaMemsetAsync(grad_b_temp, 0, static_cast<size_t>(K) * N * sizeof(float), stream);
            
            // For row-major: C[K,N] = A[K,M] @ B[M,N]
            // Here A=saved_a^T[K,M], B=grad_C[M,N], C=grad_B[K,N]
            // saved_a is [M,K] so saved_a^T is [K,M]
            // cuBLAS: C^T[N,K] = B^T[N,M] @ A^T^T[M,K] = grad_C^T @ saved_a
            cublasSgemm(cublas_handle,
                CUBLAS_OP_N,    // grad_C^T: grad_C is [M,N] row-major, transposed = [N,M] col-major
                CUBLAS_OP_N,    // saved_a: [M,K] row-major = [K,M] col-major
                N, K, M,        // C is [N,K] col-major = [K,N] row-major (grad_B)
                &alpha,
                grad_output.data, N,  // grad_C: [M,N] row-major -> ld = N
                saved_a, K,           // saved_a: [M,K] row-major -> ld = K
                &beta_accum,
                grad_b_temp, N        // grad_B: [K,N] row-major -> ld = N
            );
            
            input_b->accumulate_grad(grad_b_temp, static_cast<size_t>(K) * N, 1.0f, stream);
            // Must sync before freeing - GEMM runs async on stream
            cudaStreamSynchronize(stream);
            cudaFree(grad_b_temp);
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_a) { cudaFree(saved_a); saved_a = nullptr; }
        if (saved_b) { cudaFree(saved_b); saved_b = nullptr; }
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

Tensor matmul(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    // Validate inputs are 2D
    if (!a.shape.is_flat() || !b.shape.is_flat()) {
        throw std::invalid_argument("autograd::matmul: inputs must be 2D (BSM layout)");
    }
    
    const auto& a_shape = a.shape.as_2d();
    const auto& b_shape = b.shape.as_2d();
    
    const int M = a_shape.rows;   // A is [M, K]
    const int K = a_shape.cols;
    const int K2 = b_shape.rows;  // B is [K, N]
    const int N = b_shape.cols;
    
    if (K != K2) {
        throw std::invalid_argument("autograd::matmul: inner dimensions must match");
    }
    
    // Get cuBLAS handle
    cublasHandle_t handle = get_autograd_cublas_handle();
    if (!handle) {
        throw std::runtime_error("autograd::matmul: cuBLAS handle not set. Call set_autograd_cublas_handle() first.");
    }
    
    // Output shape: [M, N]
    auto output_shape = TensorContract::TensorShape::make_BSM(M, N);
    Tensor result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad, stream);
    
    // Forward: C = A @ B
    // Row-major: for cuBLAS we compute C^T = B^T @ A^T
    // cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSetStream(handle, stream);
    cublasStatus_t status = cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data, N,    // B: [K, N] row-major
        a.data, K,    // A: [M, K] row-major
        &beta,
        result.data, N  // C: [M, N] row-major
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("autograd::matmul: cuBLAS sgemm failed");
    }
    
    // Set up backward
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new MatMulGradFn();
        grad_fn->input_a = const_cast<Tensor*>(&a);
        grad_fn->input_b = const_cast<Tensor*>(&b);
        grad_fn->save(a, b, M, K, N, handle, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

//======================================================//
//  BF16 Conversion Kernels (for FlashAttention integration)
//======================================================//

// Simple FP32 <-> BF16 conversion (same layout)
__global__ void kernel_fp32_to_bf16(const float* __restrict__ src, 
                                     __nv_bfloat16* __restrict__ dst, 
                                     size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ void kernel_bf16_to_fp32(const __nv_bfloat16* __restrict__ src,
                                     float* __restrict__ dst,
                                     size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

// Layout conversion: BHSD (FP32) -> BSHD (BF16)
// FlashAttention uses BSHD (batch, seq, head, dim) layout
__global__ void kernel_BHSD_fp32_to_BSHD_bf16(
    const float* __restrict__ src,   // [B, H, S, D]
    __nv_bfloat16* __restrict__ dst, // [B, S, H, D]
    int batch, int heads, int seq_len, int head_dim
) {
    const size_t total = static_cast<size_t>(batch) * heads * seq_len * head_dim;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Decode BHSD source index
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq_len;
    const int h = (idx / (head_dim * seq_len)) % heads;
    const int b = idx / (head_dim * seq_len * heads);
    
    // Compute BSHD destination index
    const size_t dst_idx = (static_cast<size_t>(b) * seq_len * heads * head_dim) +
                           (static_cast<size_t>(s) * heads * head_dim) +
                           (static_cast<size_t>(h) * head_dim) + d;
    
    dst[dst_idx] = __float2bfloat16(src[idx]);
}

// Layout conversion: BSHD (BF16) -> BHSD (FP32)
__global__ void kernel_BSHD_bf16_to_BHSD_fp32(
    const __nv_bfloat16* __restrict__ src, // [B, S, H, D]
    float* __restrict__ dst,               // [B, H, S, D]
    int batch, int heads, int seq_len, int head_dim
) {
    const size_t total = static_cast<size_t>(batch) * heads * seq_len * head_dim;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Decode BSHD source index
    const int d = idx % head_dim;
    const int h = (idx / head_dim) % heads;
    const int s = (idx / (head_dim * heads)) % seq_len;
    const int b = idx / (head_dim * heads * seq_len);
    
    // Compute BHSD destination index
    const size_t dst_idx = (static_cast<size_t>(b) * heads * seq_len * head_dim) +
                           (static_cast<size_t>(h) * seq_len * head_dim) +
                           (static_cast<size_t>(s) * head_dim) + d;
    
    dst[dst_idx] = __bfloat162float(src[idx]);
}

/**
 * ScaledDotProductAttentionGradFn - Backward for attention
 * Forward: O = softmax(Q @ K^T / sqrt(d)) @ V
 * Uses FlashAttention v2 for memory-efficient backward.
 *
 * Requires saving: Q, K, V, O, and softmax_lse from forward.
 * Backward computes: grad_Q, grad_K, grad_V
 */
struct ScaledDotProductAttentionGradFn : public GradFn {
    Tensor* input_q = nullptr;
    Tensor* input_k = nullptr;
    Tensor* input_v = nullptr;
    
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
    bool causal = true;
    
    ScaledDotProductAttentionGradFn() { op_name = "scaled_dot_product_attention"; }
    
    ~ScaledDotProductAttentionGradFn() override {
        release_saved();
    }
    
    void save(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
              int b, int s, int nh, int nkv, int hd, bool is_causal,
              cudaStream_t stream) {
        batch_size = b;
        seq_len = s;
        num_heads = nh;
        num_kv_heads = nkv;
        head_dim = hd;
        causal = is_causal;
        
        const size_t q_elems = static_cast<size_t>(b) * s * nh * hd;
        const size_t kv_elems = static_cast<size_t>(b) * s * nkv * hd;
        const size_t lse_elems = static_cast<size_t>(b) * nh * s;
        
        // Allocate bf16 buffers for FlashAttention
        cudaMalloc(&saved_q_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&saved_k_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&saved_v_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&saved_out_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&saved_softmax_lse, lse_elems * sizeof(float));
        
        // Allocate backward workspace
        const size_t dq_accum_bytes = flash_attn_dq_accum_bytes(b, s, nh, hd);
        const size_t dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(b, s, nh);
        cudaMalloc(&dq_accum, dq_accum_bytes);
        cudaMalloc(&dsoftmax_sum, dsoftmax_sum_bytes);
        
        // Allocate gradient bf16 buffers
        cudaMalloc(&dq_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&dk_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&dv_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&dout_bf16, q_elems * sizeof(__nv_bfloat16));
        
        // Convert FP32 BHSD inputs to BF16 BSHD for FlashAttention
        const int block_size = 256;
        const int q_blocks = static_cast<int>((q_elems + block_size - 1) / block_size);
        const int kv_blocks = static_cast<int>((kv_elems + block_size - 1) / block_size);
        
        // Q: [B, H, S, D] FP32 -> [B, S, H, D] BF16
        kernel_BHSD_fp32_to_BSHD_bf16<<<q_blocks, block_size, 0, stream>>>(
            q.data, saved_q_bf16, b, nh, s, hd);
        
        // K: [B, Hkv, S, D] FP32 -> [B, S, Hkv, D] BF16
        kernel_BHSD_fp32_to_BSHD_bf16<<<kv_blocks, block_size, 0, stream>>>(
            k.data, saved_k_bf16, b, nkv, s, hd);
        
        // V: [B, Hkv, S, D] FP32 -> [B, S, Hkv, D] BF16
        kernel_BHSD_fp32_to_BSHD_bf16<<<kv_blocks, block_size, 0, stream>>>(
            v.data, saved_v_bf16, b, nkv, s, hd);
        
        // Output: [B, H, S, D] FP32 -> [B, S, H, D] BF16
        kernel_BHSD_fp32_to_BSHD_bf16<<<q_blocks, block_size, 0, stream>>>(
            out.data, saved_out_bf16, b, nh, s, hd);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!saved_q_bf16 || !saved_k_bf16 || !saved_v_bf16 || !saved_out_bf16) {
            return;  // No saved state
        }
        
        const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
        const size_t kv_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
        const int block_size = 256;
        const int q_blocks = static_cast<int>((q_elems + block_size - 1) / block_size);
        const int kv_blocks = static_cast<int>((kv_elems + block_size - 1) / block_size);
        
        // Convert grad_output (FP32 BHSD) to BF16 BSHD
        kernel_BHSD_fp32_to_BSHD_bf16<<<q_blocks, block_size, 0, stream>>>(
            grad_output.data, dout_bf16, batch_size, num_heads, seq_len, head_dim);
        
        // Call FlashAttention backward
        flash_attn_bwd_ex(
            saved_q_bf16,      // Q  [B, S, H, D] bf16
            saved_k_bf16,      // K  [B, S, Hkv, D] bf16
            saved_v_bf16,      // V  [B, S, Hkv, D] bf16
            saved_out_bf16,    // O  [B, S, H, D] bf16
            dout_bf16,         // dO [B, S, H, D] bf16
            saved_softmax_lse, // LSE [B, H, S] fp32
            nullptr,           // alibi_slopes (not used)
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
            causal,
            true,              // is_bf16
            stream
        );
        
        // Convert gradients back to FP32 BHSD and accumulate
        if (input_q && input_q->requires_grad) {
            input_q->ensure_grad();
            float* grad_q_fp32 = nullptr;
            cudaMalloc(&grad_q_fp32, q_elems * sizeof(float));
            kernel_BSHD_bf16_to_BHSD_fp32<<<q_blocks, block_size, 0, stream>>>(
                dq_bf16, grad_q_fp32, batch_size, num_heads, seq_len, head_dim);
            input_q->accumulate_grad(grad_q_fp32, q_elems, 1.0f, stream);
            cudaFree(grad_q_fp32);
        }
        
        if (input_k && input_k->requires_grad) {
            input_k->ensure_grad();
            float* grad_k_fp32 = nullptr;
            cudaMalloc(&grad_k_fp32, kv_elems * sizeof(float));
            kernel_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dk_bf16, grad_k_fp32, batch_size, num_kv_heads, seq_len, head_dim);
            input_k->accumulate_grad(grad_k_fp32, kv_elems, 1.0f, stream);
            cudaFree(grad_k_fp32);
        }
        
        if (input_v && input_v->requires_grad) {
            input_v->ensure_grad();
            float* grad_v_fp32 = nullptr;
            cudaMalloc(&grad_v_fp32, kv_elems * sizeof(float));
            kernel_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dv_bf16, grad_v_fp32, batch_size, num_kv_heads, seq_len, head_dim);
            input_v->accumulate_grad(grad_v_fp32, kv_elems, 1.0f, stream);
            cudaFree(grad_v_fp32);
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
    }
};

Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor* mask, float scale, cudaStream_t stream
) {
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
    
    // Output shape: same as Q [B, H, S, D]
    auto output_shape = TensorContract::TensorShape::make_BHSD(batch_size, num_heads, seq_len, head_dim);
    bool requires_grad = q.requires_grad || k.requires_grad || v.requires_grad;
    Tensor result = Tensor::zeros(output_shape, requires_grad, stream);
    
    // Compute default scale if not provided
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
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
    
    cudaMalloc(&q_bf16, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&k_bf16, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&v_bf16, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&out_bf16, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&softmax_lse, lse_elems * sizeof(float));
    
    // Convert FP32 BHSD -> BF16 BSHD
    const int block_size = 256;
    const int q_blocks = static_cast<int>((q_elems + block_size - 1) / block_size);
    const int kv_blocks = static_cast<int>((kv_elems + block_size - 1) / block_size);
    
    kernel_BHSD_fp32_to_BSHD_bf16<<<q_blocks, block_size, 0, stream>>>(
        q.data, q_bf16, batch_size, num_heads, seq_len, head_dim);
    kernel_BHSD_fp32_to_BSHD_bf16<<<kv_blocks, block_size, 0, stream>>>(
        k.data, k_bf16, batch_size, num_kv_heads, seq_len, head_dim);
    kernel_BHSD_fp32_to_BSHD_bf16<<<kv_blocks, block_size, 0, stream>>>(
        v.data, v_bf16, batch_size, num_kv_heads, seq_len, head_dim);
    
    // Forward pass with FlashAttention
    // Note: FlashAttention expects scale=1/sqrt(d) internally via softmax_scale
    // But our flash_attn_fwd_ex uses standard 1/sqrt(d) scaling
    flash_attn_fwd_ex(
        q_bf16,      // Q  [B, S, H, D] bf16
        k_bf16,      // K  [B, S, Hkv, D] bf16
        v_bf16,      // V  [B, S, Hkv, D] bf16
        out_bf16,    // O  [B, S, H, D] bf16
        softmax_lse, // LSE [B, H, S] fp32
        nullptr,     // alibi_slopes
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        true,        // causal
        true,        // is_bf16
        stream
    );
    
    // Convert BF16 BSHD -> FP32 BHSD for output
    kernel_BSHD_bf16_to_BHSD_fp32<<<q_blocks, block_size, 0, stream>>>(
        out_bf16, result.data, batch_size, num_heads, seq_len, head_dim);
    
    // Set up backward if needed
    if (requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new ScaledDotProductAttentionGradFn();
        grad_fn->input_q = const_cast<Tensor*>(&q);
        grad_fn->input_k = const_cast<Tensor*>(&k);
        grad_fn->input_v = const_cast<Tensor*>(&v);
        
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
        grad_fn->causal = true;
        
        // Allocate backward workspace
        const size_t dq_accum_bytes = flash_attn_dq_accum_bytes(batch_size, seq_len, num_heads, head_dim);
        const size_t dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(batch_size, seq_len, num_heads);
        cudaMalloc(&grad_fn->dq_accum, dq_accum_bytes);
        cudaMalloc(&grad_fn->dsoftmax_sum, dsoftmax_sum_bytes);
        cudaMalloc(&grad_fn->dq_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dk_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dv_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dout_bf16, q_elems * sizeof(__nv_bfloat16));
        
        result.grad_fn = grad_fn;
        
        // Don't free bf16 buffers - ownership transferred to grad_fn
        q_bf16 = nullptr;
        k_bf16 = nullptr;
        v_bf16 = nullptr;
        out_bf16 = nullptr;
        softmax_lse = nullptr; 
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

}  // namespace autograd

}  // namespace GRIM
