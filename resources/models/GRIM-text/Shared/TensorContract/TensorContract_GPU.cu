//======================================================//
//  TensorContract_GPU.cu
//  CUDA implementation of type-safe tensor operations
//======================================================//

#include "TensorContract_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include <cuda_runtime.h>
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
