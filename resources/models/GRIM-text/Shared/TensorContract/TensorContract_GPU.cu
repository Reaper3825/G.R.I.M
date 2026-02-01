//======================================================//
//  TensorContract_GPU.cu
//  CUDA implementation of type-safe tensor operations
//======================================================//

#include "TensorContract_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Layers/Attention/QKV_Projector.hpp"  // ISSUE #62: For launchReshapeFromBHSD
#include "../../Layers/FeedForward/Feed_Forward_GPU.hpp"  // ISSUE #97: For launchFFNBiasAdd/Backward
#include "../PBM/PositionalBiasMethod.hpp"  // ISSUE #119: For RoPE autograd backward
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <mutex>
#include <vector>
#include <atomic>

// DEBUG LOGGING - set to 1 to enable verbose tensor operation logging
#define TENSOR_VERBOSE_DEBUG 0

#if TENSOR_VERBOSE_DEBUG
#define TENSOR_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define TENSOR_LOG(...) ((void)0)
#endif

// ═══════════════════════════════════════════════════════════════════════════
// AUTOGRAD DEBUG TOGGLE - Set to true to trace autograd chain execution
// ═══════════════════════════════════════════════════════════════════════════
bool g_autograd_verbose = false;  // PRODUCTION: Disable verbose AG_TRACE logging (causes GPU syncs)

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

// ═══════════════════════════════════════════════════════════════════════════
// ISSUE #60: DEBUG GRADIENT ATTRIBUTION
// Hooks to capture gradient sources separately for tied-weight debugging

// Global buffers for capturing gradient contributions (set by TrainingState)
float* g_debug_lm_head_only_grad = nullptr;   // Where to copy LM head backward contribution
float* g_debug_embedding_only_grad = nullptr; // Where to copy embedding backward contribution
size_t g_debug_grad_buffer_size = 0;          // Size in elements (vocab_size * d_model)
bool g_debug_capture_enabled = true;         // Enable/disable capturing

// ISSUE #60 ROOT CAUSE FIX: Skip embedding backward when weights are tied
// PyTorch with tied weights does: logits = x @ embedding.weight.T
// This means autograd only runs matmul backward on embedding weight ONCE.
// In GRIM, we were running BOTH lm_head backward AND embedding backward,
// which wrote opposite gradients to the same shared buffer, canceling to ZERO!
// Set this to true when tie_embeddings=true to match PyTorch behavior.
bool g_skip_embedding_backward_for_tied_weights = false;

// ISSUE #60 PCGrad FIX: Instead of skipping, use PCGrad to combine gradients
// This preserves LM head direction while adding orthogonal embedding information
float* g_pcgrad_temp_buffer = nullptr;    // Temporary buffer for embedding grad before projection
size_t g_pcgrad_buffer_size = 0;          // Size in elements (vocab_size * d_model)

// Call this after LM head matmul backward to capture its contribution
void debugCaptureLMHeadGrad(float* grad_ptr, size_t size, cudaStream_t stream) {
    if (!g_debug_capture_enabled || !g_debug_lm_head_only_grad || !grad_ptr) return;
    const size_t copy_size = (size < g_debug_grad_buffer_size ? size : g_debug_grad_buffer_size);
    cudaMemcpyAsync(g_debug_lm_head_only_grad, grad_ptr, copy_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
}

// Call this after embedding backward to capture its contribution
void debugCaptureEmbeddingGrad(float* grad_ptr, size_t size, cudaStream_t stream) {
    if (!g_debug_capture_enabled || !g_debug_embedding_only_grad || !grad_ptr) return;
    const size_t copy_size = (size < g_debug_grad_buffer_size ? size : g_debug_grad_buffer_size);
    cudaMemcpyAsync(g_debug_embedding_only_grad, grad_ptr, copy_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
}

//======================================================//
//  ISSUE #53 FIX v2: Use cudaFreeAsync for non-blocking cleanup
//  
//  PROBLEM: cudaFree() implicitly synchronizes with pending GPU work.
//  PREVIOUS ATTEMPT: Deferred queue didn't work because CUDA may reuse
//  "queued but not freed" memory addresses for new allocations.
//  
//  SOLUTION: Use cudaFreeAsync() which is truly stream-ordered.
//  Memory is freed after all pending work on the stream completes,
//  but the call returns immediately without blocking the CPU.
//  
//  NOTE: cudaFreeAsync requires CUDA 11.2+ (we have 12.5)
//======================================================//

// Global stream for async cleanup - initialized on first use
static cudaStream_t g_cleanup_stream = nullptr;

void initCleanupStream() {
    if (g_cleanup_stream == nullptr) {
        cudaStreamCreate(&g_cleanup_stream);
    }
}

//======================================================//
//  KERNEL LAUNCH DIAGNOSTIC - Track which kernel is stuck
//  Records event after each kernel to pinpoint hangs
//======================================================//

static std::atomic<int> g_kernel_launch_count{0};
static const char* g_last_kernel_name = nullptr;
static cudaEvent_t g_last_kernel_event = nullptr;

//======================================================//
//  GRADFN OPERATION TRACKER - Track which autograd op caused error
//  RULE 20: When something fails, provide FULL CONTEXT
//======================================================//

static const char* g_current_gradfn_op = nullptr;
static void* g_current_gradfn_ptr = nullptr;

void setCurrentGradFnOp(const char* op_name, void* gradfn_ptr) {
    g_current_gradfn_op = op_name;
    g_current_gradfn_ptr = gradfn_ptr;
}

void clearCurrentGradFnOp() {
    g_current_gradfn_op = nullptr;
    g_current_gradfn_ptr = nullptr;
}

std::string getCurrentGradFnContext() {
    std::string ctx;
    if (g_current_gradfn_op) {
        ctx += "Current GradFn: op=";
        ctx += g_current_gradfn_op;
        ctx += " ptr=";
        char buf[32];
        snprintf(buf, sizeof(buf), "%p", g_current_gradfn_ptr);
        ctx += buf;
    } else {
        ctx += "Current GradFn: (none)";
    }
    ctx += " | Last kernel: ";
    if (g_last_kernel_name) {
        ctx += g_last_kernel_name;
        ctx += " (#";
        ctx += std::to_string(g_kernel_launch_count.load());
        ctx += ")";
    } else {
        ctx += "(none)";
    }
    return ctx;
}

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream) {
    ++g_kernel_launch_count;
    g_last_kernel_name = kernel_name;
    
    // Create event if first time
    if (g_last_kernel_event == nullptr) {
        cudaEventCreate(&g_last_kernel_event);
    }
    
    // Record event after kernel
    cudaEventRecord(g_last_kernel_event, stream);
    
    // Check for immediate kernel launch errors - RULE 20: Fail Loud
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + kernel_name + " - " + cudaGetErrorString(err));
    }
}

void checkLastKernelComplete() {
    // Minimal check - no logging
    if (g_last_kernel_event) {
        cudaEventQuery(g_last_kernel_event);
    }
}

// Track cuBLAS SGEMM calls
void trackCublasCall(const char* op_name, cublasHandle_t handle, cudaStream_t stream, cublasStatus_t status) {
    ++g_kernel_launch_count;
    g_last_kernel_name = op_name;
    
    // Create event if first time
    if (g_last_kernel_event == nullptr) {
        cudaEventCreate(&g_last_kernel_event);
    }
    
    // Record event after cuBLAS call
    cudaEventRecord(g_last_kernel_event, stream);
    
    // RULE 20: Fail Loud - throw immediately on cuBLAS error
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS SGEMM failed: ") + op_name + 
            " status=" + std::to_string(static_cast<int>(status)) +
            " (CUBLAS_STATUS_INVALID_VALUE=7 means dimension/leading-dim mismatch)");
    }
    
    // Also check for CUDA errors - RULE 20: Fail Loud
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after cuBLAS call: ") + op_name + 
            " - " + cudaGetErrorString(cuda_err));
    }
}

// Call this from shared_ptr deleters - uses cudaFreeAsync for non-blocking cleanup
void queueForDeferredCleanup(void* ptr) {
    if (ptr) {
        initCleanupStream();
        cudaError_t err = cudaFreeAsync(ptr, g_cleanup_stream);
        if (err != cudaSuccess) {
            cudaFree(ptr);  // Fallback - may block but at least frees
        }
    }
}

// Synchronize the cleanup stream - call this at safe points to ensure all frees complete
void flushDeferredCleanup() {
    if (g_cleanup_stream) {
        cudaStreamSynchronize(g_cleanup_stream);
    }
}

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
//  CUDA Kernels - Basic Operations (for TensorContract API)
//======================================================//

__global__ void kernel_zero(float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = 0.0f;
    }
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

// Kernel: Zero-initialize tensor (outside anonymous namespace so visible everywhere in GRIM)
__global__ void kernel_zero_autograd(float* data, size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = 0.0f;
    }
}

namespace {

// Kernel: Xavier uniform initialization (uses curand-free LCG for reproducibility)
// Issue #107 FIX: Run multiple LCG iterations to decorrelate consecutive elements
// BUG: Single iteration with state = (seed + idx*A)*A + C is LINEAR in idx,
// causing consecutive elements to have constant state difference A².
// This produced correlated values with avg|cosine| ≈ 0.37 instead of expected ≈ 0.036.
// FIX: Initialize state with hash-like mixing, then run 16 LCG iterations.
__global__ void kernel_xavier_uniform(float* data, size_t count, float scale, uint64_t seed) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Use idx to create per-element unique seed via bit mixing (splitmix64-style)
        uint64_t z = seed + idx * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        uint64_t state = z ^ (z >> 31);
        
        // Run 16 LCG iterations to fully decorrelate
        // LCG PRNG: x_{n+1} = (a * x_n + c) mod m
        // Parameters from PCG family (better statistical properties)
        constexpr uint64_t LCG_A = 6364136223846793005ULL;
        constexpr uint64_t LCG_C = 1442695040888963407ULL;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            state = state * LCG_A + LCG_C;
        }
        
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
    , grad_(std::move(other.grad_))  // ISSUE #59: Transfer shared_ptr ownership
    , grad_fn(other.grad_fn)
    , owns_grad_fn(other.owns_grad_fn)  // ISSUE #54 FIX: Transfer grad_fn ownership
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
    // grad_ already moved via std::move
    other.grad_fn = nullptr;
    other.owns_data = false;
    other.owns_grad_fn = false;  // ISSUE #54 FIX: Clear ownership flag
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Release current resources
        release();
        
        // Move from other
        data = other.data;
        shape = other.shape;
        owns_data = other.owns_data;
        grad_ = std::move(other.grad_);  // ISSUE #59: Transfer shared_ptr ownership
        grad_fn = other.grad_fn;
        owns_grad_fn = other.owns_grad_fn;  // ISSUE #54 FIX: Transfer grad_fn ownership
        requires_grad = other.requires_grad;
        is_leaf = other.is_leaf;
        retain_grad = other.retain_grad;
        stream = other.stream;
        device_id = other.device_id;
        name = other.name;
        version = other.version;
        
        // Null out other's pointers
        other.data = nullptr;
        // grad_ already moved
        other.grad_fn = nullptr;
        other.owns_data = false;
        other.owns_grad_fn = false;  // ISSUE #54 FIX: Clear ownership flag
    }
    return *this;
}

//======================================================//
//  Tensor Factory Methods
//======================================================//

Tensor Tensor::zeros(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("Tensor::zeros: stream is NULL - caller MUST provide valid stream");
    }
    
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
    kernel_zero_autograd<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(t.data, count);
    
    cudaError_t kernelErr = cudaGetLastError();
    
    if (kernelErr != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::zeros: kernel launch failed: ") + cudaGetErrorString(kernelErr));
    }
    
    // NOTE: Don't sync here - let caller control sync point
    
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
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("Tensor::empty: stream is NULL - caller MUST provide valid stream");
    }
    
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
    
    if (grad_ != nullptr && grad_->data != nullptr) {
        return;  // Already allocated
    }
    
    // ISSUE #59: Create gradient as a proper Tensor object
    // The grad Tensor has the same shape as this tensor but doesn't track gradients itself
    auto grad_tensor = std::make_shared<Tensor>();
    grad_tensor->shape = shape;
    grad_tensor->requires_grad = false;  // Gradient tensor doesn't track its own gradient
    grad_tensor->is_leaf = true;
    grad_tensor->stream = stream;
    grad_tensor->device_id = device_id;
    // NOTE: Don't copy name to avoid dangling pointer issues (temporary std::string)
    // The name field is const char* so we can't safely append ".grad" without allocation
    grad_tensor->name = nullptr;
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    cudaError_t err = cudaMalloc(&grad_tensor->data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::ensure_grad cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    grad_tensor->owns_data = true;
    
    // Zero-initialize gradient
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_zero_autograd<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(grad_tensor->data, count);
    
    grad_ = std::move(grad_tensor);
}

void Tensor::zero_grad(cudaStream_t exec_stream) {
    if (!grad_ || !grad_->data) {
        return;  // Nothing to zero
    }
    
    cudaStream_t s = exec_stream ? exec_stream : stream;
    const size_t count = shape.total_elements();
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    kernel_zero_autograd<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, s>>>(grad_->data, count);
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
    kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, s>>>(grad_->data, incoming_grad, count, scale);
}

Tensor Tensor::detach() const {
    Tensor t;
    t.data = data;
    t.shape = shape;
    t.owns_data = false;  // Non-owning view
    t.grad_ = nullptr;    // ISSUE #59: Detached tensor has no gradient
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
            cudaMemcpyAsync(grad_data(), &one, sizeof(float), cudaMemcpyHostToDevice, stream);
        } else {
            // Fill with ones (implicit broadcast for reduction operations)
            std::vector<float> ones(count, 1.0f);
            cudaMemcpyAsync(grad_data(), ones.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    } else {
        // Accumulate provided gradient
        accumulate_grad(grad_output->data, grad_output->numel(), 1.0f, stream);
    }
    
    // Traverse backward through computation graph
    if (grad_fn != nullptr) {
        // Execute backward function - create a view of our gradient
        Tensor grad_tensor;
        grad_tensor.data = grad_data();
        grad_tensor.shape = shape;
        grad_tensor.owns_data = false;  // grad_tensor is a view
        
        grad_fn->apply(grad_tensor, stream);
        
        // ISSUE #52 FIX: Synchronize stream BEFORE release_saved()!
        // release_saved() triggers destructors which call cudaFree().
        // cudaFree blocks if there are pending operations on that memory.
        // We must ensure ALL GPU work is complete before cleanup.
        
        // Check for CUDA errors before sync
        {
            cudaError_t pre_err = cudaGetLastError();
            if (pre_err != cudaSuccess) {
                // RULE 20: Fail Loud with full context!
                std::string ctx = getCurrentGradFnContext();
                std::string msg = "[Tensor::backward] CUDA error before sync: " + 
                    std::string(cudaGetErrorString(pre_err)) + " | " + ctx;
                throw std::runtime_error(msg);
            }
        }
        
        // Query stream state before blocking sync
        {
            cudaError_t query_result = cudaStreamQuery(stream);
            
            // RULE 20: If stream query returns an error (not COMPLETE, not NOT_READY), THROW!
            if (query_result != cudaSuccess && query_result != cudaErrorNotReady) {
                std::string ctx = getCurrentGradFnContext();
                std::string msg = "[Tensor::backward] Stream query detected error: " +
                    std::string(cudaGetErrorString(query_result)) + " | " + ctx;
                throw std::runtime_error(msg);
            }
        }
        
        // Poll stream until complete (non-blocking approach)
        {
            int poll_count = 0;
            cudaError_t query;
            while ((query = cudaStreamQuery(stream)) == cudaErrorNotReady) {
                poll_count++;
                if (poll_count > 100000000) {  // ~10 seconds of spinning
                    std::string ctx = getCurrentGradFnContext();
                    std::string msg = "[Tensor::backward] TIMEOUT: Stream stuck after 100M polls! | " + ctx;
                    
                    cudaError_t err = cudaGetLastError();
                    // RULE 20: Fail Loud - throw with full context
                    throw std::runtime_error(msg);
                }
            }
            if (query != cudaSuccess && query != cudaErrorNotReady) {
                // RULE 20: Fail Loud - error detected during polling
                std::string ctx = getCurrentGradFnContext();
                std::string msg = "[Tensor::backward] Stream poll detected error: " +
                    std::string(cudaGetErrorString(query)) + " | " + ctx;
                throw std::runtime_error(msg);
            }
        }
        
        // ISSUE #53 FIX: Flush deferred cleanup queue AFTER stream sync
        // The deleters queued pointers instead of calling cudaFree directly
        // to avoid blocking. Now that the stream is synced, it's safe to free.
        flushDeferredCleanup();
        
        // Release saved tensors after backward
        grad_fn->release_saved();
    }
    
    // PyTorch: Don't free intermediate gradients during backward!
    // The caches (tensor data) are needed by downstream grad_fns in the backward chain.
    // Freeing them causes SGEMM "illegal value" errors when grad_fns try to use dangling pointers.
    // 
    // Solution: Let gradients accumulate and be cleaned up AFTER the entire backward completes.
    // This matches PyTorch's behavior - gradients persist until explicitly cleared.
    //
    // if (!is_leaf && !retain_grad && owns_grad && grad) {
    //     // DISABLED: Causes SGEMM errors from dangling cache pointers
    // }
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
    
    // Final reduction: only first few threads participate
    // With blockDim.x=256, we have 8 warps, so 8 values to reduce
    // Use thread 0 to do sequential reduction (safe, no warp sync issues)
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        const int num_warps = blockDim.x / 32;
        for (int i = 0; i < num_warps; i++) {
            total += shared[i];
        }
        float rms_sq = total / d_model + eps;
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
//
// ISSUE #55 FIX: The original code used __shfl_down_sync(0xffffffff, ...) with fewer
// than 32 active threads, which is UNDEFINED BEHAVIOR and causes GPU hangs.
// Fixed by using sequential reduction in thread 0 (safe for any warp count).
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
    float* s_warp_vals = shared;  // For warp reduction results
    
    const float* x = input + static_cast<size_t>(token_idx) * d_model;
    const float* dy = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* dx = grad_input + static_cast<size_t>(token_idx) * d_model;
    
    const int num_warps = blockDim.x / 32;
    
    // Step 1: Compute mean(x^2)
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum_sq += x[i] * x[i];
    }
    
    // Warp reduction (within each warp - all 32 threads participate)
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        s_warp_vals[threadIdx.x / 32] = local_sum_sq;
    }
    __syncthreads();
    
    // ISSUE #55 FIX: Use sequential reduction in thread 0 instead of __shfl_down_sync
    // with partial warp (which is undefined behavior and hangs the GPU)
    __shared__ float s_rms_sq, s_inv_rms;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_vals[i];
        }
        s_rms_sq = total / d_model + eps;
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
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dgamma_x += __shfl_down_sync(0xffffffff, local_dgamma_x, offset);
    }
    
    // Write warp results to shared memory
    if (threadIdx.x % 32 == 0) {
        s_warp_vals[threadIdx.x / 32] = local_dgamma_x;
    }
    __syncthreads();
    
    // ISSUE #55 FIX: Sequential reduction in thread 0
    __shared__ float s_dgamma_x;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_vals[i];
        }
        s_dgamma_x = total;
    }
    __syncthreads();
    
    const float dgamma_x_sum = s_dgamma_x;
    
    // Step 3: Compute grad_input and accumulate grad_gamma
    // dx = (dy * gamma - x * dgamma_x_sum / (d_model * rms_sq)) * inv_rms
    const float scale = dgamma_x_sum / (d_model * rms_sq);
    
    // ISSUE #82 REVERTED: The 1/tokens scaling was WRONG!
    // The loss backward already applies 1/tokens through mean reduction (Issue #58),
    // so the incoming dy already reflects this scaling. Adding another 1/tokens
    // makes RMS gradients 1/tokens² too small (~300,000x too small with 3500 tokens).
    // The correct gradient is simply: grad_gamma[i] = sum_t(dy[t,i] * x[t,i] * inv_rms[t])
    
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        dx[i] = (dy[i] * gamma[i] - x[i] * scale) * inv_rms;
        
        // Accumulate grad_gamma: dgamma[i] += dy[i] * x[i] * inv_rms
        // NO 1/tokens scaling - the incoming dy already has loss mean reduction baked in
        if (grad_gamma) {
            atomicAdd(&grad_gamma[i], dy[i] * x[i] * inv_rms);
        }
    }
}

// Embedding forward: gather from embedding table with optional scaling
__global__ void kernel_embedding_forward(
    const int* token_ids,       // [tokens]
    const float* weight,        // [vocab_size, d_model]
    float* output,              // [tokens, d_model]
    int tokens,
    int d_model,
    float embedding_scale       // Scale factor (sqrt(d_model) for AIAYN-style)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const int token_id = token_ids[token_idx];
    const float* weight_row = weight + static_cast<size_t>(token_id) * d_model;
    float* output_row = output + static_cast<size_t>(token_idx) * d_model;
    
    // Gather with scaling: output[token_idx] = weight[token_id] * scale
    // Issue #92: Scale embeddings by sqrt(d_model) to bring Xavier-initialized
    // embeddings (~0.036 rms) to unit scale (~1.0 rms), matching atom injection.
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        output_row[i] = weight_row[i] * embedding_scale;
    }
}

// Embedding backward: scatter-add gradients to embedding table
__global__ void kernel_embedding_backward(
    const float* grad_output,   // [tokens, d_model]
    const int* token_ids,       // [tokens]
    float* grad_weight,         // [vocab_size, d_model]
    int tokens,
    int d_model,
    float embedding_scale       // Scale factor from forward (for chain rule)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const int token_id = token_ids[token_idx];
    const float* token_grad = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* weight_grad = grad_weight + static_cast<size_t>(token_id) * d_model;
    
    // Scatter-add: weight_grad[token_id] += grad_output[token_idx] * scale
    // Chain rule: if forward was y = w * scale, then grad_w = grad_y * scale
    // Issue #92: Embedding scale propagates through backward pass
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        atomicAdd(&weight_grad[i], token_grad[i] * embedding_scale);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════
// ISSUE #60 FIX: PCGrad kernel for tied embedding/lm_head weights
// ═══════════════════════════════════════════════════════════════════════════════════
// Combines LM head gradient (g_lm) with embedding gradient (g_emb) using PCGrad:
//   g_final = g_lm + (g_emb - proj_{g_lm}(g_emb))
//           = g_lm + g_emb - ((g_lm · g_emb) / ||g_lm||²) * g_lm
//
// For each row (one per vocab token):
// - If cosine(g_lm, g_emb) = -1: g_final = g_lm (embedding conflicts, use LM only)
// - If cosine(g_lm, g_emb) = 0:  g_final = g_lm + g_emb (orthogonal, keep both)
// - If cosine(g_lm, g_emb) = +1: g_final = g_lm (avoids double-counting)
// ═══════════════════════════════════════════════════════════════════════════════════
__global__ void kernel_pcgrad_combine(
    float* g_lm,           // [vocab_size, d_model] - LM head gradient (IN-PLACE update)
    const float* g_emb,    // [vocab_size, d_model] - Embedding gradient (temp buffer)
    int vocab_size,
    int d_model
) {
    const int vocab_idx = blockIdx.x;
    if (vocab_idx >= vocab_size) return;
    
    float* lm_row = g_lm + static_cast<size_t>(vocab_idx) * d_model;
    const float* emb_row = g_emb + static_cast<size_t>(vocab_idx) * d_model;
    
    extern __shared__ float shared[];
    float* s_dot_lm_emb = shared;                                // [num_warps]
    float* s_norm_lm_sq = shared + (blockDim.x / 32) + 1;        // [num_warps]
    
    // Step 1: Compute g_lm · g_emb and ||g_lm||² in parallel
    float local_dot = 0.0f;
    float local_norm_sq = 0.0f;
    
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        const float lm_val = lm_row[i];
        const float emb_val = emb_row[i];
        local_dot += lm_val * emb_val;
        local_norm_sq += lm_val * lm_val;
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
        local_norm_sq += __shfl_down_sync(0xffffffff, local_norm_sq, offset);
    }
    
    // Store warp results
    if (threadIdx.x % 32 == 0) {
        s_dot_lm_emb[threadIdx.x / 32] = local_dot;
        s_norm_lm_sq[threadIdx.x / 32] = local_norm_sq;
    }
    __syncthreads();
    
    // Final reduction in thread 0
    __shared__ float s_proj_coef;
    if (threadIdx.x == 0) {
        float total_dot = 0.0f;
        float total_norm_sq = 0.0f;
        const int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            total_dot += s_dot_lm_emb[w];
            total_norm_sq += s_norm_lm_sq[w];
        }
        
        // proj_coef = (g_lm · g_emb) / ||g_lm||²
        // If g_lm is zero, just use g_emb directly (proj_coef = 0)
        s_proj_coef = (total_norm_sq > 1e-12f) ? (total_dot / total_norm_sq) : 0.0f;
    }
    __syncthreads();
    
    const float proj_coef = s_proj_coef;
    
    // Step 2: g_final = g_lm + (g_emb - proj_coef * g_lm)
    //                 = g_lm * (1 - proj_coef) + g_emb
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        lm_row[i] = lm_row[i] * (1.0f - proj_coef) + emb_row[i];
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

/**
 * ISSUE #118: Row-wise centering kernel for forward activation centering
 * Subtracts the mean of each row to remove common direction accumulation.
 * 
 * Forward:  y[t,d] = x[t,d] - mean_d(x[t,:])
 * 
 * One block per row (token), 256 threads cooperatively compute row mean
 * and subtract it from each element.
 *
 * This kernel is used for BOTH forward pass (center activations before residual add)
 * and backward pass (centering is linear, so backward is also centering).
 */
__global__ void kernel_center_rows(
    const float* __restrict__ input,   // [num_rows, row_dim]
    float* __restrict__ output,        // [num_rows, row_dim]
    int row_dim,
    int num_rows
) {
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;
    
    const float* in_row = input + static_cast<size_t>(row_idx) * row_dim;
    float* out_row = output + static_cast<size_t>(row_idx) * row_dim;
    
    // Compute row mean via parallel reduction
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Each thread sums a subset of elements
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_dim; i += blockDim.x) {
        local_sum += in_row[i];
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // First thread in each warp atomically adds to shared sum
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    const float mean = s_sum / static_cast<float>(row_dim);
    
    // Subtract mean from each element
    for (int i = threadIdx.x; i < row_dim; i += blockDim.x) {
        out_row[i] = in_row[i] - mean;
    }
}

/**
 * kernel_center_columns - ISSUE #118 FIX (correct centering dimension!)
 * 
 * Centers each COLUMN by subtracting the mean computed ACROSS ROWS (positions).
 * This removes the "common direction" that causes inter-position correlation.
 * 
 * Formula: out[t,d] = in[t,d] - mean_t(in[:,d])
 *        = in[t,d] - (1/num_rows) × Σ_t(in[t,d])
 * 
 * After centering, each column (feature dimension) sums to zero across all positions.
 * This DOES affect cos(h_i, h_j) because we remove the shared component!
 * 
 * Parallelization: One block per column (d_model columns, each processes num_rows elements)
 */
__global__ void kernel_center_columns(
    const float* __restrict__ input,   // [num_rows, num_cols] row-major
    float* __restrict__ output,        // [num_rows, num_cols] row-major
    int num_cols,                      // d_model (e.g., 768)
    int num_rows                       // total_tokens (e.g., 3500)
) {
    const int col_idx = blockIdx.x;
    if (col_idx >= num_cols) return;
    
    // Compute column mean via parallel reduction across rows
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Each thread sums a subset of row elements in this column
    // Access pattern: input[row * num_cols + col_idx] for each row
    float local_sum = 0.0f;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        local_sum += input[static_cast<size_t>(row) * num_cols + col_idx];
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // First thread in each warp atomically adds to shared sum
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    const float mean = s_sum / static_cast<float>(num_rows);
    
    // Subtract column mean from each row element in this column
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        output[idx] = input[idx] - mean;
    }
}

}  // anonymous namespace

//======================================================//
//  GradFn Subclasses
//======================================================//

/**
 * ScaleGradFn - Backward for scalar multiplication
 * Forward: y = x * scale
 * Backward: grad_x = grad_y * scale (scale is constant, chain rule)
 *
 * ISSUE #98: Added to fix gradient vanishing when tie_embeddings=true.
 * When embeddings are scaled by sqrt(d_model) (Issue #92), the LM head
 * must also scale by sqrt(d_model) to maintain gradient flow symmetry.
 * Without this, gradients to encoder are ~27.7x smaller than they should be.
 */
struct ScaleGradFn : public GradFn {
    float scale_factor = 1.0f;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;
    TensorContract::TensorShape input_shape;
    size_t element_count = 0;
    bool applied = false;
    
    ScaleGradFn() { op_name = "scale"; }
    
    ~ScaleGradFn() {
        if (owns_input_grad_fn && input_grad_fn) {
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
    
    void capture_input(Tensor& input, float scale, cudaStream_t stream) {
        scale_factor = scale;
        input_shape = input.shape;
        element_count = input.numel();
        
        // Take ownership of input's grad_fn
        input_grad_fn = input.grad_fn;
        if (input.grad_fn && input.owns_grad_fn) {
            owns_input_grad_fn = true;
            input.owns_grad_fn = false;
        }
        
        // Setup gradient buffer
        if (input.requires_grad) {
            input.ensure_grad();
            if (input.is_leaf) {
                input_grad = input.grad_data();
                AG_TRACE("[ScaleGradFn] Using persistent input_grad buffer (leaf): %p\n", (void*)input_grad);
            } else {
                float* buffer = nullptr;
                cudaMalloc(&buffer, element_count * sizeof(float));
                cudaMemsetAsync(buffer, 0, element_count * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                input_grad = owned_input_grad.get();
                AG_TRACE("[ScaleGradFn] Allocated owned input_grad buffer (non-leaf): %zu floats at %p\n", element_count, (void*)input_grad);
            }
        }
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) {
            AG_TRACE("[ScaleGradFn] apply() SKIPPED (already applied)\n");
            return;
        }
        applied = true;
        
        // Backward: grad_input = grad_output * scale_factor
        // The scale is a constant, so it multiplies the incoming gradient
        if (input_grad && grad_output.data) {
            const size_t n = element_count;
            const int blocks = (n + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            
            AG_TRACE("[ScaleGradFn] apply() scale_factor=%f n=%zu input_grad=%p\n", 
                     scale_factor, n, (void*)input_grad);
            
            // grad_input += grad_output * scale  (accumulate mode)
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                input_grad, grad_output.data, n, scale_factor);
        }
        
        // Continue backward chain
        if (input_grad_fn) {
            Tensor input_grad_tensor;
            input_grad_tensor.data = input_grad;
            input_grad_tensor.shape = input_shape;
            input_grad_tensor.owns_data = false;
            input_grad_tensor.stream = stream;
            
            input_grad_fn->apply(input_grad_tensor, stream);
            input_grad_fn->release_saved();
        }
    }
    
    __host__ void release_saved() override {
        owned_input_grad.reset();
    }
};

/**
 * LayerScaleGradFn - Backward for learnable scalar multiplication (Issue #109)
 * Forward:  y[i,j] = x[i,j] * scale_param[0]  (broadcast)
 * Backward: grad_x = grad_y * scale_param     (broadcast)
 *           grad_scale = sum(grad_y * x)      (reduction)
 *
 * Similar to ScaleGradFn but scale is a LEARNABLE PARAMETER (not constant).
 */
struct LayerScaleGradFn : public GradFn {
    // Input tensor info
    float* input_data = nullptr;  // Cached for backward
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_data;  // If we need to copy input
    std::shared_ptr<float> owned_input_grad;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;
    TensorContract::TensorShape input_shape;
    size_t element_count = 0;
    
    // Scale param info (shape [1])
    float scale_value = 1.0f;     // Cached scale value for backward
    float* scale_grad = nullptr;  // Points to scale_param's grad
    
    bool applied = false;
    
    LayerScaleGradFn() { op_name = "layer_scale"; }
    
    ~LayerScaleGradFn() {
        if (owns_input_grad_fn && input_grad_fn) {
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
    
    void capture_inputs(Tensor& input, Tensor& scale_param, cudaStream_t stream) {
        input_shape = input.shape;
        element_count = input.numel();
        
        // Cache scale value
        cudaMemcpyAsync(&scale_value, scale_param.data, sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  // Need sync to read value
        
        // Take ownership of input's grad_fn
        input_grad_fn = input.grad_fn;
        if (input.grad_fn && input.owns_grad_fn) {
            owns_input_grad_fn = true;
            input.owns_grad_fn = false;
        }
        
        // Setup gradient buffer for input
        if (input.requires_grad) {
            input.ensure_grad();
            if (input.is_leaf) {
                input_grad = input.grad_data();
            } else {
                float* buffer = nullptr;
                cudaMalloc(&buffer, element_count * sizeof(float));
                cudaMemsetAsync(buffer, 0, element_count * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                input_grad = owned_input_grad.get();
            }
        }
        
        // Setup gradient buffer for scale_param (always scalar [1])
        if (scale_param.requires_grad) {
            scale_param.ensure_grad();
            scale_grad = scale_param.grad_data();
        }
        
        // Cache input data for grad_scale computation
        // For leaf tensors, data persists. For non-leaf, we need to copy.
        if (input.is_leaf) {
            input_data = input.data;
        } else {
            float* buffer = nullptr;
            cudaMalloc(&buffer, element_count * sizeof(float));
            cudaMemcpyAsync(buffer, input.data, element_count * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
            owned_input_data = std::shared_ptr<float>(buffer, [](float* p) {
                queueForDeferredCleanup(p);
            });
            input_data = owned_input_data.get();
        }
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) {
            AG_TRACE("[LayerScaleGradFn] apply() SKIPPED (already applied)\n");
            return;
        }
        applied = true;
        
        const size_t n = element_count;
        const int blocks = (n + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        // 1. grad_input = grad_output * scale_value  (broadcast)
        if (input_grad && grad_output.data) {
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                input_grad, grad_output.data, n, scale_value);
        }
        
        // 2. grad_scale = sum(grad_output * input_data)  (dot product → scalar)
        if (scale_grad && grad_output.data && input_data) {
            // Compute dot product using cublas
            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasSetStream(handle, stream);
            
            float dot_result = 0.0f;
            cublasSdot(handle, static_cast<int>(n), 
                       grad_output.data, 1, input_data, 1, &dot_result);
            
            // Accumulate to scale_grad (atomicAdd for safety)
            // Copy to temp buffer then atomicAdd
            float* h_dot = &dot_result;
            cudaMemcpyAsync(scale_grad, h_dot, sizeof(float), 
                           cudaMemcpyHostToDevice, stream);
            
            cublasDestroy(handle);
        }
        
        // Continue backward chain for input
        if (input_grad_fn) {
            Tensor input_grad_tensor;
            input_grad_tensor.data = input_grad;
            input_grad_tensor.shape = input_shape;
            input_grad_tensor.owns_data = false;
            input_grad_tensor.stream = stream;
            
            input_grad_fn->apply(input_grad_tensor, stream);
            input_grad_fn->release_saved();
        }
    }
    
    __host__ void release_saved() override {
        owned_input_grad.reset();
        owned_input_data.reset();
    }
};

/**
 * CenterRowsGradFn - Backward for row-wise centering (Issue #118)
 * Forward:  y[t,d] = x[t,d] - mean_d(x[t,:])     (per-row mean subtraction)
 * Backward: grad_x[t,d] = grad_y[t,d] - mean_d(grad_y[t,:])  (SAME centering operation!)
 *
 * Mathematical derivation:
 *   y_d = x_d - (1/D) * sum_d(x_d)
 *   Let S = sum_d(x_d), then y_d = x_d - S/D
 *   
 *   dy/dx_d = 1 - 1/D                   (for same dimension d)
 *   dy/dx_k = -1/D                      (for other dimensions k≠d)
 *   
 *   grad_x_d = sum_k (grad_y_k * dy_k/dx_d)
 *            = grad_y_d * (1 - 1/D) + sum_{k≠d} grad_y_k * (-1/D)
 *            = grad_y_d - (1/D) * sum_k(grad_y_k)
 *            = grad_y_d - mean_d(grad_y)
 *
 * BEAUTIFUL: The backward pass is ALSO row-wise centering!
 */
struct CenterRowsGradFn : public GradFn {
    // Stable data (ISSUE #48 pattern - don't store Tensor* pointers)
    float* input_grad = nullptr;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;
    TensorContract::TensorShape input_shape;
    size_t element_count = 0;
    int row_dim = 0;
    int num_rows = 0;
    bool input_requires_grad = false;
    
    // ISSUE #56 FIX: Owned gradient buffer for non-leaf tensors
    std::shared_ptr<float> owned_input_grad;
    
    CenterRowsGradFn() { op_name = "center_rows"; }
    
    ~CenterRowsGradFn() {
        if (owns_input_grad_fn && input_grad_fn) {
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
    
    __host__ void capture_input(Tensor& input, int dim, int rows, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        if (!input.requires_grad) return;
        
        input_shape = input.shape;
        element_count = input.numel();
        row_dim = dim;
        num_rows = rows;
        
        // Take ownership of input's grad_fn (Issue #50 pattern)
        input_grad_fn = input.grad_fn;
        if (input.grad_fn && input.owns_grad_fn) {
            owns_input_grad_fn = true;
            input.owns_grad_fn = false;
        }
        
        // Setup gradient buffer (Issue #54 pattern)
        input.ensure_grad();
        if (input.is_leaf) {
            input_grad = input.grad_data();
            AG_TRACE("[CenterRowsGradFn] Using persistent input_grad buffer (leaf): %p\n", (void*)input_grad);
        } else {
            float* buf = nullptr;
            cudaMalloc(&buf, element_count * sizeof(float));
            cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
            owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
            AG_TRACE("[CenterRowsGradFn] Allocated owned input_grad buffer (non-leaf): %zu floats at %p\n", element_count, (void*)input_grad);
        }
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied || !input_requires_grad || !input_grad || !grad_output.data) {
            return;
        }
        applied = true;
        
        // BACKWARD: grad_x = grad_y - mean_d(grad_y)  (reuse centering kernel!)
        // We can use kernel_center_rows to center grad_output directly to input_grad
        if (input_grad && grad_output.data) {
            kernel_center_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, input_grad, row_dim, num_rows);
        }
        
        // Continue backward chain
        if (input_grad_fn) {
            Tensor input_grad_tensor;
            input_grad_tensor.data = input_grad;
            input_grad_tensor.shape = input_shape;
            input_grad_tensor.owns_data = false;
            input_grad_tensor.stream = stream;
            
            input_grad_fn->apply(input_grad_tensor, stream);
            input_grad_fn->release_saved();
        }
    }
    
    __host__ void release_saved() override {
        owned_input_grad.reset();
    }
};

/**
 * CenterColumnsGradFn - Backward for column-wise centering (Issue #118 proper fix)
 * Forward: y[t,d] = x[t,d] - mean_t(x[:,d])   (centers each column across rows/positions)
 * Backward: Since centering is linear, grad_x = grad_y - mean_t(grad_y[:,d])  (same operation)
 * 
 * This is the CORRECT centering for reducing inter-position correlation:
 * - Removes the common direction that all positions share
 * - Actually reduces cos(h_i, h_j) between different positions
 */
struct CenterColumnsGradFn : public GradFn {
    // Input tensor info (stable data, Issue #48 pattern)
    bool input_requires_grad = false;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int num_cols = 0;      // d_model (number of columns)
    int num_rows = 0;      // total_tokens (number of positions)
    
    // Input gradient chain
    float* input_grad = nullptr;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;
    std::shared_ptr<float> owned_input_grad;
    
    CenterColumnsGradFn() { op_name = "center_columns"; }
    
    ~CenterColumnsGradFn() {
        if (owns_input_grad_fn && input_grad_fn) {
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
    
    __host__ void capture_input(Tensor& input, int cols, int rows, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        if (!input.requires_grad) return;
        
        input_shape = input.shape;
        element_count = input.numel();
        num_cols = cols;
        num_rows = rows;
        
        // Take ownership of input's grad_fn (Issue #50 pattern)
        input_grad_fn = input.grad_fn;
        if (input.grad_fn && input.owns_grad_fn) {
            owns_input_grad_fn = true;
            input.owns_grad_fn = false;
        }
        
        // Setup gradient buffer (Issue #54 pattern)
        input.ensure_grad();
        if (input.is_leaf) {
            input_grad = input.grad_data();
            AG_TRACE("[CenterColumnsGradFn] Using persistent input_grad buffer (leaf): %p\n", (void*)input_grad);
        } else {
            float* buf = nullptr;
            cudaMalloc(&buf, element_count * sizeof(float));
            cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
            owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
            AG_TRACE("[CenterColumnsGradFn] Allocated owned input_grad buffer (non-leaf): %zu floats at %p\n", element_count, (void*)input_grad);
        }
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied || !input_requires_grad || !input_grad || !grad_output.data) {
            return;
        }
        applied = true;
        
        // BACKWARD: grad_x = grad_y - mean_t(grad_y[:,d])  (same centering operation!)
        // Since column-wise centering is linear, backward is same as forward
        if (input_grad && grad_output.data) {
            kernel_center_columns<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, input_grad, num_cols, num_rows);
        }
        
        // Continue backward chain
        if (input_grad_fn) {
            Tensor input_grad_tensor;
            input_grad_tensor.data = input_grad;
            input_grad_tensor.shape = input_shape;
            input_grad_tensor.owns_data = false;
            input_grad_tensor.stream = stream;
            
            input_grad_fn->apply(input_grad_tensor, stream);
            input_grad_fn->release_saved();
        }
    }
    
    __host__ void release_saved() override {
        owned_input_grad.reset();
    }
};

/**
 * AddGradFn - Backward for element-wise addition
 * Forward: c = a + b
 * Backward: grad_a += grad_c, grad_b += grad_c
 */
/**
 * AddGradFn - Backward for element-wise addition (ISSUE #48 FIX)
 * DOES NOT store Tensor* - stores stable data instead
 */
struct AddGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    float* grad_a = nullptr;
    float* grad_b = nullptr;
    // ISSUE #56 FIX: Owned gradient buffers for non-leaf tensors
    // Without these, grad_a/grad_b point to tensor's grad buffer which gets freed
    // when the tensor goes out of scope, causing use-after-free in backward pass
    std::shared_ptr<float> owned_grad_a;
    std::shared_ptr<float> owned_grad_b;
    TensorContract::TensorShape a_shape;
    TensorContract::TensorShape b_shape;
    GradFn* a_grad_fn = nullptr;
    GradFn* b_grad_fn = nullptr;
    bool owns_a_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    bool owns_b_grad_fn = false;
    size_t element_count = 0;
    
    AddGradFn() { op_name = "add"; }
    
    ~AddGradFn() {
        // ISSUE #50: Delete owned grad_fns to complete the chain cleanup
        if (owns_a_grad_fn && a_grad_fn) { 
            delete a_grad_fn; 
            a_grad_fn = nullptr; 
        }
        if (owns_b_grad_fn && b_grad_fn) { 
            delete b_grad_fn; 
            b_grad_fn = nullptr; 
        }
    }
    
    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_shape = a.shape;
        b_shape = b.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fns to prevent use-after-free
        // When the input tensor goes out of scope, it must NOT delete its grad_fn
        // because we need it for the backward chain
        a_grad_fn = a.grad_fn;
        if (a.grad_fn && a.owns_grad_fn) {
            owns_a_grad_fn = true;
            a.owns_grad_fn = false;  // Transfer ownership to us
        }
        b_grad_fn = b.grad_fn;
        if (b.grad_fn && b.owns_grad_fn) {
            owns_b_grad_fn = true;
            b.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        element_count = a.numel();
        
        // ISSUE #56 FIX: Handle grad buffer ownership based on tensor type
        // Leaf tensors (weights) persist, safe to use their grad buffer directly
        // Non-leaf tensors (activations) are temporary, need owned buffer
        if (a_requires_grad) {
            a.ensure_grad();
            if (a.is_leaf) {
                grad_a = a.grad_data();  // ISSUE #59: Use accessor
                AG_TRACE("[AddGradFn] Using persistent grad_a buffer (leaf): %p\n", (void*)grad_a);
            } else {
                const size_t a_numel = a.numel();
                float* buffer_a = nullptr;
                cudaMalloc(&buffer_a, a_numel * sizeof(float));
                cudaMemsetAsync(buffer_a, 0, a_numel * sizeof(float), stream);
                owned_grad_a = std::shared_ptr<float>(buffer_a, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_a = owned_grad_a.get();
                AG_TRACE("[AddGradFn] Allocated owned grad_a buffer (non-leaf): %zu floats at %p\n", a_numel, (void*)grad_a);
            }
        }
        if (b_requires_grad) {
            b.ensure_grad();
            if (b.is_leaf) {
                grad_b = b.grad_data();  // ISSUE #59: Use accessor
                AG_TRACE("[AddGradFn] Using persistent grad_b buffer (leaf): %p\n", (void*)grad_b);
            } else {
                const size_t b_numel = b.numel();
                float* buffer_b = nullptr;
                cudaMalloc(&buffer_b, b_numel * sizeof(float));
                cudaMemsetAsync(buffer_b, 0, b_numel * sizeof(float), stream);
                owned_grad_b = std::shared_ptr<float>(buffer_b, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_b = owned_grad_b.get();
                AG_TRACE("[AddGradFn] Allocated owned grad_b buffer (non-leaf): %zu floats at %p\n", b_numel, (void*)grad_b);
            }
        }
    }
    
    void capture_single_input(Tensor& a, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = false;
        a_shape = a.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn
        a_grad_fn = a.grad_fn;
        if (a.grad_fn && a.owns_grad_fn) {
            owns_a_grad_fn = true;
            a.owns_grad_fn = false;
        }
        
        element_count = a.numel();
        
        // ISSUE #56 FIX: Handle grad buffer ownership for non-leaf tensors
        if (a_requires_grad) {
            a.ensure_grad();
            if (a.is_leaf) {
                grad_a = a.grad_data();  // ISSUE #59: Use accessor
            } else {
                const size_t a_numel = a.numel();
                float* buffer_a = nullptr;
                cudaMalloc(&buffer_a, a_numel * sizeof(float));
                cudaMemsetAsync(buffer_a, 0, a_numel * sizeof(float), stream);
                owned_grad_a = std::shared_ptr<float>(buffer_a, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_a = owned_grad_a.get();
            }
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("add", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        // DIAGNOSTIC: Log incoming gradient to Add backward
        static int s_add_bwd_call = 0;
        const int add_call_idx = ++s_add_bwd_call;
        {
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            std::vector<float> samp(std::min(grad_elems, static_cast<size_t>(10000)));
            cudaMemcpy(samp.data(), grad_output.data, samp.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float mx = 0.0f; double sq = 0.0;
            for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
            float rms = std::sqrt(static_cast<float>(sq / samp.size()));
            const char* a_op = a_grad_fn && a_grad_fn->op_name ? a_grad_fn->op_name : "leaf";
            const char* b_op = b_grad_fn && b_grad_fn->op_name ? b_grad_fn->op_name : "leaf";
            fprintf(stderr, "[ADD-BWD-IN] call=%d | grad: numel=%zu max=%.6f rms=%.10f | a->%s b->%s\n",
                    add_call_idx, grad_elems, mx, rms, a_op, b_op);
        }
        
        if (!grad_output.data) {
            return;
        }
        
        const size_t count = grad_output.numel();
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        // Accumulate gradients to both inputs using stored grad pointers
        if (a_requires_grad && grad_a) {
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_a, grad_output.data, count, 1.0f);
        }
        if (b_requires_grad && grad_b) {
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_b, grad_output.data, count, 1.0f);
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn pointers
        if (a_requires_grad && a_grad_fn && a_grad_fn->op_name) {
            // ISSUE #58 FIX: Pass grad_output.data (incoming gradient), NOT grad_a (local accumulator)
            Tensor view;
            view.data = grad_output.data; view.shape = a_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            a_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here!
        }
        
        if (b_requires_grad && b_grad_fn && b_grad_fn != a_grad_fn && b_grad_fn->op_name) {
            // ISSUE #58 FIX: Pass grad_output.data (incoming gradient), NOT grad_b (local accumulator)
            Tensor view;
            view.data = grad_output.data; view.shape = b_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            b_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here!
        }
    }

    
    void release_saved() override {
        GradFn::release_saved();
        
        grad_a = nullptr;
        grad_b = nullptr;
        
        // ISSUE #50: Set ownership false BEFORE delete to prevent double-delete in destructor
        if (owns_a_grad_fn && a_grad_fn) {
            owns_a_grad_fn = false;
            delete a_grad_fn;
            a_grad_fn = nullptr;
        }
        if (owns_b_grad_fn && b_grad_fn) {
            owns_b_grad_fn = false;
            delete b_grad_fn;
            b_grad_fn = nullptr;
        }
    }
};

/**
 * BiasAddGradFn - Backward for broadcast add: output = input + bias
 * Forward: output[i,j] = input[i,j] + bias[j]  (bias broadcasted)
 * Backward: grad_input = grad_output (pass-through)
 *           grad_bias[j] = sum_i(grad_output[i,j]) (reduction over tokens)
 *
 * ISSUE #97: Encoder biases (b_qkv, b_o, b1, b2) were frozen because
 * launchFFNBiasAdd is a raw CUDA kernel with NO autograd tracking.
 */
struct BiasAddGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    bool bias_requires_grad = false;
    float* grad_input = nullptr;
    float* grad_bias = nullptr;
    // ISSUE #56 FIX: Owned gradient buffers for non-leaf tensors
    std::shared_ptr<float> owned_grad_input;
    std::shared_ptr<float> owned_grad_bias;
    TensorContract::TensorShape input_shape;
    TensorContract::TensorShape bias_shape;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;
    size_t total_tokens = 0;
    size_t features = 0;
    
    BiasAddGradFn() { op_name = "bias_add"; }
    
    ~BiasAddGradFn() {
        // ISSUE #50: Delete owned grad_fn to complete the chain cleanup
        if (owns_input_grad_fn && input_grad_fn) {
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
    
    void capture_inputs(Tensor& input, Tensor& bias, int num_tokens, int num_features, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        bias_requires_grad = bias.requires_grad;
        input_shape = input.shape;
        bias_shape = bias.shape;
        total_tokens = static_cast<size_t>(num_tokens);
        features = static_cast<size_t>(num_features);
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn
        input_grad_fn = input.grad_fn;
        if (input.grad_fn && input.owns_grad_fn) {
            owns_input_grad_fn = true;
            input.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        // Handle input gradient buffer (typically non-leaf activation)
        if (input_requires_grad) {
            input.ensure_grad();
            if (input.is_leaf) {
                grad_input = input.grad_data();
                AG_TRACE("[BiasAddGradFn] Using persistent grad_input buffer (leaf): %p\n", (void*)grad_input);
            } else {
                const size_t input_numel = input.numel();
                float* buffer_input = nullptr;
                cudaMalloc(&buffer_input, input_numel * sizeof(float));
                cudaMemsetAsync(buffer_input, 0, input_numel * sizeof(float), stream);
                owned_grad_input = std::shared_ptr<float>(buffer_input, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_input = owned_grad_input.get();
                AG_TRACE("[BiasAddGradFn] Allocated owned grad_input buffer (non-leaf): %zu floats at %p\n", input_numel, (void*)grad_input);
            }
        }
        
        // Handle bias gradient buffer (typically leaf weight tensor)
        if (bias_requires_grad) {
            bias.ensure_grad();
            if (bias.is_leaf) {
                grad_bias = bias.grad_data();
                AG_TRACE("[BiasAddGradFn] Using persistent grad_bias buffer (leaf): %p\n", (void*)grad_bias);
            } else {
                float* buffer_bias = nullptr;
                cudaMalloc(&buffer_bias, features * sizeof(float));
                cudaMemsetAsync(buffer_bias, 0, features * sizeof(float), stream);
                owned_grad_bias = std::shared_ptr<float>(buffer_bias, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_bias = owned_grad_bias.get();
                AG_TRACE("[BiasAddGradFn] Allocated owned grad_bias buffer (non-leaf): %zu floats at %p\n", features, (void*)grad_bias);
            }
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("bias_add", this);
        
        // ISSUE #49: Prevent infinite loops
        if (applied) {
            return;
        }
        applied = true;
        
        // DIAGNOSTIC: Log incoming gradient and capture state
        static int s_bias_bwd_call = 0;
        const int bias_call_idx = ++s_bias_bwd_call;
        {
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            std::vector<float> samp(std::min(grad_elems, static_cast<size_t>(10000)));
            cudaMemcpy(samp.data(), grad_output.data, samp.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float mx = 0.0f; double sq = 0.0;
            for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
            float rms = std::sqrt(static_cast<float>(sq / samp.size()));
            const char* input_op = input_grad_fn && input_grad_fn->op_name ? input_grad_fn->op_name : "leaf";
            fprintf(stderr, "[BIAS-ADD-BWD-IN] call=%d | grad: numel=%zu max=%.10f rms=%.10f | input->%s | tokens=%zu features=%zu\n",
                    bias_call_idx, grad_elems, mx, rms, input_op, total_tokens, features);
            // ISSUE #97 DIAGNOSTIC: Show bias gradient state
            fprintf(stderr, "[BIAS-ADD-BWD-STATE] call=%d | bias_requires_grad=%d grad_bias=%p\n",
                    bias_call_idx, (int)bias_requires_grad, (void*)grad_bias);
        }
        
        if (!grad_output.data) {
            return;
        }
        
        const size_t count = grad_output.numel();
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        // Backward for input: grad_input = grad_output (pass-through, no shape change)
        if (input_requires_grad && grad_input) {
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_input, grad_output.data, count, 1.0f);
        }
        
        // Backward for bias: grad_bias[j] = sum_i(grad_output[i,j])
        // Use the existing bias backward kernel from Feed_Forward_GPU.cu
        if (bias_requires_grad && grad_bias) {
            launchFFNBiasBackward(grad_output.data, grad_bias, 
                                  static_cast<int>(total_tokens), static_cast<int>(features), stream);
            
            // ISSUE #97 DIAGNOSTIC: Read back and report bias gradient after computation
            {
                cudaStreamSynchronize(stream);
                std::vector<float> bias_grad_host(std::min(features, static_cast<size_t>(100)));
                cudaMemcpy(bias_grad_host.data(), grad_bias, bias_grad_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
                float bg_mx = 0.0f; double bg_sq = 0.0;
                for (auto& v : bias_grad_host) { 
                    if (!std::isnan(v) && !std::isinf(v)) { 
                        bg_mx = std::max(bg_mx, std::abs(v)); 
                        bg_sq += v*v; 
                    } 
                }
                float bg_rms = std::sqrt(static_cast<float>(bg_sq / bias_grad_host.size()));
                fprintf(stderr, "[BIAS-ADD-BWD-DONE] call=%d | grad_bias: max=%.10f rms=%.10f (sampled %zu/%zu features)\n",
                        bias_call_idx, bg_mx, bg_rms, bias_grad_host.size(), features);
            }
        } else {
            fprintf(stderr, "[BIAS-ADD-BWD-SKIP] call=%d | bias_requires_grad=%d grad_bias=%p - NOT computing bias gradient!\n",
                    bias_call_idx, (int)bias_requires_grad, (void*)grad_bias);
        }
        
        // CONTINUE AUTOGRAD CHAIN for input
        if (input_requires_grad && input_grad_fn && input_grad_fn->op_name) {
            Tensor view;
            view.data = grad_output.data;  // ISSUE #58: Pass incoming gradient
            view.shape = input_shape;
            view.owns_data = false;
            view.owns_grad_fn = false;
            view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        grad_input = nullptr;
        grad_bias = nullptr;
        if (owns_input_grad_fn && input_grad_fn) {
            owns_input_grad_fn = false;
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
};

/**
 * GeluGradFn - Backward for GELU activation (TAPE-BASED)
 * Forward: y = gelu(x)
 * Backward: grad_x = grad_y * gelu'(x)
 * 
 * TAPE-BASED: Does NOT allocate or copy. References external cache from TrainingState.
 */
/**
 * GeluGradFn - Backward for GELU activation (ISSUE #48 FIX)
 * TAPE-BASED: References external cache, DOES NOT store Tensor*
 */
struct GeluGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    // ISSUE #56 FIX: Owned gradient buffer for non-leaf tensors
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    // ISSUE #51 FIX: Own a copy of cached data instead of non-owning pointer
    std::shared_ptr<float> owned_cache;
    const float* cached_input = nullptr;  // Points to owned_cache.get()
    size_t cached_size = 0;
    
    GeluGradFn() { op_name = "gelu"; }
    
    ~GeluGradFn() {
        // ISSUE #50: Delete owned grad_fn to complete the chain cleanup
        if (owns_input_grad_fn && input_grad_fn) { 
            delete input_grad_fn; 
            input_grad_fn = nullptr; 
        }
        // shared_ptr members destruct automatically after this
    }
    
    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn to prevent use-after-free
        input_grad_fn = x.grad_fn;
        if (x.grad_fn && x.owns_grad_fn) {
            owns_input_grad_fn = true;
            x.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        // ISSUE #56 FIX: Handle grad buffer ownership for non-leaf tensors
        if (input_requires_grad) {
            x.ensure_grad();
            if (x.is_leaf) {
                input_grad = x.grad_data();  // ISSUE #59: Use accessor
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMalloc(&buffer, x_numel * sizeof(float));
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }
    
    // ISSUE #51 FIX: Copy cache data to owned buffer instead of storing dangling pointer
    void set_cache_copy(const float* external_cache, size_t size, cudaStream_t stream) {
        if (!external_cache) {
            throw std::runtime_error("GeluGradFn::set_cache_copy: external_cache is NULL - caller MUST provide cache");
        }
        cached_size = size;
        
        // Allocate and copy to owned buffer
        float* buffer = nullptr;
        cudaMalloc(&buffer, size * sizeof(float));
        cudaMemcpyAsync(buffer, external_cache, size * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Wrap in shared_ptr with DEFERRED cleanup deleter (Issue #53)
        // Instead of calling cudaFree directly (which blocks), queue for later cleanup
        owned_cache = std::shared_ptr<float>(buffer, [](float* p) {
            queueForDeferredCleanup(p);
        });
        cached_input = owned_cache.get();
        AG_TRACE("[GeluGradFn] Copied cache: %zu floats to %p\n", size, (void*)cached_input);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("gelu", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        // DIAGNOSTIC: Log incoming gradient to GELU backward
        static int s_gelu_bwd_call = 0;
        const int gelu_call_idx = ++s_gelu_bwd_call;
        {
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            std::vector<float> samp(std::min(grad_elems, static_cast<size_t>(10000)));
            cudaMemcpy(samp.data(), grad_output.data, samp.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float mx = 0.0f; double sq = 0.0;
            for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
            float rms = std::sqrt(static_cast<float>(sq / samp.size()));
            fprintf(stderr, "[GELU-BWD-IN] call=%d | grad: numel=%zu max=%.6f rms=%.6f\n",
                    gelu_call_idx, grad_elems, mx, rms);
        }
        
        if (!input_requires_grad) {
            return;  // Nothing to do
        }
        if (!input_grad) {
            throw std::runtime_error("GeluGradFn::apply: input_grad is NULL - capture_input() must be called first");
        }
        if (!cached_input) {
            throw std::runtime_error("GeluGradFn::apply: cached_input is NULL - set_cache() must be called first");
        }
        
        const size_t count = grad_output.numel();
        if (count != cached_size) {
            throw std::runtime_error("GeluGradFn::apply: size mismatch - grad_output.numel()=" + 
                                     std::to_string(count) + " cached_size=" + std::to_string(cached_size));
        }
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        // TAPE-BASED: Write directly to stored grad buffer
        kernel_gelu_backward<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_input, input_grad, count);
        trackKernelLaunch("kernel_gelu_backward", stream);

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        cached_input = nullptr;
        cached_size = 0;
        input_grad = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_input_grad_fn && input_grad_fn) { owns_input_grad_fn = false; delete input_grad_fn; input_grad_fn = nullptr; }
    }
};

/**
 * RMSNormGradFn - Backward for RMSNorm
 * Forward: y = x / rms(x) * gamma
 * Backward: Complex chain rule through normalization
 */
/**
 * RMSNormGradFn - Backward for RMS Normalization (ISSUE #48 FIX)
 * TAPE-BASED: References external cache, DOES NOT store Tensor*
 */
struct RMSNormGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    bool gamma_requires_grad = false;
    float* input_grad = nullptr;
    float* gamma_grad_ptr = nullptr;  // For gamma gradient
    float* gamma_data = nullptr;      // Gamma weights data (for forward values)
    TensorContract::TensorShape input_shape;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    // ISSUE #51 FIX: Own a copy of cached data instead of non-owning pointer
    std::shared_ptr<float> owned_cache;
    const float* cached_input = nullptr;  // Points to owned_cache.get()
    size_t cached_size = 0;
    int d_model = 0;
    float eps = 1e-5f;
    
    RMSNormGradFn() { op_name = "rms_norm"; }
    
    ~RMSNormGradFn() {
        // ISSUE #50: Delete owned grad_fn to complete the chain cleanup
        if (owns_input_grad_fn && input_grad_fn) { 
            delete input_grad_fn; 
            input_grad_fn = nullptr; 
        }
        // shared_ptr members destruct automatically after this
    }
    
    void capture_inputs(Tensor& x, Tensor& gamma_tensor) {
        input_requires_grad = x.requires_grad;
        gamma_requires_grad = gamma_tensor.requires_grad;
        input_shape = x.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn to prevent use-after-free
        input_grad_fn = x.grad_fn;
        if (x.grad_fn && x.owns_grad_fn) {
            owns_input_grad_fn = true;
            x.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        gamma_data = gamma_tensor.data;
        
        if (input_requires_grad) {
            x.ensure_grad();
            input_grad = x.grad_data();  // ISSUE #59: Use accessor
        }
        if (gamma_requires_grad) {
            gamma_tensor.ensure_grad();
            gamma_grad_ptr = gamma_tensor.grad_data();  // ISSUE #59: Use accessor
        }
    }
    
    // ISSUE #51 FIX: Copy cache data to owned buffer instead of storing dangling pointer
    void set_cache_copy(const float* external_cache, size_t size, int d, float e, cudaStream_t stream) {
        if (!external_cache) {
            throw std::runtime_error("RMSNormGradFn::set_cache_copy: external_cache is NULL - caller MUST provide cache");
        }
        cached_size = size;
        d_model = d;
        eps = e;
        
        // Allocate and copy to owned buffer
        float* buffer = nullptr;
        cudaMalloc(&buffer, size * sizeof(float));
        cudaMemcpyAsync(buffer, external_cache, size * sizeof(float), 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Wrap in shared_ptr with DEFERRED cleanup deleter (Issue #53)
        // Instead of calling cudaFree directly (which blocks), queue for later cleanup
        owned_cache = std::shared_ptr<float>(buffer, [](float* p) {
            queueForDeferredCleanup(p);
        });
        cached_input = owned_cache.get();
        AG_TRACE("[RMSNormGradFn] Copied cache: %zu floats to %p\n", size, (void*)cached_input);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("rms_norm", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        // DIAGNOSTIC: Log incoming gradient to RMSNorm backward
        static int s_rms_bwd_call = 0;
        const int rms_call_idx = ++s_rms_bwd_call;
        {
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            std::vector<float> samp(std::min(grad_elems, static_cast<size_t>(10000)));
            cudaMemcpy(samp.data(), grad_output.data, samp.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float mx = 0.0f; double sq = 0.0;
            for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
            float rms = std::sqrt(static_cast<float>(sq / samp.size()));
            fprintf(stderr, "[RMS-BWD-IN] call=%d | grad: numel=%zu max=%.10f rms=%.10f\n",
                    rms_call_idx, grad_elems, mx, rms);
        }
        
        if (!cached_input) {
            throw std::runtime_error("RMSNormGradFn::apply: cached_input is NULL - set_cache() must be called first");
        }
        if (d_model <= 0) {
            throw std::runtime_error("RMSNormGradFn::apply: d_model is " + std::to_string(d_model) + " - must be > 0");
        }
        
        const int tokens = static_cast<int>(cached_size / d_model);
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
        
        if (input_requires_grad && input_grad) {
            kernel_rmsnorm_backward<<<tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
                grad_output.data, cached_input, gamma_data,
                input_grad, gamma_grad_ptr,
                tokens, d_model, eps);
            trackKernelLaunch("kernel_rmsnorm_backward", stream);

            // CONTINUE AUTOGRAD CHAIN using stored grad_fn
            if (input_grad_fn && input_grad_fn->op_name) {
                Tensor view;
                view.data = input_grad; view.shape = input_shape;
                view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
                input_grad_fn->apply(view, stream);
                // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
            }
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        cached_input = nullptr;
        cached_size = 0;
        input_grad = nullptr;
        gamma_grad_ptr = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_input_grad_fn && input_grad_fn) { owns_input_grad_fn = false; delete input_grad_fn; input_grad_fn = nullptr; }
    }
};

/**
 * EmbeddingGradFn - Backward for embedding lookup (ISSUE #48 FIX)
 * DOES NOT store Tensor* - stores stable data instead
 */
struct EmbeddingGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool weight_requires_grad = false;
    float* weight_grad = nullptr;
    TensorContract::TensorShape weight_shape;
    GradFn* weight_grad_fn = nullptr;
    bool owns_weight_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    int* token_ids = nullptr;
    bool owns_token_ids = false;
    int num_tokens = 0;
    int d_model = 0;
    float embedding_scale = 1.0f;  // ISSUE #92: Scale factor for AIAYN-style embeddings
    
    EmbeddingGradFn() { op_name = "embedding"; }
    
    ~EmbeddingGradFn() override {
        if (owns_token_ids && token_ids) cudaFree(token_ids);
        // ISSUE #50: Delete owned grad_fn to complete the chain cleanup
        if (owns_weight_grad_fn && weight_grad_fn) { delete weight_grad_fn; weight_grad_fn = nullptr; }
    }
    
    void capture_weight(Tensor& w) {
        weight_requires_grad = w.requires_grad;
        weight_shape = w.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn to prevent use-after-free
        weight_grad_fn = w.grad_fn;
        if (w.grad_fn && w.owns_grad_fn) {
            owns_weight_grad_fn = true;
            w.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        if (weight_requires_grad) {
            w.ensure_grad();
            weight_grad = w.grad_data();  // ISSUE #59: Use accessor
        }
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
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("embedding", this);
        
        // ISSUE #60 DEBUG: Confirm EmbeddingGradFn is being called (stdout for training log)
        fprintf(stdout, "[EmbeddingGradFn::apply] ENTER - num_tokens=%d d_model=%d pcgrad_buffer=%p pcgrad_size=%zu\n",
                num_tokens, d_model, (void*)g_pcgrad_temp_buffer, g_pcgrad_buffer_size);
        fflush(stdout);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            fprintf(stdout, "[EmbeddingGradFn::apply] SKIP - already applied\n");
            fflush(stdout);
            return;
        }
        applied = true;
        
        if (!weight_requires_grad || !token_ids) {
            fprintf(stdout, "[EmbeddingGradFn::apply] SKIP - no grad needed (requires_grad=%d, ids=%p)\n",
                    weight_requires_grad, (void*)token_ids);
            fflush(stdout);
            return;
        }
        if (!weight_grad) {
            throw std::runtime_error("EmbeddingGradFn::apply: weight_grad is NULL - capture_weight() must be called first");
        }
        
        // ═══════════════════════════════════════════════════════════════════════════
        // ISSUE #60 FIX: PCGrad for tied embedding/lm_head weights
        // ═══════════════════════════════════════════════════════════════════════════
        // Problem: LM head backward and embedding backward produce OPPOSITE gradients!
        //   LM_HEAD:    wants token 277 to INCREASE (positive gradient)
        //   EMBEDDING:  wants token 277 to DECREASE (negative gradient)
        //   COMBINED:   cancels to ZERO!
        //
        // Solution: PCGrad - project out the conflicting component:
        //   g_final = g_lm + (g_emb - proj_{g_lm}(g_emb))
        //           = g_lm + orthogonal_component(g_emb)
        //
        // This preserves the LM head signal while keeping any NOVEL information
        // from embedding gradient that isn't redundant or conflicting.
        // ═══════════════════════════════════════════════════════════════════════════
        
        // Embedding shape is [vocab_size, d_model] in BSM layout
        const int vocab_size = weight_shape.as_2d().rows;
        
        if (g_pcgrad_temp_buffer && g_pcgrad_buffer_size >= static_cast<size_t>(vocab_size) * d_model) {
            // PCGrad mode: compute embedding gradient into temp buffer, then combine
            AG_TRACE("[EmbeddingGradFn] Using PCGrad mode for tied weights\n");
            
            // Step 1: Zero the temp buffer
            cudaMemsetAsync(g_pcgrad_temp_buffer, 0, 
                           static_cast<size_t>(vocab_size) * d_model * sizeof(float), stream);
            
            // Step 2: Compute embedding backward into temp buffer (NOT shared grad buffer)
            kernel_embedding_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, token_ids, g_pcgrad_temp_buffer, num_tokens, d_model, embedding_scale);
            trackKernelLaunch("kernel_embedding_backward_pcgrad", stream);
            
            // ISSUE #60 DEBUG: Capture RAW embedding gradient BEFORE PCGrad projection
            // Need sync to ensure kernel completes before capture
            fprintf(stdout, "[EmbeddingGradFn] PCGrad: embedding backward kernel launched, checking debug capture...\n");
            fprintf(stdout, "  g_debug_capture_enabled=%d g_debug_embedding_only_grad=%p\n",
                    g_debug_capture_enabled, (void*)g_debug_embedding_only_grad);
            fflush(stdout);
            
            if (g_debug_capture_enabled && g_debug_embedding_only_grad) {
                cudaStreamSynchronize(stream);  // Ensure embedding backward completes
                const size_t total_size = weight_shape.total_elements();
                fprintf(stdout, "[EmbeddingGradFn] DEBUG: Capturing raw emb grad from pcgrad_temp (size=%zu, vocab=%d, d=%d, tokens=%d)\n",
                        total_size, vocab_size, d_model, num_tokens);
                fflush(stdout);
                
                // Sample a few values from pcgrad_temp to verify it's not empty
                float sample[5];
                cudaMemcpy(sample, g_pcgrad_temp_buffer + 277 * d_model, 5 * sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stdout, "[EmbeddingGradFn] Token 277 first 5 values in pcgrad_temp: [%.6e, %.6e, %.6e, %.6e, %.6e]\n",
                        sample[0], sample[1], sample[2], sample[3], sample[4]);
                fflush(stdout);
                
                debugCaptureEmbeddingGrad(g_pcgrad_temp_buffer, total_size, stream);
                fprintf(stdout, "[EmbeddingGradFn] DEBUG: Captured raw embedding gradient (pre-PCGrad)\n");
                fflush(stdout);
            }
            
            // Step 3: Apply PCGrad - combine LM head grad (in weight_grad) with orthogonal 
            //         component of embedding grad (in temp buffer)
            const int shared_mem = 2 * ((AUTOGRAD_BLOCK_SIZE / 32) + 1) * sizeof(float);
            kernel_pcgrad_combine<<<vocab_size, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
                weight_grad, g_pcgrad_temp_buffer, vocab_size, d_model);
            trackKernelLaunch("kernel_pcgrad_combine", stream);
            
            fprintf(stdout, "[EmbeddingGradFn] PCGrad combination complete\n");
            fflush(stdout);
            
        } else if (g_skip_embedding_backward_for_tied_weights) {
            // Fallback: skip embedding backward entirely (previous fix)
            AG_TRACE("[EmbeddingGradFn] SKIPPING embedding backward - weights tied, no PCGrad buffer\n");
        } else {
            // No PCGrad, no skip: run embedding backward normally (will cancel!)
            kernel_embedding_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, token_ids, weight_grad, num_tokens, d_model, embedding_scale);
            trackKernelLaunch("kernel_embedding_backward", stream);
            
            // DEBUG: Capture embedding gradient (non-PCGrad path)
            if (g_debug_capture_enabled && g_debug_embedding_only_grad && weight_grad) {
                const size_t total_size = weight_shape.total_elements();
                debugCaptureEmbeddingGrad(weight_grad, total_size, stream);
            }
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (weight_grad_fn) {
            Tensor view;
            view.data = weight_grad; view.shape = weight_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            weight_grad_fn->apply(view, stream);
            weight_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (owns_token_ids && token_ids) { cudaFree(token_ids); token_ids = nullptr; }
        weight_grad = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_weight_grad_fn && weight_grad_fn) { owns_weight_grad_fn = false; delete weight_grad_fn; weight_grad_fn = nullptr; }
    }
};

/**
 * SoftmaxGradFn - Backward for softmax operation (ISSUE #48 FIX)
 * DOES NOT store Tensor* - stores stable data instead
 */
struct SoftmaxGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    TensorContract::TensorShape input_shape;
    GradFn* input_grad_fn = nullptr;
    float* saved_softmax = nullptr;  // Saved softmax output
    int num_tokens = 0;
    int dim = 0;
    
    SoftmaxGradFn() { op_name = "softmax"; }
    
    ~SoftmaxGradFn() override {
        if (saved_softmax) cudaFree(saved_softmax);
        // ISSUE #50: Delete owned grad_fn to complete the chain cleanup
        if (owns_input_grad_fn && input_grad_fn) { delete input_grad_fn; input_grad_fn = nullptr; }
    }
    
    bool owns_input_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    
    void capture_input(Tensor& x) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn to prevent use-after-free
        input_grad_fn = x.grad_fn;
        if (x.grad_fn && x.owns_grad_fn) {
            owns_input_grad_fn = true;
            x.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        if (input_requires_grad) {
            x.ensure_grad();
            input_grad = x.grad_data();  // ISSUE #59: Use accessor
        }
    }
    
    void save(const float* softmax_output, int tokens, int d, cudaStream_t stream) {
        num_tokens = tokens;
        dim = d;
        
        size_t bytes = static_cast<size_t>(tokens) * d * sizeof(float);
        cudaMalloc(&saved_softmax, bytes);
        cudaMemcpyAsync(saved_softmax, softmax_output, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("softmax", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        if (!input_requires_grad || !saved_softmax) {
            return;
        }
        if (!input_grad) {
            throw std::runtime_error("SoftmaxGradFn::apply: input_grad is NULL - capture_input() must be called first");
        }
        
        // Shared memory for dot product reduction: one float per warp
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
        
        kernel_softmax_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
            grad_output.data, saved_softmax, input_grad, num_tokens, dim);
        trackKernelLaunch("kernel_softmax_backward", stream);

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            input_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_softmax) { cudaFree(saved_softmax); saved_softmax = nullptr; }
        input_grad = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_input_grad_fn && input_grad_fn) { owns_input_grad_fn = false; delete input_grad_fn; input_grad_fn = nullptr; }
    }
};

/**
 * DropoutGradFn - Backward for dropout operation (ISSUE #48 FIX)
 * DOES NOT store Tensor* - stores stable data instead
 */
struct DropoutGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    TensorContract::TensorShape input_shape;
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    uint8_t* saved_mask = nullptr;  // Binary mask from forward
    float scale = 1.0f;             // 1.0 / (1.0 - dropout_prob)
    size_t count = 0;
    
    DropoutGradFn() { op_name = "dropout"; }
    
    ~DropoutGradFn() override {
        if (saved_mask) cudaFree(saved_mask);
        // ISSUE #50: Delete owned grad_fn to complete the chain cleanup
        if (owns_input_grad_fn && input_grad_fn) { delete input_grad_fn; input_grad_fn = nullptr; }
    }
    
    void capture_input(Tensor& x) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fn to prevent use-after-free
        input_grad_fn = x.grad_fn;
        if (x.grad_fn && x.owns_grad_fn) {
            owns_input_grad_fn = true;
            x.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        if (input_requires_grad) {
            x.ensure_grad();
            input_grad = x.grad_data();  // ISSUE #59: Use accessor
        }
    }
    
    void save(const uint8_t* mask, float dropout_prob, size_t n, cudaStream_t stream) {
        count = n;
        scale = (dropout_prob < 1.0f) ? 1.0f / (1.0f - dropout_prob) : 0.0f;
        
        cudaMalloc(&saved_mask, n * sizeof(uint8_t));
        cudaMemcpyAsync(saved_mask, mask, n * sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("dropout", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        if (!input_requires_grad || !saved_mask) {
            return;
        }
        if (!input_grad) {
            throw std::runtime_error("DropoutGradFn::apply: input_grad is NULL - capture_input() must be called first");
        }
        
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        kernel_dropout_backward<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, saved_mask, input_grad, scale, count);
        trackKernelLaunch("kernel_dropout_backward", stream);

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            input_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_mask) { cudaFree(saved_mask); saved_mask = nullptr; }
        input_grad = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_input_grad_fn && input_grad_fn) { owns_input_grad_fn = false; delete input_grad_fn; input_grad_fn = nullptr; }
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
/**
 * ResidualAddGradFn - Backward for residual/skip connections (ISSUE #48 FIX)
 * DOES NOT store Tensor* - stores stable data instead
 */
struct ResidualAddGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    bool residual_requires_grad = false;
    float* input_grad = nullptr;
    float* residual_grad = nullptr;
    // ISSUE #56 FIX: Owned gradient buffers for non-leaf tensors
    std::shared_ptr<float> owned_input_grad;
    std::shared_ptr<float> owned_residual_grad;
    TensorContract::TensorShape input_shape;
    TensorContract::TensorShape residual_shape;
    GradFn* input_grad_fn = nullptr;
    GradFn* residual_grad_fn = nullptr;
    bool owns_input_grad_fn = false;     // ISSUE #50: Take ownership to prevent use-after-free
    bool owns_residual_grad_fn = false;
    size_t element_count = 0;
    
    ResidualAddGradFn() { op_name = "residual_add"; }
    
    ~ResidualAddGradFn() {
        // ISSUE #50: Delete owned grad_fns to complete the chain cleanup
        if (owns_input_grad_fn && input_grad_fn) { delete input_grad_fn; input_grad_fn = nullptr; }
        if (owns_residual_grad_fn && residual_grad_fn) { delete residual_grad_fn; residual_grad_fn = nullptr; }
    }
    
    void capture_inputs(Tensor& x, Tensor& r, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        residual_requires_grad = r.requires_grad;
        input_shape = x.shape;
        residual_shape = r.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fns to prevent use-after-free
        input_grad_fn = x.grad_fn;
        if (x.grad_fn && x.owns_grad_fn) {
            owns_input_grad_fn = true;
            x.owns_grad_fn = false;  // Transfer ownership to us
        }
        residual_grad_fn = r.grad_fn;
        if (r.grad_fn && r.owns_grad_fn) {
            owns_residual_grad_fn = true;
            r.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        element_count = x.numel();
        
        // ISSUE #56 FIX: Handle grad buffer ownership for non-leaf tensors
        if (input_requires_grad) {
            x.ensure_grad();
            if (x.is_leaf) {
                input_grad = x.grad_data();  // ISSUE #59: Use accessor
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMalloc(&buffer, x_numel * sizeof(float));
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
        if (residual_requires_grad) {
            r.ensure_grad();
            if (r.is_leaf) {
                residual_grad = r.grad_data();  // ISSUE #59: Use accessor
            } else {
                const size_t r_numel = r.numel();
                float* buffer = nullptr;
                cudaMalloc(&buffer, r_numel * sizeof(float));
                cudaMemsetAsync(buffer, 0, r_numel * sizeof(float), stream);
                owned_residual_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                residual_grad = owned_residual_grad.get();
            }
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("residual_add", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        // Both inputs receive the same gradient (d(x+r)/dx = 1, d(x+r)/dr = 1)
        const size_t count = grad_output.numel();
        const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
        
        if (input_requires_grad && input_grad) {
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                input_grad, grad_output.data, count, 1.0f);
        }
        
        if (residual_requires_grad && residual_grad) {
            kernel_accumulate_grad<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                residual_grad, grad_output.data, count, 1.0f);
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn pointers
        if (input_requires_grad && input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            input_grad_fn->release_saved();
        }
        if (residual_requires_grad && residual_grad_fn && residual_grad_fn != input_grad_fn) {
            Tensor view;
            view.data = residual_grad; view.shape = residual_shape;
            view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
            residual_grad_fn->apply(view, stream);
            residual_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        input_grad = nullptr;
        residual_grad = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_input_grad_fn && input_grad_fn) { owns_input_grad_fn = false; delete input_grad_fn; input_grad_fn = nullptr; }
        if (owns_residual_grad_fn && residual_grad_fn) { owns_residual_grad_fn = false; delete residual_grad_fn; residual_grad_fn = nullptr; }
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
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new AddGradFn();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);  // ISSUE #56 FIX
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
    }
    
    return result;
}

/**
 * autograd::broadcast_add - Add bias to tensor with broadcasting: output = input + bias
 * 
 * ISSUE #97 FIX: Encoder biases were frozen because launchFFNBiasAdd bypassed autograd.
 * This function provides proper gradient tracking for the bias parameter.
 *
 * @param input Input tensor [N, D] where N=total_tokens, D=features
 * @param bias Bias tensor [D] - will be broadcasted to [N, D]
 * @param stream CUDA stream
 * @return Output tensor [N, D] with autograd graph attached
 */
Tensor broadcast_add(const Tensor& input, const Tensor& bias, cudaStream_t stream) {
    // RULE 20: Validate inputs
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::broadcast_add: stream is NULL - caller MUST provide valid stream");
    }
    if (!input.data) {
        throw std::runtime_error("autograd::broadcast_add: input.data is NULL");
    }
    if (!bias.data) {
        throw std::runtime_error("autograd::broadcast_add: bias.data is NULL");
    }
    
    // Extract dimensions from shapes
    // input: [N, D] or [B*S, D] (2D flat layout), bias: [D]
    if (!input.shape.is_flat()) {
        throw std::runtime_error("autograd::broadcast_add: input must have 2D flat layout (BSM)");
    }
    const int total_tokens = input.shape.flat.rows;
    const int features = input.shape.flat.cols;
    const int bias_size = static_cast<int>(bias.numel());
    
    if (features != bias_size) {
        throw std::runtime_error("autograd::broadcast_add: feature dimension mismatch. input features=" + 
                                 std::to_string(features) + " bias size=" + std::to_string(bias_size));
    }
    
    // Create output tensor (same shape as input)
    Tensor result = Tensor::empty(input.shape, input.requires_grad || bias.requires_grad, stream);
    
    // Forward: Copy input to output, then add bias in-place
    const size_t total_bytes = static_cast<size_t>(total_tokens) * features * sizeof(float);
    cudaMemcpyAsync(result.data, input.data, total_bytes, cudaMemcpyDeviceToDevice, stream);
    
    // Use the existing FFN bias add kernel for the forward pass
    launchFFNBiasAdd(result.data, bias.data, total_tokens, features, stream);
    
    // Set up backward - ISSUE #97: This was missing for encoder biases!
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new BiasAddGradFn();
        grad_fn->capture_inputs(const_cast<Tensor&>(input), const_cast<Tensor&>(bias), 
                                total_tokens, features, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
    }
    
    AG_TRACE("[autograd::broadcast_add] input[%d,%d] + bias[%d] -> output[%d,%d] requires_grad=%d\n",
             total_tokens, features, bias_size, total_tokens, features, result.requires_grad);
    
    // ISSUE #97 DIAGNOSTIC: Always print for debugging
    static std::atomic<int> broadcast_add_call_idx{0};
    const int fwd_call_idx = broadcast_add_call_idx.fetch_add(1);
    fprintf(stderr, "[BIAS-ADD-FWD] call=%d | input[%d,%d] + bias[%d] req_grad: input=%d bias=%d out=%d bias.grad_data=%p\n",
            fwd_call_idx, total_tokens, features, bias_size, 
            input.requires_grad, bias.requires_grad, result.requires_grad,
            (void*)const_cast<Tensor&>(bias).grad_data());
    
    return result;
}

/**
 * autograd::gelu - GELU activation with automatic differentiation
 *
 * TAPE-BASED: Requires caller to provide cache pointer from TrainingState.
 * Does NOT allocate internal copy of input.
 *
 * @param x Input tensor
 * @param stream CUDA stream
 * @param input_cache External cache pointer for input (needed for backward).
 *                    Must point to valid memory until backward() is called.
 * @return Output tensor with GELU applied
 */
Tensor gelu(const Tensor& x, cudaStream_t stream, const float* input_cache) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::gelu: stream is NULL - caller MUST provide valid stream");
    }
    
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream);
    
    // Forward: y = gelu(x)
    // gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const size_t count = x.numel();
    const int blocks = (count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
    
    // Launch GELU forward kernel
    kernel_gelu_forward<<<blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (x.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new GeluGradFn();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        
        // ISSUE #51 FIX: Copy cache to owned buffer
        const float* effective_cache = input_cache ? input_cache : x.data;
        grad_fn->set_cache_copy(effective_cache, count, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
    }
    
    return result;
}

/**
 * autograd::rms_norm - RMS Normalization with automatic differentiation
 *
 * TAPE-BASED: Requires caller to provide cache pointer from TrainingState.
 * Does NOT allocate internal copy of input.
 *
 * @param x Input tensor [tokens, d_model]
 * @param gamma Scale parameter [d_model]
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 * @param input_cache External cache pointer for input (needed for backward).
 *                    Must point to valid memory until backward() is called.
 * @return Output tensor with RMSNorm applied
 */
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps, cudaStream_t stream,
                const float* input_cache) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::rms_norm: stream is NULL - caller MUST provide valid stream");
    }
    
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
    
    // [RMSNORM_EQUATION] Rule 21 diagnostic - verify RMSNorm math
    // Equation: y = x * gamma * inv_rms, where inv_rms = 1/sqrt(mean(x²) + eps)
    // Expected: output_rms = gamma * x_rms * inv_rms = gamma * x_rms / x_rms = gamma ≈ 1.0
    #ifdef DEBUG_RMSNORM_EQUATION
    {
        cudaStreamSynchronize(stream);
        const int sample_rows = std::min(5, tokens);
        std::vector<float> h_input(sample_rows * d_model);
        std::vector<float> h_output(sample_rows * d_model);
        std::vector<float> h_gamma(d_model);
        cudaMemcpy(h_input.data(), x.data, h_input.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output.data(), result.data, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gamma.data(), gamma.data, h_gamma.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute gamma stats
        float gamma_min = h_gamma[0], gamma_max = h_gamma[0];
        double gamma_sum_sq = 0.0;
        for (int i = 0; i < d_model; i++) {
            gamma_min = std::min(gamma_min, h_gamma[i]);
            gamma_max = std::max(gamma_max, h_gamma[i]);
            gamma_sum_sq += h_gamma[i] * h_gamma[i];
        }
        float gamma_rms = std::sqrt(gamma_sum_sq / d_model);
        
        fprintf(stderr, "[RMSNORM_EQUATION] y = x * gamma * inv_rms, inv_rms = 1/sqrt(mean(x²) + eps)\n");
        fprintf(stderr, "  INPUT x: shape=[%d, %d], eps=%.2e\n", tokens, d_model, eps);
        fprintf(stderr, "  GAMMA: min=%.6f max=%.6f rms=%.6f\n", gamma_min, gamma_max, gamma_rms);
        
        for (int r = 0; r < sample_rows; r++) {
            // Compute input row stats
            float x_min = h_input[r * d_model], x_max = h_input[r * d_model];
            double x_sum_sq = 0.0;
            for (int c = 0; c < d_model; c++) {
                float val = h_input[r * d_model + c];
                x_min = std::min(x_min, val);
                x_max = std::max(x_max, val);
                x_sum_sq += val * val;
            }
            float x_rms = std::sqrt(x_sum_sq / d_model);
            float inv_rms = 1.0f / std::sqrt(x_sum_sq / d_model + eps);
            
            // Compute output row stats
            float y_min = h_output[r * d_model], y_max = h_output[r * d_model];
            double y_sum_sq = 0.0;
            for (int c = 0; c < d_model; c++) {
                float val = h_output[r * d_model + c];
                y_min = std::min(y_min, val);
                y_max = std::max(y_max, val);
                y_sum_sq += val * val;
            }
            float y_rms = std::sqrt(y_sum_sq / d_model);
            
            // Expected: y_rms = gamma_rms (when gamma=1.0, y_rms should be 1.0)
            float expected_y_rms = gamma_rms;
            
            fprintf(stderr, "  ROW %d: x_rms=%.10f inv_rms=%.10f | EXPECTED y_rms=%.10f | ACTUAL y_rms=%.10f",
                    r, x_rms, inv_rms, expected_y_rms, y_rms);
            if (std::abs(y_rms - expected_y_rms) > 0.01f) {
                fprintf(stderr, " [ANOMALY] diff=%.10f (%.4fx off)\n", 
                        y_rms - expected_y_rms, y_rms / expected_y_rms);
            } else {
                fprintf(stderr, " ✓\n");
            }
        }
    }
    #endif
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new RMSNormGradFn();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), const_cast<Tensor&>(gamma));
        
        // ISSUE #51 FIX: Copy cache to owned buffer
        const float* effective_cache = input_cache ? input_cache : x.data;
        grad_fn->set_cache_copy(effective_cache, static_cast<size_t>(tokens) * d_model, d_model, eps, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
    }
    
    return result;
}

// NOTE: cross_entropy() removed - use autograd::cross_entropy_loss() from AutogradLoss.cu
// Production training uses ComputeLossBatch.cu -> autograd::cross_entropy_loss() which
// properly computes loss AND backward with correct 1/N scaling (Issue #58 fix).

Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens, cudaStream_t stream, float embedding_scale) {
    if (!weight.shape.is_flat()) {
        throw std::invalid_argument("autograd::embedding: weight must be 2D [vocab_size, d_model]");
    }
    if (!token_ids) {
        throw std::invalid_argument("autograd::embedding: token_ids is NULL");
    }
    if (num_tokens <= 0) {
        throw std::invalid_argument("autograd::embedding: num_tokens must be > 0");
    }
    if (!weight.data) {
        throw std::invalid_argument("autograd::embedding: weight.data is NULL");
    }
    if (embedding_scale <= 0.0f) {
        throw std::invalid_argument("autograd::embedding: embedding_scale must be > 0");
    }
    
    const int d_model = weight.shape.as_2d().cols;
    auto output_shape = TensorContract::TensorShape::make_BSM(num_tokens, d_model);
    Tensor result = Tensor::empty(output_shape, weight.requires_grad, stream);
    
    // Forward: gather from weight table with scaling
    // Issue #92: Scale by sqrt(d_model) to match AIAYN-style embeddings
    kernel_embedding_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        token_ids, weight.data, result.data, num_tokens, d_model, embedding_scale);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (weight.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new EmbeddingGradFn();
        grad_fn->capture_weight(const_cast<Tensor&>(weight));
        grad_fn->save(token_ids, num_tokens, d_model, true, stream);
        grad_fn->embedding_scale = embedding_scale;  // Store for backward scaling
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
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
    // ISSUE #48: capture stable data, not Tensor*
    if (x.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new SoftmaxGradFn();
        grad_fn->capture_input(const_cast<Tensor&>(x));
        grad_fn->save(result.data, num_tokens, dim, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
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
            // Identity backward - ISSUE #48: capture stable data
            auto* grad_fn = new AddGradFn();
            grad_fn->capture_single_input(const_cast<Tensor&>(x), stream);  // ISSUE #56 FIX
            result.grad_fn = grad_fn;
            result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
        }
        return result;
    }
    
    const size_t count = x.numel();
    
    // Forward: y = x * mask / (1 - p) where mask is 0/1
    // Would call dropout forward kernel with mask
    // Placeholder: copy input
    cudaMemcpyAsync(result.data, x.data, x.size_bytes(), cudaMemcpyDeviceToDevice, stream);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (x.requires_grad && mask) {
        result.is_leaf = false;
        auto* grad_fn = new DropoutGradFn();
        grad_fn->capture_input(const_cast<Tensor&>(x));
        grad_fn->save(mask, p, count, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
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
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new ResidualAddGradFn();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), const_cast<Tensor&>(residual), stream);  // ISSUE #56 FIX
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
    }
    
    return result;
}

/**
 * autograd::scale - Scale tensor by constant scalar with gradient tracking
 * Forward: y = x * scale_factor
 * Backward: grad_x = grad_y * scale_factor (chain rule)
 *
 * ISSUE #98: Added to fix gradient vanishing when tie_embeddings=true.
 * When embeddings are scaled by sqrt(d_model) (Issue #92), the LM head
 * must also scale by sqrt(d_model) to maintain gradient flow symmetry.
 */
Tensor scale(const Tensor& x, float scale_factor, cudaStream_t stream) {
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream);
    
    // Forward: y = x * scale_factor
    TensorContract::TensorView x_view(const_cast<float*>(x.data), x.shape);
    TensorContract::TensorView out_view(result.data, result.shape);
    TensorContract::scale(x_view, scale_factor, out_view, stream);
    
    // Set up backward
    if (x.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new ScaleGradFn();
        grad_fn->capture_input(const_cast<Tensor&>(x), scale_factor, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;
    }
    
    return result;
}

/**
 * layer_scale - Element-wise multiply by learnable scalar (Issue #109)
 * 
 * Forward:  y[i,j] = x[i,j] * scale_param[0]  (scale_param is shape [1])
 * Backward: grad_x = grad_y * scale_param
 *           grad_scale_param = sum(grad_y * x)  (reduction)
 *
 * This differs from scale() which uses a constant float. LayerScale uses a
 * LEARNABLE parameter that receives its own gradients during training.
 */
Tensor layer_scale(const Tensor& x, Tensor& scale_param, cudaStream_t stream) {
    if (!scale_param.data) {
        throw std::runtime_error("layer_scale: scale_param is NULL");
    }
    
    // Read scale value from GPU
    float scale_value = 1.0f;
    cudaMemcpyAsync(&scale_value, scale_param.data, sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    const bool track_grad = x.requires_grad || scale_param.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream);
    
    // Forward: y = x * scale_value  (broadcast)
    TensorContract::TensorView x_view(const_cast<float*>(x.data), x.shape);
    TensorContract::TensorView out_view(result.data, result.shape);
    TensorContract::scale(x_view, scale_value, out_view, stream);
    
    // Set up backward
    if (track_grad) {
        result.is_leaf = false;
        auto* grad_fn = new LayerScaleGradFn();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), scale_param, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;
    }
    
    return result;
}

/**
 * center_rows - Row-wise mean centering (Issue #118)
 * 
 * Forward:  y[t,d] = x[t,d] - mean_d(x[t,:])
 * Backward: grad_x[t,d] = grad_y[t,d] - mean_d(grad_y[t,:])  (SAME centering!)
 * 
 * Purpose: Removes common direction from activations to prevent mode collapse.
 * The common direction accumulates through residual stream (12 layers × LayerScale).
 * By centering before residual add, we zero the accumulated bias.
 * 
 * @param x Input tensor [num_rows, row_dim] (typically [total_tokens, d_model])
 * @param stream CUDA stream
 * @return Centered tensor (row sums ≈ 0)
 */
Tensor center_rows(const Tensor& x, cudaStream_t stream) {
    if (!x.data) {
        throw std::runtime_error("center_rows: input tensor data is NULL");
    }
    
    // For 2D tensors [total_tokens, d_model]: rows=total_tokens, cols=d_model
    // We center each ROW (subtract mean of that row's d_model elements)
    if (!x.shape.is_flat()) {
        throw std::runtime_error("center_rows: expected 2D (flat) tensor, got 4D");
    }
    const int num_rows = x.shape.as_2d().rows;  // total_tokens
    const int row_dim = x.shape.as_2d().cols;   // d_model (768)
    
    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream);
    
    // Forward: y = x - mean(x)  (per-row)
    kernel_center_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, row_dim, num_rows);
    
    // Set up backward
    if (track_grad) {
        result.is_leaf = false;
        auto* grad_fn = new CenterRowsGradFn();
        grad_fn->capture_input(const_cast<Tensor&>(x), row_dim, num_rows, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;
    }
    
    return result;
}

/**
 * center_columns - Remove mean across rows (positions) for each column (feature)
 * ISSUE #118 PROPER FIX: This is the CORRECT centering dimension!
 * 
 * Forward:  y[t,d] = x[t,d] - mean_t(x[:,d])  (subtract column mean from each element)
 * Backward: grad_x[t,d] = grad_y[t,d] - mean_t(grad_y[:,d])  (SAME centering - linear!)
 * 
 * WHY THIS FIXES THE MODE COLLAPSE:
 * ─────────────────────────────────
 * - center_rows (old): Made each row's features sum to 0, but didn't change cos(h_i, h_j)
 *   because rows still pointed in same relative directions
 * - center_columns (new): Removes the "common direction" that all positions share
 *   by subtracting the mean vector across all positions
 * 
 * Mathematical Effect:
 *   Before: h[t,:] = signal[t,:] + common[:]  (all positions have same common component)
 *   After:  h[t,:] = signal[t,:]              (common component removed!)
 *   
 *   This DIRECTLY reduces avg_cos(h_i, h_j) because the shared direction is gone.
 * 
 * @param x Input tensor [num_rows, num_cols] (typically [total_tokens, d_model])
 * @param stream CUDA stream
 * @return Column-centered tensor (column means ≈ 0 for each feature dimension)
 */
Tensor center_columns(const Tensor& x, cudaStream_t stream) {
    if (!x.data) {
        throw std::runtime_error("center_columns: input tensor data is NULL");
    }
    
    // For 2D tensors [total_tokens, d_model]: rows=total_tokens, cols=d_model
    // We center each COLUMN (subtract mean across all rows for each column)
    if (!x.shape.is_flat()) {
        throw std::runtime_error("center_columns: expected 2D (flat) tensor, got 4D");
    }
    const int num_rows = x.shape.as_2d().rows;  // total_tokens (positions)
    const int num_cols = x.shape.as_2d().cols;  // d_model (768 features)
    
    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream);
    
    // Forward: y[t,d] = x[t,d] - mean_t(x[:,d])  (per-column mean subtraction)
    // Launch one block per column (768 blocks for d_model=768)
    kernel_center_columns<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_cols, num_rows);
    
    // Set up backward
    if (track_grad) {
        result.is_leaf = false;
        auto* grad_fn = new CenterColumnsGradFn();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, stream);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ISSUE #77 DIAGNOSTIC: Log cached activation (ln1_out) values during backward
// ═══════════════════════════════════════════════════════════════════════════════

// Kernel to compute min/max/sum/sum_sq for diagnostics
__global__ void diagCachedActivationKernel(
    const float* __restrict__ data,
    int count,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
    float* __restrict__ out_sum_sq,
    int* __restrict__ out_nan_count,
    int* __restrict__ out_inf_count
) {
    __shared__ float s_min, s_max, s_sum, s_sum_sq;
    __shared__ int s_nan, s_inf;
    
    if (threadIdx.x == 0) {
        s_min = FLT_MAX;
        s_max = -FLT_MAX;
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
        s_nan = 0;
        s_inf = 0;
    }
    __syncthreads();
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int local_nan = 0;
    int local_inf = 0;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float val = data[i];
        if (isnan(val)) {
            local_nan++;
        } else if (isinf(val)) {
            local_inf++;
        } else {
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Reduce within block using atomics (simple for small diagnostics)
    atomicMin(reinterpret_cast<int*>(&s_min), __float_as_int(local_min));
    atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(local_max));
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    atomicAdd(&s_nan, local_nan);
    atomicAdd(&s_inf, local_inf);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicMin(reinterpret_cast<int*>(out_min), __float_as_int(s_min));
        atomicMax(reinterpret_cast<int*>(out_max), __float_as_int(s_max));
        atomicAdd(out_sum, s_sum);
        atomicAdd(out_sum_sq, s_sum_sq);
        atomicAdd(out_nan_count, s_nan);
        atomicAdd(out_inf_count, s_inf);
    }
}

// Host function to log cached activation statistics
// ISSUE #77: Call this in MatMulGradFn::apply() to diagnose ln1_out values
static void logCachedActivationStats(
    const char* name,
    const float* data,
    int count,
    cudaStream_t stream
) {
    // Allocate pinned host memory for results (small, one-time alloc is OK for diagnostics)
    float h_min = FLT_MAX, h_max = -FLT_MAX, h_sum = 0.0f, h_sum_sq = 0.0f;
    int h_nan = 0, h_inf = 0;
    
    float* d_min, *d_max, *d_sum, *d_sum_sq;
    int* d_nan, *d_inf;
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMalloc(&d_nan, sizeof(int));
    cudaMalloc(&d_inf, sizeof(int));
    
    // Initialize to identity values
    float init_min = FLT_MAX, init_max = -FLT_MAX, init_zero = 0.0f;
    int init_zero_int = 0;
    cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sum_sq, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nan, &init_zero_int, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_inf, &init_zero_int, sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    int threads = 256;
    int blocks = std::min((count + threads - 1) / threads, 256);
    diagCachedActivationKernel<<<blocks, threads, 0, stream>>>(
        data, count, d_min, d_max, d_sum, d_sum_sq, d_nan, d_inf);
    
    // Sync and copy results
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nan, d_nan, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_inf, d_inf, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Compute statistics
    int valid_count = count - h_nan - h_inf;
    float mean = (valid_count > 0) ? h_sum / valid_count : 0.0f;
    float variance = (valid_count > 1) ? (h_sum_sq / valid_count - mean * mean) : 0.0f;
    float stddev = sqrtf(fmaxf(variance, 0.0f));
    
    // Log results
    fprintf(stderr, "[Issue77-CachedActivation] %s: count=%d min=%.10f max=%.10f mean=%.10f std=%.10f nan=%d inf=%d\n",
            name, count, h_min, h_max, mean, stddev, h_nan, h_inf);
    
    // Cleanup
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
    cudaFree(d_nan);
    cudaFree(d_inf);
}

// Global toggle for Issue #77 diagnostics (enable during investigation)
static bool g_issue77_diag_enabled = true;
static int g_issue77_diag_call_count = 0;

// LM head gradient correction flags (centering outside autograd)
static bool g_lm_head_recenter_gradients = false;

void set_lm_head_grad_correction(bool recenter_gradients) {
    g_lm_head_recenter_gradients = recenter_gradients;
}

__global__ void centerGradientsKernel(
    float* __restrict__ data,   // [total_tokens, d_model] in-place
    int d_model,
    int total_tokens
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    float* row = data + static_cast<size_t>(token_idx) * d_model;

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum += row[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    const float mean = s_sum / static_cast<float>(d_model);
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        row[i] -= mean;
    }
}

static inline void applyLmHeadGradCorrections(
    float* grad_a,
    int total_tokens,
    int d_model,
    cudaStream_t stream
) {
    if (!grad_a || total_tokens <= 0 || d_model <= 0) {
        return;
    }
    if (!g_lm_head_recenter_gradients) {
        return;
    }

    constexpr int kBlockSize = 256;

    centerGradientsKernel<<<total_tokens, kBlockSize, 0, stream>>>(
        grad_a, d_model, total_tokens);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[applyLmHeadGradCorrections] recenter kernel failed: ") +
                                 cudaGetErrorString(err));
    }
}

/**
 * MatMulGradFn - Backward for matrix multiplication (TAPE-BASED)
 * Forward: C = A @ B  [M,K] @ [K,N] = [M,N]
 * Backward:
 *   grad_A = grad_C @ B^T  [M,N] @ [N,K] = [M,K]
 *   grad_B = A^T @ grad_C  [K,M] @ [M,N] = [K,N]
 *
 * TAPE-BASED: Does NOT allocate. References external caches and writes directly to grad buffers.
 * 
 * ISSUE #48 FIX: Stores stable data (shapes, grad pointers, grad_fn) instead of Tensor* 
 * which may become dangling after the forward function returns.
 * 
 * ISSUE #55 FIX: For non-leaf (activation) tensors, owns grad buffers.
 * For leaf (weight) tensors, uses their persistent grad buffers directly.
 */
struct MatMulGradFn : public GradFn {
    // ISSUE #48: Don't store Tensor* - they become dangling after forward returns
    // Instead, store what we actually need for backward:
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    
    // ISSUE #55 FIX: Grad buffer management
    // - For leaf tensors (weights): grad_a/b point to persistent TrainingState buffers
    // - For non-leaf tensors (activations): grad_a/b point to owned_grad_a/b.get()
    std::shared_ptr<float> owned_grad_a;   // Owned GPU memory for grad_A (non-leaf only)
    std::shared_ptr<float> owned_grad_b;   // Owned GPU memory for grad_B (non-leaf only)
    float* grad_a = nullptr;       // Gradient buffer for A (owned or persistent)
    float* grad_b = nullptr;       // Gradient buffer for B (owned or persistent)
    
    TensorContract::TensorShape a_shape;  // Shape of A for chain continuation
    TensorContract::TensorShape b_shape;  // Shape of B for chain continuation
    GradFn* a_grad_fn = nullptr;   // Chain continuation for A
    GradFn* b_grad_fn = nullptr;   // Chain continuation for B
    bool owns_a_grad_fn = false;   // ISSUE #50: Take ownership to prevent use-after-free
    bool owns_b_grad_fn = false;
    
    // ISSUE #51 FIX: Own copies of cached data to prevent dangling pointers
    // Previously: stored non-owning pointers that became stale when input tensors freed
    // Now: copy to owned buffers that persist for backward pass
    std::shared_ptr<float> owned_cache_a;  // Owned GPU memory copy of A (for grad_B)
    std::shared_ptr<float> owned_cache_b;  // Owned GPU memory copy of B (for grad_A)
    const float* cached_a = nullptr;  // Points to owned_cache_a.get() after set_cache_copy()
    const float* cached_b = nullptr;  // Points to owned_cache_b.get() after set_cache_copy()
    int M = 0, K = 0, N = 0;   // Dimensions
    cublasHandle_t cublas_handle = nullptr;
    bool transpose_b = false;  // Was B transposed in forward?
    cudaStream_t cache_stream = nullptr;  // Stream for cache copy operations
    
    MatMulGradFn() { op_name = "matmul"; }
    
    ~MatMulGradFn() {
        // ISSUE #50: Delete owned grad_fns to complete the chain cleanup
        if (owns_a_grad_fn && a_grad_fn) {
            delete a_grad_fn;
            a_grad_fn = nullptr;
        }
        if (owns_b_grad_fn && b_grad_fn) {
            delete b_grad_fn;
            b_grad_fn = nullptr;
        }
        // shared_ptrs will be destroyed automatically after this
    }
    
    // ISSUE #48 FIX: Store stable info from tensors during forward, before they go out of scope
    // ISSUE #55 FIX: For non-leaf (intermediate) tensors, allocate OWNED grad buffers
    //                because the tensor's grad buffer gets freed when tensor goes out of scope.
    //                For leaf (weight) tensors, use their grad buffer directly since they persist.
    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_shape = a.shape;
        b_shape = b.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fns to prevent use-after-free
        a_grad_fn = a.grad_fn;
        if (a.grad_fn && a.owns_grad_fn) {
            owns_a_grad_fn = true;
            a.owns_grad_fn = false;  // Transfer ownership to us
        }
        b_grad_fn = b.grad_fn;
        if (b.grad_fn && b.owns_grad_fn) {
            owns_b_grad_fn = true;
            b.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        // ISSUE #55 FIX: Handle grad buffer ownership based on tensor type
        // - Leaf tensors (weights): persist in TrainingState, use their grad buffer directly
        // - Non-leaf tensors (activations): temporary, need owned buffer
        if (a_requires_grad) {
            a.ensure_grad();  // Ensure tensor has grad buffer
            
            if (a.is_leaf) {
                // Leaf tensor (weight): persists, safe to use directly
                grad_a = a.grad_data();  // ISSUE #59: Use accessor
                AG_TRACE("[MatMulGradFn] Using persistent grad_a buffer (leaf): %p\n", (void*)grad_a);
            } else {
                // Non-leaf tensor (activation): temporary, allocate owned buffer
                const size_t a_numel = a.numel();
                float* buffer_a = nullptr;
                cudaMalloc(&buffer_a, a_numel * sizeof(float));
                cudaMemsetAsync(buffer_a, 0, a_numel * sizeof(float), stream);
                
                owned_grad_a = std::shared_ptr<float>(buffer_a, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_a = owned_grad_a.get();
                
                // For non-leaf: gradient flows through chain, no need to sync back to tensor
                // The tensor will be destroyed anyway, grad goes to prev layer via grad_fn
                AG_TRACE("[MatMulGradFn] Allocated owned grad_a buffer (non-leaf): %zu floats at %p\n", 
                        a_numel, (void*)grad_a);
            }
        }
        if (b_requires_grad) {
            b.ensure_grad();  // Ensure tensor has grad buffer
            
            if (b.is_leaf) {
                // Leaf tensor (weight): persists, safe to use directly
                grad_b = b.grad_data();  // ISSUE #59: Use accessor
                AG_TRACE("[MatMulGradFn] Using persistent grad_b buffer (leaf): %p\n", (void*)grad_b);
                
                // ISSUE #87 DEBUG: Always print for weight tensors to verify buffer matching
                // This is critical for diagnosing the plateau bug
                if (b_shape.as_2d().rows > 10000) {  // Likely vocab_size x d_model = LM head weights
                    fprintf(stderr, "[Issue87-DEBUG] LM head weight capture: is_leaf=%d grad_b=%p shape=[%d,%d]\n",
                            (int)b.is_leaf, (void*)grad_b, b_shape.as_2d().rows, b_shape.as_2d().cols);
                    fflush(stderr);
                }
            } else {
                // Non-leaf tensor (activation): temporary, allocate owned buffer
                const size_t b_numel = b.numel();
                float* buffer_b = nullptr;
                cudaMalloc(&buffer_b, b_numel * sizeof(float));
                cudaMemsetAsync(buffer_b, 0, b_numel * sizeof(float), stream);
                
                owned_grad_b = std::shared_ptr<float>(buffer_b, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_b = owned_grad_b.get();
                
                // For non-leaf: gradient flows through chain, no need to sync back to tensor
                AG_TRACE("[MatMulGradFn] Allocated owned grad_b buffer (non-leaf): %zu floats at %p\n",
                        b_numel, (void*)grad_b);
            }
        }
    }
    
    // ISSUE #51 FIX: Copy cache data to owned buffers (not just store pointers)
    // This prevents the SGEMM "illegal value" errors caused by dangling pointers
    // when input tensors are freed before backward() runs.
    void set_cache_copy(const float* a_cache, const float* b_cache, int m, int k, int n, 
                        cublasHandle_t handle, cudaStream_t stream, bool transB = false) {
        transpose_b = transB;
        M = m; K = k; N = n;
        cublas_handle = handle;
        cache_stream = stream;
        
        // Copy cache A if needed for grad_B computation
        if (b_requires_grad && a_cache) {
            // A is [M, K] row-major
            const size_t a_size = static_cast<size_t>(m) * k;
            float* buffer_a = nullptr;
            cudaMalloc(&buffer_a, a_size * sizeof(float));
            // Issue #57: Use SYNCHRONOUS copy to ensure source data isn't freed before copy completes
            // The source tensor destructor runs on CPU and can free memory while async copy is in flight
            cudaMemcpy(buffer_a, a_cache, a_size * sizeof(float), cudaMemcpyDeviceToDevice);
            
            // Wrap in shared_ptr with DEFERRED cleanup deleter (Issue #53)
            // Instead of calling cudaFree directly (which blocks), queue for later cleanup
            owned_cache_a = std::shared_ptr<float>(buffer_a, [](float* p) {
                queueForDeferredCleanup(p);
            });
            cached_a = owned_cache_a.get();
            AG_TRACE("[MatMulGradFn] Copied cache_A: %zu floats to %p\n", a_size, (void*)cached_a);
        }
        
        // Copy cache B if needed for grad_A computation
        if (a_requires_grad && b_cache) {
            // B is [K, N] or [N, K] depending on transpose_b
            const size_t b_size = transB ? static_cast<size_t>(n) * k : static_cast<size_t>(k) * n;
            float* buffer_b = nullptr;
            cudaMalloc(&buffer_b, b_size * sizeof(float));
            // Issue #57: Use SYNCHRONOUS copy to ensure source data isn't freed before copy completes
            cudaMemcpy(buffer_b, b_cache, b_size * sizeof(float), cudaMemcpyDeviceToDevice);
            
            // Wrap in shared_ptr with DEFERRED cleanup deleter (Issue #53)
            // Instead of calling cudaFree directly (which blocks), queue for later cleanup
            owned_cache_b = std::shared_ptr<float>(buffer_b, [](float* p) {
                queueForDeferredCleanup(p);
            });
            cached_b = owned_cache_b.get();
            AG_TRACE("[MatMulGradFn] Copied cache_B: %zu floats to %p\n", b_size, (void*)cached_b);
        }
        
        // Validate that required caches are set
        if (a_requires_grad && !cached_b) {
            throw std::runtime_error("MatMulGradFn::set_cache_copy: b_cache is NULL but input_a requires grad");
        }
        if (b_requires_grad && !cached_a) {
            throw std::runtime_error("MatMulGradFn::set_cache_copy: a_cache is NULL but input_b requires grad");
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("matmul", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        // DIAGNOSTIC: Log incoming gradient to matmul backward
        static int s_matmul_bwd_call = 0;
        const int mm_call_idx = ++s_matmul_bwd_call;
        {
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            // FIX: For LM_HEAD (vocab_size=50377), first 10K elements are <1 token
            // Sample from middle of buffer (skip first 50 tokens for BOS/masked)
            const size_t start_offset = (N > 10000) ? static_cast<size_t>(50) * N : 0;
            const size_t sample_sz = std::min(grad_elems - start_offset, static_cast<size_t>(10000));
            std::vector<float> samp(sample_sz);
            cudaMemcpy(samp.data(), grad_output.data + start_offset, sample_sz * sizeof(float), cudaMemcpyDeviceToHost);
            float mx = 0.0f; double sq = 0.0;
            for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
            float rms = std::sqrt(static_cast<float>(sq / sample_sz));
            // Identify matmul by dimensions:
            // LM_HEAD: N=vocab_size (~50377)
            // QKV:     N=1280 (768 + 2*256 for GQA)
            // W_o:     K=768, N=768
            // FFN W1:  K=768, N=3072 (d_model -> d_ff)
            // FFN W2:  K=3072, N=768 (d_ff -> d_model)
            const char* label = "encoder";
            if (N > 10000) label = "LM_HEAD";
            else if (N == 1280 && K == 768) label = "QKV_proj";
            else if (K == 768 && N == 768) label = "W_o_proj";
            else if (K == 768 && N == 3072) label = "FFN_W1";
            else if (K == 3072 && N == 768) label = "FFN_W2";
            fprintf(stderr, "[MATMUL-BWD-IN] call=%d %s M=%d K=%d N=%d | grad_C: numel=%zu max=%.10f rms=%.10f PTR=%p\n",
                    mm_call_idx, label, M, K, N, grad_elems, mx, rms, (void*)grad_output.data);
        }
        
        if (!cublas_handle) {
            throw std::runtime_error("MatMulGradFn::apply: cublas_handle is NULL");
        }
        
        const float alpha = 1.0f;
        const float beta_accum = 1.0f;  // Accumulate to existing gradient
        
        cublasSetStream(cublas_handle, stream);;
        
        // Without transpose_b: Forward was C = A @ B, where A[M,K], B[K,N], C[M,N]
        //   grad_A = grad_C @ B^T   [M,N] @ [N,K] = [M,K]
        //   grad_B = A^T @ grad_C   [K,M] @ [M,N] = [K,N]
        //
        // With transpose_b: Forward was C = A @ B^T, where A[M,K], B[N,K], C[M,N]
        //   grad_A = grad_C @ B     [M,N] @ [N,K] = [M,K]  (B, not B^T)
        //   grad_B = grad_C^T @ A   [N,M] @ [M,K] = [N,K]  (gradient w.r.t. B before transpose)
        
        // ISSUE #48 FIX: Use stored grad pointers instead of Tensor* (which may be dangling)
        if (a_requires_grad) {
            AG_TRACE("[MatMulGradFn] Computing grad_A, grad_a=%p, cached_b=%p\n",
                   static_cast<void*>(grad_a), static_cast<const void*>(cached_b));
            if (!cached_b) {
                throw std::runtime_error("MatMulGradFn::apply: cached_b is NULL but input_a requires grad");
            }
            if (!grad_a) {
                throw std::runtime_error("MatMulGradFn::apply: grad_a is NULL - capture_inputs() must be called");
            }
            
            // DIAGNOSTIC: Log equation values for LM_HEAD grad_A computation
            if (N > 10000 && transpose_b) {  // LM_HEAD case
                cudaStreamSynchronize(stream);
                // Sample cached_b (weights) statistics
                const size_t b_elems = static_cast<size_t>(N) * K;
                const size_t b_sample_sz = std::min(b_elems, static_cast<size_t>(100000));
                std::vector<float> b_samp(b_sample_sz);
                cudaMemcpy(b_samp.data(), cached_b, b_sample_sz * sizeof(float), cudaMemcpyDeviceToHost);
                float b_max = 0.0f, b_min = 1e30f;
                double b_sq = 0.0, b_sum = 0.0;
                for (auto& v : b_samp) {
                    if (!std::isnan(v) && !std::isinf(v)) {
                        b_max = std::max(b_max, std::abs(v));
                        b_min = std::min(b_min, std::abs(v));
                        b_sq += v*v;
                        b_sum += v;
                    }
                }
                float b_std = std::sqrt(static_cast<float>(b_sq / b_sample_sz - (b_sum/b_sample_sz)*(b_sum/b_sample_sz)));
                float b_rms = std::sqrt(static_cast<float>(b_sq / b_sample_sz));
                
                // Sample grad_C statistics (already have from earlier diagnostic, but log again for clarity)
                const size_t c_sample_sz = std::min(static_cast<size_t>(M) * N, static_cast<size_t>(100000));
                std::vector<float> c_samp(c_sample_sz);
                cudaMemcpy(c_samp.data(), grad_output.data, c_sample_sz * sizeof(float), cudaMemcpyDeviceToHost);
                float c_max = 0.0f;
                double c_sq = 0.0;
                for (auto& v : c_samp) {
                    if (!std::isnan(v) && !std::isinf(v)) {
                        c_max = std::max(c_max, std::abs(v));
                        c_sq += v*v;
                    }
                }
                float c_rms = std::sqrt(static_cast<float>(c_sq / c_sample_sz));
                
                // Expected grad_A magnitude: sqrt(N) * grad_C_rms * B_rms
                float expected_grad_a = std::sqrt(static_cast<float>(N)) * c_rms * b_rms;
                
                fprintf(stderr, "[GRAD_A_EQUATION] LM_HEAD: grad_A = grad_C @ B\n");
                fprintf(stderr, "  grad_C: shape=[%d,%d] max=%.6f rms=%.6f\n", M, N, c_max, c_rms);
                fprintf(stderr, "  B(weights): shape=[%d,%d] max=%.6f std=%.6f rms=%.6f\n", N, K, b_max, b_std, b_rms);
                fprintf(stderr, "  EXPECTED grad_A ≈ sqrt(%d) * %.6f * %.6f = %.6f\n", N, c_rms, b_rms, expected_grad_a);
            }
            
            if (transpose_b) {
                // grad_A = grad_C @ B  where B is [N, K] (original weights, NOT transposed now)
                // Row-major: grad_A[M,K] = grad_C[M,N] @ B[N,K]
                //
                // Using row-major GEMM trick: C = A @ B row-major ≡ C^T = B^T @ A^T col-major
                // So: grad_A^T[K,M] = B^T[K,N] @ grad_C^T[N,M] col-major
                //
                // Our storage (row-major → col-major conversion):
                //   cached_b is B[N,K] row-major = [K,N] col-major (K=768 rows, N=50377 cols)
                //   grad_C is [M,N] row-major = [N,M] col-major (N=50377 rows, M=3598 cols)
                //   grad_A is [M,K] row-major = [K,M] col-major (K=768 rows, M=3598 cols)
                //
                // cuBLAS: C = op(A) @ op(B)
                //   We want: grad_A^T[K,M] = [K,N] @ [N,M]
                //   A = cached_b[K,N] col-major, op=N gives [K,N] logical ✓
                //   B = grad_C[N,M] col-major, op=N gives [N,M] logical ✓
                //   Result: [K,M] col-major = grad_A^T ✓
                cublasStatus_t sgemm_status_1 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,    // cached_b is [K,N] col-major, use as [K,N]
                    CUBLAS_OP_N,    // grad_C is [N,M] col-major, use as [N,M]
                    K, M, N,        // m=K=768, n=M=3598, k=N=50377
                    &alpha,
                    cached_b, K,          // lda=K=768 (leading dim of [K,N])
                    grad_output.data, N,  // ldb=N=50377 (leading dim of [N,M])
                    &beta_accum,
                    grad_a, K             // ldc=K=768 (leading dim of [K,M])
                );
                trackCublasCall("cublasSgemm_grad_A_transB", cublas_handle, stream, sgemm_status_1);
                
                // DIAGNOSTIC: Log ACTUAL grad_A after GEMM for LM_HEAD
                if (N > 10000) {
                    cudaStreamSynchronize(stream);
                    const size_t a_elems = static_cast<size_t>(M) * K;
                    const size_t a_sample_sz = std::min(a_elems, static_cast<size_t>(100000));
                    std::vector<float> a_samp(a_sample_sz);
                    cudaMemcpy(a_samp.data(), grad_a, a_sample_sz * sizeof(float), cudaMemcpyDeviceToHost);
                    float a_max = 0.0f;
                    double a_sq = 0.0;
                    for (auto& v : a_samp) {
                        if (!std::isnan(v) && !std::isinf(v)) {
                            a_max = std::max(a_max, std::abs(v));
                            a_sq += v*v;
                        }
                    }
                    float a_rms = std::sqrt(static_cast<float>(a_sq / a_sample_sz));
                    fprintf(stderr, "  ACTUAL grad_A: shape=[%d,%d] max=%.6f rms=%.6f\n", M, K, a_max, a_rms);
                }
            } else {
                // grad_A = grad_C @ B^T  where B is [K, N]
                // Goal: grad_A[M,K] = grad_C[M,N] @ B^T[N,K]  (row-major)
                //
                // Row-major GEMM trick: R = X @ Y row-major ≡ R^T = Y^T @ X^T col-major
                // So: grad_A^T[K,M] = B[K,N] @ grad_C^T[N,M] col-major
                //
                // Storage conversions (row-major → col-major = transpose):
                //   cached_b is B[K,N] row-major = [N,K] col-major  
                //   grad_C is [M,N] row-major = [N,M] col-major
                //   grad_A is [M,K] row-major = [K,M] col-major (output)
                //
                // We want: [K,M] = [K,N] @ [N,M] col-major
                //   cached_b is [N,K] col-major, need OP_T to get [K,N]
                //   grad_C is [N,M] col-major, use OP_N to get [N,M]
                //
                // cuBLAS: C[K,M] = OP_T(cached_b[N,K]) @ OP_N(grad_C[N,M])
                cublasStatus_t sgemm_status_2 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_T,    // cached_b is [N,K] col-major, transpose to [K,N]
                    CUBLAS_OP_N,    // grad_C is [N,M] col-major, use as [N,M]
                    K, M, N,        // m=K, n=M, k=N: result [K,M] col-major = grad_A[M,K] row-major
                    &alpha,
                    cached_b, N,          // lda=N (leading dim of [N,K] col-major, transposed)
                    grad_output.data, N,  // ldb=N (leading dim of [N,M] col-major)
                    &beta_accum,
                    grad_a, K             // ldc=K (leading dim of [K,M] col-major)
                );
                trackCublasCall("cublasSgemm_grad_A", cublas_handle, stream, sgemm_status_2);
            }
        }
        
        // ISSUE #48 FIX: Use stored grad pointers instead of Tensor*
        if (b_requires_grad) {
            AG_TRACE("[MatMulGradFn] Computing grad_B, grad_b=%p, cached_a=%p\n",
                   static_cast<void*>(grad_b), static_cast<const void*>(cached_a));
            if (!cached_a) {
                throw std::runtime_error("MatMulGradFn::apply: cached_a is NULL but input_b requires grad");
            }
            if (!grad_b) {
                throw std::runtime_error("MatMulGradFn::apply: grad_b is NULL - capture_inputs() must be called");
            }
            
            // ========================================================================
            // ISSUE #77 DIAGNOSTIC: Log cached_a (ln1_out) values BEFORE the GEMM
            // This helps diagnose why W_qkv gradients explode despite tiny FA outputs
            // ========================================================================
            if (g_issue77_diag_enabled && transpose_b) {
                g_issue77_diag_call_count++;
                // Only log first 24 calls (2 per layer * 12 layers) to avoid spam
                if (g_issue77_diag_call_count <= 24) {
                    char diag_name[128];
                    snprintf(diag_name, sizeof(diag_name), "cached_a(ln1_out)_call%d_M%d_K%d", 
                             g_issue77_diag_call_count, M, K);
                    logCachedActivationStats(diag_name, cached_a, M * K, stream);
                    
                    // Also log grad_output (grad_qkv) values
                    snprintf(diag_name, sizeof(diag_name), "grad_output(grad_qkv)_call%d_M%d_N%d", 
                             g_issue77_diag_call_count, M, N);
                    logCachedActivationStats(diag_name, grad_output.data, M * N, stream);
                    
                    // Log grad_b BEFORE the GEMM to see initial state
                    snprintf(diag_name, sizeof(diag_name), "grad_b_BEFORE_call%d_K%d_N%d", 
                             g_issue77_diag_call_count, K, N);
                    logCachedActivationStats(diag_name, grad_b, K * N, stream);
                }
            }
            
            if (transpose_b) {
                // grad_B = grad_C^T @ A  where A is [M, K], result is [N, K]
                // This is the gradient w.r.t. B BEFORE the transpose in forward.
                //
                // Row-major: grad_B[N,K] = grad_C^T[N,M] @ A[M,K]
                // Using row-major trick: R = X @ Y row-major ≡ R^T = Y^T @ X^T col-major
                // So: grad_B^T[K,N] = A^T[K,M] @ grad_C[M,N] col-major
                //
                // Storage conversions:
                //   cached_a is A[M,K] row-major = A^T[K,M] col-major
                //   grad_C is [M,N] row-major = [N,M] col-major
                //   grad_B is [N,K] row-major = [K,N] col-major
                //
                // cuBLAS: C = op(A) @ op(B)
                //   We want: grad_B^T[K,N] = A^T[K,M] @ grad_C[M,N]
                //   A = cached_a[K,M] col-major, op=N gives [K,M] = A^T ✓
                //   B = grad_C[N,M] col-major, op=T gives [M,N] = grad_C ✓
                //   Result: [K,N] col-major = grad_B^T = grad_B[N,K] row-major ✓
                cublasStatus_t sgemm_status_3 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,    // cached_a is [K,M] col-major, use as [K,M] = A^T
                    CUBLAS_OP_T,    // grad_output is [N,M] col-major, transpose to [M,N]
                    K, N, M,        // m=K=768, n=N=50377, k=M=3598
                    &alpha,
                    cached_a, K,          // lda=K (leading dim of [K,M])
                    grad_output.data, N,  // ldb=N (leading dim of [N,M])
                    &beta_accum,
                    grad_b, K             // ldc=K (leading dim of [K,N])
                );
                trackCublasCall("cublasSgemm_grad_B_transB", cublas_handle, stream, sgemm_status_3);
                
                // ISSUE #77 DIAGNOSTIC: Log grad_b AFTER the GEMM
                if (g_issue77_diag_enabled && g_issue77_diag_call_count <= 24) {
                    char diag_name[128];
                    snprintf(diag_name, sizeof(diag_name), "grad_b_AFTER_call%d_K%d_N%d", 
                             g_issue77_diag_call_count, K, N);
                    logCachedActivationStats(diag_name, grad_b, K * N, stream);
                }
            } else {
                // grad_B = A^T @ grad_C  where A is [M, K], result is [K, N]
                // Goal: grad_B[K,N] = A^T[K,M] @ grad_C[M,N]  (row-major)
                //
                // Row-major GEMM trick: R = X @ Y row-major ≡ R^T = Y^T @ X^T col-major
                // So: grad_B^T[N,K] = grad_C^T[N,M] @ A[M,K] col-major
                //
                // Storage conversions (row-major → col-major = transpose):
                //   cached_a is A[M,K] row-major = [K,M] col-major
                //   grad_C is [M,N] row-major = [N,M] col-major
                //   grad_B is [K,N] row-major = [N,K] col-major (output)
                //
                // We want: [N,K] = [N,M] @ [M,K] col-major
                //   grad_C is [N,M] col-major, use OP_N to get [N,M]
                //   cached_a is [K,M] col-major, need OP_T to get [M,K]
                //
                // cuBLAS: C[N,K] = OP_N(grad_C[N,M]) @ OP_T(cached_a[K,M])
                cublasStatus_t sgemm_status_4 = cublasSgemm(cublas_handle,
                    CUBLAS_OP_N,    // grad_C is [N,M] col-major, use as [N,M]
                    CUBLAS_OP_T,    // cached_a is [K,M] col-major, transpose to [M,K]
                    N, K, M,        // m=N, n=K, k=M: result [N,K] col-major = grad_B[K,N] row-major
                    &alpha,
                    grad_output.data, N,  // lda=N (leading dim of [N,M] col-major)
                    cached_a, K,          // ldb=K (leading dim of [K,M] col-major, transposed)
                    &beta_accum,
                    grad_b, N             // ldc=N (leading dim of [N,K] col-major)
                );
                trackCublasCall("cublasSgemm_grad_B", cublas_handle, stream, sgemm_status_4);
            }
            
            // ISSUE #60 DEBUG: Capture LM head grad contribution if debugging enabled
            // Check if this looks like LM head (N=vocab_size is large, K=d_model is small)
            // Typical LM head: N=50377, K=768
            if (g_debug_capture_enabled && N > 10000 && K < 2000) {
                debugCaptureLMHeadGrad(grad_b, static_cast<size_t>(N) * K, stream);
                AG_TRACE("[MatMulGradFn] DEBUG: Captured LM head grad, N=%d K=%d\n", N, K);
            }
        }

        // If LM head centering was applied outside autograd,
        // correct grad_A before passing it to the encoder backward chain.
        if (a_requires_grad && transpose_b && N > 10000) {
            applyLmHeadGradCorrections(
                grad_a,
                M,
                K,
                stream);
        }

        // CONTINUE AUTOGRAD CHAIN (Recursive) - ISSUE #48 FIX: Use stored grad_fn pointers
        if (a_requires_grad && a_grad_fn) {
            if (a_grad_fn->op_name) {
                Tensor view;
                view.data = grad_a; view.shape = a_shape;
                view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
                
                // DIAGNOSTIC: Log grad_a that we're passing to next grad_fn
                {
                    cudaStreamSynchronize(stream);
                    const size_t grad_elems = view.numel();
                    std::vector<float> samp(std::min(grad_elems, static_cast<size_t>(10000)));
                    cudaMemcpy(samp.data(), grad_a, samp.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    float mx = 0.0f; double sq = 0.0;
                    for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
                    float rms = std::sqrt(static_cast<float>(sq / samp.size()));
                    fprintf(stderr, "[MATMUL-BWD-TO-A] op=%s | grad_a: numel=%zu max=%.10f rms=%.10f\n",
                            a_grad_fn->op_name ? a_grad_fn->op_name : "?", grad_elems, mx, rms);
                }
                
                a_grad_fn->apply(view, stream);
                // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
            }
        }
        if (b_requires_grad && b_grad_fn && b_grad_fn != a_grad_fn) {
            if (b_grad_fn->op_name) {
                Tensor view;
                view.data = grad_b; view.shape = b_shape;
                view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
                b_grad_fn->apply(view, stream);
                // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
            }
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        // TAPE-BASED: Don't free - we don't own the caches or grad buffers
        cached_a = nullptr;
        cached_b = nullptr;
        grad_a = nullptr;
        grad_b = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_a_grad_fn && a_grad_fn) { owns_a_grad_fn = false; delete a_grad_fn; a_grad_fn = nullptr; }
        if (owns_b_grad_fn && b_grad_fn) { owns_b_grad_fn = false; delete b_grad_fn; b_grad_fn = nullptr; }
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

/**
 * autograd::matmul - Matrix multiplication with automatic differentiation
 * 
 * TAPE-BASED: Requires caller to provide cache pointers from TrainingState.
 * Does NOT allocate internal copies of A or B.
 *
 * @param a Input tensor A [M, K]
 * @param b Input tensor B [K, N]
 * @param stream CUDA stream
 * @param a_cache External cache pointer for A (needed if B requires grad). 
 *                Pass nullptr if A is a weight tensor that persists.
 * @param b_cache External cache pointer for B (needed if A requires grad).
 *                Pass nullptr if B is a weight tensor that persists.
 * @return Output tensor C [M, N]
 */
Tensor matmul(const Tensor& a, const Tensor& b, cudaStream_t stream,
              const float* a_cache, const float* b_cache, bool transpose_b) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::matmul: stream is NULL - caller MUST provide valid stream");
    }
    
    // Validate inputs are 2D
    if (!a.shape.is_flat() || !b.shape.is_flat()) {
        throw std::invalid_argument("autograd::matmul: inputs must be 2D (BSM layout)");
    }
    
    const auto& a_shape = a.shape.as_2d();
    const auto& b_shape = b.shape.as_2d();
    
    // A is [M, K]
    // B is [K, N] if !transpose_b, or [N, K] if transpose_b (so B^T is [K, N])
    const int M = a_shape.rows;
    const int K = a_shape.cols;
    const int K2 = transpose_b ? b_shape.cols : b_shape.rows;
    const int N = transpose_b ? b_shape.rows : b_shape.cols;
    
    if (K != K2) {
        throw std::invalid_argument("autograd::matmul: inner dimensions must match (K=" + std::to_string(K) + " vs K2=" + std::to_string(K2) + ")");
    }
    
    // Get cuBLAS handle
    cublasHandle_t handle = get_autograd_cublas_handle();
    
    if (!handle) {
        throw std::runtime_error("autograd::matmul: cuBLAS handle not set. Call set_autograd_cublas_handle() first.");
    }
    
    // Output shape: [M, N]
    auto output_shape = TensorContract::TensorShape::make_BSM(M, N);
    Tensor result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad, stream);
    
    // Forward: C = A @ B  (or C = A @ B^T if transpose_b)
    // Row-major storage: for cuBLAS we compute C^T = B^T @ A^T (or C^T = B @ A^T if transpose_b)
    // 
    // Without transpose_b: B is [K, N], we want B^T for cuBLAS
    //   cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    //
    // With transpose_b: B is [N, K], we want B (no transpose) for cuBLAS since B^T is [K, N]
    //   cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, B, K, A, K, beta, C, N)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSetStream(handle, stream);
    
    cublasStatus_t status;
    if (transpose_b) {
        // B is [N, K] row-major, we want A @ B^T = C[M,N]
        // Row-major A[M,K] @ (B[N,K])^T = C[M,N]
        // 
        // For cuBLAS (column-major), we compute C^T = (A @ B^T)^T = B @ A^T
        // C^T[N,M] = B[N,K] @ A[M,K]^T = B[N,K] @ A^T[K,M]
        //
        // Row-major to col-major mapping:
        // - B[N,K] row-major: element [i,j] at B + i*K + j, row stride = K
        //   cuBLAS sees as [K,N] col-major (transpose). To get [N,K], use CUBLAS_OP_T.
        //   lda = K (leading dim of stored [K,N] matrix = K)
        //
        // - A[M,K] row-major: element [i,j] at A + i*K + j, row stride = K  
        //   cuBLAS sees as [K,M] col-major (= A^T). We want A^T, so use CUBLAS_OP_N.
        //   ldb = K (leading dim of stored [K,M] matrix = K)
        //
        // - C[M,N] row-major: element [i,j] at C + i*N + j, row stride = N
        //   cuBLAS sees as [N,M] col-major (= C^T). ldc = N.
        //
        // cublasSgemm(op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        // computes C[m,n] = op(A)[m,k] @ op(B)[k,n]
        // We want C^T[N,M] = B[N,K] @ A^T[K,M]
        // So m=N, n=M, k=K, op(A)=CUBLAS_OP_T(B_stored), op(B)=CUBLAS_OP_N(A_stored)
        
        status = cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            b.data, K,    // B: [N,K] row-major stored as [K,N] col-major, lda=K
            a.data, K,    // A: [M,K] row-major stored as [K,M] col-major, ldb=K
            &beta,
            result.data, N  // C: [M,N] row-major stored as [N,M] col-major, ldc=N
        );
    } else {
        // B is [K, N] row-major
        status = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            b.data, N,    // B: [K, N] row-major
            a.data, K,    // A: [M, K] row-major
            &beta,
            result.data, N  // C: [M, N] row-major
        );
    }
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("autograd::matmul: cuBLAS sgemm failed");
    }
    
    // Set up backward (TAPE-BASED)
    if (result.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new MatMulGradFn();
        
        // ISSUE #48 FIX: Capture stable data from tensors NOW, before they go out of scope
        // Don't store Tensor* - the tensors may be stack variables that become dangling
        // ISSUE #55 FIX: Pass stream for async allocation of owned grad buffers
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        
        // TAPE-BASED: Use external caches or the tensor data directly
        // If cache not provided, assume the tensor data persists (e.g., weights)
        // ISSUE #51 FIX: Copy caches to owned buffers instead of storing dangling pointers
        // If cache not provided, use the tensor data directly (assumes weights persist)
        const float* effective_a_cache = a_cache ? a_cache : a.data;
        const float* effective_b_cache = b_cache ? b_cache : b.data;
        
        grad_fn->set_cache_copy(effective_a_cache, effective_b_cache, M, K, N, handle, stream, transpose_b);
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
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
// ISSUE #68 FIX: Correctly compute source index from decoded destination coordinates
__global__ void kernel_BSHD_bf16_to_BHSD_fp32(
    const __nv_bfloat16* __restrict__ src, // [B, S, H, D]
    float* __restrict__ dst,               // [B, H, S, D]
    int batch, int heads, int seq_len, int head_dim
) {
    const size_t total = static_cast<size_t>(batch) * heads * seq_len * head_dim;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Decode BHSD destination index (idx iterates over output layout)
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq_len;
    const int h = (idx / (head_dim * seq_len)) % heads;
    const int b = idx / (head_dim * seq_len * heads);
    
    // Compute BSHD source index from (b, s, h, d) coordinates
    // Source layout: [B, S, H, D] = b*S*H*D + s*H*D + h*D + d
    const size_t src_idx = (static_cast<size_t>(b) * seq_len * heads * head_dim) +
                           (static_cast<size_t>(s) * heads * head_dim) +
                           (static_cast<size_t>(h) * head_dim) + d;
    
    // idx is already the BHSD destination index (linear iteration over output)
    dst[idx] = __bfloat162float(src[src_idx]);
}

// ISSUE #72 FIX: Reduce GQA gradients from num_heads to num_kv_heads
// FlashAttention backward writes dK/dV for each query head separately (12 heads).
// For GQA with 4 KV heads, we need to SUM the gradients from grouped Q heads.
// E.g., Q heads 0,1,2 all use KV head 0, so dK[kv_head=0] = dK[q_head=0] + dK[q_head=1] + dK[q_head=2]
//
// ISSUE #73 FIX: External FlashAttention library does NOT apply GQA gradient scaling internally.
// When summing gradients from 3 Q heads, we get 3x the correct gradient magnitude.
// Apply gqa_grad_scale = 1.0 / heads_per_kv_group to normalize (same as old custom kernel did).
//
// Input layout:  src [B, S, num_heads, D] bf16 - gradients per query head
// Output layout: dst [B, num_kv_heads, S, D] fp32 - reduced + scaled gradients per KV head
__global__ void kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32(
    const __nv_bfloat16* __restrict__ src,  // [B, S, num_heads, D]
    float* __restrict__ dst,                // [B, num_kv_heads, S, D]
    int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim
) {
    // Each thread handles one element in the output [B, num_kv_heads, S, D]
    const size_t total = static_cast<size_t>(batch) * num_kv_heads * seq_len * head_dim;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Decode destination BHSD index where H = num_kv_heads
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq_len;
    const int kv_h = (idx / (head_dim * seq_len)) % num_kv_heads;
    const int b = idx / (head_dim * seq_len * num_kv_heads);
    
    // GQA grouping: heads_per_kv_group = num_heads / num_kv_heads
    const int heads_per_kv_group = num_heads / num_kv_heads;
    
    // ISSUE #73 FIX: Apply GQA gradient scale to normalize the sum
    // When 3 Q heads each contribute dK/dV for the same KV head, summing gives 3x magnitude.
    // Normalize by dividing by heads_per_kv_group (same as old custom Flash kernel did).
    const float gqa_grad_scale = 1.0f / static_cast<float>(heads_per_kv_group);
    
    // Sum gradients from all Q heads in this KV group
    float sum = 0.0f;
    for (int g = 0; g < heads_per_kv_group; ++g) {
        const int q_head = kv_h * heads_per_kv_group + g;
        // Source layout: [B, S, num_heads, D] = b*S*H*D + s*H*D + h*D + d
        const size_t src_idx = (static_cast<size_t>(b) * seq_len * num_heads * head_dim) +
                               (static_cast<size_t>(s) * num_heads * head_dim) +
                               (static_cast<size_t>(q_head) * head_dim) + d;
        sum += __bfloat162float(src[src_idx]);
    }
    
    // ISSUE #73: Apply GQA scaling to normalize gradient magnitude
    dst[idx] = sum * gqa_grad_scale;
}

/**
 * ScaledDotProductAttentionGradFn - Backward for attention (ISSUE #48 FIX)
 * Forward: O = softmax(Q @ K^T / sqrt(d)) @ V
 * Uses FlashAttention v2 for memory-efficient backward.
 *
 * DOES NOT store Tensor* - stores stable data instead.
 * Requires saving: Q, K, V, O, and softmax_lse from forward.
 * Backward computes: grad_Q, grad_K, grad_V
 */
struct ScaledDotProductAttentionGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool q_requires_grad = false;
    bool k_requires_grad = false;
    bool v_requires_grad = false;
    float* q_grad = nullptr;
    float* k_grad = nullptr;
    float* v_grad = nullptr;
    TensorContract::TensorShape q_shape, k_shape, v_shape;
    GradFn* q_grad_fn = nullptr;
    GradFn* k_grad_fn = nullptr;
    GradFn* v_grad_fn = nullptr;
    bool owns_q_grad_fn = false;  // ISSUE #50: Take ownership to prevent use-after-free
    bool owns_k_grad_fn = false;
    bool owns_v_grad_fn = false;
    
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
    
    // ALiBi slopes (pointer to device memory, not owned - do NOT free)
    const float* alibi_slopes = nullptr;
    
    ScaledDotProductAttentionGradFn() { op_name = "scaled_dot_product_attention"; }
    
    ~ScaledDotProductAttentionGradFn() override {
        release_saved();
        // ISSUE #50: Delete owned grad_fns to complete the chain cleanup
        if (owns_q_grad_fn && q_grad_fn) { delete q_grad_fn; q_grad_fn = nullptr; }
        if (owns_k_grad_fn && k_grad_fn) { delete k_grad_fn; k_grad_fn = nullptr; }
        if (owns_v_grad_fn && v_grad_fn) { delete v_grad_fn; v_grad_fn = nullptr; }
    }
    
    void capture_inputs(Tensor& q, Tensor& k, Tensor& v) {
        q_requires_grad = q.requires_grad;
        k_requires_grad = k.requires_grad;
        v_requires_grad = v.requires_grad;
        q_shape = q.shape;
        k_shape = k.shape;
        v_shape = v.shape;
        
        // ISSUE #50 FIX: Take ownership of captured grad_fns to prevent use-after-free
        q_grad_fn = q.grad_fn;
        if (q.grad_fn && q.owns_grad_fn) {
            owns_q_grad_fn = true;
            q.owns_grad_fn = false;  // Transfer ownership to us
        }
        k_grad_fn = k.grad_fn;
        if (k.grad_fn && k.owns_grad_fn) {
            owns_k_grad_fn = true;
            k.owns_grad_fn = false;
        }
        v_grad_fn = v.grad_fn;
        if (v.grad_fn && v.owns_grad_fn) {
            owns_v_grad_fn = true;
            v.owns_grad_fn = false;
        }
        
        if (q_requires_grad) { q.ensure_grad(); q_grad = q.grad_data(); }  // ISSUE #59: Use accessor
        if (k_requires_grad) { k.ensure_grad(); k_grad = k.grad_data(); }  // ISSUE #59: Use accessor
        if (v_requires_grad) { v.ensure_grad(); v_grad = v.grad_data(); }  // ISSUE #59: Use accessor
    }
    
    void save(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
              int b, int s, int nh, int nkv, int hd, bool is_causal,
              const float* alibi_slopes_ptr, cudaStream_t stream) {
        batch_size = b;
        seq_len = s;
        num_heads = nh;
        num_kv_heads = nkv;
        head_dim = hd;
        causal = is_causal;
        alibi_slopes = alibi_slopes_ptr;  // Save pointer (not owned)
        
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
        
        // ISSUE #72 FIX: FlashAttention backward kernel writes dK/dV using query head index (bidh=0..num_heads-1),
        // NOT the KV head index (bidh / h_h_k_ratio). With GQA (12 Q heads, 4 KV heads), the library writes
        // to positions 0-11 * head_stride, but if we only allocate for 4 KV heads, heads 4-11 write out-of-bounds!
        // This causes STATUS_STACK_BUFFER_OVERRUN crashes.
        //
        // Solution: Allocate dk_bf16/dv_bf16 for num_heads (not num_kv_heads), let FlashAttention write to all,
        // then reduce the 12-head gradients down to 4 KV heads by summing grouped heads in apply().
        const size_t dk_dv_alloc_elems = static_cast<size_t>(b) * s * nh * hd;  // Use num_heads, not num_kv_heads!
        
        cudaMalloc(&dq_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&dk_bf16, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Sized for num_heads
        cudaMalloc(&dv_bf16, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Sized for num_heads
        cudaMalloc(&dout_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMemset(dq_bf16, 0, q_elems * sizeof(__nv_bfloat16));
        cudaMemset(dk_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Zero full buffer
        cudaMemset(dv_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Zero full buffer
        
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
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("scaled_dot_product_attention", this);
        
        // ISSUE #76: Static call counter for tracking which layer/call sees explosion
        static int s_sdpa_bwd_call = 0;
        const int call_idx = ++s_sdpa_bwd_call;
        
        // DIAGNOSTIC: Log grad_output at ENTRY to SDPA backward (before any processing)
        {
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            std::vector<float> grad_sample(std::min(grad_elems, static_cast<size_t>(10000)));
            cudaMemcpy(grad_sample.data(), grad_output.data, grad_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float grad_max = 0.0f;
            double grad_sq_sum = 0.0;
            for (auto& v : grad_sample) {
                if (!std::isnan(v) && !std::isinf(v)) {
                    grad_max = std::max(grad_max, std::abs(v));
                    grad_sq_sum += static_cast<double>(v) * v;
                }
            }
            float grad_rms = std::sqrt(static_cast<float>(grad_sq_sum / grad_sample.size()));
            fprintf(stderr, "[SDPA-BWD-ENTRY] call=%d | grad_output: numel=%zu max=%.6f rms=%.6f\n",
                    call_idx, grad_elems, grad_max, grad_rms);
        }
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        if (!saved_q_bf16 || !saved_k_bf16 || !saved_v_bf16 || !saved_out_bf16) {
            return;  // No saved state
        }
        
        // ISSUE #76: Detect max_seq_len boundary - assume typical max_seq_len values
        const bool is_boundary_1024 = (seq_len >= 920);   // 90% of 1024
        const bool is_boundary_2048 = (seq_len >= 1840);  // 90% of 2048
        const bool is_boundary = is_boundary_1024 || is_boundary_2048;
        
        const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
        const size_t kv_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
        const int block_size = 256;
        const int q_blocks = static_cast<int>((q_elems + block_size - 1) / block_size);
        const int kv_blocks = static_cast<int>((kv_elems + block_size - 1) / block_size);
        
        // DIAGNOSTIC: Log grad_output (FP32 BHSD) BEFORE conversion to BF16
        {
            cudaStreamSynchronize(stream);
            std::vector<float> grad_sample(std::min(q_elems, static_cast<size_t>(10000)));
            cudaMemcpy(grad_sample.data(), grad_output.data, grad_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float grad_max = 0.0f;
            double grad_sum = 0.0, grad_sq_sum = 0.0;
            int grad_nan = 0, grad_inf = 0;
            for (auto& v : grad_sample) {
                if (std::isnan(v)) { grad_nan++; continue; }
                if (std::isinf(v)) { grad_inf++; continue; }
                grad_max = std::max(grad_max, std::abs(v));
                grad_sum += v;
                grad_sq_sum += static_cast<double>(v) * v;
            }
            float grad_mean = static_cast<float>(grad_sum / grad_sample.size());
            float grad_rms = std::sqrt(static_cast<float>(grad_sq_sum / grad_sample.size()));
            fprintf(stderr, "[FA-BWD-GRAD-IN-FP32] call=%d seqlen=%d | nan=%d inf=%d max=%.6f mean=%.6f rms=%.6f\n",
                    call_idx, seq_len, grad_nan, grad_inf, grad_max, grad_mean, grad_rms);
        }
        
        // Convert grad_output (FP32 BHSD) to BF16 BSHD
        kernel_BHSD_fp32_to_BSHD_bf16<<<q_blocks, block_size, 0, stream>>>(
            grad_output.data, dout_bf16, batch_size, num_heads, seq_len, head_dim);
        
        // ISSUE #79 DIAGNOSTIC: Log saved attention output O before FA backward
        // The FA backward computes dP_sum = sum(dO * O), so O magnitude matters
        {
            cudaStreamSynchronize(stream);
            std::vector<__nv_bfloat16> out_sample(std::min(q_elems, static_cast<size_t>(10000)));
            cudaMemcpy(out_sample.data(), saved_out_bf16, out_sample.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            float out_max = 0.0f;
            double out_sum = 0.0, out_sq_sum = 0.0;
            for (auto& v : out_sample) {
                float fv = __bfloat162float(v);
                out_max = std::max(out_max, std::abs(fv));
                out_sum += fv;
                out_sq_sum += fv * fv;
            }
            float out_mean = static_cast<float>(out_sum / out_sample.size());
            float out_rms = std::sqrt(static_cast<float>(out_sq_sum / out_sample.size()));
            fprintf(stderr, "[FA-BWD-SAVED-OUT] call=%d seqlen=%d | out_max=%.6f out_mean=%.6f out_rms=%.6f\n",
                    call_idx, seq_len, out_max, out_mean, out_rms);
        }
        
        // Call FlashAttention backward
        flash_attn_bwd_ex(
            saved_q_bf16,      // Q  [B, S, H, D] bf16
            saved_k_bf16,      // K  [B, S, Hkv, D] bf16
            saved_v_bf16,      // V  [B, S, Hkv, D] bf16
            saved_out_bf16,    // O  [B, S, H, D] bf16
            dout_bf16,         // dO [B, S, H, D] bf16
            saved_softmax_lse, // LSE [B, H, S] fp32
            alibi_slopes,      // ALiBi slopes [num_heads] (saved from forward)
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
        
        // ISSUE #76 DIAGNOSTIC: Check for gradient explosion after FlashAttention backward
        // This identifies which layer (via call_idx) sees the explosion at max_seq_len boundary
        {
            cudaStreamSynchronize(stream);  // Sync to read results
            
            // Read dQ max magnitude
            std::vector<__nv_bfloat16> dq_host(std::min(q_elems, static_cast<size_t>(1000)));
            cudaMemcpy(dq_host.data(), dq_bf16, dq_host.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            float dq_max = 0.0f;
            for (auto& v : dq_host) { dq_max = std::max(dq_max, std::abs(__bfloat162float(v))); }
            
            // Read dK/dV max magnitude (using num_heads buffer, not num_kv_heads)
            const size_t dk_dv_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
            std::vector<__nv_bfloat16> dk_host(std::min(dk_dv_elems, static_cast<size_t>(1000)));
            std::vector<__nv_bfloat16> dv_host(std::min(dk_dv_elems, static_cast<size_t>(1000)));
            cudaMemcpy(dk_host.data(), dk_bf16, dk_host.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            cudaMemcpy(dv_host.data(), dv_bf16, dv_host.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            float dk_max = 0.0f, dv_max = 0.0f;
            for (auto& v : dk_host) { dk_max = std::max(dk_max, std::abs(__bfloat162float(v))); }
            for (auto& v : dv_host) { dv_max = std::max(dv_max, std::abs(__bfloat162float(v))); }
            
            // Check softmax_lse for anomalies
            const size_t lse_elems = static_cast<size_t>(batch_size) * num_heads * seq_len;
            std::vector<float> lse_host(std::min(lse_elems, static_cast<size_t>(1000)));
            cudaMemcpy(lse_host.data(), saved_softmax_lse, lse_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float lse_min = FLT_MAX, lse_max = -FLT_MAX;
            int lse_nan = 0, lse_inf = 0;
            for (auto& v : lse_host) {
                if (std::isnan(v)) lse_nan++;
                else if (std::isinf(v)) lse_inf++;
                else { lse_min = std::min(lse_min, v); lse_max = std::max(lse_max, v); }
            }
            
            // Detect explosion: any gradient > 100 or LSE > 50
            const bool grad_explosion = (dq_max > 100.0f || dk_max > 100.0f || dv_max > 100.0f);
            const bool lse_explosion = (lse_max > 50.0f || lse_nan > 0 || lse_inf > 0);
            
            if (is_boundary || grad_explosion || lse_explosion) {
                fprintf(stderr, "[SDPA-BWD-ISSUE76] call=%d seqlen=%d%s batch=%d heads=%d/%d\n",
                        call_idx, seq_len, is_boundary ? " *** BOUNDARY ***" : "",
                        batch_size, num_heads, num_kv_heads);
                fprintf(stderr, "    dQ_max=%.10f dK_max=%.10f dV_max=%.10f%s\n",
                        dq_max, dk_max, dv_max, grad_explosion ? " *** GRAD EXPLOSION ***" : "");
                fprintf(stderr, "    softmax_lse: nan=%d inf=%d range=[%.10f, %.10f]%s\n",
                        lse_nan, lse_inf, lse_min, lse_max, lse_explosion ? " *** LSE EXPLOSION ***" : "");
            }
        }
        
        // =========================================================================
        // ISSUE #83 REMOVAL: Issue #84 fixed the root cause (missing preprocessing kernel)
        // =========================================================================
        // The dQ/dK normalization below was a BANDAID for the gradient explosion bug.
        // With Issue #84's preprocessing kernel fix, dQ/dK are now at proper magnitude.
        // Keeping this normalization would CRUSH attention gradients, causing vanishing.
        // 
        // Evidence from training_17696307607301724.log:
        //   - attn gradients: 1.96 → 0.08 (24x DECREASE - vanishing!)
        //   - ffn gradients:  1.83 → 0.07 (26x DECREASE - vanishing!)
        //   - rms gradients:  0.02 → 0.60 (30x INCREASE - still has signal)
        // This pattern shows encoder layers are frozen while RMSNorm (closer to output) learns.
        // 
        // DISABLED Issue #83 normalization - use scale=1.0 for all gradients.
        // =========================================================================
        
        // Convert gradients back to FP32 BHSD and accumulate WITHOUT normalization
        if (q_requires_grad && q_grad) {
            float* grad_q_fp32 = nullptr;
            cudaMalloc(&grad_q_fp32, q_elems * sizeof(float));
            cudaMemsetAsync(grad_q_fp32, 0, q_elems * sizeof(float), stream);
            
            kernel_BSHD_bf16_to_BHSD_fp32<<<q_blocks, block_size, 0, stream>>>(
                dq_bf16, grad_q_fp32, batch_size, num_heads, seq_len, head_dim);
            
            // Scale = 1.0 (no normalization - Issue #84 fixed root cause)
            const int acc_blocks = (q_elems + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            kernel_accumulate_grad<<<acc_blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                q_grad, grad_q_fp32, q_elems, 1.0f);
            cudaFreeAsync(grad_q_fp32, stream);
        }
        
        if (k_requires_grad && k_grad) {
            float* grad_k_fp32 = nullptr;
            cudaMalloc(&grad_k_fp32, kv_elems * sizeof(float));
            cudaMemsetAsync(grad_k_fp32, 0, kv_elems * sizeof(float), stream);
            // ISSUE #72 FIX: Use GQA reduction kernel to sum gradients from grouped Q heads
            // dk_bf16 is [B, S, num_heads, D], we reduce to [B, num_kv_heads, S, D]
            kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dk_bf16, grad_k_fp32, batch_size, num_heads, num_kv_heads, seq_len, head_dim);
            // Scale = 1.0 (no normalization - Issue #84 fixed root cause)
            const int acc_blocks = (kv_elems + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            kernel_accumulate_grad<<<acc_blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                k_grad, grad_k_fp32, kv_elems, 1.0f);
            cudaFreeAsync(grad_k_fp32, stream);
        }
        
        if (v_requires_grad && v_grad) {
            float* grad_v_fp32 = nullptr;
            cudaMalloc(&grad_v_fp32, kv_elems * sizeof(float));
            cudaMemsetAsync(grad_v_fp32, 0, kv_elems * sizeof(float), stream);
            // ISSUE #72 FIX: Use GQA reduction kernel to sum gradients from grouped Q heads
            // dv_bf16 is [B, S, num_heads, D], we reduce to [B, num_kv_heads, S, D]
            kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dv_bf16, grad_v_fp32, batch_size, num_heads, num_kv_heads, seq_len, head_dim);
            // Scale = 1.0 (no normalization needed)
            const int acc_blocks = (kv_elems + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            kernel_accumulate_grad<<<acc_blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                v_grad, grad_v_fp32, kv_elems, 1.0f);
            cudaFreeAsync(grad_v_fp32, stream);
        }
        
        // CONTINUE AUTOGRAD CHAIN - call grad_fns for Q, K, V
        if (q_requires_grad && q_grad_fn) {
            Tensor q_view;
            q_view.data = q_grad; q_view.shape = q_shape;
            q_view.owns_data = false; q_view.owns_grad_fn = false; q_view.stream = stream;
            q_grad_fn->apply(q_view, stream);
            q_grad_fn->release_saved();
        }
        if (k_requires_grad && k_grad_fn) {
            Tensor k_view;
            k_view.data = k_grad; k_view.shape = k_shape;
            k_view.owns_data = false; k_view.owns_grad_fn = false; k_view.stream = stream;
            k_grad_fn->apply(k_view, stream);
            k_grad_fn->release_saved();
        }
        if (v_requires_grad && v_grad_fn) {
            Tensor v_view;
            v_view.data = v_grad; v_view.shape = v_shape;
            v_view.owns_data = false; v_view.owns_grad_fn = false; v_view.stream = stream;
            v_grad_fn->apply(v_view, stream);
            v_grad_fn->release_saved();
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
        q_grad = nullptr; k_grad = nullptr; v_grad = nullptr;
        // ISSUE #50: Set ownership false BEFORE delete
        if (owns_q_grad_fn && q_grad_fn) { owns_q_grad_fn = false; delete q_grad_fn; q_grad_fn = nullptr; }
        if (owns_k_grad_fn && k_grad_fn) { owns_k_grad_fn = false; delete k_grad_fn; k_grad_fn = nullptr; }
        if (owns_v_grad_fn && v_grad_fn) { owns_v_grad_fn = false; delete v_grad_fn; v_grad_fn = nullptr; }
    }
};

Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const float* alibi_slopes, float scale, cudaStream_t stream,
    float* cache_softmax_lse  // NEW: Optional cache for legacy backward
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
        alibi_slopes, // ALiBi slopes [num_heads]
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
    
    // cache_softmax_lse is deprecated - autograd backward doesn't need external cache
    (void)cache_softmax_lse;
    
    // Set up backward if needed - ISSUE #48: capture stable data, not Tensor*
    if (requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new ScaledDotProductAttentionGradFn();
        
        // ISSUE #48 FIX: Capture stable data instead of storing dangling Tensor*
        grad_fn->capture_inputs(const_cast<Tensor&>(q), const_cast<Tensor&>(k), const_cast<Tensor&>(v));
        
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
        grad_fn->alibi_slopes = alibi_slopes;  // Save for backward pass (not owned)
        
        // Allocate backward workspace
        const size_t dq_accum_bytes = flash_attn_dq_accum_bytes(batch_size, seq_len, num_heads, head_dim);
        const size_t dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(batch_size, seq_len, num_heads);
        cudaMalloc(&grad_fn->dq_accum, dq_accum_bytes);
        cudaMalloc(&grad_fn->dsoftmax_sum, dsoftmax_sum_bytes);
        
        // ISSUE #72 FIX: FlashAttention backward kernel writes dK/dV using query head index (bidh=0..num_heads-1),
        // NOT the KV head index (bidh / h_h_k_ratio). With GQA (12 Q heads, 4 KV heads), the library writes
        // to positions 0-11 * head_stride, but if we only allocate for 4 KV heads, heads 4-11 write out-of-bounds!
        // This causes STATUS_STACK_BUFFER_OVERRUN crashes.
        //
        // Solution: Allocate dk_bf16/dv_bf16 for num_heads (not num_kv_heads), let FlashAttention write to all,
        // then reduce the 12-head gradients down to 4 KV heads by summing grouped heads in apply().
        const size_t dk_dv_alloc_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;  // Use num_heads!
        
        cudaMalloc(&grad_fn->dq_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dk_bf16, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Sized for num_heads
        cudaMalloc(&grad_fn->dv_bf16, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Sized for num_heads
        cudaMalloc(&grad_fn->dout_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMemset(grad_fn->dq_bf16, 0, q_elems * sizeof(__nv_bfloat16));
        cudaMemset(grad_fn->dk_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Zero full buffer
        cudaMemset(grad_fn->dv_bf16, 0, dk_dv_alloc_elems * sizeof(__nv_bfloat16));  // ISSUE #72: Zero full buffer
        
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;  // ISSUE #54 FIX: Result owns its grad_fn
        
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

// ═══════════════════════════════════════════════════════════════════════════
// ReshapeFromBHSDGradFn - ISSUE #62 FIX: Autograd-tracked BHSD->flat reshape
//
// ROOT CAUSE OF W_o/QKV GRADIENT BUG:
// Encoding_GPU.cu step 5 used Tensor::empty() + launchReshapeFromBHSD()
// which broke the autograd chain (attn_out had no grad_fn).
// When W_o matmul backward called input_grad_fn->apply(), it got nullptr
// because attn_out.grad_fn was never set.
//
// FIX: This operation takes attn_out_bhsd with its grad_fn and produces
// attn_out (flat) that has THIS GradFn. When W_o backward calls
// input_grad_fn->apply(), it calls THIS apply() which:
// 1. Reshapes the gradient from flat [tokens, d_model] to BHSD [B, H, S, D]
// 2. Continues chain to input's grad_fn (ScaledDotProductAttentionGradFn)
// ═══════════════════════════════════════════════════════════════════════════

// CUDA kernel to reshape gradient from [B*S, H*D] flat to [B, H, S, D] BHSD
// This is the inverse of launchReshapeFromBHSD
__global__ void kernel_reshape_flat_to_BHSD(
    const float* __restrict__ flat_grad,   // [B*S, H*D] row-major
    float* __restrict__ bhsd_grad,          // [B, H, S, D] row-major
    int batch_size, int seq_len, int num_heads, int head_dim
) {
    const int total = batch_size * num_heads * seq_len * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Decode BHSD position
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq_len;
    const int h = (idx / (head_dim * seq_len)) % num_heads;
    const int b = idx / (head_dim * seq_len * num_heads);
    
    // Source: flat[b * seq_len + s, h * head_dim + d]
    const int flat_row = b * seq_len + s;
    const int flat_col = h * head_dim + d;
    const int d_model = num_heads * head_dim;
    const int flat_idx = flat_row * d_model + flat_col;
    
    bhsd_grad[idx] = flat_grad[flat_idx];
}

struct ReshapeFromBHSDGradFn : public GradFn {
    // Input tensor info
    GradFn* input_grad_fn = nullptr;
    bool owns_input_grad_fn = false;
    bool input_requires_grad = false;
    float* input_grad = nullptr;  // Pre-allocated gradient buffer
    TensorContract::TensorShape input_shape;
    
    // Dimensions for reshape
    int batch_size = 0;
    int seq_len = 0;
    int num_heads = 0;
    int head_dim = 0;
    
    ReshapeFromBHSDGradFn() { op_name = "reshape_bhsd_to_flat"; }
    
    ~ReshapeFromBHSDGradFn() override {
        release_saved();
        if (owns_input_grad_fn && input_grad_fn) {
            delete input_grad_fn;
            input_grad_fn = nullptr;
        }
    }
    
    void capture_input(Tensor& bhsd_input) {
        input_requires_grad = bhsd_input.requires_grad;
        input_shape = bhsd_input.shape;
        
        // Take ownership of captured grad_fn to prevent use-after-free
        input_grad_fn = bhsd_input.grad_fn;
        if (bhsd_input.grad_fn && bhsd_input.owns_grad_fn) {
            owns_input_grad_fn = true;
            bhsd_input.owns_grad_fn = false;  // Transfer ownership to us
        }
        
        if (input_requires_grad) {
            bhsd_input.ensure_grad();
            input_grad = bhsd_input.grad_data();
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("reshape_bhsd_to_flat", this);
        
        if (applied) return;
        applied = true;
        
        // DIAGNOSTIC: Log incoming gradient to Reshape backward
        {
            static int s_reshape_call = 0;
            const int call_idx = ++s_reshape_call;
            cudaStreamSynchronize(stream);
            const size_t grad_elems = grad_output.numel();
            std::vector<float> grad_sample(std::min(grad_elems, static_cast<size_t>(10000)));
            cudaMemcpy(grad_sample.data(), grad_output.data, grad_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float grad_max = 0.0f;
            double grad_sq_sum = 0.0;
            for (auto& v : grad_sample) {
                if (!std::isnan(v) && !std::isinf(v)) {
                    grad_max = std::max(grad_max, std::abs(v));
                    grad_sq_sum += static_cast<double>(v) * v;
                }
            }
            float grad_rms = std::sqrt(static_cast<float>(grad_sq_sum / grad_sample.size()));
            fprintf(stderr, "[RESHAPE-BWD-IN] call=%d | grad_output: numel=%zu max=%.10f rms=%.10f\n",
                    call_idx, grad_elems, grad_max, grad_rms);
        }
        
        fprintf(stderr, "[ReshapeBHSDtoFlat] apply() called, input_grad_fn=%p, input_requires_grad=%d\n",
                (void*)input_grad_fn, input_requires_grad);
        
        if (!input_requires_grad) {
            fprintf(stderr, "[ReshapeBHSDtoFlat] input doesn't require grad, skipping\n");
            return;
        }
        
        // Reshape gradient from flat [tokens, d_model] to BHSD [B, H, S, D]
        const int total_elems = batch_size * num_heads * seq_len * head_dim;
        const int block_size = 256;
        const int num_blocks = (total_elems + block_size - 1) / block_size;
        
        // Allocate temporary buffer for reshaped gradient
        float* bhsd_grad = nullptr;
        cudaMalloc(&bhsd_grad, total_elems * sizeof(float));
        
        kernel_reshape_flat_to_BHSD<<<num_blocks, block_size, 0, stream>>>(
            grad_output.data, bhsd_grad, batch_size, seq_len, num_heads, head_dim);
        
        // DEBUG: Check for NaN in reshaped gradient
        cudaStreamSynchronize(stream);
        float first_val = 0.0f;
        cudaMemcpy(&first_val, bhsd_grad, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[ReshapeBHSDtoFlat] dims: B=%d S=%d H=%d D=%d total=%d first_val=%.6f\n",
                batch_size, seq_len, num_heads, head_dim, total_elems, first_val);
        
        fprintf(stderr, "[ReshapeBHSDtoFlat] reshaped grad, calling input_grad_fn->apply()...\n");
        
        // Continue chain to attention backward
        if (input_grad_fn) {
            Tensor bhsd_grad_tensor;
            bhsd_grad_tensor.data = bhsd_grad;
            bhsd_grad_tensor.shape = input_shape;
            bhsd_grad_tensor.owns_data = false;
            bhsd_grad_tensor.owns_grad_fn = false;
            bhsd_grad_tensor.stream = stream;
            
            input_grad_fn->apply(bhsd_grad_tensor, stream);
            input_grad_fn->release_saved();
            fprintf(stderr, "[ReshapeBHSDtoFlat] input_grad_fn->apply() done\n");
        } else {
            fprintf(stderr, "[ReshapeBHSDtoFlat] WARNING: no input_grad_fn to continue chain!\n");
        }
        
        // ISSUE #62: Use cudaFreeAsync to ensure kernels in apply() complete before freeing
        // Regular cudaFree might execute before async kernels finish using the buffer
        cudaFreeAsync(bhsd_grad, stream);
    }
    
    void release_saved() override {
        GradFn::release_saved();
    }
};

/**
 * Reshape BHSD tensor to flat [tokens, d_model] with autograd tracking.
 * 
 * ISSUE #62 FIX: This replaces Tensor::empty() + launchReshapeFromBHSD()
 * that broke the autograd chain (output had no grad_fn).
 */
Tensor reshape_bhsd_to_flat(
    Tensor& bhsd_input,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream
) {
    const int tokens = batch_size * seq_len;
    const int d_model = num_heads * head_dim;
    
    // Allocate output tensor in flat layout
    Tensor result = Tensor::empty(TensorContract::TensorShape::make_BSM(tokens, d_model), bhsd_input.requires_grad, stream);
    
    // Call the existing reshape kernel (declared in Encoding_GPU.hpp)
    // This reshapes [B, H, S, D] to [B*S, H*D]
    launchReshapeFromBHSD(bhsd_input.data, result.data, batch_size, seq_len, num_heads, head_dim, stream);
    
    // Set up backward if needed
    if (bhsd_input.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new ReshapeFromBHSDGradFn();
        
        // Capture input for backward
        grad_fn->capture_input(bhsd_input);
        grad_fn->batch_size = batch_size;
        grad_fn->seq_len = seq_len;
        grad_fn->num_heads = num_heads;
        grad_fn->head_dim = head_dim;
        
        result.grad_fn = grad_fn;
        result.owns_grad_fn = true;
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// SplitAndReshapeQKVGradFn - ISSUE #61 FIX: Autograd-tracked QKV split
//
// ROOT CAUSE OF QKV GRADIENT BUG:
// Previously, Encoding_GPU.cu split QKV using cudaMemcpy2D and created new
// Tensor::empty() objects. These have NO grad_fn, breaking the autograd chain!
// When attention backward calls q_grad_fn->apply(), it gets nullptr because
// Q.grad_fn was never set.
//
// FIX: This operation takes qkv_out with its grad_fn and produces Q, K, V
// tensors that have THIS GradFn as their grad_fn. When attention backward
// calls q_grad_fn->apply(), it calls THIS apply() which:
// 1. Combines grad_Q, grad_K, grad_V into grad_qkv
// 2. Calls qkv_out->grad_fn->apply() to continue the chain to W_qkv
// ═══════════════════════════════════════════════════════════════════════════

// CUDA kernel to split QKV: [tokens, qkv_dim] -> Q[tokens, d_model], K[tokens, kv_dim], V[tokens, kv_dim]
__global__ void kernel_split_qkv(
    const float* __restrict__ qkv,     // [tokens, qkv_dim] where qkv_dim = d_model + 2*kv_dim
    float* __restrict__ Q,              // [tokens, d_model]
    float* __restrict__ K,              // [tokens, kv_dim]
    float* __restrict__ V,              // [tokens, kv_dim]
    int tokens, int d_model, int kv_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_qkv_dim = d_model + 2 * kv_dim;
    const int total_q_elems = tokens * d_model;
    const int total_kv_elems = tokens * kv_dim;
    
    // Q elements: idx in [0, tokens * d_model)
    if (idx < total_q_elems) {
        const int token = idx / d_model;
        const int col = idx % d_model;
        Q[idx] = qkv[token * total_qkv_dim + col];
    }
    
    // K elements: idx in [0, tokens * kv_dim)
    // Separate kernel launch or just use offset
}

// More efficient: one kernel that handles all elements
__global__ void kernel_split_qkv_all(
    const float* __restrict__ qkv,     // [tokens, qkv_dim]
    float* __restrict__ Q,              // [tokens, d_model]
    float* __restrict__ K,              // [tokens, kv_dim]
    float* __restrict__ V,              // [tokens, kv_dim]
    int tokens, int d_model, int kv_dim
) {
    const int token = blockIdx.x;
    const int col = threadIdx.x;
    if (token >= tokens) return;
    
    const int total_qkv_dim = d_model + 2 * kv_dim;
    const float* row = qkv + token * total_qkv_dim;
    
    // Copy Q columns [0, d_model)
    if (col < d_model) {
        Q[token * d_model + col] = row[col];
    }
    
    // Copy K columns [d_model, d_model + kv_dim)
    if (col < kv_dim) {
        K[token * kv_dim + col] = row[d_model + col];
    }
    
    // Copy V columns [d_model + kv_dim, end)
    if (col < kv_dim) {
        V[token * kv_dim + col] = row[d_model + kv_dim + col];
    }
}

// Backward: combine grad_Q, grad_K, grad_V into grad_qkv
__global__ void kernel_merge_qkv_grad(
    float* __restrict__ grad_qkv,       // [tokens, qkv_dim] OUTPUT
    const float* __restrict__ grad_Q,    // [tokens, d_model]
    const float* __restrict__ grad_K,    // [tokens, kv_dim]
    const float* __restrict__ grad_V,    // [tokens, kv_dim]
    int tokens, int d_model, int kv_dim
) {
    const int token = blockIdx.x;
    const int col = threadIdx.x;
    if (token >= tokens) return;
    
    const int total_qkv_dim = d_model + 2 * kv_dim;
    float* row = grad_qkv + token * total_qkv_dim;
    
    // Q gradient columns [0, d_model)
    if (col < d_model) {
        row[col] = grad_Q[token * d_model + col];
    }
    
    // K gradient columns [d_model, d_model + kv_dim)
    if (col < kv_dim) {
        row[d_model + col] = grad_K[token * kv_dim + col];
    }
    
    // V gradient columns [d_model + kv_dim, end)
    if (col < kv_dim) {
        row[d_model + kv_dim + col] = grad_V[token * kv_dim + col];
    }
}

// Reshape from BSM to BHSD
__global__ void kernel_reshape_BSM_to_BHSD(
    const float* __restrict__ input,    // [batch*seq, heads*head_dim] BSM layout
    float* __restrict__ output,          // [batch, heads, seq, head_dim] BHSD layout
    int batch, int seq, int heads, int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elems = batch * heads * seq * head_dim;
    if (idx >= total_elems) return;
    
    // Decode BHSD index
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq;
    const int h = (idx / (head_dim * seq)) % heads;
    const int b = idx / (head_dim * seq * heads);
    
    // BSM: [batch*seq, heads*head_dim] row-major
    // Row = b*seq + s, Col = h*head_dim + d
    const int bsm_idx = (b * seq + s) * (heads * head_dim) + h * head_dim + d;
    
    output[idx] = input[bsm_idx];
}

// Backward reshape: BHSD grad to BSM grad
__global__ void kernel_reshape_BHSD_to_BSM(
    const float* __restrict__ grad_bhsd,  // [batch, heads, seq, head_dim] BHSD layout
    float* __restrict__ grad_bsm,          // [batch*seq, heads*head_dim] BSM layout
    int batch, int seq, int heads, int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elems = batch * heads * seq * head_dim;
    if (idx >= total_elems) return;
    
    // Decode BHSD index
    const int d = idx % head_dim;
    const int s = (idx / head_dim) % seq;
    const int h = (idx / (head_dim * seq)) % heads;
    const int b = idx / (head_dim * seq * heads);
    
    // BSM: [batch*seq, heads*head_dim] row-major
    const int bsm_idx = (b * seq + s) * (heads * head_dim) + h * head_dim + d;
    
    grad_bsm[bsm_idx] = grad_bhsd[idx];
}

/**
 * GradFn for split_and_reshape_qkv operation
 * 
 * This bridges the gap between qkv_out (from matmul) and Q_bhsd/K_bhsd/V_bhsd (for attention).
 * 
 * Forward: qkv_out [tokens, qkv_dim] -> Q_bhsd, K_bhsd, V_bhsd [batch, heads, seq, head_dim]
 * Backward: grad_Q_bhsd, grad_K_bhsd, grad_V_bhsd -> grad_qkv_out -> W_qkv gradients
 */
struct SplitAndReshapeQKVGradFn : public GradFn {
    // Which output this GradFn is attached to (Q, K, or V)
    enum class OutputType { Q, K, V };
    OutputType output_type = OutputType::Q;
    
    // Shared state for all three outputs (only one instance owns the upstream chain)
    struct SharedState {
        GradFn* qkv_grad_fn = nullptr;       // Grad fn of qkv_out (the matmul result)
        bool owns_qkv_grad_fn = false;
        Tensor qkv_out_ref;                  // Reference to qkv_out (for grad buffer access)
        
        // Gradients from Q, K, V (accumulated before calling upstream)
        // Using Tensor for proper RAII and shared_ptr<Tensor> grad semantics
        Tensor grad_Q_bsm;                   // [tokens, d_model] reshaped grad
        Tensor grad_K_bsm;                   // [tokens, kv_dim] reshaped grad
        Tensor grad_V_bsm;                   // [tokens, kv_dim] reshaped grad
        
        // Dimensions
        int tokens = 0;
        int d_model = 0;
        int kv_dim = 0;
        int batch = 0;
        int seq = 0;
        int num_heads = 0;
        int num_kv_heads = 0;
        int head_dim = 0;
        
        // Count of how many outputs have been processed
        std::atomic<int> apply_count{0};
        
        ~SharedState() {
            if (owns_qkv_grad_fn && qkv_grad_fn) {
                delete qkv_grad_fn;
                qkv_grad_fn = nullptr;
            }
            // Tensor members will be cleaned up automatically via RAII
        }
    };
    
    std::shared_ptr<SharedState> shared;
    
    SplitAndReshapeQKVGradFn() { op_name = "split_and_reshape_qkv"; }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        const char* type_str = (output_type == OutputType::Q) ? "Q" : 
                               (output_type == OutputType::K) ? "K" : "V";
        
        if (!shared) {
            fprintf(stderr, "[SplitQKV-%s] SKIP: shared is null!\n", type_str);
            return;
        }
        if (applied) {
            fprintf(stderr, "[SplitQKV-%s] SKIP: already applied\n", type_str);
            return;
        }
        applied = true;
        
        auto& state = *shared;
        fprintf(stderr, "[SplitQKV-%s] apply() called, current count=%d\n", type_str, state.apply_count.load());
        
        const int total_elems_q = state.batch * state.num_heads * state.seq * state.head_dim;
        const int total_elems_kv = state.batch * state.num_kv_heads * state.seq * state.head_dim;
        const int block_size = 256;
        
        // Reshape this output's gradient from BHSD to BSM
        if (output_type == OutputType::Q) {
            const int blocks = (total_elems_q + block_size - 1) / block_size;
            kernel_reshape_BHSD_to_BSM<<<blocks, block_size, 0, stream>>>(
                grad_output.data, state.grad_Q_bsm.data,
                state.batch, state.seq, state.num_heads, state.head_dim);
        } else if (output_type == OutputType::K) {
            const int blocks = (total_elems_kv + block_size - 1) / block_size;
            kernel_reshape_BHSD_to_BSM<<<blocks, block_size, 0, stream>>>(
                grad_output.data, state.grad_K_bsm.data,
                state.batch, state.seq, state.num_kv_heads, state.head_dim);
        } else {  // V
            const int blocks = (total_elems_kv + block_size - 1) / block_size;
            kernel_reshape_BHSD_to_BSM<<<blocks, block_size, 0, stream>>>(
                grad_output.data, state.grad_V_bsm.data,
                state.batch, state.seq, state.num_kv_heads, state.head_dim);
        }
        
        // Check if all three outputs have been processed
        const int count = state.apply_count.fetch_add(1) + 1;
        fprintf(stderr, "[SplitQKV-%s] after increment, count=%d (need 3)\n", type_str, count);
        
        if (count == 3) {
            fprintf(stderr, "[SplitQKV-%s] ALL THREE RECEIVED! Merging and continuing chain...\n", type_str);
            
            // ISSUE #64 FIX: Synchronize stream to ensure all three reshape kernels complete
            // before merge kernel reads from their output buffers!
            // Without this, merge kernel may read uninitialized/partial data from grad_Q/K/V_bsm
            cudaStreamSynchronize(stream);
            
            // ========== ISSUE #79 DIAGNOSTIC: Prove dQ/dK >> dV hypothesis ==========
            // Read back grad_Q, grad_K, grad_V max magnitudes BEFORE merge
            {
                static int merge_call_idx = 0;
                merge_call_idx++;
                
                const size_t q_elems = static_cast<size_t>(state.tokens) * state.d_model;
                const size_t kv_elems = static_cast<size_t>(state.tokens) * state.kv_dim;
                
                // Sample first 10000 elements for efficiency
                const size_t sample_q = std::min(q_elems, static_cast<size_t>(10000));
                const size_t sample_kv = std::min(kv_elems, static_cast<size_t>(10000));
                
                std::vector<float> q_host(sample_q), k_host(sample_kv), v_host(sample_kv);
                cudaMemcpy(q_host.data(), state.grad_Q_bsm.data, sample_q * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(k_host.data(), state.grad_K_bsm.data, sample_kv * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(v_host.data(), state.grad_V_bsm.data, sample_kv * sizeof(float), cudaMemcpyDeviceToHost);
                
                float q_max = 0.0f, k_max = 0.0f, v_max = 0.0f;
                float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
                for (auto& x : q_host) { q_max = std::max(q_max, std::abs(x)); q_sum += x * x; }
                for (auto& x : k_host) { k_max = std::max(k_max, std::abs(x)); k_sum += x * x; }
                for (auto& x : v_host) { v_max = std::max(v_max, std::abs(x)); v_sum += x * x; }
                
                const float q_rms = std::sqrt(q_sum / sample_q);
                const float k_rms = std::sqrt(k_sum / sample_kv);
                const float v_rms = std::sqrt(v_sum / sample_kv);
                
                // Compute ratios to prove hypothesis
                const float qk_to_v_max_ratio = (v_max > 1e-10f) ? (std::max(q_max, k_max) / v_max) : -1.0f;
                const float qk_to_v_rms_ratio = (v_rms > 1e-10f) ? (std::max(q_rms, k_rms) / v_rms) : -1.0f;
                
                fprintf(stderr, "[Issue79-MERGE-DIAG] call=%d tokens=%d d_model=%d kv_dim=%d\n",
                        merge_call_idx, state.tokens, state.d_model, state.kv_dim);
                fprintf(stderr, "[Issue79-MERGE-DIAG]   grad_Q: max=%.10f rms=%.10f\n", q_max, q_rms);
                fprintf(stderr, "[Issue79-MERGE-DIAG]   grad_K: max=%.10f rms=%.10f\n", k_max, k_rms);
                fprintf(stderr, "[Issue79-MERGE-DIAG]   grad_V: max=%.10f rms=%.10f\n", v_max, v_rms);
                fprintf(stderr, "[Issue79-MERGE-DIAG]   RATIO max(Q,K)/V: max_ratio=%.1fx rms_ratio=%.1fx %s\n",
                        qk_to_v_max_ratio, qk_to_v_rms_ratio,
                        (qk_to_v_max_ratio > 1000.0f) ? "*** dQ/dK >> dV CONFIRMED! ***" : "");
            }
            // ========== END ISSUE #79 DIAGNOSTIC ==========
            
            // All three gradients received - merge and continue chain
            state.qkv_out_ref.ensure_grad();
            float* qkv_grad = state.qkv_out_ref.grad_data();
            
            fprintf(stderr, "[SplitQKV-%s] qkv_grad=%p qkv_grad_fn=%p requires_grad=%d\n", 
                    type_str, (void*)qkv_grad, (void*)state.qkv_grad_fn, state.qkv_out_ref.requires_grad);
            
            const int threads = std::max(state.d_model, state.kv_dim);
            kernel_merge_qkv_grad<<<state.tokens, threads, 0, stream>>>(
                qkv_grad,
                state.grad_Q_bsm.data, state.grad_K_bsm.data, state.grad_V_bsm.data,
                state.tokens, state.d_model, state.kv_dim);
            
            // Continue the chain to qkv_out -> W_qkv
            if (state.qkv_out_ref.requires_grad && state.qkv_grad_fn) {
                fprintf(stderr, "[SplitQKV-%s] Calling qkv_grad_fn->apply()...\n", type_str);
                Tensor qkv_grad_tensor;
                qkv_grad_tensor.data = qkv_grad;
                qkv_grad_tensor.shape = state.qkv_out_ref.shape;
                qkv_grad_tensor.owns_data = false;
                qkv_grad_tensor.owns_grad_fn = false;
                qkv_grad_tensor.stream = stream;
                
                state.qkv_grad_fn->apply(qkv_grad_tensor, stream);
                state.qkv_grad_fn->release_saved();
                fprintf(stderr, "[SplitQKV-%s] qkv_grad_fn->apply() complete\n", type_str);
            } else {
                fprintf(stderr, "[SplitQKV-%s] SKIP qkv_grad_fn: requires_grad=%d qkv_grad_fn=%p\n", 
                        type_str, state.qkv_out_ref.requires_grad, (void*)state.qkv_grad_fn);
            }
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        // SharedState cleanup happens via shared_ptr destructor
    }
};

/**
 * Split qkv_out [tokens, qkv_dim] into Q, K, V and reshape to BHSD layout.
 * 
 * This replaces the manual cudaMemcpy2D + launchQKVReshapeToBHSD in Encoding_GPU.cu
 * with a properly autograd-tracked operation.
 * 
 * @param qkv_out  Input tensor [tokens, d_model + 2*kv_dim] from matmul(ln1_out, W_qkv)
 * @param batch    Batch size
 * @param seq      Sequence length (tokens = batch * seq)
 * @param num_heads     Number of Q heads
 * @param num_kv_heads  Number of K/V heads (GQA)
 * @param head_dim      Dimension per head
 * @param stream   CUDA stream
 * @return Tuple of (Q_bhsd, K_bhsd, V_bhsd) with autograd tracking
 */
std::tuple<Tensor, Tensor, Tensor> split_and_reshape_qkv(
    Tensor& qkv_out,
    int batch, int seq, int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream
) {
    const int tokens = batch * seq;
    const int d_model = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int qkv_dim = d_model + 2 * kv_dim;
    
    // Validate input shape
    if (qkv_out.shape.flat.rows != tokens || qkv_out.shape.flat.cols != qkv_dim) {
        throw std::invalid_argument(
            "split_and_reshape_qkv: qkv_out shape mismatch. Expected [" + 
            std::to_string(tokens) + ", " + std::to_string(qkv_dim) + 
            "], got [" + std::to_string(qkv_out.shape.flat.rows) + ", " + 
            std::to_string(qkv_out.shape.flat.cols) + "]");
    }
    
    // Allocate output tensors in BHSD layout
    auto q_shape = TensorContract::TensorShape::make_BHSD(batch, num_heads, seq, head_dim);
    auto k_shape = TensorContract::TensorShape::make_BHSD(batch, num_kv_heads, seq, head_dim);
    auto v_shape = TensorContract::TensorShape::make_BHSD(batch, num_kv_heads, seq, head_dim);
    
    bool requires_grad = qkv_out.requires_grad;
    Tensor Q_bhsd = Tensor::zeros(q_shape, requires_grad, stream);
    Tensor K_bhsd = Tensor::zeros(k_shape, requires_grad, stream);
    Tensor V_bhsd = Tensor::zeros(v_shape, requires_grad, stream);
    
    // Intermediate BSM tensors for split (using Tensor for RAII)
    Tensor Q_bsm = Tensor::zeros(TensorContract::TensorShape::make_BSM(tokens, d_model), false, stream);
    Tensor K_bsm = Tensor::zeros(TensorContract::TensorShape::make_BSM(tokens, kv_dim), false, stream);
    Tensor V_bsm = Tensor::zeros(TensorContract::TensorShape::make_BSM(tokens, kv_dim), false, stream);
    
    // Forward: split qkv_out into Q, K, V (BSM layout)
    const int threads = std::max(d_model, kv_dim);
    kernel_split_qkv_all<<<tokens, threads, 0, stream>>>(
        qkv_out.data, Q_bsm.data, K_bsm.data, V_bsm.data, tokens, d_model, kv_dim);
    
    // Reshape BSM -> BHSD
    const int total_q_elems = batch * num_heads * seq * head_dim;
    const int total_kv_elems = batch * num_kv_heads * seq * head_dim;
    const int block_size = 256;
    const int q_blocks = (total_q_elems + block_size - 1) / block_size;
    const int kv_blocks = (total_kv_elems + block_size - 1) / block_size;
    
    kernel_reshape_BSM_to_BHSD<<<q_blocks, block_size, 0, stream>>>(
        Q_bsm.data, Q_bhsd.data, batch, seq, num_heads, head_dim);
    kernel_reshape_BSM_to_BHSD<<<kv_blocks, block_size, 0, stream>>>(
        K_bsm.data, K_bhsd.data, batch, seq, num_kv_heads, head_dim);
    kernel_reshape_BSM_to_BHSD<<<kv_blocks, block_size, 0, stream>>>(
        V_bsm.data, V_bhsd.data, batch, seq, num_kv_heads, head_dim);
    
    // Q_bsm, K_bsm, V_bsm will be freed automatically when they go out of scope (RAII)
    
    // Set up backward if needed
    if (requires_grad) {
        // Create shared state for all three outputs
        auto shared = std::make_shared<SplitAndReshapeQKVGradFn::SharedState>();
        shared->tokens = tokens;
        shared->d_model = d_model;
        shared->kv_dim = kv_dim;
        shared->batch = batch;
        shared->seq = seq;
        shared->num_heads = num_heads;
        shared->num_kv_heads = num_kv_heads;
        shared->head_dim = head_dim;
        
        // Ensure qkv_out has gradient buffer allocated before sharing
        qkv_out.ensure_grad();
        
        // Store reference to qkv_out (for grad buffer access via ensure_grad/grad_data)
        shared->qkv_out_ref = Tensor::from_ptr(
            qkv_out.data, qkv_out.shape, false, qkv_out.requires_grad);
        shared->qkv_out_ref.share_grad(qkv_out);  // Share gradient buffer with original
        
        // Take ownership of upstream grad_fn
        shared->qkv_grad_fn = qkv_out.grad_fn;
        if (qkv_out.grad_fn && qkv_out.owns_grad_fn) {
            shared->owns_qkv_grad_fn = true;
            qkv_out.owns_grad_fn = false;  // Transfer ownership
        }
        
        // Allocate gradient Tensors for BSM intermediate results
        shared->grad_Q_bsm = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(tokens, d_model), false, stream);
        shared->grad_K_bsm = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(tokens, kv_dim), false, stream);
        shared->grad_V_bsm = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(tokens, kv_dim), false, stream);
        
        // Create GradFns for each output
        auto q_grad_fn = new SplitAndReshapeQKVGradFn();
        q_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::Q;
        q_grad_fn->shared = shared;
        
        auto k_grad_fn = new SplitAndReshapeQKVGradFn();
        k_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::K;
        k_grad_fn->shared = shared;
        
        auto v_grad_fn = new SplitAndReshapeQKVGradFn();
        v_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::V;
        v_grad_fn->shared = shared;
        
        Q_bhsd.is_leaf = false;
        Q_bhsd.grad_fn = q_grad_fn;
        Q_bhsd.owns_grad_fn = true;
        
        K_bhsd.is_leaf = false;
        K_bhsd.grad_fn = k_grad_fn;
        K_bhsd.owns_grad_fn = true;
        
        V_bhsd.is_leaf = false;
        V_bhsd.grad_fn = v_grad_fn;
        V_bhsd.owns_grad_fn = true;
    }
    
    return {std::move(Q_bhsd), std::move(K_bhsd), std::move(V_bhsd)};
}


// =============================================================================
// ISSUE #119: RoPEGradFn - Autograd wrapper for RoPE rotation
// =============================================================================
// RoPE forward rotates Q and K tensors IN-PLACE. The backward pass must apply
// the INVERSE rotation (R(-θ) = R(θ)^T) to the gradients dQ and dK to correctly
// propagate gradients through the rotation.
//
// Without this, gradients remain in the "rotated space" and are incorrect,
// causing training to learn wrong attention patterns.
//
// Uses SharedState pattern (like SplitAndReshapeQKVGradFn) to coordinate Q and K.
// When both Q and K backward are complete, continues the autograd chain.
// =============================================================================

struct RoPEGradFn : public GradFn {
    /**
     * OutputType identifies which tensor (Q or K) this GradFn is attached to.
     */
    enum class OutputType { Q, K };
    OutputType output_type = OutputType::Q;
    
    /**
     * SharedState holds the common data for coordinating Q and K backward passes.
     * - Stores upstream grad_fns for Q and K (from split_and_reshape_qkv)
     * - Stores RoPE parameters (inv_freq, dimensions)
     * - Uses atomic counter to detect when both Q and K backward are complete
     */
    struct SharedState {
        // Upstream grad_fns for Q and K (from split_and_reshape_qkv output)
        GradFn* q_upstream_grad_fn = nullptr;
        GradFn* k_upstream_grad_fn = nullptr;
        bool owns_q_upstream = false;
        bool owns_k_upstream = false;
        
        // ISSUE #48 FIX: Don't store Tensor by value - operator= is deleted
        // Instead, store only what we need for backward: requires_grad flags
        bool q_requires_grad = false;
        bool k_requires_grad = false;
        
        // RoPE parameters (captured at forward time)
        const float* inv_freq = nullptr;
        int batch_size = 0;
        int num_q_heads = 0;
        int num_kv_heads = 0;
        int seq_len = 0;
        int head_dim = 0;
        int rotary_dim = 0;
        
        // Atomic counter: when reaches 2, both Q and K backward complete
        std::atomic<int> apply_count{0};
        
        ~SharedState() {
            if (owns_q_upstream && q_upstream_grad_fn) {
                delete q_upstream_grad_fn;
            }
            if (owns_k_upstream && k_upstream_grad_fn) {
                delete k_upstream_grad_fn;
            }
        }
    };
    
    std::shared_ptr<SharedState> shared;
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!shared) {
            fprintf(stderr, "[RoPEGradFn] ERROR: shared state is null!\n");
            return;
        }
        auto& state = *shared;
        
        const char* type_str = (output_type == OutputType::Q) ? "Q" : "K";
        // Shape is 4D (BHSD layout) - use as_4d() accessor
        const auto& s4d = grad_output.shape.as_4d();
        AG_TRACE("[RoPEGradFn-%s] apply() ENTER grad_output.data=%p shape=[%d,%d,%d,%d]\n",
                 type_str, grad_output.data, 
                 s4d.batch, s4d.heads, s4d.seq, s4d.head_dim);
        
        // Apply inverse RoPE rotation to THIS gradient IN-PLACE
        // The grad_output.data points to the upstream gradient buffer (dQ or dK)
        // We need a mutable copy since grad_output is const
        float* grad_data = const_cast<float*>(grad_output.data);
        
        if (output_type == OutputType::Q) {
            // Inverse-rotate dQ only - K gradients handled by K's GradFn
            PBM::launchRoPERotationGQA_backward(
                grad_data,        // dQ - modified in-place
                nullptr,          // dK = nullptr, handle separately
                state.inv_freq,
                state.batch_size,
                state.num_q_heads,
                state.num_kv_heads,
                state.seq_len,
                state.head_dim,
                state.rotary_dim,
                stream
            );
            AG_TRACE("[RoPEGradFn-Q] Inverse RoPE applied to dQ\n");
        } else {
            // Inverse-rotate dK only - Q gradients handled by Q's GradFn
            PBM::launchRoPERotationGQA_backward(
                nullptr,          // dQ = nullptr, handle separately
                grad_data,        // dK - modified in-place
                state.inv_freq,
                state.batch_size,
                state.num_q_heads,
                state.num_kv_heads,
                state.seq_len,
                state.head_dim,
                state.rotary_dim,
                stream
            );
            AG_TRACE("[RoPEGradFn-K] Inverse RoPE applied to dK\n");
        }
        
        // Increment counter to track completion
        const int count = state.apply_count.fetch_add(1) + 1;
        AG_TRACE("[RoPEGradFn-%s] apply_count = %d/2\n", type_str, count);
        
        // Continue upstream chain for THIS output immediately
        // (Unlike SplitAndReshapeQKV, we don't need to wait for both because
        //  Q and K have independent upstream paths after split_and_reshape_qkv)
        if (output_type == OutputType::Q) {
            if (state.q_requires_grad && state.q_upstream_grad_fn) {
                AG_TRACE("[RoPEGradFn-Q] Continuing to q_upstream_grad_fn...\n");
                state.q_upstream_grad_fn->apply(grad_output, stream);
                state.q_upstream_grad_fn->release_saved();
            }
        } else {
            if (state.k_requires_grad && state.k_upstream_grad_fn) {
                AG_TRACE("[RoPEGradFn-K] Continuing to k_upstream_grad_fn...\n");
                state.k_upstream_grad_fn->apply(grad_output, stream);
                state.k_upstream_grad_fn->release_saved();
            }
        }
        
        AG_TRACE("[RoPEGradFn-%s] apply() EXIT\n", type_str);
    }
    
    void release_saved() override {
        GradFn::release_saved();
        // SharedState cleanup happens via shared_ptr destructor
    }
};


/**
 * Apply RoPE rotation to Q and K tensors IN-PLACE with autograd tracking.
 * 
 * ISSUE #119: This function wraps PBM::launchRoPERotationGQA with proper autograd
 * so that the backward pass correctly applies inverse rotation to dQ and dK.
 * 
 * RoPE Math:
 *   Forward:  Q' = R(θ) * Q,  K' = R(θ) * K   (rotation by position-dependent angle)
 *   Backward: dQ = R(-θ) * dQ', dK = R(-θ) * dK'  (inverse rotation)
 * 
 * Since R(-θ) = R(θ)^T and R is orthogonal, this is mathematically correct.
 * 
 * @param Q        Query tensor [B, num_heads, S, head_dim] - modified IN-PLACE
 * @param K        Key tensor [B, num_kv_heads, S, head_dim] - modified IN-PLACE
 * @param inv_freq Inverse frequencies for RoPE [rotary_dim/2]
 * @param batch_size   Batch size
 * @param num_q_heads  Number of Q heads
 * @param num_kv_heads Number of K/V heads (GQA: num_kv_heads < num_q_heads)
 * @param seq_len      Sequence length
 * @param head_dim     Dimension per head
 * @param rotary_dim   Number of dimensions to rotate (typically head_dim or head_dim/2)
 * @param stream       CUDA stream
 */
void rope_rotation(
    Tensor& Q,
    Tensor& K,
    const float* inv_freq,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream
) {
    // RULE 20: Fail loud validation
    if (!Q.data) {
        throw std::runtime_error("rope_rotation: Q.data is NULL");
    }
    if (!K.data) {
        throw std::runtime_error("rope_rotation: K.data is NULL");
    }
    if (!inv_freq) {
        throw std::runtime_error("rope_rotation: inv_freq is NULL");
    }
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        throw std::runtime_error("rope_rotation: invalid rotary_dim=" + std::to_string(rotary_dim) +
                                 " (head_dim=" + std::to_string(head_dim) + ")");
    }
    
    AG_TRACE("[rope_rotation] ENTER Q.data=%p K.data=%p batch=%d seq=%d heads=%d/%d dim=%d rotary=%d\n",
             Q.data, K.data, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, rotary_dim);
    
    // Forward pass: Apply RoPE rotation IN-PLACE to Q and K
    PBM::launchRoPERotationGQA(
        Q.data, K.data, inv_freq,
        batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
        stream
    );
    
    // Setup backward pass if either tensor requires gradients
    const bool requires_grad = Q.requires_grad || K.requires_grad;
    
    if (requires_grad) {
        AG_TRACE("[rope_rotation] Setting up RoPEGradFn for backward...\n");
        
        // Create shared state for coordinating Q and K backward
        auto shared = std::make_shared<RoPEGradFn::SharedState>();
        
        // Capture upstream grad_fns (transfer ownership from input tensors)
        if (Q.grad_fn) {
            shared->q_upstream_grad_fn = Q.grad_fn;
            shared->owns_q_upstream = Q.owns_grad_fn;
            Q.owns_grad_fn = false;  // Transfer ownership to SharedState
        }
        if (K.grad_fn) {
            shared->k_upstream_grad_fn = K.grad_fn;
            shared->owns_k_upstream = K.owns_grad_fn;
            K.owns_grad_fn = false;  // Transfer ownership to SharedState
        }
        
        // ISSUE #48 FIX: Store requires_grad flags, not Tensor copies
        // (Tensor::operator= is deleted, and we only need these flags)
        shared->q_requires_grad = Q.requires_grad;
        shared->k_requires_grad = K.requires_grad;
        
        // Capture RoPE parameters
        shared->inv_freq = inv_freq;
        shared->batch_size = batch_size;
        shared->num_q_heads = num_q_heads;
        shared->num_kv_heads = num_kv_heads;
        shared->seq_len = seq_len;
        shared->head_dim = head_dim;
        shared->rotary_dim = rotary_dim;
        
        // Create and attach GradFn for Q
        auto q_grad_fn = new RoPEGradFn();
        q_grad_fn->output_type = RoPEGradFn::OutputType::Q;
        q_grad_fn->shared = shared;
        Q.is_leaf = false;
        Q.grad_fn = q_grad_fn;
        Q.owns_grad_fn = true;
        
        // Create and attach GradFn for K
        auto k_grad_fn = new RoPEGradFn();
        k_grad_fn->output_type = RoPEGradFn::OutputType::K;
        k_grad_fn->shared = shared;
        K.is_leaf = false;
        K.grad_fn = k_grad_fn;
        K.owns_grad_fn = true;
        
        AG_TRACE("[rope_rotation] RoPEGradFn attached: Q.grad_fn=%p K.grad_fn=%p\n",
                 Q.grad_fn, K.grad_fn);
    }
    
    AG_TRACE("[rope_rotation] EXIT\n");
}

}  // namespace autograd

}  // namespace GRIM
