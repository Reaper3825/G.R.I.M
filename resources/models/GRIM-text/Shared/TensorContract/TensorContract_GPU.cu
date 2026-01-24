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
// ═══════════════════════════════════════════════════════════════════════════

// Global buffers for capturing gradient contributions (set by TrainingState)
float* g_debug_lm_head_only_grad = nullptr;   // Where to copy LM head backward contribution
float* g_debug_embedding_only_grad = nullptr; // Where to copy embedding backward contribution
size_t g_debug_grad_buffer_size = 0;          // Size in elements (vocab_size * d_model)
bool g_debug_capture_enabled = false;         // Enable/disable capturing

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
    
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        dx[i] = (dy[i] * gamma[i] - x[i] * scale) * inv_rms;
        
        // Accumulate grad_gamma: dgamma[i] += dy[i] * x[i] * inv_rms
        if (grad_gamma) {
            atomicAdd(&grad_gamma[i], dy[i] * x[i] * inv_rms);
        }
    }
}

// Embedding forward: gather from embedding table
__global__ void kernel_embedding_forward(
    const int* token_ids,       // [tokens]
    const float* weight,        // [vocab_size, d_model]
    float* output,              // [tokens, d_model]
    int tokens,
    int d_model
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const int token_id = token_ids[token_idx];
    const float* weight_row = weight + static_cast<size_t>(token_id) * d_model;
    float* output_row = output + static_cast<size_t>(token_idx) * d_model;
    
    // Gather: output[token_idx] = weight[token_id]
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        output_row[i] = weight_row[i];
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

}  // anonymous namespace

//======================================================//
//  GradFn Subclasses
//======================================================//

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
                grad_output.data, token_ids, g_pcgrad_temp_buffer, num_tokens, d_model);
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
                grad_output.data, token_ids, weight_grad, num_tokens, d_model);
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

Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens, cudaStream_t stream) {
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
    
    const int d_model = weight.shape.as_2d().cols;
    auto output_shape = TensorContract::TensorShape::make_BSM(num_tokens, d_model);
    Tensor result = Tensor::empty(output_shape, weight.requires_grad, stream);
    
    // Forward: gather from weight table
    kernel_embedding_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        token_ids, weight.data, result.data, num_tokens, d_model);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (weight.requires_grad) {
        result.is_leaf = false;
        auto* grad_fn = new EmbeddingGradFn();
        grad_fn->capture_weight(const_cast<Tensor&>(weight));
        grad_fn->save(token_ids, num_tokens, d_model, true, stream);
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
        
        if (!cublas_handle) {
            throw std::runtime_error("MatMulGradFn::apply: cublas_handle is NULL");
        }
        
        const float alpha = 1.0f;
        const float beta_accum = 1.0f;  // Accumulate to existing gradient
        
        cublasSetStream(cublas_handle, stream);
        
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

        // CONTINUE AUTOGRAD CHAIN (Recursive) - ISSUE #48 FIX: Use stored grad_fn pointers
        if (a_requires_grad && a_grad_fn) {
            if (a_grad_fn->op_name) {
                Tensor view;
                view.data = grad_a; view.shape = a_shape;
                view.owns_data = false; view.owns_grad_fn = false; view.stream = stream;
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
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("scaled_dot_product_attention", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
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
        
        // Convert gradients back to FP32 BHSD and accumulate - using captured data
        if (q_requires_grad && q_grad) {
            float* grad_q_fp32 = nullptr;
            cudaMalloc(&grad_q_fp32, q_elems * sizeof(float));
            kernel_BSHD_bf16_to_BHSD_fp32<<<q_blocks, block_size, 0, stream>>>(
                dq_bf16, grad_q_fp32, batch_size, num_heads, seq_len, head_dim);
            // Accumulate into pre-allocated gradient buffer
            const int acc_blocks = (q_elems + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            kernel_accumulate_grad<<<acc_blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                q_grad, grad_q_fp32, q_elems, 1.0f);
            cudaFree(grad_q_fp32);
        }
        
        if (k_requires_grad && k_grad) {
            float* grad_k_fp32 = nullptr;
            cudaMalloc(&grad_k_fp32, kv_elems * sizeof(float));
            kernel_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dk_bf16, grad_k_fp32, batch_size, num_kv_heads, seq_len, head_dim);
            // Accumulate into pre-allocated gradient buffer
            const int acc_blocks = (kv_elems + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            kernel_accumulate_grad<<<acc_blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                k_grad, grad_k_fp32, kv_elems, 1.0f);
            cudaFree(grad_k_fp32);
        }
        
        if (v_requires_grad && v_grad) {
            float* grad_v_fp32 = nullptr;
            cudaMalloc(&grad_v_fp32, kv_elems * sizeof(float));
            kernel_BSHD_bf16_to_BHSD_fp32<<<kv_blocks, block_size, 0, stream>>>(
                dv_bf16, grad_v_fp32, batch_size, num_kv_heads, seq_len, head_dim);
            // Accumulate into pre-allocated gradient buffer
            const int acc_blocks = (kv_elems + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE;
            kernel_accumulate_grad<<<acc_blocks, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                v_grad, grad_v_fp32, kv_elems, 1.0f);
            cudaFree(grad_v_fp32);
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
        cudaMalloc(&grad_fn->dq_bf16, q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dk_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dv_bf16, kv_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_fn->dout_bf16, q_elems * sizeof(__nv_bfloat16));
        
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

}  // namespace autograd

}  // namespace GRIM
