//======================================================//
//  TensorContract_GPU.cu
//  CUDA implementation of type-safe tensor operations
//======================================================//
#include "TensorContract_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../TensorConversion/TensorConversion.hpp"  // Layout conversions - single source of truth
#include "../LogRecorder/LogRecorder.hpp"
#include "../../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Layers/Attention/QKV_Projector.hpp"  // ISSUE #62: For launchReshapeFromBHSD
#include "../PBM/PositionalBiasMethod.hpp"  // ISSUE #119: For RoPE autograd backward
#include "../EquationLogging/EquationLogging.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>  // Issue #107: Philox PRNG for Xavier init
#include <device_launch_parameters.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>

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
bool g_autograd_verbose = false;  // DEBUG: Set true to trace autograd chain (KILLS PERF with fflush!)

// ═══════════════════════════════════════════════════════════════════════════
// Tensor Lifecycle Counters - sequential IDs for alloc/free/delete tracking
// ═══════════════════════════════════════════════════════════════════════════
std::atomic<int> TensorLifecycleCounters::alloc_counter{0};
std::atomic<int> TensorLifecycleCounters::free_counter{0};
std::atomic<int> TensorLifecycleCounters::gradfn_del_counter{0};
std::atomic<int> TensorLifecycleCounters::move_counter{0};

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

// ═══════════════════════════════════════════════════════════════════════════
// ISSUE #60: DEBUG GRADIENT ATTRIBUTION
// Hooks to capture gradient sources separately for tied-weight debugging

// Global buffers for capturing gradient contributions (set by TrainingState)
float* g_debug_lm_head_only_grad = nullptr;   // Where to copy LM head backward contribution
float* g_debug_embedding_only_grad = nullptr; // Where to copy embedding backward contribution
size_t g_debug_grad_buffer_size = 0;          // Size in elements (vocab_size * d_model)
bool g_debug_capture_enabled = g_autograd_verbose;         // Enable/disable capturing

// Call this after LM head matmul backward to capture its contribution
void debugCaptureLMHeadGrad(float* grad_ptr, size_t size, cudaStream_t stream) {
    if (!g_debug_capture_enabled) return;  // Debug capture not enabled - valid skip
    if (!g_debug_lm_head_only_grad) throw std::runtime_error("debugCaptureLMHeadGrad: g_debug_lm_head_only_grad buffer is NULL - initDebugGradCapture() must be called first");
    if (!grad_ptr) throw std::runtime_error("debugCaptureLMHeadGrad: grad_ptr is NULL - caller MUST provide valid gradient pointer");
    const size_t copy_size = (size < g_debug_grad_buffer_size ? size : g_debug_grad_buffer_size);
    cudaMemcpyAsync(g_debug_lm_head_only_grad, grad_ptr, copy_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
}

// Call this after embedding backward to capture its contribution
void debugCaptureEmbeddingGrad(float* grad_ptr, size_t size, cudaStream_t stream) {
    if (!g_debug_capture_enabled) return;  // Debug capture not enabled - valid skip
    if (!g_debug_embedding_only_grad) throw std::runtime_error("debugCaptureEmbeddingGrad: g_debug_embedding_only_grad buffer is NULL - initDebugGradCapture() must be called first");
    if (!grad_ptr) throw std::runtime_error("debugCaptureEmbeddingGrad: grad_ptr is NULL - caller MUST provide valid gradient pointer");
    const size_t copy_size = (size < g_debug_grad_buffer_size ? size : g_debug_grad_buffer_size);
    cudaMemcpyAsync(g_debug_embedding_only_grad, grad_ptr, copy_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
}

// Global stream for async cleanup - initialized on first use
static cudaStream_t g_cleanup_stream = nullptr;

// Static cuBLAS handle for autograd operations (LayerScaleGradFn etc.)
// Avoids creating/destroying handles per backward call (Issue #ownership-audit)
static cublasHandle_t g_autograd_cublas_handle = nullptr;

void initCleanupStream() {
    if (g_cleanup_stream == nullptr) {
        cudaStreamCreate(&g_cleanup_stream);
    }
}

static void initAutogradCublasHandle() {
    if (g_autograd_cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&g_autograd_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create autograd cuBLAS handle, status=" + std::to_string(static_cast<int>(status)));
        }
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

// DEBUG FLAG: Set to true to sync after EVERY kernel to find which one crashes
// WARNING: This is VERY slow (~10-50x) — only enable for debugging elusive CUDA errors
static bool g_debug_sync_after_every_kernel = false;

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
    
    // DEBUG: Sync after every kernel to catch errors immediately
    if (g_debug_sync_after_every_kernel) {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            fprintf(stderr, "[FATAL] Kernel '%s' (#%d) execution failed: %s\n",
                    kernel_name, static_cast<int>(g_kernel_launch_count), cudaGetErrorString(sync_err));
            fflush(stderr);
            throw std::runtime_error(std::string("CUDA kernel execution failed: ") + kernel_name + " - " + cudaGetErrorString(sync_err));
        }
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

// Destroy all module-static GPU resources (cleanup stream + cuBLAS handle)
// Call during process shutdown after all GPU work is complete
void shutdownAutogradResources() {
    if (g_autograd_cublas_handle) {
        cublasDestroy(g_autograd_cublas_handle);
        g_autograd_cublas_handle = nullptr;
    }
    if (g_cleanup_stream) {
        cudaStreamSynchronize(g_cleanup_stream);
        cudaStreamDestroy(g_cleanup_stream);
        g_cleanup_stream = nullptr;
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
    
    if (shape.is_2d_layout()) {
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
//  GQA split/merge kernels live in TensorConversion.cu
//  (single source of truth for all conversion operations)
//======================================================

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
    if (!qkv_fused.is_2d_layout()) {
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
    
    // Delegate to TensorConversion (single source of truth for GQA split/merge kernels)
    TensorConversion::split_qkv_gqa(
        qkv_fused.ptr, Q.ptr, K.ptr, V.ptr,
        batch, num_heads, num_kv_heads, seq, head_dim, stream);
    
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
    
    // Delegate to TensorConversion (single source of truth for GQA split/merge kernels)
    TensorConversion::merge_qkv_grads_gqa(
        grad_Q.ptr, grad_K.ptr, grad_V.ptr, grad_qkv.ptr,
        batch, num_heads, num_kv_heads, seq, head_dim, stream);
    
    TC_CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  Debug Utilities
//======================================================//

#ifndef NDEBUG

void debug_print_stats(const TensorView& tensor, const char* label, cudaStream_t stream) {
    if (!tensor.is_valid()) {
        throw std::runtime_error(std::string("debug_print_stats: INVALID TENSOR passed for '") + (label ? label : "<null>") + "' - caller has a bug");
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
// Supports 2D grid when block count exceeds device maxGridDim[0] (e.g. 65535 on older GPUs)
__global__ void kernel_zero_autograd(float* data, size_t count) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
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
// Issue #107 FIX: Use cuRAND Philox PRNG for high-quality random initialization.
// The previous LCG-based implementation produced correlated values causing QKV
// projection inflation (actual_row_norm=24.1 vs expected=0.86). Philox is a
// counter-based PRNG with excellent statistical properties used by PyTorch/TensorFlow.
__global__ void kernel_xavier_uniform(float* data, size_t count, float scale, uint64_t seed) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Initialize Philox4_32_10 state - each thread gets unique sequence via idx
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    
    // curand_uniform returns (0, 1], transform to (-1, 1) then scale
    const float rnd = curand_uniform(&state) * 2.0f - 1.0f;
    data[idx] = rnd * scale;
}

// Kernel: Accumulate gradient (dst += src * scale)
__global__ void kernel_accumulate_grad(float* dst, const float* src, size_t count, float scale) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] += src[idx] * scale;
    }
}

// Kernel: Dot product a·b reduced to a single scalar, ACCUMULATED to *dst (not overwritten)
// Uses shared memory block reduction. Grid must be (1) block.
__global__ void kernel_dot_accumulate_scalar(float* dst, const float* a, const float* b, size_t count) {
    __shared__ float sdata[256];
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (size_t i = tid; i < count; i += blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Thread 0 accumulates (not overwrites) to dst
    if (tid == 0) {
        atomicAdd(dst, sdata[0]);
    }
}

constexpr int AUTOGRAD_BLOCK_SIZE = 256;
constexpr int kMaxGridBlocks1D = 65535;  // Device limit on many GPUs

// Returns grid dimensions for count elements; uses 2D grid when blocks > 65535.
inline dim3 gridForCount(size_t count) {
    const int blocks = static_cast<int>((count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE);
    if (blocks <= kMaxGridBlocks1D)
        return dim3(blocks, 1, 1);
    return dim3(kMaxGridBlocks1D, (blocks + kMaxGridBlocks1D - 1) / kMaxGridBlocks1D, 1);
}

}  // anonymous namespace

//======================================================//
//  Tensor Move Constructor/Assignment
//======================================================//

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data)
    , shape(other.shape)
    , owns_data(other.owns_data)
    , grad_(std::move(other.grad_))
    , grad_fn(std::move(other.grad_fn))
    , requires_grad(other.requires_grad)
    , is_leaf(other.is_leaf)
    , retain_grad(other.retain_grad)
    , stream(other.stream)
    , device_id(other.device_id)
    , name(other.name)
    , version(other.version)
{
    other.data = nullptr;
    other.owns_data = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        
        data = other.data;
        shape = other.shape;
        owns_data = other.owns_data;
        grad_ = std::move(other.grad_);
        grad_fn = std::move(other.grad_fn);
        requires_grad = other.requires_grad;
        is_leaf = other.is_leaf;
        retain_grad = other.retain_grad;
        stream = other.stream;
        device_id = other.device_id;
        name = other.name;
        version = other.version;
        other.data = nullptr;
        other.owns_data = false;
    }
    return *this;
}

//======================================================//
//  Tensor Factory Methods
//======================================================//

Tensor Tensor::zeros(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream, const char* name) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error(std::string("Tensor::zeros: ") + (name ? name : "unknown") + " stream is NULL - caller MUST provide valid stream");
    }
    
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::zeros: invalid shape");
    }
    
    Tensor t;
    t.shape = shape;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    t.stream = stream;
    t.name = name;
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    cudaError_t err = cudaMalloc(&t.data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::zeros cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    t.owns_data = true;
    TENSOR_LOG_LIFECYCLE(alloc_counter,
        "[Tensor::alloc] #A%d cudaMalloc data=%p bytes=%zu name=%s\n",
        (void*)t.data, bytes, name ? name : "unnamed");
    
    // Zero-initialize. Use 2D grid when blocks exceeds device maxGridDim (65535 on many GPUs).
    const dim3 grid = gridForCount(count);
    kernel_zero_autograd<<<grid, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(t.data, count);
    
    cudaError_t kernelErr = cudaGetLastError();
    
    if (kernelErr != cudaSuccess) {
        const int blocks = static_cast<int>((count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE);
        throw std::runtime_error(std::string("Tensor::zeros: kernel launch failed for ") +
                                 (name ? name : "unnamed") + ": " + cudaGetErrorString(kernelErr) +
                                 " (blocks=" + std::to_string(blocks) + " count=" + std::to_string(count) + ")");
    }
    
    // NOTE: Don't sync here - let caller control sync point
    
    return t;
}

Tensor Tensor::zeros(std::initializer_list<int> dims, cudaStream_t stream, const char* name) {
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
    return zeros(shape, false, stream, name);
}

Tensor Tensor::empty(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream, const char* name) {
    // RULE 20: Fail loud on invalid stream - default stream causes race conditions
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error(std::string("Tensor::empty: ") + (name ? name : "unknown") + " stream is NULL - caller MUST provide valid stream");
    }
    
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::empty: invalid shape");
    }
    
    Tensor t;
    t.shape = shape;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    t.stream = stream;
    t.name = name;
    
    const size_t bytes = shape.total_elements() * sizeof(float);
    
    cudaError_t err = cudaMalloc(&t.data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::empty cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    t.owns_data = true;
    TENSOR_LOG_LIFECYCLE(alloc_counter,
        "[Tensor::alloc] #A%d cudaMalloc data=%p bytes=%zu name=%s\n",
        (void*)t.data, bytes, name ? name : "unnamed");
    
    // NOTE: Memory is NOT initialized (undefined values)
    return t;
}

Tensor Tensor::from_ptr(float* ptr, TensorContract::TensorShape shape, bool takes_ownership, bool requires_grad, const char* name) {
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
    t.name = name;

    if (takes_ownership) {
        TENSOR_LOG_LIFECYCLE(alloc_counter,
            "[Tensor::alloc] #A%d from_ptr data=%p bytes=%zu name=%s (takes_ownership)\n",
            (void*)ptr, shape.total_elements() * sizeof(float), name ? name : "unnamed");
    }
    
    return t;
}

Tensor Tensor::from_ptr(float* ptr, std::initializer_list<int> dims, cudaStream_t stream, const char* name) {
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
    
    Tensor t = from_ptr(ptr, shape, false, false, name);
    t.stream = stream;
    return t;
}

Tensor Tensor::xavier_uniform(TensorContract::TensorShape shape, bool requires_grad, cudaStream_t stream, const char* name) {
    if (!shape.is_valid()) {
        throw std::invalid_argument("Tensor::xavier_uniform: invalid shape");
    }
    
    Tensor t;
    t.shape = shape;
    t.requires_grad = requires_grad;
    t.is_leaf = true;
    t.stream = stream;
    t.name = name;
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    cudaError_t err = cudaMalloc(&t.data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::xavier_uniform cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    t.owns_data = true;
    TENSOR_LOG_LIFECYCLE(alloc_counter,
        "[Tensor::alloc] #A%d cudaMalloc data=%p bytes=%zu name=%s (xavier)\n",
        (void*)t.data, bytes, name ? name : "unnamed");
    
    // Xavier uniform: U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]
    // For 2D [rows, cols]: fan_in=cols, fan_out=rows
    // For 4D [B,H,S,D]: treat as [B*H*S, D] => fan_in=D, fan_out=B*H*S
    auto seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    xavier_uniform_(t, seed, stream);
    return t;
}

void Tensor::xavier_uniform_(Tensor& t, uint64_t seed, cudaStream_t stream) {
    if (!t.data) {
        throw std::invalid_argument("Tensor::xavier_uniform_: tensor has no data");
    }
    if (!t.shape.is_valid()) {
        throw std::invalid_argument("Tensor::xavier_uniform_: tensor has invalid shape");
    }
    
    const size_t count = t.shape.total_elements();
    
    // Xavier uniform calculation
    float fan_in = 1.0f, fan_out = 1.0f;
    if (t.shape.is_2d_layout()) {
        const auto& s = t.shape.as_2d();
        fan_in = static_cast<float>(s.cols);
        fan_out = static_cast<float>(s.rows);
    } else {
        throw std::invalid_argument("Tensor::xavier_uniform_: only 2D weight matrices are supported. 4D tensors are activations, not weights.");
    }
    
    float scale = std::sqrt(6.0f / (fan_in + fan_out));
    
    kernel_xavier_uniform<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(t.data, count, scale, seed);
    
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

    // RULE 20: Stream must be valid. Tensor::zeros() rejects stream==0, so we
    // avoid it here — but still validate. from_ptr() callers that set
    // requires_grad=true MUST also set .stream before ensure_grad() runs.
    if (stream == nullptr) {
        throw std::runtime_error(std::string("ensure_grad: stream is NULL on tensor '") +
                                 (name ? name : "unnamed") +
                                 "' - caller MUST set .stream before ensure_grad()");
    }
    
    const size_t count = shape.total_elements();
    const size_t bytes = count * sizeof(float);
    
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("ensure_grad cudaMalloc failed for ") +
                                 (name ? name : "unnamed") + ": " + cudaGetErrorString(err));
    }
    
    cudaMemsetAsync(ptr, 0, bytes, stream);
    
    TENSOR_LOG_LIFECYCLE(alloc_counter,
        "[Tensor::alloc] #A%d cudaMalloc data=%p bytes=%zu name=%s\n",
        (void*)ptr, bytes, name ? name : "unnamed");
    
    // Wrap in a Tensor shared_ptr for proper RAII ownership
    auto grad_tensor = std::make_shared<Tensor>();
    grad_tensor->data = ptr;
    grad_tensor->shape = shape;
    grad_tensor->owns_data = true;
    grad_tensor->requires_grad = false;
    grad_tensor->is_leaf = true;
    grad_tensor->stream = stream;
    grad_tensor->device_id = device_id;
    grad_tensor->name = name;
    grad_ = grad_tensor;
} 

void Tensor::zero_grad(cudaStream_t exec_stream) {
    if (!grad_ || !grad_->data) {
        return;  // Nothing to zero
    }
    
    cudaStream_t s = exec_stream ? exec_stream : stream;
    const size_t count = shape.total_elements();
    kernel_zero_autograd<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, s>>>(grad_->data, count);
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
    kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, s>>>(grad_->data, incoming_grad, count, scale);
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

void Tensor::backward(const Tensor* grad_output, float scale) {
    if (!requires_grad) {
        throw std::runtime_error("Tensor::backward called on tensor that doesn't require grad");
    }
    
    // Initialize gradient if this is the starting point (loss tensor)
    if (grad_output == nullptr) {
        // Default: scalar initial gradient
        ensure_grad();
        const size_t count = shape.total_elements();
        
        // For scalar loss, set grad to the provided scale (usually 1.0 or 1.0/accumulation_steps)
        if (count == 1) {
            cudaMemcpyAsync(grad_data(), &scale, sizeof(float), cudaMemcpyHostToDevice, stream);
        } else {
            // Fill with scale (implicit broadcast for reduction operations)
            std::vector<float> scales(count, scale);
            cudaMemcpyAsync(grad_data(), scales.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    } else {
        // Accumulate provided gradient scaled by the provided scale factor
        accumulate_grad(grad_output->data, grad_output->numel(), scale, stream);
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
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
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
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
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
    int vocab_size,             // RULE 20: Bounds check parameter
    float embedding_scale       // Scale factor (1.0 for production — Issue #140 removed sqrt(d_model))
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const int token_id = token_ids[token_idx];
    // RULE 20: Crash loud on OOB token ID — __trap() works in Release (assert compiles out)
    if (token_id < 0 || token_id >= vocab_size) {
        printf("FATAL: OOB token_id=%d (vocab_size=%d) at token_idx=%d in kernel_embedding_forward\n",
               token_id, vocab_size, token_idx);
        __trap();
    }
    const float* weight_row = weight + static_cast<size_t>(token_id) * d_model;
    float* output_row = output + static_cast<size_t>(token_idx) * d_model;
    
    // Gather with scaling: output[token_idx] = weight[token_id] * scale
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
    int vocab_size,             // RULE 20: Bounds check parameter
    float embedding_scale       // Scale factor from forward (for chain rule)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;
    
    const int token_id = token_ids[token_idx];
    // RULE 20: Crash loud on OOB token ID — __trap() works in Release (assert compiles out)
    if (token_id < 0 || token_id >= vocab_size) {
        printf("FATAL: OOB token_id=%d (vocab_size=%d) at token_idx=%d in kernel_embedding_backward\n",
               token_id, vocab_size, token_idx);
        __trap();
    }
    const float* token_grad = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* weight_grad = grad_weight + static_cast<size_t>(token_id) * d_model;
    
    // Scatter-add: weight_grad[token_id] += grad_output[token_idx] * scale
    // Chain rule: if forward was y = w * scale, then grad_w = grad_y * scale
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        atomicAdd(&weight_grad[i], token_grad[i] * embedding_scale);
    }
}



//========================================================================
// Log-Softmax Forward: log_softmax(x)[i] = x[i] - logsumexp(x)
// Stays in log space — no exp→log roundtrip (numerically superior to softmax→log)
// One block per row. dim = vocab_size (50K+).
//========================================================================
__global__ void kernel_log_softmax_forward(
    const float* __restrict__ input,    // [tokens, dim]
    float* __restrict__ output,         // [tokens, dim] - log-probabilities
    int tokens,
    int dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* row = input  + static_cast<size_t>(token_idx) * dim;
    float* out_row   = output + static_cast<size_t>(token_idx) * dim;

    // ── Step 1: max(x) via deterministic warp→shared reduction ──
    constexpr int kMaxWarps = 8;  // 256 threads / 32
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_val;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = s_warp[0];
        for (int w = 1; w < num_warps && w < kMaxWarps; w++) m = fmaxf(m, s_warp[w]);
        s_val = m;
    }
    __syncthreads();
    const float max_val = s_val;

    // ── Step 2: sum_exp = Σ exp(x - max) ──
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += expf(row[i] - max_val);

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_val = s;
    }
    __syncthreads();
    const float sum_exp = s_val;

    // Rule 20: sum_exp >= 1.0 is a mathematical invariant (exp(max-max)=1).
    // Diagnostic: Print to stderr if invariant violated; kernel produces NaN/Inf that
    // gets caught by downstream validation (GradFn diagnostic sampling in AutogradLoss.cu)
    if (threadIdx.x == 0 && (sum_exp < 1e-6f || isnan(sum_exp) || isinf(sum_exp))) {
        printf("[LOG_SOFTMAX_EQUATION] FATAL: sum_exp=%e at token %d — logits corrupted! "
               "max_val=%e, dim=%d\n", sum_exp, token_idx, max_val, dim);
    }

    const float log_sum_exp = logf(sum_exp) + max_val;  // logsumexp(x)

    // ── Step 3: output[i] = x[i] - logsumexp(x)  (stays in log space!) ──
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] = row[i] - log_sum_exp;
}

//========================================================================
// Log-Softmax Backward
// Given saved log_p = log_softmax(x) and upstream grad_output (dL/d(log_p)):
//
//   grad_input[i] = grad_output[i] - exp(log_p[i]) * Σ_j grad_output[j]
//
// Derivation: log_softmax = x - logsumexp(x)
//   ∂log_softmax_i/∂x_j = δ_{ij} - softmax_j
//   ⇒ grad_x[j] = Σ_i grad_y[i] * (δ_{ij} - p_j)
//                = grad_y[j] - p_j * Σ_i grad_y[i]
//
// Key: uses saved log_p, computes p = exp(log_p) on-the-fly.
// No separate softmax buffer needed.
//========================================================================
__global__ void kernel_log_softmax_backward(
    const float* __restrict__ grad_output,   // [tokens, dim] dL/d(log_p)
    const float* __restrict__ log_softmax,   // [tokens, dim] saved log-probabilities
    float* __restrict__ grad_input,          // [tokens, dim] dL/d(logits)
    int tokens,
    int dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* dy    = grad_output + static_cast<size_t>(token_idx) * dim;
    const float* log_p = log_softmax + static_cast<size_t>(token_idx) * dim;
    float* dx          = grad_input  + static_cast<size_t>(token_idx) * dim;

    // ── Step 1: Σ_j grad_output[j]  (sum of upstream gradient) ──
    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_sum;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += dy[i];

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_sum = s;
    }
    __syncthreads();

    const float sum_dy = s_sum;

    // ── Step 2: grad_x[i] = grad_y[i] - exp(log_p[i]) * sum_dy ──
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dx[i] = dy[i] - expf(log_p[i]) * sum_dy;
}

// Dropout forward: y = x * mask / (1 - p)
// mask is 0 where dropped, 1 where kept; scale = 1/(1-p) for inverted dropout
__global__ void kernel_dropout_forward(
    const float* __restrict__ input,
    const uint8_t* __restrict__ mask,
    float* __restrict__ output,
    float scale,                // 1.0 / (1.0 - dropout_prob)
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Inverted dropout: scale up kept values so expected value unchanged
        output[idx] = input[idx] * (mask[idx] ? scale : 0.0f);
    }
}

// Generate random dropout mask using Philox PRNG
// Each element is 1 (keep) with probability (1-p), 0 (drop) with probability p
__global__ void kernel_generate_dropout_mask(
    uint8_t* __restrict__ mask,
    size_t count,
    float dropout_prob,         // Probability of dropping (e.g., 0.1)
    uint64_t seed
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Initialize Philox RNG per-element with unique sequence
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random value in (0, 1]
        float rnd = curand_uniform(&state);
        
        // Keep if random > dropout_prob (so dropout_prob fraction gets dropped)
        mask[idx] = (rnd > dropout_prob) ? 1 : 0;
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
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
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

// ScaleGradFn DELETED — dead code from reverted Issue #98 (Rule 20)

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
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    size_t element_count = 0;
    
    // Scale param info (shape [1])
    float scale_value = 1.0f;     // Cached scale value for backward
    float* scale_grad = nullptr;  // Points to scale_param's grad
    
    bool applied = false;
    
    LayerScaleGradFn() { op_name = "layer_scale"; }
    
    void capture_inputs(Tensor& input, Tensor& scale_param, float cached_scale_value, cudaStream_t stream) {
        input_shape = input.shape;
        element_count = input.numel();
        
        // Use cached scale value (already read from GPU in layer_scale forward)
        scale_value = cached_scale_value;
        
        // Copy shared_ptr to input's grad_fn
        input_grad_fn = input.grad_fn;
        
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
        
        // 1. grad_input = grad_output * scale_value  (broadcast)
        if (input_grad && grad_output.data) {
            kernel_accumulate_grad<<<gridForCount(n), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                input_grad, grad_output.data, n, scale_value);
        }
        
        // 2. grad_scale = sum(grad_output * input_data)  (dot product → scalar)
        //    ACCUMULATE (not overwrite) to scale_grad for gradient accumulation windows
        if (scale_grad && grad_output.data && input_data) {
            // Single-block reduction kernel: atomicAdd(scale_grad, dot(grad_output, input_data))
            kernel_dot_accumulate_scalar<<<1, 256, 0, stream>>>(
                scale_grad, grad_output.data, input_data, n);
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
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    size_t element_count = 0;
    int row_dim = 0;
    int num_rows = 0;
    bool input_requires_grad = false;
    
    // ISSUE #56 FIX: Owned gradient buffer for non-leaf tensors
    std::shared_ptr<float> owned_input_grad;
    
    CenterRowsGradFn() { op_name = "center_rows"; }
    
    __host__ void capture_input(Tensor& input, int dim, int rows, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        if (!input.requires_grad) return;
        
        input_shape = input.shape;
        element_count = input.numel();
        row_dim = dim;
        num_rows = rows;
        
        // Copy shared_ptr to input's grad_fn
        input_grad_fn = input.grad_fn;
        
        // Setup gradient buffer (Issue #54 pattern)
        input.ensure_grad();
        
        // CRITICAL FIX (Issue #136): NEVER reuse externally-owned leaf buffers!
        // set_grad_from_buffer() marks the gradient tensor as is_leaf=true,
        // but it's wrapping an externally-owned buffer (grad_logits_tensor.data).
        // If we reuse this buffer, CenterRowsGradFn OVERWRITES it with centered gradients,
        // corrupting the original CE gradients that LogSoftmaxGradFn wrote.
        // Always allocate our own buffer so we don't destroy upstream data.
        float* buf = nullptr;
        cudaMalloc(&buf, element_count * sizeof(float));
        cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
        owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
        input_grad = owned_input_grad.get();
        AG_TRACE("[CenterRowsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p\n", element_count, (void*)input_grad);
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;  // ISSUE #49
        if (!input_requires_grad) return;  // No grad needed for this input
        if (!input_grad) throw std::runtime_error("CenterRowsGradFn::apply: input_grad is NULL - capture_input() must be called first");
        if (!grad_output.data) throw std::runtime_error("CenterRowsGradFn::apply: grad_output.data is NULL - backward called with null gradient");
        applied = true;
        
        // BACKWARD: grad_x = grad_y - mean_d(grad_y)  (reuse centering kernel!)
        // We can use kernel_center_rows to center grad_output directly to input_grad
        {
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
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_input_grad;
    
    CenterColumnsGradFn() { op_name = "center_columns"; }
    
    __host__ void capture_input(Tensor& input, int cols, int rows, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        if (!input.requires_grad) return;
        
        input_shape = input.shape;
        element_count = input.numel();
        num_cols = cols;
        num_rows = rows;
        
        // Copy shared_ptr to input's grad_fn
        input_grad_fn = input.grad_fn;
        
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
        if (applied) return;  // ISSUE #49
        if (!input_requires_grad) return;  // No grad needed for this input
        if (!input_grad) throw std::runtime_error("CenterColumnsGradFn::apply: input_grad is NULL - capture_input() must be called first");
        if (!grad_output.data) throw std::runtime_error("CenterColumnsGradFn::apply: grad_output.data is NULL - backward called with null gradient");
        applied = true;
        
        // BACKWARD: grad_x = grad_y - mean_t(grad_y[:,d])  (same centering operation!)
        // Since column-wise centering is linear, backward is same as forward
        {
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

// ═══════════════════════════════════════════════════════════════════════════════
// PC1 PROJECTION KERNELS — Issue #149
// Forward:  h̃[t] = h[t] - (h[t]·g)*g   (g = PC1 via power iteration, stop-grad)
// Backward: grad_h = (I - gg^T) * grad_h̃  (same projection, g is constant)
// ═══════════════════════════════════════════════════════════════════════════════

// Init PC1 guess from column mean: g[d] = mean_t H[t,d] / ||·||
// Launch: <<<1, 256, 0, stream>>>
static __global__ void kernel_pc1_col_mean(
    const float* __restrict__ H, float* __restrict__ g, int T, int D)
{
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float s = 0.f;
        for (int t = 0; t < T; t++) s += H[(size_t)t * D + d];
        g[d] = s / (float)T;
    }
}

// Normalize g in-place: g = g / rms(g)   where rms = sqrt(sum(g²)/D)
// Launch: <<<1, 256, 0, stream>>>
static __global__ void kernel_pc1_normalize(float* __restrict__ g, int D)
{
    __shared__ float sdata[256];
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += g[d] * g[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    // RMS normalization: inv = 1/rms(g) = 1/sqrt(sum_sq/D)
    float inv = 1.f / sqrtf(sdata[0] / (float)D + 1e-12f);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        g[d] *= inv;
}

// v[t] = H[t,:] · g  (matrix-vector product H @ g)
// Launch: <<<ceil(T/256), 256, 0, stream>>>
static __global__ void kernel_pc1_gemv_Hg(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ v, int T, int D)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float dot = 0.f;
    for (int d = 0; d < D; d++) dot += H[(size_t)t * D + d] * g[d];
    v[t] = dot;
}

// g_out[d] = H[:,d] · v  (matrix-vector product H^T @ v)
// Launch: <<<ceil(D/256), 256, 0, stream>>>
static __global__ void kernel_pc1_gemv_HtV(
    const float* __restrict__ H, const float* __restrict__ v,
    float* __restrict__ g_out, int T, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float dot = 0.f;
    for (int t = 0; t < T; t++) dot += H[(size_t)t * D + d] * v[t];
    g_out[d] = dot;
}

// H_out[t,d] = H[t,d] - (H[t,:]·g / D) * g[d]  — project each row
// With RMS-normalized g (rms(g)=1 → g·g=D), proper projection is (h·g)/(g·g) * g = (h·g)/D * g
// Used in both forward pass and backward (same linear projection since (I-gg^T/D) is symmetric)
// Launch: <<<T, 256, 0, stream>>>  (one block per row for shared-mem dot product)
static __global__ void kernel_pc1_project(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ H_out, int T, int D)
{
    int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float sdata[256];
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += H[(size_t)t * D + d] * g[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    // Divide by D because g is RMS-normalized (g·g = D, not 1)
    float coeff = sdata[0] / (float)D;
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        H_out[(size_t)t * D + d] = H[(size_t)t * D + d] - coeff * g[d];
}

/**
 * ProjectOutPC1GradFn — Issue #149
 * Backward for project_out_pc1
 * Forward:  h̃[t] = h[t] - (h[t]·g/D)*g   where g = PC1(H) via power iteration (rms-normalized), stop-grad
 * Backward: grad_h = (I - gg^T/D) * grad_h̃   (same linear projection, g constant)
 */
struct ProjectOutPC1GradFn : public GradFn {
    bool input_requires_grad = false;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int num_rows = 0;   // T (total tokens)
    int num_cols = 0;   // D (d_model)

    float* input_grad = nullptr;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_input_grad;

    // Saved PC1 direction (device memory, stop-gradient)
    float* g_saved = nullptr;
    std::shared_ptr<float> owned_g;

    ProjectOutPC1GradFn() { op_name = "project_out_pc1"; }

    __host__ void capture_input(Tensor& input, int rows, int cols,
                                const float* g_ptr, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        input_shape = input.shape;
        element_count = input.numel();
        num_rows = rows;
        num_cols = cols;

        // Save g direction (device copy, stop-gradient)
        float* gp = nullptr;
        cudaMalloc(&gp, cols * sizeof(float));
        cudaMemcpyAsync(gp, g_ptr, cols * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        owned_g.reset(gp, [](float* p) { queueForDeferredCleanup(p); });
        g_saved = owned_g.get();

        if (!input.requires_grad) return;

        input_grad_fn = input.grad_fn;

        if (input.is_leaf) {
            // Leaf tensor: use its persistent grad buffer
            input.ensure_grad();
            input_grad = input.grad_data();
            AG_TRACE("[ProjectOutPC1GradFn] Using persistent input_grad buffer (leaf): %p\n", (void*)input_grad);
        } else {
            // Non-leaf: DEFER allocation to apply() — saves ~24MB GPU during forward
            // (Issue #149 OOM fix: forward must leave room for logits allocation)
            input_grad = nullptr;
            AG_TRACE("[ProjectOutPC1GradFn] Non-leaf input: deferring grad buffer allocation to backward\n");
        }
    }

    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        if (!input_requires_grad) return;
        if (!g_saved) throw std::runtime_error("ProjectOutPC1GradFn::apply: g_saved is NULL - g direction was freed before backward");
        applied = true;

        // Allocate grad buffer on-demand for non-leaf inputs (deferred from capture_input)
        if (!input_grad) {
            float* buf = nullptr;
            cudaError_t err = cudaMalloc(&buf, element_count * sizeof(float));
            if (err != cudaSuccess || !buf) {
                throw std::runtime_error("ProjectOutPC1GradFn::apply: cudaMalloc failed for input_grad (" +
                                         std::to_string(element_count * sizeof(float)) + " bytes): " +
                                         std::string(cudaGetErrorString(err)));
            }
            cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
            owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
            AG_TRACE("[ProjectOutPC1GradFn] Allocated deferred input_grad buffer: %zu floats at %p\n", element_count, (void*)input_grad);
        }

        // BACKWARD: grad_h = (I - gg^T/D) * grad_h̃  — same projection as forward (g is RMS-normalized)
        kernel_pc1_project<<<num_rows, 256, 0, stream>>>(
            grad_output.data, g_saved, input_grad, num_rows, num_cols);

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
        owned_g.reset();
        g_saved = nullptr;
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
    std::shared_ptr<GradFn> a_grad_fn;
    std::shared_ptr<GradFn> b_grad_fn;
    size_t element_count = 0;
    
    AddGradFn() { op_name = "add"; }
    
    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_shape = a.shape;
        b_shape = b.shape;
        
        // Copy shared_ptrs to captured grad_fns
        a_grad_fn = a.grad_fn;
        b_grad_fn = b.grad_fn;
        
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
        
        // Copy shared_ptr to captured grad_fn
        a_grad_fn = a.grad_fn;
        
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
        
        // DIAGNOSTIC: Log incoming gradient to Add backward (GUARDED - expensive!)
#if TENSOR_VERBOSE_DEBUG
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
#endif
        
        if (!grad_output.data) {
            throw std::runtime_error("AddGradFn::apply: grad_output.data is NULL - backward called with null gradient");
        }
        
        const size_t count = grad_output.numel();
        
        // Accumulate gradients to both inputs using stored grad pointers
        if (a_requires_grad && grad_a) {
            kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_a, grad_output.data, count, 1.0f);
        }
        if (b_requires_grad && grad_b) {
            kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_b, grad_output.data, count, 1.0f);
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn pointers
        if (a_requires_grad && a_grad_fn && a_grad_fn->op_name) {
            // ISSUE #58 FIX: Pass grad_output.data (incoming gradient), NOT grad_a (local accumulator)
            Tensor view;
            view.data = grad_output.data; view.shape = a_shape;
            view.owns_data = false; view.stream = stream;
            a_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here!
        }
        
        if (b_requires_grad && b_grad_fn && b_grad_fn != a_grad_fn && b_grad_fn->op_name) {
            // ISSUE #58 FIX: Pass grad_output.data (incoming gradient), NOT grad_b (local accumulator)
            Tensor view;
            view.data = grad_output.data; view.shape = b_shape;
            view.owns_data = false; view.stream = stream;
            b_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here!
        }
    }

    
    void release_saved() override {
        GradFn::release_saved();
        
        grad_a = nullptr;
        grad_b = nullptr;
        
        a_grad_fn.reset();
        b_grad_fn.reset();
    }
};

//======================================================//
//  Bias Add/Backward Kernels (owned by autograd layer)
//  Moved from Feed_Forward_GPU.cu - these are generic tensor ops
//  used by BiasAddGradFn, not FFN-specific.
//======================================================//

__global__ void biasAddKernel(float* __restrict__ tensor,
                              const float* __restrict__ bias,
                              int total_elements,
                              int features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const int feature_idx = idx % features;
        tensor[idx] += bias[feature_idx];
    }
}

__global__ void biasBackwardKernel(const float* __restrict__ grad_output,
                                   float* __restrict__ grad_bias,
                                   int total_tokens,
                                   int features) {
    extern __shared__ float sdata[];

    const int feature_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (feature_idx >= features) return;

    float local_sum = 0.0f;
    for (int t = tid; t < total_tokens; t += blockDim.x) {
        local_sum += grad_output[t * features + feature_idx];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[feature_idx] += sdata[0];
    }
}

static void launchBiasAdd(float* tensor, const float* bias,
                          int total_tokens, int features,
                          cudaStream_t stream) {
    if (!tensor) throw std::runtime_error("launchBiasAdd: tensor is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (!bias) throw std::runtime_error("launchBiasAdd: bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (total_tokens <= 0 || features <= 0) throw std::runtime_error("launchBiasAdd: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");

    const int total_elements = total_tokens * features;
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = (total_elements + kBlockSize - 1) / kBlockSize;
    biasAddKernel<<<grid, kBlockSize, 0, stream>>>(tensor, bias, total_elements, features);
}

static void launchBiasBackward(const float* grad_output, float* grad_bias,
                               int total_tokens, int features,
                               cudaStream_t stream) {
    if (!grad_output) throw std::runtime_error("launchBiasBackward: grad_output is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (!grad_bias) throw std::runtime_error("launchBiasBackward: grad_bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (total_tokens <= 0 || features <= 0) throw std::runtime_error("launchBiasBackward: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");

    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int shared_bytes = kBlockSize * sizeof(float);
    biasBackwardKernel<<<features, kBlockSize, shared_bytes, stream>>>(
        grad_output, grad_bias, total_tokens, features);
}

/**
 * BiasAddGradFn - Backward for broadcast add: output = input + bias
 * Forward: output[i,j] = input[i,j] + bias[j]  (bias broadcasted)
 * Backward: grad_input = grad_output (pass-through)
 *           grad_bias[j] = sum_i(grad_output[i,j]) (reduction over tokens)
 *
 * ISSUE #97: Encoder biases (b_qkv, b_o, b1, b2) were frozen because
 * bias add was a raw CUDA kernel with NO autograd tracking.
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
    std::shared_ptr<GradFn> input_grad_fn;
    size_t total_tokens = 0;
    size_t features = 0;
    
    BiasAddGradFn() { op_name = "bias_add"; }
    
    void capture_inputs(Tensor& input, Tensor& bias, int num_tokens, int num_features, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        bias_requires_grad = bias.requires_grad;
        input_shape = input.shape;
        bias_shape = bias.shape;
        total_tokens = static_cast<size_t>(num_tokens);
        features = static_cast<size_t>(num_features);
        
        // Copy shared_ptr to captured grad_fn
        input_grad_fn = input.grad_fn;
        
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
        
        // DIAGNOSTIC: Log incoming gradient and capture state (GUARDED - expensive!)
#if TENSOR_VERBOSE_DEBUG
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
#endif
        
        if (!grad_output.data) {
            throw std::runtime_error("BiasAddGradFn::apply: grad_output.data is NULL - backward called with null gradient");
        }
        
        const size_t count = grad_output.numel();
        
        // Backward for input: grad_input = grad_output (pass-through, no shape change)
        if (input_requires_grad && grad_input) {
            kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_input, grad_output.data, count, 1.0f);
        }
        
        // Backward for bias: grad_bias[j] = sum_i(grad_output[i,j])
        // Use the existing bias backward kernel from Feed_Forward_GPU.cu
        if (bias_requires_grad && grad_bias) {
            launchBiasBackward(grad_output.data, grad_bias, 
                               static_cast<int>(total_tokens), static_cast<int>(features), stream);
#if TENSOR_VERBOSE_DEBUG
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
#endif
        }
#if TENSOR_VERBOSE_DEBUG
        else {
            fprintf(stderr, "[BIAS-ADD-BWD-SKIP] call=%d | bias_requires_grad=%d grad_bias=%p - NOT computing bias gradient!\n",
                    bias_call_idx, (int)bias_requires_grad, (void*)grad_bias);
        }
#endif
        
        // CONTINUE AUTOGRAD CHAIN for input
        if (input_requires_grad && input_grad_fn && input_grad_fn->op_name) {
            Tensor view;
            view.data = grad_output.data;  // ISSUE #58: Pass incoming gradient
            view.shape = input_shape;
            view.owns_data = false;
            view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        grad_input = nullptr;
        grad_bias = nullptr;
        input_grad_fn.reset();
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
    std::shared_ptr<GradFn> input_grad_fn;
    // ISSUE #51 FIX: Own a copy of cached data instead of non-owning pointer
    std::shared_ptr<float> owned_cache;
    const float* cached_input = nullptr;  // Points to owned_cache.get()
    size_t cached_size = 0;
    
    GeluGradFn() { op_name = "gelu"; }
    
    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        
        // Copy shared_ptr to captured grad_fn
        input_grad_fn = x.grad_fn;
        
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
        
        // DIAGNOSTIC: Log incoming gradient to GELU backward (GUARDED - expensive!)
#if TENSOR_VERBOSE_DEBUG
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
#endif
        
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
        
        // TAPE-BASED: Write directly to stored grad buffer
        kernel_gelu_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_input, input_grad, count);
        trackKernelLaunch("kernel_gelu_backward", stream);

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        cached_input = nullptr;
        cached_size = 0;
        input_grad = nullptr;
        input_grad_fn.reset();
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
    bool owns_input_grad = false;  // ISSUE #126: Track if we own input_grad buffer
    float* gamma_grad_ptr = nullptr;  // For gamma gradient
    float* gamma_data = nullptr;      // Gamma weights data (for forward values)
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    // ISSUE #51 FIX: Own a copy of cached data instead of non-owning pointer
    std::shared_ptr<float> owned_cache;
    const float* cached_input = nullptr;  // Points to owned_cache.get()
    size_t cached_size = 0;
    int d_model = 0;
    float eps = 1e-5f;
    
    RMSNormGradFn() { op_name = "rms_norm"; }
    
    ~RMSNormGradFn() {
        // ISSUE #126: Free owned input_grad buffer
        if (owns_input_grad && input_grad) {
            queueForDeferredCleanup(input_grad);
            input_grad = nullptr;
        }
        // shared_ptr members destruct automatically after this
    }
    
    void capture_inputs(Tensor& x, Tensor& gamma_tensor, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        gamma_requires_grad = gamma_tensor.requires_grad;
        input_shape = x.shape;
        
#if TENSOR_VERBOSE_DEBUG
        fprintf(stderr, "[RMSNormGradFn::capture_inputs] this=%p x.requires_grad=%d gamma.requires_grad=%d\n",
                (void*)this, x.requires_grad ? 1 : 0, gamma_tensor.requires_grad ? 1 : 0);
        fflush(stderr);
#endif
        
        // Copy shared_ptr to captured grad_fn
        input_grad_fn = x.grad_fn;
        
        gamma_data = gamma_tensor.data;
        
        // ISSUE #126 FIX: Allocate OWNED gradient buffer instead of storing pointer to x.grad_data()
        // The input tensor x may be a temporary that gets destroyed before backward() runs.
        // x.ensure_grad() allocates a gradient buffer owned by x, which gets freed when x destructs.
        // We need our own buffer that persists until backward is complete.
        if (input_requires_grad) {
            const size_t grad_size = x.shape.total_elements();
            cudaError_t err = cudaMalloc(&input_grad, grad_size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("RMSNormGradFn: Failed to allocate input_grad buffer");
            }
            // Zero-initialize the gradient buffer
            cudaMemsetAsync(input_grad, 0, grad_size * sizeof(float), stream);
            owns_input_grad = true;
#if TENSOR_VERBOSE_DEBUG
            fprintf(stderr, "[RMSNormGradFn::capture_inputs] this=%p → Allocated input_grad: %p (%zu floats)\n",
                    (void*)this, (void*)input_grad, grad_size);
#endif
        }
#if TENSOR_VERBOSE_DEBUG
        else {
            fprintf(stderr, "[RMSNormGradFn::capture_inputs] SKIPPED input_grad allocation (input_requires_grad=false)\n");
        }
        fflush(stderr);
#endif
        
        if (gamma_requires_grad) {
            gamma_tensor.ensure_grad();
            gamma_grad_ptr = gamma_tensor.grad_data();  // Gamma tensor is persistent (from TrainingState)
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

#if TENSOR_VERBOSE_DEBUG
        fprintf(stderr, "[RMSNormGradFn::apply] ENTRY - this=%p grad_output.data=%p stream=%p\n",
                (void*)this, (void*)grad_output.data, (void*)stream);
        fflush(stderr);
        
        // Check for pre-existing CUDA errors BEFORE any sync
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            fprintf(stderr, "[RMSNormGradFn::apply] PRE-EXISTING CUDA ERROR: %d (%s)\n",
                    (int)pre_err, cudaGetErrorString(pre_err));
            fflush(stderr);
        }
        
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
#endif
        
        if (!cached_input) {
            throw std::runtime_error("RMSNormGradFn::apply: cached_input is NULL - set_cache() must be called first");
        }
        if (d_model <= 0) {
            throw std::runtime_error("RMSNormGradFn::apply: d_model is " + std::to_string(d_model) + " - must be > 0");
        }
        
        const int tokens = static_cast<int>(cached_size / d_model);
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
        
#if TENSOR_VERBOSE_DEBUG
        // Add detailed logging before kernel launch
        fprintf(stderr, "[RMS-BWD-LAUNCH] tokens=%d d_model=%d shared_mem=%d\n",
                tokens, d_model, shared_mem);
        fprintf(stderr, "[RMS-BWD-LAUNCH] ptrs: grad_output=%p cached_input=%p gamma=%p\n",
                (void*)grad_output.data, (void*)cached_input, (void*)gamma_data);
        fprintf(stderr, "[RMS-BWD-LAUNCH] ptrs: input_grad=%p gamma_grad=%p\n",
                (void*)input_grad, (void*)gamma_grad_ptr);
        fprintf(stderr, "[RMS-BWD-LAUNCH] cached_size=%zu expected=%zu (tokens*d_model)\n",
                cached_size, static_cast<size_t>(tokens) * d_model);
        fflush(stderr);
        
        // VALIDATE: Try to read from each pointer to detect stale/freed memory
        {
            float test_val = 0.0f;
            cudaError_t err;
            
            err = cudaMemcpy(&test_val, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[RMS-BWD-VALIDATE] grad_output read: err=%d (%s) val=%.6f\n",
                    (int)err, cudaGetErrorString(err), test_val);
            
            err = cudaMemcpy(&test_val, cached_input, sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[RMS-BWD-VALIDATE] cached_input read: err=%d (%s) val=%.6f\n",
                    (int)err, cudaGetErrorString(err), test_val);
            
            err = cudaMemcpy(&test_val, gamma_data, sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[RMS-BWD-VALIDATE] gamma_data read: err=%d (%s) val=%.6f\n",
                    (int)err, cudaGetErrorString(err), test_val);
            
            err = cudaMemcpy(&test_val, input_grad, sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[RMS-BWD-VALIDATE] input_grad read: err=%d (%s) val=%.6f\n",
                    (int)err, cudaGetErrorString(err), test_val);
            
            if (gamma_grad_ptr) {
                err = cudaMemcpy(&test_val, gamma_grad_ptr, sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[RMS-BWD-VALIDATE] gamma_grad read: err=%d (%s) val=%.6f\n",
                        (int)err, cudaGetErrorString(err), test_val);
            }
            fflush(stderr);
        }
#endif
        
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
                view.owns_data = false; view.stream = stream;
                input_grad_fn->apply(view, stream);
                // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
            }
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        cached_input = nullptr;
        cached_size = 0;
        // ISSUE #126: Free owned input_grad buffer (if not already freed by destructor)
        if (owns_input_grad && input_grad) {
            queueForDeferredCleanup(input_grad);
            owns_input_grad = false;
        }
        input_grad = nullptr;
        gamma_grad_ptr = nullptr;
        input_grad_fn.reset();
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
    std::shared_ptr<GradFn> weight_grad_fn;
    int* token_ids = nullptr;
    bool owns_token_ids = false;
    int num_tokens = 0;
    int d_model = 0;
    int vocab_size = 0;            // RULE 20: Stored for OOB bounds checking in backward kernel
    float embedding_scale = 1.0f;  // Issue #140: No scaling (1.0f) — AIAYN sqrt(d_model) removed
    
    EmbeddingGradFn() { op_name = "embedding"; }
    
    ~EmbeddingGradFn() override {
        if (owns_token_ids && token_ids) cudaFree(token_ids);
    }
    
    void capture_weight(Tensor& w) {
        weight_requires_grad = w.requires_grad;
        weight_shape = w.shape;
        
        // Copy shared_ptr to captured grad_fn
        weight_grad_fn = w.grad_fn;
        
        if (weight_requires_grad) {
            w.ensure_grad();
            weight_grad = w.grad_data();  // ISSUE #59: Use accessor
        }
    }
    
    void save(const int* ids, int tokens, int d, bool copy_ids, cudaStream_t stream) {
        num_tokens = tokens;
        d_model = d;
        
        AG_TRACE("[EmbeddingGradFn::save] ids=%p tokens=%d d=%d copy_ids=%s\n",
                (void*)ids, tokens, d, copy_ids ? "true" : "false");
        
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
        using namespace GRIM::Logging;
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("embedding", this);
        
        AG_TRACE("[EmbeddingGradFn::apply] ENTER - tokens=%d d=%d\n",
                num_tokens, d_model);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            AG_TRACE("[EmbeddingGradFn::apply] SKIP - already applied\n");
            return;
        }
        applied = true;
        
        if (!weight_requires_grad) {
            AG_TRACE("[EmbeddingGradFn::apply] SKIP - weight does not require grad\n");
            return;
        }
        if (!token_ids) {
            throw std::runtime_error("EmbeddingGradFn::apply: token_ids is NULL - save() must store token IDs for backward scatter");
        }
        
        if (!weight_grad) {
            throw std::runtime_error("EmbeddingGradFn::apply: weight_grad is NULL - capture_weight() must be called first");
        }

        if (!grad_output.data) {
            throw std::runtime_error("EmbeddingGradFn::apply: grad_output.data is NULL");
        }

        if (!vocab_size) {
            throw std::runtime_error("EmbeddingGradFn::apply: vocab_size is 0 — save() was not called or weight_shape is invalid");
        }
        
        // PyTorch-style direct accumulation — embedding grad writes
        // to same buffer where LM head grad already lives. Natural ~90% cancellation
        // for frequent tokens acts as frequency-proportional regularization.
        kernel_embedding_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, token_ids, weight_grad, num_tokens, d_model, vocab_size, embedding_scale);
        trackKernelLaunch("kernel_embedding_backward", stream);
        
        // DEBUG: Capture embedding gradient
        if (g_debug_capture_enabled && g_debug_embedding_only_grad && weight_grad) {
            const size_t total_size = weight_shape.total_elements();
            debugCaptureEmbeddingGrad(weight_grad, total_size, stream);
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (weight_grad_fn) {
            Tensor view;
            view.data = weight_grad; view.shape = weight_shape;
            view.owns_data = false; view.stream = stream;
            weight_grad_fn->apply(view, stream);
            weight_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (owns_token_ids && token_ids) { cudaFree(token_ids); token_ids = nullptr; }
        weight_grad = nullptr;
        weight_grad_fn.reset();
    }
};

/**
 * LogSoftmaxGradFn - Backward for log_softmax operation
 * Saves log-probabilities from forward for exact backward (no recomputation).
 *
 * Gradient: grad_x[i] = grad_y[i] - exp(log_p[i]) * Σ_j grad_y[j]
 *
 * DOES NOT store Tensor* - stores stable data instead (ISSUE #48 pattern)
 */
struct LogSoftmaxGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    bool owns_input_grad = false;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    float* saved_log_softmax = nullptr;  // Saved log-probabilities from forward
    bool owns_saved_log_softmax = true;  // OOM FIX: false when borrowing from NLLLossGradFn
    int num_tokens = 0;
    int dim = 0;

    LogSoftmaxGradFn() { op_name = "log_softmax"; }

    ~LogSoftmaxGradFn() override {
        if (owns_saved_log_softmax && saved_log_softmax) { cudaFree(saved_log_softmax); }
        if (owns_input_grad && input_grad) { cudaFree(input_grad); input_grad = nullptr; }
    }

    void capture_input(Tensor& x) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;

        // Copy shared_ptr to captured grad_fn
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            // OOM FIX: Check tensor's grad buffer FIRST, only allocate if missing.
            // OLD CODE leaked 1.37GB/batch: cudaMalloc'd a buffer, then overwrote
            // the pointer with x.grad_data() without freeing the malloc'd buffer.
            x.ensure_grad();
            if (x.grad_data()) {
                // Input has grad buffer — accumulate into it during apply()
                input_grad = x.grad_data();
                owns_input_grad = false;
            } else {
                // Issue #126 fallback: allocate OWN buffer when input tensor
                // is a temporary that can't provide a grad buffer
                const size_t bytes = x.shape.total_elements() * sizeof(float);
                cudaError_t err = cudaMalloc(&input_grad, bytes);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("[LogSoftmaxGradFn::capture_input] cudaMalloc failed for input_grad (")
                        + std::to_string(bytes) + " bytes): " + cudaGetErrorString(err));
                }
                cudaMemset(input_grad, 0, bytes);
                owns_input_grad = true;
            }
        }
    }

    /// Save log-softmax output for backward pass.
    /// @param copy  When true (default), allocates GPU buffer and copies data.
    ///              When false, stores non-owning pointer — caller guarantees
    ///              the data lives until after apply() completes.
    ///              OOM FIX: unified_loss() passes false because NLLLossGradFn
    ///              owns the same data and keeps it alive through backward.
    void save(const float* log_softmax_output, int tokens, int d, cudaStream_t stream, bool copy = true) {
            num_tokens = tokens;
        dim = d;
        if (copy) {
            const size_t bytes = static_cast<size_t>(tokens) * d * sizeof(float);
            cudaError_t err = cudaMalloc(&saved_log_softmax, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("[LogSoftmaxGradFn::save] cudaMalloc failed for saved_log_softmax (")
                    + std::to_string(bytes) + " bytes): " + cudaGetErrorString(err));
            }
            cudaMemcpyAsync(saved_log_softmax, log_softmax_output, bytes,
                            cudaMemcpyDeviceToDevice, stream);
            owns_saved_log_softmax = true;
        } else {
            // Non-owning reference — saves 1.37GB by avoiding duplicate copy
            saved_log_softmax = const_cast<float*>(log_softmax_output);
            owns_saved_log_softmax = false;
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("log_softmax", this);

        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;  // No grad needed
        if (!saved_log_softmax) throw std::runtime_error("LogSoftmaxGradFn::apply: saved_log_softmax is NULL - forward must save output for backward");
        if (!input_grad) {
            throw std::runtime_error(
                "LogSoftmaxGradFn::apply: input_grad is NULL — "
                "capture_input() must be called first");
        }

        kernel_log_softmax_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, saved_log_softmax, input_grad, num_tokens, dim);
        trackKernelLaunch("kernel_log_softmax_backward", stream);

        // Continue autograd chain
        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad;
            view.shape = input_shape;
            view.owns_data = false;
            view.stream = stream;
            input_grad_fn->apply(view, stream);
            input_grad_fn->release_saved();
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (owns_saved_log_softmax && saved_log_softmax) { cudaFree(saved_log_softmax); saved_log_softmax = nullptr; }
        else { saved_log_softmax = nullptr; }  // Clear borrowed pointer without freeing
        if (owns_input_grad && input_grad) { cudaFree(input_grad); input_grad = nullptr; }
        owns_input_grad = false;
        input_grad_fn.reset();
    }
};

/**
 * DropoutGradFn - Backward for dropout operation (ISSUE #48 FIX)
 * DOES NOT store Tensor* - stores stable data instead
 * ISSUE #133 FIX: Allocates own gradient buffer (same pattern as Issue #126 RMSNormGradFn)
 */
struct DropoutGradFn : public GradFn {
    // ISSUE #48: Store stable data, NOT Tensor* pointers
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    bool owns_input_grad = false;  // ISSUE #133: Track if we own the buffer
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    uint8_t* saved_mask = nullptr;  // Binary mask from forward
    float scale = 1.0f;             // 1.0 / (1.0 - dropout_prob)
    size_t count = 0;
    
    DropoutGradFn() { op_name = "dropout"; }
    
    ~DropoutGradFn() override {
        if (saved_mask) cudaFree(saved_mask);
        // ISSUE #133: Free owned gradient buffer
        if (owns_input_grad && input_grad) { cudaFree(input_grad); input_grad = nullptr; }
    }
    
    void capture_input(Tensor& x) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        
        // Copy shared_ptr to captured grad_fn
        input_grad_fn = x.grad_fn;
        
        // ISSUE #133 FIX: Allocate our OWN gradient buffer (same as Issue #126 for RMSNormGradFn)
        // The input tensor x may be destroyed before backward() runs, so we can't borrow its buffer
        if (input_requires_grad) {
            size_t grad_size = x.numel();
            cudaMalloc(&input_grad, grad_size * sizeof(float));
            cudaMemset(input_grad, 0, grad_size * sizeof(float));
            owns_input_grad = true;
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
        
        if (!input_requires_grad) return;  // No grad needed
        if (!saved_mask) {
            throw std::runtime_error("DropoutGradFn::apply: saved_mask is NULL - forward must save dropout mask for backward");
        }
        if (!input_grad) {
            throw std::runtime_error("DropoutGradFn::apply: input_grad is NULL - capture_input() must be called first");
        }
        if (!grad_output.data) {
            throw std::runtime_error("DropoutGradFn::apply: grad_output.data is NULL");
        }
        
        kernel_dropout_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, saved_mask, input_grad, scale, count);
        trackKernelLaunch("kernel_dropout_backward", stream);

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            input_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        if (saved_mask) { cudaFree(saved_mask); saved_mask = nullptr; }
        // ISSUE #133: Free owned gradient buffer
        if (owns_input_grad && input_grad) { cudaFree(input_grad); owns_input_grad = false; }
        input_grad = nullptr;
        input_grad_fn.reset();
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
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<GradFn> residual_grad_fn;
    size_t element_count = 0;
    
    ResidualAddGradFn() { op_name = "residual_add"; }
    
    void capture_inputs(Tensor& x, Tensor& r, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        residual_requires_grad = r.requires_grad;
        input_shape = x.shape;
        residual_shape = r.shape;
        
        // Copy shared_ptrs to captured grad_fns
        input_grad_fn = x.grad_fn;
        residual_grad_fn = r.grad_fn;
        
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
        
        if (input_requires_grad && input_grad) {
            kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                input_grad, grad_output.data, count, 1.0f);
        }
        
        if (residual_requires_grad && residual_grad) {
            kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                residual_grad, grad_output.data, count, 1.0f);
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn pointers
        if (input_requires_grad && input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
            input_grad_fn->release_saved();
        }
        if (residual_requires_grad && residual_grad_fn && residual_grad_fn != input_grad_fn) {
            Tensor view;
            view.data = residual_grad; view.shape = residual_shape;
            view.owns_data = false; view.stream = stream;
            residual_grad_fn->apply(view, stream);
            residual_grad_fn->release_saved();
        }
    }
    
    void release_saved() override {
        GradFn::release_saved();
        input_grad = nullptr;
        residual_grad = nullptr;
        input_grad_fn.reset();
        residual_grad_fn.reset();
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
    
    Tensor result = Tensor::empty(a.shape, a.requires_grad || b.requires_grad, stream, "add_result");
    
    // c = a + b
    // Use TensorContract::add for the forward
    TensorContract::TensorView a_view(const_cast<float*>(a.data), a.shape);
    TensorContract::TensorView b_view(const_cast<float*>(b.data), b.shape);
    TensorContract::TensorView r_view(result.data, result.shape);
    TensorContract::add(a_view, b_view, r_view, stream);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<AddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);  // ISSUE #56 FIX
        result.grad_fn = grad_fn;
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
    if (!input.shape.is_2d_layout()) {
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
    Tensor result = Tensor::empty(input.shape, input.requires_grad || bias.requires_grad, stream, "broadcast_add_result");
    
    // Forward: Copy input to output, then add bias in-place
    const size_t total_bytes = static_cast<size_t>(total_tokens) * features * sizeof(float);
    cudaMemcpyAsync(result.data, input.data, total_bytes, cudaMemcpyDeviceToDevice, stream);
    
    // Bias add forward (owned by autograd layer, not FFN-specific)
    launchBiasAdd(result.data, bias.data, total_tokens, features, stream);
    
    // Set up backward - ISSUE #97: This was missing for encoder biases!
    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<BiasAddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(input), const_cast<Tensor&>(bias), 
                                total_tokens, features, stream);
        result.grad_fn = grad_fn;
    }
    
    AG_TRACE("[autograd::broadcast_add] input[%d,%d] + bias[%d] -> output[%d,%d] requires_grad=%d\n",
             total_tokens, features, bias_size, total_tokens, features, result.requires_grad);
    
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
    
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "gelu_result");
    
    // Forward: y = gelu(x)
    // gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const size_t count = x.numel();
    
    // Launch GELU forward kernel
    kernel_gelu_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<GeluGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        
        // ISSUE #51 FIX: Copy cache to owned buffer
        const float* effective_cache = input_cache ? input_cache : x.data;
        grad_fn->set_cache_copy(effective_cache, count, stream);
        result.grad_fn = grad_fn;
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
    
    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::rms_norm: input must be 2D (BSM)");
    }
    
    const auto& dims = x.shape.as_2d();
    const int tokens = dims.rows;
    const int d_model = dims.cols;
    
    Tensor result = Tensor::empty(x.shape, x.requires_grad || gamma.requires_grad, stream, "rms_norm_result");
    
    // Forward: y = x / rms(x) * gamma
    // Shared memory: one float per warp for reduction
    const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32 + 1) * sizeof(float);
    
    kernel_rmsnorm_forward<<<tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
        x.data, gamma.data, result.data, tokens, d_model, eps);
    
 
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<RMSNormGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), const_cast<Tensor&>(gamma), stream);  // ISSUE #126: pass stream
        
        // ISSUE #51 FIX: Copy cache to owned buffer
        const float* effective_cache = input_cache ? input_cache : x.data;
        grad_fn->set_cache_copy(effective_cache, static_cast<size_t>(tokens) * d_model, d_model, eps, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

// NOTE: cross_entropy() removed - use autograd::unified_loss() from AutogradLoss.cu
// Production training uses ComputeLossBatch.cu -> autograd::unified_loss() which
// properly computes loss AND backward with correct 1/N scaling (Issue #58 fix).
// Issue #142: cross_entropy_loss() also deleted (was thin wrapper around unified_loss).

Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens, cudaStream_t stream, float embedding_scale) {
    if (!weight.shape.is_2d_layout()) {
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
    Tensor result = Tensor::empty(output_shape, weight.requires_grad, stream, "embedding_result");
    
    // Forward: gather from weight table with scaling
    // Issue #140: Scale is 1.0f in production (AIAYN sqrt(d_model) removed for tied weights)
    const int vocab_size = weight.shape.as_2d().rows;
    kernel_embedding_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        token_ids, weight.data, result.data, num_tokens, d_model, vocab_size, embedding_scale);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (weight.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<EmbeddingGradFn>();
        grad_fn->capture_weight(const_cast<Tensor&>(weight));
        grad_fn->save(token_ids, num_tokens, d_model, true, stream);
        grad_fn->vocab_size = vocab_size;             // RULE 20: Store for backward OOB checking
        grad_fn->embedding_scale = embedding_scale;   // Store for backward scaling
        result.grad_fn = grad_fn;
    }
    
    return result;
}

Tensor log_softmax(const Tensor& x, cudaStream_t stream, bool save_output_copy) {
    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::log_softmax: input must be 2D [tokens, dim]");
    }
    if (!x.data) {
        throw std::invalid_argument("autograd::log_softmax: input data is NULL");
    }

    const auto dims = x.shape.as_2d();
    const int num_tokens = dims.rows;
    const int dim = dims.cols;

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "log_softmax_result");

    // Forward: log_softmax(x)[i] = x[i] - logsumexp(x)
    // Numerically superior to softmax→log: stays in log space entirely.
    kernel_log_softmax_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_tokens, dim);
    trackKernelLaunch("kernel_log_softmax_forward", stream);

    // Set up backward — save log-probabilities for exact gradient computation
    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<LogSoftmaxGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x));
        // OOM FIX: When save_output_copy=false (used by unified_loss), store
        // non-owning reference instead of copying 1.37GB. NLLLossGradFn owns
        // the same data and guarantees lifetime through backward.
        grad_fn->save(result.data, num_tokens, dim, stream, save_output_copy);
        result.grad_fn = grad_fn;
    }

    return result;
}


Tensor dropout(const Tensor& x, float p, uint64_t seed, bool training, cudaStream_t stream) {
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "dropout_seeded_result");
    
    if (!training || p == 0.0f) {
        // No dropout during inference or when p=0
        cudaMemcpyAsync(result.data, x.data, x.size_bytes(), cudaMemcpyDeviceToDevice, stream);
        
        if (x.requires_grad) {
            result.is_leaf = false;
            auto grad_fn = std::make_shared<AddGradFn>();
            grad_fn->capture_single_input(const_cast<Tensor&>(x), stream);
            result.grad_fn = grad_fn;
        }
        return result;
    }
    
    const size_t count = x.numel();
    const float scale = 1.0f / (1.0f - p);
    
    // Allocate mask on device
    uint8_t* mask = nullptr;
    cudaMalloc(&mask, count * sizeof(uint8_t));
    
    // Generate random mask: 1 = keep (with prob 1-p), 0 = drop (with prob p)
    kernel_generate_dropout_mask<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        mask, count, p, seed);
    trackKernelLaunch("kernel_generate_dropout_mask", stream);
    
    // Forward: y = x * mask * scale
    kernel_dropout_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, mask, result.data, scale, count);
    trackKernelLaunch("kernel_dropout_forward", stream);
    
    // Set up backward - ISSUE #48: capture stable data
    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<DropoutGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x));
        // save() copies the mask and takes ownership - we must free our copy
        grad_fn->save(mask, p, count, stream);
        result.grad_fn = grad_fn;
    }
    
    // Free our mask copy (DropoutGradFn::save() made its own copy).
    // Use stream-ordered async free to avoid host-side synchronization.
    const cudaError_t free_err = cudaFreeAsync(mask, stream);
    if (free_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("autograd::dropout: cudaFreeAsync(mask) failed: ") +
            cudaGetErrorString(free_err));
    }
    
    return result;
}

Tensor residual_add(const Tensor& x, const Tensor& residual, cudaStream_t stream) {
    if (x.numel() != residual.numel()) {
        throw std::invalid_argument("autograd::residual_add: tensor size mismatch");
    }
    
    Tensor result = Tensor::empty(x.shape, x.requires_grad || residual.requires_grad, stream, "residual_add_result");
    
    // Forward: y = x + residual
    TensorContract::TensorView x_view(const_cast<float*>(x.data), x.shape);
    TensorContract::TensorView r_view(const_cast<float*>(residual.data), residual.shape);
    TensorContract::TensorView out_view(result.data, result.shape);
    TensorContract::add(x_view, r_view, out_view, stream);
    
    // Set up backward - ISSUE #48: capture stable data, not Tensor*
    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ResidualAddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), const_cast<Tensor&>(residual), stream);  // ISSUE #56 FIX
        result.grad_fn = grad_fn;
    }
    
    return result;
}

// autograd::scale() DELETED — dead code from reverted Issue #98 (Rule 20)

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
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "layer_scale_result");
    
    // Forward: y = x * scale_value  (broadcast)
    TensorContract::TensorView x_view(const_cast<float*>(x.data), x.shape);
    TensorContract::TensorView out_view(result.data, result.shape);
    TensorContract::scale(x_view, scale_value, out_view, stream);
    
    // Set up backward
    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<LayerScaleGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), scale_param, scale_value, stream);
        result.grad_fn = grad_fn;
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
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("center_rows: expected 2D (flat) tensor, got 4D");
    }
    const int num_rows = x.shape.as_2d().rows;  // total_tokens
    const int row_dim = x.shape.as_2d().cols;   // d_model (768)
    
    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_rows_result");
    
    // Forward: y = x - mean(x)  (per-row)
    kernel_center_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, row_dim, num_rows);
    
    // Set up backward
    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterRowsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), row_dim, num_rows, stream);
        result.grad_fn = grad_fn;
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
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("center_columns: expected 2D (flat) tensor, got 4D");
    }
    const int num_rows = x.shape.as_2d().rows;  // total_tokens (positions)
    const int num_cols = x.shape.as_2d().cols;  // d_model (768 features)
    
    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_columns_result");
    
    // Forward: y[t,d] = x[t,d] - mean_t(x[:,d])  (per-column mean subtraction)
    // Launch one block per column (768 blocks for d_model=768)
    kernel_center_columns<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_cols, num_rows);
    
    // Set up backward
    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterColumnsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, stream);
        result.grad_fn = grad_fn;
    }
    
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// project_out_pc1 — Issue #149
// Projects out the dominant PC1 direction from hidden states H [T×D].
// Reduces avg_cos(h_i,h_j) to prevent causal-attention prefix-averaging
// from creating a shared direction that causes mode collapse.
//
// Forward:  h̃[t] = h[t] - (h[t]·g/D)*g   where g = PC1(H) via power iteration (rms-normalized)
// Backward: grad_h = (I - gg^T/D) * grad_h̃   (g treated as stop-gradient)
//
// g is initialised from the column mean, then refined with n_power_iters
// steps of the power method (H^T H g normalised each step).
// ─────────────────────────────────────────────────────────────────────────────
Tensor project_out_pc1(const Tensor& x, int n_power_iters, cudaStream_t stream) {
    if (x.numel() == 0)
        throw std::runtime_error("project_out_pc1: input tensor is empty");

    const int D = (int)x.shape.flat.cols;
    const int T = (int)(x.numel() / (std::size_t)D);
    if (D <= 0 || T <= 0)
        throw std::runtime_error("project_out_pc1: invalid dimensions T=" + std::to_string(T) + " D=" + std::to_string(D));

    bool track_grad = x.requires_grad;

    // Working buffers on device
    float* g_buf  = nullptr;  // [D]
    float* g_tmp  = nullptr;  // [D]
    float* v_buf  = nullptr;  // [T]
    cudaMalloc(&g_buf, D * sizeof(float));
    cudaMalloc(&g_tmp, D * sizeof(float));
    cudaMalloc(&v_buf, T * sizeof(float));

    // Initialize PC1 guess from column mean, then normalize
    kernel_pc1_col_mean<<<1, 256, 0, stream>>>(x.data, g_buf, T, D);
    kernel_pc1_normalize<<<1, 256, 0, stream>>>(g_buf, D);

    // Power iteration: g ← normalize(H^T (H g))
    const int blk = 256;
    for (int iter = 0; iter < n_power_iters; ++iter) {
        kernel_pc1_gemv_Hg<<<(T + blk - 1) / blk, blk, 0, stream>>>(x.data, g_buf, v_buf, T, D);
        kernel_pc1_gemv_HtV<<<(D + blk - 1) / blk, blk, 0, stream>>>(x.data, v_buf, g_tmp, T, D);
        kernel_pc1_normalize<<<1, 256, 0, stream>>>(g_tmp, D);
        cudaMemcpyAsync(g_buf, g_tmp, D * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    // Single sync after all PC1 kernels — needed before host-side autograd capture
    cudaError_t pc1_err = cudaStreamSynchronize(stream);
    if (pc1_err != cudaSuccess) throw std::runtime_error("project_out_pc1: PC1 kernels failed: " + std::string(cudaGetErrorString(pc1_err)));

    // Wire autograd graph — capture g BEFORE projecting (backward only needs g)
    // GradFn makes its own device copy of g_buf
    Tensor& x_mut = const_cast<Tensor&>(x);
    if (track_grad) {
        auto grad_fn    = std::make_shared<ProjectOutPC1GradFn>();
        grad_fn->capture_input(x_mut, T, D, g_buf, stream);
        x_mut.grad_fn   = grad_fn;
        x_mut.is_leaf    = false;
    }

    // Project IN-PLACE: H[t,d] -= (H[t,:]·g / D) * g[d]  (g is RMS-normalized, g·g=D)
    // Safe because kernel reads entire row into shared mem before writing.
    // Eliminates 24MB allocation that caused OOM.
    kernel_pc1_project<<<T, 256, 0, stream>>>(x_mut.data, g_buf, x_mut.data, T, D);

    // Free temporary buffers (async safe — project kernel submitted to stream before free)
    cudaFreeAsync(v_buf, stream);
    cudaFreeAsync(g_tmp, stream);
    cudaFreeAsync(g_buf, stream);  // GradFn already captured its own copy

    // Return non-owning view of the (now projected) input data
    Tensor result;
    result.data      = x_mut.data;
    result.shape     = x_mut.shape;
    result.owns_data = false;  // Input still owns the buffer
    result.requires_grad = track_grad;
    result.is_leaf   = false;
    result.grad_fn   = x_mut.grad_fn;  // Share the autograd chain
    result.stream    = stream;

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

// Issue #77 diagnostics: DISABLED. Enable for debugging weight gradient explosions.
// When enabled, logs cached activation stats for first 24 MatMulGradFn calls.
// Cost: 6 cudaMalloc/Free + 1 cudaStreamSync per call (144 total for 24 calls).
static bool g_issue77_diag_enabled = false;
static int g_issue77_diag_call_count = 0;

// Issue #142: applyLmHeadGradCorrections DELETED.
// Centering is now INSIDE autograd graph (Issues #125/#132):
//   CenterRowsGradFn::apply() row-centers grad_A in backward
//   CenterColumnsGradFn::apply() column-centers grad_A in backward
// The old external centerGradientsKernel was redundant (row centering is idempotent)
// and wasted GPU time (kernel launch + cudaStreamSynchronize + 6x fprintf per call).

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
    std::shared_ptr<GradFn> a_grad_fn;   // Chain continuation for A
    std::shared_ptr<GradFn> b_grad_fn;   // Chain continuation for B
    
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
    
    // shared_ptr members destruct automatically
    
    // ISSUE #48 FIX: Store stable info from tensors during forward, before they go out of scope
    // ISSUE #55 FIX: For non-leaf (intermediate) tensors, allocate OWNED grad buffers
    //                because the tensor's grad buffer gets freed when tensor goes out of scope.
    //                For leaf (weight) tensors, use their grad buffer directly since they persist.
    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_shape = a.shape;
        b_shape = b.shape;
        
        // Copy shared_ptrs to captured grad_fns
        a_grad_fn = a.grad_fn;
        b_grad_fn = b.grad_fn;
        
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
            
            // ========================================================================
            // ISSUE #77 DIAGNOSTIC: Log cached_a (ln1_out) values BEFORE the GEMM
            // This helps diagnose why W_qkv gradients explode despite tiny FA outputs
            // ========================================================================
            if (g_issue77_diag_enabled && transpose_b) {
                g_issue77_diag_call_count++;
                // Only log first 36 calls (3 per layer * 12 layers) to avoid spam
                if (g_issue77_diag_call_count <= 36) {
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

        // Issue #142: applyLmHeadGradCorrections removed.
        // Centering backward is handled by CenterRowsGradFn/CenterColumnsGradFn
        // inside the autograd chain (Issues #125/#132).

        // CONTINUE AUTOGRAD CHAIN (Recursive) - ISSUE #48 FIX: Use stored grad_fn pointers
        if (a_requires_grad && a_grad_fn) {
            if (a_grad_fn->op_name) {
                Tensor view;
                view.data = grad_a; view.shape = a_shape;
                view.owns_data = false; view.stream = stream;
                
                // ISSUE #127: Validate GradFn to detect use-after-free
                {
                    void** vtable_ptr = *(void***)a_grad_fn.get();
                    uint64_t vtable_value = (uint64_t)vtable_ptr;
                    if (vtable_value == 0xDDDDDDDDDDDDDDDDull ||
                        vtable_value == 0xFEEEFEEEFEEEFEEEull ||
                        vtable_value == 0xCDCDCDCDCDCDCDCDull ||
                        vtable_value == 0x0000000000000000ull) {
                        throw std::runtime_error("Use-after-free detected: a_grad_fn vtable is corrupted (value=0x" +
                            std::to_string(vtable_value) + ", op=" +
                            std::string(a_grad_fn->op_name ? a_grad_fn->op_name : "NULL") + ")");
                    }
                }
                
                a_grad_fn->apply(view, stream);
                // ISSUE #52 FIX: Do NOT call release_saved() here - cudaFree blocks while GPU busy
            }
        }
        if (b_requires_grad && b_grad_fn && b_grad_fn != a_grad_fn) {
            if (b_grad_fn->op_name) {
                Tensor view;
                view.data = grad_b; view.shape = b_shape;
                view.owns_data = false; view.stream = stream;
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
        a_grad_fn.reset();
        b_grad_fn.reset();
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
    if (!a.shape.is_2d_layout() || !b.shape.is_2d_layout()) {
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
    Tensor result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad, stream, "matmul_result");
    
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
        auto grad_fn = std::make_shared<MatMulGradFn>();
        
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
    }
    
    return result;
}

//======================================================//
//  BF16 Conversion Kernels (for FlashAttention integration)
//  NOTE: Layout conversion kernels (BHSD<->BSHD bf16) live in TensorConversion.cu
//  Only GQA-specific reduction kernel remains here.
//======================================================

// ISSUE #72 FIX: Reduce GQA gradients from num_heads to num_kv_heads
// FlashAttention backward writes dK/dV for each query head separately (12 heads).
// For GQA with 4 KV heads, we need to SUM the gradients from grouped Q heads.
// E.g., Q heads 0,1,2 all use KV head 0, so dK[kv_head=0] = dK[q_head=0] + dK[q_head=1] + dK[q_head=2]
//
// ISSUE #73 / balance fix: External FlashAttention library does NOT apply GQA gradient scaling.
// We sum gradients from heads_per_kv_group Q heads into one KV head. Two choices:
//   (1) No scale: true backprop gradient = sum → ||dK|| ~ sqrt(heads_per_kv) * per-Q magnitude (K,V get more).
//   (2) Scale 1/sqrt(heads_per_kv): so ||dK|| = ||dQ|| per head → Q,K,V comparable (was 1/heads_per_kv → Q 1.7x larger).
// We use (2) so Q, K, V receive comparable gradient magnitude (no structural 1.7x imbalance).
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
    
    // Balance Q/K/V gradient magnitude: scale so ||dK||, ||dV|| match ||dQ|| per head.
    // sum has norm ~ sqrt(heads_per_kv_group) * single-head norm → use 1/sqrt to match.
    const float gqa_grad_scale = 1.0f / sqrtf(static_cast<float>(heads_per_kv_group));
    
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
    std::shared_ptr<GradFn> q_grad_fn;
    std::shared_ptr<GradFn> k_grad_fn;
    std::shared_ptr<GradFn> v_grad_fn;
    
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
    
    // Attention dropout (saved for backward to reproduce same mask)
    float attention_dropout_p = 0.0f;
    uint64_t dropout_seed = 0;
    
    ScaledDotProductAttentionGradFn() { op_name = "scaled_dot_product_attention"; }
    
    ~ScaledDotProductAttentionGradFn() override {
        release_saved();
    }
    
    void capture_inputs(Tensor& q, Tensor& k, Tensor& v) {
        q_requires_grad = q.requires_grad;
        k_requires_grad = k.requires_grad;
        v_requires_grad = v.requires_grad;
        q_shape = q.shape;
        k_shape = k.shape;
        v_shape = v.shape;
        
        // Copy shared_ptrs to captured grad_fns
        q_grad_fn = q.grad_fn;
        k_grad_fn = k.grad_fn;
        v_grad_fn = v.grad_fn;
        
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
        // Q: [B, H, S, D] FP32 -> [B, S, H, D] BF16
        TensorConversion::convert_BHSD_to_BSHD_bf16(q.data, saved_q_bf16, b, nh, s, hd, stream);
        // K: [B, Hkv, S, D] FP32 -> [B, S, Hkv, D] BF16
        TensorConversion::convert_BHSD_to_BSHD_bf16(k.data, saved_k_bf16, b, nkv, s, hd, stream);
        // V: [B, Hkv, S, D] FP32 -> [B, S, Hkv, D] BF16
        TensorConversion::convert_BHSD_to_BSHD_bf16(v.data, saved_v_bf16, b, nkv, s, hd, stream);
        // Output: [B, H, S, D] FP32 -> [B, S, H, D] BF16
        TensorConversion::convert_BHSD_to_BSHD_bf16(out.data, saved_out_bf16, b, nh, s, hd, stream);
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // RULE 20: Track current operation for error context
        setCurrentGradFnOp("scaled_dot_product_attention", this);
        
        // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
        if (applied) {
            return;
        }
        applied = true;
        
        if (!saved_q_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_q_bf16 is NULL - save() must store Q for backward");
        if (!saved_k_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_k_bf16 is NULL - save() must store K for backward");
        if (!saved_v_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_v_bf16 is NULL - save() must store V for backward");
        if (!saved_out_bf16) throw std::runtime_error("ScaledDotProductAttentionGradFn::apply: saved_out_bf16 is NULL - save() must store output for backward");
        
        const size_t q_elems = static_cast<size_t>(batch_size) * seq_len * num_heads * head_dim;
        const size_t kv_elems = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;
        const int block_size = 256;
        const int q_blocks = static_cast<int>((q_elems + block_size - 1) / block_size);
        const int kv_blocks = static_cast<int>((kv_elems + block_size - 1) / block_size);
        
        // Convert grad_output (FP32 BHSD) to BF16 BSHD
        TensorConversion::convert_BHSD_to_BSHD_bf16(
            grad_output.data, dout_bf16, batch_size, num_heads, seq_len, head_dim, stream);
        
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
            attention_dropout_p,  // Same dropout rate as forward
            dropout_seed,         // Same seed as forward (reproduces identical mask)
            stream
        );
        
        // =========================================================================
        // ISSUE #83 REMOVAL: Issue #84 (missing preprocessing kernel)
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
            
            TensorConversion::convert_BSHD_bf16_to_BHSD(
                dq_bf16, grad_q_fp32, batch_size, seq_len, num_heads, head_dim, stream);
            
            // Scale = 1.0 (no normalization - Issue #84 fixed root cause)
            kernel_accumulate_grad<<<gridForCount(q_elems), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
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
            kernel_accumulate_grad<<<gridForCount(kv_elems), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
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
            kernel_accumulate_grad<<<gridForCount(kv_elems), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                v_grad, grad_v_fp32, kv_elems, 1.0f);
            cudaFreeAsync(grad_v_fp32, stream);
        }
        
        // CONTINUE AUTOGRAD CHAIN - call grad_fns for Q, K, V
        if (q_requires_grad && q_grad_fn) {
            Tensor q_view;
            q_view.data = q_grad; q_view.shape = q_shape;
            q_view.owns_data = false; q_view.stream = stream;
            q_grad_fn->apply(q_view, stream);
            q_grad_fn->release_saved();
        }
        if (k_requires_grad && k_grad_fn) {
            Tensor k_view;
            k_view.data = k_grad; k_view.shape = k_shape;
            k_view.owns_data = false; k_view.stream = stream;
            k_grad_fn->apply(k_view, stream);
            k_grad_fn->release_saved();
        }
        if (v_requires_grad && v_grad_fn) {
            Tensor v_view;
            v_view.data = v_grad; v_view.shape = v_shape;
            v_view.owns_data = false; v_view.stream = stream;
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
        q_grad_fn.reset();
        k_grad_fn.reset();
        v_grad_fn.reset();
    }
};

Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const float* alibi_slopes, float scale, cudaStream_t stream,
    bool causal,
    float attention_dropout_p, uint64_t dropout_seed
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
    Tensor result = Tensor::zeros(output_shape, requires_grad, stream, "sdpa_result");
    
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
    // Sentinel fill: 0xFF bytes → float NaN. If LSE shows NaN after kernel, kernel didn't write.
    // Valid LSE is always finite (LSE = max_score * scale + log(sum_exp)), never NaN.
    cudaMemsetAsync(softmax_lse, 0xFF, lse_elems * sizeof(float), stream);
    
    // Convert FP32 BHSD -> BF16 BSHD
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        q.data, q_bf16, batch_size, num_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        k.data, k_bf16, batch_size, num_kv_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        v.data, v_bf16, batch_size, num_kv_heads, seq_len, head_dim, stream);
    
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
        causal,      // Use parameter instead of hardcoded true
        true,        // is_bf16
        attention_dropout_p, // Attention dropout rate (0.0 = disabled)
        dropout_seed,        // Per-step Philox seed for reproducible masks
        stream
    );
    
    // Convert BF16 BSHD -> FP32 BHSD for output
    TensorConversion::convert_BSHD_bf16_to_BHSD(
        out_bf16, result.data, batch_size, seq_len, num_heads, head_dim, stream);
    
    // Set up backward if needed - ISSUE #48: capture stable data, not Tensor*
    if (requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ScaledDotProductAttentionGradFn>();
        
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
        grad_fn->causal = causal;  // Use parameter
        grad_fn->alibi_slopes = alibi_slopes;  // Save for backward pass (not owned)
        grad_fn->attention_dropout_p = attention_dropout_p;  // Same dropout for backward mask reproduction
        grad_fn->dropout_seed = dropout_seed;                // Same seed reproduces identical Philox mask
        
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
        
        // Ownership of bf16 buffers transferred to grad_fn for backward pass.
        // DO NOT nullify pointers - if someone accidentally frees them after transfer,
        // let the double-free crash loudly (Rule 20: fail loud).
        // Any accidental cudaFree(q_bf16) here will produce a clear double-free error
        // that immediately reveals the bug instead of silently succeeding. 
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
    std::shared_ptr<GradFn> input_grad_fn;
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
    }
    
    void capture_input(Tensor& bhsd_input) {
        input_requires_grad = bhsd_input.requires_grad;
        input_shape = bhsd_input.shape;
        
        // Copy shared_ptr to captured grad_fn
        input_grad_fn = bhsd_input.grad_fn;
        
        if (input_requires_grad) {
            bhsd_input.ensure_grad();
            input_grad = bhsd_input.grad_data();
        }
    }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("reshape_bhsd_to_flat", this);
        
        if (applied) return;
        applied = true;
        
        if (!input_requires_grad) {
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
        
        // Continue chain to attention backward
        if (input_grad_fn) {
            Tensor bhsd_grad_tensor;
            bhsd_grad_tensor.data = bhsd_grad;
            bhsd_grad_tensor.shape = input_shape;
            bhsd_grad_tensor.owns_data = false;
            bhsd_grad_tensor.stream = stream;
            
            input_grad_fn->apply(bhsd_grad_tensor, stream);
            input_grad_fn->release_saved();
        } else {
            throw std::runtime_error("[ReshapeBHSDtoFlat] input_grad_fn is NULL - autograd chain is broken at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
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
    Tensor result = Tensor::empty(TensorContract::TensorShape::make_BSM(tokens, d_model), bhsd_input.requires_grad, stream, "reshape_bhsd_to_flat_result");
    
    // Call the existing reshape kernel (declared in Encoding_GPU.hpp)
    // This reshapes [B, H, S, D] to [B*S, H*D]
    launchReshapeFromBHSD(bhsd_input.data, result.data, batch_size, seq_len, num_heads, head_dim, stream);
    
    // Set up backward if needed
    if (bhsd_input.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ReshapeFromBHSDGradFn>();
        
        // Capture input for backward
        grad_fn->capture_input(bhsd_input);
        grad_fn->batch_size = batch_size;
        grad_fn->seq_len = seq_len;
        grad_fn->num_heads = num_heads;
        grad_fn->head_dim = head_dim;
        
        result.grad_fn = grad_fn;
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

// Fused BHSD→QKV gradient merge kernel
// Reads directly from BHSD-layout gradient tensors and writes to flat QKV gradient buffer.
// Eliminates intermediate BSM buffers and the race condition from ISSUE #64.
__global__ void kernel_fused_bhsd_to_qkv_grad(
    float* __restrict__ grad_qkv,           // [tokens, qkv_dim] OUTPUT (flat)
    const float* __restrict__ grad_Q_bhsd,   // [batch, num_heads, seq, head_dim] INPUT
    const float* __restrict__ grad_K_bhsd,   // [batch, num_kv_heads, seq, head_dim] INPUT
    const float* __restrict__ grad_V_bhsd,   // [batch, num_kv_heads, seq, head_dim] INPUT
    int batch, int seq, int num_heads, int num_kv_heads, int head_dim
) {
    const int token = blockIdx.x;
    const int col = threadIdx.x;
    if (token >= batch * seq) return;

    const int b = token / seq;
    const int s = token % seq;
    const int d_model = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int qkv_dim = d_model + 2 * kv_dim;
    float* out_row = grad_qkv + token * qkv_dim;

    // Q gradient: BHSD[b, h, s, d] → flat[token, h*head_dim + d]
    if (col < d_model) {
        const int h = col / head_dim;
        const int d = col % head_dim;
        out_row[col] = grad_Q_bhsd[((b * num_heads + h) * seq + s) * head_dim + d];
    }

    // K gradient: BHSD[b, h_kv, s, d] → flat[token, d_model + h_kv*head_dim + d]
    if (col < kv_dim) {
        const int h = col / head_dim;
        const int d = col % head_dim;
        out_row[d_model + col] = grad_K_bhsd[((b * num_kv_heads + h) * seq + s) * head_dim + d];
    }

    // V gradient: BHSD[b, h_kv, s, d] → flat[token, d_model + kv_dim + h_kv*head_dim + d]
    if (col < kv_dim) {
        const int h = col / head_dim;
        const int d = col % head_dim;
        out_row[d_model + kv_dim + col] = grad_V_bhsd[((b * num_kv_heads + h) * seq + s) * head_dim + d];
    }
}

// NOTE: Reshape kernels (BSM<->BHSD) live in TensorConversion.cu - single source of truth

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
        std::shared_ptr<GradFn> qkv_grad_fn;       // Grad fn of qkv_out (the matmul result)
        Tensor qkv_out_ref;                  // Reference to qkv_out (for grad buffer access)
        
        // BHSD gradient pointers from Q, K, V backward passes
        // Stored directly from grad_output.data — no intermediate BSM buffers needed.
        // The fused kernel reads these in BHSD layout and writes flat QKV grad.
        const float* grad_Q_bhsd = nullptr;  // [batch, num_heads, seq, head_dim]
        const float* grad_K_bhsd = nullptr;  // [batch, num_kv_heads, seq, head_dim]
        const float* grad_V_bhsd = nullptr;  // [batch, num_kv_heads, seq, head_dim]
        
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
            // shared_ptr members destruct automatically
        }
    };
    
    std::shared_ptr<SharedState> shared;
    
    SplitAndReshapeQKVGradFn() { op_name = "split_and_reshape_qkv"; }
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        const char* type_str = (output_type == OutputType::Q) ? "Q" : 
                               (output_type == OutputType::K) ? "K" : "V";
        
        if (!shared) {
            throw std::runtime_error(std::string("[SplitQKV-") + type_str + "] shared is null");
        }
        if (applied) {
            return;
        }
        applied = true;
        
        auto& state = *shared;
        
        // Store this output's BHSD gradient pointer directly — no intermediate reshape
        if (output_type == OutputType::Q) {
            state.grad_Q_bhsd = grad_output.data;
        } else if (output_type == OutputType::K) {
            state.grad_K_bhsd = grad_output.data;
        } else {  // V
            state.grad_V_bhsd = grad_output.data;
        }
        
        // Check if all three outputs have been processed
        const int count = state.apply_count.fetch_add(1) + 1;
        
        if (count == 3) {
            // All three BHSD gradient pointers collected.
            // Launch ONE fused kernel that reads BHSD directly and writes flat QKV grad.
            // No intermediate BSM buffers, no race condition, no CPU sync needed.
            state.qkv_out_ref.ensure_grad();
            float* qkv_grad = state.qkv_out_ref.grad_data();
            
            const int threads = std::max(state.d_model, state.kv_dim);
            kernel_fused_bhsd_to_qkv_grad<<<state.tokens, threads, 0, stream>>>(
                qkv_grad,
                state.grad_Q_bhsd, state.grad_K_bhsd, state.grad_V_bhsd,
                state.batch, state.seq, state.num_heads, state.num_kv_heads, state.head_dim);
            
            // Continue the chain to qkv_out -> W_qkv
            if (state.qkv_out_ref.requires_grad && state.qkv_grad_fn) {
                Tensor qkv_grad_tensor;
                qkv_grad_tensor.data = qkv_grad;
                qkv_grad_tensor.shape = state.qkv_out_ref.shape;
                qkv_grad_tensor.owns_data = false;
                qkv_grad_tensor.stream = stream;
                
                state.qkv_grad_fn->apply(qkv_grad_tensor, stream);
                state.qkv_grad_fn->release_saved();
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
    Tensor Q_bhsd = Tensor::zeros(q_shape, requires_grad, stream, "qkv_split_Q");
    Tensor K_bhsd = Tensor::zeros(k_shape, requires_grad, stream, "qkv_split_K");
    Tensor V_bhsd = Tensor::zeros(v_shape, requires_grad, stream, "qkv_split_V");
    
    // Intermediate BSM tensors for split (using Tensor for RAII)
    Tensor Q_bsm = Tensor::zeros(TensorContract::TensorShape::make_BSM(tokens, d_model), false, stream, "qkv_split_Q_bsm");
    Tensor K_bsm = Tensor::zeros(TensorContract::TensorShape::make_BSM(tokens, kv_dim), false, stream, "qkv_split_K_bsm");
    Tensor V_bsm = Tensor::zeros(TensorContract::TensorShape::make_BSM(tokens, kv_dim), false, stream, "qkv_split_V_bsm");
    
    // Forward: split qkv_out into Q, K, V (BSM layout)
    const int threads = std::max(d_model, kv_dim);
    kernel_split_qkv_all<<<tokens, threads, 0, stream>>>(
        qkv_out.data, Q_bsm.data, K_bsm.data, V_bsm.data, tokens, d_model, kv_dim);
    
    // Reshape BSM -> BHSD using TensorConversion (handles block sizing internally)
    TensorConversion::convert_BSM_to_BHSD(Q_bsm.data, Q_bhsd.data, batch, seq, num_heads, head_dim, stream);
    TensorConversion::convert_BSM_to_BHSD(K_bsm.data, K_bhsd.data, batch, seq, num_kv_heads, head_dim, stream);
    TensorConversion::convert_BSM_to_BHSD(V_bsm.data, V_bhsd.data, batch, seq, num_kv_heads, head_dim, stream);
    
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
            qkv_out.data, qkv_out.shape, false, qkv_out.requires_grad, "qkv_out_ref");
        shared->qkv_out_ref.share_grad(qkv_out);  // Share gradient buffer with original
        
        // Copy shared_ptr to upstream grad_fn
        shared->qkv_grad_fn = qkv_out.grad_fn;
        
        // No intermediate BSM buffers needed — fused kernel reads BHSD directly
        
        // Create GradFns for each output
        auto q_grad_fn = std::make_shared<SplitAndReshapeQKVGradFn>();
        q_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::Q;
        q_grad_fn->shared = shared;
        
        auto k_grad_fn = std::make_shared<SplitAndReshapeQKVGradFn>();
        k_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::K;
        k_grad_fn->shared = shared;
        
        auto v_grad_fn = std::make_shared<SplitAndReshapeQKVGradFn>();
        v_grad_fn->output_type = SplitAndReshapeQKVGradFn::OutputType::V;
        v_grad_fn->shared = shared;
        
        Q_bhsd.is_leaf = false;
        Q_bhsd.grad_fn = q_grad_fn;
        
        K_bhsd.is_leaf = false;
        K_bhsd.grad_fn = k_grad_fn;
        
        V_bhsd.is_leaf = false;
        V_bhsd.grad_fn = v_grad_fn;
    }
    
    return {std::move(Q_bhsd), std::move(K_bhsd), std::move(V_bhsd)};
}




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
        std::shared_ptr<GradFn> q_upstream_grad_fn;
        std::shared_ptr<GradFn> k_upstream_grad_fn;
        
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
            // shared_ptr members destruct automatically
        }
    };
    
    std::shared_ptr<SharedState> shared;
    
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (!shared) {
            throw std::runtime_error("RoPEGradFn::apply: shared state is NULL - RoPE forward must initialize shared state");
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
        
        // Copy shared_ptrs to upstream grad_fns
        if (Q.grad_fn) {
            shared->q_upstream_grad_fn = Q.grad_fn;
        }
        if (K.grad_fn) {
            shared->k_upstream_grad_fn = K.grad_fn;
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
        auto q_grad_fn = std::make_shared<RoPEGradFn>();
        q_grad_fn->output_type = RoPEGradFn::OutputType::Q;
        q_grad_fn->shared = shared;
        Q.is_leaf = false;
        Q.grad_fn = q_grad_fn;
        
        // Create and attach GradFn for K
        auto k_grad_fn = std::make_shared<RoPEGradFn>();
        k_grad_fn->output_type = RoPEGradFn::OutputType::K;
        k_grad_fn->shared = shared;
        K.is_leaf = false;
        K.grad_fn = k_grad_fn;
        
        AG_TRACE("[rope_rotation] RoPEGradFn attached: Q.grad_fn=%p K.grad_fn=%p\n",
                 (void*)Q.grad_fn.get(), (void*)K.grad_fn.get());
    }
    
    AG_TRACE("[rope_rotation] EXIT\n");
}

}  // namespace autograd

 }  // namespace GRIM
