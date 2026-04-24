//======================================================//
//  TensorContract_GPU.cu
//  CUDA implementation of type-safe tensor operations
//======================================================//
#include "TensorContract_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../Batching/BatchPayload.hpp"  // BatchPayload for batch geometry in autograd ops
#include "../VerboseLogging.hpp"
#include "../CudaAllocUtils.hpp"
#include "../TensorConversion/TensorConversion.hpp"  // Layout conversions - single source of truth
#include "../LogRecorder/LogRecorder.hpp"
#include "../../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Layers/Attention/QKV_Projector.hpp"  // ISSUE #62: For launchReshapeFromBHSD
#include "../PBM/PositionalBiasMethod.hpp"  // ISSUE #119: For RoPE autograd backward
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>  // Issue #107: Philox PRNG for Xavier init
#include <device_launch_parameters.h>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>

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

// Call this after LM head matmul backward to capture its contribution
void debugCaptureLMHeadGrad(float* grad_ptr, size_t size, cudaStream_t stream) {
    if (!g_autograd_verbose) return;  // Debug capture not enabled - valid skip
    if (!g_debug_lm_head_only_grad) throw std::runtime_error("debugCaptureLMHeadGrad: g_debug_lm_head_only_grad buffer is NULL - initDebugGradCapture() must be called first");
    if (!grad_ptr) throw std::runtime_error("debugCaptureLMHeadGrad: grad_ptr is NULL - caller MUST provide valid gradient pointer");
    const size_t copy_size = (size < g_debug_grad_buffer_size ? size : g_debug_grad_buffer_size);
    cudaMemcpyAsync(g_debug_lm_head_only_grad, grad_ptr, copy_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
}

// Call this after embedding backward to capture its contribution
void debugCaptureEmbeddingGrad(float* grad_ptr, size_t size, cudaStream_t stream) {
    if (!g_autograd_verbose) return;  // Debug capture not enabled - valid skip
    if (!g_debug_embedding_only_grad) throw std::runtime_error("debugCaptureEmbeddingGrad: g_debug_embedding_only_grad buffer is NULL - initDebugGradCapture() must be called first");
    if (!grad_ptr) throw std::runtime_error("debugCaptureEmbeddingGrad: grad_ptr is NULL - caller MUST provide valid gradient pointer");
    const size_t copy_size = (size < g_debug_grad_buffer_size ? size : g_debug_grad_buffer_size);
    cudaMemcpyAsync(g_debug_embedding_only_grad, grad_ptr, copy_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
}

// Global stream for async cleanup - initialized on first use
static cudaStream_t g_cleanup_stream = nullptr;
static std::mutex g_cleanup_stream_mutex;

// cuBLAS for autograd is the handle from TrainingState/InferenceState; layers call
// set_autograd_cublas_handle() and autograd matmul uses get_autograd_cublas_handle() (thread-local).

void initCleanupStream() {
    std::lock_guard<std::mutex> lock(g_cleanup_stream_mutex);
    if (g_cleanup_stream == nullptr) {
        cudaStreamCreate(&g_cleanup_stream);
    }
}

// cudaMallocOrThrow and helpers now live in shared header
using GRIM::CudaAlloc::cudaMallocOrThrow;
using GRIM::CudaAlloc::detail::buildCudaAllocFailureMessage;

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
        const char* hint = "";
        switch (status) {
            case CUBLAS_STATUS_INVALID_VALUE: hint = " (dim/leading-dim mismatch)"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: hint = " (kernel crash — prior illegal memory access or bad pointer)"; break;
            case CUBLAS_STATUS_ALLOC_FAILED: hint = " (GPU OOM)"; break;
            case CUBLAS_STATUS_NOT_INITIALIZED: hint = " (handle not initialized)"; break;
            default: break;
        }
        throw std::runtime_error(std::string("cuBLAS SGEMM failed: ") + op_name + 
            " status=" + std::to_string(static_cast<int>(status)) + hint);
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

// Destroy module-static GPU resources (cleanup stream only).
// cuBLAS handle is owned by TrainingState/InferenceState and destroyed there.
// Call during process shutdown after all GPU work is complete.
void shutdownAutogradResources() {
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
        std::ostringstream oss; \
        oss << "CUDA error in TensorContract: " << cudaGetErrorString(err) \
            << " (code=" << static_cast<int>(err) << ")" \
            << " | call=" << #call \
            << " | file=" << __FILE__ << ":" << __LINE__ \
            << " | " << getCurrentGradFnContext(); \
        throw TensorContract::ContractViolation(oss.str()); \
    } \
} while(0)

// Grid dimensions for 1D kernels: cap grid.x at 65535 to avoid cudaErrorInvalidValue.
inline dim3 gridFor1D(size_t n, int block_size) {
    if (n == 0) return dim3(1, 1, 1);
    const int blocks = static_cast<int>((n + block_size - 1) / block_size);
    constexpr int kMax = 65535;
    if (blocks <= kMax) return dim3(blocks, 1, 1);
    return dim3(kMax, (blocks + kMax - 1) / kMax, 1);
}

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
        throw std::runtime_error(buildCudaAllocFailureMessage("cudaMalloc", name ? name : "TensorBuffer", bytes, err));
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
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_scale(const float* __restrict__ src,
                             float alpha,
                             float* __restrict__ dst,
                             size_t n) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
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
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
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
    kernel_zero<<<gridFor1D(n, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(tensor.ptr, n);
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
    kernel_add<<<gridFor1D(n, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(a.ptr, b.ptr, dst.ptr, n);
    TC_CUDA_CHECK(cudaGetLastError());
}

void scale(const TensorView& src, float alpha, TensorView& dst, cudaStream_t stream) {
    validate_conversion(src, dst, "scale");
    
    size_t n = src.size_elements();
    kernel_scale<<<gridFor1D(n, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(src.ptr, alpha, dst.ptr, n);
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
constexpr int kMaxGridBlocks1DFallback = 65534;  // Fallback when device query fails or reports 65535 (some drivers reject exactly 65535)

// Returns the device's max blocks per grid dimension (x). Queried at runtime via cudaDeviceGetAttribute.
// Cached per process; uses fallback if query fails (e.g. before CUDA init).
inline int getMaxGridBlocks1D() {
    static int cached = -1;
    if (cached < 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cached = kMaxGridBlocks1DFallback;
            fprintf(stderr, "[gridForCount] cudaGetDevice failed, using maxGridBlocks1D=%d\n", cached);
        } else {
            int max_x = 0;
            if (cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device) != cudaSuccess) {
                cached = kMaxGridBlocks1DFallback;
                fprintf(stderr, "[gridForCount] cudaDeviceGetAttribute failed, using maxGridBlocks1D=%d\n", cached);
            } else {
                cached = (max_x > 65534) ? 65534 : max_x;
                fprintf(stderr, "[gridForCount] device %d maxGridDimX=%d, using maxGridBlocks1D=%d\n", device, max_x, cached);
            }
        }
    }
    return cached;
}

// CUDA limits: gridDim.y and gridDim.z are 65535 (2^16-1) on all architectures.
constexpr int kMaxGridDimY = 65535;

// Returns grid dimensions for count elements; uses 2D grid when blocks exceeds device max per dimension.
// Ensures grid.y <= kMaxGridDimY to avoid cudaErrorInvalidValue (invalid argument).
inline dim3 gridForCount(size_t count) {
    // CUDA kernel launches with grid.x == 0 are invalid.
    // Treat empty tensors as a legal no-op launch configuration.
    if (count == 0) {
        return dim3(1, 1, 1);
    }
    const int blocks = static_cast<int>((count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE);
    const int max1d = getMaxGridBlocks1D();
    if (blocks <= max1d)
        return dim3(blocks, 1, 1);
    int gy = (blocks + max1d - 1) / max1d;
    if (gy <= kMaxGridDimY)
        return dim3(max1d, gy, 1);
    // grid.y would exceed 65535: switch to grid.x = ceil(blocks/65535), grid.y = 65535
    int gx = (blocks + kMaxGridDimY - 1) / kMaxGridDimY;
    return dim3(gx, kMaxGridDimY, 1);
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
    
    cudaMallocOrThrow(reinterpret_cast<void**>(&t.data), bytes, name ? name : "Tensor::zeros");
    t.owns_data = true;
    TENSOR_LOG_LIFECYCLE(alloc_counter,
        "[Tensor::alloc] #A%d cudaMalloc data=%p bytes=%zu name=%s\n",
        (void*)t.data, bytes, name ? name : "unnamed");
    
    // Zero-initialize using cudaMemsetAsync (avoids kernel launch grid limits for large tensors).
    // 0.0f has all-zero bytes, so byte-wise memset is correct.
    cudaError_t err = cudaMemsetAsync(t.data, 0, bytes, stream);
    if (err != cudaSuccess) {
        cudaFree(t.data);
        t.data = nullptr;
        throw std::runtime_error(std::string("Tensor::zeros cudaMemsetAsync failed for ") +
                                 (name ? name : "unnamed") + ": " + cudaGetErrorString(err) +
                                 " (count=" + std::to_string(count) + ")");
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
    
    cudaMallocOrThrow(reinterpret_cast<void**>(&t.data), bytes, name ? name : "Tensor::empty");
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
    
    cudaMallocOrThrow(reinterpret_cast<void**>(&t.data), bytes, name ? name : "Tensor::xavier_uniform");
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
//======================================================

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
    cudaMallocOrThrow(reinterpret_cast<void**>(&ptr), bytes, name ? name : "ensure_grad");
    
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

Tensor Tensor::detach(cudaStream_t s) const {
    Tensor t = detach();
    t.stream = s;
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
            auto poll_start = std::chrono::steady_clock::now();
            cudaError_t query;
            while ((query = cudaStreamQuery(stream)) == cudaErrorNotReady) {
                auto elapsed = std::chrono::steady_clock::now() - poll_start;
                if (elapsed > std::chrono::seconds(10)) {
                    std::string ctx = getCurrentGradFnContext();
                    cudaError_t err = cudaGetLastError();
                    std::string msg = "[Tensor::backward] TIMEOUT: Stream stuck after 10s! last_error=" +
                        std::string(cudaGetErrorString(err)) + " | " + ctx;
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

// SiLU forward: y = x * sigmoid(x)
__global__ void kernel_silu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float x = input[idx];
        const float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

// SiLU backward: grad_x = grad_y * silu'(x)
// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__global__ void kernel_silu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float x = input[idx];
        const float sig = 1.0f / (1.0f + expf(-x));
        const float dsilu = sig * (1.0f + x * (1.0f - sig));
        grad_input[idx] = grad_output[idx] * dsilu;
    }
}

// Exp forward: y = exp(x)
__global__ void kernel_exp_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = expf(input[idx]);
    }
}

// Exp backward: grad_x = grad_y * y  (uses saved output, not input)
__global__ void kernel_exp_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_output,
    float* __restrict__ grad_input,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] += grad_output[idx] * saved_output[idx];
    }
}

// Add scalar forward: y = x + scalar
__global__ void kernel_add_scalar_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] + scalar;
    }
}

// Reciprocal forward: y = 1/x
__global__ void kernel_reciprocal_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = 1.0f / input[idx];
    }
}

// Reciprocal backward: grad_x = grad_y * (-y²)  (uses saved output)
__global__ void kernel_reciprocal_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_output,
    float* __restrict__ grad_input,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float y = saved_output[idx];
        grad_input[idx] += grad_output[idx] * (-y * y);
    }
}

// Mul-scalar forward: y = x * scalar  (element-wise multiply by constant)
__global__ void kernel_mul_scalar_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] * scalar;
    }
}

// Mul-scalar backward: grad_x += grad_y * scalar
__global__ void kernel_mul_scalar_backward(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    float scalar,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] += grad_output[idx] * scalar;
    }
}

// Element-wise multiply forward: output = a ⊙ b
__global__ void kernel_elementwise_mul_forward(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = a[idx] * b[idx];
    }
}

// Broadcast row multiply forward: out[i,j] = scale[i,0] * x[i,j]
__global__ void kernel_broadcast_row_mul_forward(
    const float* __restrict__ scale,   // [rows, 1]
    const float* __restrict__ x,       // [rows, cols]
    float* __restrict__ output,        // [rows, cols]
    int rows, int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int row = idx / cols;
    output[idx] = scale[row] * x[idx];
}

// Broadcast row multiply backward for x: grad_x[i,j] += grad_out[i,j] * scale[i,0]
__global__ void kernel_broadcast_row_mul_backward_x(
    const float* __restrict__ grad_output,  // [rows, cols]
    const float* __restrict__ scale,        // [rows, 1]
    float* __restrict__ grad_x,             // [rows, cols]
    int rows, int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int row = idx / cols;
    grad_x[idx] += grad_output[idx] * scale[row];
}

// Broadcast row multiply backward for scale: grad_scale[i] += sum_j(grad_out[i,j] * x[i,j])
__global__ void kernel_broadcast_row_mul_backward_scale(
    const float* __restrict__ grad_output,  // [rows, cols]
    const float* __restrict__ x,            // [rows, cols]
    float* __restrict__ grad_scale,         // [rows, 1]
    int rows, int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        sum += grad_output[row * cols + j] * x[row * cols + j];
    // Warp reduction
    for (int mask = warpSize / 2; mask > 0; mask >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, mask);
    // Lane 0 of EVERY warp contributes (not just threadIdx.x == 0)
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(&grad_scale[row], sum);
}

// Element-wise multiply backward for input A: grad_a = grad_output ⊙ b
__global__ void kernel_elementwise_mul_backward(
    const float* grad_output,
    const float* other,
    float* grad_self,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_self[idx] = grad_output[idx] * other[idx];
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

//========================================================================
// Softmax Forward (with optional temperature scaling)
// p[i] = exp(x[i]/T - max) / sum_j exp(x[j]/T - max)
// One block per row.
//========================================================================
__global__ void kernel_softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int tokens, int dim, float inv_temperature
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* row = input  + static_cast<size_t>(token_idx) * dim;
    float* out_row   = output + static_cast<size_t>(token_idx) * dim;

    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_val;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_max = fmaxf(local_max, row[i] * inv_temperature);
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) m = fmaxf(m, s_warp[w]);
        s_val = m;
    }
    __syncthreads();
    const float max_val = s_val;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float e = expf(row[i] * inv_temperature - max_val);
        out_row[i] = e;
        local_sum += e;
    }
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_val = 1.0f / (s + 1e-7f);
    }
    __syncthreads();
    const float inv_sum = s_val;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] *= inv_sum;
}

//========================================================================
// Softmax Backward
// grad_x[i] += (1/T) * p[i] * (grad_y[i] - dot(grad_y, p))
//========================================================================
__global__ void kernel_softmax_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ softmax_out,
    float* __restrict__ grad_input,
    int tokens, int dim, float inv_temperature
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* dy = grad_output + static_cast<size_t>(token_idx) * dim;
    const float* p  = softmax_out + static_cast<size_t>(token_idx) * dim;
    float* dx       = grad_input  + static_cast<size_t>(token_idx) * dim;

    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_dot;

    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_dot += dy[i] * p[i];
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_dot += __shfl_down_sync(0xffffffff, local_dot, off);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_dot;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_dot = s;
    }
    __syncthreads();
    const float dot = s_dot;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dx[i] += p[i] * (dy[i] - dot) * inv_temperature;
}

//========================================================================
// Concat Forward: out[i, 0:D1] = a[i,:], out[i, D1:D1+D2] = b[i,:]
//========================================================================
__global__ void kernel_concat_forward(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int N, int D1, int D2
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int D = D1 + D2;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        if (j < D1)
            output[static_cast<size_t>(row) * D + j] = a[static_cast<size_t>(row) * D1 + j];
        else
            output[static_cast<size_t>(row) * D + j] = b[static_cast<size_t>(row) * D2 + (j - D1)];
    }
}

__global__ void kernel_concat_backward_a(
    float* __restrict__ grad_a,
    const float* __restrict__ grad_out,
    int N, int D1, int D_total
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D1; j += blockDim.x)
        grad_a[static_cast<size_t>(row) * D1 + j] += grad_out[static_cast<size_t>(row) * D_total + j];
}

__global__ void kernel_concat_backward_b(
    float* __restrict__ grad_b,
    const float* __restrict__ grad_out,
    int N, int D1, int D2, int D_total
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D2; j += blockDim.x)
        grad_b[static_cast<size_t>(row) * D2 + j] += grad_out[static_cast<size_t>(row) * D_total + (D1 + j)];
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), element_count * sizeof(float), "LayerScaleGradFn_input_grad");
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
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), element_count * sizeof(float), "LayerScaleGradFn_input_data");
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
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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
    bool input_is_leaf = false;       // April 2026: leaf parameter accumulation path
    float* leaf_grad_buf = nullptr;   // direct pointer to leaf parameter's .grad data

    // ISSUE #56 FIX: Owned gradient buffer for non-leaf tensors / leaf staging buffer
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

        // April 2026: Support row-centering of LEAF parameters (e.g. LM head weight matrix
        // for the "constrain Σ_d W[v,d]=0" architecture). For leaves, input.grad_fn is null,
        // so the non-leaf chain path below would silently drop gradients. We must
        // accumulate the centered gradient into the leaf's persistent .grad buffer instead.
        input_is_leaf = input.is_leaf;
        if (input_is_leaf) {
            leaf_grad_buf = input.grad_data();
            if (!leaf_grad_buf) {
                throw std::runtime_error(
                    "CenterRowsGradFn::capture_input: leaf input has requires_grad but "
                    "grad_data() is NULL after ensure_grad() at " __FILE__);
            }
        }

        // CRITICAL FIX (Issue #136): NEVER reuse externally-owned leaf buffers!
        // set_grad_from_buffer() marks the gradient tensor as is_leaf=true,
        // but it's wrapping an externally-owned buffer (grad_logits_tensor.data).
        // If we reuse this buffer, CenterRowsGradFn OVERWRITES it with centered gradients,
        // corrupting the original CE gradients that LogSoftmaxGradFn wrote.
        // Always allocate our own staging buffer so we don't destroy upstream data;
        // for leaf inputs this also serves as the centered-grad scratch that we then
        // accumulate (+=) into the leaf's persistent grad buffer.
        float* buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "CenterRowsGradFn_input_grad");
        cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
        owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
        input_grad = owned_input_grad.get();
        AG_TRACE("[CenterRowsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p (leaf=%d)\n", element_count, (void*)input_grad, (int)input_is_leaf);
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

        if (input_is_leaf) {
            // Accumulate centered gradient into leaf parameter's persistent .grad buffer.
            // MUST be += (not =) so grad-accumulation windows work and so we don't clobber
            // gradients from other autograd paths into the same parameter.
            if (!leaf_grad_buf) {
                throw std::runtime_error("CenterRowsGradFn::apply: leaf_grad_buf is NULL for leaf input at " __FILE__);
            }
            kernel_accumulate_grad<<<gridForCount(element_count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                leaf_grad_buf, input_grad, element_count, 1.0f);
        }

        // Continue backward chain (non-leaf inputs only — leaves have grad_fn=null)
        if (input_grad_fn) {
            Tensor input_grad_tensor;
            input_grad_tensor.data = input_grad;
            input_grad_tensor.shape = input_shape;
            input_grad_tensor.owns_data = false;
            input_grad_tensor.stream = stream;

            input_grad_fn->apply(input_grad_tensor, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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
        
        // Setup gradient buffer
        input.ensure_grad();
        
        // CRITICAL FIX (Issue #136 pattern): NEVER reuse externally-owned leaf buffers!
        // kernel_center_columns OVERWRITES the output buffer with centered gradients.
        // If we write into the leaf's persistent grad buffer, we corrupt upstream data
        // (same bug CenterRowsGradFn had). Always allocate our own buffer.
        float* buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "CenterColumnsGradFn_input_grad");
        cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
        owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
        input_grad = owned_input_grad.get();
        AG_TRACE("[CenterColumnsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p\n", element_count, (void*)input_grad);
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
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
        }
    }
    
    __host__ void release_saved() override {
        owned_input_grad.reset();
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// PC1 PROJECTION KERNELS — Issue #149
// g is RMS-normalized (g·g = D), so the projection coefficient is (h·g)/D, not (h·g).
// Forward:  h̃[t] = h[t] - (h[t]·g / D) * g   (g = PC1 via power iteration, stop-grad)
// Backward: grad_h += (I - gg^T/D) * grad_h̃  (same projection, ACCUMULATES; g constant)
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
// If ||g||² is below a numerical floor (degenerate / near-zero H), substitutes a
// canonical unit vector g = (1,0,…,0) so downstream power iteration cannot blow up.
// Launch: <<<1, 256, 0, stream>>>  (requires blockDim.x == 256)
static __global__ void kernel_pc1_normalize(float* __restrict__ g, int D)
{
    assert(blockDim.x == 256);  // sdata[256] reduction is hard-coded
    __shared__ float sdata[256];
    __shared__ float s_inv;
    __shared__ int   s_degenerate;
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += g[d] * g[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float sum_sq = sdata[0];
        // Floor in absolute terms — well above fp32 denormals, well below any
        // realistic ||g||² for non-collapsed H. If we trip it we substitute a
        // canonical direction rather than producing a 1e6-magnitude g from noise.
        const float kFloor = 1e-20f;
        s_degenerate = (sum_sq < kFloor) ? 1 : 0;
        s_inv = (sum_sq < kFloor) ? 0.f : (1.f / sqrtf(sum_sq / (float)D));
    }
    __syncthreads();
    if (s_degenerate) {
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            g[d] = (d == 0) ? 1.f : 0.f;
    } else {
        const float inv = s_inv;
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            g[d] *= inv;
    }
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

// H_out[t,d] = H[t,d] - (H[t,:]·g / D) * g[d]  — project each row (overwrite)
// With RMS-normalized g (rms(g)=1 → g·g=D), proper projection is (h·g)/(g·g) * g = (h·g)/D * g
// Used in FORWARD pass (writes into a fresh output buffer).
// Launch: <<<T, 256, 0, stream>>>  (one block per row for shared-mem dot product)
// Requires blockDim.x == 256 (shared-mem reduction width is hard-coded).
static __global__ void kernel_pc1_project(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ H_out, int T, int D)
{
    assert(blockDim.x == 256);  // sdata[256] reduction is hard-coded
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
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        H_out[(size_t)t * D + d] = H[(size_t)t * D + d] - coeff * g[d];
}

// H_out[t,d] += H[t,d] - (H[t,:]·g / D) * g[d]  — same projection but ACCUMULATES.
// Used in BACKWARD: grad_h += (I - gg^T/D) * grad_h̃ so multiple GradFns sharing
// a leaf parameter's persistent grad buffer compose correctly (grads must accumulate,
// never overwrite — Rule 20).
// Launch: <<<T, 256, 0, stream>>>  with blockDim.x == 256.
static __global__ void kernel_pc1_project_accum(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ H_out, int T, int D)
{
    assert(blockDim.x == 256);  // sdata[256] reduction is hard-coded
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
    float coeff = sdata[0] / (float)D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t idx = (size_t)t * D + d;
        H_out[idx] += H[idx] - coeff * g[d];
    }
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

    // Takes ownership of g_device (must be a cudaMalloc'd [cols]-float buffer).
    // Caller MUST NOT free g_device after this call — the GradFn owns it.
    __host__ void capture_input(Tensor& input, int rows, int cols,
                                float* g_device, cudaStream_t stream) {
        // ── Edge-case validation (Rule 20) ──
        if (g_device == nullptr)
            throw std::runtime_error("ProjectOutPC1GradFn::capture_input: g_device is NULL");
        if (rows <= 0 || cols <= 0)
            throw std::runtime_error("ProjectOutPC1GradFn::capture_input: invalid dims rows=" +
                                     std::to_string(rows) + " cols=" + std::to_string(cols));
        if (input.numel() != (std::size_t)rows * (std::size_t)cols)
            throw std::runtime_error("ProjectOutPC1GradFn::capture_input: input.numel()=" +
                                     std::to_string(input.numel()) + " != rows*cols=" +
                                     std::to_string((std::size_t)rows * (std::size_t)cols));

        input_requires_grad = input.requires_grad;
        input_shape = input.shape;
        element_count = input.numel();
        num_rows = rows;
        num_cols = cols;
        (void)stream;

        // Take ownership of g (stop-gradient direction). Avoids a per-step cudaMalloc
        // + D2D copy on the hot path.
        owned_g.reset(g_device, [](float* p) { queueForDeferredCleanup(p); });
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
        if (grad_output.data == nullptr)
            throw std::runtime_error("ProjectOutPC1GradFn::apply: grad_output.data is NULL");
        if (grad_output.numel() != element_count)
            throw std::runtime_error("ProjectOutPC1GradFn::apply: grad_output.numel()=" +
                                     std::to_string(grad_output.numel()) +
                                     " != captured element_count=" + std::to_string(element_count));
        applied = true;

        // Allocate grad buffer on-demand for non-leaf inputs (deferred from capture_input)
        if (!input_grad) {
            float* buf = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "ProjectOutPC1GradFn_deferred_input_grad");
            cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
            owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
            AG_TRACE("[ProjectOutPC1GradFn] Allocated deferred input_grad buffer: %zu floats at %p\n", element_count, (void*)input_grad);
        }

        // BACKWARD: grad_h += (I - gg^T/D) * grad_h̃  — same projection as forward.
        // ACCUMULATES so leaf parameters (whose grad buffer persists across multiple
        // graph branches in an accumulation window) compose correctly. The non-leaf
        // path above zero-inits input_grad first, so accumulating into zero is
        // equivalent to assigning.
        kernel_pc1_project_accum<<<num_rows, 256, 0, stream>>>(
            grad_output.data, g_saved, input_grad, num_rows, num_cols);

        if (input_grad_fn) {
            Tensor input_grad_tensor;
            input_grad_tensor.data = input_grad;
            input_grad_tensor.shape = input_shape;
            input_grad_tensor.owns_data = false;
            input_grad_tensor.stream = stream;

            input_grad_fn->apply(input_grad_tensor, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_a), a_numel * sizeof(float), "AddGradFn_grad_a");
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_b), b_numel * sizeof(float), "AddGradFn_grad_b");
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_a), a_numel * sizeof(float), "AddGradFn_single_grad_a");
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
        // For c = a + b: dc/da = 1, dc/db = 1, so upstream receives grad_output unchanged.
        // grad_a/grad_b are local accumulators (leaf buffers or owned intermediates) —
        // the chain must propagate the raw flowing gradient, not the accumulated buffer.
        if (a_requires_grad && a_grad_fn && a_grad_fn->op_name) {
            Tensor view;
            view.data = grad_output.data; view.shape = a_shape;
            view.owns_data = false; view.stream = stream;
            a_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here!
        }
        
        if (b_requires_grad && b_grad_fn && b_grad_fn != a_grad_fn && b_grad_fn->op_name) {
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_input), input_numel * sizeof(float), "BiasAddGradFn_grad_input");
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_bias), features * sizeof(float), "BiasAddGradFn_grad_bias");
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "GeluGradFn_input_grad");
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
        cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), size * sizeof(float), "GeluGradFn_cache");
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
 * SiluGradFn - Backward for SiLU activation
 * Forward: y = silu(x) = x * sigmoid(x)
 * Backward: grad_x = grad_y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 *
 * MEMORY OPTIMIZATION: Uses non-owning cache reference instead of copying.
 * Safe because ForwardIntermediates (Issue #56) guarantees the source tensor
 * persists until after backward completes.
 */
struct SiluGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_input = nullptr;
    size_t cached_size = 0;

    SiluGradFn() { op_name = "silu"; }

    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                input_grad = x.grad_data();
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "SiluGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void set_cache_ref(const float* data, size_t size) {
        if (!data) {
            throw std::runtime_error("SiluGradFn::set_cache_ref: data is NULL");
        }
        cached_input = data;
        cached_size = size;
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("silu", this);
        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;
        if (!input_grad) {
            throw std::runtime_error("SiluGradFn::apply: input_grad is NULL");
        }
        if (!cached_input) {
            throw std::runtime_error("SiluGradFn::apply: cached_input is NULL");
        }

        const size_t count = grad_output.numel();
        if (count != cached_size) {
            throw std::runtime_error("SiluGradFn::apply: size mismatch");
        }

        kernel_silu_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_input, input_grad, count);
        trackKernelLaunch("kernel_silu_backward", stream);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
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
 * ExpGradFn - Backward for element-wise exp
 * Forward: y = exp(x)
 * Backward: grad_x = grad_y * y  (uses saved output)
 *
 * Saves output (not input) — more memory-efficient since we need y anyway.
 */
struct ExpGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_output = nullptr;
    size_t cached_size = 0;

    ExpGradFn() { op_name = "exp"; }

    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                input_grad = x.grad_data();
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "ExpGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void save_output(const float* output_data, size_t size) {
        if (!output_data) {
            throw std::runtime_error("ExpGradFn::save_output: output_data is NULL");
        }
        cached_output = output_data;
        cached_size = size;
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("exp", this);
        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;
        if (!input_grad) {
            throw std::runtime_error("ExpGradFn::apply: input_grad is NULL");
        }
        if (!cached_output) {
            throw std::runtime_error("ExpGradFn::apply: cached_output is NULL");
        }

        const size_t count = grad_output.numel();
        if (count != cached_size) {
            throw std::runtime_error("ExpGradFn::apply: size mismatch");
        }

        kernel_exp_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_output, input_grad, count);
        trackKernelLaunch("kernel_exp_backward", stream);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        cached_output = nullptr;
        cached_size = 0;
        input_grad = nullptr;
        input_grad_fn.reset();
    }
};

/**
 * AddScalarGradFn - Backward for adding a constant scalar
 * Forward: y = x + c
 * Backward: grad_x = grad_y  (pure pass-through, constant has no gradient)
 */
struct AddScalarGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    size_t count = 0;

    AddScalarGradFn() { op_name = "add_scalar"; }

    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;
        count = x.numel();

        if (input_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                input_grad = x.grad_data();
            } else {
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), count * sizeof(float), "AddScalarGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, count * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("add_scalar", this);
        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;
        if (!input_grad) {
            throw std::runtime_error("AddScalarGradFn::apply: input_grad is NULL");
        }

        // Pure pass-through: grad_x += grad_y
        const size_t n = grad_output.numel();
        if (n != count) {
            throw std::runtime_error("AddScalarGradFn::apply: size mismatch");
        }
        kernel_accumulate_grad<<<gridForCount(n), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            input_grad, grad_output.data, n, 1.0f);
        trackKernelLaunch("add_scalar_backward", stream);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        input_grad = nullptr;
        input_grad_fn.reset();
    }
};

/**
 * ReciprocalGradFn - Backward for element-wise reciprocal
 * Forward: y = 1/x
 * Backward: grad_x = grad_y * (-y²)  (uses saved output)
 */
struct ReciprocalGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_output = nullptr;
    size_t cached_size = 0;

    ReciprocalGradFn() { op_name = "reciprocal"; }

    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                input_grad = x.grad_data();
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "ReciprocalGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void save_output(const float* output_data, size_t size) {
        if (!output_data) {
            throw std::runtime_error("ReciprocalGradFn::save_output: output_data is NULL");
        }
        cached_output = output_data;
        cached_size = size;
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("reciprocal", this);
        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;
        if (!input_grad) {
            throw std::runtime_error("ReciprocalGradFn::apply: input_grad is NULL");
        }
        if (!cached_output) {
            throw std::runtime_error("ReciprocalGradFn::apply: cached_output is NULL");
        }

        const size_t count = grad_output.numel();
        if (count != cached_size) {
            throw std::runtime_error("ReciprocalGradFn::apply: size mismatch");
        }

        kernel_reciprocal_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_output, input_grad, count);
        trackKernelLaunch("kernel_reciprocal_backward", stream);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        cached_output = nullptr;
        cached_size = 0;
        input_grad = nullptr;
        input_grad_fn.reset();
    }
};

/**
 * MulScalarGradFn - Backward for element-wise multiply by constant
 * Forward: y = x * scalar
 * Backward: grad_x = grad_y * scalar
 */
struct MulScalarGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    float scalar = 0.0f;
    size_t count = 0;

    MulScalarGradFn() { op_name = "mul_scalar"; }

    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                input_grad = x.grad_data();
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "MulScalarGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("mul_scalar", this);
        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;
        if (!input_grad) {
            throw std::runtime_error("MulScalarGradFn::apply: input_grad is NULL");
        }

        kernel_mul_scalar_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, input_grad, scalar, count);
        trackKernelLaunch("kernel_mul_scalar_backward", stream);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        input_grad = nullptr;
        input_grad_fn.reset();
    }
};

/**
 * BroadcastRowMulGradFn - Backward for broadcast per-row scalar multiply
 * Forward:  out[i,j] = scale[i,0] * x[i,j]
 * Backward: grad_scale[i] += sum_j(grad_out[i,j] * x[i,j])
 *           grad_x[i,j]   += grad_out[i,j] * scale[i,0]
 */
struct BroadcastRowMulGradFn : public GradFn {
    bool scale_requires_grad = false;
    bool x_requires_grad = false;
    float* scale_grad = nullptr;
    float* x_grad = nullptr;
    std::shared_ptr<float> owned_scale_grad;
    std::shared_ptr<float> owned_x_grad;
    TensorContract::TensorShape scale_shape;
    TensorContract::TensorShape x_shape;
    std::shared_ptr<GradFn> scale_grad_fn;
    std::shared_ptr<GradFn> x_grad_fn;
    const float* cached_scale = nullptr;
    const float* cached_x = nullptr;
    int rows = 0;
    int cols = 0;

    BroadcastRowMulGradFn() { op_name = "broadcast_row_mul"; }

    void capture_inputs(Tensor& s, Tensor& x, cudaStream_t stream) {
        scale_requires_grad = s.requires_grad;
        x_requires_grad = x.requires_grad;
        scale_shape = s.shape;
        x_shape = x.shape;
        scale_grad_fn = s.grad_fn;
        x_grad_fn = x.grad_fn;

        if (scale_requires_grad) {
            if (s.is_leaf) {
                s.ensure_grad();
                scale_grad = s.grad_data();
            } else {
                const size_t s_numel = s.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), s_numel * sizeof(float), "BroadcastRowMulGradFn_scale_grad");
                cudaMemsetAsync(buffer, 0, s_numel * sizeof(float), stream);
                owned_scale_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                scale_grad = owned_scale_grad.get();
            }
        }
        if (x_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                x_grad = x.grad_data();
            } else {
                const size_t x_numel = x.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "BroadcastRowMulGradFn_x_grad");
                cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
                owned_x_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                x_grad = owned_x_grad.get();
            }
        }
    }

    void set_cache_refs(const float* scale_data, const float* x_data, int r, int c) {
        if (!scale_data) throw std::runtime_error("BroadcastRowMulGradFn::set_cache_refs: scale_data is NULL");
        if (!x_data) throw std::runtime_error("BroadcastRowMulGradFn::set_cache_refs: x_data is NULL");
        cached_scale = scale_data;
        cached_x = x_data;
        rows = r;
        cols = c;
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("broadcast_row_mul", this);
        if (applied) return;
        applied = true;

        const int total = rows * cols;

        if (x_requires_grad) {
            if (!x_grad) throw std::runtime_error("BroadcastRowMulGradFn::apply: x_grad is NULL");
            kernel_broadcast_row_mul_backward_x<<<(total + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, cached_scale, x_grad, rows, cols);
            trackKernelLaunch("kernel_broadcast_row_mul_backward_x", stream);
        }

        if (scale_requires_grad) {
            if (!scale_grad) throw std::runtime_error("BroadcastRowMulGradFn::apply: scale_grad is NULL");
            kernel_broadcast_row_mul_backward_scale<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, cached_x, scale_grad, rows, cols);
            trackKernelLaunch("kernel_broadcast_row_mul_backward_scale", stream);
        }

        if (x_requires_grad && x_grad_fn) {
            Tensor view;
            view.data = x_grad; view.shape = x_shape;
            view.owns_data = false; view.stream = stream;
            x_grad_fn->apply(view, stream);
        }
        if (scale_requires_grad && scale_grad_fn) {
            Tensor view;
            view.data = scale_grad; view.shape = scale_shape;
            view.owns_data = false; view.stream = stream;
            scale_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        cached_scale = nullptr;
        cached_x = nullptr;
        scale_grad = nullptr;
        x_grad = nullptr;
        scale_grad_fn.reset();
        x_grad_fn.reset();
    }
};

/**
 * ZeroPadGradFn - Backward for row-offset zero-padding
 * Forward: result = zeros(total_rows, cols); result[offset:offset+rows, :] = x
 * Backward: grad_x += grad_result[offset:offset+rows, :]
 *
 * Used to place per-batch-row deltas at the correct offset in a full-size
 * [total_tokens, dm] tensor before autograd::add with layer_output.
 */
struct ZeroPadGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    size_t input_count = 0;
    size_t offset_elements = 0;  // row_offset * cols

    ZeroPadGradFn() { op_name = "zero_pad"; }

    void capture_input(Tensor& x, cudaStream_t stream, size_t offset_elems) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;
        input_count = x.numel();
        offset_elements = offset_elems;

        if (input_requires_grad) {
            if (x.is_leaf) {
                x.ensure_grad();
                input_grad = x.grad_data();
            } else {
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), input_count * sizeof(float), "ZeroPadGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, input_count * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("zero_pad", this);
        if (applied) return;
        applied = true;

        if (!input_requires_grad) return;
        if (!input_grad) {
            throw std::runtime_error("ZeroPadGradFn::apply: input_grad is NULL");
        }

        // grad_x += grad_output[offset_elements : offset_elements + input_count]
        kernel_accumulate_grad<<<gridForCount(input_count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            input_grad, grad_output.data + offset_elements, input_count, 1.0f);
        trackKernelLaunch("zero_pad_backward", stream);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        input_grad = nullptr;
        input_grad_fn.reset();
    }
};

/**
 * ElementwiseMulGradFn - Backward for element-wise multiply (Hadamard product)
 * Forward: y = a ⊙ b
 * Backward: grad_a = grad_y ⊙ b, grad_b = grad_y ⊙ a
 *
 * MEMORY OPTIMIZATION: Uses non-owning cache references instead of copying.
 * Safe because ForwardIntermediates (Issue #56) guarantees both input tensors
 * persist until after backward completes.
 */
struct ElementwiseMulGradFn : public GradFn {
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    float* a_grad = nullptr;
    float* b_grad = nullptr;
    std::shared_ptr<float> owned_a_grad;
    std::shared_ptr<float> owned_b_grad;
    TensorContract::TensorShape a_shape;
    TensorContract::TensorShape b_shape;
    std::shared_ptr<GradFn> a_grad_fn;
    std::shared_ptr<GradFn> b_grad_fn;

    const float* cached_a = nullptr;
    const float* cached_b = nullptr;
    size_t cached_size = 0;

    ElementwiseMulGradFn() { op_name = "elementwise_mul"; }

    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_shape = a.shape;
        b_shape = b.shape;
        a_grad_fn = a.grad_fn;
        b_grad_fn = b.grad_fn;

        if (a_requires_grad) {
            if (a.is_leaf) {
                a.ensure_grad();
                a_grad = a.grad_data();
            } else {
                const size_t n = a.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ElementwiseMulGradFn_grad_a");
                cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
                owned_a_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                a_grad = owned_a_grad.get();
            }
        }
        if (b_requires_grad) {
            if (b.is_leaf) {
                b.ensure_grad();
                b_grad = b.grad_data();
            } else {
                const size_t n = b.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ElementwiseMulGradFn_grad_b");
                cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
                owned_b_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                b_grad = owned_b_grad.get();
            }
        }
    }

    void set_cache_refs(const float* a_data, const float* b_data, size_t size) {
        cached_size = size;
        if (a_requires_grad && b_data) cached_b = b_data;
        if (b_requires_grad && a_data) cached_a = a_data;
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("elementwise_mul", this);
        if (applied) return;
        applied = true;

        const size_t count = grad_output.numel();

        if (a_requires_grad) {
            if (!a_grad || !cached_b) {
                throw std::runtime_error("ElementwiseMulGradFn::apply: a_grad or cached_b is NULL");
            }
            kernel_elementwise_mul_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, cached_b, a_grad, count);
            trackKernelLaunch("kernel_elementwise_mul_backward_a", stream);

            if (a_grad_fn) {
                Tensor view;
                view.data = a_grad; view.shape = a_shape;
                view.owns_data = false; view.stream = stream;
                a_grad_fn->apply(view, stream);
            }
        }

        if (b_requires_grad) {
            if (!b_grad || !cached_a) {
                throw std::runtime_error("ElementwiseMulGradFn::apply: b_grad or cached_a is NULL");
            }
            kernel_elementwise_mul_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_output.data, cached_a, b_grad, count);
            trackKernelLaunch("kernel_elementwise_mul_backward_b", stream);

            if (b_grad_fn) {
                Tensor view;
                view.data = b_grad; view.shape = b_shape;
                view.owns_data = false; view.stream = stream;
                b_grad_fn->apply(view, stream);
            }
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        cached_a = nullptr;
        cached_b = nullptr;
        cached_size = 0;
        a_grad = nullptr;
        b_grad = nullptr;
        a_grad_fn.reset();
        b_grad_fn.reset();
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
    std::shared_ptr<GradFn> input_grad_fn;
    // ISSUE #51 FIX: Own a copy of cached data instead of non-owning pointer
    std::shared_ptr<float> owned_cache;
    std::shared_ptr<float> owned_input_grad;
    const float* cached_input = nullptr;  // Points to owned_cache.get()
    size_t cached_size = 0;
    int d_model = 0;
    float eps = 1e-5f;
    
    RMSNormGradFn() { op_name = "rms_norm"; }
    
    ~RMSNormGradFn() {
        release_saved();
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
            float* buf = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buf), grad_size * sizeof(float), "RMSNormGradFn_input_grad");
            cudaMemsetAsync(buf, 0, grad_size * sizeof(float), stream);
            owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
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
        cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), size * sizeof(float), "RMSNormGradFn_cache");
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
        const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32) * sizeof(float);
        
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
        owned_input_grad.reset();
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
        release_saved();
    }
    
    void capture_weight(Tensor& w) {
        weight_requires_grad = w.requires_grad;
        weight_shape = w.shape;
        vocab_size = w.shape.as_2d().rows;
        if (!vocab_size) {
            throw std::runtime_error("EmbeddingGradFn::capture_weight: vocab_size is 0 — weight shape is invalid");
        }
        
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
            cudaMallocOrThrow(reinterpret_cast<void**>(&token_ids), tokens * sizeof(int), "EmbeddingGradFn_token_ids");
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
        
        // PyTorch-style direct accumulation — embedding grad writes
        // to same buffer where LM head grad already lives. Natural ~90% cancellation
        // for frequent tokens acts as frequency-proportional regularization.
        kernel_embedding_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, token_ids, weight_grad, num_tokens, d_model, vocab_size, embedding_scale);
        trackKernelLaunch("kernel_embedding_backward", stream);
        
        // DEBUG: Capture embedding gradient
        if (g_autograd_verbose && g_debug_embedding_only_grad && weight_grad) {
            const size_t total_size = weight_shape.total_elements();
            debugCaptureEmbeddingGrad(weight_grad, total_size, stream);
        }

        // CONTINUE AUTOGRAD CHAIN using stored grad_fn
        if (weight_grad_fn) {
            Tensor view;
            view.data = weight_grad; view.shape = weight_shape;
            view.owns_data = false; view.stream = stream;
            weight_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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
        release_saved();
    }

    void capture_input(Tensor& x) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;

        // Copy shared_ptr to captured grad_fn
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            x.ensure_grad();
            if (x.is_leaf) {
                // Leaf tensors (weights) persist — safe to use their grad buffer directly
                input_grad = x.grad_data();
                owns_input_grad = false;
            } else {
                // Non-leaf tensors are temporaries destroyed before backward —
                // must allocate owned buffer to avoid dangling pointer
                const size_t bytes = x.shape.total_elements() * sizeof(float);
                cudaMallocOrThrow(reinterpret_cast<void**>(&input_grad), bytes, "LogSoftmaxGradFn_input_grad");
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
            cudaMallocOrThrow(reinterpret_cast<void**>(&saved_log_softmax), bytes, "LogSoftmaxGradFn_saved");
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
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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
        release_saved();
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
            cudaMallocOrThrow(reinterpret_cast<void**>(&input_grad), grad_size * sizeof(float), "DropoutGradFn_input_grad");
            cudaMemset(input_grad, 0, grad_size * sizeof(float));
            owns_input_grad = true;
        }
    }
    
    void save(const uint8_t* mask, float dropout_prob, size_t n, cudaStream_t stream) {
        count = n;
        scale = (dropout_prob < 1.0f) ? 1.0f / (1.0f - dropout_prob) : 0.0f;
        
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_mask), n * sizeof(uint8_t), "DropoutGradFn_saved_mask");
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
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "ResidualAddGradFn_input_grad");
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
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), r_numel * sizeof(float), "ResidualAddGradFn_residual_grad");
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
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
        }
        if (residual_requires_grad && residual_grad_fn && residual_grad_fn != input_grad_fn) {
            Tensor view;
            view.data = residual_grad; view.shape = residual_shape;
            view.owns_data = false; view.stream = stream;
            residual_grad_fn->apply(view, stream);
            // ISSUE #52 FIX: Do NOT call release_saved() here — cudaFree blocks while GPU busy
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

// ScaleScalarGradFn: backward for scale_scalar; passes scale * grad_output to input
struct ScaleScalarGradFn : public GradFn {
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    float scale = 0.0f;
    ScaleScalarGradFn() { op_name = "scale_scalar"; }
    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!input_grad_fn || !input_grad_fn->op_name) return;
        if (!grad_output.data || grad_output.numel() < 1) return;
        float h_grad = 0.0f;
        cudaMemcpyAsync(&h_grad, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        const float scaled = scale * h_grad;
        float* d_scaled_raw = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_scaled_raw), sizeof(float), "ScaleScalarGradFn_d_scaled");
        std::shared_ptr<float> d_scaled_guard(d_scaled_raw, [](float* p) { queueForDeferredCleanup(p); });
        cudaMemcpyAsync(d_scaled_raw, &scaled, sizeof(float), cudaMemcpyHostToDevice, stream);
        Tensor view;
        view.data = d_scaled_raw;
        view.shape = input_shape;
        view.owns_data = false;
        view.stream = stream;
        input_grad_fn->apply(view, stream);
    }
};

Tensor scale_scalar(const Tensor& t, float scale, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::scale_scalar: stream is NULL");
    }
    if (t.numel() != 1) {
        throw std::invalid_argument("autograd::scale_scalar: input must be scalar (1 element), got " + std::to_string(t.numel()));
    }
    float h_val = 0.0f;
    cudaMemcpyAsync(&h_val, t.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const float scaled_val = scale * h_val;
    float* d_out = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_out), sizeof(float), "scale_scalar_d_out");
    cudaMemcpyAsync(d_out, &scaled_val, sizeof(float), cudaMemcpyHostToDevice, stream);
    Tensor result;
    result.data = d_out;
    result.owns_data = true;
    result.shape = TensorContract::TensorShape::make_BSM(1, 1);
    result.is_leaf = false;
    result.requires_grad = t.requires_grad;
    result.stream = stream;
    if (t.requires_grad && t.grad_fn) {
        auto grad_fn = std::make_shared<ScaleScalarGradFn>();
        grad_fn->input_grad_fn = t.grad_fn;
        grad_fn->input_shape = t.shape;
        grad_fn->scale = scale;
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
 * autograd::silu - SiLU (Swish) activation with automatic differentiation
 *
 * SiLU(x) = x * sigmoid(x)
 *
 * TAPE-BASED: Requires caller to provide cache pointer.
 * Does NOT allocate internal copy of input.
 *
 * @param x Input tensor
 * @param stream CUDA stream
 * @param input_cache External cache pointer for input (needed for backward).
 *                    Must point to valid memory until backward() is called.
 * @return Output tensor with SiLU applied
 */
Tensor silu(const Tensor& x, cudaStream_t stream, const float* input_cache) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::silu: stream is NULL - caller MUST provide valid stream");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "silu_result");

    const size_t count = x.numel();
    kernel_silu_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<SiluGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);

        const float* effective_cache = input_cache ? input_cache : x.data;
        grad_fn->set_cache_ref(effective_cache, count);
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::exp - Element-wise exponential with automatic differentiation
 *
 * Forward: y = exp(x)
 * Backward: grad_x = grad_y * y
 *
 * Saves output (not input) for backward — d/dx exp(x) = exp(x) = y.
 */
Tensor exp(const Tensor& x, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::exp: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::exp: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "exp_result");
    const size_t count = x.numel();
    kernel_exp_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ExpGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save_output(result.data, count);
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::add_scalar - Add a constant scalar to every element
 *
 * Forward: y = x + c
 * Backward: grad_x = grad_y  (pure pass-through)
 *
 * The scalar is a constant with no gradient.
 */
Tensor add_scalar(const Tensor& x, float scalar, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::add_scalar: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::add_scalar: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "add_scalar_result");
    const size_t count = x.numel();
    kernel_add_scalar_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, scalar, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<AddScalarGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::reciprocal - Element-wise reciprocal with automatic differentiation
 *
 * Forward: y = 1/x
 * Backward: grad_x = grad_y * (-y²)
 *
 * Saves output for backward — d/dx (1/x) = -1/x² = -y².
 */
Tensor reciprocal(const Tensor& x, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::reciprocal: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::reciprocal: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "reciprocal_result");
    const size_t count = x.numel();
    kernel_reciprocal_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ReciprocalGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save_output(result.data, count);
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::mul_scalar - Multiply every element by a constant
 *
 * Forward: y = x * scalar
 * Backward: grad_x = grad_y * scalar
 *
 * @param x Input tensor (any shape)
 * @param scalar Constant multiplier
 * @param stream CUDA stream
 * @return Scaled tensor with autograd tracking
 */
Tensor mul_scalar(const Tensor& x, float scalar, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::mul_scalar: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::mul_scalar: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "mul_scalar_result");
    const size_t count = x.numel();
    kernel_mul_scalar_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, scalar, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<MulScalarGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->scalar = scalar;
        grad_fn->count = count;
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::broadcast_row_mul - Broadcast per-row scalar multiply
 *
 * Forward:  out[i,j] = scale[i,0] * x[i,j]
 * Backward: grad_scale[i] += sum_j(grad_out[i,j] * x[i,j])
 *           grad_x[i,j]   += grad_out[i,j] * scale[i,0]
 *
 * @param scale Per-row scalars [rows, 1]
 * @param x     Input tensor [rows, cols]
 * @param stream CUDA stream
 * @return Broadcast-scaled tensor [rows, cols]
 */
Tensor broadcast_row_mul(const Tensor& scale, const Tensor& x, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::broadcast_row_mul: stream is NULL");
    }
    if (!scale.data || !x.data) {
        throw std::runtime_error("autograd::broadcast_row_mul: input data is NULL");
    }
    const auto s_dims = scale.shape.as_2d();
    const auto x_dims = x.shape.as_2d();
    if (s_dims.cols != 1) {
        throw std::runtime_error("autograd::broadcast_row_mul: scale must be [rows,1], got cols=" + std::to_string(s_dims.cols));
    }
    if (s_dims.rows != x_dims.rows) {
        throw std::runtime_error("autograd::broadcast_row_mul: row mismatch scale.rows=" + std::to_string(s_dims.rows) + " x.rows=" + std::to_string(x_dims.rows));
    }
    const int rows = x_dims.rows;
    const int cols = x_dims.cols;
    const int total = rows * cols;

    const bool needs_grad = scale.requires_grad || x.requires_grad;
    Tensor result = Tensor::empty(x.shape, needs_grad, stream, "brow_mul_result");

    kernel_broadcast_row_mul_forward<<<(total + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        scale.data, x.data, result.data, rows, cols);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<BroadcastRowMulGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(scale), const_cast<Tensor&>(x), stream);
        grad_fn->set_cache_refs(scale.data, x.data, rows, cols);
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::zero_pad - Place a [rows, cols] tensor at row_offset in a zero-padded
 * [total_rows, cols] result.
 *
 * Forward: result = zeros(total_rows, cols); result[row_offset : row_offset+rows, :] = x
 * Backward: grad_x += grad_result[row_offset : row_offset+rows, :]
 *
 * No custom kernel needed — forward uses cudaMemsetAsync + cudaMemcpyAsync,
 * backward uses existing kernel_accumulate_grad on the correct slice.
 */
Tensor zero_pad(const Tensor& x, int row_offset, int total_rows, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::zero_pad: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::zero_pad: input data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("autograd::zero_pad: input must be a 2D layout [rows, cols], got non-2D layout");
    }
    const int row_tokens = x.shape.flat.rows;
    const int cols = x.shape.flat.cols;
    if (row_offset < 0 || row_offset + row_tokens > total_rows) {
        throw std::runtime_error("autograd::zero_pad: offset=" + std::to_string(row_offset) +
                                 " + rows=" + std::to_string(row_tokens) +
                                 " > total_rows=" + std::to_string(total_rows));
    }

    Tensor result = Tensor::zeros(TensorContract::TensorShape::make_BSM(total_rows, cols), x.requires_grad, stream, "zero_pad_result");

    const size_t offset_elements = static_cast<size_t>(row_offset) * cols;
    const size_t slice_bytes = static_cast<size_t>(row_tokens) * cols * sizeof(float);
    cudaMemcpyAsync(result.data + offset_elements, x.data, slice_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ZeroPadGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream, offset_elements);
        result.grad_fn = grad_fn;
    }

    return result;
}

/**
 * autograd::elementwise_mul - Element-wise (Hadamard) product with automatic differentiation
 *
 * Forward: y = a ⊙ b
 * Backward: grad_a = grad_y ⊙ b, grad_b = grad_y ⊙ a
 *
 * @param a First input tensor
 * @param b Second input tensor (same shape as a)
 * @param stream CUDA stream
 * @return Output tensor with element-wise product
 */
Tensor elementwise_mul(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::elementwise_mul: stream is NULL");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("autograd::elementwise_mul: size mismatch a.numel()=" +
                                 std::to_string(a.numel()) + " b.numel()=" + std::to_string(b.numel()));
    }

    const bool needs_grad = a.requires_grad || b.requires_grad;
    Tensor result = Tensor::empty(a.shape, needs_grad, stream, "emul_result");

    const size_t count = a.numel();
    kernel_elementwise_mul_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        a.data, b.data, result.data, count);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ElementwiseMulGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        grad_fn->set_cache_refs(a.data, b.data, count);
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
    const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32) * sizeof(float);
    
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

    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument(
            "autograd::dropout: dropout probability p must be in [0, 1), got " +
            std::to_string(p));
    }
    
    const size_t count = x.numel();
    if (count == 0) {
        // Empty tensor: dropout is a no-op, keep gradient identity edge.
        if (x.requires_grad) {
            result.is_leaf = false;
            auto grad_fn = std::make_shared<AddGradFn>();
            grad_fn->capture_single_input(const_cast<Tensor&>(x), stream);
            result.grad_fn = grad_fn;
        }
        return result;
    }
    const float scale = 1.0f / (1.0f - p);
    
    // Allocate mask on device
    uint8_t* mask = nullptr;
    TC_CUDA_CHECK(cudaMalloc(&mask, count * sizeof(uint8_t)));
    if (!x.data || !result.data || !mask) {
        throw std::runtime_error("autograd::dropout: null buffer(s) before kernel launch");
    }
    
    // Generate random mask: 1 = keep (with prob 1-p), 0 = drop (with prob p)
    kernel_generate_dropout_mask<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        mask, count, p, seed);
    trackKernelLaunch("kernel_generate_dropout_mask", stream);
    TC_CUDA_CHECK(cudaGetLastError());
    
    // Forward: y = x * mask * scale
    kernel_dropout_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, mask, result.data, scale, count);
    trackKernelLaunch("kernel_dropout_forward", stream);
    TC_CUDA_CHECK(cudaGetLastError());
    
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
    // ── Edge-case validation (Rule 20: fail loud, no silent fallbacks) ──
    if (x.numel() == 0)
        throw std::runtime_error("project_out_pc1: input tensor is empty");
    if (x.data == nullptr)
        throw std::runtime_error("project_out_pc1: input tensor data is NULL");
    if (!x.shape.is_2d_layout())
        throw std::runtime_error("project_out_pc1: expected 2D (flat) tensor [T, D]");
    if (n_power_iters < 0)
        throw std::runtime_error("project_out_pc1: n_power_iters must be >= 0, got " + std::to_string(n_power_iters));

    const int D = (int)x.shape.flat.cols;
    const int T = (int)(x.numel() / (std::size_t)D);
    if (D <= 0 || T <= 0)
        throw std::runtime_error("project_out_pc1: invalid dimensions T=" + std::to_string(T) + " D=" + std::to_string(D));
    // T==1 and D==1 are degenerate: with a single row, g is collinear with H[0,:]
    // and the projection produces all-zero output, which silently zeros the LM-head
    // input. With D==1 the only direction is g itself — same outcome. Both indicate
    // a caller bug (wrong shape / micro-batch=1 row), not a recoverable runtime state.
    if (T < 2)
        throw std::runtime_error("project_out_pc1: requires T >= 2 (got T=" + std::to_string(T) + "); projection of a single row collapses to zero");
    if (D < 2)
        throw std::runtime_error("project_out_pc1: requires D >= 2 (got D=" + std::to_string(D) + "); projection in 1D collapses to zero");

    bool track_grad = x.requires_grad;

    // Working buffers on device
    float* g_buf  = nullptr;  // [D]
    float* g_tmp  = nullptr;  // [D]
    float* v_buf  = nullptr;  // [T]
    cudaMallocOrThrow(reinterpret_cast<void**>(&g_buf), D * sizeof(float), "pc1_g_buf");
    cudaMallocOrThrow(reinterpret_cast<void**>(&g_tmp), D * sizeof(float), "pc1_g_tmp");
    cudaMallocOrThrow(reinterpret_cast<void**>(&v_buf), T * sizeof(float), "pc1_v_buf");

    // Initialize PC1 guess from column mean, then normalize
    kernel_pc1_col_mean<<<1, 256, 0, stream>>>(x.data, g_buf, T, D);
    kernel_pc1_normalize<<<1, 256, 0, stream>>>(g_buf, D);

    // Power iteration: g ← normalize(H^T (H g))
    // Ping-pong g_buf <-> g_tmp instead of D2D-copying every iter.
    const int blk = 256;
    for (int iter = 0; iter < n_power_iters; ++iter) {
        kernel_pc1_gemv_Hg<<<(T + blk - 1) / blk, blk, 0, stream>>>(x.data, g_buf, v_buf, T, D);
        kernel_pc1_gemv_HtV<<<(D + blk - 1) / blk, blk, 0, stream>>>(x.data, v_buf, g_tmp, T, D);
        kernel_pc1_normalize<<<1, 256, 0, stream>>>(g_tmp, D);
        std::swap(g_buf, g_tmp);  // freshly-normalized g is now in g_buf
    }
    // Single sync after all PC1 kernels — needed before host-side autograd capture
    cudaError_t pc1_err = cudaStreamSynchronize(stream);
    if (pc1_err != cudaSuccess) throw std::runtime_error("project_out_pc1: PC1 kernels failed: " + std::string(cudaGetErrorString(pc1_err)));

    // Allocate output buffer — avoids in-place mutation of the input tensor.
    // The input's data and autograd metadata remain untouched.
    float* out_data = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&out_data), (std::size_t)T * D * sizeof(float), "pc1_output");

    // Build result tensor with its own data
    Tensor result;
    result.data      = out_data;
    result.shape     = x.shape;
    result.owns_data = true;
    result.requires_grad = track_grad;
    result.is_leaf   = false;
    result.stream    = stream;

    // Project: H_out[t,d] = H[t,d] - (H[t,:]·g / D) * g[d]  (g is RMS-normalized, g·g=D)
    kernel_pc1_project<<<T, 256, 0, stream>>>(x.data, g_buf, out_data, T, D);

    // Free scratch buffers we no longer need. g_buf is either (a) handed to the
    // GradFn below (track_grad==true), or (b) freed here.
    cudaFreeAsync(v_buf, stream);
    cudaFreeAsync(g_tmp, stream);

    if (track_grad) {
        // Transfer ownership of g_buf to the GradFn — no per-step malloc/D2D copy.
        auto grad_fn = std::make_shared<ProjectOutPC1GradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), T, D, g_buf, stream);
        result.grad_fn = grad_fn;
    } else {
        cudaFreeAsync(g_buf, stream);
    }

    return result;
}


//========================================================================
// SoftmaxGradFn
//========================================================================
struct SoftmaxGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    float* saved_softmax = nullptr;
    int num_tokens = 0;
    int dim = 0;
    float inv_temperature = 1.0f;

    SoftmaxGradFn() { op_name = "softmax"; }

    ~SoftmaxGradFn() override {
        release_saved();
    }

    void capture_input(Tensor& x, cudaStream_t stream) {
        input_requires_grad = x.requires_grad;
        input_shape = x.shape;
        input_grad_fn = x.grad_fn;

        if (input_requires_grad) {
            x.ensure_grad();
            if (x.is_leaf) {
                input_grad = x.grad_data();
            } else {
                const size_t n = x.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "SoftmaxGradFn_input_grad");
                cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
                owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                input_grad = owned_input_grad.get();
            }
        }
    }

    void save(const float* softmax_output, int tokens_, int dim_, float inv_temp, cudaStream_t stream) {
        num_tokens = tokens_;
        dim = dim_;
        inv_temperature = inv_temp;
        const size_t bytes = static_cast<size_t>(tokens_) * dim_ * sizeof(float);
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_softmax), bytes, "SoftmaxGradFn_saved");
        cudaMemcpyAsync(saved_softmax, softmax_output, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("softmax", this);
        if (applied) return;
        applied = true;
        if (!input_requires_grad) return;
        if (!saved_softmax || !input_grad) {
            throw std::runtime_error("SoftmaxGradFn::apply: saved data or grad buffer is NULL");
        }

        kernel_softmax_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, saved_softmax, input_grad, num_tokens, dim, inv_temperature);

        if (input_grad_fn) {
            Tensor view;
            view.data = input_grad; view.shape = input_shape;
            view.owns_data = false; view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_softmax) { cudaFree(saved_softmax); saved_softmax = nullptr; }
        input_grad = nullptr;
        input_grad_fn.reset();
    }
};

Tensor softmax(const Tensor& x, float temperature, cudaStream_t stream) {
    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::softmax: input must be 2D [tokens, dim]");
    }
    if (!x.data) {
        throw std::invalid_argument("autograd::softmax: input data is NULL");
    }
    if (temperature < 1e-6f) temperature = 1e-6f;
    const float inv_temp = 1.0f / temperature;

    const auto dims = x.shape.as_2d();
    const int num_tokens = dims.rows;
    const int dim = dims.cols;

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "softmax_result");

    kernel_softmax_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_tokens, dim, inv_temp);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<SoftmaxGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save(result.data, num_tokens, dim, inv_temp, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

//========================================================================
// ConcatGradFn
//========================================================================
struct ConcatGradFn : public GradFn {
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    float* grad_a = nullptr;
    float* grad_b = nullptr;
    std::shared_ptr<float> owned_grad_a;
    std::shared_ptr<float> owned_grad_b;
    TensorContract::TensorShape a_shape;
    TensorContract::TensorShape b_shape;
    std::shared_ptr<GradFn> a_grad_fn;
    std::shared_ptr<GradFn> b_grad_fn;
    int rows = 0;
    int D1 = 0;
    int D2 = 0;

    ConcatGradFn() { op_name = "concat"; }

    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
        a_requires_grad = a.requires_grad;
        b_requires_grad = b.requires_grad;
        a_shape = a.shape;
        b_shape = b.shape;
        a_grad_fn = a.grad_fn;
        b_grad_fn = b.grad_fn;

        if (a_requires_grad) {
            a.ensure_grad();
            if (a.is_leaf) {
                grad_a = a.grad_data();
            } else {
                const size_t n = a.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ConcatGradFn_grad_a");
                cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
                owned_grad_a = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                grad_a = owned_grad_a.get();
            }
        }
        if (b_requires_grad) {
            b.ensure_grad();
            if (b.is_leaf) {
                grad_b = b.grad_data();
            } else {
                const size_t n = b.numel();
                float* buffer = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ConcatGradFn_grad_b");
                cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
                owned_grad_b = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
                grad_b = owned_grad_b.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("concat", this);
        if (applied) return;
        applied = true;

        const int D_total = D1 + D2;

        if (a_requires_grad && grad_a) {
            kernel_concat_backward_a<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_a, grad_output.data, rows, D1, D_total);
            if (a_grad_fn) {
                Tensor view;
                view.data = grad_a; view.shape = a_shape;
                view.owns_data = false; view.stream = stream;
                a_grad_fn->apply(view, stream);
            }
        }
        if (b_requires_grad && grad_b) {
            kernel_concat_backward_b<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                grad_b, grad_output.data, rows, D1, D2, D_total);
            if (b_grad_fn && b_grad_fn != a_grad_fn) {
                Tensor view;
                view.data = grad_b; view.shape = b_shape;
                view.owns_data = false; view.stream = stream;
                b_grad_fn->apply(view, stream);
            }
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

Tensor concat(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (!a.shape.is_2d_layout() || !b.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::concat: both inputs must be 2D");
    }
    const auto a_dims = a.shape.as_2d();
    const auto b_dims = b.shape.as_2d();
    if (a_dims.rows != b_dims.rows) {
        throw std::invalid_argument("autograd::concat: row count mismatch (a=" +
            std::to_string(a_dims.rows) + " b=" + std::to_string(b_dims.rows) + ")");
    }
    if (!a.data || !b.data) {
        throw std::invalid_argument("autograd::concat: null data pointer");
    }

    const int N = a_dims.rows;
    const int D1 = a_dims.cols;
    const int D2 = b_dims.cols;

    const bool needs_grad = a.requires_grad || b.requires_grad;
    auto shape = TensorContract::TensorShape::make_BSM(N, D1 + D2);
    Tensor result = Tensor::empty(shape, needs_grad, stream, "concat_result");

    kernel_concat_forward<<<N, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        result.data, a.data, b.data, N, D1, D2);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ConcatGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        grad_fn->rows = N;
        grad_fn->D1 = D1;
        grad_fn->D2 = D2;
        result.grad_fn = grad_fn;
    }

    return result;
}

// ─── Cross-Entropy from Logits ───────────────────────────────────────────────

// Forward kernel: CE = log(sum(exp(z - z_max))) + z_max - z[target]
// Single-thread: num_classes ≤ 64 for selector/execution heads
__global__ void kernel_ce_logits_forward(
    const float* __restrict__ logits,   // [C]
    float* __restrict__ ce_out,         // [1]
    float* __restrict__ saved_probs,    // [C] softmax output, saved for backward
    int C, int target_idx)
{
    if (threadIdx.x != 0) return;

    // Stable log-sum-exp
    float z_max = logits[0];
    for (int i = 1; i < C; ++i) z_max = fmaxf(z_max, logits[i]);

    float sum_exp = 0.0f;
    for (int i = 0; i < C; ++i) sum_exp += expf(logits[i] - z_max);
    sum_exp = fmaxf(sum_exp, 1e-10f);

    ce_out[0] = logf(sum_exp) + z_max - logits[target_idx];

    // Save softmax probs for backward
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < C; ++i) {
        saved_probs[i] = expf(logits[i] - z_max) * inv_sum;
    }
}

// Backward kernel: d_logits[i] = grad_output * (softmax[i] - 1_{i == target})
// One thread per class.
__global__ void kernel_ce_logits_backward(
    const float* __restrict__ grad_output,  // [1]
    const float* __restrict__ saved_probs,  // [C]
    float* __restrict__ grad_logits,        // [C] accumulate
    int C, int target_idx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    float target_indicator = (i == target_idx) ? 1.0f : 0.0f;
    atomicAdd(&grad_logits[i], grad_output[0] * (saved_probs[i] - target_indicator));
}

struct CrossEntropyLogitsGradFn : public GradFn {
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    float* saved_probs = nullptr;     // Owned: [C] softmax probs
    bool owns_saved = true;
    int C = 0;
    int target_idx = 0;

    // Grad buffer for logits (non-leaf: owned)
    std::shared_ptr<float> owned_grad_logits;
    float* grad_logits = nullptr;
    bool input_is_leaf = false;

    CrossEntropyLogitsGradFn() { op_name = "cross_entropy_logits"; }

    ~CrossEntropyLogitsGradFn() {
        release_saved();
    }

    void capture_input(Tensor& logits, cudaStream_t stream) {
        input_grad_fn = logits.grad_fn;
        input_shape = logits.shape;
        C = logits.shape.as_2d().cols;
        input_is_leaf = logits.is_leaf;

        if (logits.requires_grad) {
            logits.ensure_grad();
            if (logits.is_leaf) {
                grad_logits = logits.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), static_cast<size_t>(C) * sizeof(float), "CELogitsGradFn_grad_logits");
                cudaMemsetAsync(buf, 0, static_cast<size_t>(C) * sizeof(float), stream);
                owned_grad_logits = std::shared_ptr<float>(buf, [](float* p) {
                    queueForDeferredCleanup(p);
                });
                grad_logits = owned_grad_logits.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        if (!saved_probs) {
            throw std::runtime_error("CrossEntropyLogitsGradFn::apply: saved_probs is NULL");
        }
        if (!grad_logits) {
            throw std::runtime_error("CrossEntropyLogitsGradFn::apply: grad_logits is NULL");
        }

        const int threads = (C < 256) ? C : 256;
        const int blocks = (C + threads - 1) / threads;
        kernel_ce_logits_backward<<<blocks, threads, 0, stream>>>(
            grad_output.data, saved_probs, grad_logits, C, target_idx);

        // Chain upstream
        if (input_grad_fn && input_grad_fn->op_name) {
            Tensor view;
            view.data = grad_logits;
            view.shape = input_shape;
            view.owns_data = false;
            view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (owns_saved && saved_probs) {
            queueForDeferredCleanup(saved_probs);
        }
        saved_probs = nullptr;
        grad_logits = nullptr;
        input_grad_fn.reset();
    }
};

Tensor cross_entropy_logits(const Tensor& logits, int target_idx, cudaStream_t stream) {
    if (!logits.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::cross_entropy_logits: logits must be 2D");
    }
    const auto dims = logits.shape.as_2d();
    if (dims.rows != 1) {
        throw std::invalid_argument("autograd::cross_entropy_logits: logits must be [1, C], got rows=" +
                                     std::to_string(dims.rows));
    }
    const int C = dims.cols;
    if (target_idx < 0 || target_idx >= C) {
        throw std::invalid_argument("autograd::cross_entropy_logits: target_idx=" +
                                     std::to_string(target_idx) + " out of range [0, " +
                                     std::to_string(C) + ")");
    }
    if (!logits.data) {
        throw std::invalid_argument("autograd::cross_entropy_logits: logits.data is NULL");
    }

    auto out_shape = TensorContract::TensorShape::make_BSM(1, 1);
    Tensor result = Tensor::zeros(out_shape, logits.requires_grad, stream, "ce_logits_result");

    // Allocate saved_probs buffer for backward
    float* saved_probs = nullptr;
    if (logits.requires_grad) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_probs), static_cast<size_t>(C) * sizeof(float), "ce_logits_saved_probs");
    }

    // Forward: single-thread kernel (C ≤ 64 for selector/execution)
    kernel_ce_logits_forward<<<1, 32, 0, stream>>>(
        logits.data, result.data, saved_probs ? saved_probs : result.data,
        C, target_idx);

    if (logits.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CrossEntropyLogitsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(logits), stream);
        grad_fn->saved_probs = saved_probs;
        grad_fn->target_idx = target_idx;
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd

 }  // namespace GRIM
