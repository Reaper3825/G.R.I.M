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
#include <cassert>
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

}

}  // namespace GRIM
