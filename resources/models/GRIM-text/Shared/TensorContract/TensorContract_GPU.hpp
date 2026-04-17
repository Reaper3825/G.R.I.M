#pragma once
//======================================================//
//  TensorContract_GPU.hpp
//  Type-safe tensor abstraction + Native Autograd for CUDA
//======================================================//
//
//  PURPOSE
//  =======
//  This module provides:
//  1. Type-safe tensor abstraction with layout metadata
//  2. Native autograd system (PyTorch-style gradient tracking)
//  3. Parameter group management for optimizer integration
//
//  Eliminates the need to reason about layout, stride, transpose, and
//  storage order simultaneously. All tensor metadata is encapsulated
//  in TensorView, and conversion/validation is handled by the API.
//
//  DESIGN PRINCIPLES
//  =================
//  1. SINGLE SOURCE OF TRUTH: All layout conventions defined here
//  2. RUNTIME VALIDATION: All safety checks are runtime (not compile-time)
//     True compile-time layout safety would require templates or strong typedefs.
//  3. LAYOUT-AWARE SHAPES: 2D and 4D tensors are represented differently
//  4. ZERO-COPY WHERE POSSIBLE: Views are lightweight, conversions only when needed
//  5. CUDA-FIRST: All operations are GPU-accelerated
//  6. FAIL CLOSED: Unsafe operations throw, not return false silently
//  7. AUTOGRAD: Gradient tracking via computation graph (GradFn nodes)
//
//  AUTOGRAD DEBUG TOGGLE
//  =====================
//  Set g_autograd_verbose = true to trace autograd chain execution
//  (defined in TensorContract_GPU.cu)
extern bool g_autograd_verbose;

// ═══════════════════════════════════════════════════════════════════════════
// ISSUE #60: DEBUG GRADIENT ATTRIBUTION
// Global hooks for capturing gradient sources separately (tied-weight debugging)
// Set these from TrainingState to enable gradient attribution logging
// ═══════════════════════════════════════════════════════════════════════════
extern float* g_debug_lm_head_only_grad;   // Buffer to capture LM head backward contribution
extern float* g_debug_embedding_only_grad; // Buffer to capture embedding backward contribution  
extern size_t g_debug_grad_buffer_size;    // Size in elements (vocab_size * d_model)
extern bool g_debug_capture_enabled;       // Set to true to enable capturing


// Call these to capture gradient sources (called internally by GradFn::apply)
void debugCaptureLMHeadGrad(float* grad_ptr, size_t size, cudaStream_t stream);
void debugCaptureEmbeddingGrad(float* grad_ptr, size_t size, cudaStream_t stream);

//  TENSOR LAYOUTS
//  ==============
//  2D LAYOUTS (flat tensors):
//    BSM       = [tokens, d_model]         - Collapsed activation format
//    QKV_FUSED = [tokens, total_qkv_dim]   - Fused QKV projection output
//
//  4D LAYOUTS (attention tensors):
//    BHSD = [batch, heads, seq, head_dim]  - Standard attention format
//    BSHD = [batch, seq, heads, head_dim]  - Merged head format
//
//  GQA SUPPORT
//  ===========
//  For Grouped Query Attention:
//  - Q uses num_heads
//  - K, V use num_kv_heads (num_kv_heads <= num_heads)
//  - total_qkv_dim = d_model + 2 * kv_dim (canonical formula)
//
//  CUBLAS ROW-MAJOR CONVENTION
//  ===========================
//  cuBLAS assumes column-major storage. For row-major tensors:
//  - A row-major [M,K] is seen as column-major [K,M]^T
//  - To compute C = A @ B in row-major: swap A and B in cuBLAS call
//  - Or interpret as: C^T = B^T @ A^T
//
//  LIMITATIONS (future work)
//  =========================
//  - No stride model: all tensors assumed contiguous
//  - No sub-tensor views: slicing requires copy
//  - No blockwise kernel support without explicit stride handling
//
//======================================================//

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <string>
#include <algorithm>  // for std::max in merge_qkv_grads_gqa
#include <memory>     // for std::shared_ptr (ISSUE #59: grad as Tensor)
#include <tuple>      // for std::tuple (ISSUE #61: split_and_reshape_qkv return type)
#include <atomic>     // for tensor lifecycle counters

//======================================================//
//  Tensor Lifecycle Counters
//  Sequential IDs for every alloc/free/delete to trace memory lifecycle.
//  Each log line gets a monotonic counter so you can correlate across operations.
//  Define TENSOR_LIFECYCLE_LOGGING=1 to enable (adds overhead from fprintf + atomics).
//======================================================//
#ifndef TENSOR_LIFECYCLE_LOGGING
#define TENSOR_LIFECYCLE_LOGGING 0
#endif

struct TensorLifecycleCounters {
    static std::atomic<int> alloc_counter;   // Incremented on cudaMalloc (zeros/empty)
    static std::atomic<int> free_counter;    // Incremented on cudaFree (release)
    static std::atomic<int> gradfn_del_counter; // Incremented on grad_fn delete
    static std::atomic<int> move_counter;    // Incremented on move-construct/assign
};

#if TENSOR_LIFECYCLE_LOGGING
#define TENSOR_LOG_LIFECYCLE(counter, fmt, ...) \
    do { \
        const int _tlc_id = TensorLifecycleCounters::counter.fetch_add(1); \
        fprintf(stderr, fmt, _tlc_id, ##__VA_ARGS__); \
    } while(0)
#else
#define TENSOR_LOG_LIFECYCLE(counter, fmt, ...) ((void)0)
#endif

//======================================================//
//  ISSUE #53 FIX: Deferred GPU Memory Cleanup (GLOBAL SCOPE)
//  
//  Call flushDeferredCleanup() after cudaStreamSynchronize() to 
//  free GPU memory that was queued by shared_ptr deleters.
//  This prevents cudaFree from blocking when called during 
//  destructor chains while the GPU is still busy.
//  
//  NOTE: In global scope so lambdas in shared_ptr deleters can access them.
//======================================================//

/**
 * Queue a GPU pointer for deferred cleanup.
 * Called from shared_ptr deleters instead of cudaFree() directly.
 */
void queueForDeferredCleanup(void* ptr);

/**
 * Flush the deferred GPU memory cleanup queue.
 * Call this AFTER cudaStreamSynchronize() when it's safe to free GPU memory.
 */
void flushDeferredCleanup();

/**
 * Destroy all module-static autograd GPU resources (cleanup stream + cuBLAS handle).
 * Call during process shutdown after all GPU work is complete.
 */
void shutdownAutogradResources();

/**
 * RULE 20 Error Context: Track current GradFn operation for detailed error messages.
 * Call at start of each GradFn::apply() to record which operation is executing.
 */
void setCurrentGradFnOp(const char* op_name, void* gradfn_ptr);
void clearCurrentGradFnOp();
std::string getCurrentGradFnContext();

namespace TensorContract {

//======================================================//
//  Tensor Layout Enum
//======================================================//

enum class Layout : uint8_t {
    // 2D formats (flat tensors) - use Shape2D
    BSM,               // [tokens, d_model] - Collapsed activation format
    QKV_FUSED,         // [tokens, total_qkv_dim] - Fused QKV projection output
    LOGITS,            // [tokens, vocab_size] - Language model logits (output projections)
    
    // 4D formats (attention tensors) - use Shape4D
    BHSD,              // [batch, heads, seq, head_dim] - Standard attention format
    BSHD,              // [batch, seq, heads, head_dim] - Merged head format
    
    // Unknown/invalid
    UNKNOWN,
};

/**
 * Check if a layout is 2D (flat) or 4D (multi-dimensional)
 */
constexpr bool is_2d_layout(Layout l) {
    return l == Layout::BSM || l == Layout::QKV_FUSED || l == Layout::LOGITS;
}

constexpr bool is_4d_layout(Layout l) {
    return l == Layout::BHSD || l == Layout::BSHD;
}

//======================================================//
//  Shape Types - Separate 2D and 4D representations
//======================================================//

/**
 * Shape2D - For flat/collapsed tensors (BSM, QKV_FUSED)
 * 
 * These are TRUE 2D tensors, not 4D tensors pretending to have seq=1.
 */
struct Shape2D {
    int rows = 0;      // tokens (batch * seq collapsed)
    int cols = 0;      // d_model or total_qkv_dim
    
    size_t total_elements() const {
        return static_cast<size_t>(rows) * cols;
    }
    
    bool is_valid() const {
        return rows > 0 && cols > 0;
    }
    
    bool operator==(const Shape2D& other) const {
        return rows == other.rows && cols == other.cols;
    }
    bool operator!=(const Shape2D& other) const { return !(*this == other); }
};

/**
 * Shape4D - For multi-head attention tensors (BHSD, BHDS, BSHD)
 */
struct Shape4D {
    int batch = 0;
    int heads = 0;      // num_heads for Q, num_kv_heads for K/V in GQA
    int seq = 0;
    int head_dim = 0;
    
    // Derived dimensions
    int d_model() const { return heads * head_dim; }
    int tokens() const { return batch * seq; }
    
    size_t total_elements() const {
        return static_cast<size_t>(batch) * heads * seq * head_dim;
    }
    
    bool is_valid() const {
        return batch > 0 && heads > 0 && seq > 0 && head_dim > 0;
    }
    
    bool operator==(const Shape4D& other) const {
        return batch == other.batch && heads == other.heads &&
               seq == other.seq && head_dim == other.head_dim;
    }
    bool operator!=(const Shape4D& other) const { return !(*this == other); }
};

//======================================================//
//  TensorShape - Layout-aware discriminated union
//======================================================//

/**
 * TensorShape - Holds either Shape2D or Shape4D based on layout
 * 
 * This is NOT a 4D shape pretending to be 2D. It's a proper union
 * that represents the actual tensor dimensionality.
 */
struct TensorShape {
    Layout layout = Layout::UNKNOWN;
    
    union {
        Shape2D flat;   // For BSM, QKV_FUSED
        Shape4D multi;  // For BHSD, BHDS, BSHD
    };
    
    // Default constructor
    TensorShape() : layout(Layout::UNKNOWN), flat{0, 0} {}
    
    // Construct from 2D shape
    TensorShape(Layout l, Shape2D s) : layout(l), flat(s) {
        if (!TensorContract::is_2d_layout(l)) {
            layout = Layout::UNKNOWN;  // Invalid: 4D layout with 2D shape
        }
    }
    
    // Construct from 4D shape  
    TensorShape(Layout l, Shape4D s) : layout(l), multi(s) {
        if (!is_4d_layout(l)) {
            layout = Layout::UNKNOWN;  // Invalid: 2D layout with 4D shape
        }
    }
    
    // Query dimensionality
    bool is_2d_layout() const { return TensorContract::is_2d_layout(layout); }
    bool is_4d() const { return is_4d_layout(layout); }
    
    // Safe accessors (throw if wrong type)
    const Shape2D& as_2d() const {
        if (!is_2d_layout()) throw std::logic_error("TensorShape: not a 2D layout");
        return flat;
    }
    
    const Shape4D& as_4d() const {
        if (!is_4d()) throw std::logic_error("TensorShape: not a 4D layout");
        return multi;
    }
    
    Shape2D& as_2d() {
        if (!is_2d_layout()) throw std::logic_error("TensorShape: not a 2D layout");
        return flat;
    }
    
    Shape4D& as_4d() {
        if (!is_4d()) throw std::logic_error("TensorShape: not a 4D layout");
        return multi;
    }
    
    // Layout-aware element count
    size_t total_elements() const {
        if (is_2d_layout()) return flat.total_elements();
        if (is_4d()) return multi.total_elements();
        return 0;
    }
    
    // Layout-aware validation
    bool is_valid() const {
        if (layout == Layout::UNKNOWN) return false;
        if (is_2d_layout()) return flat.is_valid();
        if (is_4d()) return multi.is_valid();
        return false;
    }
    
    // RULE 20: Fail loud - throws if shape is invalid
    const TensorShape& require(const char* context) const {
        if (layout == Layout::UNKNOWN) {
            throw std::runtime_error(std::string(context) + ": TensorShape has UNKNOWN layout");
        }
        if (is_2d_layout() && !flat.is_valid()) {
            throw std::runtime_error(std::string(context) + ": TensorShape 2D is invalid (rows=" + 
                                     std::to_string(flat.rows) + ", cols=" + std::to_string(flat.cols) + ")");
        }
        if (is_4d() && !multi.is_valid()) {
            throw std::runtime_error(std::string(context) + ": TensorShape 4D is invalid (batch=" + 
                                     std::to_string(multi.batch) + ", heads=" + std::to_string(multi.heads) +
                                     ", seq=" + std::to_string(multi.seq) + ", head_dim=" + std::to_string(multi.head_dim) + ")");
        }
        return *this;
    }
    
    // Factory methods with explicit layout binding
    static TensorShape make_BSM(int tokens, int d_model) {
        return TensorShape(Layout::BSM, Shape2D{tokens, d_model});
    }
    
    static TensorShape make_QKV_FUSED(int tokens, int total_qkv_dim) {
        return TensorShape(Layout::QKV_FUSED, Shape2D{tokens, total_qkv_dim});
    }
    
    static TensorShape make_LOGITS(int tokens, int vocab_size) {
        return TensorShape(Layout::LOGITS, Shape2D{tokens, vocab_size});
    }
    
    static TensorShape make_BHSD(int batch, int heads, int seq, int head_dim) {
        return TensorShape(Layout::BHSD, Shape4D{batch, heads, seq, head_dim});
    }
    
    static TensorShape make_BSHD(int batch, int seq, int heads, int head_dim) {
        return TensorShape(Layout::BSHD, Shape4D{batch, heads, seq, head_dim});
    }
};

//======================================================//
//  TensorView - Lightweight non-owning tensor reference
//======================================================//

/**
 * TensorView - A non-owning view into a GPU tensor with layout metadata
 * 
 * This is the primary abstraction for passing tensors between functions.
 * It encapsulates:
 *   - Pointer to GPU memory
 *   - Layout-aware shape (2D or 4D)
 *   - Optional name for debugging
 * 
 * IMPORTANT: TensorView does NOT own the memory. The caller is responsible
 * for ensuring the underlying buffer outlives the view.
 */
struct TensorView {
    float* ptr = nullptr;
    TensorShape shape;
    const char* name = nullptr;  // Optional debug name
    
    // Constructors
    TensorView() = default;
    
    // Construct with pre-built shape
    TensorView(float* p, TensorShape s, const char* n = nullptr)
        : ptr(p), shape(s), name(n) {}
    
    // Convenience constructors for specific layouts
    static TensorView make_BSM(float* p, int tokens, int d_model, const char* n = nullptr) {
        return TensorView(p, TensorShape::make_BSM(tokens, d_model), n);
    }
    
    static TensorView make_QKV_FUSED(float* p, int tokens, int total_qkv_dim, const char* n = nullptr) {
        return TensorView(p, TensorShape::make_QKV_FUSED(tokens, total_qkv_dim), n);
    }
    
    static TensorView make_LOGITS(float* p, int tokens, int vocab_size, const char* n = nullptr) {
        return TensorView(p, TensorShape::make_LOGITS(tokens, vocab_size), n);
    }
    
    static TensorView make_BHSD(float* p, int batch, int heads, int seq, int head_dim, const char* n = nullptr) {
        return TensorView(p, TensorShape::make_BHSD(batch, heads, seq, head_dim), n);
    }
    
    // Query layout type
    Layout layout() const { return shape.layout; }
    bool is_2d_layout() const { return shape.is_2d_layout(); }
    bool is_4d() const { return shape.is_4d(); }
    
    // Validation
    bool is_valid() const {
        return ptr != nullptr && shape.is_valid();
    }
    
    // RULE 20: Fail loud - throws if view is invalid
    const TensorView& require(const char* context) const {
        if (!ptr) {
            throw std::runtime_error(std::string(context) + ": TensorView has NULL pointer" +
                                     (name ? std::string(" (name=") + name + ")" : ""));
        }
        shape.require(context);
        return *this;
    }
    
    // RULE 20: Returns mutable reference, throws if invalid
    TensorView& require_mut(const char* context) {
        if (!ptr) {
            throw std::runtime_error(std::string(context) + ": TensorView has NULL pointer" +
                                     (name ? std::string(" (name=") + name + ")" : ""));
        }
        shape.require(context);
        return *this;
    }
    
    // Size in bytes (layout-aware)
    size_t size_bytes() const {
        return shape.total_elements() * sizeof(float);
    }
    
    // Size in elements (layout-aware)
    size_t size_elements() const {
        return shape.total_elements();
    }
    
    // Debug string
    std::string to_string() const;
};

//======================================================//
//  TensorBuffer - Owning GPU tensor with automatic cleanup
//======================================================//

/**
 * TensorBuffer - RAII wrapper for GPU tensor memory
 * 
 * Use this when you need to allocate temporary GPU memory for
 * tensor operations. Memory is automatically freed on destruction.
 */
class TensorBuffer {
public:
    TensorBuffer() = default;
    ~TensorBuffer();
    
    // Non-copyable, movable
    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;
    TensorBuffer(TensorBuffer&& other) noexcept;
    TensorBuffer& operator=(TensorBuffer&& other) noexcept;
    
    // Allocation (TensorShape already contains the layout)
    bool allocate(TensorShape shape, const char* name = nullptr);
    void free();
    
    // Access
    TensorView view() const { return view_; }
    float* ptr() const { return view_.ptr; }
    bool is_allocated() const { return view_.ptr != nullptr; }
    Layout layout() const { return view_.layout(); }
    
    // Size
    size_t size_bytes() const { return view_.size_bytes(); }
    size_t size_elements() const { return view_.size_elements(); }
    
private:
    TensorView view_;
    bool owns_memory_ = false;
};

//======================================================//
//  Tensor Contract Validation
//======================================================//

/**
 * ContractViolation - Exception thrown when tensor contract is violated
 */
class ContractViolation : public std::runtime_error {
public:
    ContractViolation(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * Validate that two tensors can be used together in an operation
 * Checks: non-null, valid shapes, compatible dimensions
 * 
 * @param a First tensor
 * @param b Second tensor  
 * @param op_name Name of operation (for error messages)
 * @throws ContractViolation if validation fails
 */
void validate_binary_op(const TensorView& a, const TensorView& b, const char* op_name);

/**
 * Validate that source and destination are compatible for conversion
 * Checks: non-null, same total elements, no aliasing
 * 
 * @param src Source tensor
 * @param dst Destination tensor
 * @param op_name Name of operation (for error messages)
 * @throws ContractViolation if validation fails
 */
void validate_conversion(const TensorView& src, const TensorView& dst, const char* op_name);

/**
 * Check if two tensors alias (overlap in memory)
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return true if tensors overlap in memory
 */
bool tensors_alias(const TensorView& a, const TensorView& b);

//======================================================//
//  Common Tensor Operations
//======================================================//

/**
 * Zero a tensor buffer
 * 
 * @param tensor Tensor to zero
 * @param stream CUDA stream
 */
void zero(TensorView& tensor, cudaStream_t stream = nullptr);

/**
 * Copy tensor data (same layout)
 * 
 * @param src Source tensor
 * @param dst Destination tensor (must have same shape and layout)
 * @param stream CUDA stream
 */
void copy(const TensorView& src, TensorView& dst, cudaStream_t stream = nullptr);

/**
 * Add two tensors element-wise: dst = a + b
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @param dst Output tensor (can alias a or b for in-place)
 * @param stream CUDA stream
 */
void add(const TensorView& a, const TensorView& b, TensorView& dst, cudaStream_t stream = nullptr);

/**
 * Scale tensor by scalar: dst = alpha * src
 * 
 * @param src Input tensor
 * @param alpha Scale factor
 * @param dst Output tensor (can alias src for in-place)
 * @param stream CUDA stream
 */
void scale(const TensorView& src, float alpha, TensorView& dst, cudaStream_t stream = nullptr);

//======================================================//
//  GQA-Aware QKV Operations
//======================================================//

/**
 * QKV dimensions for GQA (Grouped Query Attention)
 * 
 * In GQA:
 *   - Q has full num_heads
 *   - K, V have reduced num_kv_heads
 *   - Each KV head is shared by (num_heads / num_kv_heads) Q heads
 */
struct GQADims {
    int num_heads;       // Q heads
    int num_kv_heads;    // K, V heads (num_kv_heads <= num_heads)
    int head_dim;        // Dimension per head
    
    // Derived dimensions
    int d_model() const { return num_heads * head_dim; }
    int q_dim() const { return num_heads * head_dim; }
    int kv_dim() const { return num_kv_heads * head_dim; }
    int total_qkv_dim() const { return q_dim() + 2 * kv_dim(); }  // Canonical formula
    int heads_per_kv_group() const { return num_heads / num_kv_heads; }
    
    // Validation
    bool is_valid() const {
        return num_heads > 0 && num_kv_heads > 0 && head_dim > 0 &&
               num_heads >= num_kv_heads && (num_heads % num_kv_heads) == 0;
    }
    
    // RULE 20: Fail loud - throws if dimensions are invalid
    const GQADims& require(const char* context) const {
        if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
            throw std::runtime_error(std::string(context) + ": GQADims has invalid dimensions (num_heads=" +
                                     std::to_string(num_heads) + ", num_kv_heads=" + std::to_string(num_kv_heads) +
                                     ", head_dim=" + std::to_string(head_dim) + ")");
        }
        if (num_heads < num_kv_heads) {
            throw std::runtime_error(std::string(context) + ": GQADims num_heads (" + std::to_string(num_heads) +
                                     ") must be >= num_kv_heads (" + std::to_string(num_kv_heads) + ")");
        }
        if (num_heads % num_kv_heads != 0) {
            throw std::runtime_error(std::string(context) + ": GQADims num_heads (" + std::to_string(num_heads) +
                                     ") must be divisible by num_kv_heads (" + std::to_string(num_kv_heads) + ")");
        }
        return *this;
    }
    
    bool is_mha() const { return num_heads == num_kv_heads; }  // Multi-head attention
    bool is_gqa() const { return num_heads > num_kv_heads; }   // Grouped query attention
};

/**
 * Split fused QKV tensor into separate Q, K, V tensors (GQA-aware)
 * 
 * @param qkv_fused Input: [tokens, total_qkv_dim] where total_qkv_dim = q_dim + 2*kv_dim
 * @param Q Output: [tokens, q_dim] or [batch, num_heads, seq, head_dim] (depends on target_layout)
 * @param K Output: [tokens, kv_dim] or [batch, num_kv_heads, seq, head_dim]
 * @param V Output: [tokens, kv_dim] or [batch, num_kv_heads, seq, head_dim]
 * @param gqa GQA dimensions
 * @param target_layout Desired output layout (BSM or BHSD)
 * @param stream CUDA stream
 */
void split_qkv_gqa(const TensorView& qkv_fused,
                   TensorView& Q, TensorView& K, TensorView& V,
                   const GQADims& gqa, Layout target_layout,
                   cudaStream_t stream = nullptr);

/**
 * Merge separate Q, K, V gradients back into fused format (GQA-aware)
 * 
 * @param grad_Q Input: Q gradients [batch, num_heads, seq, head_dim]
 * @param grad_K Input: K gradients [batch, num_kv_heads, seq, head_dim]
 * @param grad_V Input: V gradients [batch, num_kv_heads, seq, head_dim]
 * @param grad_qkv Output: Fused gradients [tokens, total_qkv_dim]
 * @param gqa GQA dimensions
 * @param stream CUDA stream
 */
void merge_qkv_grads_gqa(const TensorView& grad_Q, const TensorView& grad_K, const TensorView& grad_V,
                         TensorView& grad_qkv, const GQADims& gqa,
                         cudaStream_t stream = nullptr);

//======================================================//
//  Utility Functions
//======================================================//

/**
 * Get human-readable name for layout
 */
const char* layout_name(Layout layout);

/**
 * Compute required buffer size for a tensor
 */
size_t compute_buffer_size(const TensorShape& shape);

//======================================================//
//  Debug Utilities (disabled in release builds)
//======================================================//

#ifndef NDEBUG
/**
 * Sample and print tensor statistics for debugging
 */
void debug_print_stats(const TensorView& tensor, const char* label, cudaStream_t stream = nullptr);

/**
 * Verify tensor contains valid floating point values (no NaN/Inf)
 */
bool debug_check_finite(const TensorView& tensor, cudaStream_t stream = nullptr);
#else
inline void debug_print_stats(const TensorView&, const char*, cudaStream_t = nullptr) {}
inline bool debug_check_finite(const TensorView&, cudaStream_t = nullptr) { return true; }
#endif

}  // namespace TensorContract


//======================================================//
//======================================================//
//  GRIM NATIVE AUTOGRAD SYSTEM
//======================================================//
//======================================================//
//
//  This section provides PyTorch-style automatic differentiation
//  for GRIM-text training. Key components:
//
//  1. ParamGroupType - Categories for parameter groups (for optimizer)
//  2. ParameterGroup - Weights + gradients + Adam state
//  3. GradFn - Base class for backward function nodes
//  4. Tensor - Autograd-aware tensor with gradient tracking
//
//  Unlike PyTorch, this is CUDA-first and contiguous-only.
//
//======================================================//

namespace GRIM {

//======================================================//
//  Parameter Group Type (for optimizer categorization)
//======================================================//

/**
 * @brief Category of parameters for optimizer and gradient analysis
 * 
 * Used to:
 * - Compute per-component gradient norms (for debugging)
 * - Apply different learning rates or weight decay per group
 * - Track training dynamics by component type
 */
enum class ParamGroupType : uint8_t {
    EMBEDDING = 0,      ///< Token embeddings
    LM_HEAD = 1,        ///< Language model head (output projection)
    ATTENTION = 2,      ///< Attention weights (W_qkv, W_o)
    FFN = 3,            ///< Feed-forward network weights (W1, W2)
    RMSNORM = 4,        ///< RMSNorm gamma parameters
    SCRATCHBLOCK = 5,   ///< Atom type embeddings + projection
    MTP = 6,            ///< Multi-token prediction auxiliary heads (weight + bias per head)
    REASONING_HEAD = 7, ///< Reasoning head weights (W_op, b_op, w_arg1, w_arg2)
    EXECUTION_BLOCK = 8,///< Execution block weights (decode MLP, arg/op/slot select, cross-attn)
    SLOT_SELECTOR = 9,  ///< Decode-time slot selector (W_q, W_k, null_key, null_bias)
    COUNT = 10          ///< Number of parameter group types
};

//======================================================//
//  Parameter Group (Weights + Gradients + Optimizer State)
//======================================================//

// Forward declaration (full definition below GradFn)
struct Tensor;

/**
 * @brief A group of parameters with associated gradient and optimizer state
 * 
 * Holds a pointer to the owning Tensor for weights/grads/size (single source
 * of truth — no cached raw-pointer copies that can go stale).  Optimizer
 * moment buffers (m_state, v_state) are NOT part of Tensor so they remain
 * raw float* here.
 * 
 * OWNERSHIP: ParameterGroup does NOT own the Tensor or moment memory.
 * The TrainingState/LanguageModel that creates it owns the underlying buffers.
 * 
 * WEIGHT TYING: When tie_embeddings=true, embedding and LM head share
 * the same Tensor data/grad.  The optimizer must NOT update aliased
 * parameters twice.
 */
struct ParameterGroup {
    std::string name;        ///< Human-readable name (e.g., "layer0_attn_W_qkv")
    Tensor* tensor;          ///< Non-owning pointer to the parameter Tensor
    Tensor* m_tensor = nullptr;  ///< Adam first moment Tensor (nullptr until optimizer init)
    Tensor* v_tensor = nullptr;  ///< Adam second moment Tensor (nullptr until optimizer init)
    ParamGroupType type;     ///< Category for optimizer and analysis
    int layer_index = -1;    ///< Encoder layer index (0-based), -1 for non-layer params
    float upsilon = 1.0f;    ///< Depth-aware regularization scale: Υ_l = 0.1 * sqrt(L_ref / L)
    float weight_decay_multiplier = 1.0f;  ///< 0.0 for biases/norms (no weight decay), 1.0 for weights
    float lr_multiplier = 1.0f;  ///< Per-group learning rate scale (e.g., 0.1 = 10x slower learning)
    
    // Live accessors — always read through the Tensor, never stale
    // Defined after struct Tensor (forward-declared only here)
    inline float* weights() const;
    inline float* grads() const;
    inline float* m_state() const;
    inline float* v_state() const;
    inline size_t size() const;
    inline size_t size_bytes() const;
    bool has_optimizer_state() const { return m_tensor != nullptr && v_tensor != nullptr; }
};

//======================================================//
//  GradFn - Backward Function Node (Computation Graph)
//======================================================//

// Forward declaration
struct Tensor;

/**
 * @brief Base class for backward computation nodes
 * 
 * Each differentiable operation creates a GradFn that knows how to compute
 * gradients with respect to its inputs. When backward() is called on a
 * tensor, the graph is traversed and each GradFn::apply() is invoked.
 * 
 * SAVED TENSORS: Operations save input tensors needed for gradient computation.
 * For example, matmul saves A and B to compute:
 *   grad_A = grad_output @ B^T
 *   grad_B = A^T @ grad_output
 * 
 * OWNERSHIP: GradFn owns saved tensor data (copies are made during forward).
 * The tensor's grad_fn pointer is a raw pointer; the Tensor owns the GradFn.
 */
struct GradFn {
    //--------------------------------------------------//
    // Saved State
    //--------------------------------------------------//
    
    std::vector<Tensor*> saved_tensors;  ///< Inputs saved for backward (owned copies)
    std::vector<TensorContract::TensorView> saved_views;  ///< Lightweight views (non-owning)
    
    //--------------------------------------------------//
    // Operation Metadata
    //--------------------------------------------------//
    
    const char* op_name = nullptr;  ///< Operation name ("matmul", "gelu", "add", etc.)
    int call_id = 0;                ///< Unique call ID for deterministic replay
    bool applied = false;           ///< ISSUE #49: Prevent infinite loops when grad_fn is shared
    bool released_ = false;         ///< ISSUE #50: Prevent double release_saved calls
    
    //--------------------------------------------------//
    // Virtual Interface
    //--------------------------------------------------//
    
    virtual ~GradFn() = default;
    
    /**
     * @brief Execute the backward pass for this node
     * 
     * @param grad_output Gradient of loss with respect to this operation's output
     * @param stream CUDA stream for async execution
     */
    virtual void apply(const Tensor& grad_output, cudaStream_t stream) = 0;
    
    /**
     * @brief Release saved tensors (called after backward to free memory)
     */
    virtual void release_saved() {
        if (released_) return;  // ISSUE #50: Prevent double release
        released_ = true;
        saved_tensors.clear();
        saved_views.clear();
    }
};

//======================================================//
//  Autograd Tensor
//======================================================//

/**
 * @brief GPU tensor with automatic differentiation support
 * 
 * This extends TensorBuffer with autograd tracking:
 * - data:          GPU memory for tensor values
 * - grad:          GPU memory for accumulated gradients (lazy-allocated)
 * - grad_fn:       Backward function if tensor resulted from an operation
 * - requires_grad: Whether to track this tensor in the compute graph
 * - is_leaf:       True if created by user (not from an operation)
 * 
 * MEMORY MODEL (CUDA-first):
 * - All data lives on GPU
 * - Gradients are lazy-allocated on first backward
 * - Contiguous storage only (no strides)
 * 
 * LEAF vs NON-LEAF:
 * - Leaf tensors: Created directly (parameters, inputs). Gradients accumulated.
 * - Non-leaf tensors: Results of operations. Gradients computed but not stored
 *   unless retain_grad=true.
 * 
 * GRADIENT ACCUMULATION:
 * When multiple operations depend on the same tensor, gradients are accumulated
 * via accumulate_grad(). This is equivalent to PyTorch's in-place +=.
 */
struct Tensor {
    //--------------------------------------------------//
    // Data Storage
    //--------------------------------------------------//
    
    float* data = nullptr;              ///< GPU memory for values
    TensorContract::TensorShape shape;  ///< Layout-aware shape (2D or 4D)
    bool owns_data = false;             ///< RAII ownership flag for data
    
    //--------------------------------------------------//
    // Autograd Fields
    //--------------------------------------------------//
    
    // ISSUE #59: Gradient is now a full Tensor object (via shared_ptr)
    // - shared_ptr enables weight tying: multiple tensors share same grad Tensor
    // - Gradient Tensor has same shape as parameter, holds GPU buffer
    // - Use grad_data() to access raw float* for CUDA kernels
    // - Use grad() to access Tensor* for autograd operations
    std::shared_ptr<Tensor> grad_ = nullptr;  ///< Gradient tensor (lazy-allocated, shared for weight tying)
    
    std::shared_ptr<GradFn> grad_fn;    ///< Backward function (null for leaf tensors). shared_ptr manages lifetime.
    bool requires_grad = false;         ///< Track in compute graph?
    bool is_leaf = true;                ///< Created by user (not from operation)?
    bool retain_grad = false;           ///< Keep grad even if non-leaf?
    
    //--------------------------------------------------//
    // Execution Context
    //--------------------------------------------------//
    
    cudaStream_t stream = nullptr;      ///< Associated CUDA stream
    int device_id = 0;                  ///< GPU device ID
    
    //--------------------------------------------------//
    // Debug/Telemetry
    //--------------------------------------------------//
    
    const char* name = nullptr;         ///< Optional debug name
    uint64_t version = 0;               ///< Incremented on each in-place modification
    
    //--------------------------------------------------//
    // Constructors / Destructor
    //--------------------------------------------------//
    
    Tensor() = default;
    ~Tensor() { release(); }
    
    // Non-copyable (GPU memory is expensive to copy)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Movable
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    //--------------------------------------------------//
    // Factory Methods
    //--------------------------------------------------//
    
    /// Create zero-initialized tensor with explicit layout
    static Tensor zeros(TensorContract::TensorShape shape, 
                        bool requires_grad = false,
                        cudaStream_t stream = nullptr,
                        const char* name = nullptr);
    
    /// Create zero-initialized tensor with raw dimensions (convenience)
    /// 1D: treated as [1, dim] BSM
    /// 2D: treated as BSM [tokens, features]
    /// 4D: treated as BHSD [batch, heads, seq, head_dim]
    static Tensor zeros(std::initializer_list<int> dims,
                        cudaStream_t stream = nullptr,
                        const char* name = nullptr);
    
    /// Create uninitialized tensor (values undefined)
    static Tensor empty(TensorContract::TensorShape shape,
                        bool requires_grad = false,
                        cudaStream_t stream = nullptr,
                        const char* name = nullptr);
    
    /// Wrap existing GPU pointer (takes ownership if specified)
    static Tensor from_ptr(float* ptr, 
                           TensorContract::TensorShape shape,
                           bool takes_ownership = false,
                           bool requires_grad = false,
                           const char* name = nullptr);
    
    /// Wrap existing GPU pointer with raw dimensions (convenience)
    static Tensor from_ptr(float* ptr,
                           std::initializer_list<int> dims,
                           cudaStream_t stream = nullptr,
                           const char* name = nullptr);
    
    /// Xavier uniform initialization (for weights)
    static Tensor xavier_uniform(TensorContract::TensorShape shape,
                                 bool requires_grad = true,
                                 cudaStream_t stream = nullptr,
                                 const char* name = nullptr);
    
    /// In-place Xavier uniform initialization
    static void xavier_uniform_(Tensor& t, uint64_t seed, cudaStream_t stream = nullptr);
    
    //--------------------------------------------------//
    // Gradient Management
    //--------------------------------------------------//
    
    /// Enable gradient tracking (PyTorch-style)
    Tensor& requires_grad_(bool enable = true) {
        requires_grad = enable;
        return *this;
    }
    
    /// Lazy-allocate gradient Tensor (zeros it)
    void ensure_grad();
    
    /// Get gradient Tensor pointer (nullptr if not allocated)
    Tensor* grad() { return grad_.get(); }
    const Tensor* grad() const { return grad_.get(); }
    
    /// Get raw gradient data pointer (nullptr if grad not allocated)
    /// Use this for CUDA kernel launches
    float* grad_data() { return grad_ ? grad_->data : nullptr; }
    const float* grad_data() const { return grad_ ? grad_->data : nullptr; }
    
    /// Check if gradient is allocated
    bool has_grad() const { return grad_ != nullptr && grad_->data != nullptr; }
    
    /// Share gradient storage with another tensor (for weight tying)
    /// This tensor's grad will point to other's grad Tensor
    void share_grad(Tensor& other) {
        if (!other.grad_) {
            throw std::runtime_error("share_grad: other tensor has no grad to share");
        }
        grad_ = other.grad_;  // shared_ptr copy = shared ownership
    }
    
    /// Set gradient from an externally-owned buffer (non-owning view).
    /// The caller MUST keep external_grad alive for the lifetime of this Tensor's grad.
    void set_grad_from_buffer(float* external_grad) {
        if (!external_grad) {
            throw std::runtime_error("set_grad_from_buffer: external_grad is NULL - caller MUST provide valid pointer");
        }
        grad_ = std::make_shared<Tensor>();
        grad_->data = external_grad;
        grad_->shape = shape;
        grad_->owns_data = false;  // Always non-owning. Caller manages lifetime.
        grad_->requires_grad = false;
        grad_->is_leaf = true;
    }
    
    /// Zero gradient buffer (async on stream)
    void zero_grad(cudaStream_t stream = nullptr);
    
    /// Accumulate incoming gradient (for multi-use tensors)
    void accumulate_grad(const float* incoming_grad, 
                         size_t count,
                         float scale = 1.0f,
                         cudaStream_t stream = nullptr);
    
    /// Detach from compute graph (returns view with requires_grad=false)
    Tensor detach() const;
    Tensor detach(cudaStream_t stream) const;
    
    //--------------------------------------------------//
    // Backward Pass
    //--------------------------------------------------//
    
    /**
     * @brief Trigger backward pass from this tensor
     * 
     * Typically called on the loss tensor. Walks the compute graph
     * backward, calling each GradFn::apply().
     * 
     * @param grad_output Initial gradient (nullptr for scalar 1.0)
     * @param scale       Scalar scale to apply to the initial gradient (default 1.0)
     */
    void backward(const Tensor* grad_output = nullptr, float scale = 1.0f);
    
    //--------------------------------------------------//
    // View Conversion (for compatibility with existing code)
    //--------------------------------------------------//
    
    /// Get a non-owning view of data
    TensorContract::TensorView view() const {
        return TensorContract::TensorView(data, shape, name);
    }
    
    /// Get a non-owning view of gradient data
    /// Note: Returns mutable view even from const Tensor (gradient is logically separate)
    TensorContract::TensorView grad_view() const {
        return TensorContract::TensorView(grad_ ? grad_->data : nullptr, shape, name);
    }
    
    //--------------------------------------------------//
    // Size Queries
    //--------------------------------------------------//
    
    size_t numel() const { return shape.total_elements(); }
    size_t size_bytes() const { return numel() * sizeof(float); }
    TensorContract::Layout layout() const { return shape.layout; }
    bool is_contiguous() const { return true; }  // Always true (no stride support)
    
    //--------------------------------------------------//
    // RULE 20: Validation Methods (Fail Loud)
    //--------------------------------------------------//
    
    /// Returns true if tensor has valid data pointer and shape
    bool is_valid() const {
        return data != nullptr && shape.is_valid();
    }
    
    /// RULE 20: Throws if tensor is invalid - returns self for chaining
    const Tensor& require(const char* context) const {
        if (!data) {
            throw std::runtime_error(std::string(context) + ": Tensor has NULL data pointer" +
                                     (name ? std::string(" (name=") + name + ")" : ""));
        }
        shape.require(context);
        return *this;
    }
    
    /// RULE 20: Throws if tensor is invalid - returns mutable self for chaining
    Tensor& require_mut(const char* context) {
        if (!data) {
            throw std::runtime_error(std::string(context) + ": Tensor has NULL data pointer" +
                                     (name ? std::string(" (name=") + name + ")" : ""));
        }
        shape.require(context);
        return *this;
    }
    
    /// RULE 20: Throws if gradient is not available
    const Tensor& require_grad_present(const char* context) const {
        require(context);
        if (!requires_grad) {
            throw std::runtime_error(std::string(context) + ": Tensor does not require grad" +
                                     (name ? std::string(" (name=") + name + ")" : ""));
        }
        if (!has_grad()) {
            throw std::runtime_error(std::string(context) + ": Tensor grad buffer is NULL" +
                                     (name ? std::string(" (name=") + name + ")" : ""));
        }
        return *this;
    }
    
    //--------------------------------------------------//
    // Memory Management
    //--------------------------------------------------//
    
    void release() {
        if (grad_fn) {
            TENSOR_LOG_LIFECYCLE(gradfn_del_counter,
                "[Tensor::release] #D%d releasing grad_fn=%p op=%s refcount=%ld (data=%p owns_data=%d name=%s)\n",
                (void*)grad_fn.get(), grad_fn->op_name ? grad_fn->op_name : "null",
                grad_fn.use_count(),
                (void*)data, (int)owns_data, name ? name : "unnamed");
        }
        if (owns_data && data) {
            TENSOR_LOG_LIFECYCLE(free_counter,
                "[Tensor::release] #F%d cudaFree data=%p name=%s\n",
                (void*)data, name ? name : "unnamed");
            cudaError_t free_err = cudaFree(data);
            if (free_err != cudaSuccess) {
                fprintf(stderr, "[Tensor::release] cudaFree(%p) failed: %s (name=%s)\n",
                        (void*)data, cudaGetErrorString(free_err),
                        name ? name : "unnamed");
                if (free_err == cudaErrorIllegalAddress) {
                    fprintf(stderr, "[Tensor::release] Hint: illegal memory access on cudaFree usually means "
                            "a prior kernel faulted. Check [CUDA] ERROR after first batch in logs, or run under compute-sanitizer.\n");
                }
            }
        }
        grad_.reset();
        grad_fn.reset();
        data = nullptr;
    }
};

//======================================================//
//  ParameterGroup accessor definitions (need full Tensor)
//======================================================//

inline float* ParameterGroup::weights() const   { return tensor->data; }
inline float* ParameterGroup::grads()   const   { return tensor->grad_data(); }
inline float* ParameterGroup::m_state() const   { return m_tensor ? m_tensor->data : nullptr; }
inline float* ParameterGroup::v_state() const   { return v_tensor ? v_tensor->data : nullptr; }
inline size_t ParameterGroup::size()    const   { return tensor->numel(); }
inline size_t ParameterGroup::size_bytes() const { return size() * sizeof(float); }

}  // namespace GRIM (temporarily close for cuBLAS forward decl)

//======================================================//
//  Autograd Operations (create computation graph nodes)
//======================================================//

// Forward declarations for cuBLAS types (must be in global namespace)
struct cublasContext;
typedef cublasContext* cublasHandle_t;

namespace GRIM {  // reopen namespace

namespace autograd {

/**
 * Set/get the cuBLAS handle for autograd matmul operations.
 * Must be called before using matmul() function.
 * Thread-local: each thread can have its own handle.
 */
void set_autograd_cublas_handle(cublasHandle_t handle);
cublasHandle_t get_autograd_cublas_handle();

// Issue #142: set_lm_head_grad_correction() DELETED.
// Centering backward is now handled by CenterRowsGradFn/CenterColumnsGradFn
// inside the autograd graph (Issues #125/#132).

/**
 * Matrix multiplication: C = A @ B  (or C = A @ B^T if transpose_b=true)
 * Creates MatMulGradFn node if either input requires_grad
 * Requires: call set_autograd_cublas_handle() first
 *
 * TAPE-BASED: Uses external cache pointers for backward pass.
 * If a_cache/b_cache is nullptr, uses tensor data directly (assumes it persists).
 * 
 * @param a_cache External cache for A (needed if B requires grad)
 * @param b_cache External cache for B (needed if A requires grad)
 * @param transpose_b If true, computes A @ B^T instead of A @ B
 */
Tensor matmul(const Tensor& a, const Tensor& b, cudaStream_t stream = nullptr,
              const float* a_cache = nullptr, const float* b_cache = nullptr,
              bool transpose_b = false);

/**
 * Element-wise addition: C = A + B
 */
Tensor add(const Tensor& a, const Tensor& b, cudaStream_t stream = nullptr);

/**
 * Scale a scalar tensor by a constant (for loss weighting, e.g. MTP alpha/K).
 * Forward: result = scale * input (input must be 1 element).
 * Backward: gradient passed to input is scale * grad_output.
 */
Tensor scale_scalar(const Tensor& t, float scale, cudaStream_t stream = nullptr);

// autograd::scale() DELETED — dead code from reverted Issue #98 (Rule 20)

/**
 * Element-wise exponential: y = exp(x)
 * Backward: grad_x = grad_y * y
 */
Tensor exp(const Tensor& x, cudaStream_t stream = nullptr);

/**
 * Add a constant scalar to every element: y = x + c
 * Backward: grad_x = grad_y  (pure pass-through, constant has no gradient)
 */
Tensor add_scalar(const Tensor& x, float scalar, cudaStream_t stream = nullptr);

/**
 * Element-wise reciprocal: y = 1/x
 * Backward: grad_x = grad_y * (-y²)
 */
Tensor reciprocal(const Tensor& x, cudaStream_t stream = nullptr);

/**
 * Multiply every element of a tensor by a constant float.
 * Forward: y = x * scalar
 * Backward: grad_x = grad_y * scalar
 */
Tensor mul_scalar(const Tensor& x, float scalar, cudaStream_t stream = nullptr);

/**
 * Broadcast per-row scalar multiply: out[i,j] = scale[i,0] * x[i,j]
 * scale must be [rows, 1], x must be [rows, cols].
 * Backward: grad_scale[i] = sum_j(grad_out[i,j] * x[i,j])
 *           grad_x[i,j] = grad_out[i,j] * scale[i,0]
 */
Tensor broadcast_row_mul(const Tensor& scale, const Tensor& x, cudaStream_t stream = nullptr);

/**
 * Zero-pad: places [rows, cols] input at row_offset in a [total_rows, cols] zero tensor.
 * Forward:  result = 0; result[row_offset:row_offset+rows, :] = x
 * Backward: grad_x += grad_result[row_offset:row_offset+rows, :]
 */
Tensor zero_pad(const Tensor& x, int row_offset, int total_rows, cudaStream_t stream = nullptr);

/**
 * LayerScale: Scale tensor by a learned scalar parameter (tensor of shape [1])
 * Forward:  y[i,j] = x[i,j] * scale_param[0]
 * Backward: grad_x = grad_y * scale_param
 *           grad_scale = sum(grad_y * x)
 *
 * ISSUE #109: LayerScale from CaiT paper - learnable residual connection scaling
 * to reduce high input row correlation in deeper transformer layers.
 *
 * @param x Input tensor [N, D]
 * @param scale_param Learnable scalar tensor [1]
 * @param stream CUDA stream
 * @return Scaled tensor with autograd tracking for both input and scale_param
 */
Tensor layer_scale(const Tensor& x, Tensor& scale_param, cudaStream_t stream = nullptr);

/**
 * Row-wise mean centering: output[t,d] = input[t,d] - mean_d(input[t,:])
 *
 * ISSUE #118 FIX: Removes common direction from activations to prevent mode collapse.
 * The common direction (learned by V projection and FFN) accumulates through 
 * residual stream across 12 encoder layers. By centering before residual add,
 * we zero this accumulated bias.
 *
 * Mathematical property: Backward is ALSO centering (grad_x = grad_y - mean(grad_y))
 * This is because centering is a linear operation with symmetric Jacobian.
 *
 * @param x Input tensor [N, D] where N=total_tokens, D=d_model
 * @param stream CUDA stream
 * @return Centered tensor (each row has sum ≈ 0)
 */
Tensor center_rows(const Tensor& x, cudaStream_t stream = nullptr);

/**
 * Center columns (positions) - subtract mean across rows for each column
 * ISSUE #118 PROPER FIX: This is the CORRECT centering dimension!
 * 
 * Forward:  y[t,d] = x[t,d] - mean_t(x[:,d])   (per-column mean subtraction)
 * Backward: grad_x = grad_y - mean_t(grad_y)  (same operation - linear!)
 *
 * WHY THIS IS DIFFERENT FROM center_rows:
 * - center_rows: Subtracts mean across columns (features) → row sums = 0
 *   BUT doesn't change cos(h_i, h_j) between positions!
 * - center_columns: Subtracts mean across rows (positions) → column sums = 0
 *   This REMOVES the common direction all positions share → reduces avg_cos!
 *
 * Mathematical effect:
 *   Before: h[t,:] = signal[t,:] + common[:]   (all positions share common component)
 *   After:  h[t,:] = signal[t,:]              (common component removed!)
 *
 * @param x Input tensor [N, D] where N=total_tokens (positions), D=d_model (features)
 * @param stream CUDA stream
 * @return Centered tensor (each column has mean ≈ 0 across positions)
 */
Tensor center_columns(const Tensor& x, cudaStream_t stream = nullptr);

    // Issue #149: project out dominant PC1 direction to prevent mode collapse
    // g = PC1(H) via n_power_iters steps of power iteration (stop-gradient)
    // Forward:  h̃[t] = h[t] - (h[t]·g)*g
    // Backward: grad_h = (I - gg^T) * grad_h̃
    Tensor project_out_pc1(const Tensor& x, int n_power_iters = 5, cudaStream_t stream = nullptr);

/**
 * Broadcast add with bias: output[i,j] = input[i,j] + bias[j]
 * 
 * ISSUE #97 FIX: This replaces raw launchFFNBiasAdd calls with proper autograd tracking.
 * Without this, encoder biases (b_qkv, b_o, b1, b2) were frozen at initialization.
 *
 * @param input Input tensor [N, D] where N=total_tokens, D=features  
 * @param bias Bias tensor [D] - will be broadcasted across tokens
 * @param stream CUDA stream
 * @return Output tensor [N, D] with autograd graph attached for both input and bias
 */
Tensor broadcast_add(const Tensor& input, const Tensor& bias, cudaStream_t stream = nullptr);

/**
 * GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * TAPE-BASED: Uses external cache pointer for backward pass.
 * If input_cache is nullptr, uses tensor data directly (assumes it persists).
 * 
 * @param input_cache External cache for input (needed for backward)
 */
Tensor gelu(const Tensor& x, cudaStream_t stream = nullptr, 
            const float* input_cache = nullptr);

/**
 * SiLU (Swish) activation: y = x * sigmoid(x)
 *
 * Used as the gate activation in SwiGLU feed-forward networks.
 * TAPE-BASED: Uses external cache pointer for backward pass.
 *
 * @param input_cache External cache for input (needed for backward)
 */
Tensor silu(const Tensor& x, cudaStream_t stream = nullptr,
            const float* input_cache = nullptr);

/**
 * Element-wise (Hadamard) product: y = a ⊙ b
 *
 * Used for the gating operation in SwiGLU: SiLU(gate) ⊙ up
 * Both inputs must have the same shape.
 */
Tensor elementwise_mul(const Tensor& a, const Tensor& b, cudaStream_t stream = nullptr);

/**
 * RMSNorm: y = x / rms(x) * (learnable)gamma
 *
 * TAPE-BASED: Uses external cache pointer for backward pass.
 * If input_cache is nullptr, uses tensor data directly (assumes it persists).
 *
 * @param input_cache External cache for input (needed for backward)
 */
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps = 1e-5f, 
                cudaStream_t stream = nullptr, const float* input_cache = nullptr);

/**
 * Embedding lookup: output[i] = weight[token_ids[i]] * embedding_scale
 * @param embedding_scale Scale factor for embeddings (default 1.0, use sqrt(d_model) for AIAYN-style)
 */
Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens,
                 cudaStream_t stream = nullptr, float embedding_scale = 1.0f);

/**
 * Log-Softmax: y[i] = x[i] - logsumexp(x) — numerically stable log(softmax(x))
 * Input: [tokens, dim] - log_softmax computed along dim axis
 * Creates LogSoftmaxGradFn if input.requires_grad
 * @param save_output_copy When true (default), LogSoftmaxGradFn copies the
 *   log-probs for backward. When false, stores a non-owning pointer — caller
 *   must guarantee the data lives until after backward completes.
 *   OOM FIX: unified_loss() passes false because NLLLossGradFn owns the data.
 */
Tensor log_softmax(const Tensor& x, cudaStream_t stream = nullptr, bool save_output_copy = true);

/**
 * Dropout with auto-generated mask: y = x * mask / (1 - p)
 * Generates mask internally using Philox PRNG.
 * 
 * @param x Input tensor
 * @param p Dropout probability (fraction to drop, e.g., 0.1 = drop 10%)
 * @param seed Random seed for mask generation (use batch_idx * step + layer_offset for reproducibility)
 * @param training If false, no dropout is applied (identity function) 
 */
Tensor dropout(const Tensor& x, float p, uint64_t seed, bool training = true,
               cudaStream_t stream = nullptr);

/**
 * Residual/skip connection add: y = x + residual
 * Optimized backward: both inputs receive the unmodified gradient
 */
Tensor residual_add(const Tensor& x, const Tensor& residual,
                    cudaStream_t stream = nullptr);

/**
 * Scaled dot-product attention with optional mask and ALiBi
 * 
 * @param causal If true, applies causal mask (autoregressive). If false, all positions attend all others.
 * @param attention_dropout_p Attention dropout DROP rate (0.0 = disabled, 0.15 = 15% drop rate).
 *        Converted internally to keep probability for FlashAttention (keep_p = 1.0 - attention_dropout_p).
 * @param dropout_seed Per-step Philox RNG seed for reproducible dropout masks.
 *        Use ctx.step * 2654435761ULL + layer_offset for per-step, per-layer uniqueness.
 */
Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const float* alibi_slopes = nullptr, float scale = 0.0f,
    cudaStream_t stream = nullptr,
    bool causal = true,
    float attention_dropout_p = 0.0f, uint64_t dropout_seed = 0);

/**
 * Split QKV projection output and reshape to BHSD layout with autograd tracking.
 * 
 * ISSUE #61 FIX: This replaces manual cudaMemcpy2D + Tensor::empty() operations
 * that broke the autograd chain (Q/K/V had no grad_fn, causing zero gradients
 * for W_qkv).
 * 
 * @param qkv_out      Input tensor [tokens, d_model + 2*kv_dim] from matmul(ln1_out, W_qkv)
 * @param batch        Batch size
 * @param seq          Sequence length (tokens = batch * seq)
 * @param num_heads    Number of Q heads
 * @param num_kv_heads Number of K/V heads (GQA)
 * @param head_dim     Dimension per head
 * @param stream       CUDA stream
 * @return Tuple of (Q_bhsd, K_bhsd, V_bhsd) with properly linked autograd chains
 */
std::tuple<Tensor, Tensor, Tensor> split_and_reshape_qkv(
    Tensor& qkv_out,
    int batch, int seq, int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream = nullptr);

/**
 * Reshape attention output from BHSD to flat layout with autograd tracking.
 * 
 * ISSUE #62 FIX: This replaces Tensor::empty() + launchReshapeFromBHSD()
 * that broke the autograd chain (output had no grad_fn, causing W_o and 
 * downstream gradients to not flow through attention backward).
 * 
 * @param bhsd_input   Input tensor [B, H, S, D] from attention output
 * @param batch_size   Batch size
 * @param seq_len      Sequence length
 * @param num_heads    Number of attention heads
 * @param head_dim     Dimension per head
 * @param stream       CUDA stream
 * @return Tensor [tokens, d_model] with properly linked autograd chain
 */
Tensor reshape_bhsd_to_flat(
    Tensor& bhsd_input,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream = nullptr);

/**
 * Apply RoPE rotation to Q and K tensors with autograd tracking.
 * 
 * ISSUE #119 FIX: This wraps PBM::launchRoPERotationGQA() to provide proper
 * autograd tracking. Previously, RoPE was called with raw .data pointers,
 * completely bypassing the autograd chain. The backward kernel existed but
 * was NEVER called because no RoPEGradFn was attached.
 * 
 * Forward: Rotates Q and K in-place using R(θ) where θ = position * inv_freq
 * Backward: Applies inverse rotation R(-θ) to dQ and dK gradients
 * 
 * Uses SharedState pattern to coordinate backward calls from both Q and K.
 * 
 * @param Q          Q tensor [B, H, S, D] - modified in-place
 * @param K          K tensor [B, Hkv, S, D] - modified in-place  
 * @param inv_freq   Device pointer to inverse frequencies [rotary_dim/2]
 * @param batch_size Batch size (B)
 * @param num_q_heads Number of query heads (H)
 * @param num_kv_heads Number of key/value heads (Hkv) for GQA
 * @param seq_len    Sequence length (S)
 * @param head_dim   Dimension per head (D)
 * @param rotary_dim Number of dimensions to rotate (must be <= head_dim, typically 64)
 * @param stream     CUDA stream
 */
void rope_rotation(
    Tensor& Q, Tensor& K,
    const float* inv_freq,
    int batch_size, int num_q_heads, int num_kv_heads,
    int seq_len, int head_dim, int rotary_dim,
    cudaStream_t stream = nullptr,
    int pos_offset = 0);

/**
 * Softmax: p[i] = exp(x[i]) / sum_j exp(x[j])  (along last dim, numerically stable)
 * Input: [tokens, dim] - softmax computed along dim axis
 * Optionally divides logits by temperature before computing softmax.
 * Creates SoftmaxGradFn if input.requires_grad
 *
 * @param temperature  Divide logits by this value before softmax. 1.0 = no scaling.
 */
Tensor softmax(const Tensor& x, float temperature = 1.0f, cudaStream_t stream = nullptr);

/**
 * Concatenate two 2D tensors along columns (last dim).
 * Input:  a [N, D1], b [N, D2]  (same number of rows)
 * Output: [N, D1 + D2]
 * Creates ConcatGradFn if either input requires_grad
 */
Tensor concat(const Tensor& a, const Tensor& b, cudaStream_t stream = nullptr);

/**
 * Cross-entropy loss from logits (stable log-sum-exp formulation).
 * Input:  logits [1, num_classes] — raw scores (NOT probabilities)
 * Target: single integer class index in [0, num_classes)
 * Output: scalar CE tensor [1, 1]
 *
 * Forward:  CE = log(sum(exp(z_i - z_max))) + z_max - z[target]
 * Backward: d_logits[i] = softmax(z)[i] - 1_{i == target}
 *
 * Creates CrossEntropyLogitsGradFn if input requires_grad.
 * Designed for small classification heads (selector, execution block).
 */
Tensor cross_entropy_logits(const Tensor& logits, int target_idx, cudaStream_t stream = nullptr);

}  // namespace autograd

}  // namespace GRIM
