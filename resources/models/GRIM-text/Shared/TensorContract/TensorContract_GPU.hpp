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
//  TENSOR LAYOUTS
//  ==============
//  2D LAYOUTS (flat tensors):
//    BSM       = [tokens, d_model]         - Collapsed activation format
//    QKV_FUSED = [tokens, total_qkv_dim]   - Fused QKV projection output
//
//  4D LAYOUTS (attention tensors):
//    BHSD = [batch, heads, seq, head_dim]  - Standard attention format
//    BHDS = [batch, heads, head_dim, seq]  - Key transposed for Q @ K^T
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
    BHDS,              // [batch, heads, head_dim, seq] - Key transposed for Q @ K^T
    BSHD,              // [batch, seq, heads, head_dim] - Merged head format
    
    // Unknown/invalid
    UNKNOWN,
};

/**
 * Check if a layout is 2D (flat) or 4D (multi-dimensional)
 */
constexpr bool is_flat_layout(Layout l) {
    return l == Layout::BSM || l == Layout::QKV_FUSED || l == Layout::LOGITS;
}

constexpr bool is_4d_layout(Layout l) {
    return l == Layout::BHSD || l == Layout::BHDS || l == Layout::BSHD;
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
        if (!is_flat_layout(l)) {
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
    bool is_flat() const { return is_flat_layout(layout); }
    bool is_4d() const { return is_4d_layout(layout); }
    
    // Safe accessors (throw if wrong type)
    const Shape2D& as_2d() const {
        if (!is_flat()) throw std::logic_error("TensorShape: not a 2D layout");
        return flat;
    }
    
    const Shape4D& as_4d() const {
        if (!is_4d()) throw std::logic_error("TensorShape: not a 4D layout");
        return multi;
    }
    
    Shape2D& as_2d() {
        if (!is_flat()) throw std::logic_error("TensorShape: not a 2D layout");
        return flat;
    }
    
    Shape4D& as_4d() {
        if (!is_4d()) throw std::logic_error("TensorShape: not a 4D layout");
        return multi;
    }
    
    // Layout-aware element count
    size_t total_elements() const {
        if (is_flat()) return flat.total_elements();
        if (is_4d()) return multi.total_elements();
        return 0;
    }
    
    // Layout-aware validation
    bool is_valid() const {
        if (layout == Layout::UNKNOWN) return false;
        if (is_flat()) return flat.is_valid();
        if (is_4d()) return multi.is_valid();
        return false;
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
    
    static TensorShape make_BHDS(int batch, int heads, int head_dim, int seq) {
        // Note: BHDS stores head_dim before seq in the logical name,
        // but Shape4D always stores in BHSD order for consistency
        return TensorShape(Layout::BHDS, Shape4D{batch, heads, seq, head_dim});
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
    bool is_flat() const { return shape.is_flat(); }
    bool is_4d() const { return shape.is_4d(); }
    
    // Validation
    bool is_valid() const {
        return ptr != nullptr && shape.is_valid();
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
//  Layout Conversion Operations
//======================================================//

/**
 * Convert tensor from one layout to another
 * 
 * Supported conversions:
 *   - BSM <-> BHSD (embedding to multi-head and back)
 *   - BHSD <-> BHDS (key transpose for attention)
 *   - BHSD <-> BSHD (head dimension reordering)
 * 
 * NOTE: This operation is ASYNCHRONOUS. The kernel is launched on the given stream
 * but does not synchronize. Caller must cudaStreamSynchronize() if the result is
 * needed immediately, or chain subsequent operations on the same stream.
 * 
 * @param src Source tensor view
 * @param dst Destination tensor view (must be pre-allocated)
 * @param stream CUDA stream for async execution
 * @throws ContractViolation if conversion is invalid
 */
void convert(const TensorView& src, TensorView& dst, cudaStream_t stream = nullptr);

/**
 * Check if in-place conversion is possible between two layouts
 * 
 * CRITICAL: Call this BEFORE calling convert_inplace to avoid data corruption.
 * In-place conversion requires specific geometric conditions that are NOT
 * generally satisfied for arbitrary tensor dimensions.
 * 
 * @param tensor Current tensor
 * @param target_layout Desired output layout
 * @return true if in-place conversion is safe, false otherwise
 */
bool can_convert_inplace(const TensorView& tensor, Layout target_layout);

/**
 * In-place layout conversion (when possible)
 * 
 * WARNING: Only call this after can_convert_inplace() returns true!
 * Calling this when in-place conversion is not possible leads to data corruption.
 * 
 * Some conversions can be done in-place (e.g., BHSD <-> BSHD for certain dims).
 * Returns false if in-place conversion failed.
 * 
 * @param tensor Tensor to convert (layout field is updated on success)
 * @param target_layout Desired output layout
 * @param stream CUDA stream for async execution
 * @return true if in-place conversion succeeded, false otherwise
 * @pre can_convert_inplace(tensor, target_layout) == true
 */
bool convert_inplace(TensorView& tensor, Layout target_layout, cudaStream_t stream = nullptr);

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

/**
 * Check if a conversion between two layouts is supported
 */
bool is_conversion_supported(Layout from, Layout to);

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
    NUMERIC_HEAD = 2,   ///< Numeric prediction head
    ATTENTION = 3,      ///< Attention weights (W_qkv, W_o)
    FFN = 4,            ///< Feed-forward network weights (W1, W2)
    RMSNORM = 5,        ///< RMSNorm gamma parameters
    SCRATCHBLOCK = 6,   ///< Atom type embeddings + projection
    COUNT = 7           ///< Number of parameter group types
};

//======================================================//
//  Parameter Group (Weights + Gradients + Optimizer State)
//======================================================//

/**
 * @brief A group of parameters with associated gradient and optimizer state
 * 
 * This struct holds pointers to GPU memory for:
 * - weights:  The actual model parameters [size elements]
 * - grads:    Accumulated gradients [size elements]
 * - m_state:  Adam first moment (momentum) [size elements]
 * - v_state:  Adam second moment (variance) [size elements]
 * 
 * OWNERSHIP: ParameterGroup does NOT own the memory. The TrainingState
 * or LanguageModel that creates it owns the underlying buffers.
 * 
 * WEIGHT TYING: When tie_embeddings=true, embedding and LM head share
 * the same weights/grads pointers (aliased). The optimizer must NOT
 * update aliased parameters twice.
 */
struct ParameterGroup {
    std::string name;        ///< Human-readable name (e.g., "layer0_attn_W_qkv")
    float* weights;          ///< Pointer to GPU weights [size elements]
    float* grads;            ///< Pointer to GPU gradients [size elements]
    size_t size;             ///< Number of elements (not bytes)
    float* m_state;          ///< Adam first moment (can be nullptr until optimizer init)
    float* v_state;          ///< Adam second moment (can be nullptr until optimizer init)
    ParamGroupType type;     ///< Category for optimizer and analysis
    
    // Convenience accessors
    size_t size_bytes() const { return size * sizeof(float); }
    bool has_optimizer_state() const { return m_state != nullptr && v_state != nullptr; }
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
    
    float* grad = nullptr;              ///< GPU memory for gradients (lazy-allocated)
    bool owns_grad = false;             ///< RAII ownership flag for grad
    
    GradFn* grad_fn = nullptr;          ///< Backward function (null for leaf tensors)
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
                        cudaStream_t stream = nullptr);
    
    /// Create zero-initialized tensor with raw dimensions (convenience)
    /// 1D: treated as [1, dim] BSM
    /// 2D: treated as BSM [tokens, features]
    /// 4D: treated as BHSD [batch, heads, seq, head_dim]
    static Tensor zeros(std::initializer_list<int> dims,
                        cudaStream_t stream = nullptr);
    
    /// Create uninitialized tensor (values undefined)
    static Tensor empty(TensorContract::TensorShape shape,
                        bool requires_grad = false,
                        cudaStream_t stream = nullptr);
    
    /// Wrap existing GPU pointer (takes ownership if specified)
    static Tensor from_ptr(float* ptr, 
                           TensorContract::TensorShape shape,
                           bool takes_ownership = false,
                           bool requires_grad = false);
    
    /// Wrap existing GPU pointer with raw dimensions (convenience)
    static Tensor from_ptr(float* ptr,
                           std::initializer_list<int> dims,
                           cudaStream_t stream = nullptr);
    
    /// Xavier uniform initialization (for weights)
    static Tensor xavier_uniform(TensorContract::TensorShape shape,
                                 bool requires_grad = true,
                                 cudaStream_t stream = nullptr);
    
    /// In-place Xavier uniform initialization
    static void xavier_uniform_(Tensor& t, cudaStream_t stream = nullptr);
    
    //--------------------------------------------------//
    // Gradient Management
    //--------------------------------------------------//
    
    /// Enable gradient tracking (PyTorch-style)
    Tensor& requires_grad_(bool enable = true) {
        requires_grad = enable;
        return *this;
    }
    
    /// Lazy-allocate gradient buffer (zeros it)
    void ensure_grad();
    
    /// Zero gradient buffer (async on stream)
    void zero_grad(cudaStream_t stream = nullptr);
    
    /// Accumulate incoming gradient (for multi-use tensors)
    void accumulate_grad(const float* incoming_grad, 
                         size_t count,
                         float scale = 1.0f,
                         cudaStream_t stream = nullptr);
    
    /// Detach from compute graph (returns view with requires_grad=false)
    Tensor detach() const;
    
    //--------------------------------------------------//
    // Backward Pass
    //--------------------------------------------------//
    
    /**
     * @brief Trigger backward pass from this tensor
     * 
     * Typically called on the loss tensor. Walks the compute graph
     * backward, calling each GradFn::apply().
     * 
     * @param grad_output Initial gradient (default: scalar 1.0)
     */
    void backward(const Tensor* grad_output = nullptr);
    
    //--------------------------------------------------//
    // View Conversion (for compatibility with existing code)
    //--------------------------------------------------//
    
    /// Get a non-owning view of data
    TensorContract::TensorView view() const {
        return TensorContract::TensorView(data, shape, name);
    }
    
    /// Get a non-owning view of gradient
    TensorContract::TensorView grad_view() const {
        return TensorContract::TensorView(grad, shape, name);
    }
    
    //--------------------------------------------------//
    // Size Queries
    //--------------------------------------------------//
    
    size_t numel() const { return shape.total_elements(); }
    size_t size_bytes() const { return numel() * sizeof(float); }
    TensorContract::Layout layout() const { return shape.layout; }
    bool is_contiguous() const { return true; }  // Always true (no stride support)
    
    //--------------------------------------------------//
    // Memory Management
    //--------------------------------------------------//
    
    void release() {
        if (owns_data && data) { cudaFree(data); }
        if (owns_grad && grad) { cudaFree(grad); }
        if (grad_fn) { delete grad_fn; }
        data = nullptr;
        grad = nullptr;
        grad_fn = nullptr;
    }
};

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

/**
 * Matrix multiplication: C = A @ B
 * Creates MatMulGradFn node if either input requires_grad
 * Requires: call set_autograd_cublas_handle() first
 */
Tensor matmul(const Tensor& a, const Tensor& b, cudaStream_t stream = nullptr);

/**
 * Element-wise addition: C = A + B
 */
Tensor add(const Tensor& a, const Tensor& b, cudaStream_t stream = nullptr);

/**
 * GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
Tensor gelu(const Tensor& x, cudaStream_t stream = nullptr);

/**
 * RMSNorm: y = x / rms(x) * gamma
 */
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps = 1e-5f, 
                cudaStream_t stream = nullptr);

/**
 * Softmax cross-entropy loss (fused for numerical stability)
 * Automatically creates CrossEntropyGradFn in computation graph
 * @param logits Input logits [tokens, vocab_size] - MUST have Layout::LOGITS
 * @param targets Target token IDs [tokens]
 * @param num_tokens Number of tokens
 * @param vocab_size Vocabulary size
 * @return scalar loss tensor with backward graph attached
 */
Tensor cross_entropy(const Tensor& logits, const int* targets, 
                     int num_tokens, int vocab_size,
                     cudaStream_t stream = nullptr);

/**
 * Focal loss with automatic gradient tracking
 * Focal loss formula: L = α(1-p_t)^γ * CE
 * @param logits Input logits [tokens, vocab_size] - MUST have Layout::LOGITS
 * @param targets Target token IDs [tokens]
 * @param focal_alpha Weighting factor for positive class (default 1.0)
 * @param focal_gamma Focusing parameter (default 2.0, higher = more focus on hard examples)
 * @return scalar loss tensor with backward graph attached
 */
Tensor focal_loss(const Tensor& logits, const int* targets,
                  int num_tokens, int vocab_size,
                  float focal_alpha = 1.0f,
                  float focal_gamma = 2.0f,
                  cudaStream_t stream = nullptr);

/**
 * Unified loss: focal + label smoothing with automatic gradient tracking
 * Combines focal loss weighting with label smoothing regularization
 * @param logits Input logits [tokens, vocab_size] - MUST have Layout::LOGITS
 * @param targets Target token IDs [tokens]
 * @param focal_alpha Focal loss alpha parameter
 * @param focal_gamma Focal loss gamma parameter
 * @param smoothing Label smoothing epsilon (fraction of probability mass to spread)
 * @return scalar loss tensor with backward graph attached
 */
Tensor unified_loss(const Tensor& logits, const int* targets,
                    int num_tokens, int vocab_size,
                    float focal_alpha = 1.0f,
                    float focal_gamma = 2.0f,
                    float smoothing = 0.1f,
                    cudaStream_t stream = nullptr);

/**
 * Embedding lookup: output[i] = weight[token_ids[i]]
 */
Tensor embedding(const Tensor& weight, const int* token_ids, int num_tokens,
                 cudaStream_t stream = nullptr);

/**
 * Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
 * Input: [tokens, dim] - softmax computed along dim axis
 */
Tensor softmax(const Tensor& x, cudaStream_t stream = nullptr);

/**
 * Dropout: y = x * mask / (1 - p), where mask is binary
 * @param x Input tensor
 * @param p Dropout probability (fraction to drop, e.g., 0.1)
 * @param training If false, no dropout is applied
 * @param mask External binary mask (nullptr to generate internally - not yet supported)
 */
Tensor dropout(const Tensor& x, float p, bool training = true,
               const uint8_t* mask = nullptr, cudaStream_t stream = nullptr);

/**
 * Residual/skip connection add: y = x + residual
 * Optimized backward: both inputs receive the unmodified gradient
 */
Tensor residual_add(const Tensor& x, const Tensor& residual,
                    cudaStream_t stream = nullptr);

/**
 * Scaled dot-product attention with optional mask
 */
Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor* mask = nullptr, float scale = 0.0f,
    cudaStream_t stream = nullptr);

}  // namespace autograd

}  // namespace GRIM
