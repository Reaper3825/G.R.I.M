#pragma once
//======================================================//
//  TensorContract_GPU.hpp
//  Type-safe tensor abstraction layer for CUDA operations
//======================================================//
//
//  PURPOSE
//  =======
//  This module provides a type-safe abstraction for tensor operations,
//  eliminating the need to reason about layout, stride, transpose, and
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
#include <algorithm>  // for std::max in merge_qkv_grads_gqa

namespace TensorContract {

//======================================================//
//  Tensor Layout Enum
//======================================================//

enum class Layout : uint8_t {
    // 2D formats (flat tensors) - use Shape2D
    BSM,               // [tokens, d_model] - Collapsed activation format
    QKV_FUSED,         // [tokens, total_qkv_dim] - Fused QKV projection output
    
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
    return l == Layout::BSM || l == Layout::QKV_FUSED;
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
