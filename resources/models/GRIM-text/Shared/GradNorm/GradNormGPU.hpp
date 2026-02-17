#pragma once
/**
 * @file GradNormGPU.hpp
 * @brief GPU-resident gradient norm computation with no CPU sync
 * 
 * ARCHITECTURE (Rules 20-22 Compliant):
 * =====================================
 * - All computation on GPU - NO cudaStreamSynchronize during compute
 * - Computes norms only - clipping done in Phase2 via scaleGradients()
 * - Integrates with TelemetryLattice for observability
 * - Uses TensorContract for buffer validation
 * - Centralized controller pattern - uses TrainingState's stream
 * 
 * KERNEL DESIGN:
 * ==============
 * 1. sumSquaresPerGroupKernel: Parallel reduction per parameter group
 *    - Each group has its own partial sum
 *    - Block-level reduction in shared memory
 *    - Grid-level atomicAdd to group's partial sum
 * 
 * 2. finalizeNormsKernel: Single-block kernel for final reduction
 *    - Computes sqrt(sum) for each group
 *    - Computes total_norm = sqrt(sum of all squared norms)
 *    - Detects NaN/Inf (fail loud)
 *    - Optional: applies global clip scale in-place
 * 
 * MEMORY LAYOUT:
 * ==============
 * Device:
 *   d_partial_sums_[num_groups]  - Per-group squared sum (intermediate)
 *   d_group_norms_[num_groups]   - Per-group L2 norm (final)
 *   d_metrics_                   - GradMetrics struct on device
 * 
 * Host (pinned):
 *   h_metrics_                   - GradMetrics for async D2H copy
 * 
 * FAIL LOUD:
 * ==========
 * - NaN/Inf detection sets flags in GradMetrics (no silent failures)
 * - Kernel launch errors return error codes
 * - Invalid pointers validated via TensorContract
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Include TensorContract for ParamGroupType (now defined there)
#include "../TensorContract/TensorContract_GPU.hpp"

//=============================================================================
// FORWARD DECLARATIONS (outside GradNorm namespace)
// ParamGroupType is now defined in TensorContract_GPU.hpp (GRIM namespace)
//=============================================================================

namespace GRIM::GradNorm {

//=============================================================================
// STATUS CODES
//=============================================================================

enum class GradNormStatus : uint8_t {
    SUCCESS = 0,
    NOT_INITIALIZED = 1,
    INVALID_PARAM = 2,
    CUDA_ERROR = 3,
    ALLOC_FAILED = 4,
    KERNEL_LAUNCH_FAILED = 5,
    NAN_DETECTED = 6,
    INF_DETECTED = 7,
};

const char* statusToString(GradNormStatus status);

//=============================================================================
// METRICS STRUCTURE (GPU + Host)
//=============================================================================

/**
 * GradMetrics - Packed metrics for GPU->Host async copy
 * 
 * Aligned to 64 bytes for efficient memory transfer.
 * Contains per-type norms + flags.
 */
struct alignas(64) GradMetrics {
    // Per-group-type norms (L2) - must match ParamGroupType enum order
    float embedding_norm = 0.0f;      // ParamGroupType::EMBEDDING = 0
    float lm_head_norm = 0.0f;        // ParamGroupType::LM_HEAD = 1
    float attention_norm = 0.0f;      // ParamGroupType::ATTENTION = 2
    float ffn_norm = 0.0f;            // ParamGroupType::FFN = 3
    float rmsnorm_norm = 0.0f;        // ParamGroupType::RMSNORM = 4
    float scratchblock_norm = 0.0f;   // ParamGroupType::SCRATCHBLOCK = 5
    
    // Global metrics
    float total_norm = 0.0f;        // sqrt(sum of all squared norms)
    float clip_scale = 1.0f;        // Applied scaling (1.0 if no clipping)
    float max_norm = 0.0f;          // Maximum per-group norm
    
    // Status flags
    uint32_t has_nan = 0;           // 1 if any NaN detected
    uint32_t has_inf = 0;           // 1 if any Inf detected
    uint32_t groups_processed = 0;  // Number of groups computed
    uint32_t _pad = 0;              // Alignment padding

    // First NaN/Inf details (group index + norm value)
    int32_t first_nan_group = -1;
    int32_t first_inf_group = -1;
    float first_nan_value = 0.0f;
    float first_inf_value = 0.0f;
};

//=============================================================================
// GRAD NORM CONTROLLER
//=============================================================================

/**
 * GradNormController - GPU-resident gradient norm computation
 * 
 * USAGE:
 *   1. initialize() with max number of groups
 *   2. computeAndClip() each batch - GPU-only, no sync
 *   3. asyncCopyToHost() when metrics needed for logging
 *   4. getHostMetrics() to read (after sync point)
 *   5. shutdown() on destruction
 * 
 * INTEGRATION:
 *   - Lives in TrainingState (centralized controller pattern)
 *   - Uses TrainingState's StreamController
 *   - Feeds metrics to TelemetryLattice
 */
class GradNormController {
public:
    GradNormController() = default;
    ~GradNormController();
    
    // No copy
    GradNormController(const GradNormController&) = delete;
    GradNormController& operator=(const GradNormController&) = delete;
    
    // Move allowed
    GradNormController(GradNormController&& other) noexcept;
    GradNormController& operator=(GradNormController&& other) noexcept;
    
    /**
     * @brief Initialize GPU resources
     * 
     * @param max_groups  Maximum number of parameter groups
     * @param stream      CUDA stream (from StreamController)
     * @return            Status code
     */
    GradNormStatus initialize(size_t max_groups, cudaStream_t stream);
    
    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Compute gradient norms (no clipping - that's done in Phase2)
     * 
     * FULLY GPU-RESIDENT: No CPU sync during this call.
     * Clipping is done separately via scaleGradients() after CPU-side decision.
     * 
     * @param groups        Array of ParameterGroup structs
     * @param num_groups    Number of groups
     * @param stream        CUDA stream
     * @return              Status code
     */
    GradNormStatus computeNorms(
        const GRIM::ParameterGroup* groups,
        size_t num_groups,
        cudaStream_t stream
    );
    
    /**
     * @brief Async copy metrics to pinned host buffer
     * 
     * Call this before a sync point where you need metrics for logging.
     * 
     * @param stream  CUDA stream
     * @return        Status code
     */
    GradNormStatus asyncCopyToHost(cudaStream_t stream);
    
    /**
     * @brief Get host metrics (valid after stream sync)
     * 
     * WARNING: Only valid after cudaStreamSynchronize following asyncCopyToHost.
     */
    const GradMetrics& getHostMetrics() const { return *h_metrics_; }
    
    /**
     * @brief Release GPU resources
     */
    void shutdown();
    
private:
    // Initialization state
    bool initialized_ = false;
    size_t max_groups_ = 0;
    
    // Device memory
    float* d_partial_sums_ = nullptr;   // [max_groups] - partial squared sums
    float* d_group_norms_ = nullptr;    // [max_groups] - final per-group norms
    GradMetrics* d_metrics_ = nullptr;  // Device-side metrics struct
    GRIM::ParamGroupType* d_types_temp_ = nullptr;  // [max_groups] - reusable types buffer (PERF FIX)
    
    // Host pinned memory (for async copy)
    GradMetrics* h_metrics_ = nullptr;
    GRIM::ParamGroupType* h_types_temp_ = nullptr;  // [max_groups] - pinned host buffer for types
    
    // Last stream used (for debugging)
    cudaStream_t last_stream_ = nullptr;
};

} // namespace GRIM::GradNorm
