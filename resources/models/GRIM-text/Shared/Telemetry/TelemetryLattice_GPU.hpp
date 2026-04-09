#pragma once
/**
 * @file TelemetryLattice_GPU.hpp
 * @brief Multi-scale hierarchical telemetry system (Pattern B: self-managing)
 * 
 * ARCHITECTURE:
 *   Level 0: Updates every step (raw stream)
 *   Level k: Updates every n_k = 2^k steps (aggregates level k-1)
 * 
 * USAGE (Pattern B):
 *   1. Construct: TelemetryLattice lattice(config);
 *   2. Each step: lattice.updateFromBatch(payload, loss, grad, lr, step);
 *   3. Read: lattice.readVector(level, stream, &vec);
 *   4. Destructor frees all GPU memory (RAII)
 * 
 * GUARANTEES:
 *   - Pure GPU execution (no CPU round-trips)
 *   - Fail loud on NaN/Inf (if strict_mode)
 *   - Thread-safe single-stream updates
 */

#include "TelemetryState_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include <cuda_runtime.h>

namespace GRIM::Telemetry {

//=============================================================================
// LATTICE CONFIGURATION
//=============================================================================

struct LatticeConfig {
    int num_levels = 5;                 // k ∈ [0, 4]: strides [1, 2, 4, 8, 16]
    int num_streams = 5;                // Number of parallel metric streams
    TelemetryHyperParams hyperparams;   // Global hyperparameters
    cudaStream_t stream = nullptr;      // CUDA stream for kernels
};

//=============================================================================
// TELEMETRY LATTICE (Pattern B: self-allocating, self-managing)
//=============================================================================

class TelemetryLattice {
public:
    /**
     * @brief Construct and allocate lattice on GPU
     * 
     * Allocates device memory for all levels and streams.
     * Throws std::runtime_error on allocation failure (Rule 20: fail loud).
     */
    explicit TelemetryLattice(const LatticeConfig& config);
    ~TelemetryLattice();

    TelemetryLattice(const TelemetryLattice&) = delete;
    TelemetryLattice& operator=(const TelemetryLattice&) = delete;
    TelemetryLattice(TelemetryLattice&& other) noexcept;
    TelemetryLattice& operator=(TelemetryLattice&& other) noexcept;

    //=========================================================================
    // API
    //=========================================================================

    /**
     * @brief Update all levels with new observations
     * 
     * @param observations  Raw metric values [num_streams] (host memory)
     * @param global_step   Current training step
     * @return              Error code (OK on success)
     */
    TelemetryError update(const float* observations, uint32_t global_step);

    /**
     * @brief Update from BatchPayload (single source of truth)
     *
     * Extracts tokens from payload.token_stats.total_tokens.
     * Fails loud if total_tokens <= 0.
     */
    TelemetryError updateFromBatch(
        const GRIM::Batching::BatchPayload& payload,
        float loss, float grad_norm, float learning_rate,
        uint32_t global_step);

    /**
     * @brief Read telemetry vector from specific level and stream
     * Performs GPU->CPU copy of 10-element telemetry vector.
     */
    TelemetryError readVector(int level, int stream_idx,
                              TelemetryVector* out_vector) const;

    /**
     * @brief Read 3 telemetry vectors in single sync (performance optimization)
     * Batches 3 GPU kernel launches + 1 sync + 3 memcpy.
     */
    TelemetryError readBatched(
        int level0, int stream_idx0, TelemetryVector* out_vector0,
        int level1, int stream_idx1, TelemetryVector* out_vector1,
        int level2, int stream_idx2, TelemetryVector* out_vector2) const;

    /**
     * @brief Read full internal state (for debugging/logging)
     */
    TelemetryError readState(int level, int stream_idx,
                             TelemetryState* out_state) const;

    /**
     * @brief Reset all anchors to current values (call after soft restart)
     */
    TelemetryError resetAnchors(cudaStream_t stream);

    //=========================================================================
    // Accessors
    //=========================================================================

    const LatticeConfig& config() const { return config_; }

    LatticeLevelState* levels() { return levels_; }
    const LatticeLevelState* levels() const { return levels_; }
    int* error_flag() { return d_error_flag_; }

private:
    LatticeConfig config_;

    // Pattern B: GRIM::Tensor for float-based GPU buffers
    GRIM::Tensor observations_;       // [num_streams]
    GRIM::Tensor scratch_vectors_;    // [num_streams * 10] — reinterpret as TelemetryVector*

    // Non-float GPU memory (managed by RAII, not Tensor)
    LatticeLevelState* levels_ = nullptr;
    int* d_error_flag_ = nullptr;

    void releaseNonTensor();
};

//=============================================================================
// UTILITY - Stream naming for logging
//=============================================================================

enum class MetricStream : int {
    LOSS = 0,
    GRAD_NORM_MEAN = 1,
    GRAD_NORM_MAX = 2,
    LEARNING_RATE = 3,
    TOKENS_PER_BATCH = 4,
    RHO_FINAL = 5,
    RHO_GROWTH = 6,
    RHO_WORST_DELTA = 7,
    H_RMS_GROWTH = 8,
};

const char* getMetricStreamName(MetricStream stream);

} // namespace GRIM::Telemetry
