#pragma once
/**
 * @file TelemetryLattice_GPU.hpp
 * @brief Multi-scale hierarchical telemetry system
 * 
 * ARCHITECTURE:
 *   Level 0: Updates every step (raw stream)
 *   Level k: Updates every n_k = 2^k steps (aggregates level k-1)
 * 
 * USAGE:
 *   1. Allocate lattice with initTelemetryLattice()
 *   2. Each training step:
 *      - Get UnifiedLossTelemetry from loss kernel
 *      - Feed scalars (loss, grad_norm, etc.) via updateTelemetryLattice()
 *   3. Read telemetry vectors for monitoring/logging
 *   4. Cleanup with freeTelemetryLattice()
 * 
 * GUARANTEES:
 *   - Pure GPU execution (no CPU round-trips)
 *   - Fail loud on NaN/Inf (if strict_mode)
 *   - Thread-safe single-stream updates
 */

#include "TelemetryState_GPU.hpp"
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
// LATTICE HANDLE (opaque)
//=============================================================================

struct TelemetryLattice;

//=============================================================================
// API - GPU-resident operations
//=============================================================================

/**
 * @brief Allocate and initialize lattice on GPU
 * 
 * Creates device memory for all levels and streams.
 * No CPU-side state; everything lives on GPU.
 * 
 * @param config  Lattice configuration
 * @return        Lattice handle (nullptr on failure)
 */
TelemetryLattice* initTelemetryLattice(const LatticeConfig& config);

/**
 * @brief Update all levels of lattice with new observations
 * 
 * Performs:
 *   1. Level 0 update (every step)
 *   2. Check strides for higher levels
 *   3. Aggregate level k-1 telemetry for level k input
 *   4. Update higher levels if stride aligns
 * 
 * INPUTS (host memory, copied to GPU):
 *   observations[num_streams] - Raw metric values (e.g., loss, grad_norm)
 * 
 * EXAMPLE:
 *   float obs[5] = {
 *       telemetry.loss_mean,        // Stream 0: loss
 *       telemetry.grad_norm_mean,   // Stream 1: gradient
 *       telemetry.grad_norm_max,    // Stream 2: max gradient
 *       learning_rate,              // Stream 3: LR
 *       tokens_per_batch            // Stream 4: efficiency
 *   };
 *   TelemetryError err = updateTelemetryLattice(lattice, obs, global_step);
 * 
 * @param lattice       Lattice handle
 * @param observations  Raw metric values [num_streams]
 * @param global_step   Current training step
 * @return              Error code (OK on success)
 */
TelemetryError updateTelemetryLattice(
    TelemetryLattice* lattice,
    const float* observations,
    uint32_t global_step
);

/**
 * @brief Read telemetry vector from specific level and stream
 * 
 * Performs GPU->CPU copy of 10-element telemetry vector.
 * 
 * @param lattice       Lattice handle
 * @param level         Level index [0, num_levels-1]
 * @param stream_idx    Stream index [0, num_streams-1]
 * @param out_vector    Output buffer (10 floats)
 * @return              Error code
 */
TelemetryError readTelemetryVector(
    const TelemetryLattice* lattice,
    int level,
    int stream_idx,
    TelemetryVector* out_vector
);

/**
 * @brief Read 3 telemetry vectors in single sync (PERFORMANCE OPTIMIZATION)
 * 
 * Batches 3 GPU kernel launches + 1 sync + 3 memcpy operations.
 * Reduces 3 sequential syncs (~6700ms) to 1 sync (~100ms).
 * 
 * @param lattice       Lattice handle
 * @param level0        Level for first vector
 * @param stream_idx0   Stream for first vector
 * @param out_vector0   Output buffer for first vector
 * @param level1        Level for second vector
 * @param stream_idx1   Stream for second vector
 * @param out_vector1   Output buffer for second vector
 * @param level2        Level for third vector
 * @param stream_idx2   Stream for third vector
 * @param out_vector2   Output buffer for third vector
 * @return              Error code
 */
TelemetryError readTelemetryBatched(
    const TelemetryLattice* lattice,
    int level0, int stream_idx0, TelemetryVector* out_vector0,
    int level1, int stream_idx1, TelemetryVector* out_vector1,
    int level2, int stream_idx2, TelemetryVector* out_vector2
);

/**
 * @brief Read current state from specific level and stream
 * 
 * For debugging/logging - returns full internal state.
 * 
 * @param lattice       Lattice handle
 * @param level         Level index
 * @param stream_idx    Stream index
 * @param out_state     Output state
 * @return              Error code
 */
TelemetryError readTelemetryState(
    const TelemetryLattice* lattice,
    int level,
    int stream_idx,
    TelemetryState* out_state
);

/**
 * @brief Free all GPU memory for lattice
 * 
 * @param lattice  Lattice handle (set to nullptr after free)
 */
void freeTelemetryLattice(TelemetryLattice** lattice);

/**
 * @brief Reset all anchor values (mu_a, sigma_a) to current values (mu, sigma)
 * 
 * Clears delta_mu and delta_sigma across all levels/streams.
 * Call after soft restart to prevent repeated drift triggers.
 * 
 * @param lattice  Lattice handle
 * @param stream   CUDA stream for kernel launch
 * @return         Error code
 */
TelemetryError resetTelemetryAnchors(TelemetryLattice* lattice, cudaStream_t stream);

/**
 * @brief Get configuration from lattice
 */
LatticeConfig getTelemetryLatticeConfig(const TelemetryLattice* lattice);

//=============================================================================
// UTILITY - Stream naming for logging
//=============================================================================

enum class MetricStream : int {
    LOSS = 0,
    GRAD_NORM_MEAN = 1,
    GRAD_NORM_MAX = 2,
    LEARNING_RATE = 3,
    TOKENS_PER_BATCH = 4,
    // Extend as needed
};

const char* getMetricStreamName(MetricStream stream);

} // namespace GRIM::Telemetry
