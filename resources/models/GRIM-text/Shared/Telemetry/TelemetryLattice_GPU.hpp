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
    // Adam warmup causation tracking (β₂ convergence diagnostics)
    ADAM_BC2_V_CONVERGENCE = 9,   // 1 - β₂^(step+1): v-estimate quality, half-life=693
    ADAM_SIGNAL_DOMINANCE = 10,   // bc2/(1-bc2): >1 = learning, <1 = destroying
    ADAM_CUMULATIVE_DISP = 11,    // Σlr(t): total weight displacement from Xavier init
    ADAM_DISRUPTION_EMB = 12,     // cumulative_disp / xavier_emb_scale: displacement in Xavier units
    ADAM_INV_BC2_AMP = 13,        // 1/(1-β₂^(step+1)): v bias correction amplification factor
    // Execution Block health tracking
    EXEC_GRAD_NORM = 14,             // RMS of execution block parameter gradients
    EXEC_GRAD_RATIO = 15,            // exec_grad_norm / encoder_grad_norm: relative learning signal
    EXEC_SELECTION_ENTROPY = 16,     // mean(H(arg1)+H(arg2)+H(op)+H(write))/4: decision sharpness
    EXEC_OP_ENTROPY = 17,            // mean(H(op)): operation diversity (collapse = single op)
    EXEC_DIV_CLAMP_RATE = 18,        // div_clamp_count / total_steps: numerical stability
    EXEC_MAX_P_WRITE = 19,           // mean(max(p_write)): write slot concentration
    EXEC_ACTIVE_RATIO = 20,          // active_rows / batch_size: exec block utilization
    // EB/SB injection diagnostics (poisoning hypothesis)
    EB_INJECT_GATE = 21,             // mean sigmoid inject gate across active rows*steps
    EB_READ_GATE_MEAN = 22,          // mean sigmoid cross-attn read gate across tokens*layers
    EB_INJECT_WEIGHT_NORM = 23,      // RMS(w_inject_gate) — gate parameter evolution
    EB_READ_WEIGHT_NORM = 24,        // RMS(W_gate_read) — read gate parameter evolution
    EB_LOSS_FRAC = 25,               // (exec_ce + exec_entropy) / total_loss
    SB_ATOM_EMBED_RMS = 26,          // RMS(atom_type_embeddings) — ScratchBlock injection scale
    // PBM (Positional Bias Method) diagnostics
    PBM_ALIBI_SLOPE_RMS = 27,        // RMS of ALiBi slopes (constant; verifies init integrity)
    PBM_ALIBI_EFF_BIAS_MAX = 28,     // max|slope| * batch_max_seq_len (varies per batch)
    PBM_ROPE_INV_FREQ_RMS = 29,      // RMS of RoPE inverse frequencies (constant; verifies init)
    PBM_BATCH_MAX_SEQ_LEN = 30,      // Actual max sequence length in current batch
    // Raw ρ decomposition (final layer) — trace WHY correlation moves
    RHO_RAW_AVG_ABS_DOT = 31,        // mean|dot(h_i,h_j)| — alignment numerator
    RHO_RAW_AVG_NORM_PROD = 32,      // mean(‖h_i‖·‖h_j‖·d) — normalization denominator
    RHO_RAW_H_RMS_MIN = 33,          // min per-position h_rms — collapse detector
    RHO_RAW_H_RMS_MAX = 34,          // max per-position h_rms — explosion detector
};

const char* getMetricStreamName(MetricStream stream);

} // namespace GRIM::Telemetry
