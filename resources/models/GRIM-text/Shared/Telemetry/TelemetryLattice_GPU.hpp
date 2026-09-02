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
     * Extracts tokens from payload.actual_tokens.
     * Fails loud if actual_tokens <= 0.
     */
    TelemetryError updateFromBatch(
        const ::GRIM::Batching::BatchPayload& payload,
        float loss, float grad_norm, float learning_rate,
        uint32_t global_step);

    /**
     * @brief Read telemetry vector from specific level and stream
     * Performs GPU->CPU copy of 10-element telemetry vector.
     */
    TelemetryError readVector(int level, int stream_idx,
                              TelemetryVector* out_vector) const;

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
    int* error_flag() { return d_error_flag_; }

private:
    friend const LatticeLevelState* latticeLevelsDevicePtr(const TelemetryLattice& lattice);

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
    RESERVED_23 = 23,
    RESERVED_24 = 24,
    EB_LOSS_FRAC = 25,               // execution_loss / total_loss
    RESERVED_26 = 26,
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
    // RMSNorm learned gamma tracking — detect gamma drift/collapse
    RMS_GAMMA_PRE_ATTN_RMS = 35,     // mean RMS(γ₁) across encoder layers (pre-attention)
    RMS_GAMMA_PRE_FFN_RMS = 36,      // mean RMS(γ₂) across encoder layers (pre-FFN)
    RMS_GAMMA_FINAL_RMS = 37,        // RMS(γ_final) — LM head final RMSNorm gamma
    // ρ-denominator collapse detector (April 2026)
    RHO_RAW_RMS_SPREAD = 38,         // rms_max / rms_min — per-position rms bifurcation
                                     // (>2.0 indicates row-centering or upstream collapse
                                     //  is stripping certain positions to near-zero,
                                     //  driving spurious high ρ via tiny denominators)
    // h↔W alignment diagnostics (April 2026) — LM-head leak channel detector.
    // Random-baseline RMS(cos) = 1/sqrt(d_model) ≈ 0.0361 at d=768.
    // logit_std_ratio² ≈ 1 + d_model · cos_hW_rms²; ratio>1 implies coherent
    // h-aligned drift in W rows (e.g., from Σ_t(p_v−y_v)·h_t accumulating in grad_W).
    HW_COS_RMS = 39,                 // sqrt(mean cos²(h_t, W_v)) over sampled t,v — primary alignment metric
    HW_COS_SIGNED_MEAN = 40,         // mean cos(h_t, W_v) — DC-leak channel (rank-1 component)
    HW_COS_ABS_MAX = 41,             // max |cos(h_t, W_v)| — worst-case single-row alignment
    HW_HBAR_WBAR_COS = 42,           // cos(mean_t h_t, mean_v W_v) — explicit rank-1 DC coupling
    HW_H_DC_MEAN = 43,               // mean over t of (1/d) Σ_d h[t,d] — DC component of hidden state
    HW_H_DC_ABS_MAX = 44,            // max  over t of |(1/d) Σ_d h[t,d]| — worst-position DC offset
    // Unigram-frequency-direction collapse detector (Apr 2026).
    // e_uf_dir := mean over valid positions of embedding[input_ids[t]] — empirical
    // estimator of Σ_v p(v)·E[v] since positions are sampled from p(v).
    // High |cos(h_t, e_uf_dir)| during steps 0–600 → unigram-frequency collapse mode
    // (h aligned with the dominant-token direction). Compared at the LM-input tensor
    // (post-centering when `lm_head_centering.project_out_pc1` is enabled).
    UNIGRAM_DIR_COS_ABS_MEAN = 45,    // mean_t |cos(h_t, e_uf_dir)| at LM-input tensor
    UNIGRAM_DIR_COS_SIGNED_MEAN = 46, // mean_t  cos(h_t, e_uf_dir) at LM-input tensor

    // Raw LM-head weight RMS that enters the logit-scale equation
    //   logit_std ≈ sqrt(d_model) × h_rms × W_rms_rms
    // where W_rms_rms = sqrt(E[||W_v||² / d_model]) over a strided sample of vocab rows.
    // Published every step the LOGIT_SCALE diagnostic runs (see Phase2_TrainingLoop.cu).
    LM_HEAD_W_RMS_RMS = 47,

    // Init-time structural invariants — published once by Phase1 (after
    // buildParameterGroups) and held constant for the whole run. Phase2's
    // per-step lattice update keeps re-pushing them, so mu == value and
    // sigma == 0 across every level. Full pointer/config/parameter-group
    // init facts are dumped to LogRecorder in the shared session log
    // (`training_<session>.log`).
    // Pointer values do not fit in float streams; invariant failures include
    // them in the thrown error text.
    INIT_TIE_CFG          = 48,  // 1.0 if config.tie_embeddings, else 0.0
    INIT_TIE_PTRS_SAME    = 49,  // 1.0 if emb_weight_ptr == lm_weight_ptr
    INIT_TIE_GRADS_SAME   = 50,  // 1.0 if emb_grad_ptr  == lm_grad_ptr
    INIT_LM_OWNS_WEIGHTS  = 51,  // 1.0 if LMHeadLayer owns its weight buffer
    INIT_OPT_GROUPS_TOTAL = 52,  // count of parameter groups built
    INIT_OPT_GROUPS_EMB   = 53,  // groups whose tensor matches emb_weight_ptr
    INIT_OPT_GROUPS_LM    = 54,  // groups whose tensor matches lm_weight_ptr

    // Rho mean-removal / signed-dot diagnostics — final collected layer only,
    // emitted by RHO_BUILDUP_EQUATION at the rho diagnostic cadence.
    RHO_RAW_AVG_SIGNED_DOT   = 55, // mean dot(h_i,h_j) over sampled valid pairs
    RHO_CENTERED_AVG_ABS_DOT = 56, // mean|dot(h_i-μ,h_j-μ)| over sampled valid pairs
    RHO_MEAN_VECTOR_RMS      = 57, // sqrt(mean_d μ_d²), μ = mean_i h_i

    // Atom-specialized path isolator — final collected layer, RHO_BUILDUP cadence.
    // If atom-only processing is what drives ρ up, atom-only ρ spikes while
    // non-atom ρ stays flatter. The split pins the cause directly.
    RHO_ATOM_ONLY    = 58, // avg|cos(h_i,h_j)| over atom-token positions only
    RHO_NONATOM_ONLY = 59, // avg|cos(h_i,h_j)| over non-atom positions only (baseline)

    // Optimizer-boundary counter mirrored into telemetry so Adam/RAdamW derived
    // diagnostics can be aligned against the exact zero-based optimizer step + 1
    // iteration used by the optimizer kernels.
    OPTIMIZER_ITERATION = 60, // input.optimizer_step + 1 as consumed by optimizer diagnostics

    // Raw loss decomposition from the completed BatchResult. These remain in
    // objective units so they can be graphed directly beside stream 0 (LOSS).
    TEXT_LOSS                = 61,
    LOCAL_ATOM_RETRIEVAL_LOSS = 62,
    SELECTOR_LOSS            = 63,
    RESERVED_64              = 64,
    RESERVED_65              = 65,
    RESERVED_66              = 66,
    RESERVED_67              = 67,
    EXECUTION_LOSS           = 68,

    // Legacy execution-objective stream IDs are retained for telemetry schema
    // compatibility. The removed teacher-forced execution path no longer
    // writes them.
    EXEC_LOSS_GATE_CE_RAW          = 69,
    EXEC_LOSS_STOP_CE_RAW          = 70,
    EXEC_LOSS_OP_CE_RAW            = 71,
    EXEC_LOSS_ARG1_CE_RAW          = 72,
    EXEC_LOSS_ARG2_CE_RAW          = 73,
    EXEC_LOSS_WRITE_CE_RAW         = 74,
    EXEC_LOSS_DIV_PRE_NORM         = 75,
    EXEC_LOSS_ENTROPY_CONTRIBUTION = 76,
    EXEC_LOSS_GATE_CONTRIBUTION    = 77,
    EXEC_LOSS_STOP_CONTRIBUTION    = 78,
    EXEC_LOSS_OP_CONTRIBUTION      = 79,
    EXEC_LOSS_ARG1_CONTRIBUTION    = 80,
    EXEC_LOSS_ARG2_CONTRIBUTION    = 81,
    EXEC_LOSS_WRITE_CONTRIBUTION   = 82,
    EXEC_LOSS_DIV_CONTRIBUTION     = 83,
    EXEC_LOSS_RECONSTRUCTED        = 84,
    EXEC_LOSS_RESIDUAL             = 85,
    EXEC_GATE_ACCURACY             = 86,
    EXEC_STOP_ACCURACY             = 87,
    EXEC_OP_ACCURACY               = 88,
    EXEC_ARG1_ACCURACY             = 89,
    EXEC_ARG2_ACCURACY             = 90,
    EXEC_WRITE_ACCURACY            = 91,
    RESERVED_92                    = 92,
    EXEC_LOSS_SCALAR_TERM_COUNT    = 93,
};

const char* getMetricStreamName(MetricStream stream);

} // namespace GRIM::Telemetry
