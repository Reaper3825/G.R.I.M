/**
 * @file TelemetryLattice_GPU.cu
 * @brief GPU kernels for hierarchical telemetry tracking
 * 
 * Pattern B: TelemetryLattice self-allocates GPU memory in constructor,
 * frees in destructor. Float buffers stored as GRIM::Tensor.
 */

#include "TelemetryLattice_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace GRIM::Telemetry {

//=============================================================================
// CONSTANTS (from HyperParameters)
//=============================================================================

constexpr int kMaxLevels = HyperParameters::TELEMETRY_MAX_LEVELS;
constexpr int kMaxStreams = HyperParameters::TELEMETRY_MAX_STREAMS;

//=============================================================================
// DEVICE HELPERS
//=============================================================================

__device__ __forceinline__ float safeSign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

__device__ __forceinline__ float safeSqrt(float x, float eps) {
    return sqrtf(fmaxf(x, eps));
}

__device__ __forceinline__ bool isFiniteValue(float x) {
    return isfinite(x);
}

//=============================================================================
// CORE UPDATE KERNEL (per-stream, per-level)
//=============================================================================

__global__ void updateTelemetryStateKernel(
    LatticeLevelState* levels,      // [num_levels * num_streams]
    const float* observations,      // [num_streams] - input x_t
    const TelemetryHyperParams hp,
    uint32_t global_step,
    int num_levels,
    int num_streams,
    bool strict_mode,
    int* error_flag                 // [1] - atomicOr on error
) {
    const int stream_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream_idx >= num_streams) {
        return;
    }

    const float x_t = observations[stream_idx];
    
    if (strict_mode && !isFiniteValue(x_t)) {
        atomicOr(error_flag, (int)TelemetryError::ERR_NAN_IN_INPUT);
        return;
    }

    //=========================================================================
    // Level 0: Always update (every step)
    //=========================================================================
    
    int level_0_idx = stream_idx;
    LatticeLevelState* level_0 = &levels[level_0_idx];
    TelemetryState* s = &level_0->state;
    
    // STEP 1: Fast magnitude statistics (bias-corrected EMA)
    // Raw EMA accumulators (zero-initialized, no init special case)
    s->mu_raw = hp.beta_mu * s->mu_raw + (1.0f - hp.beta_mu) * x_t;
    s->m2_raw = hp.beta_mu * s->m2_raw + (1.0f - hp.beta_mu) * (x_t * x_t);
    
    // Adam-style bias correction: corrected = raw / (1 - beta^t)
    s->beta_mu_power *= hp.beta_mu;
    const float bc = 1.0f - s->beta_mu_power;
    s->mu = s->mu_raw / bc;
    s->m2 = s->m2_raw / bc;
    
    // Scale-aware epsilon: variance floor proportional to mu^2
    const float raw_variance = s->m2 - s->mu * s->mu;
    const float scale_eps = hp.epsilon * s->mu * s->mu;
    const float variance = fmaxf(raw_variance, scale_eps);
    s->sigma = sqrtf(fmaxf(variance, 0.0f));
    s->sigma_tilde = s->sigma / (fabsf(s->mu) + hp.epsilon);

    // STEP 2: Volatility-of-volatility (σ-decoupled mirror: sigma_jump)
    const float sigma_delta = s->sigma - s->sigma_prev;
    s->sigma_jump = sigma_delta;
    s->v_sigma = hp.beta_v * s->v_sigma + (1.0f - hp.beta_v) * (sigma_delta * sigma_delta);
    s->sigma_prev = s->sigma;

    // STEP 3: Adaptive outlier threshold
    const float v_sigma_sqrt = safeSqrt(s->v_sigma, hp.epsilon);
    s->k_out = hp.k_out0 * (1.0f + hp.alpha_v * v_sigma_sqrt);
    s->c_out = s->mu + s->k_out * s->sigma;

    // STEP 4: Normalized slope + direction (σ-decoupled mirrors: delta_raw, delta_bar_raw)
    // First sample: sigma_prev=0, mu_prev=0 → delta_hat=0 and delta_raw=0 to avoid garbage
    const float delta_raw_inst = (s->step_count == 0) ? 0.0f : (x_t - s->mu_prev);
    s->delta_raw = delta_raw_inst;
    s->delta_bar_raw = hp.beta_delta * s->delta_bar_raw + (1.0f - hp.beta_delta) * delta_raw_inst;
    const float delta_hat = (s->step_count == 0) ? 0.0f
        : (x_t - s->mu_prev) / (s->sigma_prev + hp.epsilon);
    s->delta_bar = hp.beta_delta * s->delta_bar + (1.0f - hp.beta_delta) * delta_hat;
    s->p = hp.beta_delta * s->p + (1.0f - hp.beta_delta) * safeSign(delta_hat);
    s->mu_prev = s->mu;

    // STEP 5: Soft outlier gating (σ-decoupled mirror: outlier_raw)
    s->outlier_raw = x_t - s->c_out;
    const float outlier_arg = (x_t - s->c_out) / (s->sigma + hp.epsilon);
    const float w_out = 1.0f / (1.0f + expf(-outlier_arg));
    s->r_out = hp.beta_r * s->r_out + (1.0f - hp.beta_r) * w_out;

    // STEP 6: Excess magnitude
    const float eta = fmaxf(0.0f, x_t - s->c_out);
    s->mu_ex = hp.beta_mu * s->mu_ex + (1.0f - hp.beta_mu) * eta;

    // STEP 7: Persistence
    s->ell_out = hp.beta_run * s->ell_out + (1.0f - hp.beta_run) * w_out;

    // STEP 8: Slow anchor drift
    s->mu_a = hp.beta_a * s->mu_a + (1.0f - hp.beta_a) * s->mu;
    s->sigma_a = hp.beta_a * s->sigma_a + (1.0f - hp.beta_a) * s->sigma;
    s->delta_mu = s->mu - s->mu_a;
    s->delta_sigma = s->sigma - s->sigma_a;

    // STEP 9: Update metadata
    s->step_count++;
    s->initialized = 1;
    level_0->last_update = global_step;

    if (strict_mode) {
        if (!isFiniteValue(s->mu) || !isFiniteValue(s->sigma) ||
            !isFiniteValue(s->delta_bar) || !isFiniteValue(s->v_sigma) ||
            !isFiniteValue(s->delta_bar_raw)) {
            atomicOr(error_flag, (int)TelemetryError::ERR_NAN_IN_STATE);
        }
    }

    //=========================================================================
    // Higher levels: Check stride and update if aligned
    //=========================================================================
    
    for (int level = 1; level < num_levels; ++level) {
        const uint32_t stride = 1u << level;
        
        if ((global_step % stride) != 0) {
            continue;
        }

        const int prev_level_idx = (level - 1) * num_streams + stream_idx;
        const TelemetryState* prev_state = &levels[prev_level_idx].state;
        const float x_k = prev_state->mu;
        
        const int level_k_idx = level * num_streams + stream_idx;
        LatticeLevelState* level_k = &levels[level_k_idx];
        TelemetryState* sk = &level_k->state;
        
        if (sk->step_count == 0) {
            sk->k_out = hp.k_out0;
        }

        // Bias-corrected EMA (same as Level 0)
        sk->mu_raw = hp.beta_mu * sk->mu_raw + (1.0f - hp.beta_mu) * x_k;
        sk->m2_raw = hp.beta_mu * sk->m2_raw + (1.0f - hp.beta_mu) * (x_k * x_k);
        
        sk->beta_mu_power *= hp.beta_mu;
        const float bc_k = 1.0f - sk->beta_mu_power;
        sk->mu = sk->mu_raw / bc_k;
        sk->m2 = sk->m2_raw / bc_k;
        
        const float raw_var_k = sk->m2 - sk->mu * sk->mu;
        const float scale_eps_k = hp.epsilon * sk->mu * sk->mu;
        const float var_k = fmaxf(raw_var_k, scale_eps_k);
        sk->sigma = sqrtf(fmaxf(var_k, 0.0f));
        sk->sigma_tilde = sk->sigma / (fabsf(sk->mu) + hp.epsilon);
        
        const float sd_k = sk->sigma - sk->sigma_prev;
        sk->sigma_jump = sd_k;
        sk->v_sigma = hp.beta_v * sk->v_sigma + (1.0f - hp.beta_v) * (sd_k * sd_k);
        sk->sigma_prev = sk->sigma;
        
        sk->k_out = hp.k_out0 * (1.0f + hp.alpha_v * safeSqrt(sk->v_sigma, hp.epsilon));
        sk->c_out = sk->mu + sk->k_out * sk->sigma;
        
        const float delta_raw_k = (sk->step_count == 0) ? 0.0f : (x_k - sk->mu_prev);
        sk->delta_raw = delta_raw_k;
        sk->delta_bar_raw = hp.beta_delta * sk->delta_bar_raw + (1.0f - hp.beta_delta) * delta_raw_k;
        const float dh_k = (sk->step_count == 0) ? 0.0f
            : (x_k - sk->mu_prev) / (sk->sigma_prev + hp.epsilon);
        sk->delta_bar = hp.beta_delta * sk->delta_bar + (1.0f - hp.beta_delta) * dh_k;
        sk->p = hp.beta_delta * sk->p + (1.0f - hp.beta_delta) * safeSign(dh_k);
        sk->mu_prev = sk->mu;
        
        sk->outlier_raw = x_k - sk->c_out;
        const float out_arg_k = (x_k - sk->c_out) / (sk->sigma + hp.epsilon);
        const float w_k = 1.0f / (1.0f + expf(-out_arg_k));
        sk->r_out = hp.beta_r * sk->r_out + (1.0f - hp.beta_r) * w_k;
        
        const float eta_k = fmaxf(0.0f, x_k - sk->c_out);
        sk->mu_ex = hp.beta_mu * sk->mu_ex + (1.0f - hp.beta_mu) * eta_k;
        
        sk->ell_out = hp.beta_run * sk->ell_out + (1.0f - hp.beta_run) * w_k;
        
        sk->mu_a = hp.beta_a * sk->mu_a + (1.0f - hp.beta_a) * sk->mu;
        sk->sigma_a = hp.beta_a * sk->sigma_a + (1.0f - hp.beta_a) * sk->sigma;
        sk->delta_mu = sk->mu - sk->mu_a;
        sk->delta_sigma = sk->sigma - sk->sigma_a;
        
        sk->step_count++;
        sk->initialized = 1;
        level_k->last_update = global_step;
    }
}

//=============================================================================
// EXTRACTION KERNEL (read telemetry vector)
//=============================================================================

__global__ void extractTelemetryVectorKernel(
    const LatticeLevelState* levels,
    TelemetryVector* output,
    int level,
    int stream_idx,
    int num_streams
) {
    const int idx = level * num_streams + stream_idx;
    const TelemetryState* s = &levels[idx].state;
    
    if (s->initialized == 0) {
        output->mu = 0.0f;
        output->sigma_tilde = 0.0f;
        output->v_sigma = 0.0f;
        output->delta_bar = 0.0f;
        output->p = 0.0f;
        output->r_out = 0.0f;
        output->ell_out = 0.0f;
        output->mu_ex = 0.0f;
        output->delta_mu = 0.0f;
        output->delta_sigma = 0.0f;
        return;
    }
    
    output->mu = s->mu;
    output->sigma_tilde = s->sigma_tilde;
    output->v_sigma = s->v_sigma;
    output->delta_bar = s->delta_bar;
    output->p = s->p;
    output->r_out = s->r_out;
    output->ell_out = s->ell_out;
    output->mu_ex = s->mu_ex;
    output->delta_mu = s->delta_mu;
    output->delta_sigma = s->delta_sigma;
}

//=============================================================================
// ANCHOR RESET KERNEL
//=============================================================================

__global__ void resetAnchorsKernel(
    LatticeLevelState* __restrict__ levels,
    int num_levels,
    int num_streams
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_states = num_levels * num_streams;
    
    if (idx >= total_states) return;
    
    TelemetryState* s = &levels[idx].state;
    s->mu_a = s->mu;
    s->sigma_a = s->sigma;
    s->delta_mu = 0.0f;
    s->delta_sigma = 0.0f;
}

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

const char* getTelemetryErrorMessage(TelemetryError err) {
    switch (err) {
        case TelemetryError::OK: return "Success";
        case TelemetryError::ERR_NAN_IN_INPUT: return "FATAL: NaN in input observation";
        case TelemetryError::ERR_INF_IN_INPUT: return "FATAL: Inf in input observation";
        case TelemetryError::ERR_NAN_IN_STATE: return "FATAL: NaN detected in telemetry state";
        case TelemetryError::ERR_KERNEL_LAUNCH: return "FATAL: CUDA kernel launch failed";
        case TelemetryError::ERR_NULL_POINTER: return "FATAL: NULL pointer";
        case TelemetryError::ERR_INVALID_PARAMS: return "FATAL: Invalid parameters";
        default: return "Unknown error";
    }
}

const char* getMetricStreamName(MetricStream stream) {
    switch (stream) {
        case MetricStream::LOSS: return "loss";
        case MetricStream::GRAD_NORM_MEAN: return "grad_norm_mean";
        case MetricStream::GRAD_NORM_MAX: return "grad_norm_max";
        case MetricStream::LEARNING_RATE: return "learning_rate";
        case MetricStream::TOKENS_PER_BATCH: return "tokens_per_batch";
        case MetricStream::RHO_FINAL: return "rho_final";
        case MetricStream::RHO_GROWTH: return "rho_growth";
        case MetricStream::RHO_WORST_DELTA: return "rho_worst_delta";
        case MetricStream::H_RMS_GROWTH: return "h_rms_growth";
        case MetricStream::ADAM_BC2_V_CONVERGENCE: return "adam_bc2_v_convergence";
        case MetricStream::ADAM_SIGNAL_DOMINANCE: return "adam_signal_dominance";
        case MetricStream::ADAM_CUMULATIVE_DISP: return "adam_cumulative_disp";
        case MetricStream::ADAM_DISRUPTION_EMB: return "adam_disruption_emb";
        case MetricStream::ADAM_INV_BC2_AMP: return "adam_inv_bc2_amp";
        case MetricStream::EXEC_GRAD_NORM: return "exec_grad_norm";
        case MetricStream::EXEC_GRAD_RATIO: return "exec_grad_ratio";
        case MetricStream::EXEC_SELECTION_ENTROPY: return "exec_selection_entropy";
        case MetricStream::EXEC_OP_ENTROPY: return "exec_op_entropy";
        case MetricStream::EXEC_DIV_CLAMP_RATE: return "exec_div_clamp_rate";
        case MetricStream::EXEC_MAX_P_WRITE: return "exec_max_p_write";
        case MetricStream::EXEC_ACTIVE_RATIO: return "exec_active_ratio";
        case MetricStream::EB_INJECT_GATE: return "eb_inject_gate";
        case MetricStream::EB_READ_GATE_MEAN: return "eb_read_gate_mean";
        case MetricStream::EB_INJECT_WEIGHT_NORM: return "eb_inject_weight_norm";
        case MetricStream::EB_READ_WEIGHT_NORM: return "eb_read_weight_norm";
        case MetricStream::EB_LOSS_FRAC: return "eb_loss_frac";
        case MetricStream::SB_ATOM_EMBED_RMS: return "sb_atom_embed_rms";
        case MetricStream::PBM_ALIBI_SLOPE_RMS: return "pbm_alibi_slope_rms";
        case MetricStream::PBM_ALIBI_EFF_BIAS_MAX: return "pbm_alibi_eff_bias_max";
        case MetricStream::PBM_ROPE_INV_FREQ_RMS: return "pbm_rope_inv_freq_rms";
        case MetricStream::PBM_BATCH_MAX_SEQ_LEN: return "pbm_batch_max_seq_len";
        case MetricStream::RHO_RAW_AVG_ABS_DOT: return "rho_raw_avg_abs_dot";
        case MetricStream::RHO_RAW_AVG_NORM_PROD: return "rho_raw_avg_norm_prod";
        case MetricStream::RHO_RAW_H_RMS_MIN: return "rho_raw_h_rms_min";
        case MetricStream::RHO_RAW_H_RMS_MAX: return "rho_raw_h_rms_max";
        case MetricStream::RMS_GAMMA_PRE_ATTN_RMS: return "rms_gamma_pre_attn_rms";
        case MetricStream::RMS_GAMMA_PRE_FFN_RMS: return "rms_gamma_pre_ffn_rms";
        case MetricStream::RMS_GAMMA_FINAL_RMS: return "rms_gamma_final_rms";
        case MetricStream::RHO_RAW_RMS_SPREAD: return "rho_raw_rms_spread";
        case MetricStream::HW_COS_RMS: return "hw_cos_rms";
        case MetricStream::HW_COS_SIGNED_MEAN: return "hw_cos_signed_mean";
        case MetricStream::HW_COS_ABS_MAX: return "hw_cos_abs_max";
        case MetricStream::HW_HBAR_WBAR_COS: return "hw_hbar_wbar_cos";
        case MetricStream::HW_H_DC_MEAN: return "hw_h_dc_mean";
        case MetricStream::HW_H_DC_ABS_MAX: return "hw_h_dc_abs_max";
        case MetricStream::UNIGRAM_DIR_COS_ABS_MEAN: return "unigram_dir_cos_abs_mean";
        case MetricStream::UNIGRAM_DIR_COS_SIGNED_MEAN: return "unigram_dir_cos_signed_mean";
        case MetricStream::LM_HEAD_W_RMS_RMS: return "lm_head_w_rms_rms";
        case MetricStream::INIT_TIE_CFG: return "init_tie_cfg";
        case MetricStream::INIT_TIE_PTRS_SAME: return "init_tie_ptrs_same";
        case MetricStream::INIT_TIE_GRADS_SAME: return "init_tie_grads_same";
        case MetricStream::INIT_LM_OWNS_WEIGHTS: return "init_lm_owns_weights";
        case MetricStream::INIT_OPT_GROUPS_TOTAL: return "init_opt_groups_total";
        case MetricStream::INIT_OPT_GROUPS_EMB: return "init_opt_groups_emb";
        case MetricStream::INIT_OPT_GROUPS_LM: return "init_opt_groups_lm";
        case MetricStream::RHO_RAW_AVG_SIGNED_DOT: return "rho_raw_avg_signed_dot";
        case MetricStream::RHO_CENTERED_AVG_ABS_DOT: return "rho_centered_avg_abs_dot";
        case MetricStream::RHO_MEAN_VECTOR_RMS: return "rho_mean_vector_rms";
        default: return "unknown";
    }
}

//=============================================================================
// CONSTRUCTOR (Pattern B: self-allocating)
//=============================================================================

TelemetryLattice::TelemetryLattice(const LatticeConfig& config)
    : config_(config)
{
    if (config_.num_levels <= 0 || config_.num_levels > kMaxLevels ||
        config_.num_streams <= 0 || config_.num_streams > kMaxStreams) {
        throw std::runtime_error("[Telemetry] FATAL: Invalid config: levels=" +
            std::to_string(config_.num_levels) + ", streams=" +
            std::to_string(config_.num_streams));
    }
    StreamController::fatalIfDefaultStream(config_.stream, "TelemetryLattice::TelemetryLattice");

    const size_t total_states = config_.num_levels * config_.num_streams;

    // Allocate struct array (non-float, managed manually)
    cudaError_t err = cudaMalloc(&levels_, total_states * sizeof(LatticeLevelState));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[Telemetry] FATAL: cudaMalloc levels failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMemsetAsync(levels_, 0, total_states * sizeof(LatticeLevelState), config_.stream);
    if (err != cudaSuccess) {
        cudaFree(levels_); levels_ = nullptr;
        throw std::runtime_error(std::string("[Telemetry] FATAL: cudaMemsetAsync failed: ") +
                                 cudaGetErrorString(err));
    }

    // Set strides for each level
    std::vector<LatticeLevelState> host_levels(total_states);
    for (int level = 0; level < config_.num_levels; ++level) {
        const uint32_t stride = 1u << level;
        for (int stream = 0; stream < config_.num_streams; ++stream) {
            int idx = level * config_.num_streams + stream;
            host_levels[idx].stride = stride;
            host_levels[idx].last_update = 0;
        }
    }
    err = cudaMemcpyAsync(levels_, host_levels.data(),
                          total_states * sizeof(LatticeLevelState),
                          cudaMemcpyHostToDevice, config_.stream);
    if (err != cudaSuccess) {
        cudaFree(levels_); levels_ = nullptr;
        throw std::runtime_error(std::string("[Telemetry] FATAL: cudaMemcpyAsync strides failed: ") +
                                 cudaGetErrorString(err));
    }

    // Pattern B: float buffers as GRIM::Tensor
    observations_ = GRIM::Tensor::zeros({config_.num_streams}, config_.stream,
                                        "telemetry.observations");
    scratch_vectors_ = GRIM::Tensor::zeros({config_.num_streams, 10}, config_.stream,
                                           "telemetry.scratch_vectors");

    // Error flag (int, not float — managed manually)
    err = cudaMalloc(&d_error_flag_, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(levels_); levels_ = nullptr;
        throw std::runtime_error(std::string("[Telemetry] FATAL: cudaMalloc error_flag failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(config_.stream);
    if (err != cudaSuccess) {
        releaseNonTensor();
        throw std::runtime_error(std::string("[Telemetry] FATAL: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(err));
    }

    fprintf(stdout, "[Telemetry] Lattice initialized: %d levels, %d streams, GPU-resident (Pattern B)\n",
            config_.num_levels, config_.num_streams);
}

//=============================================================================
// DESTRUCTOR
//=============================================================================

TelemetryLattice::~TelemetryLattice() {
    // Log any pending CUDA error before teardown (often explains cudaFree failures below)
    cudaError_t pending = cudaGetLastError();
    if (pending != cudaSuccess) {
        fprintf(stderr, "[Telemetry] Lattice teardown: GPU already in error state: %s. "
                "cudaFree failures below are a consequence; fix the earlier fault.\n",
                cudaGetErrorString(pending));
    }
    releaseNonTensor();
    // observations_ and scratch_vectors_ auto-release via Tensor destructor
    fprintf(stdout, "[Telemetry] Lattice freed\n");
}

void TelemetryLattice::releaseNonTensor() {
    if (levels_) {
        cudaError_t err = cudaFree(levels_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Telemetry] cudaFree(levels_) failed: %s (ptr=%p)\n",
                    cudaGetErrorString(err), (void*)levels_);
        }
        levels_ = nullptr;
    }
    if (d_error_flag_) {
        cudaError_t err = cudaFree(d_error_flag_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Telemetry] cudaFree(d_error_flag_) failed: %s (ptr=%p)\n",
                    cudaGetErrorString(err), (void*)d_error_flag_);
        }
        d_error_flag_ = nullptr;
    }
}

//=============================================================================
// MOVE OPERATIONS
//=============================================================================

TelemetryLattice::TelemetryLattice(TelemetryLattice&& other) noexcept
    : config_(other.config_),
      observations_(std::move(other.observations_)),
      scratch_vectors_(std::move(other.scratch_vectors_)),
      levels_(other.levels_),
      d_error_flag_(other.d_error_flag_)
{
    other.levels_ = nullptr;
    other.d_error_flag_ = nullptr;
}

TelemetryLattice& TelemetryLattice::operator=(TelemetryLattice&& other) noexcept {
    if (this != &other) {
        releaseNonTensor();

        config_ = other.config_;
        observations_ = std::move(other.observations_);
        scratch_vectors_ = std::move(other.scratch_vectors_);
        levels_ = other.levels_;
        d_error_flag_ = other.d_error_flag_;

        other.levels_ = nullptr;
        other.d_error_flag_ = nullptr;
    }
    return *this;
}

//=============================================================================
// UPDATE
//=============================================================================

TelemetryError TelemetryLattice::update(const float* observations, uint32_t global_step) {
    if (!observations) {
        fprintf(stderr, "[Telemetry] %s\n",
                getTelemetryErrorMessage(TelemetryError::ERR_NULL_POINTER));
        return TelemetryError::ERR_NULL_POINTER;
    }
    StreamController::fatalIfDefaultStream(config_.stream,
                                           "TelemetryLattice::update");

    cudaMemcpyAsync(observations_.data, observations,
                    config_.num_streams * sizeof(float),
                    cudaMemcpyHostToDevice, config_.stream);

    cudaMemsetAsync(d_error_flag_, 0, sizeof(int), config_.stream);

    const int num_blocks = (config_.num_streams + 255) / 256;
    updateTelemetryStateKernel<<<num_blocks, 256, 0, config_.stream>>>(
        levels_,
        observations_.data,
        config_.hyperparams,
        global_step,
        config_.num_levels,
        config_.num_streams,
        config_.hyperparams.strict_mode,
        d_error_flag_
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Telemetry] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return TelemetryError::ERR_KERNEL_LAUNCH;
    }

    cudaStreamSynchronize(config_.stream);

    int host_error_flag = 0;
    cudaMemcpy(&host_error_flag, d_error_flag_, sizeof(int), cudaMemcpyDeviceToHost);

    if (host_error_flag != 0) {
        TelemetryError error = static_cast<TelemetryError>(host_error_flag);
        fprintf(stderr, "[Telemetry] %s at step %u\n",
                getTelemetryErrorMessage(error), global_step);
        return error;
    }

    return TelemetryError::OK;
}

TelemetryError TelemetryLattice::updateFromBatch(
    const GRIM::Batching::BatchPayload& payload,
    float loss, float grad_norm, float learning_rate,
    uint32_t global_step)
{
    if (payload.token_stats.total_tokens <= 0) {
        fprintf(stderr, "[Telemetry] FATAL: BatchPayload.token_stats.total_tokens=%lld (must be > 0)\n",
                static_cast<long long>(payload.token_stats.total_tokens));
        return TelemetryError::ERR_INVALID_PARAMS;
    }
    const float tokens = static_cast<float>(payload.token_stats.total_tokens);
    float obs[5] = { loss, grad_norm, grad_norm, learning_rate, tokens };
    return update(obs, global_step);
}

//=============================================================================
// READ OPERATIONS
//=============================================================================

TelemetryError TelemetryLattice::readVector(
    int level, int stream_idx, TelemetryVector* out_vector) const
{
    if (!out_vector) {
        return TelemetryError::ERR_NULL_POINTER;
    }
    if (level < 0 || level >= config_.num_levels ||
        stream_idx < 0 || stream_idx >= config_.num_streams) {
        return TelemetryError::ERR_INVALID_PARAMS;
    }

    auto* scratch = reinterpret_cast<TelemetryVector*>(scratch_vectors_.data);
    TelemetryVector* d_temp = &scratch[0];

    extractTelemetryVectorKernel<<<1, 1, 0, config_.stream>>>(
        levels_, d_temp, level, stream_idx, config_.num_streams
    );

    cudaStreamSynchronize(config_.stream);
    cudaMemcpy(out_vector, d_temp, sizeof(TelemetryVector), cudaMemcpyDeviceToHost);

    return TelemetryError::OK;
}

TelemetryError TelemetryLattice::readBatched(
    int level0, int stream_idx0, TelemetryVector* out_vector0,
    int level1, int stream_idx1, TelemetryVector* out_vector1,
    int level2, int stream_idx2, TelemetryVector* out_vector2) const
{
    if (!out_vector0 || !out_vector1 || !out_vector2) {
        return TelemetryError::ERR_NULL_POINTER;
    }

    if (level0 < 0 || level0 >= config_.num_levels ||
        stream_idx0 < 0 || stream_idx0 >= config_.num_streams) {
        return TelemetryError::ERR_INVALID_PARAMS;
    }
    if (level1 < 0 || level1 >= config_.num_levels ||
        stream_idx1 < 0 || stream_idx1 >= config_.num_streams) {
        return TelemetryError::ERR_INVALID_PARAMS;
    }
    if (level2 < 0 || level2 >= config_.num_levels ||
        stream_idx2 < 0 || stream_idx2 >= config_.num_streams) {
        return TelemetryError::ERR_INVALID_PARAMS;
    }

    auto* scratch = reinterpret_cast<TelemetryVector*>(scratch_vectors_.data);
    TelemetryVector* d_temp0 = &scratch[0];
    TelemetryVector* d_temp1 = &scratch[1];
    TelemetryVector* d_temp2 = &scratch[2];

    extractTelemetryVectorKernel<<<1, 1, 0, config_.stream>>>(
        levels_, d_temp0, level0, stream_idx0, config_.num_streams
    );
    extractTelemetryVectorKernel<<<1, 1, 0, config_.stream>>>(
        levels_, d_temp1, level1, stream_idx1, config_.num_streams
    );
    extractTelemetryVectorKernel<<<1, 1, 0, config_.stream>>>(
        levels_, d_temp2, level2, stream_idx2, config_.num_streams
    );

    cudaStreamSynchronize(config_.stream);

    cudaMemcpy(out_vector0, d_temp0, sizeof(TelemetryVector), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_vector1, d_temp1, sizeof(TelemetryVector), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_vector2, d_temp2, sizeof(TelemetryVector), cudaMemcpyDeviceToHost);

    return TelemetryError::OK;
}

TelemetryError TelemetryLattice::readState(
    int level, int stream_idx, TelemetryState* out_state) const
{
    if (!out_state) {
        return TelemetryError::ERR_NULL_POINTER;
    }
    if (level < 0 || level >= config_.num_levels ||
        stream_idx < 0 || stream_idx >= config_.num_streams) {
        return TelemetryError::ERR_INVALID_PARAMS;
    }

    const int idx = level * config_.num_streams + stream_idx;

    cudaMemcpy(out_state,
               &levels_[idx].state,
               sizeof(TelemetryState),
               cudaMemcpyDeviceToHost);

    return TelemetryError::OK;
}

//=============================================================================
// ANCHOR RESET
//=============================================================================

TelemetryError TelemetryLattice::resetAnchors(cudaStream_t stream) {
    if (!levels_) {
        return TelemetryError::ERR_NULL_POINTER;
    }
    StreamController::fatalIfDefaultStream(stream, "TelemetryLattice::resetAnchors");
    
    const int total_states = config_.num_levels * config_.num_streams;
    const int threads = 256;
    const int blocks = (total_states + threads - 1) / threads;
    
    resetAnchorsKernel<<<blocks, threads, 0, stream>>>(
        levels_, config_.num_levels, config_.num_streams
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Telemetry] resetAnchorsKernel launch failed: %s\n", cudaGetErrorString(err));
        return TelemetryError::ERR_KERNEL_LAUNCH;
    }
    
    return TelemetryError::OK;
}

} // namespace GRIM::Telemetry
