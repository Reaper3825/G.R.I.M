#pragma once
/**
 * @file TelemetryControl_GPU.hpp
 * @brief Production-grade telemetry-driven adaptive control (GPU-native)
 * 
 * ARCHITECTURE:
 *   - Single fused kernel reads telemetry + computes decision on GPU
 *   - Only ONE small D2H transfer (ControlDecision struct)
 *   - State kept GPU-resident between calls
 *   - NO per-call allocations (Rule 22)
 * 
 * DESIGN GOALS:
 *   1. Fair comparison across batches (normalize by tokens/seq_len)
 *   2. Gradient spike detection with severity levels
 *   3. Accumulation bug detection (zero gradients with non-zero loss)
 *   4. False alarm prevention for long sequence transitions
 *   5. Multi-timescale coherent decisions using TelemetryLattice
 * 
 * USAGE:
 *   TelemetryControl ctrl(config);
 *   ctrl.initGPU(stream);
 *   auto decision = ctrl.evaluate(lattice, grad_norm, loss, valid_tokens, seq_len, step);
 *   if (decision.action == ControlAction::SkipStep) { ... }
 *   grad_scale *= decision.grad_scale_factor;
 */

#include "TelemetryLattice_GPU.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace GRIM::Telemetry {

//=============================================================================
// CONFIGURATION (CPU-side, copied to GPU at init)
//=============================================================================

struct TelemetryControlConfig {
    // Reference values for normalization
    float reference_seq_len = 512.0f;         // Baseline sequence length
    float reference_tokens = 720.0f;          // Baseline tokens per batch
    
    // Spike detection thresholds (multiples of baseline)
    float spike_mild_threshold = 3.0f;        // > 3x baseline = mild
    float spike_moderate_threshold = 5.0f;    // > 5x baseline = moderate
    float spike_severe_threshold = 10.0f;     // > 10x baseline = severe
    
    // Spike response
    float moderate_grad_scale = 0.5f;         // Scale gradients by 50% on moderate spike
    int moderate_cooldown_extension = 3;      // Extend cooldown by 3 steps
    
    // Accumulation bug detection
    float min_grad_for_nonzero_loss = 1e-10f; // If loss > threshold, grad should exceed this
    float loss_threshold_for_grad_check = 0.01f;
    int max_consecutive_zero_grad_steps = 3;  // FATAL after this many zero-grad steps
    
    // False alarm prevention
    float seq_len_regime_change_threshold = 0.3f;  // 30% change triggers suppression
    int regime_change_suppression_steps = 2;       // Suppress alarms for N steps
    
    // Volatility-based scaling - DISABLED (was causing plateau by damping gradients 35%)
    float volatility_damping_threshold = 100.0f;   // Effectively disabled (unreachable threshold)
    float max_volatility_damping = 1.0f;           // No damping (was 0.5 = 50% reduction)
    
    // Gradient health (decay detection) - DISABLED (normal gradient decay during convergence is expected)
    float gradient_decay_threshold = 0.0f;         // Disabled (was 0.1 = 10% of baseline triggers decay)
    float max_decay_boost = 1.0f;                  // No boost (was 1.5x)
    
    // Progress acceleration - DISABLED (loss fluctuations during plateau falsely trigger boost,
    // creating feedback loop: dip→boost→overshoot→dip→boost. This prevented convergence.)
    float progress_boost_threshold = 100.0f;       // Disabled (unreachable: would need 100-sigma drop)
    float max_progress_boost = 1.0f;               // No boost (was 1.3x)
    
    // Outlier-based soft restart triggers
    float outlier_frequency_trigger = 0.95f;       // r_out threshold (raised: 95% outliers required)
    float outlier_persistence_trigger = 0.90f;     // ell_out threshold (raised: persistent outliers only)
    
    // Drift detection
    float anchor_drift_sigma_multiplier = 5.0f;    // |delta_mu| > N * sigma_tilde triggers (raised: 5-sigma events only)
    
    // Soft restart cooldown
    int soft_restart_cooldown_steps = 10;          // Wait N batches before allowing another soft restart
    
    // Warmup and baseline stabilization
    int warmup_steps = 100;                        // Skip decay detection during LR warmup
    int baseline_stabilization_steps = 50;         // Additional steps after warmup for telemetry baseline to stabilize
    
    // Logging
    bool verbose_logging = true;
    bool fail_loud_on_accumulation_bug = true;
    
    // Plateau noise injection - escape local minima by perturbing weights
    bool plateau_noise_enabled = true;             // Master switch
    int plateau_noise_patience = 50;               // Consecutive low-variance batches before inject
    float plateau_noise_variance_threshold = 0.001f; // Loss variance below this = plateau
    float plateau_noise_std = 0.001f;              // Base noise standard deviation
    bool plateau_noise_proportional = true;        // Scale by |weight| if true, else fixed
    int plateau_noise_cooldown = 500;              // Min batches between injections
    int plateau_noise_max_per_epoch = 3;           // Safety limit
};

//=============================================================================
// SPIKE SEVERITY (GPU-compatible enum)
//=============================================================================

enum class SpikeSeverity : int {
    None = 0,
    Mild = 1,
    Moderate = 2,
    Severe = 3
};

const char* getSpikeSeverityName(SpikeSeverity severity);

//=============================================================================
// CONTROL ACTION (GPU-compatible enum)
//=============================================================================

enum class ControlAction : int {
    Continue = 0,           // Normal operation
    ScaleGradients = 1,     // Apply grad_scale_factor
    ExtendCooldown = 2,     // Extend LR cooldown
    SkipStep = 3,           // Skip optimizer step entirely
    TriggerSoftRestart = 4, // Reset optimizer momentum
    InjectPlateauNoise = 5, // Inject Gaussian noise into weights to escape plateau
    FatalError = 6          // Accumulation bug or unrecoverable state
};

const char* getControlActionName(ControlAction action);

//=============================================================================
// CONTROL DECISION (GPU-resident output, small D2H transfer)
//=============================================================================

struct alignas(16) ControlDecision {
    // Primary action (4 bytes)
    ControlAction action = ControlAction::Continue;
    
    // Gradient scaling - always applied, default 1.0 (4 bytes)
    float grad_scale_factor = 1.0f;
    
    // LR cooldown extension - 0 = no extension (4 bytes)
    int cooldown_extension = 0;
    
    // Spike severity (4 bytes)
    SpikeSeverity spike_severity = SpikeSeverity::None;
    
    // Bit flags for detected conditions (4 bytes)
    uint32_t flags = 0;
    static constexpr uint32_t FLAG_ACCUMULATION_BUG   = 1 << 0;
    static constexpr uint32_t FLAG_REGIME_CHANGE      = 1 << 1;
    static constexpr uint32_t FLAG_OUTLIER_REGIME     = 1 << 2;
    static constexpr uint32_t FLAG_ANCHOR_DRIFT       = 1 << 3;
    static constexpr uint32_t FLAG_GRADIENT_DECAY     = 1 << 4;
    static constexpr uint32_t FLAG_RAPID_PROGRESS     = 1 << 5;
    static constexpr uint32_t FLAG_PLATEAU_NOISE      = 1 << 6;
    
    // Telemetry values used for decision (20 bytes)
    float spike_ratio = 0.0f;           // current / baseline
    float volatility_damping = 1.0f;    // Applied volatility damping
    float decay_boost = 1.0f;           // Applied decay boost
    float progress_boost = 1.0f;        // Applied progress boost (reward rapid improvement)
    float normalized_grad = 0.0f;       // Token-normalized gradient
    
    // Error code from kernel (4 bytes)
    int error_code = 0;
    
    // Padding to align to 48 bytes total
    int _pad0 = 0;
    
    // Helper accessors (CPU-side only)
    bool accumulationBugDetected() const { return flags & FLAG_ACCUMULATION_BUG; }
    bool regimeChangeDetected() const { return flags & FLAG_REGIME_CHANGE; }
    bool outlierRegimeActive() const { return flags & FLAG_OUTLIER_REGIME; }
    bool anchorDriftDetected() const { return flags & FLAG_ANCHOR_DRIFT; }
    bool gradientDecayDetected() const { return flags & FLAG_GRADIENT_DECAY; }
    bool rapidProgressDetected() const { return flags & FLAG_RAPID_PROGRESS; }
    bool plateauNoiseTriggered() const { return flags & FLAG_PLATEAU_NOISE; }
};
static_assert(sizeof(ControlDecision) == 48, "ControlDecision must be 48 bytes for efficient transfer");

//=============================================================================
// GPU-RESIDENT STATE (persists between calls, no per-call allocations)
//=============================================================================

struct alignas(16) TelemetryControlState_GPU {
    // Regime change tracking (20 bytes)
    int suppression_steps_remaining = 0;
    float last_avg_seq_len = 0.0f;
    float last_avg_tokens = 0.0f;
    int consecutive_zero_grad_steps = 0;
    int soft_restart_cooldown = 0;
    
    // Step counter (4 bytes)
    uint32_t step_count = 0;
    
    // Plateau noise state (28 bytes)
    int consecutive_low_variance_batches = 0;  // Counts plateau duration
    int batches_since_noise_injection = 0;     // Cooldown counter
    int noise_injections_this_epoch = 0;       // Per-epoch limit
    int total_noise_injections = 0;            // Lifetime counter
    float loss_sum = 0.0f;                     // Rolling sum for variance (Welford's)
    float loss_sum_sq = 0.0f;                  // Rolling sum of squares
    int loss_count = 0;                        // Number of losses in window
    
    // Padding to 64 bytes
    int _pad[1] = {0};
};
static_assert(sizeof(TelemetryControlState_GPU) == 64, "GPU state must be 64 bytes");

//=============================================================================
// KERNEL INPUT (passed to GPU, small H2D)
//=============================================================================

struct alignas(16) ControlKernelInput {
    float raw_grad_norm;
    float loss;
    float avg_seq_len;
    int valid_tokens;
    uint32_t global_step;
    int _pad[3];
};
static_assert(sizeof(ControlKernelInput) == 32, "Kernel input must be 32 bytes");

//=============================================================================
// TELEMETRY CONTROL CLASS (manages GPU resources)
//=============================================================================

class TelemetryControl {
public:
    explicit TelemetryControl(const TelemetryControlConfig& config = {});
    ~TelemetryControl();
    
    // Non-copyable, movable
    TelemetryControl(const TelemetryControl&) = delete;
    TelemetryControl& operator=(const TelemetryControl&) = delete;
    TelemetryControl(TelemetryControl&&) noexcept;
    TelemetryControl& operator=(TelemetryControl&&) noexcept;
    
    /**
     * @brief Initialize GPU resources (call once after construction)
     * Stream is NOT cached - must be passed to evaluate() on every call (Rule 22)
     */
    void initGPU();
    
    /**
     * @brief Check if GPU resources are initialized
     */
    bool isInitialized() const { return d_config_ != nullptr; }
    
    /**
     * @brief Evaluate telemetry and produce control decision (GPU-accelerated)
     * 
     * Single fused kernel:
     *   1. Reads 5 telemetry vectors from lattice (GPU memory)
     *   2. Computes all checks (spike, accumulation, regime, drift)
     *   3. Produces ControlDecision (GPU→CPU transfer at end)
     * 
     * @param lattice       TelemetryLattice (GPU-resident)
     * @param raw_grad_norm Raw gradient norm (before any scaling)
     * @param loss          Current batch loss
     * @param valid_tokens  Number of valid (non-masked) tokens in batch
     * @param avg_seq_len   Average sequence length in batch
     * @param global_step   Current training step
     * @return              ControlDecision with action and scaling factors
     */
    ControlDecision evaluate(
        const TelemetryLattice* lattice,
        float raw_grad_norm,
        float loss,
        int valid_tokens,
        float avg_seq_len,
        uint32_t global_step,
        cudaStream_t stream
    );
    
    /**
     * @brief Reset state (call at epoch boundary)
     */
    void reset();
    
    /**
     * @brief Get config (CPU-side copy)
     */
    const TelemetryControlConfig& config() const { return config_; }
    
    /**
     * @brief Get description of last decision (for logging)
     */
    std::string describeDecision(const ControlDecision& d) const;
    
private:
    TelemetryControlConfig config_;
    // NO stream storage - stream passed explicitly to evaluate()
    
    // GPU-resident buffers (allocated once in initGPU)
    TelemetryControlConfig* d_config_ = nullptr;      // Config mirror
    TelemetryControlState_GPU* d_state_ = nullptr;    // Persistent state
    ControlDecision* d_decision_ = nullptr;           // Output buffer
    ControlKernelInput* d_input_ = nullptr;           // Input buffer
    
    void freeGPU();
};

//=============================================================================
// KERNEL LAUNCH (internal - called by TelemetryControl::evaluate)
//=============================================================================

void launchControlDecisionKernel(
    const LatticeLevelState* d_lattice_levels,
    int num_lattice_streams,
    const TelemetryControlConfig* d_config,
    TelemetryControlState_GPU* d_state,
    const ControlKernelInput* d_input,
    ControlDecision* d_decision,
    cudaStream_t stream
);

//=============================================================================
// PLATEAU NOISE INJECTION (GPU kernel)
//=============================================================================

/**
 * @brief Inject Gaussian noise into weight buffer to escape plateau
 * 
 * Uses cuRAND Philox for high-quality random numbers.
 * Two modes: fixed (noise_std) or proportional (noise_std * |weight|)
 * 
 * @param weights       GPU buffer to perturb (in-place)
 * @param num_elements  Number of float elements
 * @param noise_std     Base noise standard deviation
 * @param proportional  If true, scale noise by |weight|
 * @param seed          RNG seed (combine global_step + injection_count)
 * @param stream        CUDA stream
 */
void launchPlateauNoiseInjection(
    float* weights,
    size_t num_elements,
    float noise_std,
    bool proportional,
    uint64_t seed,
    cudaStream_t stream
);

} // namespace GRIM::Telemetry
