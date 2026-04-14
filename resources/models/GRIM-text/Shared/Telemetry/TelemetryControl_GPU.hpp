#pragma once
/**
 * @file TelemetryControl_GPU.hpp
 * @brief Monitoring-only telemetry observer (GPU-native)
 * 
 * ARCHITECTURE:
 *   - Single fused kernel reads telemetry + computes diagnostics on GPU
 *   - Only ONE small D2H transfer (ControlDecision struct)
 *   - State kept GPU-resident between calls
 *   - NO per-call allocations (Rule 22)
 * 
 * DESIGN PRINCIPLES (Rule 20 — Fail Loud):
 *   This system is MONITORING ONLY. It does NOT modify training behavior.
 *   If something is wrong (gradient explosion, data corruption), the correct
 *   response is to CRASH with a clear error, not silently skip/scale/inject.
 *
 *   The ONLY intervention is FatalError on accumulation bugs (zero gradients
 *   with non-zero loss = disconnected autograd graph = must crash).
 *
 *   Spike severity, ratio, and flags are DIAGNOSTIC — logged for analysis,
 *   never acted upon. If spikes are frequent, fix the root cause.
 * 
 * USAGE:
 *   TelemetryControl ctrl(config);
 *   ctrl.initGPU(stream);
 *   auto decision = ctrl.evaluate(lattice, grad_norm, loss, valid_tokens, seq_len, step);
 *   // decision.action is always Continue or FatalError
 *   // decision.spike_severity/spike_ratio are for logging only
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
    
    // Spike detection thresholds (DIAGNOSTIC ONLY — logged, never acted upon)
    float spike_mild_threshold = 3.0f;        // > 3x baseline = mild
    float spike_moderate_threshold = 5.0f;    // > 5x baseline = moderate
    float spike_severe_threshold = 10.0f;     // > 10x baseline = severe
    
    // Accumulation bug detection (Rule 20: crash on disconnected autograd)
    float min_grad_for_nonzero_loss = 1e-10f; // If loss > threshold, grad should exceed this
    float loss_threshold_for_grad_check = 0.01f;
    int max_consecutive_zero_grad_steps = 3;  // FATAL after this many zero-grad steps
    
    // Warmup and baseline stabilization
    int warmup_steps = 100;                        // Skip spike classification during LR warmup
    int baseline_stabilization_steps = 50;         // Additional steps for telemetry baseline to stabilize
    
    // Logging
    bool verbose_logging = true;
    bool fail_loud_on_accumulation_bug = true;
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
    Continue = 0,           // Normal operation (monitoring only)
    FatalError = 1          // Accumulation bug — crash immediately (Rule 20)
};

const char* getControlActionName(ControlAction action);

//=============================================================================
// CONTROL DECISION (GPU-resident output, small D2H transfer)
//=============================================================================

struct alignas(16) ControlDecision {
    // Primary action — always Continue or FatalError (4 bytes)
    ControlAction action = ControlAction::Continue;
    
    // Spike severity — DIAGNOSTIC ONLY, never acted upon (4 bytes)
    SpikeSeverity spike_severity = SpikeSeverity::None;
    
    // Bit flags for detected conditions (4 bytes)
    uint32_t flags = 0;
    static constexpr uint32_t FLAG_ACCUMULATION_BUG   = 1 << 0;
    
    // Diagnostic telemetry values (12 bytes)
    float spike_ratio = 0.0f;           // current / baseline (for logging)
    float normalized_grad = 0.0f;       // Token-normalized gradient
    int error_code = 0;
    
    // Reserved (20 bytes padding to maintain 48-byte D2H transfer)
    int _reserved[5] = {};
    
    // Helper accessor (CPU-side only)
    bool accumulationBugDetected() const { return flags & FLAG_ACCUMULATION_BUG; }
};
static_assert(sizeof(ControlDecision) == 48, "ControlDecision must be 48 bytes for efficient transfer");

//=============================================================================
// GPU-RESIDENT STATE (persists between calls, no per-call allocations)
//=============================================================================

struct alignas(16) TelemetryControlState_GPU {
    // Accumulation bug tracking
    int consecutive_zero_grad_steps = 0;
    
    // Step counter
    uint32_t step_count = 0;
    
    // Reserved (padding to 64 bytes)
    int _reserved[14] = {};
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
     * @param stream CUDA stream for async memset
     */
    void reset(cudaStream_t stream);
    
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
    bool gpu_state_initialized_ = false;              // True after first evaluate() copies config/state
    
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

} // namespace GRIM::Telemetry
