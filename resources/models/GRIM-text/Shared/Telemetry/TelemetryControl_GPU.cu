/**
 * @file TelemetryControl_GPU.cu
 * @brief Monitoring-only telemetry observer (GPU-native)
 * 
 * ARCHITECTURE:
 *   - Single fused kernel computes diagnostics on GPU
 *   - Only ONE 48-byte D2H transfer per evaluate() call
 *   - State persists GPU-resident between calls
 *   - NO per-call allocations (Rule 22 compliant)
 *
 * Rule 20: This system is MONITORING ONLY. The only intervention is
 * FatalError on accumulation bugs. All other conditions are diagnostic.
 */

#include "TelemetryControl_GPU.hpp"
#include "TelemetryLattice_Internal.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM::Telemetry {

//=============================================================================
// STRING HELPERS (CPU-side)
//=============================================================================

const char* getSpikeSeverityName(SpikeSeverity severity) {
    switch (severity) {
        case SpikeSeverity::None:     return "none";
        case SpikeSeverity::Mild:     return "mild";
        case SpikeSeverity::Moderate: return "moderate";
        case SpikeSeverity::Severe:   return "severe";
        default: return "unknown";
    }
}

const char* getControlActionName(ControlAction action) {
    switch (action) {
        case ControlAction::Continue:           return "continue";
        case ControlAction::FatalError:         return "fatal_error";
        default: return "unknown";
    }
}

//=============================================================================
// DEVICE HELPER FUNCTIONS
//=============================================================================

__device__ __forceinline__ float d_computeNormalizedGrad(
    float raw_grad_norm, 
    int valid_tokens, 
    float reference_tokens
) {
    // CRITICAL: backward() scales each gradient by 1/valid_tokens, but the L2 norm
    // (sum of squared gradients) is still proportional to sqrt(valid_tokens) because
    // fewer tokens = less gradient energy overall.
    // 
    // To fairly compare gradients across batches with different token counts, we must
    // normalize by sqrt(reference_tokens/valid_tokens). This makes grad_norm independent
    // of batch size, so diagnostics can detect true pathologies vs token variance.
    
    if (valid_tokens <= 0 || reference_tokens <= 0.0f) return raw_grad_norm;
    const float scale = sqrtf(reference_tokens / static_cast<float>(valid_tokens));
    return raw_grad_norm * scale;
}

__device__ __forceinline__ SpikeSeverity d_computeSpikeSeverity(
    float current_grad, 
    float baseline_grad,
    const TelemetryControlConfig* cfg
) {
    if (baseline_grad < 1e-8f) return SpikeSeverity::None;
    
    const float ratio = current_grad / baseline_grad;
    
    if (ratio >= cfg->spike_severe_threshold) return SpikeSeverity::Severe;
    if (ratio >= cfg->spike_moderate_threshold) return SpikeSeverity::Moderate;
    if (ratio >= cfg->spike_mild_threshold) return SpikeSeverity::Mild;
    return SpikeSeverity::None;
}

__device__ __forceinline__ bool d_checkAccumulationBug(
    float grad_norm, 
    float loss,
    const TelemetryControlConfig* cfg
) {
    return (loss > cfg->loss_threshold_for_grad_check && 
            grad_norm < cfg->min_grad_for_nonzero_loss);
}

//=============================================================================
// MONITORING-ONLY DIAGNOSTIC KERNEL
//
// Single thread kernel - decision logic is sequential, not data-parallel.
// Benefits: 
//   - Reads telemetry directly from GPU memory (no D2H for vectors)
//   - Updates state GPU-resident
//   - Only final ControlDecision needs D2H transfer
//
// Rule 20: The ONLY intervention is FatalError on accumulation bugs.
// Everything else is diagnostic (spike severity, ratio) for logging.
//=============================================================================

__global__ void controlDecisionKernel(
    const LatticeLevelState* __restrict__ d_lattice_levels,
    int num_lattice_streams,
    const TelemetryControlConfig* __restrict__ cfg,
    TelemetryControlState_GPU* __restrict__ state,
    const ControlKernelInput* __restrict__ input,
    ControlDecision* __restrict__ decision
) {
    // Single-thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize decision to defaults
    // CRITICAL: CUDA device memory is NOT automatically initialized!
    decision->action = ControlAction::Continue;
    decision->spike_severity = SpikeSeverity::None;
    decision->flags = 0;
    decision->spike_ratio = 0.0f;
    decision->normalized_grad = 0.0f;
    decision->error_code = 0;
    for (int i = 0; i < 5; ++i) decision->_reserved[i] = 0;
    
    // Increment step counter
    state->step_count++;
    
    // Unpack input
    const float raw_grad_norm = input->raw_grad_norm;
    const float loss = input->loss;
    const int valid_tokens = input->valid_tokens;
    
    // Compute normalized gradient (diagnostic)
    decision->normalized_grad = d_computeNormalizedGrad(raw_grad_norm, valid_tokens, cfg->reference_tokens);
    
    //=========================================================================
    // CHECK 1: Accumulation Bug (FATAL — Rule 20)
    // Zero gradients with non-zero loss = disconnected autograd graph.
    // This MUST crash. There is no recovery.
    //=========================================================================
    
    if (d_checkAccumulationBug(raw_grad_norm, loss, cfg)) {
        state->consecutive_zero_grad_steps++;
        
        if (state->consecutive_zero_grad_steps >= cfg->max_consecutive_zero_grad_steps) {
            decision->flags |= ControlDecision::FLAG_ACCUMULATION_BUG;
            decision->action = ControlAction::FatalError;
            decision->error_code = 1;  // Accumulation bug
            return;
        }
    } else {
        state->consecutive_zero_grad_steps = 0;
    }
    
    //=========================================================================
    // If no lattice, return baseline decision
    //=========================================================================
    
    if (d_lattice_levels == nullptr) {
        return;
    }
    
    //=========================================================================
    // READ TELEMETRY FOR DIAGNOSTICS (direct GPU memory access)
    //=========================================================================
    
    const int num_streams = num_lattice_streams;
    
    // Level 0 (immediate) grad stream
    const int idx_L0_grad = 0 * num_streams + 1;
    const float L0_grad_mu = d_lattice_levels[idx_L0_grad].state.mu;
    
    // Level 2 (4-step baseline) grad stream
    const int idx_L2_grad = 2 * num_streams + 1;
    const float L2_grad_mu = d_lattice_levels[idx_L2_grad].state.mu;
    
    //=========================================================================
    // DIAGNOSTIC: Gradient Spike Classification (logged, NEVER acted upon)
    //=========================================================================
    
    const float raw_baseline = (L2_grad_mu > 1e-8f) ? L2_grad_mu : L0_grad_mu;
    const float normalized_baseline = d_computeNormalizedGrad(raw_baseline, cfg->reference_tokens, cfg->reference_tokens);
    const float baseline_grad = (normalized_baseline > 1e-8f) ? normalized_baseline : decision->normalized_grad;
    decision->spike_ratio = (baseline_grad > 1e-8f) 
        ? decision->normalized_grad / baseline_grad 
        : 1.0f;
    
    // Classify spike severity (for logging only)
    const uint32_t min_step_for_classification = cfg->warmup_steps + cfg->baseline_stabilization_steps;
    if (input->global_step >= min_step_for_classification) {
        decision->spike_severity = d_computeSpikeSeverity(
            decision->normalized_grad, baseline_grad, cfg);
    }
    
    // Action is always Continue (monitoring only)
    decision->action = ControlAction::Continue;
}

//=============================================================================
// KERNEL LAUNCH WRAPPER
//=============================================================================

void launchControlDecisionKernel(
    const LatticeLevelState* d_lattice_levels,
    int num_lattice_streams,
    const TelemetryControlConfig* d_config,
    TelemetryControlState_GPU* d_state,
    const ControlKernelInput* d_input,
    ControlDecision* d_decision,
    cudaStream_t stream
) {
    // Single thread kernel - decision logic is sequential
    controlDecisionKernel<<<1, 1, 0, stream>>>(
        d_lattice_levels, num_lattice_streams, d_config, d_state, d_input, d_decision
    );
}

//=============================================================================
// CLASS IMPLEMENTATION
//=============================================================================

TelemetryControl::TelemetryControl(const TelemetryControlConfig& config)
    : config_(config)
{
}

TelemetryControl::~TelemetryControl() {
    freeGPU();
}

TelemetryControl::TelemetryControl(TelemetryControl&& other) noexcept
    : config_(other.config_)
    , d_config_(other.d_config_)
    , d_state_(other.d_state_)
    , d_decision_(other.d_decision_)
    , d_input_(other.d_input_)
    , gpu_state_initialized_(other.gpu_state_initialized_)
{
    other.d_config_ = nullptr;
    other.d_state_ = nullptr;
    other.d_decision_ = nullptr;
    other.d_input_ = nullptr;
    other.gpu_state_initialized_ = false;
}

TelemetryControl& TelemetryControl::operator=(TelemetryControl&& other) noexcept {
    if (this != &other) {
        freeGPU();
        config_ = other.config_;
        d_config_ = other.d_config_;
        d_state_ = other.d_state_;
        d_decision_ = other.d_decision_;
        d_input_ = other.d_input_;
        gpu_state_initialized_ = other.gpu_state_initialized_;
        other.d_config_ = nullptr;
        other.d_state_ = nullptr;
        other.d_decision_ = nullptr;
        other.d_input_ = nullptr;
        other.gpu_state_initialized_ = false;
    }
    return *this;
}

void TelemetryControl::initGPU() {
    if (d_config_ != nullptr) {
        throw std::runtime_error("TelemetryControl::initGPU called twice (double-init)");
    }
    
    // Allocate GPU buffers (no stream ops - pure allocation)
    cudaError_t err;
    
    err = cudaMalloc(&d_config_, sizeof(TelemetryControlConfig));
    if (err != cudaSuccess) {
        throw std::runtime_error("TelemetryControl: Failed to allocate d_config_");
    }
    
    err = cudaMalloc(&d_state_, sizeof(TelemetryControlState_GPU));
    if (err != cudaSuccess) {
        freeGPU();
        throw std::runtime_error("TelemetryControl: Failed to allocate d_state_");
    }
    
    err = cudaMalloc(&d_decision_, sizeof(ControlDecision));
    if (err != cudaSuccess) {
        freeGPU();
        throw std::runtime_error("TelemetryControl: Failed to allocate d_decision_");
    }
    
    err = cudaMalloc(&d_input_, sizeof(ControlKernelInput));
    if (err != cudaSuccess) {
        freeGPU();
        throw std::runtime_error("TelemetryControl: Failed to allocate d_input_");
    }
    
    // Config/state init deferred to first evaluate() call with proper stream
}

void TelemetryControl::freeGPU() {
    if (d_config_) { cudaFree(d_config_); d_config_ = nullptr; }
    if (d_state_) { cudaFree(d_state_); d_state_ = nullptr; }
    if (d_decision_) { cudaFree(d_decision_); d_decision_ = nullptr; }
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    gpu_state_initialized_ = false;
}

void TelemetryControl::reset(cudaStream_t stream) {
    if (!d_state_) {
        throw std::runtime_error("[TelemetryControl::reset] d_state_ is NULL - call initGPU first");
    }
    StreamController::fatalIfDefaultStream(stream, "TelemetryControl::reset");
    
    // Reset GPU state to zeros
    cudaError_t err = cudaMemsetAsync(d_state_, 0, sizeof(TelemetryControlState_GPU), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::reset] cudaMemsetAsync failed: ") +
            cudaGetErrorString(err));
    }
    
    // Mark that config needs to be re-copied on next evaluate (in case config changed)
    gpu_state_initialized_ = false;
}

ControlDecision TelemetryControl::evaluate(
    const TelemetryLattice* lattice,
    float raw_grad_norm,
    float loss,
    int valid_tokens,
    float avg_seq_len,
    uint32_t global_step,
    cudaStream_t stream
) {
    if (!isInitialized()) {
        throw std::runtime_error("TelemetryControl::evaluate called before initGPU");
    }
    if (!lattice) {
        throw std::runtime_error("TelemetryControl::evaluate: lattice is NULL");
    }
    StreamController::fatalIfDefaultStream(stream, "TelemetryControl::evaluate");
    
    // Per-instance lazy init: copy config and zero state on first call
    // BUG FIX: Was using static bool which shared state across ALL instances!
    if (!gpu_state_initialized_) {
        cudaError_t err = cudaMemcpyAsync(d_config_, &config_, sizeof(TelemetryControlConfig),
                        cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemcpyAsync config failed: ") +
                cudaGetErrorString(err));
        }
        err = cudaMemsetAsync(d_state_, 0, sizeof(TelemetryControlState_GPU), stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemsetAsync state failed: ") +
                cudaGetErrorString(err));
        }
        gpu_state_initialized_ = true;
    }
    
    // Pack input
    ControlKernelInput input{};
    input.raw_grad_norm = raw_grad_norm;
    input.loss = loss;
    input.avg_seq_len = avg_seq_len;
    input.valid_tokens = valid_tokens;
    input.global_step = global_step;
    
    // Copy input to GPU (32 bytes H2D)
    cudaError_t err = cudaMemcpyAsync(d_input_, &input, sizeof(ControlKernelInput),
                    cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemcpyAsync input failed: ") +
            cudaGetErrorString(err));
    }
    
    // Extract device pointers from lattice (Pattern B: public accessors)
    const LatticeLevelState* d_lattice_levels = latticeLevelsDevicePtr(*lattice);
    const int num_lattice_streams = lattice->config().num_streams;
    
    // Launch kernel with device pointers (not host struct pointer!)
    launchControlDecisionKernel(d_lattice_levels, num_lattice_streams, d_config_, d_state_, d_input_, d_decision_, stream);
    
    // Check kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::evaluate] kernel launch failed: ") +
            cudaGetErrorString(err));
    }
    
    // Copy decision back (48 bytes D2H)
    ControlDecision result{};
    err = cudaMemcpyAsync(&result, d_decision_, sizeof(ControlDecision),
                    cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[TelemetryControl::evaluate] cudaMemcpyAsync decision failed: ") +
            cudaGetErrorString(err));
    }
    
    // Must sync to get result
    cudaStreamSynchronize(stream);
    
    return result;
}

std::string TelemetryControl::describeDecision(const ControlDecision& d) const {
    std::ostringstream oss;
    oss << "Action=" << getControlActionName(d.action);
    
    if (d.spike_severity != SpikeSeverity::None) {
        oss << " spike=" << getSpikeSeverityName(d.spike_severity) 
            << "(ratio=" << d.spike_ratio << ")";
    }
    
    if (d.accumulationBugDetected()) oss << " [ACCUM_BUG]";
    
    return oss.str();
}

} // namespace GRIM::Telemetry
