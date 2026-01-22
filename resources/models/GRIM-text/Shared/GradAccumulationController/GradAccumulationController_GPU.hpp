/**
 * @file GradAccumulationController_GPU.hpp
 * @brief Centralized gradient accumulation traffic controller
 *
 * DESIGN PHILOSOPHY:
 * This is the SINGLE SOURCE OF TRUTH for gradient state. All gradient
 * operations flow through this controller. No more chasing red herrings.
 *
 * TRAFFIC LIGHT STATES:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  IDLE ──────► ACCUMULATING ──────► READY_FOR_STEP ──────► IDLE │
 * │    │              │                      │                      │
 * │    │         (backward x N)         (optimizer.step)            │
 * │    │              │                      │                      │
 * │    └──────────────┴──────────────────────┘                      │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * USAGE PATTERN:
 * ─────────────────────────────────────────────────────────────────
 *   GradAccumulationController controller;
 *   controller.configure(accum_steps, stream);
 *   controller.registerGradientBuffer("embedding", ptr, size);
 *   // ... register all buffers
 *
 *   for (batch : batches) {
 *       controller.beginAccumulationWindow();  // IDLE -> ACCUMULATING, zeros grads
 *
 *       for (int micro = 0; micro < accum_steps; ++micro) {
 *           float scale = controller.getScaleFactor();  // 1.0 / accum_steps
 *           
 *           controller.beginBackward();    // Validates state
 *           float scaled_loss = loss * scale;
 *           backward(scaled_loss);
 *           controller.endBackward();      // Increments micro-step
 *       }
 *
 *       controller.beginOptimizerStep();   // ACCUMULATING -> READY_FOR_STEP
 *       clip_grad_norm();
 *       optimizer.step();
 *       controller.endOptimizerStep();     // READY_FOR_STEP -> IDLE, zeros grads
 *   }
 * ─────────────────────────────────────────────────────────────────
 *
 * DEBUG FEATURES:
 * - State transition logging (verbose mode)
 * - Invalid state detection with clear error messages
 * - Gradient buffer validation
 * - Statistics tracking (zero ops, bytes, windows)
 * - Per-buffer RMS monitoring for gradient explosion detection
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

namespace GRIM {

//======================================================//
//  Traffic Light State Machine
//======================================================//

/**
 * @brief Controller state - the "traffic light"
 * 
 * State transitions:
 *   IDLE -> ACCUMULATING: beginAccumulationWindow()
 *   ACCUMULATING -> ACCUMULATING: beginBackward(), endBackward() (micro-steps)
 *   ACCUMULATING -> READY_FOR_STEP: beginOptimizerStep() (when micro_step == accum_steps)
 *   READY_FOR_STEP -> IDLE: endOptimizerStep()
 *   
 * Invalid transitions trigger errors with descriptive messages.
 */
enum class GradControllerState : uint8_t
{
    IDLE = 0 ,              ///< Not in accumulation window, ready to start
    ACCUMULATING = 1 ,      ///< In backward pass accumulation (micro-steps in progress)
    READY_FOR_STEP = 2 ,    ///< Accumulation complete, optimizer step allowed
    STEPPING = 3            ///< Optimizer step in progress (transient)
    // Rule 20: Removed ERR state - invalid transitions crash immediately with clear message
};

/**
 * @brief Convert state to string for logging
 */
const char* stateToString(GradControllerState state);

//======================================================//
//  Gradient Buffer Descriptor
//======================================================//

/**
 * @brief Single gradient buffer registration
 * @note Pointers are NOT owned - caller manages lifetime
 */
struct GradBuffer {
    std::string name;           ///< Descriptive name for debugging
    float* ptr;                 ///< Device pointer to gradient buffer
    std::size_t size;           ///< Number of float elements
    bool enabled;               ///< Include in teardown operations
    
    // Monitoring (computed on demand)
    mutable float last_rms;     ///< Last computed RMS (for explosion detection)
    mutable bool rms_valid;     ///< Whether last_rms is valid
    
    GradBuffer()
        : name("unnamed"), ptr(nullptr), size(0), enabled(true)
        , last_rms(0.0f), rms_valid(false) {}
    
    GradBuffer(const std::string& n, float* p, std::size_t s)
        : name(n), ptr(p), size(s), enabled(true)
        , last_rms(0.0f), rms_valid(false) {}
    
    std::size_t bytes() const { return size * sizeof(float); }
};

//======================================================//
//  Configuration
//======================================================//

struct GradAccumulationConfig {
    int accum_steps = 1;                    ///< Gradient accumulation steps
    cudaStream_t stream = nullptr;          ///< CUDA stream for async ops
    
    // Zeroing policy
    bool zero_on_window_start = false;      ///< Zero grads at accumulation window start (DISABLED - redundant after optimizer_complete)
    bool zero_on_optimizer_complete = true; ///< Zero grads after optimizer step
    
    // Scaling
    bool auto_scale_loss = false;            ///< Automatically compute 1/accum_steps
    float manual_scale_factor = 1.0f;       ///< Manual scale (if auto_scale_loss=false)
    
    // Debugging
    bool verbose = false;                   ///< Log state transitions
    bool strict_mode = true;                ///< Throw on invalid state transitions
    bool sync_after_zero = true;           ///< Sync stream after zeroing (debug)
    bool monitor_gradients = true;         ///< Compute RMS after each backward (expensive)
    float gradient_explosion_threshold = 1e6f; ///< RMS threshold for explosion warning
};

//======================================================//
//  Statistics
//======================================================//

struct GradAccumulationStats {
    // Counters
    std::size_t total_zero_ops = 0;
    std::size_t total_bytes_zeroed = 0;
    std::size_t accumulation_windows = 0;
    std::size_t micro_steps_completed = 0;
    std::size_t optimizer_steps = 0;
    
    // Error tracking
    std::size_t invalid_transitions = 0;
    std::size_t gradient_explosions_detected = 0;
    
    // Timing (optional)
    double total_zero_time_ms = 0.0;
    
    void reset() {
        total_zero_ops = 0;
        total_bytes_zeroed = 0;
        accumulation_windows = 0;
        micro_steps_completed = 0;
        optimizer_steps = 0;
        invalid_transitions = 0;
        gradient_explosions_detected = 0;
        total_zero_time_ms = 0.0;
    }
};

//======================================================//
//  Main Controller Class
//======================================================//

class GradAccumulationController {
public:
    //--------------------------------------------------//
    //  Lifecycle
    //--------------------------------------------------//
    GradAccumulationController();
    ~GradAccumulationController();
    
    // Non-copyable, movable
    GradAccumulationController(const GradAccumulationController&) = delete;
    GradAccumulationController& operator=(const GradAccumulationController&) = delete;
    GradAccumulationController(GradAccumulationController&&) noexcept;
    GradAccumulationController& operator=(GradAccumulationController&&) noexcept;
    
    //--------------------------------------------------//
    //  Configuration
    //--------------------------------------------------//
    
    /// Configure with accumulation steps and optional stream
    void configure(int accum_steps, cudaStream_t stream = nullptr);
    
    /// Configure with full config struct
    void configure(const GradAccumulationConfig& config);
    
    /// Get current configuration
    const GradAccumulationConfig& config() const { return config_; }
    
    /// Update stream after construction
    void setStream(cudaStream_t stream) { config_.stream = stream; }
    
    //--------------------------------------------------//
    //  Buffer Registration
    //--------------------------------------------------//
    
    /// Register a gradient buffer for management
    /// @param name Descriptive name (used in logs and debugging)
    /// @param ptr Device pointer to gradient buffer (NOT owned)
    /// @param size Number of float elements
    void registerGradientBuffer(const std::string& name, float* ptr, std::size_t size);
    
    /// Register multiple buffers
    void registerGradientBuffers(const std::vector<GradBuffer>& buffers);
    
    /// Unregister a buffer by name
    /// @return true if buffer was found and removed
    bool unregisterGradientBuffer(const std::string& name);
    
    /// Clear all registered buffers
    void clearBuffers();
    
    /// Enable/disable a specific buffer
    void setBufferEnabled(const std::string& name, bool enabled);
    
    /// Get buffer by name (nullptr if not found)
    const GradBuffer* getBuffer(const std::string& name) const;
    
    /// Get all registered buffers
    const std::vector<GradBuffer>& buffers() const { return buffers_; }
    
    /// Get buffer count
    std::size_t bufferCount() const { return buffers_.size(); }
    
    /// Get total gradient elements
    std::size_t totalGradientElements() const;
    
    /// Get total gradient bytes
    std::size_t totalGradientBytes() const;
    
    //--------------------------------------------------//
    //  TRAFFIC LIGHT API - Main Workflow
    //--------------------------------------------------//
    
    /**
     * @brief Get the loss scale factor
     * @return 1.0 / accum_steps (or manual_scale_factor if auto disabled)
     * 
     * Usage: scaled_loss = loss * controller.getScaleFactor();
     */
    float getScaleFactor() const;
    
    /**
     * @brief Start a new accumulation window
     * 
     * State transition: IDLE -> ACCUMULATING
     * Actions:
     *   - Validates current state is IDLE
     *   - Zeros all gradients (if zero_on_window_start=true)
     *   - Resets micro-step counter
     *   - Logs transition (if verbose=true)
     * 
     * @return true if transition successful, false on error
     */
    bool beginAccumulationWindow();
    
    /**
     * @brief Called BEFORE loss.backward()
     * 
     * State: Must be ACCUMULATING
     * Actions:
     *   - Validates state is ACCUMULATING
     *   - Validates micro_step < accum_steps
     *   - Logs (if verbose=true)
     * 
     * @return true if backward allowed, false on error
     */
    bool beginBackward();
    
    /**
     * @brief Called AFTER loss.backward()
     * 
     * State: ACCUMULATING
     * Actions:
     *   - Increments micro-step counter
     *   - Monitors gradient RMS (if monitor_gradients=true)
     *   - Logs (if verbose=true)
     * 
     * @return true if successful
     */
    bool endBackward();
    
    /**
     * @brief Called BEFORE optimizer.step()
     * 
     * State transition: ACCUMULATING -> READY_FOR_STEP
     * Actions:
     *   - Validates micro_step == accum_steps (accumulation complete)
     *   - Transitions to READY_FOR_STEP
     *   - Logs (if verbose=true)
     * 
     * @return true if optimizer step allowed, false if accumulation incomplete
     */
    bool beginOptimizerStep();
    
    /**
     * @brief Called AFTER optimizer.step()
     * 
     * State transition: READY_FOR_STEP -> IDLE
     * Actions:
     *   - Zeros all gradients (if zero_on_optimizer_complete=true)
     *   - Resets micro-step counter
     *   - Transitions to IDLE
     *   - Updates statistics
     *   - Logs (if verbose=true)
     * 
     * @return true if successful
     */
    bool endOptimizerStep();
    
    //--------------------------------------------------//
    //  State Queries
    //--------------------------------------------------//
    
    /// Get current state
    GradControllerState state() const { return state_; }
    
    /// Get state as string
    const char* stateString() const { return stateToString(state_); }
    
    /// Get current micro-step (0 to accum_steps-1)
    int currentMicroStep() const { return micro_step_; }
    
    /// Get total accumulation steps
    int accumSteps() const { return config_.accum_steps; }
    
    /// Check if accumulation is complete (micro_step == accum_steps)
    bool isAccumulationComplete() const { 
        return micro_step_ >= config_.accum_steps; 
    }
    
    /// Check if in valid accumulating state
    bool isAccumulating() const { 
        return state_ == GradControllerState::ACCUMULATING; 
    }
    
    /// Check if ready for optimizer step
    bool isReadyForStep() const { 
        return state_ == GradControllerState::READY_FOR_STEP ||
               (state_ == GradControllerState::ACCUMULATING && isAccumulationComplete());
    }
    
    //--------------------------------------------------//
    //  Manual Control
    //--------------------------------------------------//
    
    /// Manually zero all registered gradients
    void zeroAllGradients();
    
    /// Zero a specific buffer by name
    /// @return true if buffer found and zeroed
    bool zeroGradientBuffer(const std::string& name);
    
    /// Force reset to IDLE state (emergency use)
    void forceReset();
    
    //--------------------------------------------------//
    //  Gradient Monitoring (Debug)
    //--------------------------------------------------//
    
    /// Compute RMS of all gradient buffers
    /// @return Vector of (buffer_name, rms_value) pairs
    std::vector<std::pair<std::string, float>> computeGradientRMS() const;
    
    /// Check for gradient explosion (any buffer RMS > threshold)
    /// @return true if explosion detected
    bool checkGradientExplosion() const;
    
    /// Get the name of the buffer with highest RMS
    std::string getHottestBuffer() const;
    
    //--------------------------------------------------//
    //  Callbacks
    //--------------------------------------------------//
    
    using StateCallback = std::function<void(GradControllerState from, GradControllerState to)>;
    using GradientCallback = std::function<void(const std::string& buffer_name, float rms)>;
    using ZeroCallback = std::function<void(cudaStream_t stream)>;
    
    /// Called on state transitions
    void setStateTransitionCallback(StateCallback cb) { state_callback_ = std::move(cb); }
    
    /// Called when gradient explosion detected
    void setExplosionCallback(GradientCallback cb) { explosion_callback_ = std::move(cb); }
    
    /// Called during gradient zeroing phase to zero additional Tensor-managed buffers
    /// This callback is used by the Tensor migration (Issue #45) to zero intermediate
    /// gradient buffers that are now managed by Tensor::zero_grad() instead of raw registration.
    void setAdditionalZeroCallback(ZeroCallback cb) { additional_zero_callback_ = std::move(cb); }
    
    //--------------------------------------------------//
    //  Statistics & Debugging
    //--------------------------------------------------//
    
    /// Get runtime statistics
    const GradAccumulationStats& stats() const { return stats_; }
    
    /// Reset statistics
    void resetStats() { stats_.reset(); }
    
    /// Print current state and buffers
    void printState() const;
    
    /// Print all registered buffers with sizes
    void printBuffers() const;
    
    /// Validate configuration and buffers
    /// @return true if valid
    bool validate() const;
    
    /// Get last error message
    const std::string& lastError() const { return last_error_; }

private:
    //--------------------------------------------------//
    //  Internal Implementation
    //--------------------------------------------------//
    
    /// Transition to new state with logging
    bool transitionTo(GradControllerState new_state, const char* action);
    
    /// Set error state with message
    void setError(const std::string& message);
    
    /// Zero all enabled gradient buffers
    void zeroGradientsInternal(const char* phase);
    
    /// Launch gradient RMS computation kernel
    float computeBufferRMS(const GradBuffer& buffer) const;
    
    //--------------------------------------------------//
    //  State
    //--------------------------------------------------//
    
    GradAccumulationConfig config_;
    std::vector<GradBuffer> buffers_;
    GradControllerState state_ = GradControllerState::IDLE;
    int micro_step_ = 0;
    GradAccumulationStats stats_;
    std::string last_error_;
    
    // Callbacks
    StateCallback state_callback_;
    GradientCallback explosion_callback_;
    ZeroCallback additional_zero_callback_;  // Issue #45: For Tensor-managed intermediate grads
    
    // Device memory for RMS computation (allocated lazily)
    mutable float* d_rms_scratch_ = nullptr;
    mutable std::size_t rms_scratch_size_ = 0;
};

//======================================================//
//  RAII Accumulation Scope (Optional Helper)
//======================================================//

/**
 * @brief RAII helper for accumulation window
 * 
 * Usage:
 *   {
 *       AccumulationWindowScope scope(controller);
 *       for (int i = 0; i < accum_steps; ++i) {
 *           scope.backward([&]() {
 *               loss.backward();
 *           });
 *       }
 *       scope.optimizerStep([&]() {
 *           optimizer.step();
 *       });
 *   } // Automatic cleanup
 */
class AccumulationWindowScope {
public:
    explicit AccumulationWindowScope(GradAccumulationController& controller);
    ~AccumulationWindowScope();
    
    /// Execute backward pass with proper begin/end calls
    template<typename BackwardFn>
    void backward(BackwardFn&& fn) {
        controller_.beginBackward();
        fn();
        controller_.endBackward();
    }
    
    /// Execute optimizer step with proper begin/end calls
    template<typename StepFn>
    void optimizerStep(StepFn&& fn) {
        controller_.beginOptimizerStep();
        fn();
        controller_.endOptimizerStep();
        stepped_ = true;
    }
    
    /// Get scale factor for loss
    float scaleFactor() const { return controller_.getScaleFactor(); }
    
private:
    GradAccumulationController& controller_;
    bool stepped_ = false;
};

//======================================================//
//  CUDA Kernels (Internal)
//======================================================//

namespace detail {

/// Compute RMS of a gradient buffer
/// @return RMS value (computed synchronously for debugging)
float computeGradientRMS(const float* grads, std::size_t size, cudaStream_t stream);

/// Zero gradient buffer (wraps cudaMemsetAsync)
void zeroGradientBuffer(float* grads, std::size_t size, cudaStream_t stream);

} // namespace detail

} // namespace GRIM
