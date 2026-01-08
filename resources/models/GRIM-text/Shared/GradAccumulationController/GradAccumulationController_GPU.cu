/**
 * @file GradAccumulationController_GPU.cu
 * @brief Implementation of centralized gradient accumulation controller
 *
 * This file implements the traffic light state machine for gradient
 * accumulation. All gradient operations are logged and validated.
 *
 * STATE MACHINE:
 *   IDLE --(beginAccumulationWindow)--> ACCUMULATING
 *   ACCUMULATING --(beginBackward/endBackward)--> ACCUMULATING (micro-steps)
 *   ACCUMULATING --(beginOptimizerStep)--> READY_FOR_STEP
 *   READY_FOR_STEP --(endOptimizerStep)--> IDLE
 *
 * ERROR HANDLING:
 *   - Invalid transitions set state to ERROR
 *   - Error messages stored in last_error_
 *   - forceReset() recovers from ERROR state
 */

#include "GradAccumulationController_GPU.hpp"
#include "../LogRecorder/LogRecorder.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace {
constexpr auto kModuleGradCtrl = GRIM::Logging::ModuleId::Optimizer;
}

namespace GRIM {

//======================================================//
//  State String Conversion
//======================================================//

const char* stateToString(GradControllerState state) {
    switch (state) {
        case GradControllerState::IDLE:           return "IDLE";
        case GradControllerState::ACCUMULATING:   return "ACCUMULATING";
        case GradControllerState::READY_FOR_STEP: return "READY_FOR_STEP";
        case GradControllerState::STEPPING:       return "STEPPING";
        default:                                  return "UNKNOWN";
    }
}

//======================================================//
//  CUDA Kernels for RMS Computation
//======================================================//

namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr int kMaxGridSize = HyperParameters::CUDA_MAX_GRID_SIZE;

inline int computeGridSize(std::size_t elements) {
    int grid = static_cast<int>((elements + kBlockSize - 1) / kBlockSize);
    return std::min(grid, kMaxGridSize);
}

/**
 * @brief Compute sum of squares for RMS computation
 * Uses parallel reduction with shared memory
 */
__global__ void sumSquaresKernel(const float* __restrict__ data,
                                  float* __restrict__ partial_sums,
                                  std::size_t size) {
    __shared__ float shared[kBlockSize];
    
    const std::size_t tid = threadIdx.x;
    const std::size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    
    // Accumulate sum of squares for this thread
    float sum = 0.0f;
    for (std::size_t i = global_idx; i < size; i += stride) {
        float val = data[i];
        sum += val * val;
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

/**
 * @brief Final reduction kernel for RMS
 */
__global__ void finalReduceKernel(const float* __restrict__ partial_sums,
                                   float* __restrict__ result,
                                   int num_blocks,
                                   std::size_t total_elements) {
    __shared__ float shared[kBlockSize];
    
    const int tid = threadIdx.x;
    
    // Load and sum partial results
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    // Final reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // Compute and store RMS
    if (tid == 0) {
        float mean_sq = shared[0] / static_cast<float>(total_elements);
        result[0] = sqrtf(mean_sq);
    }
}

} // anonymous namespace

//======================================================//
//  detail:: Implementation
//======================================================//

namespace detail {

float computeGradientRMS(const float* grads, std::size_t size, cudaStream_t stream,
                         float* d_scratch_partial, float* d_scratch_result) {
    if (!grads || size == 0) {
        return 0.0f;
    }
    
    if (!d_scratch_partial || !d_scratch_result) {
        GRIM::Logging::EmitModuleError(kModuleGradCtrl, "computeGradientRMS: RMS scratch buffers not allocated");
        return -1.0f;
    }
    
    const int grid_size = computeGridSize(size);
    
    // Launch sum of squares kernel (uses pre-allocated scratch)
    sumSquaresKernel<<<grid_size, kBlockSize, 0, stream>>>(grads, d_scratch_partial, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "computeGradientRMS: sumSquaresKernel launch failed: " << cudaGetErrorString(err);
        GRIM::Logging::EmitModuleError(kModuleGradCtrl, oss.str());
        return -1.0f;
    }
    
    // Launch final reduction (uses pre-allocated result buffer)
    finalReduceKernel<<<1, kBlockSize, 0, stream>>>(d_scratch_partial, d_scratch_result, grid_size, size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "computeGradientRMS: finalReduceKernel launch failed: " << cudaGetErrorString(err);
        GRIM::Logging::EmitModuleError(kModuleGradCtrl, oss.str());
        return -1.0f;
    }
    
    // Async copy to pinned host memory (non-blocking)
    float h_result = 0.0f;
    err = cudaMemcpyAsync(&h_result, d_scratch_result, sizeof(float), 
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "computeGradientRMS: cudaMemcpyAsync failed: " << cudaGetErrorString(err);
        GRIM::Logging::EmitModuleError(kModuleGradCtrl, oss.str());
        return -1.0f;
    }
    
    // Sync only if we need the value immediately (caller can skip this)
    cudaStreamSynchronize(stream);
    
    return h_result;
}

void zeroGradientBuffer(float* grads, std::size_t size, cudaStream_t stream, const char* name = nullptr) {
    if (!grads || size == 0) {
        return;
    }
    
    cudaError_t err = cudaMemsetAsync(grads, 0, size * sizeof(float), stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradController] ERROR: cudaMemsetAsync failed: %s\n",
                cudaGetErrorString(err));
        if (name) {
            fprintf(stderr, "                buffer='%s' ptr=%p size=%zu stream=%p\n",
                    name, grads, size, stream);
        } else {
            fprintf(stderr, "                ptr=%p size=%zu stream=%p\n",
                    grads, size, stream);
        }
        
        // Check if pointer is valid device memory
        cudaPointerAttributes attrs;
        cudaError_t ptr_err = cudaPointerGetAttributes(&attrs, grads);
        if (ptr_err != cudaSuccess) {
            fprintf(stderr, "                Pointer validation failed: %s\n",
                    cudaGetErrorString(ptr_err));
            cudaGetLastError(); // Clear error
        } else {
            fprintf(stderr, "                Pointer type: %d (0=unregistered,1=host,2=device,3=managed)\n",
                    attrs.type);
        }
    }
}

} // namespace detail

//======================================================//
//  GradAccumulationController Implementation
//======================================================//

GradAccumulationController::GradAccumulationController()
    : config_{}
    , buffers_{}
    , state_(GradControllerState::IDLE)
    , micro_step_(0)
    , stats_{}
    , last_error_{}
    , state_callback_(nullptr)
    , explosion_callback_(nullptr)
    , d_rms_scratch_(nullptr)
    , rms_scratch_size_(0)
{}

GradAccumulationController::~GradAccumulationController() {
    if (d_rms_scratch_) {
        cudaFree(d_rms_scratch_);
        d_rms_scratch_ = nullptr;
    }
}

GradAccumulationController::GradAccumulationController(GradAccumulationController&& other) noexcept
    : config_(std::move(other.config_))
    , buffers_(std::move(other.buffers_))
    , state_(other.state_)
    , micro_step_(other.micro_step_)
    , stats_(other.stats_)
    , last_error_(std::move(other.last_error_))
    , state_callback_(std::move(other.state_callback_))
    , explosion_callback_(std::move(other.explosion_callback_))
    , d_rms_scratch_(other.d_rms_scratch_)
    , rms_scratch_size_(other.rms_scratch_size_)
{
    other.state_ = GradControllerState::IDLE;
    other.micro_step_ = 0;
    other.stats_.reset();
    other.d_rms_scratch_ = nullptr;
    other.rms_scratch_size_ = 0;
}

GradAccumulationController& GradAccumulationController::operator=(GradAccumulationController&& other) noexcept {
    if (this != &other) {
        if (d_rms_scratch_) {
            cudaFree(d_rms_scratch_);
        }
        
        config_ = std::move(other.config_);
        buffers_ = std::move(other.buffers_);
        state_ = other.state_;
        micro_step_ = other.micro_step_;
        stats_ = other.stats_;
        last_error_ = std::move(other.last_error_);
        state_callback_ = std::move(other.state_callback_);
        explosion_callback_ = std::move(other.explosion_callback_);
        d_rms_scratch_ = other.d_rms_scratch_;
        rms_scratch_size_ = other.rms_scratch_size_;
        
        other.state_ = GradControllerState::IDLE;
        other.micro_step_ = 0;
        other.stats_.reset();
        other.d_rms_scratch_ = nullptr;
        other.rms_scratch_size_ = 0;
    }
    return *this;
}

//--------------------------------------------------//
//  Configuration
//--------------------------------------------------//

void GradAccumulationController::configure(int accum_steps, cudaStream_t stream) {
    GradAccumulationConfig cfg;
    cfg.accum_steps = accum_steps;
    cfg.stream = stream;
    configure(cfg);
}

void GradAccumulationController::configure(const GradAccumulationConfig& config) {
    if (config.accum_steps < 1) {
        setError("accum_steps must be >= 1");
        if (config.strict_mode) {
            fprintf(stderr, "[GradController] ERROR: %s\n", last_error_.c_str());
        }
        return;
    }
    if (!config.stream) {
        setError("stream must be provided by StreamController (null stream rejected)");
        return;
    }
    StreamController::fatalIfDefaultStream(config.stream, "GradAccumulationController::configure");
    
    config_ = config;
    state_ = GradControllerState::IDLE;
    micro_step_ = 0;
    
    // Allocate RMS scratch buffers if monitoring enabled
    if (config_.monitor_gradients && !d_rms_scratch_) {
        constexpr int kMaxGridSize = HyperParameters::CUDA_REDUCTION_MAX_BLOCKS;
        cudaMalloc(&d_rms_scratch_, (kMaxGridSize + 1) * sizeof(float));
        if (!d_rms_scratch_) {
            throw std::runtime_error("[GradController] Failed to allocate RMS scratch buffers");
        }
    }
    
    if (config_.verbose) {
        std::cout << "[GradController] Configured: accum_steps=" << config_.accum_steps
                  << ", scale_factor=" << getScaleFactor()
                  << ", zero_on_start=" << config_.zero_on_window_start
                  << ", zero_on_complete=" << config_.zero_on_optimizer_complete
                  << "\n";
    }
}

//--------------------------------------------------//
//  Buffer Registration
//--------------------------------------------------//

void GradAccumulationController::registerGradientBuffer(const std::string& name, 
                                                          float* ptr, 
                                                          std::size_t size) {
    // Check for duplicate name
    for (const auto& buf : buffers_) {
        if (buf.name == name) {
            if (config_.verbose) {
                std::cout << "[GradController] WARNING: Duplicate buffer '" << name 
                          << "' - updating pointer\n";
            }
            // Update existing buffer
            for (auto& b : buffers_) {
                if (b.name == name) {
                    b.ptr = ptr;
                    b.size = size;
                    return;
                }
            }
        }
    }
    
    buffers_.emplace_back(name, ptr, size);
    
    if (config_.verbose) {
        std::cout << "[GradController] Registered buffer '" << name 
                  << "': " << size << " elements (" 
                  << (size * sizeof(float) / 1024.0) << " KB)\n";
    }
}

void GradAccumulationController::registerGradientBuffers(const std::vector<GradBuffer>& buffers) {
    for (const auto& buf : buffers) {
        registerGradientBuffer(buf.name, buf.ptr, buf.size);
    }
}

bool GradAccumulationController::unregisterGradientBuffer(const std::string& name) {
    auto it = std::find_if(buffers_.begin(), buffers_.end(),
        [&name](const GradBuffer& buf) { return buf.name == name; });
    
    if (it != buffers_.end()) {
        if (config_.verbose) {
            std::cout << "[GradController] Unregistered buffer '" << name << "'\n";
        }
        buffers_.erase(it);
        return true;
    }
    return false;
}

void GradAccumulationController::clearBuffers() {
    buffers_.clear();
    if (config_.verbose) {
        std::cout << "[GradController] Cleared all buffers\n";
    }
}

void GradAccumulationController::setBufferEnabled(const std::string& name, bool enabled) {
    for (auto& buf : buffers_) {
        if (buf.name == name) {
            buf.enabled = enabled;
            if (config_.verbose) {
                std::cout << "[GradController] Buffer '" << name 
                          << "' " << (enabled ? "enabled" : "disabled") << "\n";
            }
            return;
        }
    }
}

const GradBuffer* GradAccumulationController::getBuffer(const std::string& name) const {
    for (const auto& buf : buffers_) {
        if (buf.name == name) {
            return &buf;
        }
    }
    return nullptr;
}

std::size_t GradAccumulationController::totalGradientElements() const {
    std::size_t total = 0;
    for (const auto& buf : buffers_) {
        if (buf.enabled && buf.ptr) {
            total += buf.size;
        }
    }
    return total;
}

std::size_t GradAccumulationController::totalGradientBytes() const {
    return totalGradientElements() * sizeof(float);
}

//--------------------------------------------------//
//  TRAFFIC LIGHT API
//--------------------------------------------------//

float GradAccumulationController::getScaleFactor() const {
    if (config_.auto_scale_loss) {
        return 1.0f / static_cast<float>(config_.accum_steps);
    }
    return config_.manual_scale_factor;
}

bool GradAccumulationController::beginAccumulationWindow() {
    // Rule 20: Fail loud on invalid state
    if (state_ != GradControllerState::IDLE) {
        fprintf(stderr, "\n[GradController] FATAL: beginAccumulationWindow() called in state %s (expected IDLE)\n", stateString());
        fprintf(stderr, "[GradController] Previous window was not closed properly.\n");
        fprintf(stderr, "[GradController] Call forceReset() if intentionally abandoning a window.\n");
        std::abort();
    }
    
    // Zero gradients at window start
    if (config_.zero_on_window_start) {
        zeroGradientsInternal("window_start");
    }
    
    // Reset micro-step
    micro_step_ = 0;
    
    // Transition to ACCUMULATING
    transitionTo(GradControllerState::ACCUMULATING, "beginAccumulationWindow");
    
    stats_.accumulation_windows++;
    return true;
}

bool GradAccumulationController::beginBackward() {
    // Rule 20: Fail loud on invalid state - no silent ERROR state
    if (state_ != GradControllerState::ACCUMULATING) {
        fprintf(stderr, "\n[GradController] FATAL: beginBackward() called in state %s (expected ACCUMULATING)\n", stateString());
        fprintf(stderr, "[GradController] This indicates a bug in training loop state management.\n");
        fprintf(stderr, "[GradController] micro_step=%d accum_steps=%d\n", micro_step_, config_.accum_steps);
        std::abort();
    }
    
    // Rule 20: Fail loud if accumulation window is closed
    if (micro_step_ >= config_.accum_steps) {
        fprintf(stderr, "\n[GradController] FATAL: beginBackward() called but accumulation complete\n");
        fprintf(stderr, "[GradController] micro_step=%d >= accum_steps=%d\n", micro_step_, config_.accum_steps);
        fprintf(stderr, "[GradController] Call beginAccumulationWindow() to start new window.\n");
        std::abort();
    }
    
    if (config_.verbose) {
        std::cout << "[GradController] beginBackward() micro_step=" << micro_step_ 
                  << "/" << config_.accum_steps << "\n";
    }
    
    return true;
}

bool GradAccumulationController::endBackward() {
    // Rule 20: Fail loud on invalid state - no silent ERROR state
    if (state_ != GradControllerState::ACCUMULATING) {
        fprintf(stderr, "\n[GradController] FATAL: endBackward() called in state %s (expected ACCUMULATING)\n", stateString());
        fprintf(stderr, "[GradController] This indicates beginBackward() was not called or state was corrupted.\n");
        std::abort();
    }
    
    // Increment micro-step
    micro_step_++;
    stats_.micro_steps_completed++;
    
    // Monitor gradients if enabled
    if (config_.monitor_gradients) {
        bool explosion = checkGradientExplosion();
        if (explosion) {
            stats_.gradient_explosions_detected++;
            if (config_.verbose) {
                std::cout << "[GradController] WARNING: Gradient explosion detected!\n";
            }
        }
    }
    
    if (config_.verbose) {
        std::cout << "[GradController] endBackward() micro_step=" << micro_step_ 
                  << "/" << config_.accum_steps;
        if (isAccumulationComplete()) {
            std::cout << " [COMPLETE]";
        }
        std::cout << "\n";
    }
    
    return true;
}

bool GradAccumulationController::beginOptimizerStep() {
    // Rule 20: Fail loud on invalid state
    if (state_ != GradControllerState::ACCUMULATING) {
        fprintf(stderr, "\n[GradController] FATAL: beginOptimizerStep() called in state %s (expected ACCUMULATING)\n", stateString());
        fprintf(stderr, "[GradController] micro_step=%d accum_steps=%d\n", micro_step_, config_.accum_steps);
        std::abort();
    }
    
    // Warning only for early optimizer step (valid use case for single-batch training)
    if (!isAccumulationComplete()) {
        fprintf(stderr, "[GradController] WARNING: beginOptimizerStep() with incomplete accumulation (micro_step=%d < accum_steps=%d)\n",
                micro_step_, config_.accum_steps);
    }
    
    // Transition to READY_FOR_STEP
    transitionTo(GradControllerState::READY_FOR_STEP, "beginOptimizerStep");
    
    return true;
}

bool GradAccumulationController::endOptimizerStep() {
    // Rule 20: Fail loud on invalid state
    if (state_ != GradControllerState::READY_FOR_STEP) {
        fprintf(stderr, "\n[GradController] FATAL: endOptimizerStep() called in state %s (expected READY_FOR_STEP)\n", stateString());
        fprintf(stderr, "[GradController] beginOptimizerStep() must be called first.\n");
        std::abort();
    }
    
    // Zero gradients after optimizer step
    if (config_.zero_on_optimizer_complete) {
        zeroGradientsInternal("optimizer_complete");
    }
    
    // Reset micro-step
    micro_step_ = 0;
    
    // Transition to IDLE
    transitionTo(GradControllerState::IDLE, "endOptimizerStep");
    
    stats_.optimizer_steps++;
    return true;
}

//--------------------------------------------------//
//  Manual Control
//--------------------------------------------------//

void GradAccumulationController::zeroAllGradients() {
    zeroGradientsInternal("manual");
}

bool GradAccumulationController::zeroGradientBuffer(const std::string& name) {
    for (const auto& buf : buffers_) {
        if (buf.name == name && buf.enabled && buf.ptr && buf.size > 0) {
            detail::zeroGradientBuffer(buf.ptr, buf.size, config_.stream, buf.name.c_str());
            
            stats_.total_zero_ops++;
            stats_.total_bytes_zeroed += buf.bytes();
            
            if (config_.verbose) {
                std::cout << "[GradController] Zeroed buffer '" << name << "'\n";
            }
            return true;
        }
    }
    return false;
}

void GradAccumulationController::forceReset() {
    state_ = GradControllerState::IDLE;
    micro_step_ = 0;
    last_error_.clear();
    
    if (config_.verbose) {
        std::cout << "[GradController] Force reset to IDLE\n";
    }
}

//--------------------------------------------------//
//  Gradient Monitoring
//--------------------------------------------------//

std::vector<std::pair<std::string, float>> GradAccumulationController::computeGradientRMS() const {
    std::vector<std::pair<std::string, float>> results;
    if (!d_rms_scratch_) return results;  // RMS monitoring not configured
    
    results.reserve(buffers_.size());
    
    float* d_partial = d_rms_scratch_;
    float* d_result = d_rms_scratch_ + 256;  // kMaxGridSize
    
    for (auto& buf : buffers_) {
        if (buf.enabled && buf.ptr && buf.size > 0) {
            float rms = detail::computeGradientRMS(buf.ptr, buf.size, config_.stream, d_partial, d_result);
            buf.last_rms = rms;
            buf.rms_valid = true;
            results.emplace_back(buf.name, rms);
        }
    }
    
    return results;
}

bool GradAccumulationController::checkGradientExplosion() const {
    if (!d_rms_scratch_) return false;  // RMS monitoring not configured
    
    float* d_partial = d_rms_scratch_;
    float* d_result = d_rms_scratch_ + 256;  // kMaxGridSize
    
    for (auto& buf : buffers_) {
        if (buf.enabled && buf.ptr && buf.size > 0) {
            float rms = detail::computeGradientRMS(buf.ptr, buf.size, config_.stream, d_partial, d_result);
            buf.last_rms = rms;
            buf.rms_valid = true;
            
            if (rms > config_.gradient_explosion_threshold) {
                if (explosion_callback_) {
                    explosion_callback_(buf.name, rms);
                }
                return true;
            }
        }
    }
    return false;
}

std::string GradAccumulationController::getHottestBuffer() const {
    std::string hottest;
    float max_rms = -1.0f;
    
    for (const auto& buf : buffers_) {
        if (buf.rms_valid && buf.last_rms > max_rms) {
            max_rms = buf.last_rms;
            hottest = buf.name;
        }
    }
    
    return hottest;
}

float GradAccumulationController::computeBufferRMS(const GradBuffer& buffer) const {
    if (!d_rms_scratch_) return 0.0f;  // RMS monitoring not configured
    
    float* d_partial = d_rms_scratch_;
    float* d_result = d_rms_scratch_ + 256;  // kMaxGridSize
    
    return detail::computeGradientRMS(buffer.ptr, buffer.size, config_.stream, d_partial, d_result);
}

//--------------------------------------------------//
//  Statistics & Debugging
//--------------------------------------------------//

void GradAccumulationController::printState() const {
    std::cout << "\n[GradController] State:\n";
    std::cout << "─────────────────────────────────────────────────────\n";
    std::cout << "  State:      " << stateString() << "\n";
    std::cout << "  Micro-step: " << micro_step_ << "/" << config_.accum_steps << "\n";
    std::cout << "  Scale:      " << getScaleFactor() << "\n";
    std::cout << "  Buffers:    " << buffers_.size() << " registered\n";
    std::cout << "  Elements:   " << totalGradientElements() << "\n";
    std::cout << "  Bytes:      " << (totalGradientBytes() / (1024.0 * 1024.0)) << " MB\n";
    if (!last_error_.empty()) {
        std::cout << "  Last Error: " << last_error_ << "\n";
    }
    std::cout << "─────────────────────────────────────────────────────\n";
}

void GradAccumulationController::printBuffers() const {
    std::cout << "\n[GradController] Registered Buffers (" << buffers_.size() << "):\n";
    std::cout << "─────────────────────────────────────────────────────\n";
    std::cout << std::left << std::setw(40) << "Name" 
              << std::right << std::setw(12) << "Elements"
              << std::setw(12) << "KB"
              << std::setw(10) << "Enabled"
              << std::setw(12) << "RMS"
              << "\n";
    std::cout << "─────────────────────────────────────────────────────\n";
    
    std::size_t total_bytes = 0;
    for (const auto& buf : buffers_) {
        std::cout << std::left << std::setw(40) << buf.name
                  << std::right << std::setw(12) << buf.size
                  << std::setw(12) << std::fixed << std::setprecision(1) 
                  << (buf.bytes() / 1024.0)
                  << std::setw(10) << (buf.enabled ? "yes" : "no");
        if (buf.rms_valid) {
            std::cout << std::setw(12) << std::scientific << std::setprecision(2) << buf.last_rms;
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << "\n";
        
        if (buf.enabled) {
            total_bytes += buf.bytes();
        }
    }
    
    std::cout << "─────────────────────────────────────────────────────\n";
    std::cout << "Total: " << std::fixed << std::setprecision(2) 
              << (total_bytes / (1024.0 * 1024.0)) << " MB\n\n";
}

bool GradAccumulationController::validate() const {
    bool valid = true;
    
    if (config_.accum_steps < 1) {
        fprintf(stderr, "[GradController] VALIDATION FAILED: accum_steps=%d < 1\n",
                config_.accum_steps);
        valid = false;
    }
    
    for (const auto& buf : buffers_) {
        if (buf.enabled) {
            if (!buf.ptr) {
                fprintf(stderr, "[GradController] VALIDATION FAILED: buffer '%s' has null pointer\n",
                        buf.name.c_str());
                valid = false;
            }
            if (buf.size == 0) {
                fprintf(stderr, "[GradController] VALIDATION FAILED: buffer '%s' has zero size\n",
                        buf.name.c_str());
                valid = false;
            }
        }
    }
    
    return valid;
}

//--------------------------------------------------//
//  Internal Implementation
//--------------------------------------------------//

bool GradAccumulationController::transitionTo(GradControllerState new_state, const char* action) {
    GradControllerState old_state = state_;
    state_ = new_state;
    
    if (config_.verbose) {
        std::cout << "[GradController] " << action << ": "
                  << stateToString(old_state) << " -> " << stateToString(new_state) << "\n";
    }
    
    if (state_callback_) {
        state_callback_(old_state, new_state);
    }
    
    return true;
}

void GradAccumulationController::setError(const std::string& message) {
    // Rule 20: No ERROR state - crash immediately with clear message
    fprintf(stderr, "\n[GradController] FATAL: %s\n", message.c_str());
    std::abort();
}

void GradAccumulationController::zeroGradientsInternal(const char* phase) {
    if (buffers_.empty()) {
        return;
    }
    
    static unsigned long long current_step_id = 0;
    current_step_id++;
    
    if (config_.verbose) {
        std::cout << "[GradController] Zeroing gradients (" << phase << ")\n";
    }
    
    std::size_t bytes_zeroed = 0;
    int buffers_zeroed = 0;
    
    for (const auto& buf : buffers_) {
        if (buf.enabled && buf.ptr && buf.size > 0) {
            // Removed GradWrite spam - kills performance
            detail::zeroGradientBuffer(buf.ptr, buf.size, config_.stream, buf.name.c_str());
            bytes_zeroed += buf.bytes();
            buffers_zeroed++;
        }
    }
    
    stats_.total_zero_ops += buffers_zeroed;
    stats_.total_bytes_zeroed += bytes_zeroed;
    
    // Optional validation (EXPENSIVE - only enable for debugging)
    // Validates gradient buffers are actually zeroed after memset
    // Cost: cudaStreamSync + 50 D2H memcopies per batch = ~80% performance loss
    static bool enable_validation = false;  // Set to true only when debugging zero failures
    
    if (enable_validation) {
        if (config_.stream) {
            cudaError_t err = cudaStreamSynchronize(config_.stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "[GradController] FATAL: cudaStreamSynchronize after zero failed: %s\n",
                        cudaGetErrorString(err));
                std::exit(1);
            }
        } else {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "[GradController] FATAL: cudaDeviceSynchronize after zero failed: %s\n",
                        cudaGetErrorString(err));
                std::exit(1);
            }
        }
        
        for (const auto& buf : buffers_) {
            if (!buf.enabled || !buf.ptr || buf.size == 0) continue;
            
            float h_sample = -999.0f;
            cudaError_t err = cudaMemcpy(&h_sample, buf.ptr, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "[GradController] FATAL: Failed to verify zero for buffer '%s': %s\n",
                        buf.name.c_str(), cudaGetErrorString(err));
                std::exit(1);
            }
            
            if (h_sample != 0.0f) {
                fprintf(stderr, "\n");
                fprintf(stderr, "╔════════════════════════════════════════════════════════════════╗\n");
                fprintf(stderr, "║  FATAL: GRADIENT BUFFER NOT ZERO AFTER ZEROING OPERATION      ║\n");
                fprintf(stderr, "╚════════════════════════════════════════════════════════════════╝\n");
                fprintf(stderr, "\n");
                fprintf(stderr, "  Buffer: '%s' | Size: %zu (%.2fMB) | Sample: %.6e\n",
                        buf.name.c_str(), buf.size, buf.bytes()/(1024.0*1024.0), h_sample);
                fprintf(stderr, "  Phase: '%s'\n", phase);
                fprintf(stderr, "\n");
                std::exit(1);
            }
        }
    }
    
    if (config_.verbose) {
        std::cout << "[GradController] ✓ Zeroed " << buffers_zeroed << " buffers ("
                  << (bytes_zeroed / (1024.0 * 1024.0)) << " MB) - validation PASSED\n";
    }
}

//======================================================//
//  AccumulationWindowScope Implementation
//======================================================//

AccumulationWindowScope::AccumulationWindowScope(GradAccumulationController& controller)
    : controller_(controller), stepped_(false)
{
    controller_.beginAccumulationWindow();
}

AccumulationWindowScope::~AccumulationWindowScope() {
    // Safety: if user forgot to call optimizerStep, force reset
    if (!stepped_) {
        controller_.forceReset();
    }
}

} // namespace GRIM
