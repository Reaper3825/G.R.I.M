/**
 * @file StreamController_GPU.cu
 * @brief Implementation of centralized CUDA stream management
 * 
 * AUDIT (Feb 2026): Deleted all dead code per YAGNI:
 *   - StreamType enum (Transfer, Auxiliary) - zero production callers
 *   - StreamEvent class - zero production callers (16 CUDA events wasted on init)
 *   - StreamDescriptor struct - replaced with simple members
 *   - getTransferStream/getAuxiliaryStream - zero callers, Rule 20 fallback violations
 *   - getStream(StreamType) - zero callers, default case returned Primary silently
 *   - setExternalPrimaryStream/setExternalStream - zero callers
 *   - recordEvent/waitEvent - zero callers
 *   - logStatus/resetStats/checkCudaError - zero callers
 *   - syncAllStreams/syncStream(StreamType) - only called internally, inlined to syncPrimaryStream
 *   Rule 20 fixes:
 *   - initialize() throws on double-init instead of silently returning true
 *   - getPrimaryStream() throws if not initialized instead of returning nullptr
 */

#include "StreamController_GPU.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace GRIM {

//======================================================//
//  StreamController Implementation
//======================================================//

StreamController::StreamController() = default;

StreamController::~StreamController() {
    if (initialized_.load(std::memory_order_acquire)) {
        shutdown();
    }
}

bool StreamController::initialize(const StreamControllerConfig& config) {
    // Rule 20: Double-init is a lifecycle bug - fail loud
    if (initialized_.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            "[StreamController] FATAL: initialize() called when already initialized. "
            "This is a lifecycle bug - call shutdown() first or fix the double-init.");
    }
    
    config_ = config;
    
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "Initializing StreamController...");
    }
    
    // Create primary stream
    unsigned int flags = config_.primary_flags;
    int priority = config_.primary_priority;
    
    cudaError_t err;
    if (priority >= 0) {
        err = cudaStreamCreateWithPriority(&primary_stream_, flags, priority);
    } else {
        err = cudaStreamCreateWithFlags(&primary_stream_, flags);
    }
    
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to create primary stream: " << cudaGetErrorString(err);
        Logging::EmitModuleError(kModuleName, oss.str());
        return false;
    }
    
    fatalIfDefaultStream(primary_stream_, "StreamController::initialize primary");
    
    stream_owned_ = true;
    stats_.streams_created++;
    
    initialized_.store(true, std::memory_order_release);
    
    if (config_.verbose) {
        std::ostringstream oss;
        oss << "StreamController initialized: primary=yes";
        if (priority >= 0) {
            oss << " (priority=" << priority << ")";
        }
        Logging::EmitModuleInfo(kModuleName, oss.str());
    }
    
    return true;
}

void StreamController::shutdown() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }
    
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "Shutting down StreamController...");
    }
    
    // Sync before destroying
    if (primary_stream_) {
        cudaStreamSynchronize(primary_stream_);
    }
    
    // Destroy stream if we own it
    if (stream_owned_ && primary_stream_) {
        cudaStreamDestroy(primary_stream_);
        stats_.streams_destroyed++;
        
        if (config_.verbose) {
            Logging::EmitModuleInfo(kModuleName, "Destroyed primary stream");
        }
    }
    
    primary_stream_ = nullptr;
    stream_owned_ = true;
    initialized_.store(false, std::memory_order_release);
    
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "StreamController shutdown complete");
    }
}

cudaStream_t StreamController::getPrimaryStream() const {
    // Rule 20: No silent nullptr - caller MUST have initialized first
    if (!initialized_.load(std::memory_order_acquire) || !primary_stream_) {
        throw std::runtime_error(
            "[StreamController] FATAL: getPrimaryStream() called but StreamController is NOT initialized. "
            "Call initialize() first.");
    }
    return primary_stream_;
}

bool StreamController::syncPrimaryStream() {
    if (!initialized_.load(std::memory_order_acquire) || !primary_stream_) {
        return true;  // Nothing to sync
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = cudaStreamSynchronize(primary_stream_);
    
    auto end = std::chrono::high_resolution_clock::now();
    double sync_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to sync primary stream: " << cudaGetErrorString(err);
        Logging::EmitModuleError(kModuleName, oss.str());
        return false;
    }
    
    // Update stats
    sync_count_.fetch_add(1, std::memory_order_relaxed);
    stats_.total_syncs++;
    stats_.total_sync_time_ms += sync_time_ms;
    
    return true;
}

void StreamController::fatalIfDefaultStream(cudaStream_t stream, const char* context) {
    bool is_default = (stream == nullptr);
#if defined(cudaStreamLegacy)
    is_default = is_default || (stream == cudaStreamLegacy);
#endif
#if defined(cudaStreamPerThread)
    is_default = is_default || (stream == cudaStreamPerThread);
#endif

    if (!is_default) {
        return;
    }

    std::ostringstream oss;
    oss << "FATAL: default CUDA stream is forbidden";
    if (context && context[0] != '\0') {
        oss << " (" << context << ")";
    }
    Logging::EmitModuleError(kModuleName, oss.str());
    std::abort();
}

} // namespace GRIM
