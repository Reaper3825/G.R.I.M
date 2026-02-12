/**
 * @file StreamController_GPU.hpp
 * @brief Centralized CUDA stream management for GRIM-text
 *
 * DESIGN PHILOSOPHY:
 * StreamController is OWNED by TrainingState (not a singleton). This ensures
 * proper ownership semantics and prevents stream fragmentation. All components
 * that need streams should get them from training_state_.stream_ctrl.
 *
 * STREAM TOPOLOGY:
 * ┌────────────────────────────────────────────────────┐
 * │              TrainingState.stream_ctrl              │
 * │  ┌──────────────────────────────────────────────┐  │
 * │  │  Primary Stream (all compute + transfers)    │  │
 * │  └──────────────────────────────────────────────┘  │
 * └────────────────────────────────────────────────────┘
 *
 * USAGE PATTERN (via TrainingState):
 * ─────────────────────────────────────────────────────────────────
 *   // Access via TrainingState (NO SINGLETON)
 *   cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
 *   
 *   // Kernel launch
 *   myKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(...);
 *   
 *   // Sync
 *   training_state_.stream_ctrl.syncPrimaryStream();
 * ─────────────────────────────────────────────────────────────────
 *
 * THREAD SAFETY:
 * - Stream getters are lock-free after initialization
 * - Initialize/shutdown are NOT thread-safe (call from main thread)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <atomic>

#include "../LogRecorder/LogRecorder.hpp"

namespace GRIM {

//======================================================//
//  Configuration
//======================================================//

struct StreamControllerConfig {
    // Stream creation flags
    unsigned int primary_flags = cudaStreamNonBlocking;
    
    // Priority (lower = higher priority, -1 = default)
    int primary_priority = -1;
    
    // Behavior
    bool verbose = false;                   ///< Log stream operations
};

//======================================================//
//  Statistics
//======================================================//

struct StreamControllerStats {
    uint64_t streams_created = 0;
    uint64_t streams_destroyed = 0;
    uint64_t total_syncs = 0;
    double total_sync_time_ms = 0.0;
};

//======================================================//
//  Main Controller Class
//======================================================//

class StreamController {
public:
    //--------------------------------------------------//
    //  Lifecycle (PUBLIC - owned by TrainingState)
    //--------------------------------------------------//
    
    StreamController();
    ~StreamController();
    
    // Non-copyable, non-movable (CUDA resources)
    StreamController(const StreamController&) = delete;
    StreamController& operator=(const StreamController&) = delete;
    StreamController(StreamController&&) = delete;
    StreamController& operator=(StreamController&&) = delete;
    
    /**
     * @brief Initialize the primary stream
     * @param config Configuration options
     * @return true on success
     * @throws std::runtime_error if already initialized (Rule 20: fail loud)
     * @note NOT thread-safe - call from main thread
     */
    bool initialize(const StreamControllerConfig& config = {});
    
    /**
     * @brief Shutdown and destroy the owned stream
     * @note NOT thread-safe - call from main thread
     */
    void shutdown();
    
    /**
     * @brief Check if controller is initialized
     */
    bool isInitialized() const { return initialized_.load(std::memory_order_acquire); }
    
    //--------------------------------------------------//
    //  Stream Access
    //--------------------------------------------------//
    
    /**
     * @brief Get the primary compute stream
     * @return CUDA stream handle
     * @throws std::runtime_error if not initialized (Rule 20: no silent nullptr)
     */
    cudaStream_t getPrimaryStream() const;
    
    //--------------------------------------------------//
    //  Synchronization
    //--------------------------------------------------//
    
    /**
     * @brief Synchronize the primary stream
     * @return true on success
     */
    bool syncPrimaryStream();
    
    //--------------------------------------------------//
    //  Diagnostics
    //--------------------------------------------------//
    
    /**
     * @brief Get statistics
     */
    const StreamControllerStats& getStats() const { return stats_; }

    /**
     * @brief Fatal guard against default (null) stream usage
     * @param stream CUDA stream handle
     * @param context Context string for diagnostics
     */
    static void fatalIfDefaultStream(cudaStream_t stream, const char* context);

private:
    //--------------------------------------------------//
    //  Member Data
    //--------------------------------------------------//
    
    std::atomic<bool> initialized_{false};
    StreamControllerConfig config_;
    
    // Primary stream
    cudaStream_t primary_stream_ = nullptr;
    bool stream_owned_ = true;
    
    // Sync statistics
    std::atomic<uint64_t> sync_count_{0};
    
    // Statistics
    StreamControllerStats stats_;
    
    // Module name for logging
    static constexpr const char* kModuleName = "StreamController";
};

// NO GLOBAL CONVENIENCE FUNCTIONS
// Access streams via: training_state_.stream_ctrl.getPrimaryStream()

} // namespace GRIM
