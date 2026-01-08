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
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                     TrainingState.stream_ctrl                          │
 * │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
 * │  │ Primary      │  │ Transfer     │  │ Auxiliary    │                 │
 * │  │ (Compute)    │  │ (H2D/D2H)    │  │ (Optional)   │                 │
 * │  └──────────────┘  └──────────────┘  └──────────────┘                 │
 * │         │                 │                 │                          │
 * │         ▼                 ▼                 ▼                          │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │              Event-based Synchronization                         │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
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
 *   
 *   // Cross-stream synchronization
 *   int event_id = training_state_.stream_ctrl.recordEvent(StreamType::Primary);
 *   training_state_.stream_ctrl.waitEvent(StreamType::Transfer, event_id);
 * ─────────────────────────────────────────────────────────────────
 *
 * THREAD SAFETY:
 * - Stream getters are lock-free after initialization
 * - Initialize/shutdown are NOT thread-safe (call from main thread)
 * - Event recording/waiting is thread-safe
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <atomic>
#include <array>
#include <functional>
#include <vector>
#include <mutex>

#include "../LogRecorder/LogRecorder.hpp"

namespace GRIM {

//======================================================//
//  Stream Type Enumeration
//======================================================//

/**
 * @brief Stream categories for different workload types
 */
enum class StreamType : uint8_t
{
    Primary = 0,        ///< Main compute stream (forward/backward)
    Transfer = 1,       ///< Host-Device transfers (async memcpy)
    Auxiliary = 2,      ///< Secondary compute (validation, stats)
    kCount = 3          ///< Number of stream types
};

const char* StreamTypeToString(StreamType type);

//======================================================//
//  Stream Events (for cross-stream synchronization)
//======================================================//

/**
 * @brief Manages CUDA events for stream synchronization
 */
struct StreamEvent {
    cudaEvent_t event = nullptr;
    bool valid = false;
    std::string name;

    StreamEvent() = default;
    explicit StreamEvent(const std::string& event_name);
    ~StreamEvent();

    // Non-copyable, movable
    StreamEvent(const StreamEvent&) = delete;
    StreamEvent& operator=(const StreamEvent&) = delete;
    StreamEvent(StreamEvent&& other) noexcept;
    StreamEvent& operator=(StreamEvent&& other) noexcept;

    bool create();
    void destroy();
    bool record(cudaStream_t stream);
    bool synchronize();
    bool isReady() const;
};

//======================================================//
//  Stream Descriptor
//======================================================//

/**
 * @brief Holds stream metadata and ownership info
 */
struct StreamDescriptor {
    cudaStream_t stream = nullptr;
    StreamType type = StreamType::Primary;
    bool owned = true;                      ///< If true, controller destroys on shutdown
    bool initialized = false;
    std::string name;
    
    // Statistics
    std::atomic<uint64_t> sync_count{0};
    std::atomic<uint64_t> total_sync_time_us{0};

    StreamDescriptor() = default;
    StreamDescriptor(StreamType t, const std::string& n)
        : type(t), name(n) {}
};

//======================================================//
//  Configuration
//======================================================//

struct StreamControllerConfig {
    // Stream creation flags
    unsigned int primary_flags = cudaStreamNonBlocking;
    unsigned int transfer_flags = cudaStreamNonBlocking;
    unsigned int auxiliary_flags = cudaStreamNonBlocking;
    
    // Priority (lower = higher priority, -1 = default)
    int primary_priority = -1;
    int transfer_priority = -1;
    int auxiliary_priority = -1;
    
    // Behavior
    bool create_transfer_stream = true;     ///< Create separate transfer stream
    bool create_auxiliary_stream = false;   ///< Create auxiliary stream on demand
    bool verbose = false;                   ///< Log stream operations
    bool sync_on_error = true;              ///< Full sync on CUDA error
};

//======================================================//
//  Statistics
//======================================================//

struct StreamControllerStats {
    uint64_t streams_created = 0;
    uint64_t streams_destroyed = 0;
    uint64_t total_syncs = 0;
    uint64_t events_created = 0;
    uint64_t events_recorded = 0;
    double total_sync_time_ms = 0.0;
    
    void reset() {
        streams_created = 0;
        streams_destroyed = 0;
        total_syncs = 0;
        events_created = 0;
        events_recorded = 0;
        total_sync_time_ms = 0.0;
    }
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
     * @brief Initialize streams based on configuration
     * @param config Configuration options
     * @return true on success
     * @note NOT thread-safe - call from main thread
     */
    bool initialize(const StreamControllerConfig& config = {});
    
    /**
     * @brief Shutdown and destroy all owned streams
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
     * @return CUDA stream handle (may be nullptr if not initialized)
     */
    cudaStream_t getPrimaryStream() const;
    
    /**
     * @brief Get the transfer stream (for H2D/D2H copies)
     * @return CUDA stream handle (may be nullptr if not created)
     */
    cudaStream_t getTransferStream() const;
    
    /**
     * @brief Get the auxiliary stream
     * @return CUDA stream handle (may be nullptr if not created)
     */
    cudaStream_t getAuxiliaryStream() const;
    
    /**
     * @brief Get stream by type
     * @param type Stream type enum
     * @return CUDA stream handle
     */
    cudaStream_t getStream(StreamType type) const;
    
    //--------------------------------------------------//
    //  External Stream Injection
    //--------------------------------------------------//
    
    /**
     * @brief Set an externally-managed primary stream
     * @param stream External stream (controller does NOT own this)
     * @note Useful when receiving stream from Python/external source
     */
    void setExternalPrimaryStream(cudaStream_t stream);
    
    /**
     * @brief Set an externally-managed stream of any type
     * @param type Stream type to set
     * @param stream External stream (controller does NOT own this)
     */
    void setExternalStream(StreamType type, cudaStream_t stream);
    
    //--------------------------------------------------//
    //  Synchronization
    //--------------------------------------------------//
    
    /**
     * @brief Synchronize the primary stream
     * @return true on success
     */
    bool syncPrimaryStream();
    
    /**
     * @brief Synchronize a specific stream
     * @param type Stream type to sync
     * @return true on success
     */
    bool syncStream(StreamType type);
    
    /**
     * @brief Synchronize all managed streams
     * @return true if all syncs succeeded
     */
    bool syncAllStreams();
    
    /**
     * @brief Record an event on a stream for later synchronization
     * @param type Stream to record on
     * @param event_name Name for the event
     * @return Event index (or -1 on failure)
     */
    int recordEvent(StreamType type, const std::string& event_name = "");
    
    /**
     * @brief Wait for an event on a different stream
     * @param wait_stream Stream that should wait
     * @param event_index Index returned by recordEvent
     * @return true on success
     */
    bool waitEvent(StreamType wait_stream, int event_index);
    
    //--------------------------------------------------//
    //  Diagnostics
    //--------------------------------------------------//
    
    /**
     * @brief Get statistics
     */
    const StreamControllerStats& getStats() const { return stats_; }
    
    /**
     * @brief Reset statistics
     */
    void resetStats() { stats_.reset(); }
    
    /**
     * @brief Print current state to log
     */
    void logStatus() const;
    
    /**
     * @brief Check CUDA error and log if present
     * @param operation Description of the operation
     * @return true if no error
     */
    bool checkCudaError(const char* operation) const;

    /**
     * @brief Fatal guard against default (null) stream usage
     * @param stream CUDA stream handle
     * @param context Context string for diagnostics
     */
    static void fatalIfDefaultStream(cudaStream_t stream, const char* context);

private:
    //--------------------------------------------------//
    //  Private Implementation
    //--------------------------------------------------//
    
    bool createStream(StreamType type);
    void destroyStream(StreamType type);
    
    //--------------------------------------------------//
    //  Member Data
    //--------------------------------------------------//
    
    std::atomic<bool> initialized_{false};
    StreamControllerConfig config_;
    
    // Stream storage (fixed array for lock-free access)
    std::array<StreamDescriptor, static_cast<size_t>(StreamType::kCount)> streams_;
    
    // Event pool for cross-stream synchronization
    static constexpr int kMaxEvents = 16;
    std::array<StreamEvent, kMaxEvents> events_;
    std::atomic<int> next_event_idx_{0};
    
    // Statistics
    StreamControllerStats stats_;
    
    // Module name for logging
    static constexpr const char* kModuleName = "StreamController";
};

// NO GLOBAL CONVENIENCE FUNCTIONS
// Access streams via: training_state_.stream_ctrl.getPrimaryStream()

} // namespace GRIM
