/**
 * @file StreamController_GPU.cu
 * @brief Implementation of centralized CUDA stream management
 */

#include "StreamController_GPU.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace GRIM {

//======================================================//
//  StreamType Utilities
//======================================================//

const char* StreamTypeToString(StreamType type) {
    switch (type) {
        case StreamType::Primary:   return "Primary";
        case StreamType::Transfer:  return "Transfer";
        case StreamType::Auxiliary: return "Auxiliary";
        default:                    return "Unknown";
    }
}

//======================================================//
//  StreamEvent Implementation
//======================================================//

StreamEvent::StreamEvent(const std::string& event_name)
    : name(event_name) {
    create();
}

StreamEvent::~StreamEvent() {
    destroy();
}

StreamEvent::StreamEvent(StreamEvent&& other) noexcept
    : event(other.event)
    , valid(other.valid)
    , name(std::move(other.name)) {
    other.event = nullptr;
    other.valid = false;
}

StreamEvent& StreamEvent::operator=(StreamEvent&& other) noexcept {
    if (this != &other) {
        destroy();
        event = other.event;
        valid = other.valid;
        name = std::move(other.name);
        other.event = nullptr;
        other.valid = false;
    }
    return *this;
}

bool StreamEvent::create() {
    if (valid) return true;
    
    cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        Logging::EmitModuleError("StreamController",
            std::string("Failed to create event '") + name + "': " + cudaGetErrorString(err));
        return false;
    }
    valid = true;
    return true;
}

void StreamEvent::destroy() {
    if (valid && event) {
        cudaEventDestroy(event);
        event = nullptr;
        valid = false;
    }
}

bool StreamEvent::record(cudaStream_t stream) {
    if (!valid) {
        if (!create()) return false;
    }
    
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        Logging::EmitModuleError("StreamController",
            std::string("Failed to record event '") + name + "': " + cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool StreamEvent::synchronize() {
    if (!valid) return false;
    
    cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess) {
        Logging::EmitModuleError("StreamController",
            std::string("Failed to sync event '") + name + "': " + cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool StreamEvent::isReady() const {
    if (!valid) return true;  // No event = nothing to wait for
    
    cudaError_t err = cudaEventQuery(event);
    return (err == cudaSuccess);
}

//======================================================//
//  StreamController Implementation
//======================================================//

StreamController::StreamController() {
    // Initialize stream descriptors with names
    // Use direct member initialization since StreamDescriptor may contain non-copyable members
    streams_[static_cast<size_t>(StreamType::Primary)].type = StreamType::Primary;
    streams_[static_cast<size_t>(StreamType::Primary)].name = "Primary";
    
    streams_[static_cast<size_t>(StreamType::Transfer)].type = StreamType::Transfer;
    streams_[static_cast<size_t>(StreamType::Transfer)].name = "Transfer";
    
    streams_[static_cast<size_t>(StreamType::Auxiliary)].type = StreamType::Auxiliary;
    streams_[static_cast<size_t>(StreamType::Auxiliary)].name = "Auxiliary";
}

StreamController::~StreamController() {
    if (initialized_.load(std::memory_order_acquire)) {
        shutdown();
    }
}

bool StreamController::initialize(const StreamControllerConfig& config) {
    if (initialized_.load(std::memory_order_acquire)) {
        Logging::EmitModuleWarning(kModuleName, "Already initialized, ignoring");
        return true;
    }
    
    config_ = config;
    
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "Initializing StreamController...");
    }
    
    // Create primary stream (always)
    if (!createStream(StreamType::Primary)) {
        Logging::EmitModuleError(kModuleName, "Failed to create primary stream");
        return false;
    }
    fatalIfDefaultStream(streams_[static_cast<size_t>(StreamType::Primary)].stream,
                         "StreamController::initialize primary");
    
    // Create transfer stream (optional)
    if (config_.create_transfer_stream) {
        if (!createStream(StreamType::Transfer)) {
            Logging::EmitModuleWarning(kModuleName, 
                "Failed to create transfer stream, using primary for transfers");
        } else {
            fatalIfDefaultStream(streams_[static_cast<size_t>(StreamType::Transfer)].stream,
                                 "StreamController::initialize transfer");
        }
    }
    
    // Create auxiliary stream (optional)
    if (config_.create_auxiliary_stream) {
        if (!createStream(StreamType::Auxiliary)) {
            Logging::EmitModuleWarning(kModuleName, 
                "Failed to create auxiliary stream");
        } else {
            fatalIfDefaultStream(streams_[static_cast<size_t>(StreamType::Auxiliary)].stream,
                                 "StreamController::initialize auxiliary");
        }
    }
    
    // Pre-create events
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "Pre-creating events...");
    }
    for (int i = 0; i < kMaxEvents; ++i) {
        events_[i].name = "Event_" + std::to_string(i);
        if (events_[i].create()) {
            stats_.events_created++;
        }
    }
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, std::string("Created ") + std::to_string(stats_.events_created) + " events");
    }
    
    initialized_.store(true, std::memory_order_release);
    
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "Setting initialized flag...");
        std::ostringstream oss;
        oss << "StreamController initialized: "
            << "primary=" << (streams_[0].initialized ? "yes" : "no")
            << ", transfer=" << (streams_[1].initialized ? "yes" : "no")
            << ", auxiliary=" << (streams_[2].initialized ? "yes" : "no");
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
    
    // Sync all streams before destroying
    syncAllStreams();
    
    // Destroy events
    for (auto& event : events_) {
        event.destroy();
    }
    
    // Destroy streams (only owned ones)
    for (size_t i = 0; i < static_cast<size_t>(StreamType::kCount); ++i) {
        destroyStream(static_cast<StreamType>(i));
    }
    
    initialized_.store(false, std::memory_order_release);
    
    if (config_.verbose) {
        Logging::EmitModuleInfo(kModuleName, "StreamController shutdown complete");
    }
}

bool StreamController::createStream(StreamType type) {
    size_t idx = static_cast<size_t>(type);
    StreamDescriptor& desc = streams_[idx];
    
    if (desc.initialized) {
        return true;  // Already created
    }
    
    // Determine flags and priority
    unsigned int flags = cudaStreamDefault;
    int priority = -1;
    
    switch (type) {
        case StreamType::Primary:
            flags = config_.primary_flags;
            priority = config_.primary_priority;
            break;
        case StreamType::Transfer:
            flags = config_.transfer_flags;
            priority = config_.transfer_priority;
            break;
        case StreamType::Auxiliary:
            flags = config_.auxiliary_flags;
            priority = config_.auxiliary_priority;
            break;
        default:
            break;
    }
    
    cudaError_t err;
    
    if (priority >= 0) {
        // Create with priority
        err = cudaStreamCreateWithPriority(&desc.stream, flags, priority);
    } else {
        // Create with flags only
        err = cudaStreamCreateWithFlags(&desc.stream, flags);
    }
    
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to create " << StreamTypeToString(type) 
            << " stream: " << cudaGetErrorString(err);
        Logging::EmitModuleError(kModuleName, oss.str());
        return false;
    }
    
    desc.initialized = true;
    desc.owned = true;
    stats_.streams_created++;
    
    if (config_.verbose) {
        std::ostringstream oss;
        oss << "Created " << StreamTypeToString(type) << " stream";
        if (priority >= 0) {
            oss << " (priority=" << priority << ")";
        }
        Logging::EmitModuleInfo(kModuleName, oss.str());
    }
    
    return true;
}

void StreamController::destroyStream(StreamType type) {
    size_t idx = static_cast<size_t>(type);
    StreamDescriptor& desc = streams_[idx];
    
    if (!desc.initialized) {
        return;
    }
    
    // Only destroy if we own it
    if (desc.owned && desc.stream) {
        cudaStreamDestroy(desc.stream);
        stats_.streams_destroyed++;
        
        if (config_.verbose) {
            std::ostringstream oss;
            oss << "Destroyed " << StreamTypeToString(type) << " stream";
            Logging::EmitModuleInfo(kModuleName, oss.str());
        }
    }
    
    desc.stream = nullptr;
    desc.initialized = false;
    desc.owned = true;
}

cudaStream_t StreamController::getPrimaryStream() const {
    return streams_[static_cast<size_t>(StreamType::Primary)].stream;
}

cudaStream_t StreamController::getTransferStream() const {
    const auto& desc = streams_[static_cast<size_t>(StreamType::Transfer)];
    // Fall back to primary if transfer not created
    return desc.initialized ? desc.stream : getPrimaryStream();
}

cudaStream_t StreamController::getAuxiliaryStream() const {
    const auto& desc = streams_[static_cast<size_t>(StreamType::Auxiliary)];
    // Fall back to primary if auxiliary not created
    return desc.initialized ? desc.stream : getPrimaryStream();
}

cudaStream_t StreamController::getStream(StreamType type) const {
    switch (type) {
        case StreamType::Primary:   return getPrimaryStream();
        case StreamType::Transfer:  return getTransferStream();
        case StreamType::Auxiliary: return getAuxiliaryStream();
        default:                    return getPrimaryStream();
    }
}

void StreamController::setExternalPrimaryStream(cudaStream_t stream) {
    setExternalStream(StreamType::Primary, stream);
}

void StreamController::setExternalStream(StreamType type, cudaStream_t stream) {
    size_t idx = static_cast<size_t>(type);
    StreamDescriptor& desc = streams_[idx];

    fatalIfDefaultStream(stream, "StreamController::setExternalStream");
    
    // If we had an owned stream, destroy it first
    if (desc.owned && desc.initialized && desc.stream) {
        cudaStreamDestroy(desc.stream);
        stats_.streams_destroyed++;
    }
    
    desc.stream = stream;
    desc.owned = false;  // External stream - we don't own it
    desc.initialized = (stream != nullptr);
    
    if (config_.verbose) {
        std::ostringstream oss;
        oss << "Set external " << StreamTypeToString(type) << " stream";
        Logging::EmitModuleInfo(kModuleName, oss.str());
    }
}

bool StreamController::syncPrimaryStream() {
    return syncStream(StreamType::Primary);
}

bool StreamController::syncStream(StreamType type) {
    size_t idx = static_cast<size_t>(type);
    StreamDescriptor& desc = streams_[idx];
    
    if (!desc.initialized || !desc.stream) {
        return true;  // Nothing to sync
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaError_t err = cudaStreamSynchronize(desc.stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    double sync_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to sync " << StreamTypeToString(type) 
            << " stream: " << cudaGetErrorString(err);
        Logging::EmitModuleError(kModuleName, oss.str());
        return false;
    }
    
    // Update stats
    desc.sync_count.fetch_add(1, std::memory_order_relaxed);
    stats_.total_syncs++;
    stats_.total_sync_time_ms += sync_time_ms;
    
    return true;
}

bool StreamController::syncAllStreams() {
    bool success = true;
    
    for (size_t i = 0; i < static_cast<size_t>(StreamType::kCount); ++i) {
        if (!syncStream(static_cast<StreamType>(i))) {
            success = false;
        }
    }
    
    return success;
}

int StreamController::recordEvent(StreamType type, const std::string& event_name) {
    // Get next event slot (circular)
    int idx = next_event_idx_.fetch_add(1, std::memory_order_relaxed) % kMaxEvents;
    StreamEvent& event = events_[idx];
    
    if (!event_name.empty()) {
        event.name = event_name;
    }
    
    cudaStream_t stream = getStream(type);
    if (!event.record(stream)) {
        return -1;
    }
    
    stats_.events_recorded++;
    return idx;
}

bool StreamController::waitEvent(StreamType wait_stream, int event_index) {
    if (event_index < 0 || event_index >= kMaxEvents) {
        Logging::EmitModuleError(kModuleName, "Invalid event index");
        return false;
    }
    
    StreamEvent& event = events_[event_index];
    if (!event.valid) {
        Logging::EmitModuleError(kModuleName, "Event not valid");
        return false;
    }
    
    cudaStream_t stream = getStream(wait_stream);
    cudaError_t err = cudaStreamWaitEvent(stream, event.event, 0);
    
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "Failed to wait on event: " << cudaGetErrorString(err);
        Logging::EmitModuleError(kModuleName, oss.str());
        return false;
    }
    
    return true;
}

void StreamController::logStatus() const {
    std::ostringstream oss;
    oss << "StreamController Status:\n";
    oss << "  Initialized: " << (initialized_.load() ? "yes" : "no") << "\n";
    
    for (size_t i = 0; i < static_cast<size_t>(StreamType::kCount); ++i) {
        const auto& desc = streams_[i];
        oss << "  " << StreamTypeToString(static_cast<StreamType>(i)) << ": ";
        if (desc.initialized) {
            oss << "active (owned=" << (desc.owned ? "yes" : "no") 
                << ", syncs=" << desc.sync_count.load() << ")";
        } else {
            oss << "not created";
        }
        oss << "\n";
    }
    
    oss << "  Stats: created=" << stats_.streams_created
        << ", destroyed=" << stats_.streams_destroyed
        << ", total_syncs=" << stats_.total_syncs
        << ", total_sync_time=" << stats_.total_sync_time_ms << "ms";
    
    Logging::EmitModuleInfo(kModuleName, oss.str());
}

bool StreamController::checkCudaError(const char* operation) const {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << operation << " failed: " << cudaGetErrorString(err);
        Logging::EmitModuleError(kModuleName, oss.str());
        
        if (config_.sync_on_error) {
            cudaDeviceSynchronize();
        }
        return false;
    }
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
