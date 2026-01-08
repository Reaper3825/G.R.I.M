#pragma once
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "../../Layers/grim_layer_gpu.hpp"
#include "../Delegate/Delegate.hpp"

namespace GRIM {

// Simple handle for callback registration (GRIM-text local, not from core/delegate.hpp)
class DelegateHandle {
public:
    DelegateHandle() : id_(0) {}
    explicit DelegateHandle(std::uint64_t id) : id_(id) {}
    
    bool IsValid() const { return id_ != 0; }
    bool operator==(const DelegateHandle& other) const { return id_ == other.id_; }
    bool operator!=(const DelegateHandle& other) const { return id_ != other.id_; }
    std::uint64_t id() const { return id_; }
    
private:
    friend struct std::hash<DelegateHandle>;
    std::uint64_t id_;
};

} // namespace GRIM

// Hash specialization for DelegateHandle (must be in std namespace)
namespace std {
    template<> struct hash<GRIM::DelegateHandle> {
        std::size_t operator()(const GRIM::DelegateHandle& h) const noexcept {
            return std::hash<std::uint64_t>{}(h.id_);
        }
    };
}

namespace GRIM {

// CPU-side multicast delegate for module logging callbacks (GRIM-text local)
template<typename... Args>
class MulticastDelegate {
public:
    using Callback = std::function<void(Args...)>;

    DelegateHandle Add(const Callback& func) {
        DelegateHandle handle(++last_id_);
        listeners_[handle] = func;
        return handle;
    }

    bool Remove(const DelegateHandle& handle) {
        return listeners_.erase(handle) > 0;
    }

    void Clear() { listeners_.clear(); }

    void Broadcast(Args... args) const {
        for (const auto& [_, func] : listeners_) {
            func(args...);
        }
    }

    bool IsEmpty() const { return listeners_.empty(); }
    std::size_t Count() const { return listeners_.size(); }

private:
    std::unordered_map<DelegateHandle, Callback> listeners_;
    std::uint64_t last_id_ = 0;
};

std::string layerTypeToString(LayerType type);
bool CreateLayerFolder(const std::string& absolutePath, LayerType type);
void InitLogRecorder();

} // namespace GRIM

namespace GRIM::Logging {

enum class ModuleLogLevel : int {
    Info = 0,
    Warning = 1,
    Error = 2
};

enum class ModuleId : int {
    ForwardPass = 0,
    BackwardPass = 1,
    Optimizer = 2,
    Scheduler = 3,
    Activations = 4,
    GuessCache = 5,
    Validation = 6,
    Checkpoint = 7,
    DataLoader = 8,
    Inference = 9,
    LogRecorder = 10,
    Training = 11,
    TrainingOrchestrator = 12,
    StreamController = 13,
    Loss = 14,
    Attention = 15,
    Custom = 16
};

struct ModuleLogOverride {
    std::string module;
    ModuleLogLevel level = ModuleLogLevel::Info;
};

struct ModuleLogEvent {
    std::string module;
    ModuleLogLevel level = ModuleLogLevel::Info;
    std::string message;
    std::uint64_t global_step = 0;
};

using ModuleLogCallback = std::function<void(const ModuleLogEvent&)>;

class ModuleLogSink {
public:
    ModuleLogSink() = default;
    ModuleLogSink(const std::string& module_name, ModuleLogCallback callback);
    ModuleLogSink(const ModuleLogSink&) = delete;
    ModuleLogSink& operator=(const ModuleLogSink&) = delete;
    ModuleLogSink(ModuleLogSink&& other) noexcept;
    ModuleLogSink& operator=(ModuleLogSink&& other) noexcept;
    ~ModuleLogSink();

    bool bind(const std::string& module_name, ModuleLogCallback callback);
    void reset();
    bool active() const { return handle_.IsValid(); }

private:
    std::string module_;
    GRIM::DelegateHandle handle_{};
};

const char* ModuleIdToString(ModuleId id);
const char* ModuleLogLevelToString(ModuleLogLevel level);
inline ModuleLogOverride MakeOverride(ModuleId id, ModuleLogLevel level) {
    return {ModuleIdToString(id), level};
}

void SetDefaultModuleLogLevel(ModuleLogLevel level);
ModuleLogLevel GetDefaultModuleLogLevel();
void SetModuleLogLevel(const std::string& module_name, ModuleLogLevel level);
void ClearModuleLogLevel(const std::string& module_name);
ModuleLogLevel GetModuleLogLevel(const std::string& module_name);
bool ApplyModuleLogOverride(const ModuleLogOverride& override_desc);
bool ApplyModuleLogOverrides(const std::vector<ModuleLogOverride>& overrides);
bool RegisterModuleLogProfile(const std::string& profile_name,
                              const std::vector<ModuleLogOverride>& overrides);
bool ApplyModuleLogProfile(const std::string& profile_name);
bool HasModuleLogProfile(const std::string& profile_name);

GRIM::DelegateHandle RegisterModuleLogSink(const std::string& module_name,
                                     ModuleLogCallback callback);
bool UnregisterModuleLogSink(const std::string& module_name,
                             const GRIM::DelegateHandle& handle);

// Async logging control
void SetModuleLogAsync(bool enabled);
bool IsModuleLogAsync();
void FlushModuleLogQueue();

void EmitModuleLog(const std::string& module_name,
                   ModuleLogLevel level,
                   std::string_view message,
                   std::uint64_t global_step = 0,
                   bool force_sync = false);
inline void EmitModuleInfo(const std::string& module_name,
                           std::string_view message,
                           std::uint64_t global_step = 0,
                           bool force_sync = false) {
    EmitModuleLog(module_name, ModuleLogLevel::Info, message, global_step, force_sync);
}
inline void EmitModuleWarning(const std::string& module_name,
                              std::string_view message,
                              std::uint64_t global_step = 0,
                              bool force_sync = false) {
    EmitModuleLog(module_name, ModuleLogLevel::Warning, message, global_step, force_sync);
}
inline void EmitModuleError(const std::string& module_name,
                            std::string_view message,
                            std::uint64_t global_step = 0,
                            bool force_sync = false) {
    EmitModuleLog(module_name, ModuleLogLevel::Error, message, global_step, force_sync);
}

inline void EmitModuleLog(ModuleId id,
                          ModuleLogLevel level,
                          std::string_view message,
                          std::uint64_t global_step = 0,
                          bool force_sync = false) {
    EmitModuleLog(ModuleIdToString(id), level, message, global_step, force_sync);
}

inline void EmitModuleInfo(ModuleId id,
                           std::string_view message,
                           std::uint64_t global_step = 0) {
    EmitModuleLog(id, ModuleLogLevel::Info, message, global_step);
}

inline void EmitModuleWarning(ModuleId id,
                              std::string_view message,
                              std::uint64_t global_step = 0) {
    EmitModuleLog(id, ModuleLogLevel::Warning, message, global_step);
}

inline void EmitModuleError(ModuleId id,
                            std::string_view message,
                            std::uint64_t global_step = 0) {
    EmitModuleLog(id, ModuleLogLevel::Error, message, global_step);
}

constexpr std::size_t kMaxTagLength = 32;
constexpr std::size_t kMaxMessageLength = 96;
constexpr int kMaxDeviceLogCallbacks = 8;

struct LayerLogEntry {
    LayerType type = LayerType::kUnknown;
    int layer_index = -1;
    std::uint64_t global_step = 0;
    float primary_value = 0.0f;
    float secondary_value = 0.0f;
    char tag[kMaxTagLength] = {0};
    char message[kMaxMessageLength] = {0};
};

struct DeviceLogBuffer {
    LayerLogEntry* entries = nullptr;
    int* cursor = nullptr;
    int capacity = 0;
};

using DeviceLogDelegate = GPUMulticastDelegate<kMaxDeviceLogCallbacks, const LayerLogEntry&>;

bool InitLogRecorder(const std::string& rootPath = std::string(),
                     std::size_t maxDeviceEntries = 4096);
void ShutdownLogRecorder();
void FlushDeviceLogs();
void ResetDeviceLogs();
bool LogsInitialized();
const std::string& GetLogsRoot();
DeviceLogBuffer GetDeviceLogBuffer();

cudaError_t InstallDefaultDeviceLogger();
cudaError_t RegisterDeviceLogCallback(DeviceLogDelegate::Callback callback);
cudaError_t ClearDeviceLogCallbacks();

__device__ void RecordLayerLog(const LayerLogEntry& entry);
__device__ void RecordLayerLogSimple(LayerType type,
                                     int layer_index,
                                     std::uint64_t global_step,
                                     float primary_value,
                                     float secondary_value,
                                     const char* tag,
                                     const char* message);

// Host-side helper to write a single entry immediately (bypasses the device buffer).
bool RecordLayerLogHost(LayerType type,
                        int layer_index,
                        std::uint64_t global_step,
                        float primary_value,
                        float secondary_value,
                        const std::string& tag,
                        const std::string& message);

// Configure which layer types are logged (call from ai_config.json parsing)
void ConfigureLayerLogging(bool master_enabled,
                           bool embedding,
                           bool rms_norm,
                           bool attention,
                           bool feed_forward,
                           bool residual,
                           bool encoding,
                           bool serialization);

// Check if a specific layer type is enabled for logging
bool IsLayerLoggingEnabled(LayerType type);

class ScopedLogRecorder {
public:
    ScopedLogRecorder() = default;
    ScopedLogRecorder(const ScopedLogRecorder&) = delete;
    ScopedLogRecorder& operator=(const ScopedLogRecorder&) = delete;
    ScopedLogRecorder(ScopedLogRecorder&& other) noexcept;
    ScopedLogRecorder& operator=(ScopedLogRecorder&& other) noexcept;
    ~ScopedLogRecorder();

    bool init(const std::string& root_path, std::size_t max_device_entries = 4096);
    void shutdown();
    void flush() const;
    bool active() const { return active_; }

private:
    bool active_ = false;
};

//======================================================//
//  LogMirrorScope - Mirrors log file to console
//  Background thread tails a log file and writes to stdout
//======================================================//

class LogMirrorScope {
public:
    explicit LogMirrorScope(const std::string& log_path);
    LogMirrorScope(const LogMirrorScope&) = delete;
    LogMirrorScope& operator=(const LogMirrorScope&) = delete;
    ~LogMirrorScope();

    bool active() const { return !stop_.load(); }

private:
    bool start(const std::string& log_path);
    void mirrorLoop();
    void writeToConsole(const std::string& text);

    std::ifstream log_stream_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
#ifdef _WIN32
    HANDLE console_out_ = INVALID_HANDLE_VALUE;
#else
    int console_fd_ = -1;
#endif
};

//======================================================//
//  Logging Configuration Parsing Utilities
//======================================================//

// Parse a string representation of log level (e.g., "info", "warn", "error")
ModuleLogLevel ParseModuleLogLevelString(
    const std::string& text,
    ModuleLogLevel fallback = ModuleLogLevel::Info);

// Parse a module:level spec string (e.g., "ForwardPass:warn")
bool ParseModuleOverrideSpec(const std::string& spec, ModuleLogOverride& out);

// Register standard logging profiles (forward_pass, backward_pass, optimizer, validation)
void RegisterDefaultLoggingProfiles();

} // namespace GRIM::Logging
