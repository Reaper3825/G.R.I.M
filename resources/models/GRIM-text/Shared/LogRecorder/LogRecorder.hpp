#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../Layers/grim_layer_gpu.hpp"

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
    Custom = 16,
    Autograd = 17,
    ExecutionBlock = 18
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

struct LayerLogEntry {
    LayerType type = LayerType::kUnknown;
    int layer_index = -1;
    std::uint64_t global_step = 0;
    float primary_value = 0.0f;
    float secondary_value = 0.0f;
    char tag[kMaxTagLength] = {0};
    char message[kMaxMessageLength] = {0};
};

bool InitLogRecorder(const std::string& rootPath,
                     std::size_t maxDeviceEntries = 4096);
void ShutdownLogRecorder();
void FlushDeviceLogs();
void ResetDeviceLogs();
bool LogsInitialized();
const std::string& GetLogsRoot();

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
                           bool serialization,
                           bool execution_block);

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
//  Module Log Formatting Helpers
//======================================================//

// Create a standard module log formatter that routes events to a logging function.
// The log_fn parameter is called with formatted log strings.
// The returned callback converts ModuleLogEvent to formatted string and logs via log_fn.
ModuleLogCallback CreateStandardModuleLogFormatter(std::function<void(const std::string&)> log_fn);

} // namespace GRIM::Logging
