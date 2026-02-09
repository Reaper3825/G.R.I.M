#include "LogRecorder.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace GRIM;
using namespace GRIM::Logging;

namespace fs = std::filesystem;

namespace {

std::mutex g_host_mutex;
bool g_initialized = false;
std::string g_logs_root;

std::unordered_map<std::string, std::FILE*> g_log_handles;

using ModuleDelegate = GRIM::MulticastDelegate<const ModuleLogEvent&>;
std::mutex g_module_log_mutex;
std::unordered_map<std::string, std::shared_ptr<ModuleDelegate>> g_module_loggers;
std::unordered_map<std::string, ModuleLogLevel> g_module_level_overrides;
ModuleLogLevel g_default_module_level = ModuleLogLevel::Info;
std::unordered_map<std::string, std::vector<ModuleLogOverride>> g_log_profiles;

// Async logging infrastructure
std::atomic<bool> g_async_logging_enabled{true};  // Default to async
std::queue<ModuleLogEvent> g_log_queue;
std::mutex g_log_queue_mutex;
std::condition_variable g_log_queue_cv;
std::thread g_log_worker;
std::atomic<bool> g_log_worker_stop{false};

// Layer logging enables (global state, set via ConfigureLayerLogging)
bool g_layer_logging_master_enabled = true;
bool g_layer_enables[static_cast<int>(LayerType::kCount)] = {
    false,  // kUnknown
    true,   // kEmbedding
    false,  // kLayerNorm (deprecated)
    true,   // kRMSNorm
    true,   // kAttention
    true,   // kFeedForward
    true,   // kResidual
    true,   // kEncoding
    true    // kSerialization
};

std::string SafeString(const char* buffer, std::size_t max_len) {
    if (!buffer || max_len == 0) {
        return {};
    }
    std::size_t actual = strnlen(buffer, max_len);
    return std::string(buffer, actual);
}

std::FILE* GetLayerFile(LayerType type, int index) {
    const std::string type_name = layerTypeToString(type);
    fs::path folder = fs::path(g_logs_root) / type_name;
    std::error_code ec;
    fs::create_directories(folder, ec);

    std::string file_key = (folder / (std::to_string(index) + ".log")).string();
    auto it = g_log_handles.find(file_key);
    if (it != g_log_handles.end() && it->second) {
        return it->second;
    }
    std::FILE* handle = std::fopen(file_key.c_str(), "a");
    if (!handle) {
        return nullptr;
    }
    g_log_handles[file_key] = handle;
    return handle;
}

void CloseAllFiles() {
    for (auto& kv : g_log_handles) {
        if (kv.second) {
            std::fclose(kv.second);
        }
    }
    g_log_handles.clear();
}

std::shared_ptr<ModuleDelegate> EnsureModuleDelegateLocked(const std::string& module_name) {
    auto it = g_module_loggers.find(module_name);
    if (it == g_module_loggers.end()) {
        auto delegate = std::make_shared<ModuleDelegate>();
        it = g_module_loggers.emplace(module_name, std::move(delegate)).first;
    }
    return it->second;
}

std::shared_ptr<ModuleDelegate> FindModuleDelegateLocked(const std::string& module_name) {
    auto it = g_module_loggers.find(module_name);
    if (it == g_module_loggers.end()) {
        return nullptr;
    }
    return it->second;
}

ModuleLogLevel ResolveModuleThresholdLocked(const std::string& module_name) {
    auto it = g_module_level_overrides.find(module_name);
    if (it != g_module_level_overrides.end()) {
        return it->second;
    }
    return g_default_module_level;
}

bool ShouldEmitLocked(const std::string& module_name, ModuleLogLevel level) {
    const ModuleLogLevel threshold = ResolveModuleThresholdLocked(module_name);
    return static_cast<int>(level) >= static_cast<int>(threshold);
}

void WriteEntryToDisk(const LayerLogEntry& entry) {
    // Check if this layer type is enabled
    if (!g_layer_logging_master_enabled) {
        return;
    }
    const int type_idx = static_cast<int>(entry.type);
    if (type_idx < 0 || type_idx >= static_cast<int>(LayerType::kCount) || !g_layer_enables[type_idx]) {
        return;
    }
    
    std::FILE* handle = GetLayerFile(entry.type, entry.layer_index);
    if (!handle) {
        return;
    }
    const std::string tag = SafeString(entry.tag, kMaxTagLength);
    const std::string message = SafeString(entry.message, kMaxMessageLength);
    std::fprintf(handle,
                 "step=%llu primary=%.6f secondary=%.6f tag=%s message=%s\n",
                 static_cast<unsigned long long>(entry.global_step),
                 static_cast<double>(entry.primary_value),
                 static_cast<double>(entry.secondary_value),
                 tag.c_str(),
                 message.c_str());
    std::fflush(handle);
}

void LogWorkerThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(g_log_queue_mutex);
        g_log_queue_cv.wait(lock, [] {
            return !g_log_queue.empty() || g_log_worker_stop.load();
        });
        
        if (g_log_worker_stop.load() && g_log_queue.empty()) {
            break;
        }
        
        if (!g_log_queue.empty()) {
            ModuleLogEvent event = std::move(g_log_queue.front());
            g_log_queue.pop();
            lock.unlock();
            
            // Process event outside lock
            std::shared_ptr<ModuleDelegate> delegate;
            {
                std::lock_guard<std::mutex> module_lock(g_module_log_mutex);
                delegate = FindModuleDelegateLocked(event.module);
            }
            if (delegate) {
                delegate->Broadcast(event);
            }
        }
    }
}

void CleanupResources() {
    // Shutdown async logging first
    if (g_log_worker.joinable()) {
        g_log_worker_stop.store(true);
        g_log_queue_cv.notify_one();
        g_log_worker.join();
    }
    
    CloseAllFiles();
    g_initialized = false;
}

std::string TrimCopyInternal(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(start, end - start + 1);
}

std::string ToLowerCopyInternal(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

} // namespace

namespace GRIM {

std::string layerTypeToString(LayerType type) {
    switch (type) {
        case LayerType::kEmbedding: return "Embedding";
        case LayerType::kLayerNorm: return "LayerNorm";
        case LayerType::kRMSNorm: return "RMSNorm";
        case LayerType::kAttention: return "Attention";
        case LayerType::kFeedForward: return "FeedForward";
        case LayerType::kResidual: return "Residual";
        case LayerType::kEncoding: return "Encoding";
        case LayerType::kSerialization: return "Serialization";
        case LayerType::kUnknown:
        default:
            return "Unknown";
    }
}

bool CreateLayerFolder(const std::string& absolutePath, LayerType type) {
    fs::path folder = fs::path(absolutePath) / layerTypeToString(type);
    std::error_code ec;
    fs::create_directories(folder, ec);
    return !ec;
}

} // namespace GRIM

namespace GRIM::Logging {

const char* ModuleIdToString(ModuleId id) {
    switch (id) {
        case ModuleId::ForwardPass: return "ForwardPass";
        case ModuleId::BackwardPass: return "BackwardPass";
        case ModuleId::Optimizer: return "Optimizer";
        case ModuleId::Scheduler: return "Scheduler";
        case ModuleId::Activations: return "Activations";
        case ModuleId::GuessCache: return "GuessCache";
        case ModuleId::Validation: return "Validation";
        case ModuleId::Checkpoint: return "Checkpoint";
        case ModuleId::DataLoader: return "DataLoader";
        case ModuleId::Inference: return "Inference";
        case ModuleId::LogRecorder: return "LogRecorder";
        case ModuleId::Training: return "Training";
        case ModuleId::TrainingOrchestrator: return "TrainingOrchestrator";
        case ModuleId::StreamController: return "StreamController";
        case ModuleId::Loss: return "Loss";
        case ModuleId::Attention: return "Attention";
        case ModuleId::Autograd: return "Autograd";
        case ModuleId::Custom:
        default:
            return "Custom";
    }
}

const char* ModuleLogLevelToString(ModuleLogLevel level) {
    switch (level) {
        case ModuleLogLevel::Info: return "info";
        case ModuleLogLevel::Warning: return "warning";
        case ModuleLogLevel::Error: return "error";
        default: return "unknown";
    }
}

bool InitLogRecorder(const std::string& rootPath, std::size_t /*maxDeviceEntries*/) {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    if (g_initialized) {
        return true;
    }

    if (rootPath.empty()) {
        throw std::runtime_error("InitLogRecorder: rootPath is EMPTY - caller MUST provide valid logs directory");
    }
    g_logs_root = rootPath;
    fs::create_directories(g_logs_root);
    for (int i = 1; i < static_cast<int>(LayerType::kCount); ++i) {
        CreateLayerFolder(g_logs_root, static_cast<LayerType>(i));
    }

    g_initialized = true;
    
    // Start async logging worker
    g_log_worker_stop.store(false);
    g_log_worker = std::thread(LogWorkerThread);
    
    return true;
}

void ShutdownLogRecorder() {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    if (!g_initialized) {
        return;
    }
    FlushDeviceLogs();
    CleanupResources();
}

void FlushDeviceLogs() {
    // No-op: Device-side logging was removed (zero callers in production).
    // Only host-side RecordLayerLogHost() is used.
}

void ResetDeviceLogs() {
    // No-op: Device-side logging was removed (zero callers in production).
}

bool LogsInitialized() {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    return g_initialized;
}

const std::string& GetLogsRoot() {
    return g_logs_root;
}

bool RecordLayerLogHost(LayerType type,
                        int layer_index,
                        std::uint64_t global_step,
                        float primary_value,
                        float secondary_value,
                        const std::string& tag,
                        const std::string& message) {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    if (!g_initialized) {
        return false;
    }
    LayerLogEntry entry{};
    entry.type = type;
    entry.layer_index = layer_index;
    entry.global_step = global_step;
    entry.primary_value = primary_value;
    entry.secondary_value = secondary_value;
    std::snprintf(entry.tag, kMaxTagLength, "%s", tag.c_str());
    std::snprintf(entry.message, kMaxMessageLength, "%s", message.c_str());
    WriteEntryToDisk(entry);
    return true;
}

DelegateHandle RegisterModuleLogSink(const std::string& module_name,
                                     ModuleLogCallback callback) {
    if (!callback) {
        return {};
    }
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    auto delegate = EnsureModuleDelegateLocked(module_name);
    return delegate ? delegate->Add(callback) : DelegateHandle{};
}

bool UnregisterModuleLogSink(const std::string& module_name,
                             const DelegateHandle& handle) {
    if (!handle.IsValid()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    auto delegate = FindModuleDelegateLocked(module_name);
    if (!delegate) {
        return false;
    }
    const bool removed = delegate->Remove(handle);
    if (delegate->IsEmpty()) {
        g_module_loggers.erase(module_name);
    }
    return removed;
}

void EmitModuleLog(const std::string& module_name,
                   ModuleLogLevel level,
                   std::string_view message,
                   std::uint64_t global_step,
                   bool force_sync) {
    ModuleLogEvent event;
    event.module = module_name;
    event.level = level;
    event.message = std::string(message);
    event.global_step = global_step;
    
    // Check if we should emit (level filtering)
    {
        std::lock_guard<std::mutex> lock(g_module_log_mutex);
        if (!ShouldEmitLocked(module_name, level)) {
            return;
        }
    }
    
    // Force sync or async disabled -> synchronous broadcast
    if (force_sync || !g_async_logging_enabled.load()) {
        std::shared_ptr<ModuleDelegate> delegate;
        {
            std::lock_guard<std::mutex> lock(g_module_log_mutex);
            delegate = FindModuleDelegateLocked(module_name);
        }
        if (!delegate) {
            return;
        }
        delegate->Broadcast(event);
        return;
    }
    
    // Async path - queue event
    {
        std::lock_guard<std::mutex> lock(g_log_queue_mutex);
        g_log_queue.push(std::move(event));
    }
    g_log_queue_cv.notify_one();
}

ModuleLogSink::ModuleLogSink(const std::string& module_name, ModuleLogCallback callback) {
    bind(module_name, std::move(callback));
}

ModuleLogSink::ModuleLogSink(ModuleLogSink&& other) noexcept {
    *this = std::move(other);
}

ModuleLogSink& ModuleLogSink::operator=(ModuleLogSink&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    module_ = std::move(other.module_);
    handle_ = other.handle_;
    other.handle_ = {};
    other.module_.clear();
    return *this;
}

ModuleLogSink::~ModuleLogSink() {
    reset();
}

bool ModuleLogSink::bind(const std::string& module_name, ModuleLogCallback callback) {
    reset();
    if (!callback) {
        return false;
    }
    module_ = module_name;
    handle_ = RegisterModuleLogSink(module_, std::move(callback));
    if (!handle_.IsValid()) {
        module_.clear();
        return false;
    }
    return true;
}

void ModuleLogSink::reset() {
    if (!handle_.IsValid()) {
        module_.clear();
        return;
    }
    UnregisterModuleLogSink(module_, handle_);
    handle_ = {};
    module_.clear();
}

bool ApplyModuleLogOverride(const ModuleLogOverride& override_desc) {
    if (override_desc.module.empty()) {
        return false;
    }
    SetModuleLogLevel(override_desc.module, override_desc.level);
    return true;
}

bool ApplyModuleLogOverrides(const std::vector<ModuleLogOverride>& overrides) {
    bool result = true;
    for (const auto& entry : overrides) {
        result &= ApplyModuleLogOverride(entry);
    }
    return result;
}

bool RegisterModuleLogProfile(const std::string& profile_name,
                              const std::vector<ModuleLogOverride>& overrides) {
    if (profile_name.empty() || overrides.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    g_log_profiles[profile_name] = overrides;
    return true;
}

bool HasModuleLogProfile(const std::string& profile_name) {
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    return g_log_profiles.find(profile_name) != g_log_profiles.end();
}

bool ApplyModuleLogProfile(const std::string& profile_name) {
    std::vector<ModuleLogOverride> overrides;
    {
        std::lock_guard<std::mutex> lock(g_module_log_mutex);
        auto it = g_log_profiles.find(profile_name);
        if (it == g_log_profiles.end()) {
            return false;
        }
        overrides = it->second;
    }
    return ApplyModuleLogOverrides(overrides);
}

void SetDefaultModuleLogLevel(ModuleLogLevel level) {
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    g_default_module_level = level;
}

ModuleLogLevel GetDefaultModuleLogLevel() {
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    return g_default_module_level;
}

void SetModuleLogLevel(const std::string& module_name, ModuleLogLevel level) {
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    if (module_name.empty()) {
        return;
    }
    g_module_level_overrides[module_name] = level;
}

void ClearModuleLogLevel(const std::string& module_name) {
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    g_module_level_overrides.erase(module_name);
}

ModuleLogLevel GetModuleLogLevel(const std::string& module_name) {
    std::lock_guard<std::mutex> lock(g_module_log_mutex);
    if (module_name.empty()) {
        return g_default_module_level;
    }
    return ResolveModuleThresholdLocked(module_name);
}

ScopedLogRecorder::ScopedLogRecorder(ScopedLogRecorder&& other) noexcept {
    *this = std::move(other);
}

ScopedLogRecorder& ScopedLogRecorder::operator=(ScopedLogRecorder&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (active_) {
        shutdown();
    }
    active_ = other.active_;
    other.active_ = false;
    return *this;
}

ScopedLogRecorder::~ScopedLogRecorder() {
    if (active_) {
        shutdown();
    }
}

bool ScopedLogRecorder::init(const std::string& root_path, std::size_t max_device_entries) {
    if (active_) {
        return true;
    }
    active_ = InitLogRecorder(root_path, max_device_entries);
    if (active_) {
        EmitModuleInfo(ModuleId::LogRecorder,
                       std::string("Layer diagnostics enabled at ") + root_path);
    } else {
        EmitModuleWarning(ModuleId::LogRecorder,
                          std::string("Failed to initialize layer diagnostics at ") + root_path);
    }
    return active_;
}

void ScopedLogRecorder::shutdown() {
    if (!active_) {
        return;
    }
    ShutdownLogRecorder();
    EmitModuleInfo(ModuleId::LogRecorder, "Layer diagnostics shutdown");
    active_ = false;
}

void ScopedLogRecorder::flush() const {
    if (!active_) {
        return;
    }
    FlushDeviceLogs();
}



void SetModuleLogAsync(bool enabled) {
    g_async_logging_enabled.store(enabled);
}

bool IsModuleLogAsync() {
    return g_async_logging_enabled.load();
}

void FlushModuleLogQueue() {
    if (!g_async_logging_enabled.load()) {
        return;  // Nothing to flush in sync mode
    }
    
    // Wait until queue is empty
    while (true) {
        {
            std::lock_guard<std::mutex> lock(g_log_queue_mutex);
            if (g_log_queue.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void ConfigureLayerLogging(bool master_enabled,
                           bool embedding,
                           bool rms_norm,
                           bool attention,
                           bool feed_forward,
                           bool residual,
                           bool encoding,
                           bool serialization) {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    g_layer_logging_master_enabled = master_enabled;
    g_layer_enables[static_cast<int>(LayerType::kUnknown)] = false;
    g_layer_enables[static_cast<int>(LayerType::kEmbedding)] = embedding;
    g_layer_enables[static_cast<int>(LayerType::kLayerNorm)] = false; // Deprecated, always false
    g_layer_enables[static_cast<int>(LayerType::kRMSNorm)] = rms_norm;
    g_layer_enables[static_cast<int>(LayerType::kAttention)] = attention;
    g_layer_enables[static_cast<int>(LayerType::kFeedForward)] = feed_forward;
    g_layer_enables[static_cast<int>(LayerType::kResidual)] = residual;
    g_layer_enables[static_cast<int>(LayerType::kEncoding)] = encoding;
    g_layer_enables[static_cast<int>(LayerType::kSerialization)] = serialization;
}

bool IsLayerLoggingEnabled(LayerType type) {
    if (!g_layer_logging_master_enabled) {
        return false;
    }
    const int idx = static_cast<int>(type);
    if (idx < 0 || idx >= static_cast<int>(LayerType::kCount)) {
        return false;
    }
    return g_layer_enables[idx];
}

//======================================================//
//  Module Log Formatting Helpers
//======================================================//

ModuleLogCallback CreateStandardModuleLogFormatter(std::function<void(const std::string&)> log_fn) {
    if (!log_fn) {
        // Return a no-op callback if log_fn is null
        return [](const ModuleLogEvent&) {};
    }
    
    // Return a formatter that converts ModuleLogEvent to string and logs via log_fn
    return [log_fn](const ModuleLogEvent& evt) {
        const char* level = "INFO";
        switch (evt.level) {
            case ModuleLogLevel::Warning: level = "WARN"; break;
            case ModuleLogLevel::Error: level = "ERR"; break;
            default: break;
        }
        std::ostringstream msg;
        msg << "[" << evt.module << "][" << level << "] " << evt.message;
        log_fn(msg.str());
    };
}

} // namespace GRIM::Logging
