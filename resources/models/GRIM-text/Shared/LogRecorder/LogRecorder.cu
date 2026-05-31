#include "LogRecorder.hpp"
#include "BatchLogTape.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace GRIM;
using namespace GRIM::Logging;

namespace fs = std::filesystem;

namespace {

//======================================================//
//  Layer-file logging state
//======================================================//

std::mutex g_host_mutex;
bool g_initialized = false;
std::string g_logs_root;

std::unordered_map<std::string, std::FILE*> g_log_handles;

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
    true,   // kSerialization
    true    // kExecutionBlock
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

void WriteEntryToDisk(const LayerLogEntry& entry) {
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

} // namespace

//======================================================//
//  GRIM namespace — LayerType helpers
//======================================================//

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
        case LayerType::kExecutionBlock: return "ExecutionBlock";
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

//======================================================//
//  GRIM::Logging — Module log + layer logging
//======================================================//

namespace GRIM::Logging {

const char* ModuleIdToString(ModuleId id) {
    switch (id) {
        case ModuleId::ForwardPass: return "ForwardPass";
        case ModuleId::BackwardPass: return "BackwardPass";
        case ModuleId::Optimizer: return "Optimizer";
        case ModuleId::Scheduler: return "Scheduler";
        case ModuleId::Activations: return "Activations";
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
        case ModuleId::ExecutionBlock: return "ExecutionBlock";
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

//------------------------------------------------------
//  Layer-file logging (InitLogRecorder / RecordLayerLogHost)
//------------------------------------------------------

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
    return true;
}

void ShutdownLogRecorder() {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    if (!g_initialized) {
        return;
    }
    CloseAllFiles();
    g_initialized = false;
}

void FlushDeviceLogs() {
    // No-op: Device-side logging was removed.
}

void ResetDeviceLogs() {
    // No-op: Device-side logging was removed.
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

//------------------------------------------------------
//  EmitModuleLog — routes through global BatchLogTape
//------------------------------------------------------

void EmitModuleLog(const std::string& module_name,
                   ModuleLogLevel level,
                   std::string_view message,
                   std::uint64_t global_step,
                   bool force_sync) {
    const LogLevel log_level = moduleLogLevelToLogLevel(static_cast<int>(level));
    
    auto* tape = getGlobalTape();
    if (tape) {
        LogGroup group = LogGroup::System;
        
        if (module_name == "ForwardPass")           group = moduleIdToLogGroup(0);
        else if (module_name == "BackwardPass")     group = moduleIdToLogGroup(1);
        else if (module_name == "Optimizer")        group = moduleIdToLogGroup(2);
        else if (module_name == "Scheduler")        group = moduleIdToLogGroup(3);
        else if (module_name == "Activations")      group = moduleIdToLogGroup(4);
        else if (module_name == "Validation")       group = moduleIdToLogGroup(5);
        else if (module_name == "Checkpoint")       group = moduleIdToLogGroup(6);
        else if (module_name == "DataLoader")       group = moduleIdToLogGroup(7);
        else if (module_name == "Inference")        group = moduleIdToLogGroup(8);
        else if (module_name == "LogRecorder")      group = moduleIdToLogGroup(9);
        else if (module_name == "Training")         group = moduleIdToLogGroup(10);
        else if (module_name == "TrainingOrchestrator") group = moduleIdToLogGroup(11);
        else if (module_name == "StreamController") group = moduleIdToLogGroup(12);
        else if (module_name == "Loss")             group = moduleIdToLogGroup(13);
        else if (module_name == "Attention")        group = moduleIdToLogGroup(14);
        else if (module_name == "Custom")           group = moduleIdToLogGroup(15);
        else if (module_name == "Autograd")         group = moduleIdToLogGroup(16);
        else if (module_name == "ExecutionBlock")   group = moduleIdToLogGroup(17);
        
        if (!tape->accepts(log_level, group)) {
            return;
        }
        
        LogEntry entry{};
        entry.level = log_level;
        entry.group = group;
        entry.phase = LogPhase::LIFECYCLE;
        entry.layer_idx = -1;
        entry.global_step = static_cast<int32_t>(global_step);
        entry.batch_idx = tape->currentBatch();
        entry.setTag(module_name.c_str());
        entry.setMessageView(message);
        entry.primary = __builtin_nanf("");
        entry.secondary = __builtin_nanf("");
        tape->emitImmediate(entry);
        // Shared training_<session>.log must remain chronological relative to
        // TrainingLogger writes. Flush each module/lifecycle line immediately
        // instead of deferring it behind the next batch tape flush.
        tape->flushSinks();
    } else {
        // Pre-tape initialization: write to stderr so messages aren't lost
        const char* lvl_str = "INFO";
        if (level == ModuleLogLevel::Warning) lvl_str = "WARN";
        else if (level == ModuleLogLevel::Error) lvl_str = "ERR";
        std::fprintf(stderr, "[pre-tape][%s][%s] %.*s\n",
                     module_name.c_str(), lvl_str,
                     static_cast<int>(message.size()), message.data());
    }
}

//------------------------------------------------------
//  ConfigureLayerLogging / IsLayerLoggingEnabled
//------------------------------------------------------

void ConfigureLayerLogging(bool master_enabled,
                           bool embedding,
                           bool rms_norm,
                           bool attention,
                           bool feed_forward,
                           bool residual,
                           bool encoding,
                           bool serialization,
                           bool execution_block) {
    std::lock_guard<std::mutex> lock(g_host_mutex);
    g_layer_logging_master_enabled = master_enabled;
    g_layer_enables[static_cast<int>(LayerType::kUnknown)] = false;
    g_layer_enables[static_cast<int>(LayerType::kEmbedding)] = embedding;
    g_layer_enables[static_cast<int>(LayerType::kLayerNorm)] = false;
    g_layer_enables[static_cast<int>(LayerType::kRMSNorm)] = rms_norm;
    g_layer_enables[static_cast<int>(LayerType::kAttention)] = attention;
    g_layer_enables[static_cast<int>(LayerType::kFeedForward)] = feed_forward;
    g_layer_enables[static_cast<int>(LayerType::kResidual)] = residual;
    g_layer_enables[static_cast<int>(LayerType::kEncoding)] = encoding;
    g_layer_enables[static_cast<int>(LayerType::kSerialization)] = serialization;
    g_layer_enables[static_cast<int>(LayerType::kExecutionBlock)] = execution_block;
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

} // namespace GRIM::Logging
