#pragma once
//======================================================//
//  LogRecorder.hpp — Module-level logging API
//======================================================//
//
//  EmitModuleLog and convenience wrappers now route through
//  the unified BatchLogTape (single logging point).
//  The old delegate/callback/async-queue infrastructure
//  has been deleted.
//
//  Layer logging (RecordLayerLogHost) still writes to
//  per-layer-type disk files — retained for the single
//  caller in Phase2_TrainingLoop.cu.
//
//  Author: Austin Wadkins
//======================================================//

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "../../Layers/grim_layer_gpu.hpp"

namespace GRIM {

std::string layerTypeToString(LayerType type);
bool CreateLayerFolder(const std::string& absolutePath, LayerType type);

} // namespace GRIM

namespace GRIM::Logging {

//======================================================//
//  Module identity enums (used by 366+ call sites)
//======================================================//

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

const char* ModuleIdToString(ModuleId id);
const char* ModuleLogLevelToString(ModuleLogLevel level);

//======================================================//
//  EmitModuleLog — routes through global BatchLogTape
//======================================================//
//
//  Before the tape is constructed (early Phase1), messages
//  go to stderr with a [pre-tape] prefix. After setGlobalTape(),
//  they are recorded onto the tape and flow through all sinks.
//

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

//======================================================//
//  Layer-file logging (per-type disk files)
//======================================================//

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

bool RecordLayerLogHost(LayerType type,
                        int layer_index,
                        std::uint64_t global_step,
                        float primary_value,
                        float secondary_value,
                        const std::string& tag,
                        const std::string& message);

void ConfigureLayerLogging(bool master_enabled,
                           bool embedding,
                           bool rms_norm,
                           bool attention,
                           bool feed_forward,
                           bool residual,
                           bool encoding,
                           bool serialization,
                           bool execution_block);

bool IsLayerLoggingEnabled(LayerType type);

} // namespace GRIM::Logging
