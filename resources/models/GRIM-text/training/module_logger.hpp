#pragma once

#include "../Shared/LogRecorder/LogRecorder.hpp"
#include <string_view>
#include <cstdint>

// Utility helper to route module logs through the centralized TrainingLogger sink.
template <GRIM::Logging::ModuleId Module>
struct ModuleLogger {
    static void info(std::string_view message, std::uint64_t global_step = 0) {
        GRIM::Logging::EmitModuleInfo(Module, message, global_step);
    }

    static void warn(std::string_view message, std::uint64_t global_step = 0) {
        GRIM::Logging::EmitModuleWarning(Module, message, global_step);
    }

    static void error(std::string_view message, std::uint64_t global_step = 0) {
        GRIM::Logging::EmitModuleError(Module, message, global_step);
    }
};
