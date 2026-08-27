/**
 * BatchLogTape.cu — Compilation unit for BatchLogTape
 *
 * Most BatchLogTape methods are inline in BatchLogTape.hpp. This compilation
 * unit owns the global registration lifetime and the config parsing helper
 * that depends on <string> operations.
 */

#include "BatchLogTape.hpp"
#include "LogTypes.hpp"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace GRIM::Logging {

//======================================================//
//  Config parsing helpers
//======================================================//

TapeConfig parseTapeConfig(
    const char* default_level_str,
    bool equation_csv_enabled,
    bool stderr_enabled,
    size_t initial_capacity)
{
    TapeConfig config;
    config.default_level = logLevelFromString(default_level_str);
    config.equation_csv_enabled = equation_csv_enabled;
    config.stderr_enabled = stderr_enabled;
    if (initial_capacity > 0) {
        config.initial_capacity = initial_capacity;
    }
    return config;
}

void applyGroupOverride(TapeConfig& config, const char* group_str, const char* level_str) {
    if (!group_str || !level_str) {
        throw std::runtime_error("applyGroupOverride: group_str and level_str must not be NULL");
    }
    LogGroup group = logGroupFromString(group_str);
    LogLevel level = logLevelFromString(level_str);
    config.group_levels[static_cast<size_t>(group)] = level;
}

//======================================================//
//  Diagnostic dump (for debugging the logging system itself)
//======================================================//

std::string dumpTapeConfig(const TapeConfig& config) {
    std::ostringstream ss;
    ss << "BatchLogTape config:\n";
    ss << "  default_level: " << logLevelToString(config.default_level) << "\n";
    ss << "  capacity: " << config.initial_capacity << "\n";
    ss << "  equation_csv: " << (config.equation_csv_enabled ? "on" : "off") << "\n";
    ss << "  stderr: " << (config.stderr_enabled ? "on" : "off") << "\n";

    bool has_overrides = false;
    for (size_t i = 0; i < static_cast<size_t>(LogGroup::COUNT); ++i) {
        if (config.group_levels[i] != LogLevel::COUNT) {
            if (!has_overrides) {
                ss << "  group overrides:\n";
                has_overrides = true;
            }
            ss << "    " << logGroupToString(static_cast<LogGroup>(i))
               << " = " << logLevelToString(config.group_levels[i]) << "\n";
        }
    }
    if (!has_overrides) {
        ss << "  group overrides: (none)\n";
    }

    return ss.str();
}

//======================================================//
//  Global tape pointer
//======================================================//

static BatchLogTape* g_global_tape = nullptr;

BatchLogTape::~BatchLogTape() {
    if (g_global_tape == this) {
        g_global_tape = nullptr;
    }
}

void setGlobalTape(BatchLogTape* tape) {
    g_global_tape = tape;
}

BatchLogTape* getGlobalTape() {
    return g_global_tape;
}

} // namespace GRIM::Logging
