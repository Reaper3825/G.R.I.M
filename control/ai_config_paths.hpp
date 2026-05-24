#pragma once

//======================================================//
// AI CONFIG ORGANIZATION
//
// This file defines the raw C++ snapshot that loads ai_config.json.
// The JSON→C++ mapping is:
//
// JSON Section                → C++ Owner
// ---------------------------------------
// ai_config.json              → AiConfigSnapshot::document
// training.config leaves      → HyperParameterGroupings_GPU.hpp typed owners
//
// RULE: All runtime fields in TrainingHyperparameters MUST
// be authored in ai_config.json training.config or derived in HyperParameterGroupings_GPU.hpp.
// HyperParameter_GPU.hpp may keep only formulas/static kernel capabilities,
// never runtime policy fallbacks.
//
// For compile-time constants (CUDA blocks, epsilons, etc.),
// see HyperParameter_GPU.hpp - DO NOT duplicate them here.
//======================================================//

// Include guard macro for detection by other headers
#define GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

// Strict layering: this file is the JSON reader and is allowed to be
// included by EXACTLY ONE place — HyperParameters_GPU.hpp. AiConfigSnapshot
// stays raw/flat; downstream typed config construction lives in HP_GPU.hpp.
// Including this header from anywhere else is a layering violation; HP_GPU.hpp
// is the single entry point.
#ifndef GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS
# error "ai_config_paths.hpp must only be included via HyperParameters_GPU.hpp. Include HyperParameters_GPU.hpp instead."
#endif

namespace GRIM {
namespace Config {

struct AiConfigSnapshot {
    std::filesystem::path config_path;
    nlohmann::json document;
};

inline AiConfigSnapshot loadAiConfigSnapshot();

namespace detail {

inline std::filesystem::path resolveAiConfigPath() {
    namespace fs = std::filesystem;
    fs::path defaultCandidate = fs::current_path() / "ai_config.json";
    if (fs::exists(defaultCandidate)) {
        return defaultCandidate;
    }

    fs::path searchPath = fs::current_path();
    for (int i = 0; i < 5 && searchPath.has_parent_path(); ++i) {
        searchPath = searchPath.parent_path();
        fs::path parentCandidate = searchPath / "ai_config.json";
        if (fs::exists(parentCandidate)) {
            return parentCandidate;
        }
    }

    throw std::runtime_error(
        "resolveAiConfigPath: ai_config.json not found in current directory or parent directories");
}

} // namespace detail

inline AiConfigSnapshot loadAiConfigSnapshot() {
    auto resolved_path = detail::resolveAiConfigPath();

    std::ifstream configFile(resolved_path);
    if (!configFile.is_open()) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: could not open ai_config.json at: " + resolved_path.string());
    }

    try {
        nlohmann::json document;
        configFile >> document;
        if (!document.is_object()) {
            throw std::runtime_error("ai_config.json: root document must be an object");
        }

        AiConfigSnapshot snapshot;
        snapshot.config_path = resolved_path;
        snapshot.document = document;
        return snapshot;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: failed to parse ai_config.json at: " + resolved_path.string() +
            ": " + e.what());
    }
}

} // namespace Config
} // namespace GRIM
