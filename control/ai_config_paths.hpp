#pragma once

//======================================================//
// AI CONFIG ORGANIZATION
//
// This file defines the startup snapshot that loads ai_config.json and the
// selected immutable model configuration.
// The JSON→C++ mapping is:
//
// JSON Section                → C++ Owner
// ---------------------------------------
// selected model.grimcfg      -> AiConfigSnapshot::model_config
// ai_config.json              → AiConfigSnapshot::document
// training.config leaves      → HyperParameters_GPU.hpp direct document compute/validate boundary
//
// RULE: Model-semantic fields come from the validated compiled artifact.
// Training/runtime fields remain authored in ai_config.json training.config.
//
// For compile-time constants (CUDA blocks, epsilons, etc.),
// see HyperParameters_GPU.hpp - DO NOT duplicate them here.
//======================================================//

// Include guard macro for detection by other headers
#define GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED

#include <string>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <nlohmann/json.hpp>

#include "../resources/models/GRIM-text/Shared/ModelConfig/CompiledModelConfig.hpp"

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
    nlohmann::json document;
    std::optional<CompiledModelConfigSnapshot> model_config;
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

    nlohmann::json document;
    try {
        configFile >> document;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: failed to parse ai_config.json at: " + resolved_path.string() +
            ": " + e.what());
    }
    if (!document.is_object()) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: ai_config.json root document must be an object at: " +
            resolved_path.string());
    }

    AiConfigSnapshot snapshot;
    snapshot.document = std::move(document);
    try {
        const auto model_config_path =
            resolveCompiledModelConfigPath(snapshot.document, resolved_path);
        if (model_config_path) {
            snapshot.model_config = loadCompiledModelConfig(*model_config_path);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "loadAiConfigSnapshot: failed to load selected model.grimcfg: " +
            std::string(e.what()));
    }
    return snapshot;
}

} // namespace Config
} // namespace GRIM
