#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>

// Centralized config + memory bootstrap for GRIM
namespace bootstrap_config {

    // Run all config/memory/error/synonym bootstrap
    void initAll();

    // Generic loader → ensures defaults, patches missing keys, saves back
    bool loadConfig(const std::filesystem::path& path,
                    const nlohmann::json& defaults,
                    nlohmann::json& outConfig,
                    const std::string& name,
                    const std::string& errorCode = "");

    // Canonical defaults
    nlohmann::json defaultAI();
    nlohmann::json defaultErrors();
    nlohmann::json defaultMemory();
    nlohmann::json defaultAliases();
}
