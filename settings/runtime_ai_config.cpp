#include "runtime_ai_config.hpp"
#include "../resources.hpp"
#include "../logger.hpp"
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

extern nlohmann::json aiConfig;

namespace Settings {

std::filesystem::path resolveAiConfigPath() {
    fs::path grimRoot = fs::path(getGrimRootDir());
    fs::path configPath = grimRoot / AI_CONFIG_FILE;
    if (fs::exists(configPath)) {
        return configPath;
    }
    throw std::runtime_error(
        "ai_config.json not found at " + configPath.string() +
        " (GRIM root: " + grimRoot.string() + ")");
}

std::filesystem::path resolveAiConfigLocalPath() {
    fs::path grimRoot = fs::path(getGrimRootDir());
    return grimRoot / AI_CONFIG_LOCAL_FILE;
}

void deepMerge(nlohmann::json& base, const nlohmann::json& overlay) {
    if (!overlay.is_object()) return;
    for (auto& [key, val] : overlay.items()) {
        if (val.is_object() && base.contains(key) && base[key].is_object()) {
            deepMerge(base[key], val);
        } else {
            base[key] = val;
        }
    }
}

nlohmann::json loadRuntimeAiConfig() {
    fs::path configPath = resolveAiConfigPath();

    nlohmann::json doc;
    {
        std::ifstream f(configPath);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open " + configPath.string());
        }
        f >> doc;
    }
    LOG_DEBUG("Settings", "Loaded " + configPath.string());

    // Merge local overlay (secrets, machine-specific overrides)
    fs::path localPath = resolveAiConfigLocalPath();
    if (fs::exists(localPath)) {
        std::ifstream fl(localPath);
        nlohmann::json local;
        fl >> local;
        deepMerge(doc, local);
        LOG_DEBUG("Settings", "Merged " + localPath.string());
    }

    aiConfig = doc;
    return doc;
}

nlohmann::json saveRuntimeAiConfig(const nlohmann::json& pending) {
    // Reload the on-disk document to avoid losing keys the UI doesn't expose
    fs::path configPath = resolveAiConfigPath();

    nlohmann::json doc;
    {
        std::ifstream f(configPath);
        if (f.is_open()) {
            f >> doc;
        }
    }

    // Deep-merge pending edits into full document
    deepMerge(doc, pending);

    // Write back
    {
        std::ofstream f(configPath);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot write " + configPath.string());
        }
        f << doc.dump(4);
    }
    LOG_DEBUG("Settings", "Saved " + configPath.string());

    // Re-apply the local overlay so the in-memory copy matches what loadRuntimeAiConfig would return
    fs::path localPath = resolveAiConfigLocalPath();
    if (fs::exists(localPath)) {
        std::ifstream fl(localPath);
        nlohmann::json local;
        fl >> local;
        deepMerge(doc, local);
    }

    aiConfig = doc;
    return doc;
}

} // namespace Settings
