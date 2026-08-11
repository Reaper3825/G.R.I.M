#include "bootstrap_config.hpp"

#include "../settings/runtime_ai_config.hpp"
#include "error_manager.hpp"
#include "logger.hpp"
#include "nlp/grammar_parser.hpp"
#include "resources.hpp"

#include <fstream>

namespace fs = std::filesystem;

// Retained temporarily for legacy GrammarParser consumers. Phase 2 no longer
// loads or invokes grammar/NLP rules during application bootstrap.
namespace GRIM {
GrammarParser g_grammarParser;
}

namespace {
bool mergeDefaults(nlohmann::json& config,
                   const nlohmann::json& defaults,
                   int* patchedCount = nullptr) {
    bool patched = false;
    for (const auto& [key, defaultValue] : defaults.items()) {
        if (!config.contains(key) || config[key].is_null()) {
            config[key] = defaultValue;
            patched = true;
            if (patchedCount) {
                ++(*patchedCount);
            }
        } else if (defaultValue.is_object() && config[key].is_object()) {
            if (mergeDefaults(config[key], defaultValue, patchedCount)) {
                patched = true;
            }
        } else if (config[key].type() != defaultValue.type()) {
            config[key] = defaultValue;
            patched = true;
            if (patchedCount) {
                ++(*patchedCount);
            }
        }
    }
    return patched;
}
} // namespace

namespace bootstrap_config {

nlohmann::json defaultAliases() {
    return nlohmann::json::object();
}

bool loadConfig(const fs::path& path,
                const nlohmann::json& defaults,
                nlohmann::json& outConfig,
                const std::string& name,
                const std::string& errorCode) {
    if (!fs::exists(path)) {
        outConfig = defaults;
        std::ofstream(path) << outConfig.dump(2);
        LOG_PHASE(name + " created", true);
        return true;
    }

    try {
        std::ifstream input(path);
        input >> outConfig;

        int patchedCount = 0;
        if (mergeDefaults(outConfig, defaults, &patchedCount)) {
            std::ofstream(path) << outConfig.dump(2);
            LOG_PHASE(name + " patched", true);
            LOG_DEBUG("Config", name + " patched (" +
                std::to_string(patchedCount) + " keys)");
        } else {
            LOG_PHASE(name + " load", true);
        }
        return true;
    } catch (...) {
        LOG_ERROR("Config", name + " invalid; reset to defaults");
        LOG_PHASE(name + " load", false);

        if (!errorCode.empty()) {
            ErrorManager::report(errorCode);
        }

        outConfig = defaults;
        std::ofstream(path) << outConfig.dump(2);
        return false;
    }
}

void initAll() {
    try {
        Settings::loadRuntimeAiConfig();
        LOG_PHASE("AI config load", true);
    } catch (const std::exception& e) {
        LOG_ERROR("Config", std::string("ai_config load error: ") + e.what());
        LOG_PHASE("AI config load", false);
    }

    const fs::path synonymPath = fs::path(getResourcePath()) / "synonyms.json";
    if (!fs::exists(synonymPath)) {
        std::ofstream(synonymPath) << "{}\n";
        LOG_PHASE("Synonyms config created", true);
    } else {
        LOG_PHASE("Synonyms config load", true);
    }
}

} // namespace bootstrap_config
