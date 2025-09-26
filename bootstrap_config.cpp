#include "bootstrap_config.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "aliases.hpp"
#include "nlp.hpp"
#include "console_history.hpp"
#include "ai.hpp"
#include "logger.hpp"

namespace fs = std::filesystem;

// ----------------- helpers -----------------
static bool mergeDefaults(nlohmann::json& cfg,
                          const nlohmann::json& defs,
                          const std::string& prefix = "",
                          int* patchedCount = nullptr) {
    bool patched = false;
    for (auto& [key, defVal] : defs.items()) {
        if (!cfg.contains(key) || cfg[key].is_null()) {
            cfg[key] = defVal;
            patched = true;
            if (patchedCount) (*patchedCount)++;
        } else if (defVal.is_object() && cfg[key].is_object()) {
            if (mergeDefaults(cfg[key], defVal,
                              prefix.empty() ? key : prefix + "." + key,
                              patchedCount))
                patched = true;
        } else if (cfg[key].type() != defVal.type()) {
            cfg[key] = defVal;
            patched = true;
            if (patchedCount) (*patchedCount)++;
        }
    }
    return patched;
}

// ----------------- defaults -----------------
namespace bootstrap_config {

nlohmann::json defaultAI() {
    return {
        {"backend", "auto"},
        {"ollama_url", "http://127.0.0.1:11434"},
        {"localai_url", "http://127.0.0.1:8080/v1"},
        {"default_model", "mistral"},

        {"whisper_language", "en"},
        {"whisper_max_tokens", 32},
        {"silence_threshold", 0.02},
        {"silence_timeout_ms", 4000},

        {"voice", {
            {"mode", "local"},
            {"engine", "coqui"},
            {"local_engine", "en_US-amy-medium.onnx"},
            {"speaker", "p225"},
            {"speed", 1.0},
            {"rules", {
                {"startup", "sapi"},
                {"reminder", "coqui"},
                {"summary", "coqui"},
                {"banter", "coqui"}
            }},
            {"input_device_index", -1},
            {"coqui", {
                {"model", "tts_models/en/vctk/vits"},
                {"speaker", "p225"}
            }},
            {"sapi", {
                {"voice", "en_US-amy-medium.onnx"}
            }}
        }},

        {"api_keys", {
            {"openai", ""},
            {"elevenlabs", ""},
            {"azure", ""}
        }},

        {"whisper", {
            {"sampling_strategy", "beam"},
            {"temperature", 0.2},
            {"min_speech_ms", 500},
            {"min_silence_ms", 1200}
        }}
    };
}

nlohmann::json defaultErrors() {
    return {
        {"ERR_FS_MISSING_DIR", {
            {"user", "[FS] Usage: cd/mkdir <directory>"},
            {"debug", "Filesystem command called without directory argument."}
        }},
        {"ERR_FS_DIR_NOT_FOUND", {
            {"user", "[FS] Directory does not exist."},
            {"debug", "Target directory not found in cmdChangeDir."}
        }},
        {"ERR_APP_NO_ARGUMENT", {
            {"user", "[App] Usage: open <application>"},
            {"debug", "Application command called without argument."}
        }},
        {"ERR_AI_CONFIG_INVALID", {
            {"user", "[AI] Config file invalid → reset to defaults."},
            {"debug", "ai_config.json failed parsing or validation."}
        }},
        {"ERR_ALIAS_NOT_FOUND", {
            {"user", "[Alias] Application not found."},
            {"debug", "Alias lookup failed in user, auto, and fallback."}
        }}
    };
}

nlohmann::json defaultMemory() {
    return nlohmann::json::object();
}

nlohmann::json defaultAliases() {
    return nlohmann::json::object();
}

// ----------------- loader -----------------
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
        std::ifstream f(path);
        f >> outConfig;

        int patchedCount = 0;
        if (mergeDefaults(outConfig, defaults, "", &patchedCount)) {
            std::ofstream(path) << outConfig.dump(2);
            LOG_PHASE(name + " patched", true);
            LOG_DEBUG("Config", name + " patched (" + std::to_string(patchedCount) + " keys)");
        } else {
            LOG_PHASE(name + " load", true);
        }
        return true;
    } catch (...) {
        LOG_ERROR("Config", name + " invalid → reset to defaults");
        LOG_PHASE(name + " load", false);

        if (!errorCode.empty())
            ErrorManager::report(errorCode);

        outConfig = defaults;
        std::ofstream(path) << outConfig.dump(2);
        return false;
    }
}

// ----------------- entry -----------------
void initAll() {
    // memory.json
    nlohmann::json memoryCfg;
    loadConfig("memory.json", defaultMemory(), memoryCfg, "Memory config", "");

    // ai_config.json
    fs::path cfgPath = fs::current_path() / AI_CONFIG_FILE;
    loadConfig(cfgPath, defaultAI(), aiConfig, "AI config", "ERR_AI_CONFIG_INVALID");

    // errors.json
    fs::path errPath = fs::path(getResourcePath()) / "errors.json";
    nlohmann::json errorsCfg;
    loadConfig(errPath, defaultErrors(), errorsCfg, "Errors config", "");

    // NLP rules
    fs::path nlpPath = fs::path(getResourcePath()) / "nlp_rules.json";
    if (!fs::exists(nlpPath)) {
        std::ofstream(nlpPath) << "[]\n";
        LOG_PHASE("NLP rules created", true);
    }
    std::string err;
    if (!g_nlp.load_rules(nlpPath.string(), &err)) {
        LOG_ERROR("Config", "Failed to load NLP rules: " + err);
        LOG_PHASE("NLP rules load", false);
    } else {
        LOG_PHASE("NLP rules load", true);
    }

    // synonyms.json
    fs::path synPath = fs::path(getResourcePath()) / "synonyms.json";
    if (!fs::exists(synPath)) {
        std::ofstream(synPath) << "{}\n";
        LOG_PHASE("Synonyms config created", true);
    } else {
        LOG_PHASE("Synonyms config load", true);
    }
}

} // namespace bootstrap_config
