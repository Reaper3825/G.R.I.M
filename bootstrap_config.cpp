#include "bootstrap_config.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "aliases.hpp"
#include "nlp.hpp"
#include "console_history.hpp"
#include "ai.hpp"

namespace fs = std::filesystem;

// Forward declare grimLog
extern void grimLog(const std::string& msg);

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
            if (mergeDefaults(cfg[key], defVal, prefix.empty() ? key : prefix + "." + key, patchedCount))
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

        // 🔹 top-level silence detection keys (for backward compat)
        {"whisper_language", "en"},
        {"whisper_max_tokens", 32},
        {"silence_threshold", 0.02},
        {"silence_timeout_ms", 4000},

        {"voice", {
            {"mode", "local"},  
            {"engine", "coqui"},       
            // local_engine kept for SAPI/Piper fallback
            {"local_engine", "en_US-amy-medium.onnx"},

            // 🔹 Default speaker (valid for VCTK)
            {"speaker", "p225"},                      

            // 🔹 Default speed multiplier
            {"speed", 1.0},                          

            // 🔹 Per-category routing rules
            {"rules", {
                {"startup", "sapi"},
                {"reminder", "coqui"},
                {"summary", "coqui"},
                {"banter", "coqui"}
            }},

            // 🔹 Device index for input (mic)
            {"input_device_index", -1}
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
    return nlohmann::json::object(); // {}
}

nlohmann::json defaultAliases() {
    return nlohmann::json::object(); // {}
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
        grimLog("[Config] " + name + " created");
        return true;
    }

    try {
        std::ifstream f(path);
        f >> outConfig;

        int patchedCount = 0;
        if (mergeDefaults(outConfig, defaults, "", &patchedCount)) {
            std::ofstream(path) << outConfig.dump(2);
            grimLog("[Config] " + name + " patched (" +
                    std::to_string(patchedCount) + " keys) and saved");
        } else {
            grimLog("[Config] " + name + " loaded");
        }
        return true;
    } catch (...) {
        grimLog("[Config] " + name + " invalid → reset to defaults");
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
    loadConfig("memory.json", defaultMemory(), memoryCfg, "memory.json");

    // ai_config.json
    fs::path cfgPath = fs::current_path() / AI_CONFIG_FILE;
    loadConfig(cfgPath, defaultAI(), aiConfig, AI_CONFIG_FILE, "ERR_AI_CONFIG_INVALID");

    // errors.json
    fs::path errPath = fs::path(getResourcePath()) / "errors.json";
    nlohmann::json errorsCfg;
    loadConfig(errPath, defaultErrors(), errorsCfg, "errors.json");

    // NLP rules
    fs::path nlpPath = fs::path(getResourcePath()) / "nlp_rules.json";
    if (!fs::exists(nlpPath)) {
        std::ofstream(nlpPath) << "[]\n";
        grimLog("[Config] nlp_rules.json created");
    }
    std::string err;
    if (!g_nlp.load_rules(nlpPath.string(), &err))
        grimLog("[Config] Failed to load NLP rules: " + err);
    else
        grimLog("[Config] NLP rules loaded");

    // synonyms.json
    fs::path synPath = fs::path(getResourcePath()) / "synonyms.json";
    if (!fs::exists(synPath)) {
        std::ofstream(synPath) << "{}\n";
        grimLog("[Config] synonyms.json created");
    }
}

} // namespace bootstrap_config
