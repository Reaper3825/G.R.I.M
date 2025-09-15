#include "bootstrap.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "ai.hpp"              // 🔹 Needed for g_silenceThreshold / g_silenceTimeoutMs
#include "system_detect.hpp"
#include "nlp.hpp"
#include "error_manager.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// Global system info
SystemInfo g_systemInfo;

// ----------------------------
// Simple GRIM-style logger
// ----------------------------
static void grimLog(const std::string& msg) {
    std::cout << msg << std::endl;
}

// -------------------------------------------------------------
// Default config JSON (uses AI globals for initial values)
// -------------------------------------------------------------
static nlohmann::json defaultConfig() {
    return {
        {"backend", "auto"},
        {"ollama_url", "http://127.0.0.1:11434"},
        {"localai_url", "http://127.0.0.1:8080/v1"},
        {"default_model", "mistral"},
        {"whisper_language", "en"},
        {"whisper_max_tokens", 32},
        {"silence_threshold", g_silenceThreshold},
        {"silence_timeout_ms", g_silenceTimeoutMs},
        {"voice", {
            {"mode", "hybrid"},
            {"local_engine", "en_US-amy-medium.onnx"},
            {"cloud_engine", "openai"},
            {"rules", {
                {"startup", "local"},
                {"reminder", "local"},
                {"summary", "cloud"},
                {"banter", "cloud"}
            }},
            {"silence_threshold", 0.02},
            {"silence_timeout_ms", 4000},
            {"input_device_index", -1},
            {"tts_url", "http://127.0.0.1:8080/tts"}
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

// -------------------------------------------------------------
// Validate and patch ai_config.json
// -------------------------------------------------------------
static bool validateAndPatch(nlohmann::json& cfg) {
    bool patched = false;
    nlohmann::json defs = defaultConfig();

    for (auto& [key, val] : defs.items()) {
        if (!cfg.contains(key) || cfg[key].is_null() || cfg[key].type() != val.type()) {
            cfg[key] = val;
            grimLog("[Config] " + std::string(AI_CONFIG_FILE) +
                    " patched (missing key: " + key + ")");
            patched = true;
        } else if (val.is_object()) {
            for (auto& [subKey, subVal] : val.items()) {
                if (!cfg[key].contains(subKey) || cfg[key][subKey].is_null()) {
                    cfg[key][subKey] = subVal;
                    grimLog("[Config] " + std::string(AI_CONFIG_FILE) +
                            " patched (missing key: " + key + "." + subKey + ")");
                    patched = true;
                }
            }
        }
    }

    return patched;
}

// -------------------------------------------------------------
// Bootstrap main entry
// -------------------------------------------------------------
void runBootstrapChecks(int argc, char** argv) {
    grimLog("[GRIM] Startup begin");

    // memory.json
    fs::path memPath = "memory.json";
    if (!fs::exists(memPath)) {
        std::ofstream(memPath) << "{}\n";
        grimLog("[Config] memory.json created");
    } else {
        grimLog("[Config] memory.json found (" +
                std::to_string(fs::file_size(memPath)) + " bytes)");
    }

    // ai_config.json
    fs::path cfgPath = fs::current_path() / AI_CONFIG_FILE;
    nlohmann::json cfg;

    if (!fs::exists(cfgPath)) {
        cfg = defaultConfig();
        std::ofstream(cfgPath) << cfg.dump(2);
        grimLog("[Config] " + std::string(AI_CONFIG_FILE) + " created");
    } else {
        try {
            std::ifstream f(cfgPath);
            f >> cfg;
            std::cout << "[DEBUG][Bootstrap] Parsed JSON from "
                      << cfgPath << ":\n" << cfg.dump(2) << "\n";

            if (validateAndPatch(cfg)) {
                std::ofstream(cfgPath) << cfg.dump(2);
                grimLog("[Config] " + std::string(AI_CONFIG_FILE) +
                        " patched and saved");
            } else {
                grimLog("[Config] " + std::string(AI_CONFIG_FILE) + " loaded");
            }
        } catch (const std::exception& e) {
            grimLog("[Config] " + std::string(AI_CONFIG_FILE) +
                    " invalid → reset to defaults");
            ErrorManager::report("ERR_AI_CONFIG_INVALID");
            cfg = defaultConfig();
            std::ofstream(cfgPath) << cfg.dump(2);
        }
    }

    aiConfig = cfg;

    // Debug: final config
    std::cout << "[DEBUG][Bootstrap] Final aiConfig:\n"
              << aiConfig.dump(2) << "\n";

    // Resources
    fs::path resDir = getResourcePath();
    if (!fs::exists(resDir)) {
        grimLog("[Config] creating resources directory at " + resDir.string());
        fs::create_directories(resDir);
    }

    // NLP rules
    fs::path nlpPath = resDir / "nlp_rules.json";
    if (!fs::exists(nlpPath)) {
        std::ofstream(nlpPath) << "[]\n";
        grimLog("[Config] nlp_rules.json created");
    }

    std::string err;
    if (!g_nlp.load_rules(nlpPath.string(), &err)) {
        grimLog("[Config] Failed to load NLP rules: " + err);
    } else {
        grimLog("[Config] NLP rules loaded");
    }

    // Synonyms and aliases
    if (!fs::exists(resDir / "synonyms.json")) {
        std::ofstream(resDir / "synonyms.json") << "{}\n";
        grimLog("[Config] synonyms.json created");
    }
    if (!fs::exists(resDir / "app_aliases.json")) {
        std::ofstream(resDir / "app_aliases.json") << "{}\n";
        grimLog("[Config] app_aliases.json created");
    }

    // Fonts
    std::string fontPath = findAnyFontInResources(argc, argv, &history);
    if (!fontPath.empty())
        grimLog("[Config] Font found: " + fontPath);
    else
        grimLog("[Config] No system font found, UI may render incorrectly");

    grimLog("[GRIM] ---- Bootstrap Complete ----");
}
