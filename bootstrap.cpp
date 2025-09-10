#include "bootstrap.hpp"
#include "resources.hpp"       // declares getResourcePath()
#include "console_history.hpp"
#include "ai.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>             // for system()
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// --- Utility: create file if missing ---
static void ensureFileExists(const fs::path& path, const std::string& contents, const std::string& tag) {
    if (fs::exists(path)) {
        std::cout << "[OK] " << tag << " found (" << fs::file_size(path) << " bytes)\n";
    } else {
        std::ofstream out(path);
        if (out) {
            out << contents;
            std::cout << "[NEW] " << tag << " created at " << path << "\n";
        } else {
            std::cerr << "[ERROR] Failed to create " << tag << " at " << path << "\n";
        }
    }
}

// --- Utility: patch ai_config.json with missing defaults ---
static void validateAIConfig(const fs::path& path) {
    nlohmann::json defaults = {
        {"backend", "auto"},
        {"ollama_url", "http://127.0.0.1:11434"},
        {"localai_url", "http://127.0.0.1:8080/v1"}
    };

    nlohmann::json cfg;
    bool changed = false;

    if (fs::exists(path)) {
        try {
            std::ifstream f(path);
            f >> cfg;
        } catch (...) {
            std::cerr << "[WARN] ai_config.json invalid, resetting to defaults.\n";
            cfg = defaults;
            changed = true;
        }
    } else {
        cfg = defaults;
        changed = true;
    }

    for (auto& [k, v] : defaults.items()) {
        if (!cfg.contains(k)) {
            cfg[k] = v;
            changed = true;
            std::cout << "[PATCH] ai_config.json missing key \"" << k << "\" → added default.\n";
        }
    }

    if (changed) {
        std::ofstream out(path);
        if (out) {
            out << cfg.dump(2);
            std::cout << "[OK] ai_config.json updated with defaults.\n";
        }
    } else {
        std::cout << "[OK] ai_config.json valid.\n";
    }
}

// --- Whisper model check ---
static void ensureWhisperModel(const fs::path& modelPath) {
    if (fs::exists(modelPath)) {
        std::cout << "[OK] Whisper model found: " << modelPath << " ("
                  << fs::file_size(modelPath) / (1024*1024) << " MB)\n";
        return;
    }

    std::cout << "[NEW] Whisper model missing, attempting download...\n";
    fs::create_directories(modelPath.parent_path());

    std::string url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";
    std::string cmd = "curl -L " + url + " -o " + modelPath.string();

    int result = system(cmd.c_str());
    if (result == 0 && fs::exists(modelPath)) {
        std::cout << "[OK] Whisper model downloaded successfully.\n";
    } else {
        std::cerr << "[ERROR] Failed to download Whisper model.\n"
                  << "        Please download manually from:\n"
                  << "        " << url << "\n"
                  << "        and place it at " << modelPath << "\n";
    }
}

// --- Main bootstrap sequence ---
void runBootstrapChecks(int argc, char** argv) {
    std::cout << "---- GRIM Bootstrap Check ----\n";

    // Memory
    ensureFileExists("memory.json", "{}\n", "memory.json");

    // AI config
    validateAIConfig("ai_config.json");

    // Resources directory
    fs::path resDir = getResourcePath();
    if (!fs::exists(resDir)) {
        std::cout << "[INFO] Creating resources directory at " << resDir << "\n";
        fs::create_directories(resDir);
    }

    // NLP rules
    ensureFileExists(resDir / "nlp_rules.json",
        "[\n"
        "  {\"intent\":\"open_app\",\"pattern\":\"^\\\\s*(open|launch|start)\\\\s+([\\\\w\\\\.\\\\-]+)\\\\s*$\",\"slot_names\":[\"verb\",\"app\"],\"score_boost\":0.3,\"case_insensitive\":true},\n"
        "  {\"intent\":\"search_web\",\"pattern\":\"^(google|search|look up)\\\\s+(.+)$\",\"slot_names\":[\"verb\",\"query\"],\"score_boost\":0.2,\"case_insensitive\":true},\n"
        "  {\"intent\":\"set_timer\",\"pattern\":\"^(set\\\\s+)?timer\\\\s+for\\\\s+(\\\\d+)\\\\s*(seconds|s|minutes|min|m|hours|h)\\\\b\",\"slot_names\":[\"_opt\",\"value\",\"unit\"],\"score_boost\":0.25,\"case_insensitive\":true}\n"
        "]\n",
        "nlp_rules.json"
    );

    // Synonyms
    ensureFileExists(resDir / "synonyms.json", "{}\n", "synonyms.json");

    // Aliases
    ensureFileExists(resDir / "app_aliases.json", "{}\n", "app_aliases.json");

    // Fonts
    std::string fontPath = findAnyFontInResources(argc, argv, nullptr);
    if (!fontPath.empty()) {
        std::cout << "[OK] Font available: " << fontPath << "\n";
    } else {
        std::cout << "[ERROR] No usable font found! UI may fail.\n";
    }

    // Whisper model
    fs::path whisperModel = fs::path("whisper.cpp/models/ggml-small.bin");
    ensureWhisperModel(whisperModel);

    std::cout << "-------------------------------\n";
}
