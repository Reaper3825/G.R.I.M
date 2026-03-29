#include "settings_apply.hpp"
#include "../resources.hpp"
#include "../logger.hpp"
#include "../voice/voice_speak.hpp"
#include "../perception/perception_context.hpp"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace Settings {

void applyRuntimeSideEffects(const nlohmann::json& config,
                             const std::unordered_map<std::string, std::string>& fontMap,
                             SetFontFn setFont,
                             SetBlurFn setBlur) {
    // Speaker
    if (config.contains("voice") && config["voice"].is_object() &&
        config["voice"].contains("speaker")) {
        std::string speaker = config["voice"]["speaker"].get<std::string>();
        Voice::setSpeaker(speaker);
        LOG_DEBUG("Settings", "Speaker set to: " + speaker);
    }

    // Vision AI
    if (config.contains("vision") && config["vision"].is_object() &&
        config["vision"].contains("enabled")) {
        bool enabled = config["vision"]["enabled"].get<bool>();
        if (GRIM::Perception::g_contextManager) {
            GRIM::Perception::g_contextManager->setFeatureEnabled("vision_ai", enabled);
            LOG_DEBUG("Settings", "Vision AI " + std::string(enabled ? "enabled" : "disabled"));
        }
    }

    // Font
    if (config.contains("ui") && config["ui"].is_object() &&
        config["ui"].contains("font_name") && setFont) {
        std::string fontName = config["ui"]["font_name"].get<std::string>();
        int fontSize = config["ui"].value("font_size", 16);
        auto it = fontMap.find(fontName);
        if (it != fontMap.end()) {
            setFont(it->second, fontSize);
            LOG_DEBUG("Settings", "Font set to: " + fontName + " size " + std::to_string(fontSize));
        } else {
            LOG_ERROR("Settings", "Font file not found for: " + fontName);
        }
    }

    // Blur
    if (config.contains("blur") && config["blur"].is_object() && setBlur) {
        bool enabled = config["blur"].value("enabled", true);
        float opacity = config["blur"].value("opacity", 0.99f);
        int intensity = config["blur"].value("intensity", 2);
        setBlur(enabled, opacity, intensity);
        LOG_DEBUG("Settings", "Blur updated: enabled=" + std::string(enabled ? "true" : "false"));
    }
}

std::vector<std::string> scanSpeakerEmbeddings() {
    std::vector<std::string> embeddings;
    embeddings.push_back("default");

    fs::path embeddingDir = fs::path(getResourcePath()) / "voices" / "embeddings";
    if (!fs::exists(embeddingDir)) {
        return embeddings;
    }

    for (const auto& entry : fs::directory_iterator(embeddingDir)) {
        if (entry.path().extension() == ".npz") {
            std::string name = entry.path().stem().string();
            if (name != "default") {
                embeddings.push_back(name);
            }
        }
    }

    LOG_DEBUG("Settings", "Found " + std::to_string(embeddings.size()) + " speaker embeddings");
    return embeddings;
}

std::vector<std::string> scanFonts(std::unordered_map<std::string, std::string>& outFontMap) {
    outFontMap.clear();
    std::vector<std::string> fonts;

    auto scanDir = [&](const fs::path& dir) {
        if (!fs::exists(dir)) return;
        for (const auto& entry : fs::recursive_directory_iterator(
                 dir, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            if (ext != ".ttf" && ext != ".otf") continue;
            std::string name = entry.path().stem().string();
            if (outFontMap.count(name)) continue;
            outFontMap[name] = entry.path().string();
            fonts.push_back(name);
        }
    };

    scanDir(fs::path(getResourcePath()));
    scanDir(fs::path(getGrimRootDir()) / "resources" / "fonts");

    std::sort(fonts.begin(), fonts.end());
    LOG_DEBUG("Settings", "Found " + std::to_string(fonts.size()) + " TTF/OTF fonts");
    return fonts;
}

} // namespace Settings
