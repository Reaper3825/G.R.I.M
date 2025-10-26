#include "ui_settings_menu.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include <fstream>

UISettingsMenu::UISettingsMenu()
    : UIPanel("Settings", true)
{
    position = { 200, 200 };
    size = { 440, 400 };
    setVisible(false);
    setBackground(0xC0202020);
    loadConfig();
}

void UISettingsMenu::loadConfig() {
    try {
        std::ifstream f("ai_config.json");
        if (f.is_open()) {
            f >> config;
        } else {
            config = {
                {"backend", "mistral"},
                {"voice", "coqui"},
                {"model", "ggml-base.en.bin"}
            };
        }
        LOG_DEBUG("UISettingsMenu", "Config loaded successfully");
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to load ai_config.json");
    }
    refreshButtons();
}

void UISettingsMenu::saveConfig() {
    try {
        std::ofstream f("ai_config.json");
        f << config.dump(4);
        LOG_DEBUG("UISettingsMenu", "Saved ai_config.json");
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to save ai_config.json");
    }
}

void UISettingsMenu::refreshButtons() {
    buttons.clear();

    auto getString = [&](const std::string& key, const std::string& fallback) -> std::string {
        try {
            if (config.contains(key) && config[key].is_string())
                return config[key].get<std::string>();
            if (config.contains("ai") && config["ai"].contains(key) && config["ai"][key].is_string())
                return config["ai"][key].get<std::string>();
        } catch (...) {}
        return fallback;
    };

    std::string backend = getString("backend", "mistral");
    std::string voice   = getString("voice", "coqui");
    std::string model   = getString("model", "ggml-base.en.bin");

    buttons.emplace_back("Backend: " + backend, [this] {
        std::string current = config.value("backend", "mistral");
        std::string next = (current == "mistral") ? "openai" :
                           (current == "openai") ? "localai" : "mistral";
        config["backend"] = next;
        saveConfig();
        refreshButtons();
    });

    buttons.emplace_back("Voice: " + voice, [this] {
        std::string current = config.value("voice", "coqui");
        std::string next = (current == "coqui") ? "whisper" : "coqui";
        config["voice"] = next;
        saveConfig();
        refreshButtons();
    });

    buttons.emplace_back("Model: " + model, [this] {
        std::string current = config.value("model", "ggml-base.en.bin");
        std::string next = (current == "ggml-base.en.bin") ? "ggml-large.bin" : "ggml-base.en.bin";
        config["model"] = next;
        saveConfig();
        refreshButtons();
    });

    buttons.emplace_back("Close", [this] {
        setVisible(false);
    });

    float startY = position.y + 50.0f;
    for (size_t i = 0; i < buttons.size(); ++i) {
        buttons[i].setPosition(position.x + 20.0f, startY + i * 70.0f);
        buttons[i].setSize(400.0f, 50.0f);
    }

    LOG_DEBUG("UISettingsMenu", "Buttons refreshed");
}

void UISettingsMenu::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    
    if (!isVisible()) return;
    
    for (auto& b : buttons)
        b.update(input, dt);
}

void UISettingsMenu::drawOverlay(OverlayRenderer& renderer)
{
    if (!isVisible()) return;
    
    UIPanel::drawOverlay(renderer);
    
    // Draw buttons would go here - not implemented yet
    // for (auto& btn : buttons) { btn.draw(renderer); }
}
