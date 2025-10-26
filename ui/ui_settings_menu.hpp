#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include <vector>
#include <nlohmann/json.hpp>

class OverlayRenderer;  // Forward declaration
struct InputState; // Forward declaration

class UISettingsMenu : public UIPanel {
public:
    UISettingsMenu();

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;  // Changed to drawOverlay

private:
    void loadConfig();
    void saveConfig();
    void refreshButtons();

    nlohmann::json config;
    std::vector<UIButton> buttons;
};
