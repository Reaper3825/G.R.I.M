#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_toggle.hpp"
#include "ui_scrollbox.hpp"
#include "ui_dropdown.hpp"  // ? NEW: Include dropdown
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

class OverlayRenderer;
struct InputState;

class UISettingsMenu : public UIPanel {
public:
    UISettingsMenu();

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;

private:
    void loadConfig();
    void saveConfig();
    void applyChanges();
    void createWidgets();
    
    void cycleBackend();
    void cycleVoice();
    void cycleModel();
    void cyclePersonality();
    void doSaveAndClose();
    void doCancel();
    
    // ? NEW: Helper to scan for available speaker embeddings
    std::vector<std::string> getSpeakerEmbeddings();

    nlohmann::json config;
    nlohmann::json pendingConfig;
    
    std::shared_ptr<UIScrollBox> scrollBox;
    std::vector<std::shared_ptr<UIButton>> buttons;
    std::vector<std::string> buttonLabels;
    std::vector<std::shared_ptr<UISlider>> sliders;
    std::vector<std::shared_ptr<UIToggle>> toggles;
    std::vector<std::shared_ptr<UIDropdown>> dropdowns;  // ? NEW: Dropdown storage
    
    bool hasChanges = false;
    bool isRefreshing = false;
    bool needsWidgetRefresh = false;
};
