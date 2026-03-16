#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_toggle.hpp"
#include "ui_scrollbox.hpp"
#include "ui_dropdown.hpp"  // ? NEW: Include dropdown
#include <vector>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

class OverlayRenderer;
struct InputState;

class UISettingsMenu : public UIPanel {
public:
    UISettingsMenu();

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

private:
    void loadConfig();
    void saveConfig();
    void applyChanges();
    void createWidgets();
    
    void cycleBackend();
    void cycleVoice();
    void cycleSpeaker();
    void cycleModel();
    void cyclePersonality();
    void doSaveAndClose();
    void doCancel();
    
    // ? NEW: Helper to scan for available speaker embeddings
    std::vector<std::string> getSpeakerEmbeddings();
    
    // ? NEW: Helper to scan for available fonts
    std::vector<std::string> getFontList();

    nlohmann::json config;
    nlohmann::json pendingConfig;
    
    std::shared_ptr<UIScrollBox> scrollBox;
    
    // Keep references to action buttons for external access (Save/Cancel)
    std::shared_ptr<UIButton> saveButton;
    std::shared_ptr<UIButton> cancelButton;
    
    bool hasChanges = false;
    bool isRefreshing = false;
    bool needsWidgetRefresh = false;

    std::unordered_map<std::string, std::string> m_fontPathMap;
};
