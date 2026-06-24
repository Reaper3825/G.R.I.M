#pragma once
#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_slider.hpp"
#include "primitives/ui_toggle.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "primitives/ui_dropdown.hpp"
#include "primitives/ui_textarea.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

class OverlayRenderer;
struct InputState;

enum class SettingsTab : uint8_t {
    General     = 0,
    Voice       = 1,
    Audio       = 2,
    Vision      = 3,
    UIGraphics  = 4,
    Preferences = 5,
    Memory      = 6,
    Intents     = 7
};

class UISettingsMenu : public UIPanel {
public:
    UISettingsMenu();

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

    void setTab(SettingsTab tab);
    SettingsTab currentTab() const { return activeTab_; }

private:
    void loadConfig();
    void saveConfig();
    void applyChanges();
    void createWidgets();
    
    // Per-tab widget creation
    void createGeneralWidgets();
    void createVoiceWidgets();
    void createAudioWidgets();
    void createVisionWidgets();
    void createUIGraphicsWidgets();
    void createPreferencesWidgets();
    void createMemoryWidgets();
    void createIntentsWidgets();

    void cycleBackend();
    void cycleVoice();
    void cycleSpeaker();
    void cycleModel();
    void cyclePersonality();
    void doSaveAndClose();
    void doCancel();
    
    std::vector<std::string> getSpeakerEmbeddings();
    std::vector<std::string> getFontList();

    nlohmann::json config;
    nlohmann::json pendingConfig;
    
    // Tab system
    SettingsTab activeTab_ = SettingsTab::General;
    std::shared_ptr<UIButton> tabGeneralBtn_;
    std::shared_ptr<UIButton> tabVoiceBtn_;
    std::shared_ptr<UIButton> tabAudioBtn_;
    std::shared_ptr<UIButton> tabVisionBtn_;
    std::shared_ptr<UIButton> tabUIGraphicsBtn_;
    std::shared_ptr<UIButton> tabPreferencesBtn_;
    std::shared_ptr<UIButton> tabMemoryBtn_;
    std::shared_ptr<UIButton> tabIntentsBtn_;

    std::shared_ptr<UIScrollBox> scrollBox;
    
    std::shared_ptr<UIButton> saveButton;
    std::shared_ptr<UIButton> cancelButton;
    
    bool hasChanges = false;
    bool isRefreshing = false;
    bool needsWidgetRefresh = false;

    std::unordered_map<std::string, std::string> m_fontPathMap;
};
