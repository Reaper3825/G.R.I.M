#pragma once
#include "core/host_exports.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/primitives/ui_textarea.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include <utility>

// Registry inspection only. This view never dispatches or changes tools.
class UITrainingToolsView {
public:
    GRIM_HOST_API ~UITrainingToolsView();
    GRIM_HOST_API UITrainingToolsView();
    GRIM_HOST_API void update(const InputState&, float dt, const PanelRect&, uint64_t panelId);
    GRIM_HOST_API void draw(OverlayRenderer&, const PanelRect&);
    GRIM_HOST_API void hide();
private:
    void layout(const PanelRect&);
    void refresh();
    void refreshProviders();
    void filter();
    void showSelection();
    UIInputBox search_;
    UIDropdown usageFilter_, pluginFilter_;
    UIScrollBox list_;
    UITextArea details_;
    std::vector<GRIM::MMO::RegisteredTool> tools_;
    std::vector<size_t> matches_;
    std::vector<std::string> providers_;
    std::string query_, selectedId_;
    float refreshTimer_ = 2.0f;
};
