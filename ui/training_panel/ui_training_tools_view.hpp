#pragma once
#include "MMO/Core/ToolRegistry.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/primitives/ui_textarea.hpp"
#include "ui/primitives/ui_panel.hpp"

// Registry inspection only. This view never dispatches or changes tools.
class UITrainingToolsView {
public:
    UITrainingToolsView();
    void update(const InputState&, float dt, const PanelRect&, uint64_t panelId);
    void draw(OverlayRenderer&, const PanelRect&);
    void hide();
private:
    void layout(const PanelRect&);
    void refresh();
    void filter();
    void showSelection();
    UIInputBox search_;
    UIScrollBox list_;
    UITextArea details_;
    std::vector<GRIM::MMO::ToolDescriptor> tools_;
    std::vector<size_t> matches_;
    std::string query_, selectedId_;
    float refreshTimer_ = 2.0f;
};
