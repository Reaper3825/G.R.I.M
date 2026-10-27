#pragma once
#include "world_actions.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_panel.hpp"
#include <memory>

// Shared offline inspector / Play composer. Never evaluates action applicability.
class UIWorldActions {
public:
    explicit UIWorldActions(const GRIM::WorldLab::Editor&);
    void select(const std::string& id);
    void update(const InputState&,float,const PanelRect&,uint64_t panel);
    void draw(OverlayRenderer&,const PanelRect&,bool details=true);
    void hide();
    std::string command() const;
    std::function<void(const std::string&)> onPrepare;
    std::function<bool()> beforePrepare;
private:
    void refresh();
    void parameters();
    void layout(const PanelRect&);
    std::vector<Widget*> controls();
    const GRIM::WorldLab::Editor& editor_;
    nlohmann::json snapshot_,catalog_;
    std::string selection_,message_;
    std::unique_ptr<UIDropdown> actions_;
    std::vector<std::unique_ptr<UIDropdown>> parameters_;
    std::vector<size_t> indices_;
    std::vector<std::vector<std::string>> candidates_;
    std::unique_ptr<UIButton> prepare_;
};
