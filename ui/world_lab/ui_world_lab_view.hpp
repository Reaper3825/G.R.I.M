#pragma once
#include "ui_world_canvas.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui_world_session.hpp"
#include "world_asset.hpp"
#include "ui_world_actions.hpp"
#include <memory>

class UIWorldLabView {
public:
    explicit UIWorldLabView(std::filesystem::path recoveryPath = {}, std::filesystem::path gamesDirectory = {});
    ~UIWorldLabView();
    const GRIM::WorldLab::Document& document() const { return editor_.document(); }
    void update(const InputState&, float dt, const PanelRect&, uint64_t panel);
    void draw(OverlayRenderer&, const PanelRect&);
    void hide();
    void insertAsset(const GRIM::WorldLab::Asset&);
    std::function<void(const GRIM::WorldLab::Document&,const std::string&)> onSaveAsset;
private:
    void layout(const PanelRect&);
    void refresh();
    bool apply();
    void act(const std::function<void()>&);
    void addObject();
    void autosave();
    void refreshGames();
    void newGamePath();
    void openGame(const std::filesystem::path&);
    std::vector<Widget*> controls();
    GRIM::WorldLab::Editor editor_;
    UIWorldCanvas canvas_{editor_};
    UIWorldActions actions_{editor_};
    std::vector<std::unique_ptr<UIButton>> toolbar_;
    UIInputBox path_, title_, name_, description_;
    UIScrollBox inspectorScroll_;
    std::unique_ptr<UIDropdown> kind_, location_, state_;
    std::unique_ptr<UIDropdown> doorState_, doorKey_;
    std::vector<std::string> doorKeys_;
    std::unique_ptr<UIWorldSession> session_;
    bool sessionVisible_ = false;
    std::unique_ptr<UIButton> apply_, add_, spawn_, erase_, saveAsset_, actionsShortcut_;
    std::vector<std::string> locations_;
    PanelRect area_{}, inspector_{};
    std::string status_ = "Drag a compass handle to connect rooms or create a new room.";
    std::filesystem::path recoveryPath_, gamesDirectory_;
    std::unique_ptr<UIDropdown> games_;
    std::vector<std::filesystem::path> gameFiles_;
    float gamesTimer_ = 2;
    nlohmann::json recoveredSnapshot_;
    float recoveryTimer_ = 0;
};
