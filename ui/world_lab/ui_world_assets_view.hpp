#pragma once
#include "world_asset.hpp"
#include "ui_world_actions.hpp"
#include "ui_world_canvas.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/primitives/ui_panel.hpp"
#include <memory>

class UIWorldAssetsView {
public:
    explicit UIWorldAssetsView(std::filesystem::path directory);
    void update(const InputState&,float,const PanelRect&,uint64_t);
    void draw(OverlayRenderer&,const PanelRect&);
    void hide();
    void capture(const GRIM::WorldLab::Document&,const std::string& root);
    std::function<void(const GRIM::WorldLab::Asset&)> onInsert;
    const GRIM::WorldLab::Document& document() const {return editor_.document();}
private:
    void layout(const PanelRect&);
    void refresh();
    void refreshLibrary();
    bool apply();
    bool dirty() const;
    void requireClean() const;
    void setAsset(const GRIM::WorldLab::Asset&,const std::filesystem::path& = {});
    GRIM::WorldLab::Asset asset() const;
    void save(bool copy);
    void act(const std::function<void()>&);
    std::vector<Widget*> controls();
    GRIM::WorldLab::Editor editor_;
    UIWorldCanvas canvas_{editor_};
    UIWorldActions actions_{editor_};
    std::filesystem::path directory_,path_;
    std::string root_,status_="Create an asset or select a saved one. Add to world inserts an independent copy.";
    nlohmann::json baseline_;
    std::string baselineRoot_;
    std::vector<GRIM::WorldLab::AssetEntry> entries_;
    std::unique_ptr<UIDropdown> library_,kind_,state_;
    UIInputBox assetName_,name_,description_;
    std::vector<std::unique_ptr<UIButton>> buttons_;
    UIScrollBox scroll_;
    PanelRect inspector_{};
    float libraryTimer_=2;
};
