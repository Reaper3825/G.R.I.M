#pragma once
#include "world_asset.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <algorithm>
#include <cmath>
#include <optional>

// Owns the pointer throughout a library-to-canvas gesture. The destination
// resolves placement; the browser never mutates the world or the asset editor.
class UIWorldAssetBrowser : public Widget {
public:
    explicit UIWorldAssetBrowser(std::filesystem::path directory):directory_(std::move(directory)),
        type_("",{"All types","room","object","key","container","supporter","food"},0,
            [this](int,const std::string&){filter();}) {
        search_.setPlaceholder("Search assets...");
    }
    std::function<bool(const GRIM::WorldLab::Asset&,Vec2)> canDrop;
    std::function<void(const GRIM::WorldLab::Asset&,Vec2)> onDrop;
    std::function<void(const GRIM::WorldLab::Asset&,const std::filesystem::path&)> onOpen;
    std::function<void(const std::string&)> onError;
    void refreshLibrary() {timer_=2;}
    void selectPath(const std::filesystem::path& path) {selected_=path;}
    bool wantsFocus() const override {return true;}
    void cancelInput() override {drag_.reset();moved_=false;Widget::cancelInput();}
    void hide() {cancelInput();search_.cancelInput();type_.cancelInput();type_.collapse();scroll_.cancelInput();timer_=2;}
    void update(const InputState& source,float dt) override {
        if(source.interactionCancelled){hide();return;}
        layout();search_.setPanelID(panelID);type_.setPanelID(panelID);scroll_.setPanelID(panelID);
        timer_+=std::max(0.f,dt);
        if(timer_>=2 && !drag_ && !type_.isExpanded()) {
            timer_=0;
            try {entries_=GRIM::WorldLab::listAssets(directory_);filter(false);}
            catch(const std::exception& e){if(onError)onError(e.what());}
        }
        type_.update(source,dt);search_.update(source,dt);
        if(query_!=search_.getText()){query_=search_.getText();filter();}
        layout();
        const auto input=routeInput(source);
        if(drag_) {
            if(!hasPointerCapture() || !input.mouseInputEnabled || input.keyPressed[VK_ESCAPE]){cancelInput();return;}
            pointer_=input.mousePos;
            moved_|=std::hypot(pointer_.x-start_.x,pointer_.y-start_.y)>4;
            if(!input.mouseDown[0]) {
                const auto asset=*drag_;
                const bool commit=input.mouseReleased[0] && moved_ && source.pointerInClip() && canDrop && canDrop(asset,pointer_);
                const bool open=input.mouseReleased[0] && !moved_ && source.pointerInClip() && inside(pointer_,gridOrigin(),gridSize()) && onOpen;
                const auto path=pressedPath_;
                cancelInput();
                if(commit && onDrop)onDrop(asset,pointer_);
                if(open)try {onOpen(asset,path);selected_=path;}
                    catch(const std::exception& e){if(onError)onError(e.what());}
            }
            return;
        }
        scroll_.update(source.clipped(gridOrigin(),gridSize()),dt);
        if(!input.mousePressed[0] || !input.mouseInputEnabled || !inside(input.mousePos,gridOrigin(),gridSize()) || input.mousePos.x>=position.x+size.x-18)return;
        const int column=static_cast<int>((input.mousePos.x-position.x)/tileWidth());
        const int row=static_cast<int>((input.mousePos.y-gridOrigin().y+scroll_.getScrollOffset())/72);
        const size_t index=static_cast<size_t>(row*columns()+column);
        if(index>=filtered_.size())return;
        const auto& entry=entries_[filtered_[index]];
        if(!onOpen)selected_=entry.path;
        if(!entry.valid)return;
        try {
            auto asset=GRIM::WorldLab::Asset::parse(GRIM::WorldLab::loadJsonFile(entry.path));
            if(capturePointer()){setFocused(true);pressedPath_=entry.path;drag_=std::move(asset);start_=pointer_=input.mousePos;moved_=false;}
        }catch(const std::exception& e){if(onError)onError(e.what());}
    }
    void drawOverlay(OverlayRenderer& r,const Vec2&) override {
        using namespace UITheme::Colors;layout();
        r.drawText(position,"Asset Library",TextPrimary,14);
        search_.drawOverlay(r,{});type_.drawOverlay(r,{});scroll_.drawOverlay(r,{});
        r.pushClipRect(gridOrigin(),{std::max(0.f,size.x-18),gridSize().y});
        if(filtered_.empty())r.drawText({position.x+8,gridOrigin().y+10},"No matching assets. Create and save assets in the Assets tab.",TextMuted,12);
        for(size_t i=0;i<filtered_.size();++i) {
            const auto& e=entries_[filtered_[i]];
            const Vec2 p{position.x+(i%columns())*tileWidth(),gridOrigin().y+(i/columns())*72-scroll_.getScrollOffset()};
            if(p.y+68<gridOrigin().y || p.y>=gridOrigin().y+gridSize().y)continue;
            const Vec2 extent{tileWidth()-6,66};
            r.drawRect(p,extent,selected_==e.path?BorderFocus:BorderPrimary);
            r.drawRect({p.x+1,p.y+1},{extent.x-2,extent.y-2},CardSurface);
            r.pushClipRect({p.x+6,p.y+4},{std::max(0.f,extent.x-12),58});
            r.drawText({p.x+8,p.y+8},e.name,TextPrimary,12);
            r.drawText({p.x+8,p.y+30},e.valid?e.kind+" / "+std::to_string(e.entityCount)+" entities":"Invalid asset",e.valid?TextMuted:Warning,11);
            r.drawText({p.x+8,p.y+47},e.valid?(onOpen?"Click to edit":"Drag into world"):"Invalid file",TextMuted,10);
            r.popClipRect();
        }
        r.popClipRect();
    }
    // Draw after the canvas and inspector, outside the tile viewport clip.
    void drawFloating(OverlayRenderer& r) {
        using namespace UITheme::Colors;
        if(type_.isExpanded())type_.drawExpandedList(r,{});
        if(!drag_ || !moved_ || !onDrop)return;
        const bool valid=canDrop && canDrop(*drag_,pointer_);
        const Vec2 p{pointer_.x+12,pointer_.y+12};
        r.drawRect(p,{250,48},CardSurface);
        r.drawText({p.x+6,p.y+5},drag_->name,TextPrimary,12);
        r.drawText({p.x+6,p.y+26},valid?"Release to insert":"Drop on canvas / a valid holder",valid?Success:Warning,11);
    }
private:
    static bool inside(Vec2 p,Vec2 origin,Vec2 extent){return p.x>=origin.x && p.y>=origin.y && p.x<origin.x+extent.x && p.y<origin.y+extent.y;}
    int columns() const {return std::max(1,static_cast<int>((size.x-18)/190));}
    float tileWidth() const {return std::max(1.f,(size.x-18)/columns());}
    Vec2 gridOrigin() const {return {position.x,position.y+58};}
    Vec2 gridSize() const {return {size.x,std::max(0.f,size.y-58)};}
    void layout() {
        search_.setPosition(position.x,position.y+22);search_.setSize(std::max(20.f,size.x-170),28);
        type_.setMaxVisibleItems(std::max(1,static_cast<int>((size.y-50)/25)));
        type_.setPosition(position.x+std::max(20.f,size.x-164),position.y+22);type_.setSize(160,28);
        scroll_.setPosition(gridOrigin());scroll_.setSize(gridSize());
        scroll_.setContentHeight(static_cast<float>((filtered_.size()+columns()-1)/columns())*72);
    }
    void filter(bool reset=true) {
        filtered_=GRIM::WorldLab::searchAssets(entries_,query_,type_.getSelectedIndex()==0?"":type_.getSelectedItem());
        if(reset)scroll_.scrollTo(0);
    }
    std::filesystem::path directory_,selected_,pressedPath_;
    UIInputBox search_;
    UIDropdown type_;
    UIScrollBox scroll_;
    std::vector<GRIM::WorldLab::AssetEntry> entries_;
    std::vector<size_t> filtered_;
    std::string query_;
    std::optional<GRIM::WorldLab::Asset> drag_;
    Vec2 start_{},pointer_{};
    bool moved_=false;
    float timer_=2;
};
