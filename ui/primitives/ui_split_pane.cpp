#include "ui_split_pane.hpp"
#include "core/input_parser.hpp"
#include "ui/overlay_renderer.hpp"
#include <algorithm>
#include <cmath>

namespace {
bool inside(Vec2 p,PanelRect r) {
    return p.x>=r.origin.x && p.y>=r.origin.y && p.x<r.origin.x+r.size.x && p.y<r.origin.y+r.size.y;
}
}
void UISplitPane::setMinimumSizes(float a,float b) {
    if(std::isfinite(a) && std::isfinite(b) && a>=0 && b>=0){firstMinimum_=a;secondMinimum_=b;}
}
void UISplitPane::setDefaultSecondSize(float value) {if(std::isfinite(value) && value>=0)defaultSecond_=value;}
void UISplitPane::setSecondSize(float value) {if(std::isfinite(value) && value>=0){preferredSecond_=value;custom_=true;}}
void UISplitPane::reset() {cancelInput();custom_=false;}
float UISplitPane::extent() const {return std::max(0.f,axis_==Axis::Horizontal?size.x:size.y);}
float UISplitPane::dividerExtent() const {return std::min(12.f,extent());}
float UISplitPane::secondExtent() const {
    const float available=extent()-dividerExtent(),minimum=firstMinimum_+secondMinimum_;
    // A tiny parent must never produce negative or overlapping rectangles.
    // Keep the user's preferred size intact so it returns when space permits.
    if(available<minimum)return minimum>0?available*secondMinimum_/minimum:available/2;
    return std::clamp(custom_?preferredSecond_:defaultSecond_,secondMinimum_,available-firstMinimum_);
}
PanelRect UISplitPane::firstRect() const {
    auto r=PanelRect{position,{std::max(0.f,size.x),std::max(0.f,size.y)}};
    (axis_==Axis::Horizontal?r.size.x:r.size.y)=extent()-dividerExtent()-secondExtent();return r;
}
PanelRect UISplitPane::dividerRect() const {
    auto r=firstRect();
    if(axis_==Axis::Horizontal){r.origin.x+=r.size.x;r.size.x=dividerExtent();}
    else {r.origin.y+=r.size.y;r.size.y=dividerExtent();}return r;
}
PanelRect UISplitPane::secondRect() const {
    auto r=dividerRect();
    if(axis_==Axis::Horizontal){r.origin.x+=r.size.x;r.size.x=secondExtent();}
    else {r.origin.y+=r.size.y;r.size.y=secondExtent();}return r;
}
void UISplitPane::cancelInput() {
    if(dragging_){preferredSecond_=savedPreference_;custom_=savedCustom_;}
    dragging_=hovered_=false;Widget::cancelInput();
}
void UISplitPane::update(const InputState& source,float) {
    if(!visible || source.interactionCancelled){cancelInput();return;}
    const auto input=routeInput(source);
    hovered_=input.mouseInputEnabled && source.pointerInClip() && inside(input.mousePos,dividerRect());
    if(dragging_ && (!hasPointerCapture() || !input.mouseInputEnabled || input.keyPressed[VK_ESCAPE])){cancelInput();return;}
    if(hovered_ && input.mousePressed[1]){reset();return;}
    const float pointer=axis_==Axis::Horizontal?input.mousePos.x:input.mousePos.y;
    if(hovered_ && input.mousePressed[0] && capturePointer()) {
        dragging_=true;startPointer_=pointer;startExtent_=secondExtent();
        savedPreference_=preferredSecond_;savedCustom_=custom_;
    }
    if(!dragging_)return;
    if(input.mouseDown[0] || input.mouseReleased[0]) {
        const float available=extent()-dividerExtent();
        if(available>=firstMinimum_+secondMinimum_)
            setSecondSize(std::clamp(startExtent_-(pointer-startPointer_),secondMinimum_,available-firstMinimum_));
    }
    if(!input.mouseDown[0]) {
        if(input.mouseReleased[0]){dragging_=false;releasePointer();}
        else cancelInput();
    }
}
void UISplitPane::drawOverlay(OverlayRenderer& r,const Vec2&) {
    if(!visible)return;
    using namespace UITheme::Colors;
    const auto d=dividerRect();r.drawRect(d.origin,d.size,dragging_?BorderFocus:hovered_?BorderPrimary:ContentAreaBg);
    const Vec2 center{d.origin.x+d.size.x/2,d.origin.y+d.size.y/2};
    if(axis_==Axis::Horizontal)r.drawRect({center.x-1,center.y-std::min(18.f,d.size.y/2)},{2,std::min(36.f,d.size.y)},TextMuted);
    else r.drawRect({center.x-std::min(18.f,d.size.x/2),center.y-1},{std::min(36.f,d.size.x),2},TextMuted);
}
