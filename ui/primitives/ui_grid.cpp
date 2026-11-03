#include "ui_grid.hpp"
#include "core/input_parser.hpp"
#include "ui/overlay_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace {
float nonnegative(float value) {
    if(!std::isfinite(value) || value<0) throw std::invalid_argument("Grid dimension must be finite and nonnegative");
    return value;
}
}
UIGrid::UIGrid(size_t count) {setColumns(count);}
void UIGrid::setColumns(size_t count) {
    if(!count) throw std::invalid_argument("Grid needs at least one column");
    columns=count;minimumWidth=0;layout();
}
void UIGrid::setAutoColumns(float width) {
    nonnegative(width);if(width==0) throw std::invalid_argument("Minimum tile width must be positive");
    minimumWidth=width;layout();
}
void UIGrid::setSpacing(float x,float y) {nonnegative(x);nonnegative(y);gapX=x;gapY=y;layout();}
void UIGrid::setPadding(float value) {padding=nonnegative(value);layout();}
void UIGrid::setRowHeight(float value) {rowHeight=nonnegative(value);fitRows=false;layout();}
void UIGrid::setFitRows(bool value) {fitRows=value;layout();}
void UIGrid::addWidget(std::shared_ptr<Widget> child) {
    if(!child || child.get()==this || std::find(widgets.begin(),widgets.end(),child)!=widgets.end()) return;
    child->setPanelID(getPanelID());widgets.push_back(std::move(child));layout();
}
void UIGrid::removeWidget(Widget* child) {
    for(const auto& w:widgets) if(w.get()==child) w->cancelInput();
    widgets.erase(std::remove_if(widgets.begin(),widgets.end(),[child](const auto& w){return w.get()==child;}),widgets.end());layout();
}
void UIGrid::clearWidgets() {for(auto& w:widgets) w->cancelInput();widgets.clear();layout();}
void UIGrid::cancelInput() {Widget::cancelInput();for(auto& w:widgets) w->cancelInput();}
void UIGrid::layout() {
    const float width=(std::max)(0.0f,size.x-2*padding);
    actualColumns=columns;
    if(minimumWidth>0) actualColumns=static_cast<size_t>((std::max)(1.0f,std::floor((width+gapX)/(minimumWidth+gapX))));
    size_t count=0;for(const auto& w:widgets) if(w->isVisible()) ++count;
    if(!count) {contentHeight=0;return;}
    const size_t rows=1+(count-1)/actualColumns;
    // Compress gaps in tiny viewports instead of positioning tiles outside them.
    const float spacingX=actualColumns>1 ? (std::min)(gapX,width/float(actualColumns-1)) : 0;
    const float tileWidth=(std::max)(0.0f,(width-spacingX*float(actualColumns-1))/float(actualColumns));
    const float height=(std::max)(0.0f,size.y-2*padding);
    const float spacingY=fitRows && rows>1 ? (std::min)(gapY,height/float(rows-1)) : gapY;
    const float tileHeight=fitRows ? (std::max)(0.0f,(height-spacingY*float(rows-1))/float(rows)) : rowHeight;
    contentHeight=2*padding+tileHeight*float(rows)+spacingY*float(rows-1);
    size_t index=0;
    for(auto& child:widgets) {
        child->setPanelID(getPanelID());
        if(!child->isVisible()) continue;
        child->setPosition(position.x+padding+float(index%actualColumns)*(tileWidth+spacingX),
            position.y+padding+float(index/actualColumns)*(tileHeight+spacingY));
        child->setSize(tileWidth,tileHeight);++index;
    }
}
void UIGrid::update(const InputState& input,float dt) {
    if(!isVisible()) return;layout();
    const auto clipped=input.clipped(position,size);
    const auto snapshot=widgets;
    for(const auto& child:snapshot) if(child->isVisible() && std::find(widgets.begin(),widgets.end(),child)!=widgets.end()) child->update(clipped,dt);
}
void UIGrid::draw(UIRenderer& renderer) {
    if(!isVisible()) return;layout();for(auto& child:widgets) if(child->isVisible()) child->draw(renderer);
}
void UIGrid::drawOverlay(OverlayRenderer& renderer,const Vec2& panelPos) {
    if(!isVisible()) return;layout();renderer.pushClipRect(position,size);
    for(auto& child:widgets) if(child->isVisible()) child->drawOverlay(renderer,panelPos);
    renderer.popClipRect();
}
