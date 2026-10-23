#include "ui_world_canvas.hpp"
#include "core/input_parser.hpp"
#include "ui/overlay_renderer.hpp"
#include <algorithm>
#include <cmath>

using namespace GRIM::WorldLab;
namespace {
constexpr float width=230, header=44, rowHeight=25;
bool inside(Vec2 p, CanvasRect r) { return p.x>=r.origin.x && p.y>=r.origin.y && p.x<r.origin.x+r.size.x && p.y<r.origin.y+r.size.y; }
float distance(Vec2 a,Vec2 b) { return std::hypot(a.x-b.x,a.y-b.y); }
const std::vector<std::string> directions{"north","east","south","west"};
float segmentDistance(Vec2 p,Vec2 a,Vec2 b) {
    const float dx=b.x-a.x,dy=b.y-a.y,len=dx*dx+dy*dy;
    const float t=len>0 ? std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/len,0.f,1.f) : 0;
    return distance(p,{a.x+t*dx,a.y+t*dy});
}
}
std::vector<UIWorldCanvas::Row> UIWorldCanvas::rows(const std::string& room) const {
    std::vector<Row> out;
    // Iterative traversal avoids recursion on imported deeply nested worlds.
    std::vector<Row> pending{{room,-1}};
    while(!pending.empty()) {
        auto parent=pending.back(); pending.pop_back();
        if(parent.depth>=0) out.push_back(parent);
        const auto& entities=editor_.document().entities;
        for(auto it=entities.rbegin();it!=entities.rend();++it) if(it->parent==parent.id) pending.push_back({it->id,parent.depth+1});
    }
    return out;
}
CanvasRect UIWorldCanvas::roomRect(const std::string& id) const {
    const auto* p=editor_.document().position(id); if(!p) return {};
    Vec2 at{p->x,p->y};
    if(drag_==id && direction_.empty() && moved_) {at.x+=pointer_.x-start_.x; at.y+=pointer_.y-start_.y;}
    return {at,{width,header+std::max(1.f,static_cast<float>(rows(id).size()))*rowHeight+12}};
}
Vec2 UIWorldCanvas::port(const std::string& id,const std::string& dir) const {
    const auto r=roomRect(id); Vec2 p{r.origin.x+r.size.x/2,r.origin.y+r.size.y/2};
    if(dir=="north") p.y=r.origin.y;
    if(dir=="south") p.y+=r.size.y/2;
    if(dir=="east") p.x+=r.size.x/2;
    if(dir=="west") p.x=r.origin.x;
    return p;
}
std::string UIWorldCanvas::hitRoom(Vec2 p) const {
    const auto& layout=editor_.document().layout;
    for(auto it=layout.rbegin();it!=layout.rend();++it) if(inside(p,roomRect(it->id))) return it->id;
    return {};
}
std::string UIWorldCanvas::hitEntity(Vec2 p) const {
    auto id=hitRoom(p); if(id.empty()) return {};
    const auto r=roomRect(id); const auto children=rows(id);
    const int index=static_cast<int>(std::floor((p.y-r.origin.y-header)/rowHeight));
    return index>=0 && index<static_cast<int>(children.size()) ? children[index].id : id;
}
bool UIWorldCanvas::select(std::string id) {
    if(onSelect && !onSelect(id)) return false;
    selected=id; return true;
}
void UIWorldCanvas::cancelInput() { drag_.clear(); direction_.clear(); moved_=false; UICanvas::cancelInput(); }
void UIWorldCanvas::setPlayState(const nlohmann::json& state) {
    playRoom_=state.is_object()?state.value("room_id",std::string{}):"";
    const auto facts=state.is_object()?state.value("facts",nlohmann::json::array()):nlohmann::json::array();
    changedEntities_.clear();
    if(playFacts_.is_array())for(const auto* list:std::initializer_list<const nlohmann::json*>{&facts,&playFacts_})for(const auto& fact:*list){
        const auto& other=list==&facts?playFacts_:facts;
        if(std::find(other.begin(),other.end(),fact)==other.end())for(const auto& id:fact.at("arguments"))changedEntities_.push_back(id.get<std::string>());
    }
    playFacts_=facts;
    playStates_.clear();
    for(const auto& fact:facts){
        const auto predicate=fact.at("predicate").get<std::string>();
        if((predicate=="open" || predicate=="closed" || predicate=="locked") && fact.at("arguments").size()==1)
            playStates_[fact.at("arguments")[0].get<std::string>()]=predicate;
    }
}
void UIWorldCanvas::fitWorld() {
    if(editor_.document().layout.empty()) {resetView();return;}
    float x=1e9f,y=1e9f,right=-1e9f,bottom=-1e9f;
    for(const auto& p:editor_.document().layout) {auto r=roomRect(p.id);x=std::min(x,r.origin.x);y=std::min(y,r.origin.y);right=std::max(right,r.origin.x+r.size.x);bottom=std::max(bottom,r.origin.y+r.size.y);}
    fitToBounds({{x,y},{right-x,bottom-y}},48);
}
bool UIWorldCanvas::updateCanvasContent(const InputState& source,float) {
    const auto input=routeInput(source);
    if(selectionOnly) {
        if(input.mousePressed[0] && input.mouseInputEnabled) {
            const auto id=hitEntity(screenToWorld(input.mousePos));
            if(!id.empty()){select(id);return true;}
        }
        return false;
    }
    if(input.keyPressed[VK_ESCAPE]) {cancelInput();return true;}
    if(!drag_.empty() && !hasPointerCapture()) {cancelInput();return false;}
    if(!drag_.empty()) {
        if(!input.mouseInputEnabled) {cancelInput();return true;}
        pointer_=screenToWorld(input.mousePos);
        moved_ |= distance(pointer_,start_)*getZoom()>4;
        if(!input.mouseDown[0]) {
            auto id=drag_; const auto dir=direction_; const auto point=pointer_; const auto start=start_; const bool moved=moved_;
            // Hit test against authored positions rather than the moving preview.
            cancelInput();
            setFocused(true);
            const auto* entity=editor_.document().find(id);
            const bool movingRoom=entity && entity->kind=="room" && dir.empty();
            const bool validDrop=movingRoom || (containsScreenPoint(input.mousePos) && source.pointerInClip());
            if(input.mouseReleased[0] && moved && validDrop) {
                try {
                    std::string newSelection=id;
                    editor_.change([&](Document& d){
                        if(!dir.empty()) {
                            auto target=hitRoom(point);
                            if(target.empty()) target=d.addRoom(point.x-width/2,point.y-header/2);
                            d.connect(id,target,dir); newSelection=target;
                        } else if(d.find(id)->kind=="room") {
                            const auto p=*d.position(id); d.moveRoom(id,p.x+point.x-start.x,p.y+point.y-start.y);
                        } else {
                            const auto target=hitEntity(point);
                            if(target.empty() || target==id) return;
                            d.find(id)->parent=target; // Document validation rejects cycles/invalid holders.
                        }
                    });
                    select(newSelection); if(onChanged) onChanged();
                } catch(const std::exception& e) {if(onError) onError(e.what());}
            }
        }
        return true;
    }
    if(input.keyboardInputEnabled) {
        try {
            bool changed=false;
            if(input.ctrl && input.keyPressed['Z']) changed=input.shift ? editor_.redo() : editor_.undo();
            else if(input.ctrl && input.keyPressed['Y']) changed=editor_.redo();
            else if(input.keyPressed[VK_DELETE] && !selected.empty()) changed=editor_.change([&](Document& d){d.erase(selected);});
            if(changed) {if(onChanged) onChanged();return true;}
        } catch(const std::exception& e) {if(onError) onError(e.what());}
    }
    if(!input.mousePressed[0] || !containsScreenPoint(input.mousePos)) return false;
    const auto p=screenToWorld(input.mousePos);
    for(auto it=editor_.document().layout.rbegin();it!=editor_.document().layout.rend();++it) {
        for(const auto& dir:directions) if(distance(p,port(it->id,dir))*getZoom()<=10) {
            const auto id=it->id;
            if(select(id) && capturePointer()) {setFocused(true);drag_=id;direction_=dir;start_=pointer_=p;}
            return true;
        }
    }
    auto id=hitEntity(p);
    if(!id.empty()) {
        if(select(id) && capturePointer()) {setFocused(true);drag_=id;start_=pointer_=p;}
        return true;
    }
    for(const auto& edge:editor_.document().passages) {
        if(segmentDistance(p,port(edge.from,edge.direction),port(edge.to,opposite(edge.direction)))*getZoom()<8) {
            if(select(edge.id)) setFocused(true); return true;
        }
    }
    if(select("")) setFocused(true);
    return true;
}
void UIWorldCanvas::drawCanvasContent(OverlayRenderer& r) {
    using namespace UITheme::Colors;
    const auto& d=editor_.document();
    for(const auto& edge:d.passages) {
        const auto a=port(edge.from,edge.direction),b=port(edge.to,opposite(edge.direction));
        drawWorldLine(r,a,b,selected==edge.id?PrimaryLight:Primary,selected==edge.id?3.f:2.f);
        const auto state=playStates_.find(edge.id);
        const auto doorState=state==playStates_.end()?"start: "+edge.doorState:"now: "+state->second;
        drawWorldText(r,{(a.x+b.x)/2+6,(a.y+b.y)/2+6},edge.doorName.empty()?edge.direction:edge.doorName+" ("+doorState+")",TextMuted,12);
    }
    for(const auto& p:d.layout) {
        const auto rect=roomRect(p.id); if(!isWorldRectVisible(rect)) continue;
        const auto* e=d.find(p.id); const auto at=rect.origin;
        drawWorldRect(r,rect,playRoom_==p.id?Success:selected==p.id?BorderFocus:BorderPrimary);
        drawWorldRect(r,{{at.x+1,at.y+1},{rect.size.x-2,rect.size.y-2}},CardSurface);
        r.pushClipRect(worldToScreen({at.x+8,at.y+2}),{(width-16)*getZoom(),(rect.size.y-4)*getZoom()});
        drawWorldText(r,{at.x+12,at.y+9},e->name,TextPrimary,15);
        drawWorldText(r,{at.x+12,at.y+28},playRoom_==p.id?"PLAYTEST LOCATION":d.spawn==p.id?"PLAYER START":"ROOM",d.spawn==p.id || playRoom_==p.id?Success:TextMuted,10);
        const auto children=rows(p.id);
        if(children.empty()) drawWorldText(r,{at.x+12,at.y+header},"Select room to add objects",TextMuted,12);
        for(size_t i=0;i<children.size();++i) {
            const auto* child=d.find(children[i].id); const float y=at.y+header+i*rowHeight;
            if(selected==child->id) drawWorldRect(r,{{at.x+4,y-2},{width-8,rowHeight}},RowSelected);
            const float indent=std::min(children[i].depth,6)*12.f;
            const bool changed=std::find(changedEntities_.begin(),changedEntities_.end(),child->id)!=changedEntities_.end();
            drawWorldText(r,{at.x+12+indent,y},child->name,changed?Warning:TextSecondary,13);
        }
        r.popClipRect();
        if(!selectionOnly)for(const auto& dir:directions) {
            const auto pt=port(p.id,dir);
            drawWorldRect(r,{{pt.x-5,pt.y-5},{10,10}},Primary);
            drawWorldText(r,{pt.x+8,pt.y-7},dir.substr(0,1),TextMuted,11);
        }
    }
    if(!drag_.empty() && !direction_.empty()) drawWorldLine(r,port(drag_,direction_),pointer_,PrimaryLight,2);
    else if(!drag_.empty() && moved_ && d.find(drag_)->kind!="room") drawWorldText(r,{pointer_.x+12,pointer_.y},d.find(drag_)->name,PrimaryLight);
}
