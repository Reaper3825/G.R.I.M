#include "ui_world_actions.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <algorithm>
#include <sstream>
using namespace GRIM::WorldLab;
namespace {
std::string parameterLabel(const std::string& type) {
    if(type=="o")return "item";
    if(type=="c")return "container";
    if(type=="s")return "surface";
    if(type=="d")return "door";
    if(type=="k")return "key";
    if(type=="f")return "food";
    if(type=="t")return "entity";
    return type;
}
std::string actionLabel(const nlohmann::json& action) {
    auto label=action["command"].get<std::string>();
    for(const auto& parameter:action["parameters"]){
        const auto token="{"+parameter["name"].get<std::string>()+"}";
        const auto at=label.find(token);
        if(at!=std::string::npos)label.replace(at,token.size(),"<"+parameterLabel(parameter["type"])+">");
    }
    // Examination variants share text but have different prerequisite rules.
    const auto id=action["id"].get<std::string>();
    if(id=="examine/I")label+=" (carried)";
    if(id=="examine/c")label+=" (inside container)";
    if(id=="examine/s")label+=" (on surface)";
    if(id=="examine/t")label+=" (in room)";
    return label;
}
}
UIWorldActions::UIWorldActions(const Editor& editor):editor_(editor) {
    actions_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[this](int,const auto&){parameters();});
    actions_->setFullWidth(true);actions_->setMaxVisibleItems(4);
    prepare_=std::make_unique<UIButton>("Prepare command",[this]{
        try{if(beforePrepare && !beforePrepare())return;const auto value=command();if(onPrepare)onPrepare(value);message_="Prepared; review in Play before Send.";}
        catch(const std::exception& e){message_=e.what();}
    });
    refresh();
}
void UIWorldActions::select(const std::string& id){if(selection_!=id){hide();selection_=id;refresh();}}
void UIWorldActions::refresh(){
    std::string previous;
    const auto old=actions_->getSelectedIndex();
    if(old>=0 && old<static_cast<int>(indices_.size()))previous=actionCatalog()["actions"][indices_[old]]["id"];
    snapshot_=editor_.document().json();indices_.clear();std::vector<std::string> labels;
    int chosen=0;
    for(size_t i=0;i<actionCatalog()["actions"].size();++i){const auto& a=actionCatalog()["actions"][i];
        if(!actionRelevant(editor_.document(),a,selection_))continue;
        if(a["id"]==previous)chosen=static_cast<int>(indices_.size());
        indices_.push_back(i);labels.push_back(actionLabel(a));
    }
    actions_->collapse();actions_->setItems(labels);actions_->setSelectedIndex(chosen);parameters();
}
void UIWorldActions::parameters(){
    for(auto& p:parameters_){p->collapse();p->cancelInput();}
    parameters_.clear();candidates_.clear();message_.clear();
    const auto index=actions_->getSelectedIndex();if(index<0 || index>=static_cast<int>(indices_.size()))return;
    const auto& action=actionCatalog()["actions"][indices_[index]];
    for(const auto& parameter:action["parameters"]){
        auto candidates=actionCandidates(editor_.document(),parameter["type"]);
        candidates.insert(candidates.begin(),"");std::vector<std::string> labels{"Choose "+parameterLabel(parameter["type"])};int selected=0;
        for(size_t i=1;i<candidates.size();++i){labels.push_back(actionEntityName(editor_.document(),candidates[i])+" ["+candidates[i]+"]");if(candidates[i]==selection_)selected=static_cast<int>(i);}
        candidates_.push_back(std::move(candidates));
        parameters_.push_back(std::make_unique<UIDropdown>("",labels,selected,[this](int,const auto&){message_.clear();}));
        parameters_.back()->setFullWidth(true);parameters_.back()->setMaxVisibleItems(4);
    }
}
std::string UIWorldActions::command() const {
    const auto index=actions_->getSelectedIndex();if(index<0 || index>=static_cast<int>(indices_.size()))throw std::runtime_error("Select an action");
    std::vector<std::string> bindings;
    for(size_t i=0;i<parameters_.size();++i){const int p=parameters_[i]->getSelectedIndex();bindings.push_back(p>=0 && p<static_cast<int>(candidates_[i].size())?candidates_[i][p]:"");}
    return composeAction(editor_.document(),actionCatalog()["actions"][indices_[index]],bindings);
}
std::vector<Widget*> UIWorldActions::controls(){std::vector<Widget*> result{actions_.get()};for(auto& p:parameters_)result.push_back(p.get());if(onPrepare)result.push_back(prepare_.get());return result;}
void UIWorldActions::layout(const PanelRect& a){
    auto place=[&](Widget& w,float y){w.setPosition(a.origin.x,a.origin.y+y);w.setSize(a.size.x,26);};
    place(*actions_,24);for(size_t i=0;i<parameters_.size();++i)place(*parameters_[i],54+30*static_cast<float>(i));
    place(*prepare_,116);
}
void UIWorldActions::update(const InputState& input,float dt,const PanelRect& a,uint64_t panel){
    if(snapshot_!=editor_.document().json()){hide();refresh();}layout(a);
    const auto& clipped=input;auto widgets=controls();
    for(auto* w:widgets)w->setPanelID(panel);
    // A changed action rebuilds parameter widgets; never retain pointers across callbacks.
    if(actions_->isExpanded()){actions_->update(clipped,dt);return;}
    for(auto& p:parameters_)if(p->isExpanded()){p->update(clipped,dt);return;}
    actions_->update(clipped,dt);
    for(auto& p:parameters_)p->update(clipped,dt);
    if(onPrepare)prepare_->update(clipped,dt);
}
void UIWorldActions::hide(){for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}}
void UIWorldActions::draw(OverlayRenderer& r,const PanelRect& a,bool details){
    using namespace UITheme::Colors;layout(a);r.pushClipRect(a.origin,a.size);
    r.drawText(a.origin,"Actions / TextWorld 1.7",TextPrimary,14);
    if(details){
        std::string text="Supported rules; live availability is checked in Play.\n";
        const auto index=actions_->getSelectedIndex();
        if(index>=0 && index<static_cast<int>(indices_.size())){
            const auto& action=actionCatalog()["actions"][indices_[index]];
            text+="Rule: "+action["id"].get<std::string>()+"\n";
            if(action["id"]=="lock/c" || action["id"]=="unlock/c")text+="Container key matching is not yet editable.\n";
            text+="Parameters: o=item, c=container, s=surface, d=door, k=key, f=food, t=thing.\n";
            text+="Rule variables: P=player, I=inventory, r=room.\n";
            for(const auto* field:{"requires","removes","adds"}){
                text+=std::string(field)+": ";for(const auto& fact:action[field])text+=fact.get<std::string>()+"; ";text+='\n';
            }
        }
        if(!message_.empty())text=message_+"\n"+text;
        float y=a.origin.y+150;std::istringstream input(text);std::string paragraph;
        while(std::getline(input,paragraph)){std::istringstream words(paragraph);std::string word,line;
            while(words>>word){if(!line.empty() && r.measureTextWidth(line+" "+word,11)>a.size.x-8){r.drawText({a.origin.x,y},line,TextSecondary,11);y+=16;line.clear();}if(!line.empty())line+=' ';line+=word;}
            r.drawText({a.origin.x,y},line,TextSecondary,11);y+=16;
        }
    }else if(!message_.empty())r.drawText({a.origin.x,a.origin.y+146},message_,TextMuted,10);
    for(auto* w:controls())w->drawOverlay(r,{});
    r.popClipRect();
    for(auto* w:controls())if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isExpanded())d->drawExpandedList(r,{});
}
