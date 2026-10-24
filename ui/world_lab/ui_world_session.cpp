#include "ui_world_session.hpp"
#include "core/input_parser.hpp"
#include "ui/overlay_renderer.hpp"
#include "ui/input/ui_input_router.hpp"
#include <algorithm>
#include <sstream>

using namespace GRIM::WorldLab;
namespace {
const std::vector<std::string> predicates{"player_at","carrying","at","in","on","open","closed","locked","eaten"};
const std::vector<std::string> predicateLabels{"Player is in room","Player carries","Object is in room","Object is inside","Object is on","Is open","Is closed","Is locked","Has been eaten"};
void place(Widget& w,float x,float y,float width=168,float height=28){w.setPosition(x,y);w.setSize(width,height);}
std::string label(const Document& d,const std::string& id){if(auto* e=d.find(id))return e->name;for(const auto& p:d.passages)if(p.id==id)return p.doorName;return id;}
std::string describe(const Document& d,const Condition& c){return c.predicate+" "+label(d,c.subject)+(c.target.empty()?"":" / "+label(d,c.target));}
}
UIWorldSession::UIWorldSession(Editor& editor):editor_(editor),actions_(editor) {
    actions_.onPrepare=[this](const std::string& command){
        if(!catalogCompatible_)throw std::runtime_error("Action catalog changed; refresh the bundled catalog first");
        prepareCommand(command);
    };
    available_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Live available commands"},0,[this](int index,const auto&){
        if(currentGame() && !done_ && index>0 && index<=static_cast<int>(availableCommands_.size()))prepareCommand(availableCommands_[index-1]);
        available_->setSelectedIndex(0);
    });
    available_->setFullWidth(true);available_->setMaxVisibleItems(4);
    objective_.setPlaceholder("Describe the objective");command_.setPlaceholder("Type a game command");
    objective_.OnTextSubmitted.Bind([this](const auto&){apply();});
    command_.OnTextSubmitted.Bind([this](const auto&){request("step");});
    predicate_=std::make_unique<UIDropdown>("",predicateLabels,0,[this](int,const auto&){conditionOptions();});
    subject_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[](int,const auto&){});
    target_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[](int,const auto&){});
    conditions_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[](int,const auto&){});
    history_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[this](int index,const auto&){
        if(index>=0 && index<static_cast<int>(historyEntries_.size())){historyIndex_=index;scroll_.scrollTo(0);if(onState)onState(historyEntries_[index].state);}
    });
    auto challengeButton=[&](const std::string& name,std::function<void()> fn){challengeButtons_.push_back(std::make_unique<UIButton>(name,[this,fn]{report(fn);}));};
    challengeButton("Add win",[this]{addCondition(false);});
    challengeButton("Add fail",[this]{addCondition(true);});
    challengeButton("Remove",[this]{const int index=conditions_->getSelectedIndex();editor_.change([&](Document& d){
        if(index>=0 && index<static_cast<int>(d.winConditions.size()))d.winConditions.erase(d.winConditions.begin()+index);
        else if(index>=static_cast<int>(d.winConditions.size()) && index<static_cast<int>(d.winConditions.size()+d.failConditions.size()))d.failConditions.erase(d.failConditions.begin()+index-d.winConditions.size());
    });refresh();});
    challengeButton("Validate",[this]{request("validate");});
    challengeButton("Apply goal",[this]{apply();});
    auto playButton=[&](const std::string& name,std::function<void()> fn){playButtons_.push_back(std::make_unique<UIButton>(name,[this,fn]{report(fn);}));};
    playButton("Compile & Play",[this]{request("compile");});
    playButton("Reset",[this]{request("reset");});
    playButton("Stop",[this]{client_.stop();live_=false;recording_=false;message_="Stopping worker...";if(onState)onState({});});
    playButton("Record",[this]{request("record");});
    playButton("Save solution",[this]{request("verify_solution");});
    playButton("Send",[this]{request("step");});
    playButton("Check saved",[this]{request("check_saved");});
    playButton("Actions",[this]{composerVisible_=!composerVisible_;actions_.hide();});
    refresh();
}
void UIWorldSession::report(const std::function<void()>& action){try{action();}catch(const std::exception& e){message_=e.what();}}
bool UIWorldSession::apply(){
    try {
        // External undo/load may have replaced the objective. Only commit a field
        // that the user actually changed since its last refresh.
        if(lastDocument_.is_object() && objective_.getText()==lastDocument_.value("objective",std::string{}))return true;
        editor_.change([&](Document& d){d.objective=objective_.getText();});
        lastDocument_["objective"]=objective_.getText();
        return true;
    }catch(const std::exception& e){message_=e.what();return false;}
}
void UIWorldSession::conditionOptions(){
    const auto& d=editor_.document();const auto pred=predicates.at(predicate_->getSelectedIndex());
    subjects_.clear();targets_.clear();std::vector<std::string> names,targetNames;
    for(const auto& e:d.entities){
        const bool portable=e.kind=="object" || e.kind=="key" || e.kind=="food";
        const bool accept=pred=="player_at"?e.kind=="room":pred=="eaten"?e.kind=="food":
            (pred=="open" || pred=="closed" || pred=="locked")?e.kind=="container":portable;
        if(accept){subjects_.push_back(e.id);names.push_back(e.name+" ["+e.id+"]");}
        if((pred=="at" && e.kind=="room") || (pred=="in" && e.kind=="container") || (pred=="on" && e.kind=="supporter")){
            targets_.push_back(e.id);targetNames.push_back(e.name+" ["+e.id+"]");
        }
    }
    if(pred=="open" || pred=="closed" || pred=="locked")for(const auto& p:d.passages)if(!p.doorName.empty()){subjects_.push_back(p.id);names.push_back(p.doorName+" ["+p.id+"]");}
    subject_->setItems(names);subject_->setSelectedIndex(0);target_->setItems(targetNames);target_->setSelectedIndex(0);
    target_->setVisible(challenge_ && (pred=="at" || pred=="in" || pred=="on"));
}
void UIWorldSession::refresh(){
    const auto& d=editor_.document();objective_.setText(d.objective);lastDocument_=d.json();
    if(live_ && !currentGame() && onState)onState({});
    conditionOptions();std::vector<std::string> list;
    for(const auto& c:d.winConditions)list.push_back("Win: "+describe(d,c));
    for(const auto& c:d.failConditions)list.push_back("Fail: "+describe(d,c));
    conditions_->setItems(list);conditions_->setSelectedIndex(0);
}
void UIWorldSession::addCondition(bool failure){
    if(!apply())return;
    int s=subject_->getSelectedIndex(),t=target_->getSelectedIndex();
    if(s<0 || s>=static_cast<int>(subjects_.size()))throw std::runtime_error("Choose a matching subject first");
    Condition c{predicates.at(predicate_->getSelectedIndex()),subjects_[s],{}};
    if(c.predicate=="at" || c.predicate=="in" || c.predicate=="on"){
        if(t<0 || t>=static_cast<int>(targets_.size()))throw std::runtime_error("Choose a target location");c.target=targets_[t];
    }
    editor_.change([&](Document& d){(failure?d.failConditions:d.winConditions).push_back(c);});refresh();
}
bool UIWorldSession::currentGame() const{return live_ && compiled_==editor_.document().gameJson();}
void UIWorldSession::request(const std::string& op){
    report([&]{
        if(client_.busy())throw std::runtime_error("Wait for the current request or press Stop");
        if(!apply())return;
        if(op!="compile" && op!="validate" && !currentGame())throw std::runtime_error("Compile & Play the current world first");
        nlohmann::json request;request["op"]=op;
        if(op=="compile" || op=="validate"){sentDocument_=editor_.document().gameJson();request["document"]=sentDocument_;}
        if(op=="step"){
            if(done_)throw std::runtime_error("Episode finished; reset to play again");
            sentCommand_=command_.getText();if(sentCommand_.empty())return;request["command"]=sentCommand_;
        }
        if(op=="record")request["op"]="reset";
        if(op=="verify_solution"){
            if(!recording_)throw std::runtime_error("Press Record to reset and record a solution first");
            request["commands"]=recorded_;
        }
        if(op=="check_saved"){request["op"]="verify_solution";request["commands"]=editor_.document().solution;}
        if(client_.begin(std::move(request))){operation_=op;details_.clear();message_="Working: "+op+"...";}
    });
}
void UIWorldSession::receive(){
    auto response=client_.poll();if(!response)return;
    if(!response->value("ok",false)){
        if(response->value("transport_error",false)){live_=false;recording_=false;if(onState)onState({});}
        details_=response->value("error",std::string{"TextWorld request failed"});
        message_="Request failed. See details below.";
        const auto id=response->value("entity_id",std::string{});if(!id.empty() && onErrorEntity)onErrorEntity(id);
        return;
    }
    const auto& result=response->at("result");
    if(result.value("stopped",false)){live_=false;message_="Worker stopped.";return;}
    if(operation_=="validate"){message_="Compiled and reset successfully. Solvability is not proven.";}
    if(operation_=="verify_solution" || operation_=="check_saved"){
        if(!currentGame()){message_="World changed during replay; recompile and verify again.";return;}
        if(operation_=="verify_solution"){
            editor_.change([&](Document& d){d.solution=result.at("commands").get<std::vector<std::string>>();});recording_=false;
        }
        message_="Solution replay reached a win. Save the world to keep it.";
    }
    if(result.contains("state")){
        if(operation_=="compile")catalogCompatible_=result.contains("action_catalog") && result.at("action_catalog")==actionCatalog();
        if(operation_=="compile"){compiled_=sentDocument_;live_=true;recording_=false;recorded_.clear();historyEntries_.clear();}
        if(operation_=="reset" || operation_=="record"){historyEntries_.clear();recorded_.clear();recording_=operation_=="record";}
        if(operation_=="step" && recording_)recorded_.push_back(sentCommand_);
        const auto& state=result.at("state");done_=state.value("done",false);
        availableCommands_=state.value("admissible_commands",std::vector<std::string>{});
        std::vector<std::string> availableLabels{"Live available commands"};
        if(!done_)availableLabels.insert(availableLabels.end(),availableCommands_.begin(),availableCommands_.end());
        available_->collapse();available_->setItems(availableLabels);available_->setSelectedIndex(0);
        historyEntries_.push_back({operation_=="step"?sentCommand_:"Start",state.value("feedback",std::string{}),state});
        std::vector<std::string> labels;for(size_t i=0;i<historyEntries_.size();++i)labels.push_back(std::to_string(i)+": "+historyEntries_[i].command);
        history_->setItems(labels);historyIndex_=static_cast<int>(labels.size())-1;history_->setSelectedIndex(historyIndex_);scroll_.scrollTo(0);
        if(onState)onState(currentGame()?state:nlohmann::json{});
        if(operation_=="step")command_.clear();
        message_=state.value("won",false)?"Won!":state.value("lost",false)?"Challenge failed.":recording_?"Recording from the starting state.":"Ready for a command.";
        if(!catalogCompatible_)message_="Catalog mismatch: composer disabled; free-text commands remain available.";
    }
    if(result.contains("warnings"))for(const auto& warning:result["warnings"])message_+=" "+warning.get<std::string>();
    if((operation_=="compile" || operation_=="validate") && sentDocument_!=editor_.document().gameJson())message_="World changed during compilation; compile again.";
}
std::vector<Widget*> UIWorldSession::controls(){
    std::vector<Widget*> result;
    if(challenge_){result={&objective_,predicate_.get(),subject_.get(),target_.get(),conditions_.get()};for(auto& b:challengeButtons_)result.push_back(b.get());}
    else{result={&command_,history_.get(),available_.get()};for(auto& b:playButtons_)result.push_back(b.get());}
    return result;
}
void UIWorldSession::setChallenge(bool value){
    apply();actions_.hide();for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}challenge_=value;refresh();scroll_.scrollTo(0);
}
void UIWorldSession::layout(const PanelRect& area){
    area_=area;const float x=area.origin.x+8,y=area.origin.y,w=area.size.x-16;
    place(objective_,x,y+42,w);place(*predicate_,x,y+100);
    place(*subject_,x,y+154);place(*target_,x,y+208);place(*conditions_,x,y+302);
    place(*challengeButtons_[0],x,y+248,90);place(*challengeButtons_[1],x+98,y+248,90);
    place(*challengeButtons_[2],x+176,y+302,80);place(*challengeButtons_[3],x,y+350,100);place(*challengeButtons_[4],x+108,y+350,110);
    place(*playButtons_[0],x,y+36,140);place(*playButtons_[1],x+148,y+36,70);place(*playButtons_[2],x+226,y+36,65);
    place(*playButtons_[3],x,y+74,75);place(*playButtons_[4],x+83,y+74,115);place(*playButtons_[6],x+206,y+74,100);
    place(*history_,x,y+114);place(command_,x,y+154,std::max(80.f,w-74));place(*playButtons_[5],x+w-66,y+154,66);
    place(*available_,x,y+190,w-92);place(*playButtons_[7],x+w-84,y+190,84);
    const float transcriptY=challenge_?414.f:composerVisible_?390.f:226.f;
    transcript_={{x,y+transcriptY},{w,std::max(24.f,area.size.y-transcriptY-50)}};
    scroll_.setPosition(transcript_.origin);scroll_.setSize(transcript_.size);
}
void UIWorldSession::update(const InputState& input,float dt,const PanelRect& area){
    report([&]{receive();});
    if(editor_.document().json()!=lastDocument_){apply();refresh();}
    layout(area);const auto clipped=input.clipped(area.origin,area.size);
    available_->setVisible(!challenge_ && currentGame() && !done_ && !client_.busy());
    if(!available_->isVisible())available_->collapse();
    auto widgets=controls();Widget* popup=nullptr;
    for(auto* w:widgets)if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isVisible() && d->isExpanded()){popup=w;w->update(clipped,dt);break;}
    for(auto* w:widgets)if(w!=popup && w->isVisible())w->update(clipped,dt);
    if(!challenge_ && composerVisible_)actions_.update(clipped,dt,{{area.origin.x+8,area.origin.y+226},{area.size.x-16,160}},GRIM::UI::InputRouter::get().currentPanel);
    scroll_.update(clipped,dt);
}
void UIWorldSession::hide(){apply();actions_.hide();for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}scroll_.cancelInput();client_.stop();live_=false;recording_=false;if(onState)onState({});}
void UIWorldSession::draw(OverlayRenderer& r,const PanelRect& area){
    using namespace UITheme::Colors;layout(area);r.pushClipRect(area.origin,area.size);r.drawRect(area.origin,area.size,CardSurface);
    const float x=area.origin.x+8,y=area.origin.y;
    r.drawText({x,y+10},challenge_?"Challenge":"Manual playtest",TextPrimary,16);
    if(challenge_){
        r.drawText({x,y+26},"Objective",TextMuted,11);r.drawText({x,y+82},"Condition",TextMuted,12);
        r.drawText({x,y+136},"Subject",TextMuted,12);if(target_->isVisible())r.drawText({x,y+190},"Target",TextMuted,12);
        r.drawText({x,y+284},"Win: ALL conditions / Fail: ANY",TextSecondary,12);
        r.drawText({x,y+390},"Saved solution: "+std::to_string(editor_.document().solution.size())+" commands",TextMuted,12);
    }else{
        r.drawText({x+178,y+121},live_&&!currentGame()?"World changed - recompile":recording_?"REC":"History",live_&&!currentGame()?Warning:TextMuted,11);
    }
        scroll_.drawOverlay(r,{});r.pushClipRect(transcript_.origin,transcript_.size);
        std::string text=details_.empty()?"":details_+"\n\n";
        if(challenge_){
            for(const auto& c:editor_.document().winConditions)text+="Win: "+describe(editor_.document(),c)+"\n";
            for(const auto& c:editor_.document().failConditions)text+="Fail: "+describe(editor_.document(),c)+"\n";
        }else if(historyIndex_>=0 && historyIndex_<static_cast<int>(historyEntries_.size())){
            const auto& e=historyEntries_[historyIndex_];text+="> "+e.command+"\n"+e.feedback+"\n\n"+e.state.value("inventory",std::string{});
            text+="\n\nAvailable commands (author aid):\n";
            for(const auto& command:e.state.value("admissible_commands",std::vector<std::string>{}))text+=command+"\n";
        }
        std::vector<std::string> lines;std::istringstream input(text);std::string logical;
        while(std::getline(input,logical)){
            std::istringstream words(logical);std::string word,line;
            while(words>>word){if(!line.empty() && r.measureTextWidth(line+" "+word,12)>transcript_.size.x-22){lines.push_back(line);line.clear();}if(!line.empty())line+=' ';line+=word;}
            lines.push_back(line);
        }
        scroll_.setContentHeight(static_cast<float>(lines.size())*17+8);
        float top=transcript_.origin.y+4-scroll_.getScrollOffset();for(const auto& line:lines){r.drawText({x+4,top},line,TextSecondary,12);top+=17;}
        r.popClipRect();
    for(auto* w:controls())if(w->isVisible())w->drawOverlay(r,{});
    if(!challenge_ && composerVisible_)actions_.draw(r,{{area.origin.x+8,area.origin.y+226},{area.size.x-16,160}},false);
    for(auto* w:controls())if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isVisible() && d->isExpanded())d->drawExpandedList(r,{});
    r.drawText({x,area.origin.y+area.size.y-34},message_,client_.busy()?Warning:TextSecondary,11);
    r.popClipRect();
}
