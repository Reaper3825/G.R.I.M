#include "ui_world_session.hpp"
#include "world_transcript.hpp"
#include "core/input_parser.hpp"
#include "ui/overlay_renderer.hpp"
#include "ui/input/ui_input_router.hpp"
#include <algorithm>

using namespace GRIM::WorldLab;
namespace {
void place(Widget& w,float x,float y,float width=168,float height=28){w.setPosition(x,y);w.setSize(width,height);}
nlohmann::json sandbox(){return {{"source_id",""},{"name","Sandbox"},{"goal",nullptr}};}
}
UIWorldSession::UIWorldSession(Editor& editor,std::filesystem::path recordsDirectory)
    :editor_(editor),recordsDirectory_(std::move(recordsDirectory)),actions_(editor),selected_(sandbox()) {
    if(recordsDirectory_.empty())recordsDirectory_=std::filesystem::current_path()/"resources/models/games/episodes";
    selected_.erase("name");
    transcriptText_.setReadOnly(true);
    command_.setPlaceholder("Type a game command");
    goalSearch_.setPlaceholder("Find Goal by ID, name, or requirement (first 100)");
    goalSearch_.OnTextSubmitted.Bind([this](const auto&){report([this]{reloadGoals();});});
    command_.OnTextSubmitted.Bind([this](const auto&){request("step");});
    goals_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Sandbox"},0,[this](int i,const auto&){
        report([&]{if(i>=0 && i<static_cast<int>(choices_.size()))selectGoal(choices_[i]);});
    });
    goals_->setFullWidth(true);goals_->setMaxVisibleItems(5);
    history_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[this](int i,const auto&){
        if(i>=0 && i<static_cast<int>(historyEntries_.size())){historyIndex_=i;transcriptText_.scrollToTop();if(onState)onState(historyEntries_[i].state);}
    });
    available_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Live available commands"},0,[this](int i,const auto&){
        if(currentEpisode() && !done_ && i>0 && i<=static_cast<int>(availableCommands_.size()))prepareCommand(availableCommands_[i-1]);
        available_->setSelectedIndex(0);
    });
    available_->setFullWidth(true);available_->setMaxVisibleItems(4);
    actions_.onPrepare=[this](const auto& command){if(!catalogCompatible_)throw std::runtime_error("Action catalog changed; refresh the bundled catalog first");prepareCommand(command);};
    auto goalButton=[&](const auto& name,std::function<void()> fn){challengeButtons_.push_back(std::make_unique<UIButton>(name,[this,fn]{report(fn);}));};
    goalButton("Reload Goals",[this]{reloadGoals();});
    goalButton("Import legacy",[this]{importLegacyGoal();});
    goalButton("Compile / Start",[this]{request(currentGame()?"start_episode":"compile");});
    goalButton("Validate world",[this]{request("validate");});
    auto playButton=[&](const auto& name,std::function<void()> fn){playButtons_.push_back(std::make_unique<UIButton>(name,[this,fn]{report(fn);}));};
    playButton("Compile / Start",[this]{request(currentGame()?"start_episode":"compile");});
    playButton("Reset",[this]{request("reset");});
    playButton("Stop",[this]{client_.stop();live_=recording_=false;message_="Stopping worker...";if(onState)onState({});});
    playButton("Record",[this]{request("record");});
    playButton("Save solution",[this]{request("verify_solution");});
    playButton("Send",[this]{request("step");});
    playButton("Check saved",[this]{request("check_saved");});
    playButton("Actions",[this]{composerVisible_=!composerVisible_;actions_.hide();});
    choices_={sandbox()};refresh();
}
void UIWorldSession::report(const std::function<void()>& action){try{action();}catch(const std::exception& e){message_=e.what();}}
void UIWorldSession::selectGoal(nlohmann::json selection){
    if(client_.busy())throw std::runtime_error("Wait for the current request before changing Goal");
    if(!selection.is_object() || !selection.contains("goal") || !selection.at("source_id").is_string())throw std::runtime_error("Invalid Goal selection");
    if(!selection["goal"].is_null()){
        if(selection.at("source_id").get<std::string>().empty())throw std::runtime_error("Selected Goal requires a ConceptBlock ID");
        selection["goal"]=GRIM::goalToJson(GRIM::goalFromJson(selection["goal"]));
    }
    selection.erase("name"); // Display labels do not affect episode compatibility.
    selected_=std::move(selection);recording_=false;
    message_=currentGame()?"Goal selected. Compile / Start begins a fresh episode without recompiling.":"Goal selected. Compile / Start to play.";
    if(onState)onState({});
}
void UIWorldSession::reloadGoals(){
    if(client_.busy())throw std::runtime_error("Wait for the current request before reloading Goals");
    std::vector<nlohmann::json> choices{sandbox()};
    if(access_.list){auto values=access_.list(goalSearch_.getText());choices.insert(choices.end(),values.begin(),values.end());}
    else throw std::runtime_error("ConceptBlock database is unavailable");
    std::vector<std::string> labels;int selected=0;
    for(size_t i=0;i<choices.size();++i){
        const auto& value=choices[i];
        if(value.at("source_id")==selected_.at("source_id"))selected=static_cast<int>(i);
        labels.push_back(value.value("name",value.at("source_id").get<std::string>())+" ["+value.at("source_id").get<std::string>()+"]");
    }
    choices_=std::move(choices);goals_->setItems(labels);goals_->setSelectedIndex(selected);
    // Reloading the database does not silently replace a running Goal snapshot.
    message_="Goals refreshed. Select a Goal to load its snapshot.";
}
void UIWorldSession::importLegacyGoal(){
    if(client_.busy())throw std::runtime_error("Wait for the current request before importing");
    if(!hasLegacyChallenge(editor_.document()))throw std::runtime_error("This world has no legacy challenge to import");
    if(!access_.import)throw std::runtime_error("ConceptBlock database import is unavailable");
    const auto selection=legacyGoal(editor_.document());
    const auto record=nlohmann::json{{"format","grim-episode"},{"version",1},{"selection",selection},
        {"environment",editor_.document().gameJson()},{"commands",editor_.document().solution},{"verified",false}};
    std::filesystem::create_directories(recordsDirectory_);
    // Preserve the old solution before clearing any document metadata. Both
    // writes are idempotent; a failed database save leaves the legacy world intact.
    saveJsonFile(recordsDirectory_/("legacy-"+recordKey(selection.at("source_id"))+".grimepisode.json"),record);
    access_.import(selection);
    editor_.change([](Document& d){d.objective.clear();d.winConditions.clear();d.failConditions.clear();d.solution.clear();});
    goalSearch_.setText(selection.at("source_id").get<std::string>());
    reloadGoals();selectGoal(selection);refresh();
    message_="Imported ConceptBlock Goal and archived solution. Save the world to persist its environment-only format.";
}
void UIWorldSession::refresh(){lastDocument_=editor_.document().json();if(live_ && !currentGame() && onState)onState({});}
bool UIWorldSession::currentGame() const{return live_ && compiled_==editor_.document().gameJson();}
bool UIWorldSession::currentEpisode() const{return currentGame() && active_==selected_;}
std::filesystem::path UIWorldSession::recordPath() const {
    return recordsDirectory_/(recordKey(episode_.at("identity").dump())+".grimepisode.json");
}
void UIWorldSession::saveWitness(const nlohmann::json& result){
    if(selected_.at("goal").is_null())throw std::runtime_error("Sandbox has no verified Goal solution");
    std::filesystem::create_directories(recordsDirectory_);
    saveJsonFile(recordPath(),{{"format","grim-episode"},{"version",1},{"selection",active_},
        {"environment",compiled_},{"commands",result.at("commands")},{"witness",result.at("witness")},{"verified",true}});
}
void UIWorldSession::request(const std::string& op){report([&]{
    if(client_.busy())throw std::runtime_error("Wait for the current request or press Stop");
    if(op!="compile" && op!="validate" && !currentGame())throw std::runtime_error("Compile & Play the current world first");
    if(op!="compile" && op!="validate" && op!="start_episode" && !currentEpisode())throw std::runtime_error("Start the selected Goal first");
    nlohmann::json request{{"op",op}};
    sentGoal_=selected_;
    if(op=="compile" || op=="validate"){
        sentDocument_=editor_.document().gameJson();request["document"]=sentDocument_;
    }
    if(op=="compile" || op=="validate" || op=="start_episode"){
        request["goal"]=selected_.at("goal");request["source_id"]=selected_.at("source_id");
    }
    if(op=="step"){
        if(done_)throw std::runtime_error("Episode finished; reset to play again");
        sentCommand_=command_.getText();if(sentCommand_.empty())return;request["command"]=sentCommand_;
    }
    if(op=="record")request["op"]="reset";
    if(op=="verify_solution"){
        if(!recording_)throw std::runtime_error("Press Record to reset and record a solution first");
        request["commands"]=recorded_;
    }
    if(op=="check_saved"){
        request["op"]="verify_solution";
        const auto saved=recordPath();
        const auto legacy=recordsDirectory_/("legacy-"+recordKey(selected_.at("source_id"))+".grimepisode.json");
        const auto record=readEpisodeRecord(std::filesystem::exists(saved)?saved:legacy);
        auto recordSelection=record.at("selection");recordSelection.erase("name");
        if(record.at("environment")!=compiled_ || recordSelection!=active_)throw std::runtime_error("Saved solution is stale for this environment or Goal");
        request["commands"]=record.at("commands");if(record.contains("witness"))request["witness"]=record.at("witness");
    }
    if(client_.begin(std::move(request))){operation_=op;details_.clear();message_="Working: "+op+"...";}
});}
void UIWorldSession::receive(){
    auto response=client_.poll();if(!response)return;
    if(!response->value("ok",false)){
        if(response->value("transport_error",false)){live_=recording_=false;if(onState)onState({});}
        details_=response->value("error",std::string{"TextWorld request failed"});
        if(!response->value("collection",std::string{}).empty())details_+=" ["+response->value("source_id",std::string{})+" / "+response->at("collection").get<std::string>()+" / "+std::to_string(response->value("index",-1))+"]";
        message_="Request failed. See details below.";
        const auto id=response->value("entity_id",std::string{});if(!id.empty() && onErrorEntity)onErrorEntity(id);return;
    }
    const auto& result=response->at("result");
    if(result.value("stopped",false)){live_=false;message_="Worker stopped.";return;}
    if(operation_=="validate")message_="World compiled/reset and Goal bound. Achievability: "+result.at("episode").at("achievability").get<std::string>()+".";
    if(operation_=="verify_solution" || operation_=="check_saved"){
        if(!currentEpisode()){message_="Environment or Goal changed during replay; start and verify again.";return;}
        saveWitness(result);episode_["achievability"]="achievable";recording_=false;message_="Solution verified and saved for this environment/Goal pairing.";
    }
    if(result.contains("state")){
        if(operation_=="compile"){
            catalogCompatible_=result.contains("action_catalog") && result.at("action_catalog")==actionCatalog();
            compiled_=sentDocument_;live_=true;
        }
        if(operation_=="compile" || operation_=="start_episode")active_=sentGoal_;
        if(operation_=="compile" || operation_=="start_episode" || operation_=="reset" || operation_=="record"){
            historyEntries_.clear();recorded_.clear();recording_=operation_=="record";
        }
        if(operation_=="step" && recording_)recorded_.push_back(sentCommand_);
        const auto& state=result.at("state");done_=state.at("done");episode_=state.at("episode");
        availableCommands_=state.value("admissible_commands",std::vector<std::string>{});
        std::vector<std::string> availableLabels{"Live available commands"};
        if(!done_)availableLabels.insert(availableLabels.end(),availableCommands_.begin(),availableCommands_.end());
        available_->collapse();available_->setItems(availableLabels);available_->setSelectedIndex(0);
        std::string feedback=cleanTranscript(state.value("feedback",std::string{}));
        if(operation_!="step"){
            const auto description=cleanTranscript(state.value("description",std::string{}));
            if(!description.empty())feedback=description;
            if(active_.at("goal").is_object())feedback=active_.at("goal").value("target_state",std::string{})+"\n\n"+feedback;
        }
        historyEntries_.push_back({operation_=="step"?sentCommand_:"Start",std::move(feedback),state});
        std::vector<std::string> labels;for(size_t i=0;i<historyEntries_.size();++i)labels.push_back(std::to_string(i)+": "+historyEntries_[i].command);
        history_->setItems(labels);historyIndex_=static_cast<int>(labels.size())-1;history_->setSelectedIndex(historyIndex_);transcriptText_.scrollToTop();
        if(onState)onState(currentEpisode()?state:nlohmann::json{});
        if(operation_=="step")command_.clear();
        const auto status=episode_.at("status").get<std::string>();
        message_=status=="succeeded"?"Won!":status=="failed"?"Goal failed.":status=="stopped"?"Episode stopped without verified Goal success.":recording_?"Recording from the starting state.":"Ready for a command.";
        if(!catalogCompatible_)message_+=" Catalog mismatch: composer disabled.";
    }
    if((operation_=="compile" || operation_=="validate") && sentDocument_!=editor_.document().gameJson())message_="World changed during compilation; compile again.";
}
std::vector<Widget*> UIWorldSession::controls(){
    std::vector<Widget*> result;
    if(challenge_){result={&goalSearch_,goals_.get()};for(auto& b:challengeButtons_)result.push_back(b.get());}
    else{result={&command_,history_.get(),available_.get()};for(auto& b:playButtons_)result.push_back(b.get());}
    result.push_back(&transcriptText_);return result;
}
void UIWorldSession::setChallenge(bool value){
    actions_.hide();for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}
    challenge_=value;refresh();transcriptText_.scrollToTop();
    if(value && access_.list)report([this]{reloadGoals();});
}
void UIWorldSession::layout(const PanelRect& area){
    area_=area;const float x=area.origin.x+8,y=area.origin.y,w=area.size.x-16;
    place(goalSearch_,x,y+38,w);place(*goals_,x,y+74,w);
    place(*challengeButtons_[0],x,y+110,130);place(*challengeButtons_[1],x+138,y+110,130);
    place(*challengeButtons_[2],x,y+146,140);place(*challengeButtons_[3],x+148,y+146,140);
    place(*playButtons_[0],x,y+36,140);place(*playButtons_[1],x+148,y+36,70);place(*playButtons_[2],x+226,y+36,65);
    place(*playButtons_[3],x,y+74,75);place(*playButtons_[4],x+83,y+74,115);place(*playButtons_[6],x+206,y+74,100);
    place(*history_,x,y+114);place(command_,x,y+154,std::max(80.f,w-74));place(*playButtons_[5],x+w-66,y+154,66);
    place(*available_,x,y+190,w-92);place(*playButtons_[7],x+w-84,y+190,84);
    const float transcriptY=challenge_?186.f:composerVisible_?390.f:226.f;
    transcript_={{x,y+transcriptY},{w,std::max(24.f,area.size.y-transcriptY-50)}};
    transcriptText_.setPosition(transcript_.origin);transcriptText_.setSize(transcript_.size);
}
void UIWorldSession::update(const InputState& input,float dt,const PanelRect& area){
    report([&]{receive();});if(editor_.document().json()!=lastDocument_)refresh();
    layout(area);const auto clipped=input.clipped(area.origin,area.size);
    available_->setVisible(!challenge_ && currentEpisode() && !done_ && !client_.busy());if(!available_->isVisible())available_->collapse();
    auto widgets=controls();Widget* popup=nullptr;
    for(auto* w:widgets)if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isVisible() && d->isExpanded()){popup=w;w->update(clipped,dt);break;}
    for(auto* w:widgets)if(w!=popup && w->isVisible())w->update(clipped,dt);
    if(!challenge_ && composerVisible_)actions_.update(clipped,dt,{{area.origin.x+8,area.origin.y+226},{area.size.x-16,160}},GRIM::UI::InputRouter::get().currentPanel);
}
void UIWorldSession::hide(){actions_.hide();for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}client_.stop();live_=recording_=false;if(onState)onState({});}
void UIWorldSession::draw(OverlayRenderer& r,const PanelRect& area){
    using namespace UITheme::Colors;layout(area);r.pushClipRect(area.origin,area.size);r.drawRect(area.origin,area.size,CardSurface);
    const float x=area.origin.x+8,y=area.origin.y;
    r.drawText({x,y+10},challenge_?"ConceptBlock Goal":"Manual playtest",TextPrimary,16);
    std::string text=details_.empty()?"":details_+"\n\n";
    if(challenge_){
        text+="Selected: "+selected_.at("source_id").get<std::string>()+"\n";
        if(!selected_.at("goal").is_null()){
            const auto& goal=selected_.at("goal");text+=goal.value("target_state",std::string{})+"\n\n";
            for(const auto& c:goal.at("success_criteria"))text+="Success: "+c.at("criterion").get<std::string>()+"\n";
            for(const auto& c:goal.at("constraints"))text+="Constraint: "+c.at("constraint").get<std::string>()+"\n";
        }else text+="Sandbox exploration; no automatic Goal success.\n";
        if(hasLegacyChallenge(editor_.document()))text+="\nLegacy conditions are inactive. Import legacy to create a database Goal.\n";
        if(currentEpisode() && episode_.is_object()){
            text+="\nCoverage: "+episode_.value("coverage",std::string{})+" / achievability: "+episode_.value("achievability",std::string{})+"\n";
            for(const auto& check:episode_.at("requirements")){
                const bool constraint=check.at("collection")=="constraints";
                const auto status=check.at("status").get<std::string>();
                const auto label=constraint && status!="unverified"?(status=="satisfied"?"VIOLATION":"not violated"):status;
                text+=(constraint?"Constraint ":"Success criterion ")+std::to_string(check.at("index").get<int>()+1)+": "+label+"\n";
            }
        }
    }else if(historyIndex_>=0 && historyIndex_<static_cast<int>(historyEntries_.size())){
        const auto& e=historyEntries_[historyIndex_];text+="> "+e.command+"\n"+e.feedback+"\n\n"+cleanTranscript(e.state.value("inventory",std::string{}));
        text+="\n\nAvailable commands (author aid):\n";for(const auto& command:e.state.value("admissible_commands",std::vector<std::string>{}))text+=command+"\n";
    }
    transcriptText_.setText(text);for(auto* w:controls())if(w->isVisible())w->drawOverlay(r,{});
    if(!challenge_ && composerVisible_)actions_.draw(r,{{area.origin.x+8,area.origin.y+226},{area.size.x-16,160}},false);
    for(auto* w:controls())if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isVisible() && d->isExpanded())d->drawExpandedList(r,{});
    r.drawText({x,area.origin.y+area.size.y-34},message_,client_.busy()?Warning:TextSecondary,11);r.popClipRect();
}
