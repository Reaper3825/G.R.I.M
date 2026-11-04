#include "ui_world_session.hpp"
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
    curriculums_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Select curriculum"},0,[this](int i,const auto&){
        report([&]{selectCurriculum(i-1);});curriculums_->setSelectedIndex(curriculumIndex_+1);
    });
    courses_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Select course"},0,[this](int i,const auto&){
        report([&]{selectCourse(i-1);});courses_->setSelectedIndex(courseIndex_+1);
    });
    for(auto* dropdown:{curriculums_.get(),courses_.get()}){dropdown->setFullWidth(true);dropdown->setMaxVisibleItems(5);}
    previousBlock_=std::make_unique<UIButton>("Back",[this]{report([&]{moveConceptBlock(-1);});});
    nextBlock_=std::make_unique<UIButton>("Next",[this]{report([&]{moveConceptBlock(1);});});
    history_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[this](int i,const auto&){
        report([&]{selectHistory(i);});
    });
    historyView_=std::make_unique<UIButton>("Host details",[this]{
        hostDetailsVisible_=!hostDetailsVisible_;
        historyView_->setText(hostDetailsVisible_?"Trajectory":"Host details");
        transcriptText_.scrollToTop();
    });
    available_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Live available commands"},0,[this](int i,const auto&){
        if(currentEpisode() && !done_ && i>0 && i<=static_cast<int>(availableCommands_.size()))prepareCommand(availableCommands_[i-1]);
        available_->setSelectedIndex(0);
    });
    available_->setFullWidth(true);available_->setMaxVisibleItems(4);
    actions_.onPrepare=[this](const auto& command){if(!catalogCompatible_)throw std::runtime_error("Action catalog changed; refresh the bundled catalog first");prepareCommand(command);};
    auto goalButton=[&](const auto& name,std::function<void()> fn){challengeButtons_.push_back(std::make_unique<UIButton>(name,[this,fn]{report(fn);}));};
    goalButton("Reload Goals",[this]{if(access_.registry)reloadRegistry();reloadGoals();});
    goalButton("Import legacy",[this]{importLegacyGoal();});
    goalButton("Compile / Start",[this]{request(currentGame()?"start_episode":"compile");});
    goalButton("Validate world",[this]{request("validate");});
    auto playButton=[&](const auto& name,std::function<void()> fn){playButtons_.push_back(std::make_unique<UIButton>(name,[this,fn]{report(fn);}));};
    playButton("Compile / Start",[this]{request(currentGame()?"start_episode":"compile");});
    playButton("Reset",[this]{request("reset");});
    playButton("Stop",[this]{client_.stop();live_=recording_=false;initialModelInput_.reset();continuationStatus_="terminated: stopped";message_="Stopping worker...";if(onState)onState({});});
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
    if(selection.contains("prompt") && !selection.at("prompt").is_string())throw std::runtime_error("ConceptBlock prompt must be text");
    if(!selection["goal"].is_null()){
        if(selection.at("source_id").get<std::string>().empty())throw std::runtime_error("Selected Goal requires a ConceptBlock ID");
        selection["goal"]=GRIM::goalToJson(GRIM::goalFromJson(selection["goal"]));
    }
    selection.erase("name"); // Display labels do not affect episode compatibility.
    selected_=std::move(selection);recording_=false;blockIndex_=-1;blockName_.clear();
    initialModelInput_.reset();clearHistory();
    refreshGoalChoices();
    message_=currentGame()?"Goal selected. Compile / Start begins a fresh episode without recompiling.":"Goal selected. Compile / Start to play.";
    if(onState)onState({});
}
void UIWorldSession::reloadRegistry(){
    if(client_.busy())throw std::runtime_error("Wait for the current request before reloading curricula");
    if(!access_.registry)throw std::runtime_error("Curriculum registry is unavailable");
    auto registry=access_.registry();
    if(!registry.at("curriculums").is_array() || !registry.at("courses").is_array())
        throw std::runtime_error("Invalid curriculum registry");
    std::vector<std::string> labels{"Select curriculum"};
    for(const auto& c:registry.at("curriculums"))labels.push_back(c.at("name").get<std::string>());
    registry_=std::move(registry);curriculums_->setItems(labels);
    curriculumIndex_=courseIndex_=blockIndex_=-1;courseChoices_.clear();courseBlockIds_.clear();blockName_.clear();
    curriculums_->collapse();curriculums_->setSelectedIndex(0);
    courses_->collapse();courses_->setItems({"Select course"});courses_->setSelectedIndex(0);
    message_="Curricula refreshed. Select a curriculum and course.";
}
void UIWorldSession::selectCurriculum(int index){
    if(client_.busy())throw std::runtime_error("Wait for the current request before changing curriculum");
    std::vector<nlohmann::json> choices;
    std::vector<std::string> labels{"Select course"};
    if(index>=0){
        const auto& curriculum=registry_.at("curriculums").at(index);
        for(const auto& id:curriculum.at("course_ids")){
            const auto& all=registry_.at("courses");
            const auto found=std::find_if(all.begin(),all.end(),[&](const auto& c){return c.at("id")==id;});
            if(found==all.end())throw std::runtime_error("Registered course is missing: "+id.get<std::string>());
            choices.push_back(*found);labels.push_back(found->at("name").get<std::string>());
        }
    }
    selectGoal(sandbox());
    curriculumIndex_=index;courseIndex_=blockIndex_=-1;courseBlockIds_.clear();
    courseChoices_=std::move(choices);courses_->collapse();courses_->setItems(labels);courses_->setSelectedIndex(0);
    curriculums_->setSelectedIndex(index+1);
    message_=index<0?"Select a curriculum.":courseChoices_.empty()?"This curriculum has no courses.":"Select a course to browse its ConceptBlocks.";
}
void UIWorldSession::selectCourse(int index){
    if(client_.busy())throw std::runtime_error("Wait for the current request before changing course");
    std::vector<std::string> ids;
    if(index>=0)ids=courseChoices_.at(index).at("concept_block_ids").get<std::vector<std::string>>();
    nlohmann::json selection=sandbox();
    if(!ids.empty()){
        if(!access_.byId)throw std::runtime_error("Exact ConceptBlock lookup is unavailable");
        selection=access_.byId(ids.front());
        if(selection.at("source_id")!=ids.front())throw std::runtime_error("ConceptBlock lookup returned a different ID");
    }
    const auto name=selection.value("name",std::string{});
    selectGoal(std::move(selection));
    courseIndex_=index;courseBlockIds_=std::move(ids);blockIndex_=courseBlockIds_.empty()?-1:0;blockName_=name;
    courses_->setSelectedIndex(index+1);
    if(courseBlockIds_.empty())message_=index<0?"Select a course.":"This course has no ConceptBlocks.";
}
void UIWorldSession::selectCourseBlock(int index){
    if(index<0 || index>=static_cast<int>(courseBlockIds_.size()))throw std::runtime_error("End of course; no further ConceptBlock in that direction");
    if(!access_.byId)throw std::runtime_error("Exact ConceptBlock lookup is unavailable");
    auto selection=access_.byId(courseBlockIds_[index]);
    if(selection.at("source_id")!=courseBlockIds_[index])throw std::runtime_error("ConceptBlock lookup returned a different ID");
    const auto name=selection.value("name",std::string{});
    selectGoal(std::move(selection));blockIndex_=index;blockName_=name;
}
void UIWorldSession::moveConceptBlock(int delta){
    if(client_.busy())throw std::runtime_error("Wait for the current request before changing ConceptBlock");
    if(courseBlockIds_.empty())throw std::runtime_error("Select a nonempty course first");
    selectCourseBlock(blockIndex_<0?(delta<0?static_cast<int>(courseBlockIds_.size())-1:0):blockIndex_+delta);
}
void UIWorldSession::reloadGoals(){
    if(client_.busy())throw std::runtime_error("Wait for the current request before reloading Goals");
    std::vector<nlohmann::json> choices{sandbox()};
    if(access_.list){auto values=access_.list(goalSearch_.getText());choices.insert(choices.end(),values.begin(),values.end());}
    else throw std::runtime_error("ConceptBlock database is unavailable");
    choices_=std::move(choices);searchChoiceCount_=choices_.size();refreshGoalChoices();
    // Reloading the database does not silently replace a running Goal snapshot.
    message_="Goals refreshed. Select a Goal to load its snapshot.";
}
void UIWorldSession::refreshGoalChoices(){
    choices_.resize(std::min(searchChoiceCount_,choices_.size()));
    if(std::none_of(choices_.begin(),choices_.end(),[&](const auto& choice){return choice.at("source_id")==selected_.at("source_id");}))
        choices_.push_back(selected_);
    std::vector<std::string> labels;int selected=0;
    for(size_t i=0;i<choices_.size();++i){
        const auto& value=choices_[i];
        if(value.at("source_id")==selected_.at("source_id"))selected=static_cast<int>(i);
        labels.push_back(value.value("name",value.at("source_id").get<std::string>())+" ["+value.at("source_id").get<std::string>()+"]");
    }
    goals_->collapse();goals_->setItems(labels);goals_->setSelectedIndex(selected);
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
void UIWorldSession::clearHistory(){
    historyEntries_.clear();trajectory_.clear();pendingExecute_.clear();historyIndex_=-1;
    history_->collapse();history_->setItems({});continuationStatus_="not_started";
}
void UIWorldSession::appendHistory(HistoryEntry entry){
    historyEntries_.push_back(std::move(entry));
    std::vector<std::string> labels;
    for(std::size_t i=0;i<historyEntries_.size();++i)
        labels.push_back(std::to_string(i)+": "+historyEntries_[i].command+" ["+historyEntries_[i].continuationStatus+"]");
    history_->setItems(labels);selectHistory(static_cast<int>(historyEntries_.size())-1);
}
void UIWorldSession::selectHistory(int index){
    if(index<0 || index>=static_cast<int>(historyEntries_.size()))throw std::out_of_range("History boundary is unavailable");
    historyIndex_=index;history_->setSelectedIndex(index);transcriptText_.scrollToTop();
    if(onState)onState(currentEpisode()?historyEntries_[index].state:nlohmann::json{});
}
std::string UIWorldSession::continuationStatus() const {
    if(continuationStatus_=="starting" || continuationStatus_=="not_started")return continuationStatus_;
    return currentEpisode() || !live_ ? continuationStatus_ : "unavailable: environment or Goal changed";
}
std::string UIWorldSession::historyHostDetails(std::size_t index) const {
    const auto& e=historyEntries_.at(index);
    return "Boundary status: "+e.continuationStatus+"\nLive status: "+continuationStatus()+
        "\n\nCommand: "+e.command+"\n\nDisclosed observation\n"+e.observation+
        "\n\nHost receipt (private)\n"+e.receipt.dump(2);
}
void UIWorldSession::refresh(){lastDocument_=editor_.document().json();if(live_ && !currentGame() && onState)onState({});}
bool UIWorldSession::currentGame() const{return live_ && compiled_==editor_.document().gameJson();}
bool UIWorldSession::currentEpisode() const{return currentGame() && initialModelInput_ && !historyEntries_.empty() && active_==selected_;}
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
    const bool beginsAttempt=op=="compile" || op=="start_episode" || op=="reset" || op=="record";
    if(!beginsAttempt && op!="validate" && !currentEpisode())throw std::runtime_error("Start the selected Goal first");
    nlohmann::json request{{"op",op}};
    sentGoal_=selected_;
    if(op=="compile" || op=="validate"){
        sentDocument_=editor_.document().gameJson();request["document"]=sentDocument_;
    }
    if(beginsAttempt && !access_.renderInput)throw std::runtime_error("Model-input preview renderer is unavailable");
    if(beginsAttempt || op=="validate"){
        request["goal"]=selected_.at("goal");request["source_id"]=selected_.at("source_id");
    }
    if(op=="step"){
        if(done_)throw std::runtime_error("Episode finished; reset to play again");
        sentCommand_=command_.getText();if(sentCommand_.empty())return;request["command"]=sentCommand_;
        if(!access_.renderInput)throw std::runtime_error("Model-input preview renderer is unavailable");
        pendingExecute_=access_.renderInput({{"execute",sentCommand_}});
        if(pendingExecute_.empty())throw std::runtime_error("Execute span is missing or ignored in the active configuration");
    }
    // Rebind the selected frozen Goal and reset the world as one operation.
    // Worker reset alone intentionally retains its previous bound Goal.
    if(op=="reset" || op=="record")request["op"]="start_episode";
    if(op=="verify_solution"){
        if(!recording_)throw std::runtime_error("Press Record to reset and record a solution first");
        request["commands"]=recorded_;
    }
    if(op=="check_saved"){
        request["op"]="verify_solution";
        const auto saved=recordPath();
        const auto legacy=recordsDirectory_/("legacy-"+recordKey(selected_.at("source_id"))+".grimepisode.json");
        const auto record=readEpisodeRecord(std::filesystem::exists(saved)?saved:legacy);
        auto recordSelection=record.at("selection");recordSelection.erase("name");recordSelection.erase("prompt");
        auto activeBinding=active_;activeBinding.erase("prompt");
        if(record.at("environment")!=compiled_ || recordSelection!=activeBinding)throw std::runtime_error("Saved solution is stale for this environment or Goal");
        request["commands"]=record.at("commands");if(record.contains("witness"))request["witness"]=record.at("witness");
    }
    if(client_.begin(std::move(request))){operation_=op;
        if(op=="step"){
            trajectory_+=pendingExecute_;pendingExecute_.clear();continuationStatus_="awaiting_action_result";
            // Keep the last authoritative live observation even when inspecting an older boundary.
            const auto state=historyEntries_.back().state;
            appendHistory({sentCommand_,"",continuationStatus_,state,
                {{"status","pending"},{"command",sentCommand_}},trajectory_.size()});
        }
        else if(beginsAttempt)continuationStatus_="starting";
        details_.clear();message_="Working: "+op+"...";}
});}
void UIWorldSession::receive(){
    auto response=client_.poll();if(!response)return;
    try{
        if(!response->value("ok",false)){
            if(response->value("transport_error",false)){
                live_=recording_=false;initialModelInput_.reset();
                if(continuationStatus_!="terminated: stopped")continuationStatus_="terminated: infrastructure_error";
                if(onState)onState({});
            }else if(operation_=="step")continuationStatus_="action_rejected";
            else if(continuationStatus_=="starting")continuationStatus_=historyEntries_.empty()?"not_started":historyEntries_.back().continuationStatus;
            if(operation_=="step" && !historyEntries_.empty()){
                const auto state=historyEntries_.back().state;
                appendHistory({sentCommand_,"",continuationStatus_,state,*response,trajectory_.size()});
            }
            details_=response->value("error",std::string{"TextWorld request failed"});
            if(!response->value("collection",std::string{}).empty())details_+=" ["+response->value("source_id",std::string{})+" / "+response->at("collection").get<std::string>()+" / "+std::to_string(response->value("index",-1))+"]";
            message_="Request failed. See details below.";
            const auto id=response->value("entity_id",std::string{});if(!id.empty() && onErrorEntity)onErrorEntity(id);return;
        }
        const auto& result=response->at("result");
        if(result.value("stopped",false)){
            live_=false;
            if(continuationStatus_!="terminated: infrastructure_error"){
                continuationStatus_="terminated: stopped";message_="Worker stopped.";
            }
            return;
        }
        if(operation_=="validate")message_="World compiled/reset and Goal bound. Achievability: "+result.at("episode").at("achievability").get<std::string>()+".";
        if(operation_=="verify_solution" || operation_=="check_saved"){
            if(!currentEpisode()){message_="Environment or Goal changed during replay; start and verify again.";return;}
            saveWitness(result);episode_["achievability"]="achievable";recording_=false;message_="Solution verified and saved for this environment/Goal pairing.";
        }
        if(result.contains("state")){
            // Only live start/action responses extend history. Validation and independent
            // witness replay must not create a fake action or observation boundary.
            const bool beginsAttempt=operation_=="compile" || operation_=="start_episode" || operation_=="reset" || operation_=="record";
            if(!beginsAttempt && operation_!="step")return;
            const auto goal=sentGoal_.at("goal").is_null()?GRIM::ConceptBlockGoal{}:GRIM::goalFromJson(sentGoal_.at("goal"));
            const auto input=GRIM::Environment::buildEnvironmentInferenceInput(goal,*response,
                beginsAttempt?sentGoal_.value("prompt",std::string{}):std::string{});
            const auto observation=access_.renderInput({{"environment_context",input.reasoning_state.at("environment_context")}});
            if(observation.empty() && !input.reasoning_state.at("environment_context").get<std::string>().empty())
                throw std::runtime_error("Environment context span is missing or ignored in the active configuration");
            std::string modelInputText;
            if(operation_=="compile"){
                catalogCompatible_=result.contains("action_catalog") && result.at("action_catalog")==actionCatalog();
                compiled_=sentDocument_;live_=true;
            }
            if(beginsAttempt){
                auto supplied=input.reasoning_state;supplied["prompt"]=input.prompt;
                modelInputText=access_.renderInput(supplied);
                active_=sentGoal_;initialModelInput_=input;
                clearHistory();trajectory_=std::move(modelInputText);
                recorded_.clear();recording_=operation_=="record";command_.clear();
            }else{
                trajectory_+="\n\n"+observation+"\n\n";
            }
            if(operation_=="step" && recording_)recorded_.push_back(sentCommand_);
            const auto& state=result.at("state");done_=state.at("done");episode_=state.at("episode");
            availableCommands_=state.value("admissible_commands",std::vector<std::string>{});
            std::vector<std::string> availableLabels{"Live available commands"};
            if(!done_)availableLabels.insert(availableLabels.end(),availableCommands_.begin(),availableCommands_.end());
            available_->collapse();available_->setItems(availableLabels);available_->setSelectedIndex(0);
            continuationStatus_=done_?"world_terminal: "+input.episode_status+"; awaiting_answer":"ready_for_command";
            appendHistory({operation_=="step"?sentCommand_:"Start",observation,continuationStatus_,state,*response,trajectory_.size()});
            if(operation_=="step")command_.clear();
            const auto status=episode_.at("status").get<std::string>();
            message_=status=="succeeded"?"Won!":status=="failed"?"Goal failed.":status=="stopped"?"Episode stopped without verified Goal success.":recording_?"Recording from the starting state.":"Ready for a command.";
            if(!catalogCompatible_)message_+=" Catalog mismatch: composer disabled.";
        }
        if((operation_=="compile" || operation_=="validate") && sentDocument_!=editor_.document().gameJson())message_="World changed during compilation; compile again.";
    }catch(const std::exception& error){
        // Preserve the actual receipt even if projection/rendering fails after dispatch.
        client_.stop();live_=recording_=false;initialModelInput_.reset();
        continuationStatus_="terminated: infrastructure_error";
        if(operation_=="step" && !historyEntries_.empty()){
            const auto state=historyEntries_.back().state;
            const bool completed=response->is_object() && response->contains("ok") && response->at("ok")==true &&
                response->contains("result") && response->at("result").contains("state");
            appendHistory({sentCommand_,"",continuationStatus_,state,
                {{"status",completed?"completed":"unknown"},{"response",*response},{"error",error.what()}},trajectory_.size()});
        }
        if(onState)onState({});throw;
    }
}
std::vector<Widget*> UIWorldSession::controls(){
    std::vector<Widget*> result;
    if(challenge_){result={&goalSearch_,goals_.get()};for(auto& b:challengeButtons_)result.push_back(b.get());}
    else{result={&command_,history_.get(),historyView_.get(),available_.get()};for(auto& b:playButtons_)result.push_back(b.get());}
    result.insert(result.end(),{curriculums_.get(),courses_.get(),previousBlock_.get(),nextBlock_.get()});
    result.push_back(&transcriptText_);return result;
}
void UIWorldSession::setChallenge(bool value){
    actions_.hide();for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}
    challenge_=value;refresh();transcriptText_.scrollToTop();
    if(registry_.is_null() && access_.registry)report([this]{reloadRegistry();});
    if(value && access_.list)report([this]{reloadGoals();});
}
void UIWorldSession::layout(const PanelRect& area){
    area_=area;const float x=area.origin.x+8,y=area.origin.y+116,w=area.size.x-16;
    place(*curriculums_,x,area.origin.y+36,w);
    place(*courses_,x,area.origin.y+72,std::max(80.f,w-144));
    place(*previousBlock_,x+w-136,area.origin.y+72,64);place(*nextBlock_,x+w-64,area.origin.y+72,64);
    place(goalSearch_,x,y+38,w);place(*goals_,x,y+74,w);
    place(*challengeButtons_[0],x,y+110,130);place(*challengeButtons_[1],x+138,y+110,130);
    place(*challengeButtons_[2],x,y+146,140);place(*challengeButtons_[3],x+148,y+146,140);
    place(*playButtons_[0],x,y+36,140);place(*playButtons_[1],x+148,y+36,70);place(*playButtons_[2],x+226,y+36,65);
    place(*playButtons_[3],x,y+74,75);place(*playButtons_[4],x+83,y+74,115);place(*playButtons_[6],x+206,y+74,100);
    place(*history_,x,y+114,std::max(80.f,w-116));place(*historyView_,x+w-108,y+114,108);place(command_,x,y+154,std::max(80.f,w-74));place(*playButtons_[5],x+w-66,y+154,66);
    place(*available_,x,y+190,w-92);place(*playButtons_[7],x+w-84,y+190,84);
    const float transcriptY=challenge_?186.f:composerVisible_?390.f:226.f;
    transcript_={{x,y+transcriptY},{w,std::max(24.f,area.size.y-transcriptY-166)}};
    transcriptText_.setPosition(transcript_.origin);transcriptText_.setSize(transcript_.size);
}
void UIWorldSession::update(const InputState& input,float dt,const PanelRect& area){
    report([&]{receive();});if(editor_.document().json()!=lastDocument_)refresh();
    layout(area);const auto clipped=input.clipped(area.origin,area.size);
    available_->setVisible(!challenge_ && currentEpisode() && !done_ && !client_.busy());if(!available_->isVisible())available_->collapse();
    auto widgets=controls();Widget* popup=nullptr;
    for(auto* w:widgets)if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isVisible() && d->isExpanded()){popup=w;w->update(clipped,dt);break;}
    for(auto* w:widgets)if(w!=popup && w->isVisible())w->update(clipped,dt);
    if(!challenge_ && composerVisible_)actions_.update(clipped,dt,{{area.origin.x+8,area.origin.y+342},{area.size.x-16,160}},GRIM::UI::InputRouter::get().currentPanel);
}
void UIWorldSession::hide(){actions_.hide();for(auto* w:controls()){w->cancelInput();if(auto* d=dynamic_cast<UIDropdown*>(w))d->collapse();}client_.stop();live_=recording_=false;initialModelInput_.reset();continuationStatus_="terminated: stopped";if(onState)onState({});}
void UIWorldSession::draw(OverlayRenderer& r,const PanelRect& area){
    using namespace UITheme::Colors;layout(area);r.pushClipRect(area.origin,area.size);r.drawRect(area.origin,area.size,CardSurface);
    const float x=area.origin.x+8,y=area.origin.y;
    r.drawText({x,y+10},challenge_?"ConceptBlock Goal":"Manual playtest",TextPrimary,16);
    const auto progress=blockIndex_<0?"No course block selected":std::to_string(blockIndex_+1)+" / "+std::to_string(courseBlockIds_.size())+"  "+blockName_;
    r.drawText({x,y+108},progress,TextSecondary,11);
    std::string text=details_.empty() || (!challenge_ && !hostDetailsVisible_)?"":details_+"\n\n";
    if(challenge_){
        text+="Selected: "+selected_.at("source_id").get<std::string>()+"\n";
        const auto prompt=selected_.value("prompt",std::string{});
        if(!prompt.empty())text+="\nPrompt\n"+prompt+"\n\n";
        if(!selected_.at("goal").is_null()){
            const auto& goal=selected_.at("goal");text+=goal.value("target_state",std::string{})+"\n\n";
            for(const auto& c:goal.at("success_criteria"))text+="Success: "+c.at("criterion").get<std::string>()+"\n";
            for(const auto& c:goal.at("constraints"))text+="Constraint: "+c.at("constraint").get<std::string>()+"\n";
        }else text+=selected_.at("source_id").get<std::string>().empty()?"Sandbox exploration; no automatic Goal success.\n":"This ConceptBlock has no Goal; no automatic Goal success.\n";
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
        if(hostDetailsVisible_)text+=historyHostDetails(historyIndex_);
        else text+="Manual trajectory preview - boundary "+std::to_string(historyIndex_)+
            "\n"+historyEntries_[historyIndex_].continuationStatus+"\n\n"+historyModelText(historyIndex_);
    }
    transcriptText_.setText(text);for(auto* w:controls())if(w->isVisible())w->drawOverlay(r,{});
    if(!challenge_ && composerVisible_)actions_.draw(r,{{area.origin.x+8,area.origin.y+342},{area.size.x-16,160}},false);
    for(auto* w:controls())if(auto* d=dynamic_cast<UIDropdown*>(w);d && d->isVisible() && d->isExpanded())d->drawExpandedList(r,{});
    r.drawText({x,area.origin.y+area.size.y-34},message_,client_.busy()?Warning:TextSecondary,11);r.popClipRect();
}
