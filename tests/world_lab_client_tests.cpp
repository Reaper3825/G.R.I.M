#include "ui/world_lab/textworld_client.hpp"
#include "ui/world_lab/world_document.hpp"
#include "ui/world_lab/ui_world_session.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/overlay_renderer.hpp"
extern std::string renderedTestText;
#include "core/input_parser.hpp"
#include "resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp"
#include <fstream>
#include <chrono>
#include <atomic>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <set>

using namespace GRIM::WorldLab;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
nlohmann::json wait(TextWorldClient& client) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(135);
    while(std::chrono::steady_clock::now()<deadline){
        if(auto response=client.poll())return *response;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    client.stop();throw std::runtime_error("Client integration test timed out");
}
nlohmann::json request(TextWorldClient& client,nlohmann::json body){
    check(client.begin(std::move(body)),"Request unexpectedly busy");
    auto response=wait(client);
    if(!response.value("ok",false))throw std::runtime_error(response.dump());
    return response.at("result");
}
}
void runTextWorldClientTests(){
    const auto sessions=[] {
        std::set<std::filesystem::path> found;
        for(const auto& entry:std::filesystem::directory_iterator(std::filesystem::temp_directory_path()))
            if(entry.is_directory() && entry.path().filename().wstring().find(L"grim-world-session-")==0)found.insert(entry.path());
        return found;
    };
    const auto sessionsBefore=sessions();
    {
        auto& registry=GRIM::MMO::ToolRegistry::instance();
        GRIM::MMO::ToolDescriptor tool;
        tool.tool_id="catalog_fixture";
        tool.short_description="Inspect information supplied by a selected entity";
        tool.description="Full documentation returned only by deeper inspection";
        tool.usage="catalog_fixture {t}";tool.bindings={{"t","t"}};
        tool.parameters={{"target","string","The selected thing to inspect",true}};
        int calls=0;
        auto& providers = GRIM::MMO::ProviderRegistry::instance();
        const GRIM::MMO::ProviderDescriptor owner{GRIM::MMO::ToolProviderType::Plugin, "Client catalog fixture", "fixture-1", {}};
        check(providers.publish(owner, {{tool, [&](const std::string&) {
            ++calls;
            return CommandResult{};
        }}}), "Client catalog fixture publication failed");
        Editor catalog;
        catalog.load(std::filesystem::path(__FILE__).parent_path().parent_path()/"resources/models/games/grim_internal.grimworld.json");
        TextWorldClient catalogClient;
        request(catalogClient,{{"op","compile"},{"document",catalog.document().gameJson()}});
        auto inaccessible=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture"}}).at("state");
        check(!inaccessible.contains("tool_details"),"Tool details were accessible outside the catalog room");
        request(catalogClient,{{"op","step"},{"command","enter tools"}});
        auto description=request(catalogClient,{{"op","step"},{"command","x tool catalog"}}).at("state");
        check(!description.contains("tool_catalog"),"Examining the catalog still listed tools");
        auto closed=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture"}}).at("state");
        check(!closed.contains("tool_details"),"Closed catalog exposed its entries");
        auto observation=request(catalogClient,{{"op","step"},{"command","open tool catalog"}}).at("state");
        check(observation.at("tool_catalog").at("tools")[0].at("tool_id")==tool.tool_id &&
            observation.at("feedback").get<std::string>().find(tool.short_description)!=std::string::npos,
            "Opening the container did not return the host registry catalog");
        check(calls==0 && registry.getTool(tool.tool_id)->usage_count==0,"Catalog invoked a listed tool");
        auto detail=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture"}}).at("state");
        check(detail.at("tool_details").at("description")==tool.description &&
            detail.at("feedback").get<std::string>().find("Tool usage: catalog_fixture {t}")!=std::string::npos &&
            detail.at("tool_details").at("parameters").size()==1,"Native deeper inspection did not return tool documentation");
        tool.description="Updated tool documentation without world recompilation";registry.updateMetadata(tool, owner);
        detail=request(catalogClient,{{"op","step"},{"command","x catalog_fixture in tool catalog"}}).at("state");
        check(detail.at("tool_details").at("description")==tool.description && calls==0,"Scoped details were stale or invoked a tool");
        request(catalogClient,{{"op","step"},{"command","close tool catalog"}});
        closed=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture in tool catalog"}}).at("state");
        check(!closed.contains("tool_details"),"Closing the catalog did not hide its entries");
        request(catalogClient,{{"op","step"},{"command","open tool catalog"}});
        providers.retire(owner.name, owner.instance);
        detail=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture in tool catalog"}}).at("state");
        check(!detail.contains("tool_details") && detail.contains("observation_error"),"Removed tool inspection leaked stale documentation");
        observation=request(catalogClient,{{"op","step"},{"command","open tool catalog"}}).at("state");
        check(observation.at("tool_catalog").at("tools").empty(),"Registry removal required recompiling the catalog");
        auto failed=request(catalogClient,{{"op","step"},{"command","examine absent"}}).at("state");
        check(!failed.contains("tool_catalog"),"Failed examine reused a stale catalog observation");
    }
    Editor example;
    {
        // Real mailbox/client/worker path; controlled handlers stand in for UI Plugin.
        auto& providers = GRIM::MMO::ProviderRegistry::instance();
        const GRIM::MMO::ProviderDescriptor owner{GRIM::MMO::ToolProviderType::Plugin, "Client UI fixture", "fixture-1", {}};
        std::vector<GRIM::MMO::ToolRegistration> registrations;
        int shows = 0, hides = 0;
        bool visible = false;
        std::atomic<bool> stall{false}, submitted{false}, released{false};
        for (const auto& id : {"ui.list", "ui.inspect", "ui.show", "ui.hide", "ui.poll"}) {
            GRIM::MMO::ToolDescriptor tool;
            tool.tool_id = tool.display_name = id;
            tool.argument_format = "json";
            if (tool.tool_id == "ui.inspect" || tool.tool_id == "ui.show" || tool.tool_id == "ui.hide") {
                tool.usage = tool.tool_id + " {t}"; tool.bindings = {{"t", "t"}};
                tool.parameters = {{"panel_id", "string", "Panel key", true}};
            }
            registrations.push_back({tool, [&, id = std::string(id)](const std::string& text) {
                const auto args = nlohmann::json::parse(text);
                nlohmann::json result = {{"success", true}};
                if ((id == "ui.list" && stall) || id == "ui.poll") {
                    submitted = true;
                    if (args.value("cancel", false)) released = true;
                    else {
                        CommandResult pending;
                        pending.structuredOutputJson = nlohmann::json{{"format", "grim-ui-panel-response"}, {"version", 1},
                            {"request_id", 1}, {"status", "pending"}}.dump();
                        return pending;
                    }
                }
                if (id == "ui.list") result["panels"] = {{{"panel_id", "settings"}, {"title", "Settings panel"}, {"runtime_id", 42}}};
                else if (id != "ui.poll") {
                    check(args.at("panel_id") == "settings", "Client adapter encoded incorrect key");
                    if (id == "ui.show" || id == "ui.hide") {
                        check(args.at("expected_runtime_id") == 42, "Client omitted generation guard");
                        visible = id == "ui.show";
                        if (visible) ++shows; else ++hides;
                    }
                    result["panel"] = {{"panel_id", "settings"}, {"title", "Settings panel"}, {"runtime_id", 42},
                        {"visible", visible}, {"minimized", false}, {"maximized", false}};
                }
                CommandResult output;
                output.structuredOutputJson = nlohmann::json{{"format", "grim-ui-panel-response"}, {"version", 1},
                    {"request_id", 1}, {"status", "completed"}, {"result", result}}.dump();
                return output;
            }});
        }
        check(providers.publish(owner, registrations), "Client UI fixture publication failed");
        TextWorldClient uiClient;
        const auto initial = request(uiClient, {{"op", "compile"}, {"document", Document::initial().gameJson()}}).at("state");
        auto state = request(uiClient, {{"op", "step"}, {"command", " ui.list "}}).at("state");
        check(state.at("host_action").at("resources")[0].at("reference") == "panel:settings", "Client did not project UI discovery");
        check(!state.contains("host_action_request"), "Internal dispatch request leaked");
        state = request(uiClient, {{"op", "step"}, {"command", "ui.inspect Settings panel"}}).at("state");
        check(state.at("host_action").at("status") == "completed" && shows == 0, "Client inspection caused effects");
        state = request(uiClient, {{"op", "step"}, {"command", "ui.show panel:settings"}}).at("state");
        check(state.at("host_action").at("status") == "completed" && shows == 1, "Client did not execute one UI action");
        state = request(uiClient, {{"op", "step"}, {"command", "ui.hide Settings panel"}}).at("state");
        check(state.at("host_action").at("resource").at("visible") == false && hides == 1,
              "Client did not execute hide and report completion");
        check(state.at("facts") == initial.at("facts") && state.at("moves") == initial.at("moves") &&
            state.at("episode") == initial.at("episode"), "Host effect altered native verification state");
        check(uiClient.begin({{"op", "verify_solution"}, {"commands", {"ui.show panel:settings"}}}), "Cannot submit replay");
        check(!wait(uiClient).at("ok").get<bool>() && shows == 1, "Replay executed a host effect");
        check(uiClient.begin({{"op", "verify_solution"}, {"commands", {"ui.hide panel:settings"}}}), "Cannot submit hide replay");
        check(!wait(uiClient).at("ok").get<bool>() && hides == 1, "Replay repeated hide");
        request(uiClient, {{"op", "reset"}});
        state = request(uiClient, {{"op", "step"}, {"command", "ui.show panel:settings"}}).at("state");
        check(state.at("host_action").at("status") == "failed" && shows == 1, "Reset retained observed resources");
        stall = true;
        check(uiClient.begin({{"op", "step"}, {"command", "ui.list"}}), "Cannot start pending host request");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!submitted && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(submitted, "Host request did not reach submission");
        uiClient.stop();
        const auto cancelled = wait(uiClient);
        check(!cancelled.at("ok").get<bool>() && cancelled.at("transport_error").get<bool>() && released,
              "Stop during host dispatch did not cancel owned work and stop the worker");
        providers.retire(owner.name, owner.instance);
    }
    example.load(std::filesystem::path(__FILE__).parent_path().parent_path()/"examples/world_lab/locked_door.grimworld.json");
    const Document d=example.document();
    auto selection=legacyGoal(d);
    selection["prompt"]="Find the brass key and carry it through the oak door.";
    TextWorldClient client;
    auto start=request(client,{{"op","compile"},{"document",d.gameJson()},
        {"source_id",selection.at("source_id")},{"goal",selection.at("goal")}});
    check(start.at("action_catalog")==actionCatalog(),"Runtime catalog differs from native action definitions");
    check(start.at("state").at("room_id")==d.spawn,"Client did not receive initial room");
    const std::vector<std::string> solution{"open toolbox","take brass key from toolbox","unlock oak door with brass key","open oak door","go east"};
    nlohmann::json last;
    for(const auto& command:solution)last=request(client,{{"op","step"},{"command",command}});
    check(last.at("state").at("won")==true,"Client puzzle did not reach win");
    check(request(client,{{"op","verify_solution"},{"commands",solution}}).at("verified")==true,"Client solution verification failed");
    check(request(client,{{"op","reset"}}).at("state").at("room_id")==d.spawn,"Client reset failed");
    check(client.begin({{"op","validate"},{"document",d.gameJson()}}),"Cannot start validation");
    check(!client.begin({{"op","step"},{"command","look"}}),"Client allowed overlapping requests");
    client.stop();auto cancelled=wait(client);check(!cancelled.value("ok",true),"Cancellation did not report an error");
    check(request(client,{{"op","compile"},{"document",d.gameJson()}}).contains("state"),"Client did not restart after cancellation");
    client.stop();wait(client);
    // Drive the actual play controls with routed input and the installed worker.
    example.change([](Document& world){world.solution.clear();});
    const auto records=std::filesystem::temp_directory_path()/("grim-episode-tests-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    UIWorldSession panel(example,records);
    std::ifstream configFile(std::filesystem::path(__FILE__).parent_path().parent_path()/"ai_config.json");
    nlohmann::json config;configFile>>config;
    const auto definitions=config.at("training").at("config").at("concept_spans").get<GRIM::NamedConceptSpanDefinitions>();
    bool failObservation=false;
    panel.setGoalAccess({{}, {}, {}, {},[&](const nlohmann::json& supplied){
        if(failObservation && supplied.contains("environment_context"))throw std::runtime_error("Injected observation rendering failure");
        return GRIM::SpanText::renderInput(supplied,definitions);
    }, [](std::string text) {
        std::promise<GRIM::WorldLab::PreviewTokenResult> count;
        count.set_value({text.find("<execute>") == std::string::npos ? 42 : 87, {}});
        return count.get_future();
    }});
    panel.selectGoal(selection);panel.setChallenge(false);
    auto frame=[&](const InputState& input){auto& router=GRIM::UI::InputRouter::get();router.beginFrame();router.currentPanel=90;panel.update(input,0,{{0,0},{340,650}});router.endFrame();};
    auto click=[&](float x,float y){InputState input;input.mousePos={x,y+116};input.mousePressed[0]=input.mouseDown[0]=true;frame(input);input.mousePressed[0]=input.mouseDown[0]=false;input.mouseReleased[0]=true;frame(input);};
    auto ready=[&]{const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(135);while(panel.busy() && std::chrono::steady_clock::now()<deadline){frame({});std::this_thread::sleep_for(std::chrono::milliseconds(20));}check(!panel.busy(),"Play panel request timed out");};
    frame({});click(50,48);ready();check(panel.status()=="Ready for a command.","Play panel did not compile the example");
    check(panel.initialModelInput().has_value(),"Start did not publish model input");
    auto supplied=panel.initialModelInput()->reasoning_state;supplied["prompt"]=panel.initialModelInput()->prompt;
    check(panel.initialModelText()==GRIM::SpanText::renderInput(supplied,definitions),"World Lab preview differs from the configured inference renderer");
    check(panel.initialModelText().starts_with("<prompt>\n") && panel.initialModelText().find("<target_state>\n")!=std::string::npos &&
          panel.initialModelText().find("<criteria>\n")!=std::string::npos,"Initial preview omitted configured span markers");
    check(panel.initialModelText().find("<evidence>")==std::string::npos && panel.initialModelText().find("<determine>")==std::string::npos &&
          panel.initialModelText().find("admissible_commands")==std::string::npos,"Initial preview exposed private data or authored continuation");
    check(panel.initialModelInput()->prompt==selection.at("prompt").get<std::string>() &&
          panel.initialModelInput()->reasoning_state.at("environment_context").get<std::string>().find("Description:\n")!=std::string::npos,"Start lost the authored prompt or live observation");
    check(panel.initialModelText().find("<prompt>\n"+selection.at("prompt").get<std::string>()+"\n</prompt>")==0 &&
          panel.initialModelText().find("<environment_context>\n")!=std::string::npos,"Task and environment did not render in separate spans");
    check(panel.initialModelText().find("$$$$")==std::string::npos && panel.initialModelText().find("-= ")==std::string::npos &&
          panel.initialModelText().find("LOCATION:")==std::string::npos,"Model input retained interpreter chrome");
    check(panel.historyEntries().size()==1 && panel.historyIndex()==0 && panel.continuationStatus()=="ready_for_command",
          "Start did not publish snapshot zero and its continuation status");
    const auto initialPreview=panel.initialModelText();
    check(panel.historyEntries().front().receipt.contains("result") &&
          panel.historyHostDetails(0).find("Host receipt (private)")!=std::string::npos,
          "Initial boundary lost its separate host receipt");
    const auto firstGoal=panel.initialModelInput()->reasoning_state.at("goal");
    check(firstGoal.at("target_state")==selection.at("goal").at("target_state") &&
          firstGoal.at("success_criteria").size()==selection.at("goal").at("success_criteria").size(),"Selected Goal did not reach model input");
    check(firstGoal.dump().find("evidence")==std::string::npos &&
          panel.initialModelInput()->reasoning_state.at("knowns").empty() &&
          panel.initialModelInput()->reasoning_state.at("unknowns").empty(),"Initial model state leaked evidence or retained intermediate state");
    // Preparing a typed action does not execute it. Dropdown selection cannot
    // click through to the Prepare button underneath its popup.
    click(280,202); // Actions composer.
    click(40,292);click(40,343); // Container parameter -> toolbox.
    check(panel.preparedCommand().empty(),"Parameter popup clicked through to Prepare");
    click(40,354);
    check(panel.preparedCommand()=="close toolbox","Play composer failed typed binding");
    panel.prepareCommand("");click(280,202); // Close composer.
    click(40,202);click(40,254); // Latest live available command.
    check(!panel.preparedCommand().empty(),"Live available-command picker failed");
    panel.prepareCommand("");
    click(40,86);ready();check(panel.status()=="Recording from the starting state.","Record did not reset the game");
    check(panel.historyEntries().size()==1 && panel.initialModelText()==initialPreview,
          "Record did not reset trajectory history to the new initial input");
    std::string expectedTrajectory=panel.initialModelText();
    for(const auto& command:solution){
        const auto before=panel.historyEntries().size();
        // Inspect snapshot zero while executing against the latest live state.
        panel.selectHistory(0);
        check(panel.historyIndex()==0 && panel.historyModelText(0)==initialPreview,
              "History selection changed snapshot zero");
        click(20,166);InputState input;input.textInput=command;frame(input);input.textInput.clear();input.keyPressed[VK_RETURN]=true;frame(input);
        expectedTrajectory+=GRIM::SpanText::renderInput({{"execute",command}},definitions);
        check(panel.historyEntries().size()==before+1 && panel.continuationStatus()=="awaiting_action_result" &&
              panel.historyModelText(before)==expectedTrajectory && panel.historyEntries().back().observation.empty(),
              "Pending boundary lost Execute or included future observation");
        const auto pendingPreview=panel.historyModelText(before);
        ready();
        const auto& completed=panel.historyEntries().back();
        expectedTrajectory+=completed.observation;
        if(panel.historyEntries().size()!=before+2 || panel.historyModelText(before+1)!=expectedTrajectory ||
           panel.historyModelText(before)!=pendingPreview || panel.historyModelText(0)!=initialPreview){
            throw std::runtime_error("Response boundary mismatch for "+command+"; status="+panel.status()+
                "; entries="+std::to_string(panel.historyEntries().size())+" expected="+std::to_string(before+2)+
                "; actual bytes="+std::to_string(panel.historyModelText(panel.historyEntries().size()-1).size())+
                " expected bytes="+std::to_string(expectedTrajectory.size()));
        }
        const auto& response=completed.receipt;
        const auto projected=GRIM::Environment::buildEnvironmentInferenceInput(GRIM::goalFromJson(selection.at("goal")),response);
        check(completed.observation==GRIM::SpanText::renderInput(
                  {{"environment_context",projected.reasoning_state.at("environment_context")}},definitions),
              "Historical observation differs from inference projection");
        check(expectedTrajectory.find("admissible_commands")==std::string::npos &&
              expectedTrajectory.find("<evidence>")==std::string::npos && expectedTrajectory.find("<answer>")==std::string::npos &&
              expectedTrajectory.find("<determine>")==std::string::npos,
              "Manual trajectory leaked host data or invented reasoning/submission");
    }
    check(panel.status()=="Won!","Play panel commands did not win");
    check(panel.continuationStatus()=="world_terminal: succeeded; awaiting_answer",
          "World success incorrectly submitted the attempt");
    const auto terminalCount=panel.historyEntries().size();
    const auto terminalPreview=panel.historyModelText(terminalCount-1);
    nlohmann::json historicalState;
    panel.onState=[&](const auto& state){historicalState=state;};
    panel.selectHistory(0);
    check(historicalState.at("room_id")==d.spawn,"History selection did not project the historical canvas state");
    panel.prepareCommand("look");click(280,166);
    check(!panel.busy() && panel.status()=="Episode finished; reset to play again" && panel.historyEntries().size()==terminalCount,
          "Historical selection rewound the live terminal world");
    panel.prepareCommand("");panel.selectHistory(static_cast<int>(terminalCount)-1);
    OverlayRenderer historyRenderer;
    frame({});frame({});
    renderedTestText.clear();panel.draw(historyRenderer,{{0,0},{340,650}});
    check(renderedTestText.find("Preview tokens: 87")!=std::string::npos,
          "Trajectory footer did not count the selected command history");
    panel.selectHistory(0);frame({});frame({});
    renderedTestText.clear();panel.draw(historyRenderer,{{0,0},{340,650}});
    check(renderedTestText.find("Preview tokens: 42")!=std::string::npos,
          "Trajectory footer retained the count for a different boundary");
    panel.selectHistory(static_cast<int>(terminalCount)-1);
    click(270,126);frame({});renderedTestText.clear();panel.draw(historyRenderer,{{0,0},{340,650}});
    check(renderedTestText.find("Boundary status:")!=std::string::npos &&
          renderedTestText.find("Manual trajectory preview")==std::string::npos &&
          renderedTestText.find("Preview tokens:")==std::string::npos,"Host-details toggle did not select the separate inspection view");
    click(270,126);renderedTestText.clear();panel.draw(historyRenderer,{{0,0},{340,650}});
    check(renderedTestText.find("Manual trajectory preview")!=std::string::npos &&
          renderedTestText.find("Host receipt (private)")==std::string::npos,"Trajectory view mixed private host details into model text");
    check(panel.historyIndex()==static_cast<int>(terminalCount)-1 && panel.historyModelText(terminalCount-1)==terminalPreview,
          "Host-details inspection changed the trajectory or selection");
    click(130,86);ready();check(example.document().solution.empty(),"Solution leaked back into environment document");
    check(std::filesystem::is_directory(records),"Play panel did not save an episode record");
    const auto recordFile=std::filesystem::directory_iterator(records)->path();
    const auto record=readEpisodeRecord(recordFile);
    check(panel.historyEntries().size()==terminalCount && panel.historyModelText(terminalCount-1)==terminalPreview,
          "Independent verification created a live trajectory boundary");
    check(record.at("commands")==solution && record.at("verified").get<bool>(),"Play panel did not save verified recording");
    check(record.at("selection").at("prompt")==selection.at("prompt"),"Saved recording lost the task prompt");
    auto legacyRecord=record;legacyRecord["selection"].erase("prompt");saveJsonFile(recordFile,legacyRecord);
    click(250,86);ready();check(panel.status()=="Solution verified and saved for this environment/Goal pairing.","Saved witness replay failed");
    GRIM::ConceptBlockGoal initialGoal;
    initialGoal.target_state="Remain in the starting room.";
    initialGoal.success_criteria.push_back({"At starting room",GRIM::EvidenceCondition{"player_at",d.spawn,""}});
    panel.selectGoal({{"source_id","initial-goal"},{"prompt","Stay in this room."},{"goal",GRIM::goalToJson(initialGoal)}});
    check(!panel.initialModelInput() && panel.historyEntries().empty() && panel.historyIndex()==-1,
          "Selecting another block retained the previous attempt history");
    panel.prepareCommand("a stale command");
    click(180,48);ready();check(panel.status()=="Won!","Reset did not bind the newly selected Goal");
    check(panel.preparedCommand().empty() && panel.initialModelInput()->reasoning_state.at("goal").at("target_state")==initialGoal.target_state &&
          !panel.initialModelInput()->reasoning_state.at("environment_context").get<std::string>().empty(),"Reset did not refresh Goal and starting observation together");
    check(panel.initialModelInput()->prompt=="Stay in this room." &&
          panel.initialModelInput()->prompt.find(selection.at("prompt").get<std::string>())==std::string::npos,"Reset retained the previous block's prompt");
    check(panel.initialModelText().find("Stay in this room.")!=std::string::npos &&
          panel.initialModelText().find(selection.at("prompt").get<std::string>())==std::string::npos,"Reset retained the previous rendered span text");
    click(40,86);ready();click(130,86);ready();
    check(panel.status()=="Solution verified and saved for this environment/Goal pairing.","Zero-command solution was not saved");
    panel.selectGoal({{"source_id",""},{"goal",nullptr}});click(40,86);ready();
    check(panel.initialModelInput() && panel.initialModelInput()->reasoning_state.at("goal").at("target_state")=="" &&
          panel.initialModelInput()->reasoning_state.at("goal").at("success_criteria").empty(),"Recording a sandbox retained the previous Goal");
    check(panel.initialModelInput()->prompt.empty(),"Sandbox retained an authored task prompt");
    panel.selectGoal(selection);click(50,48);ready();
    panel.selectGoal(selection);panel.prepareCommand("look");click(280,166);
    check(!panel.busy() && panel.historyEntries().empty() && panel.status()=="Start the selected Goal first",
          "Reselecting the active Goal allowed continuation without fresh history");
    click(180,48);ready();
    example.change([](Document& world){world.objective="A changed objective.";});frame({});
    click(180,48);ready();check(panel.status()=="Ready for a command.","Legacy metadata incorrectly invalidated environment compilation");
    const auto originalDescription=example.document().find(example.document().spawn)->description;
    example.change([](Document& world){world.find(world.spawn)->description="Changed environment description.";});frame({});
    click(180,48);check(panel.status()=="Compile & Play the current world first","Play panel reused a stale compiled world");
    example.change([&](Document& world){world.find(world.spawn)->description=originalDescription;});frame({});
    panel.prepareCommand("look");click(280,166);
    check(panel.continuationStatus()=="awaiting_action_result","Fault test did not submit an action");
    const auto pendingFailurePreview=panel.historyModelText(panel.historyEntries().size()-1);
    failObservation=true;ready();failObservation=false;
    const auto& failedBoundary=panel.historyEntries().back();
    check(panel.continuationStatus()=="terminated: infrastructure_error" && !panel.initialModelInput() &&
          failedBoundary.receipt.at("status")=="completed" && failedBoundary.receipt.at("response").at("result").contains("state") &&
          failedBoundary.observation.empty() && panel.historyModelText(panel.historyEntries().size()-1)==pendingFailurePreview,
          "Post-dispatch render failure lost its receipt, invented feedback, or remained resumable");
    const auto failedCount=panel.historyEntries().size();panel.prepareCommand("look");click(280,166);
    check(!panel.busy() && panel.historyEntries().size()==failedCount && panel.status()=="Compile & Play the current world first",
          "Infrastructure failure allowed implicit action resubmission");
    panel.hide();ready();
    check(!panel.initialModelInput() && panel.continuationStatus()=="terminated: stopped",
          "Hidden session retained live model input or a resumable status");
    for(const auto& entry:std::filesystem::directory_iterator(records))std::filesystem::remove(entry.path());
    std::filesystem::remove(records);
    check(sessions()==sessionsBefore,"TextWorld client leaked temporary session folders");
    std::cout<<"TextWorld client compile/play/reset/solution/cancellation integration passed\n";
}
