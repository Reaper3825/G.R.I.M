#include "ui/world_lab/textworld_client.hpp"
#include "ui/world_lab/world_document.hpp"
#include "ui/world_lab/ui_world_session.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include "ui/input/ui_input_router.hpp"
#include "core/input_parser.hpp"
#include <chrono>
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
        tool.tool_id="catalog_fixture";tool.provider_name="catalog-client-tests";
        tool.short_description="Inspect information supplied by a selected entity";
        tool.description="Full documentation returned only by deeper inspection";
        tool.usage="catalog_fixture {t}";tool.bindings={{"t","t"}};
        tool.parameters={{"target","string","The selected thing to inspect",true}};
        int calls=0;
        registry.registerTool(tool,[&](const std::string&){++calls;return CommandResult{};});
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
        tool.description="Updated tool documentation without world recompilation";registry.updateMetadata(tool);
        detail=request(catalogClient,{{"op","step"},{"command","x catalog_fixture in tool catalog"}}).at("state");
        check(detail.at("tool_details").at("description")==tool.description && calls==0,"Scoped details were stale or invoked a tool");
        request(catalogClient,{{"op","step"},{"command","close tool catalog"}});
        closed=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture in tool catalog"}}).at("state");
        check(!closed.contains("tool_details"),"Closing the catalog did not hide its entries");
        request(catalogClient,{{"op","step"},{"command","open tool catalog"}});
        registry.unregisterTool(tool.tool_id);
        detail=request(catalogClient,{{"op","step"},{"command","examine catalog_fixture in tool catalog"}}).at("state");
        check(!detail.contains("tool_details") && detail.contains("observation_error"),"Removed tool inspection leaked stale documentation");
        observation=request(catalogClient,{{"op","step"},{"command","open tool catalog"}}).at("state");
        check(observation.at("tool_catalog").at("tools").empty(),"Registry removal required recompiling the catalog");
        auto failed=request(catalogClient,{{"op","step"},{"command","examine absent"}}).at("state");
        check(!failed.contains("tool_catalog"),"Failed examine reused a stale catalog observation");
    }
    Editor example;
    example.load(std::filesystem::path(__FILE__).parent_path().parent_path()/"examples/world_lab/locked_door.grimworld.json");
    const Document d=example.document();
    const auto selection=legacyGoal(d);
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
    UIWorldSession panel(example,records);panel.selectGoal(selection);panel.setChallenge(false);
    auto frame=[&](const InputState& input){auto& router=GRIM::UI::InputRouter::get();router.beginFrame();router.currentPanel=90;panel.update(input,0,{{0,0},{340,650}});router.endFrame();};
    auto click=[&](float x,float y){InputState input;input.mousePos={x,y};input.mousePressed[0]=input.mouseDown[0]=true;frame(input);input.mousePressed[0]=input.mouseDown[0]=false;input.mouseReleased[0]=true;frame(input);};
    auto ready=[&]{const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(135);while(panel.busy() && std::chrono::steady_clock::now()<deadline){frame({});std::this_thread::sleep_for(std::chrono::milliseconds(20));}check(!panel.busy(),"Play panel request timed out");};
    frame({});click(50,48);ready();check(panel.status()=="Ready for a command.","Play panel did not compile the example");
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
    for(const auto& command:solution){
        click(20,166);InputState input;input.textInput=command;frame(input);input.textInput.clear();input.keyPressed[VK_RETURN]=true;frame(input);ready();
    }
    check(panel.status()=="Won!","Play panel commands did not win");
    click(130,86);ready();check(example.document().solution.empty(),"Solution leaked back into environment document");
    check(std::filesystem::is_directory(records),"Play panel did not save an episode record");
    const auto recordFile=std::filesystem::directory_iterator(records)->path();
    const auto record=readEpisodeRecord(recordFile);
    check(record.at("commands")==solution && record.at("verified").get<bool>(),"Play panel did not save verified recording");
    click(250,86);ready();check(panel.status()=="Solution verified and saved for this environment/Goal pairing.","Saved witness replay failed");
    GRIM::ConceptBlockGoal initialGoal;
    initialGoal.success_criteria.push_back({"At starting room",GRIM::EvidenceCondition{"player_at",d.spawn,""}});
    panel.selectGoal({{"source_id","initial-goal"},{"goal",GRIM::goalToJson(initialGoal)}});
    click(50,48);ready();check(panel.status()=="Won!","Goal-only start did not evaluate initial success");
    click(40,86);ready();click(130,86);ready();
    check(panel.status()=="Solution verified and saved for this environment/Goal pairing.","Zero-command solution was not saved");
    panel.selectGoal(selection);click(50,48);ready();
    example.change([](Document& world){world.objective="A changed objective.";});frame({});
    click(180,48);ready();check(panel.status()=="Ready for a command.","Legacy metadata incorrectly invalidated environment compilation");
    example.change([](Document& world){world.find(world.spawn)->description="Changed environment description.";});frame({});
    click(180,48);check(panel.status()=="Compile & Play the current world first","Play panel reused a stale compiled world");
    panel.hide();ready();
    for(const auto& entry:std::filesystem::directory_iterator(records))std::filesystem::remove(entry.path());
    std::filesystem::remove(records);
    check(sessions()==sessionsBefore,"TextWorld client leaked temporary session folders");
    std::cout<<"TextWorld client compile/play/reset/solution/cancellation integration passed\n";
}
