#include "ui/world_lab/world_document.hpp"
#include "ui/world_lab/ui_world_canvas.hpp"
#include "ui/world_lab/ui_world_lab_view.hpp"
#include "ui/world_lab/ui_world_session.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

using namespace GRIM::WorldLab;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> void rejects(F operation,const char* message){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}check(failed,message);}
auto& router=GRIM::UI::InputRouter::get();
void frame(UIWorldCanvas& canvas,const InputState& input,bool cancel=false){router.beginFrame(cancel);router.currentPanel=90;canvas.update(input,0);router.endFrame();}
void drag(UIWorldCanvas& canvas,Vec2 from,Vec2 to,bool cancel=false){
    InputState in;in.mousePos=from;in.mousePressed[0]=in.mouseDown[0]=true;frame(canvas,in);
    in.mousePressed[0]=false;in.mousePos=to;frame(canvas,in);
    if(cancel){frame(canvas,in,true);return;}
    in.mouseDown[0]=false;in.mouseReleased[0]=true;frame(canvas,in);
}
}
void runWorldLabTests(){
    Editor editor;
    std::string room=editor.document().spawn,other,box,key,edge;
    editor.change([&](Document& d){other=d.addRoom(430,80);box=d.addObject(room,"container");key=d.addObject(box,"key");edge=d.connect(room,other,"east");});
    const auto original=editor.document().json();
    rejects([&]{editor.change([&](Document& d){d.find(box)->parent=key;});},"Invalid holder accepted");
    rejects([&]{editor.change([&](Document& d){d.find(box)->parent=box;});},"Containment cycle accepted");
    rejects([&]{editor.change([&](Document& d){auto third=d.addRoom(0,0);d.connect(room,third,"east");});},"Occupied exit accepted");
    check(editor.document().json()==original,"Failed edit partially mutated document");
    editor.change([&](Document& d){d.erase(room);});
    check(!editor.document().find(box)&&!editor.document().find(key)&&editor.document().passages.empty()&&editor.document().spawn==other,"Delete left dangling content/start/passages");
    check(editor.undo()&&editor.document().json()==original,"Undo did not restore deletion atomically");
    check(editor.redo()&&!editor.document().find(room),"Redo failed");editor.undo();
    editor.change([&](Document& d){d.find(key)->name="Brass key";});check(!editor.canRedo(),"New edit did not clear redo");
    auto invalid=editor.document().json();invalid["version"]=9;rejects([&]{Document::parse(invalid);},"Future format accepted");
    auto legacy=original;legacy["version"]=1;legacy.erase("objective");legacy.erase("solution");legacy.erase("win_conditions");legacy.erase("fail_conditions");
    for(auto& p:legacy["passages"]){p.erase("door_name");p.erase("door_state");p.erase("key");}
    check(Document::parse(legacy).winConditions.empty(),"Version-1 world migration failed");
    Editor challenge;
    std::string door,goal,doorKey;
    challenge.change([&](Document& d){goal=d.addRoom(400,80);doorKey=d.addObject(d.spawn,"key");door=d.connect(d.spawn,goal,"east");auto& p=d.passages.back();p.doorName="Oak door";p.doorState="locked";p.key=doorKey;d.winConditions={{"player_at",goal,""},{"carrying",doorKey,""}};});
    auto game=challenge.document().gameJson();challenge.change([&](Document& d){d.moveRoom(goal,500,100);d.solution={"look"};});
    check(challenge.document().gameJson()==game,"Editor layout/solution invalidated compiled game");
    rejects([&]{challenge.change([&](Document& d){d.winConditions.push_back({"carrying",goal,""});});},"Invalid condition accepted");
    challenge.change([&](Document& d){d.erase(doorKey);});
    check(challenge.document().passages.front().doorState=="closed" && challenge.document().winConditions.size()==1,"Deleting key left dangling door/condition");
    challenge.undo();check(challenge.document().passages.front().key==doorKey && challenge.document().winConditions.size()==2,"Challenge undo lost references");
    invalid=original;invalid["entities"][1]["id"]=room;rejects([&]{Document::parse(invalid);},"Duplicate ID accepted");
    invalid=original;invalid["editor"]["rooms"][0]["x"]=1e20;rejects([&]{Document::parse(invalid);},"Out-of-range geometry accepted");
    invalid=original;invalid["spawn"]="missing";rejects([&]{Document::parse(invalid);},"Dangling spawn accepted");
    invalid=original;invalid["passages"][0]["direction"]="up";rejects([&]{Document::parse(invalid);},"Invalid direction accepted");
    const auto directory=std::filesystem::temp_directory_path()/("grim-world-tests-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    const auto path=directory/"world.json",bad=directory/"invalid.json";
    editor.save(path);check(!editor.dirty(),"Save did not clear dirty flag");
    const auto saved=editor.document().json();
    editor.change([&](Document& d){d.title="Second version";});check(editor.dirty(),"Edit did not mark dirty");
    editor.undo();check(!editor.dirty(),"Undo to saved state remains dirty");editor.redo();editor.save(path);
    Editor loaded;loaded.load(path);check(loaded.document().json()==editor.document().json(),"Save replacement/roundtrip failed");
#ifdef _WIN32
    const HANDLE locked=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    check(locked!=INVALID_HANDLE_VALUE,"Could not lock saved file for failure test");
    editor.change([](Document& d){d.title="Must not overwrite locked file";});
    rejects([&]{editor.save(path);},"Replacing a locked file unexpectedly succeeded");
    check(editor.dirty(),"Failed replacement cleared dirty state");
    CloseHandle(locked);
    Editor preserved;preserved.load(path);check(preserved.document().json()==loaded.document().json(),"Failed replacement damaged previous save");
#endif
    {std::ofstream out(bad);out<<"{ invalid";}
    const auto before=loaded.document().json();rejects([&]{loaded.load(bad);},"Malformed save accepted");check(loaded.document().json()==before,"Failed load destroyed document");
    {std::ofstream out(bad);out<<before.dump()<<" garbage";}
    rejects([&]{loaded.load(bad);},"Trailing garbage accepted");
    loaded.newWorld();check(loaded.undo()&&loaded.document().json()==before,"New world not undoable");
    rejects([&]{loaded.save(directory/"missing"/"file.json");},"Save to missing directory succeeded");
    Editor verify;verify.load(path);check(verify.document().json()==before,"Failed save affected existing file");
    std::filesystem::remove(path);std::filesystem::remove(bad);

    Editor gestures;UIWorldCanvas canvas(gestures);canvas.setSize(1000,700);
    const auto first=gestures.document().spawn;canvas.selected=first;
    drag(canvas,{100,100},{160,140});
    check(gestures.document().position(first)->x==140 && gestures.document().position(first)->y==120,"Room drag failed");
    gestures.undo();check(gestures.document().position(first)->x==80,"Drag was not one undo step");
    const auto state=gestures.document().json();drag(canvas,{100,100},{200,200},true);check(gestures.document().json()==state,"Cancelled drag changed document");
    drag(canvas,{310,120.5f},{1100,120.5f});check(gestures.document().json()==state,"Outside port drop created a room");
    // East port is centered on the initial empty room's 81px height.
    drag(canvas,{310,120.5f},{550,120.5f});
    check(gestures.document().layout.size()==2 && gestures.document().passages.size()==1,"Drag-to-create connection failed");
    gestures.undo();check(gestures.document().layout.size()==1 && gestures.document().passages.empty(),"Connection creation not atomic");
    std::string target,object;
    gestures.change([&](Document& d){target=d.addRoom(450,80);object=d.addObject(first,"key");});
    drag(canvas,{110,130},{470,100});check(gestures.document().find(object)->parent==target,"Object reparent drag failed");
    gestures.undo();check(gestures.document().find(object)->parent==first,"Object move undo failed");
    // Drawing and tab lifecycle execute with the existing renderer sinks.
    OverlayRenderer renderer;canvas.drawOverlay(renderer,{});
    UIWorldLabView view(directory/"recovery.json",directory/"games");
    auto viewFrame=[&](const InputState& in){router.beginFrame();router.currentPanel=90;view.update(in,0,{{0,0},{1000,650}},90);view.draw(renderer,{{0,0},{1000,650}});router.endFrame();};
    auto click=[&](Vec2 p){InputState in;in.mousePos=p;in.mousePressed[0]=in.mouseDown[0]=true;viewFrame(in);in.mousePressed[0]=in.mouseDown[0]=false;in.mouseReleased[0]=true;viewFrame(in);};
    viewFrame({});
    click({900,455});check(view.document().entities.size()==2,"Inspector Add button failed");
    click({110,220}); // Select starting room again.
    click({750,455}); // Open kind dropdown.
    click({750,530}); // Select container (third item).
    click({900,455});
    check(view.document().entities.size()==3 && view.document().entities.back().kind=="container","Kind dropdown or popup ownership failed");
    view.hide();
    Editor recovery;recovery.load(directory/"recovery.json");
    check(recovery.document().json()==view.document().json(),"Hide did not preserve recovery snapshot");
    viewFrame({});click({240,57}); // Undo toolbar.
    check(view.document().entities.size()==2,"View undo did not restore document");
    // Minimized/clipped inspector still renders safely with scrolling.
    router.beginFrame();router.currentPanel=90;view.update({},0,{{0,0},{800,350}},90);view.draw(renderer,{{0,0},{800,350}});router.endFrame();
    view.hide();
    viewFrame({});click({170,57}); // Save into the library.
    const auto library=directory/"games";
    check(std::filesystem::exists(library/"world.grimworld.json"),"Default save missed games folder");
    const auto savedWorld=view.document().json();
    click({30,57});click({170,57}); // New must not overwrite the first game.
    check(std::filesystem::exists(library/"world-2.grimworld.json"),"New reused existing game filename");
    click({50,90});click({50,145}); // First alphabetic entry: world-2.
    click({50,90});click({50,170}); // Second entry: world.
    check(view.document().json()==savedWorld,"Saved-game dropdown did not load selected world");
    {std::ofstream malformed(library/"aaa.grimworld.json");malformed<<"invalid";}
    router.beginFrame();router.currentPanel=90;view.update({},2,{{0,0},{1000,650}},90);router.endFrame();
    click({50,90});click({50,145});
    check(view.document().json()==savedWorld,"Malformed library game replaced document");
    view.hide();
    std::filesystem::remove(library/"aaa.grimworld.json");
    std::filesystem::remove(library/"world.grimworld.json");
    std::filesystem::remove(library/"world-2.grimworld.json");
    std::filesystem::remove(library);
    std::filesystem::remove(directory/"recovery.json");std::filesystem::remove(directory);
    Editor conditionEditor;
    conditionEditor.change([](Document& d){d.addObject(d.spawn,"key");});
    UIWorldSession challengePanel(conditionEditor);
    auto challengeFrame=[&](const InputState& in){router.beginFrame();router.currentPanel=90;challengePanel.update(in,0,{{0,0},{340,600}});challengePanel.draw(renderer,{{0,0},{340,600}});router.endFrame();};
    auto challengeClick=[&](Vec2 p){InputState in;in.mousePos=p;in.mousePressed[0]=in.mouseDown[0]=true;challengeFrame(in);in.mousePressed[0]=in.mouseDown[0]=false;in.mouseReleased[0]=true;challengeFrame(in);};
    challengeFrame({});challengeClick({40,260});
    check(conditionEditor.document().winConditions.size()==1 && conditionEditor.document().winConditions.front().predicate=="player_at","Challenge condition creation failed");
    challengeClick({40,112});challengeClick({40,164});challengeClick({40,260});
    check(conditionEditor.document().winConditions.size()==2 && conditionEditor.document().winConditions.back().predicate=="carrying","Challenge predicate selection failed");
    challengeClick({40,54});InputState typed;typed.textInput="Find the key";challengeFrame(typed);typed.textInput.clear();typed.keyPressed[VK_RETURN]=true;challengeFrame(typed);
    check(conditionEditor.document().objective=="Find the key","Challenge objective commit failed");
    conditionEditor.undo();challengeFrame({});
    check(conditionEditor.document().objective.empty(),"Challenge field overwrote external undo");
    challengePanel.hide();
    // Catalog binding is typed and ID-based, independent of starting/live state.
    auto rule=[](const std::string& id)->nlohmann::json {
        for(const auto& a:actionCatalog()["actions"])if(a["id"]==id)return a;
        throw std::runtime_error("Missing built-in action");
    };
    Editor actionEditor;std::string actionKey,actionBox,actionDoor;
    actionEditor.change([&](Document& d){
        actionKey=d.addObject(d.spawn,"key");d.find(actionKey)->name="brass key";
        actionBox=d.addObject(d.spawn,"container");d.find(actionBox)->name="toolbox";
        const auto next=d.addRoom(400,80);actionDoor=d.connect(d.spawn,next,"east");
        d.passages.back().doorName="garden gate";d.passages.back().key=actionKey;
    });
    check(composeAction(actionEditor.document(),rule("unlock/d"),{actionDoor,actionKey})=="unlock garden gate with brass key","Typed door binding failed");
    check(composeAction(actionEditor.document(),rule("go/north"),{})=="go north","Zero parameter action failed");
    rejects([&]{composeAction(actionEditor.document(),rule("eat"),{actionKey});},"Wrong action parameter kind accepted");
    rejects([&]{composeAction(actionEditor.document(),rule("unlock/d"),{actionDoor});},"Missing action parameter accepted");
    check(actionRelevant(actionEditor.document(),rule("take/c"),actionBox),"Holder action missing from inspector");
    check(!actionRelevant(actionEditor.document(),rule("eat"),actionBox),"Unrelated action in inspector");
    check(actionRelevant(actionEditor.document(),rule("go/west"),actionDoor),"Reverse passage traversal missing");
    UIWorldActions actionView(actionEditor);actionView.select(actionBox);
    std::string prepared;actionView.onPrepare=[&](const auto& value){prepared=value;};
    auto actionFrame=[&](const InputState& in){router.beginFrame();router.currentPanel=90;actionView.update(in,0,{{0,0},{260,530}},90);actionView.draw(renderer,{{0,0},{260,530}});router.endFrame();};
    auto actionClick=[&](Vec2 p){InputState in;in.mousePos=p;in.mousePressed[0]=in.mouseDown[0]=true;actionFrame(in);in.mousePressed[0]=in.mouseDown[0]=false;in.mouseReleased[0]=true;actionFrame(in);};
    actionFrame({}); // Sorted catalog starts with close/c for a container.
    check(actionView.command()=="close toolbox","Selection did not bind container parameter");
    const auto actionBefore=actionEditor.document().json();actionClick({40,128});
    check(prepared=="close toolbox" && actionEditor.document().json()==actionBefore,"Prepare changed world or failed");
    actionClick({40,36});actionClick({40,62}); // Full-width dropdown first row.
    check(actionView.command()=="close toolbox","Action dropdown lost selection binding");
    actionEditor.change([&](Document& d){d.find(actionBox)->name="supply box";});actionFrame({});
    check(actionView.command()=="close supply box","Renamed entity left stale command");
    actionEditor.change([&](Document& d){d.erase(actionBox);});actionFrame({});
    rejects([&]{actionView.command();},"Deleted selection left a usable stale action");
    actionView.hide();
    challengePanel.setChallenge(false);challengePanel.prepareCommand("look");challengeFrame({});
    check(challengePanel.preparedCommand()=="look","Play did not retain prepared command");
    challengePanel.hide();
    std::cout<<"World Lab document, persistence, gesture, and lifecycle tests passed\n";
}
