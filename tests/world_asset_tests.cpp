#include "ui/world_lab/world_asset.hpp"
#include "ui/world_lab/ui_world_assets_view.hpp"
#include "ui/world_lab/ui_world_lab_view.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace GRIM::WorldLab;
namespace {
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F f,const char* message){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,message);}
}
void runWorldAssetTests() {
    auto source=Document::initial();const auto room=source.spawn;
    const auto box=source.addObject(room,"container");source.find(box)->name="seed box";source.find(box)->state="closed";
    const auto key=source.addObject(box,"key");source.find(key)->name="brass key";
    const auto other=source.addRoom(400,80);source.connect(room,other,"east");
    source.winConditions.push_back({"carrying",key,""});
    const auto original=source.json();
    auto roomAsset=Asset::capture(source,room,"Garden shed");
    auto objectAsset=Asset::capture(source,box,"Seed box");
    check(roomAsset.entities.size()==3 && roomAsset.preview().passages.empty() && roomAsset.preview().winConditions.empty(),"Room asset copied external game data");
    check(objectAsset.entities.size()==2 && objectAsset.entities.front().parent.empty(),"Object asset lost contents or kept external parent");
    check(Asset::parse(objectAsset.json()).json()==objectAsset.json(),"Asset round trip changed data");
    auto bad=objectAsset.json();bad["entities"][1]["parent"]="outside";
    rejects([&]{Asset::parse(bad);},"External asset reference accepted");
    bad=objectAsset.json();bad["entities"][1]["id"]=box;
    rejects([&]{Asset::parse(bad);},"Duplicate asset ID accepted");
    bad=objectAsset.json();bad["entities"][1]["parent"]=key;
    rejects([&]{Asset::parse(bad);},"Asset containment cycle accepted");
    bad=objectAsset.json();bad["version"]=2;
    rejects([&]{Asset::parse(bad);},"Unsupported asset version accepted");
    Editor destination;const auto before=destination.document().json();std::string first,second;
    destination.change([&](Document& d){first=roomAsset.insert(d,"",400,100);});
    destination.change([&](Document& d){second=roomAsset.insert(d,"",800,100);});
    check(first!=second && destination.document().entities.size()==7,"Repeated room insertion collided IDs");
    std::set<std::string> names,ids;
    for(const auto& e:destination.document().entities){check(names.insert(e.name).second,"Repeated insert collided names");check(ids.insert(e.id).second,"Repeated insert collided IDs");}
    destination.undo();destination.undo();check(destination.document().json()==before,"Asset insertion was not atomic undo");destination.redo();
    auto invalid=destination.document();const auto invalidBefore=invalid.json();
    rejects([&]{objectAsset.insert(invalid,"missing",0,0);},"Invalid object destination accepted");
    check(invalid.json()==invalidBefore,"Failed insertion partially mutated world");
    check(source.json()==original && objectAsset.entities.back().name=="brass key","Insertion changed source asset/world");
    const auto root=std::filesystem::temp_directory_path()/("grim-asset-tests-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    const auto file=newAssetPath(root,objectAsset);saveJsonFile(file,objectAsset.json());
    check(Asset::parse(loadJsonFile(file)).json()==objectAsset.json(),"Saved asset did not reload");
    check(newAssetPath(root,objectAsset)!=file,"Save copy chose existing file");
    check(listAssets(root).size()==1,"Asset discovery missed saved object");
    auto renamed=objectAsset;renamed.name="Renamed box";saveJsonFile(file,renamed.json());
    check(listAssets(root).front().name=="objects / Renamed box","Asset rename missing from library");
    {std::ofstream out(root/"objects"/"bad.grimasset.json");out<<"bad";}
    check(listAssets(root).size()==2,"Malformed asset hid healthy library entries");
    // Exercise the real UI with renderer sinks, including copy insertion callback.
    OverlayRenderer renderer;auto& router=GRIM::UI::InputRouter::get();
    UIWorldAssetsView view(root);UIWorldLabView world(root/"recovery.json",root/"games");
    view.onInsert=[&](const Asset& a){world.insertAsset(a);};
    auto frame=[&](const InputState& in){router.beginFrame();router.currentPanel=90;view.update(in,0,{{0,0},{1000,700}},90);view.draw(renderer,{{0,0},{1000,700}});router.endFrame();};
    auto click=[&](Vec2 p){InputState in;in.mousePos=p;in.mousePressed[0]=in.mouseDown[0]=true;frame(in);in.mousePressed[0]=in.mouseDown[0]=false;in.mouseReleased[0]=true;frame(in);};
    view.capture(source,room);frame({});click({130,104}); // Save captured room.
    check(listAssets(root).size()==3,"Assets Save button did not persist captured room");
    click({325,104});check(world.document().entities.size()==4,"Add to world callback failed");
    click({750,396});click({750,496}); // Type dropdown -> container.
    click({770,430});check(view.document().entities.size()==4,"Asset Add inside failed");
    rejects([&]{view.capture(source,box);},"Dirty asset was silently replaced");
    click({750,478});check(view.document().entities.size()==3,"Asset Undo failed");
    click({840,478});check(view.document().entities.size()==4,"Asset Redo failed");
    click({550,104});check(view.document().entities.size()==3,"Discard did not restore saved asset");
    view.capture(source,box);click({750,478}); // Undo cannot cross asset-open boundary.
    check(view.document().entities.size()==3 && view.document().find(box),"Undo crossed asset boundary");
    click({130,104}); // Save object, then delete it to recoverable trash.
    const auto count=listAssets(root).size();click({420,104});
    check(listAssets(root).size()==count-1 && !std::filesystem::is_empty(root/".trash"),"Delete did not archive asset");
    // Library selection reloads a saved room after New.
    click({200,25});click({200,130}); // fourth row: saved room, after placeholder and two objects.
    check(view.document().entities.size()==3 && view.document().find(box),"Saved asset dropdown failed to load room");
    view.hide();world.hide();
    // Remove only files created in this test-owned directory, no recursive deletion.
    for(const auto* folder:{"rooms","objects",".trash","games"}) {
        const auto dir=root/folder;
        if(std::filesystem::exists(dir)){for(const auto& entry:std::filesystem::directory_iterator(dir))std::filesystem::remove(entry.path());std::filesystem::remove(dir);}
    }
    std::filesystem::remove(root/"recovery.json");std::filesystem::remove(root);
    std::cout<<"World asset persistence, isolation, insertion, and UI tests passed\n";
}
