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
    const auto catalog=listAssets(root);
    check(searchAssets(catalog,"brass","container").size()==1,"Nested contents missing from search");
    check(searchAssets(catalog,"brass","room").empty(),"Type filter used child type instead of root");
    check(searchAssets(catalog,"unmatched words","").empty(),"Unrelated entries returned");
    auto ranking=catalog;auto related=ranking.front();related.name="Other asset";related.searchText="Renamed box";
    ranking.push_back(related);
    check(searchAssets(ranking,"Renamed box","").front()==0,"Name relevance did not outrank contents");
    {
        const auto tiles=root/"tile-tests";
        for(int i=0;i<12;++i){auto a=roomAsset;a.name="Tile "+std::to_string(100+i);saveJsonFile(newAssetPath(tiles,a),a.json());}
        UIWorldAssetBrowser browser(tiles);browser.setPosition(0,0);browser.setSize(420,150);
        std::string dropped;
        browser.canDrop=[](const Asset&,Vec2 p){return p.x>420;};
        browser.onDrop=[&](const Asset& a,Vec2){dropped=a.name;};
        auto& inputRouter=GRIM::UI::InputRouter::get();OverlayRenderer sink;
        auto tick=[&](const InputState& in){inputRouter.beginFrame();inputRouter.currentPanel=92;browser.update(in,0);browser.drawOverlay(sink,{});browser.drawFloating(sink);inputRouter.endFrame();};
        tick({});InputState in;in.mousePos={20,80};in.mouseWheelDelta=-360;tick(in);
        in.mouseWheelDelta=0;in.mousePressed[0]=in.mouseDown[0]=true;tick(in);
        in.mousePressed[0]=false;in.mousePos={500,200};tick(in);
        in.mouseDown[0]=false;in.mouseReleased[0]=true;tick(in);
        check(dropped=="Tile 104","Scrolled grid chose the wrong asset");
        in={};in.mousePos={20,80};in.mousePressed[0]=in.mouseDown[0]=true;tick(in);
        browser.hide();check(!browser.hasPointerCapture(),"Hidden browser retained drag capture");
        for(const auto& file:std::filesystem::directory_iterator(tiles/"rooms"))std::filesystem::remove(file.path());
        std::filesystem::remove(tiles/"rooms");std::filesystem::remove(tiles);
    }
    // Exercise the real UI with renderer sinks, including copy insertion callback.
    OverlayRenderer renderer;auto& router=GRIM::UI::InputRouter::get();
    UIWorldAssetsView view(root);UIWorldLabView world(root/"recovery.json",root/"games");
    view.onInsert=[&](const Asset& a){world.insertAsset(a);};
    auto frame=[&](const InputState& in){router.beginFrame();router.currentPanel=90;view.update(in,0,{{0,0},{1000,900}},90);view.draw(renderer,{{0,0},{1000,900}});router.endFrame();};
    auto click=[&](Vec2 p){InputState in;in.mousePos=p;in.mousePressed[0]=in.mouseDown[0]=true;frame(in);in.mousePressed[0]=in.mouseDown[0]=false;in.mouseReleased[0]=true;frame(in);};
    view.capture(source,room);frame({});click({130,104}); // Save captured room.
    check(listAssets(root).size()==3,"Assets Save button did not persist captured room");
    // Real browser gestures: search, cancellation, placement, invalid holders,
    // nested object copy and a single undo for the entire insertion.
    {
        UIWorldLabView browserWorld(root/"browser-recovery.json",root/"games",root);
        auto browserFrame=[&](const InputState& in){router.beginFrame();router.currentPanel=91;browserWorld.update(in,0,{{0,0},{1000,700}},91);browserWorld.draw(renderer,{{0,0},{1000,700}});router.endFrame();};
        auto browserClick=[&](Vec2 p){InputState in;in.mousePos=p;in.mousePressed[0]=in.mouseDown[0]=true;browserFrame(in);in.mousePressed[0]=in.mouseDown[0]=false;in.mouseReleased[0]=true;browserFrame(in);};
        auto query=[&](const std::string& text){browserClick({40,490});InputState in;in.ctrl=true;in.keyPressed['A']=true;browserFrame(in);in={};in.textInput=text;browserFrame(in);};
        auto drop=[&](Vec2 point,bool escape=false){InputState in;in.mousePos={40,535};in.mousePressed[0]=in.mouseDown[0]=true;browserFrame(in);in.mousePressed[0]=false;in.mousePos=point;in.keyPressed[VK_ESCAPE]=escape;browserFrame(in);in.keyPressed[VK_ESCAPE]=false;in.mouseDown[0]=false;in.mouseReleased[0]=true;browserFrame(in);};
        browserFrame({});query("Starting room");
        // The inspector divider changes the actual canvas drop bounds.
        InputState resize;resize.mousePos={718,250};resize.mousePressed[0]=resize.mouseDown[0]=true;browserFrame(resize);
        resize.mousePressed[0]=false;resize.mousePos.x=618;browserFrame(resize);
        resize.mouseDown[0]=false;resize.mouseReleased[0]=true;browserFrame(resize);
        drop({650,250});check(browserWorld.document().entities.size()==1,"Resized inspector did not constrain canvas drops");
        resize={};resize.mousePos={618,250};resize.mousePressed[1]=true;browserFrame(resize);
        browserClick({900,490});browserClick({900,615}); // Container filter excludes room roots.
        drop({400,250});check(browserWorld.document().entities.size()==1,"Browser type filter failed");
        browserClick({900,490});
        InputState filterWheel;filterWheel.mousePos={900,515};filterWheel.mouseWheelDelta=240;browserFrame(filterWheel);
        browserClick({900,515}); // All types, after scrolling to the first option.
        drop({400,250},true);check(browserWorld.document().entities.size()==1,"Escape inserted an asset");
        drop({400,250});check(browserWorld.document().entities.size()==4,"Browser room drop failed");
        const auto placed=browserWorld.document().layout.back();
        check(placed.x==392 && placed.y==130,"Browser ignored drop coordinates");
        browserClick({240,57});check(browserWorld.document().entities.size()==1,"Browser insertion was not one undo");
        query("Renamed box");drop({600,250});check(browserWorld.document().entities.size()==1,"Object inserted without holder");
        drop({110,215});check(browserWorld.document().entities.size()==3,"Browser object drop lost nested contents");
        browserWorld.hide();
        std::filesystem::remove(root/"browser-recovery.json");
    }
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
    // The Assets browser opens templates, preserving dirty drafts and file identity.
    frame({});
    InputState resizeLibrary;resizeLibrary.mousePos={400,634};resizeLibrary.mousePressed[0]=resizeLibrary.mouseDown[0]=true;frame(resizeLibrary);
    resizeLibrary.mousePressed[0]=false;resizeLibrary.mousePos.y=574;frame(resizeLibrary);
    resizeLibrary.mouseDown[0]=false;resizeLibrary.mouseReleased[0]=true;frame(resizeLibrary);
    click({40,660}); // First tile has moved up with the library divider.
    check(view.document().title=="Renamed box","Assets tile did not open the saved template");
    resizeLibrary={};resizeLibrary.mousePos={400,574};resizeLibrary.mousePressed[1]=true;frame(resizeLibrary);
    const auto opened=view.document().json();click({220,720});
    check(view.document().json()==opened,"Invalid tile replaced the asset draft");
    click({40,60});InputState rename;rename.ctrl=true;rename.keyPressed['A']=true;frame(rename);
    rename={};rename.textInput="Edited from browser";frame(rename);
    click({420,720});check(view.document().json()==opened,"Tile selection discarded pending edits");
    click({130,104});
    check(Asset::parse(loadJsonFile(file)).name=="Edited from browser","Browser-opened asset saved to the wrong file");
    frame({});click({420,720});
    check(view.document().title=="Starting room","Clean tile selection did not load room");
    view.hide();world.hide();
    // Remove only files created in this test-owned directory, no recursive deletion.
    for(const auto* folder:{"rooms","objects",".trash","games"}) {
        const auto dir=root/folder;
        if(std::filesystem::exists(dir)){for(const auto& entry:std::filesystem::directory_iterator(dir))std::filesystem::remove(entry.path());std::filesystem::remove(dir);}
    }
    std::filesystem::remove(root/"recovery.json");std::filesystem::remove(root);
    std::cout<<"World asset persistence, isolation, insertion, and UI tests passed\n";
}
