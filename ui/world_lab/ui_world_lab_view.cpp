#include "ui_world_lab_view.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <algorithm>
#include <cstdlib>
#include <cctype>

using namespace GRIM::WorldLab;
namespace {
std::string pathText(const std::filesystem::path& path) {
    const auto text=path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()),text.size());
}
const std::vector<std::string> kinds{"object","key","container","supporter","food"};
void place(Widget& w,float x,float y,float width,float height=28) {w.setPosition(x,y);w.setSize(width,height);}
}
UIWorldLabView::UIWorldLabView(std::filesystem::path recoveryPath, std::filesystem::path gamesDirectory)
    : recoveryPath_(std::move(recoveryPath)), gamesDirectory_(std::move(gamesDirectory)) {
    if(gamesDirectory_.empty())gamesDirectory_=std::filesystem::current_path()/"resources"/"models"/"games";
    games_=std::make_unique<UIDropdown>("",std::vector<std::string>{"Saved games"},0,[this](int index,const std::string&){
        if(index>0 && index<=static_cast<int>(gameFiles_.size())) {
            const auto file=gameFiles_[index-1];
            act([&]{openGame(file);});
        }
        games_->setSelectedIndex(0);
    });
    canvas_.selected=editor_.document().spawn;
    canvas_.onSelect=[this](const std::string& id){
        if(!apply()) return false;
        canvas_.selected=id; refresh(); return true;
    };
    canvas_.onError=[this](const std::string& message){status_=message;};
    canvas_.onChanged=[this]{refresh();};
    act([this]{std::filesystem::create_directories(gamesDirectory_);newGamePath();refreshGames();});
    path_.setPlaceholder("World file path (.grimworld.json)");
    name_.setPlaceholder("Name"); description_.setPlaceholder("Description");
    auto button=[this](const std::string& text,std::function<void()> action){
        toolbar_.push_back(std::make_unique<UIButton>(text,[this,action]{act(action);}));
    };
    button("New",[this]{if(!apply())return;editor_.newWorld();newGamePath();canvas_.cancelInput();canvas_.selected=editor_.document().spawn;refresh();canvas_.fitWorld();status_="New world. Undo restores the previous world.";});
    button("Open",[this]{openGame(std::filesystem::u8path(path_.getText()));});
    button("Save",[this]{if(!apply())return;editor_.save(std::filesystem::u8path(path_.getText()));status_="Saved world.";refreshGames();autosave();});
    button("Undo",[this]{if(!apply())return;canvas_.cancelInput();editor_.undo();refresh();});
    button("Redo",[this]{if(!apply())return;canvas_.cancelInput();editor_.redo();refresh();});
    button("+ Room",[this]{if(!apply())return;std::string id;const auto r=canvas_.visibleWorldRect();editor_.change([&](Document& d){id=d.addRoom(r.origin.x+r.size.x/2-115,r.origin.y+r.size.y/2-45);});canvas_.selected=id;refresh();});
    button("Fit",[this]{canvas_.fitWorld();});
    button("Recover",[this]{if(!apply())return;editor_.load(recoveryPath_);editor_.markUnsaved();canvas_.cancelInput();canvas_.selected=editor_.document().spawn;refresh();canvas_.fitWorld();status_="Recovered last autosave. Save to the chosen world file; Undo restores prior edits.";});
    session_=std::make_unique<UIWorldSession>(editor_);
    session_->onErrorEntity=[this](const std::string& id){canvas_.selected=id;status_="TextWorld validation points to "+id;refresh();};
    session_->onState=[this](const nlohmann::json& state){canvas_.setPlayState(state);};
    actions_.beforePrepare=[this]{return apply();};
    actions_.onPrepare=[this](const std::string& command){
        sessionVisible_=true;session_->setChallenge(false);session_->prepareCommand(command);
    };
    button("Challenge",[this]{if(!apply())return;sessionVisible_=true;session_->setChallenge(true);});
    button("Play",[this]{if(!apply())return;sessionVisible_=true;session_->setChallenge(false);});
    button("Inspect",[this]{if(!apply())return;sessionVisible_=false;refresh();});
    kind_=std::make_unique<UIDropdown>("",kinds,0,[](int,const std::string&){});
    location_=std::make_unique<UIDropdown>("",std::vector<std::string>{},0,[this](int index,const std::string&){
        if(index<0 || index>=static_cast<int>(locations_.size()))return;
        const auto parent=locations_[index];
        act([&]{if(!apply())return;editor_.change([&](Document& d){auto* e=d.find(canvas_.selected);if(e && e->kind!="room")e->parent=parent;});refresh();});
    });
    state_=std::make_unique<UIDropdown>("",std::vector<std::string>{"open","closed"},0,[this](int,const std::string& state){
        act([&]{if(!apply())return;editor_.change([&](Document& d){auto* e=d.find(canvas_.selected);if(e && e->kind=="container")e->state=state;});refresh();});
    });
    doorState_=std::make_unique<UIDropdown>("",std::vector<std::string>{"open","closed","locked"},0,[](int,const std::string&){});
    doorKey_=std::make_unique<UIDropdown>("",std::vector<std::string>{"No key"},0,[](int,const std::string&){});
    apply_=std::make_unique<UIButton>("Apply changes",[this]{apply();});
    actionsShortcut_=std::make_unique<UIButton>("Actions",[this]{if(apply())inspectorScroll_.scrollTo(500);});
    add_=std::make_unique<UIButton>("+ Add here",[this]{act([this]{addObject();});});
    spawn_=std::make_unique<UIButton>("Set player start",[this]{act([this]{if(!apply())return;editor_.change([&](Document& d){d.spawn=canvas_.selected;});refresh();});});
    erase_=std::make_unique<UIButton>("Delete selection",[this]{act([this]{if(!apply())return;editor_.change([&](Document& d){d.erase(canvas_.selected);});refresh();status_="Deleted selection and its contents. Undo restores them.";});});
    saveAsset_=std::make_unique<UIButton>("Save as asset",[this]{act([this]{
        if(!apply())return;
        if(!onSaveAsset)throw std::runtime_error("Asset library is unavailable");
        onSaveAsset(editor_.document(),canvas_.selected);
    });});
    title_.OnTextSubmitted.Bind([this](const std::string&){apply();});
    name_.OnTextSubmitted.Bind([this](const std::string&){apply();});
    description_.OnTextSubmitted.Bind([this](const std::string&){apply();});
    std::filesystem::path recoveryRoot=std::filesystem::temp_directory_path();
#ifdef _WIN32
    wchar_t* local=nullptr;size_t length=0;
    if(_wdupenv_s(&local,&length,L"LOCALAPPDATA")==0 && local)recoveryRoot=local;
    std::free(local);
#else
    if(const char* local=std::getenv("XDG_STATE_HOME"))recoveryRoot=std::filesystem::u8path(local);
#endif
    if(recoveryPath_.empty())recoveryPath_=recoveryRoot / "GRIM" / "WorldLab" / "recovery.grimworld.json";
    // Do not overwrite a previous session's recovery until the user edits this one.
    recoveredSnapshot_=editor_.document().json();
    refresh();
}
std::vector<Widget*> UIWorldLabView::controls() {
    std::vector<Widget*> result{&path_,&title_,games_.get()};
    if(!sessionVisible_)for(auto* w:std::vector<Widget*>{&name_,&description_,kind_.get(),location_.get(),state_.get(),doorState_.get(),doorKey_.get(),apply_.get(),add_.get(),spawn_.get(),erase_.get(),saveAsset_.get(),actionsShortcut_.get()})result.push_back(w);
    for(auto& b:toolbar_)result.push_back(b.get());
    return result;
}
UIWorldLabView::~UIWorldLabView() { hide(); }
void UIWorldLabView::act(const std::function<void()>& action) {
    try {action();}catch(const std::exception& e){status_=e.what();}
}
bool UIWorldLabView::apply() {
    try {
        if(session_ && !session_->apply())return false;
        const auto id=canvas_.selected;
        editor_.change([&](Document& d){
            d.title=title_.getText();
            if(auto* e=d.find(id)) {e->name=name_.getText();e->description=description_.getText();}
            for(auto& p:d.passages)if(p.id==id){
                p.doorName=name_.getText();p.doorState=p.doorName.empty()?"open":doorState_->getSelectedItem();
                const int key=doorKey_->getSelectedIndex();
                p.key=!p.doorName.empty() && key>=0 && key<static_cast<int>(doorKeys_.size())?doorKeys_[key]:"";
            }
        });
        return true;
    }catch(const std::exception& e){status_=e.what();return false;}
}
void UIWorldLabView::refresh() {
    const auto& d=editor_.document();
    const bool passage=std::any_of(d.passages.begin(),d.passages.end(),[&](const auto& p){return p.id==canvas_.selected;});
    const auto* e=d.find(canvas_.selected);
    if(!e && !passage)canvas_.selected.clear();
    title_.setText(d.title);name_.setText(e?e->name:"");description_.setText(e?e->description:"");
    const bool holder=e && (e->kind=="room" || e->kind=="container" || e->kind=="supporter");
    name_.setVisible(e!=nullptr || passage);description_.setVisible(e!=nullptr);apply_->setVisible(true);
    doorState_->setVisible(passage);doorKey_->setVisible(passage);
    doorKeys_={""};std::vector<std::string> keyNames{"No key"};
    for(const auto& key:d.entities)if(key.kind=="key"){doorKeys_.push_back(key.id);keyNames.push_back(key.name+" ["+key.id+"]");}
    doorKey_->setItems(keyNames);doorKey_->setSelectedIndex(0);
    for(const auto& p:d.passages)if(p.id==canvas_.selected){
        name_.setText(p.doorName);doorState_->setSelectedIndex(p.doorState=="locked"?2:p.doorState=="closed"?1:0);
        auto it=std::find(doorKeys_.begin(),doorKeys_.end(),p.key);if(it!=doorKeys_.end())doorKey_->setSelectedIndex(static_cast<int>(it-doorKeys_.begin()));
    }
    kind_->setVisible(holder);add_->setVisible(holder);spawn_->setVisible(e && e->kind=="room");
    saveAsset_->setVisible(e!=nullptr);
    erase_->setVisible(e || passage);location_->setVisible(e && e->kind!="room");state_->setVisible(e && e->kind=="container");
    locations_.clear();std::vector<std::string> labels;int selected=0;
    if(e && e->kind!="room") for(const auto& candidate:d.entities) {
        if(candidate.kind!="room" && candidate.kind!="container" && candidate.kind!="supporter")continue;
        bool descendant=false;
        for(auto* p=&candidate;p;p=d.find(p->parent)) if(p->id==e->id){descendant=true;break;}
        if(descendant)continue;
        if(candidate.id==e->parent)selected=static_cast<int>(locations_.size());
        locations_.push_back(candidate.id);labels.push_back(candidate.name+" ["+candidate.id+"]");
    }
    location_->setItems(labels);location_->setSelectedIndex(selected);
    if(e)state_->setSelectedIndex(e->state=="closed"?1:0);
    if(session_)session_->refresh();
    actions_.select(canvas_.selected);
}
void UIWorldLabView::addObject() {
    if(!apply())return;
    std::string id;
    editor_.change([&](Document& d){id=d.addObject(canvas_.selected,kind_->getSelectedItem());});
    canvas_.selected=id;refresh();
}
void UIWorldLabView::layout(const PanelRect& content) {
    area_=content;
    float x=content.origin.x+8,y=content.origin.y+8;
    const float w=std::max(0.f,content.size.x-16);
    place(title_,x,y,std::max(120.f,w-280));
    float bx=x;
    for(size_t i=0;i<toolbar_.size();++i){const float bw=i>=8?84.f:i==5?78.f:64.f;place(*toolbar_[i],bx,y+36,bw);bx+=bw+5;}
    place(*games_,x,y+72,168);
    place(path_,x+180,y+72,std::max(40.f,w-180));
    const float paneWidth=sessionVisible_?340.f:268.f;
    const float canvasWidth=std::max(80.f,w-paneWidth-12);
    canvas_.setPosition(x,y+112);canvas_.setSize(canvasWidth,std::max(40.f,content.size.y-174));
    inspector_={{x+canvasWidth+12,y+112},{paneWidth,std::max(40.f,content.size.y-174)}};
    inspectorScroll_.setPosition(inspector_.origin);inspectorScroll_.setSize(inspector_.size);inspectorScroll_.setContentHeight(1040);
    x=inspector_.origin.x+10;y=inspector_.origin.y-inspectorScroll_.getScrollOffset();
    place(name_,x,y+48,232);place(description_,x,y+102,232);
    place(*apply_,x,y+140,150);
    place(*actionsShortcut_,x+156,y+140,76);
    place(*location_,x,y+204,168);place(*state_,x,y+260,168);
    place(*kind_,x,y+322,130);place(*add_,x+138,y+322,94);
    place(*spawn_,x,y+366,150);place(*erase_,x,y+408,150);
    place(*saveAsset_,x,y+452,150);
    place(*doorState_,x,y+204,168);place(*doorKey_,x,y+260,168);
}
void UIWorldLabView::newGamePath() {
    auto candidate=gamesDirectory_/"world.grimworld.json";
    for(unsigned i=2;std::filesystem::exists(candidate);++i)
        candidate=gamesDirectory_/("world-"+std::to_string(i)+".grimworld.json");
    path_.setText(pathText(candidate));
}
void UIWorldLabView::refreshGames() {
    std::vector<std::filesystem::path> files;
    if(std::filesystem::exists(gamesDirectory_))for(const auto& entry:std::filesystem::directory_iterator(gamesDirectory_)) {
        auto name=pathText(entry.path().filename());
        std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
        if(entry.is_regular_file() && name.ends_with(".grimworld.json"))files.push_back(entry.path());
    }
    std::sort(files.begin(),files.end());
    if(files==gameFiles_)return;
    std::vector<std::string> labels{"Saved games"};
    for(const auto& file:files) {
        auto name=pathText(file.filename());name.resize(name.size()-15);
        if(name.size()>20)name=name.substr(0,17)+"...";
        labels.push_back(name);
    }
    gameFiles_=std::move(files);games_->setItems(labels);games_->setSelectedIndex(0);
}
void UIWorldLabView::openGame(const std::filesystem::path& file) {
    if(!apply())return;
    editor_.load(file);path_.setText(pathText(file));
    canvas_.cancelInput();canvas_.selected=editor_.document().spawn;
    refresh();canvas_.fitWorld();status_="Opened world. Undo restores previous edits.";
}
void UIWorldLabView::insertAsset(const Asset& asset) {
    if(!apply())throw std::runtime_error("Apply valid World Lab edits before inserting an asset");
    const auto rect=canvas_.visibleWorldRect();std::string id;
    editor_.change([&](Document& d){id=asset.insert(d,canvas_.selected,rect.origin.x+rect.size.x/2-115,rect.origin.y+rect.size.y/2-45);});
    canvas_.cancelInput();canvas_.selected=id;refresh();canvas_.fitWorld();
    status_="Inserted asset. Undo removes the entire copy. Save the game to keep it.";
}
void UIWorldLabView::autosave() {
    const auto snapshot=editor_.document().json();
    if(snapshot==recoveredSnapshot_)return;
    std::filesystem::create_directories(recoveryPath_.parent_path());
    auto copy=editor_;copy.save(recoveryPath_);recoveredSnapshot_=snapshot;
}
void UIWorldLabView::update(const InputState& input,float dt,const PanelRect& content,uint64_t panel) {
    layout(content);canvas_.setPanelID(panel);
    const auto clipped=input.clipped(content.origin,content.size);
    gamesTimer_+=std::max(0.f,dt);
    if(gamesTimer_>=2 && !games_->isExpanded()){gamesTimer_=0;act([this]{refreshGames();});}
    // Open dropdowns receive input before siblings so dismissals never click through.
    std::vector<UIDropdown*> dropdowns=sessionVisible_?std::vector<UIDropdown*>{}:std::vector<UIDropdown*>{kind_.get(),location_.get(),state_.get(),doorState_.get(),doorKey_.get()};
    dropdowns.insert(dropdowns.begin(),games_.get());
    for(auto* dropdown:dropdowns)dropdown->setPanelID(panel);
    auto dropdownInput=[&](UIDropdown* dropdown){return dropdown==games_.get()?clipped:clipped.clipped(inspector_.origin,inspector_.size);};
    UIDropdown* expanded=nullptr;
    for(auto* dropdown:dropdowns)if(dropdown->isVisible() && dropdown->isExpanded()){expanded=dropdown;dropdown->update(dropdownInput(dropdown),dt);break;}
    for(auto* w:controls()) {
        w->setPanelID(panel);if(!w->isVisible())continue;
        auto* dropdown=dynamic_cast<UIDropdown*>(w);
        if(dropdown)continue;
        const bool isInspector=w==&name_ || w==&description_ || w==apply_.get() || w==add_.get() || w==spawn_.get() || w==erase_.get() || w==saveAsset_.get() || w==actionsShortcut_.get();
        w->update(isInspector?clipped.clipped(inspector_.origin,inspector_.size):clipped,dt);
    }
    // A popup may have closed above; don't process the same press twice.
    for(auto* dropdown:dropdowns)if(dropdown!=expanded && dropdown->isVisible())dropdown->update(dropdownInput(dropdown),dt);
    if(sessionVisible_)session_->update(clipped,dt,inspector_);
    else{
        actions_.update(clipped.clipped(inspector_.origin,inspector_.size),dt,
            {{inspector_.origin.x+10,inspector_.origin.y+500-inspectorScroll_.getScrollOffset()},{232,530}},panel);
        inspectorScroll_.setPanelID(panel);inspectorScroll_.update(clipped,dt);
    }
    canvas_.update(clipped,dt);
    recoveryTimer_+=std::max(0.f,dt);
    if(recoveryTimer_>=10){recoveryTimer_=0;act([this]{if(apply())autosave();});}
}
void UIWorldLabView::hide() {
    actions_.hide();
    act([this]{if(apply())autosave();});
    canvas_.cancelInput();inspectorScroll_.cancelInput();for(auto* w:controls())w->cancelInput();
    games_->collapse();gamesTimer_=2;
    kind_->collapse();location_->collapse();state_->collapse();
    doorState_->cancelInput();doorState_->collapse();doorKey_->cancelInput();doorKey_->collapse();
    if(session_)session_->hide();
}
void UIWorldLabView::draw(OverlayRenderer& r,const PanelRect& content) {
    using namespace UITheme::Colors;
    layout(content);
    r.pushClipRect(content.origin,content.size);
    for(size_t i=0;i<toolbar_.size();++i) toolbar_[i]->drawOverlay(r,{});
    games_->drawOverlay(r,{});path_.drawOverlay(r,{});title_.drawOverlay(r,{});
    const auto& d=editor_.document();
    bool pending=title_.getText()!=d.title || session_->hasPendingEdits() || (d.find(canvas_.selected) && (name_.getText()!=d.find(canvas_.selected)->name || description_.getText()!=d.find(canvas_.selected)->description));
    for(const auto& p:d.passages)if(p.id==canvas_.selected){
        const int key=doorKey_->getSelectedIndex();
        pending |= name_.getText()!=p.doorName || doorState_->getSelectedItem()!=p.doorState ||
            (key>=0 && key<static_cast<int>(doorKeys_.size()) && doorKeys_[key]!=p.key);
    }
    r.drawText({content.origin.x+std::max(150.f,content.size.x-265),content.origin.y+14},editor_.dirty()||pending?"Unsaved changes":"Saved",editor_.dirty()||pending?Warning:Success,13);
    canvas_.drawOverlay(r,{});
    if(sessionVisible_)session_->draw(r,inspector_);
    else {
    r.pushClipRect(inspector_.origin,inspector_.size);
    inspectorScroll_.drawOverlay(r,{});
    const float x=inspector_.origin.x+10,y=inspector_.origin.y-inspectorScroll_.getScrollOffset();
    const auto* e=d.find(canvas_.selected);
    r.drawText({x,y+10},e?"Inspect / "+e->kind:canvas_.selected.empty()?"World":"Passage",TextPrimary,15);
    if(e){
        r.drawText({x,y+32},"Name",TextMuted,11);name_.drawOverlay(r,{});
        r.drawText({x,y+86},"Description",TextMuted,11);description_.drawOverlay(r,{});
        if(e->kind!="room")r.drawText({x,y+184},"Location",TextMuted,12);
        if(e->kind=="container")r.drawText({x,y+240},"Starting state",TextMuted,12);
        if(kind_->isVisible())r.drawText({x,y+302},"Add an object",TextMuted,12);
    } else if(!canvas_.selected.empty()) {
        for(const auto& p:d.passages)if(p.id==canvas_.selected){
            r.drawText({x,y+32},"Door name (blank = passage)",TextMuted,11);name_.drawOverlay(r,{});
            r.drawText({x,y+96},p.direction+" / "+opposite(p.direction),PrimaryLight,13);
            r.drawText({x,y+184},"Starting door state",TextMuted,12);
            r.drawText({x,y+240},"Matching key",TextMuted,12);
        }
    } else r.drawText({x,y+48},"Select a room or object.",TextMuted,13);
    for(auto* w:std::vector<Widget*>{apply_.get(),kind_.get(),location_.get(),state_.get(),doorState_.get(),doorKey_.get(),add_.get(),spawn_.get(),erase_.get(),saveAsset_.get(),actionsShortcut_.get()})if(w->isVisible())w->drawOverlay(r,{});
    actions_.draw(r,{{x,y+500},{232,530}});
    for(auto* dropdown:{kind_.get(),location_.get(),state_.get(),doorState_.get(),doorKey_.get()})if(dropdown->isVisible() && dropdown->isExpanded())dropdown->drawExpandedList(r,{});
    r.popClipRect();
    }
    const float bottom=content.origin.y+content.size.y;
    r.drawText({content.origin.x+8,bottom-42},status_,TextSecondary,12);
    r.drawText({content.origin.x+8,bottom-22},"Drag rooms / objects  |  Drag compass handles to connect  |  Middle-drag to pan  |  Wheel to zoom",TextMuted,12);
    if(games_->isExpanded())games_->drawExpandedList(r,{});
    r.popClipRect();
}
