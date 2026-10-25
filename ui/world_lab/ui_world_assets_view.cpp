#include "ui_world_assets_view.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

using namespace GRIM::WorldLab;
namespace {
const std::vector<std::string> types{"room","object","key","container","supporter","food"};
void place(Widget& w,float x,float y,float width,float height=28){w.setPosition(x,y);w.setSize(width,height);}
}
UIWorldAssetsView::UIWorldAssetsView(std::filesystem::path directory):directory_(std::move(directory)) {
    browser_=std::make_unique<UIWorldAssetBrowser>(directory_);
    browser_->onOpen=[this](const Asset& asset,const std::filesystem::path& file){
        requireClean();setAsset(asset,file);status_="Loaded asset for editing.";
    };
    browser_->onError=[this](const std::string& message){status_=message;};
    library_=std::make_unique<UIDropdown>("Saved assets",std::vector<std::string>{"Choose asset"},0,[this](int index,const std::string&){
        act([&]{if(index<=0 || index>static_cast<int>(entries_.size()))return;requireClean();
            const auto file=entries_[index-1].path;const auto candidate=Asset::parse(loadJsonFile(file));setAsset(candidate,file);status_="Loaded asset.";});
        library_->setSelectedIndex(0);
    });
    kind_=std::make_unique<UIDropdown>("",types,0,[](int,const std::string&){});
    state_=std::make_unique<UIDropdown>("",std::vector<std::string>{"open","closed"},0,[](int,const std::string&){});
    auto button=[&](const std::string& label,std::function<void()> action){buttons_.push_back(std::make_unique<UIButton>(label,[this,action]{act(action);}));};
    button("New asset",[this]{requireClean();setAsset(Asset::create(kind_->getSelectedItem()));status_="New asset. Edit its name and contents, then Save.";});
    button("Save",[this]{save(false);});
    button("Save copy",[this]{save(true);});
    button("Add to world",[this]{if(!apply())return;if(!onInsert)throw std::runtime_error("World Lab is unavailable");onInsert(asset());status_="Inserted an independent copy into World Lab.";});
    button("Delete asset",[this]{
        requireClean();if(path_.empty())throw std::runtime_error("Select a saved asset first");
        const auto trash=directory_/".trash";std::filesystem::create_directories(trash);
        auto destination=trash/path_.filename();destination+="."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::rename(path_,destination);setAsset(Asset::create("room"));refreshLibrary();status_="Asset moved to world_assets/.trash. Existing games keep their copies.";
    });
    button("Discard edits",[this]{editor_.reset(Document::parse(baseline_));root_=baselineRoot_;canvas_.selected=root_;refresh();status_="Restored the last saved or opened asset.";});
    button("Apply",[this]{if(apply())status_="Changes applied. Save to update the library.";});
    button("Add inside",[this]{
        if(!apply())return;
        if(kind_->getSelectedItem()=="room")throw std::runtime_error("Choose an object type to add inside a room or container");
        std::string id;editor_.change([&](Document& d){id=d.addObject(canvas_.selected,kind_->getSelectedItem());});canvas_.selected=id;refresh();
    });
    button("Remove item",[this]{
        if(!apply())return;
        if(canvas_.selected==root_)throw std::runtime_error("The asset root cannot be removed");
        editor_.change([&](Document& d){d.erase(canvas_.selected);});canvas_.selected=root_;refresh();
    });
    button("Undo",[this]{if(!apply())return;editor_.undo();refresh();});
    button("Redo",[this]{if(!apply())return;editor_.redo();refresh();});
    button("Actions",[this]{if(apply())scroll_.scrollTo(390);});
    canvas_.selectionOnly=true;
    canvas_.onSelect=[this](const std::string& id){if(!apply() || (id!=root_ && id==editor_.document().spawn))return false;canvas_.selected=id;refresh();return true;};
    canvas_.onError=[this](const std::string& message){status_=message;};
    assetName_.setPlaceholder("Asset library name");name_.setPlaceholder("In-game name");description_.setPlaceholder("Description");
    setAsset(Asset::create("room"));
    act([this]{std::filesystem::create_directories(directory_/"rooms");std::filesystem::create_directories(directory_/"objects");refreshLibrary();});
}
void UIWorldAssetsView::act(const std::function<void()>& action){try{action();}catch(const std::exception& e){status_=e.what();}}
Asset UIWorldAssetsView::asset() const {return Asset::capture(editor_.document(),root_,editor_.document().title);}
bool UIWorldAssetsView::dirty() const {
    const auto* e=editor_.document().find(canvas_.selected);
    return editor_.document().json()!=baseline_ || assetName_.getText()!=editor_.document().title ||
        (e && (name_.getText()!=e->name || description_.getText()!=e->description || (e->kind=="container" && state_->getSelectedItem()!=e->state)));
}
void UIWorldAssetsView::requireClean() const {if(dirty())throw std::runtime_error("Save or Discard edits before replacing this asset.");}
void UIWorldAssetsView::setAsset(const Asset& a,const std::filesystem::path& path) {
    const auto candidate=a.preview();editor_.reset(candidate);
    const auto serialized=candidate.json();
    root_=a.root;path_=path;browser_->selectPath(path_);baseline_=serialized;baselineRoot_=root_;
    canvas_.cancelInput();canvas_.selected=root_;refresh();canvas_.fitWorld();
}
void UIWorldAssetsView::capture(const Document& d,const std::string& root) {
    requireClean();const auto* entity=d.find(root);if(!entity)throw std::runtime_error("Select a room or object first");
    setAsset(Asset::capture(d,root,entity->name));status_="Selection copied with all contents. Passages and goals are not included. Choose Save.";
}
bool UIWorldAssetsView::apply() {
    try {
        editor_.change([&](Document& d){
            d.title=assetName_.getText();
            if(auto* e=d.find(canvas_.selected)){e->name=name_.getText();e->description=description_.getText();if(e->kind=="container")e->state=state_->getSelectedItem();}
            Asset::capture(d,root_,d.title).preview();
        });return true;
    }catch(const std::exception& e){status_=e.what();return false;}
}
void UIWorldAssetsView::save(bool copy) {
    if(!apply())return;
    auto a=asset();auto target=copy || path_.empty()?newAssetPath(directory_,a):path_;
    saveJsonFile(target,a.json());path_=target;baseline_=editor_.document().json();baselineRoot_=root_;refreshLibrary();status_="Saved asset. Library name changes appear in the browser.";
}
void UIWorldAssetsView::refresh() {
    const auto& d=editor_.document();if(!d.find(canvas_.selected))canvas_.selected=root_;
    assetName_.setText(d.title);const auto* e=d.find(canvas_.selected);
    name_.setText(e?e->name:"");description_.setText(e?e->description:"");
    state_->setVisible(e && e->kind=="container");state_->setSelectedIndex(e && e->state=="closed"?1:0);
    actions_.select(canvas_.selected);
}
void UIWorldAssetsView::refreshLibrary() {
    browser_->refreshLibrary();browser_->selectPath(path_);
    auto entries=listAssets(directory_);std::vector<std::string> labels{"Choose asset"};
    for(const auto& entry:entries)labels.push_back(entry.name);
    entries_=std::move(entries);library_->setItems(labels);library_->setSelectedIndex(0);
}
std::vector<Widget*> UIWorldAssetsView::controls() {
    std::vector<Widget*> result{library_.get(),kind_.get(),state_.get(),&assetName_,&name_,&description_};
    for(auto& b:buttons_)result.push_back(b.get());return result;
}
void UIWorldAssetsView::layout(const PanelRect& area) {
    const float x=area.origin.x+8,y=area.origin.y+8,w=std::max(0.f,area.size.x-16);
    place(*library_,x,y,w);place(assetName_,x,y+44,std::max(80.f,w-140));
    float bx=x;const float widths[]={92,60,90,110,100,110};
    for(int i=0;i<6;++i){place(*buttons_[i],bx,y+82,widths[i]);bx+=widths[i]+6;}
    librarySplit_.setPosition(x,y+126);librarySplit_.setSize(w,std::max(0.f,area.size.y-184));
    librarySplit_.setMinimumSizes(100,130);
    librarySplit_.setDefaultSecondSize(std::min(210.f,std::max(130.f,area.size.y*0.28f)));
    const auto upper=librarySplit_.firstRect(),library=librarySplit_.secondRect();
    browser_->setPosition(library.origin);browser_->setSize(library.size);
    inspectorSplit_.setPosition(upper.origin);inspectorSplit_.setSize(upper.size);
    inspectorSplit_.setMinimumSizes(160,268);inspectorSplit_.setDefaultSecondSize(268);
    const auto canvas=inspectorSplit_.firstRect();inspector_=inspectorSplit_.secondRect();
    canvas_.setPosition(canvas.origin);canvas_.setSize(canvas.size);
    scroll_.setPosition(inspector_.origin);scroll_.setSize(inspector_.size);scroll_.setContentHeight(930);
    const auto ix=inspector_.origin.x+8,iy=inspector_.origin.y-scroll_.getScrollOffset();
    const float fields=std::max(0.f,inspector_.size.x-20);
    place(name_,ix,iy+44,fields);place(description_,ix,iy+100,fields);place(*state_,ix,iy+150,168);
    place(*buttons_[6],ix,iy+190,80);place(*kind_,ix,iy+248,168);
    place(*buttons_[7],ix,iy+286,110);place(*buttons_[8],ix+118,iy+286,120);
    place(*buttons_[9],ix,iy+330,80);place(*buttons_[10],ix+88,iy+330,80);
    place(*buttons_[11],ix+88,iy+190,100);
}
void UIWorldAssetsView::update(const InputState& input,float dt,const PanelRect& area,uint64_t panel) {
    layout(area);const auto clipped=input.clipped(area.origin,area.size);
    librarySplit_.setPanelID(panel);inspectorSplit_.setPanelID(panel);
    librarySplit_.update(clipped,dt);layout(area);
    inspectorSplit_.update(clipped,dt);layout(area);
    browser_->setPanelID(panel);browser_->update(clipped,dt);
    for(auto* w:controls())w->setPanelID(panel);
    auto source=[&](Widget* w){return w==library_.get() || w==&assetName_ || std::any_of(buttons_.begin(),buttons_.begin()+6,[&](const auto& b){return b.get()==w;})?clipped:clipped.clipped(inspector_.origin,inspector_.size);};
    UIDropdown* expanded=nullptr;
    for(auto* dropdown:{library_.get(),kind_.get(),state_.get()})if(dropdown->isVisible() && dropdown->isExpanded()){expanded=dropdown;dropdown->update(source(dropdown),dt);break;}
    for(auto* w:controls())if(w->isVisible() && w!=expanded)w->update(source(w),dt);
    actions_.update(clipped.clipped(inspector_.origin,inspector_.size),dt,
        {{inspector_.origin.x+8,inspector_.origin.y+390-scroll_.getScrollOffset()},{std::max(0.f,inspector_.size.x-20),530}},panel);
    canvas_.setPanelID(panel);canvas_.update(clipped,dt);scroll_.setPanelID(panel);scroll_.update(clipped,dt);
    libraryTimer_+=std::max(0.f,dt);
    if(libraryTimer_>=2 && !library_->isExpanded()){libraryTimer_=0;act([this]{refreshLibrary();});}
}
void UIWorldAssetsView::hide(){librarySplit_.cancelInput();inspectorSplit_.cancelInput();browser_->hide();actions_.hide();for(auto* w:controls())w->cancelInput();for(auto* d:{library_.get(),kind_.get(),state_.get()})d->collapse();canvas_.cancelInput();scroll_.cancelInput();libraryTimer_=2;}
void UIWorldAssetsView::draw(OverlayRenderer& r,const PanelRect& area) {
    using namespace UITheme::Colors;layout(area);r.pushClipRect(area.origin,area.size);
    library_->drawOverlay(r,{});assetName_.drawOverlay(r,{});
    r.drawText({area.origin.x+area.size.x-135,area.origin.y+60},dirty()?"Unsaved edits":path_.empty()?"New asset":"Saved",dirty()?Warning:TextMuted,12);
    for(int i=0;i<6;++i)buttons_[i]->drawOverlay(r,{});
    canvas_.drawOverlay(r,{});
    r.pushClipRect(inspector_.origin,inspector_.size);scroll_.drawOverlay(r,{});
    const auto x=inspector_.origin.x+8,y=inspector_.origin.y-scroll_.getScrollOffset();
    const auto* e=editor_.document().find(canvas_.selected);
    r.drawText({x,y+8},e?"Inspect / "+e->kind:"Select an item",TextPrimary,15);
    r.drawText({x,y+28},"In-game name",TextMuted,11);name_.drawOverlay(r,{});
    r.drawText({x,y+84},"Description",TextMuted,11);description_.drawOverlay(r,{});
    if(state_->isVisible())state_->drawOverlay(r,{});
    r.drawText({x,y+228},"Type for New asset / Add inside",TextMuted,12);kind_->drawOverlay(r,{});
    for(size_t i=6;i<buttons_.size();++i)buttons_[i]->drawOverlay(r,{});
    actions_.draw(r,{{x,y+390},{std::max(0.f,inspector_.size.x-20),530}});
    for(auto* d:{kind_.get(),state_.get()})if(d->isVisible() && d->isExpanded())d->drawExpandedList(r,{});
    r.popClipRect();
    librarySplit_.drawOverlay(r,{});inspectorSplit_.drawOverlay(r,{});
    browser_->drawOverlay(r,{});browser_->drawFloating(r);
    r.drawText({area.origin.x+8,area.origin.y+area.size.y-34},status_,TextSecondary,12);
    if(library_->isExpanded())library_->drawExpandedList(r,{});r.popClipRect();
}
