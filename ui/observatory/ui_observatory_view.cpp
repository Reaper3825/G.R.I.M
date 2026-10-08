#include "ui_observatory_view.hpp"
#include "observatory_viewport.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_slider.hpp"
#include "ui/primitives/ui_graph.hpp"
#include "ui/overlay_renderer.hpp"
#include "resources.hpp"
#include <nlohmann/json.hpp>
#include <future>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace GRIM::Observatory;
namespace {
std::string fixed(double v,int precision=3) { std::ostringstream s;s<<std::fixed<<std::setprecision(precision)<<v;return s.str(); }
std::string oneLine(std::string s) { for(char& c:s) if(static_cast<unsigned char>(c)<32)c=' ';return s; }
}
struct UIObservatoryView::Impl {
    struct Loaded {
        std::shared_ptr<GRIM::Config::CompiledModelConfigSnapshot> config;
        std::shared_ptr<Report> report;
    };
    std::shared_ptr<GRIM::Config::CompiledModelConfigSnapshot> config;
    std::shared_ptr<Report> report;
    std::future<Loaded> loading;
    struct StoredModel { std::string id; std::filesystem::path configPath; };
    std::vector<StoredModel> models;
    int modelIndex=0;
    std::string configPath,capturePath,message="Select a model from the model store to begin.";
    UIInputBox captureInput{&capturePath};
    UIButton loadConfig,refresh,loadCapture,preview,play,view,reset;
    UIDropdown model;
    UIDropdown position,candidate;
    UISlider layer;
    UIGraph graph{"Candidate probability (X) / entropy in nats (Y)",GraphType::Scatter};
    std::unique_ptr<ObservatoryViewport> viewport;
    size_t positionIndex=0;
    unsigned selectedLayer=0;
    int selectedToken=-1;
    bool flat=false,playing=false,readyLayout=false;
    float elapsed=0;
    PanelRect scene{},inspector{};
    std::vector<ObservatoryViewport::Selection> graphSelection;

    Impl():
      loadConfig("Reload model",[this]{beginLoad(true);}),
      refresh("Refresh store",[this]{refreshModels();}),
      loadCapture("Load capture",[this]{beginLoad(false);}),
      preview("Synthetic preview",[this]{
          if(!config || loading.valid()) {message="Select a model first; wait for any current load.";return;}
          try {report=std::make_shared<Report>(makePreview(*config));installed();}
          catch(const std::exception& e){message=e.what();}
      }),
      play("Play depth",[this]{if(report){playing=!playing;elapsed=0;play.setText(playing?"Pause":"Play depth");}}),
      view("Flat view",[this]{flat=!flat;view.setText(flat?"3D view":"Flat view");if(viewport)viewport->hide();}),
      reset("Reset camera",[this]{if(viewport)viewport->resetCamera();}),
      model("Model store",{"Select model"},0,[this](int i,const std::string&){
          if(loading.valid()) {model.setSelectedIndex(modelIndex);message="Wait for the current load before changing models.";return;}
          modelIndex=i;
          configPath=i>0 && size_t(i)<=models.size()?models[size_t(i)-1].configPath.string():std::string{};
          clearModel();
          if(!configPath.empty())beginLoad(true);
          else message="Select a model from the model store to begin.";
      }),
      position("",{},0,[this](int i,const std::string&){if(i>=0){positionIndex=size_t(i);selectedToken=-1;rebuild();}}),
      candidate("",{},0,[this](int i,const std::string&){
          const auto* l=readout();
          if(l && i>0 && size_t(i)<=l->candidates.size())selectedToken=l->candidates[size_t(i)-1].tokenId;
          else selectedToken=-1;
          rebuild();
      }),
      layer("Encoder layer",1,1,1,[this](float value){selectedLayer=unsigned(value)-1;selectedToken=-1;rebuild();},1) {
        captureInput.setPlaceholder("Absolute path to saved Observatory capture.json");
        auto& gc=graph.getConfig();gc.useDownsampling=false;gc.pointRadius=5;gc.showValues=false;
        graph.setXAxisRange(0,1);
        graph.setOnPointClick([this](int i,const DataPoint&){
            if(i>=0 && size_t(i)<graphSelection.size()) {
                selectedLayer=graphSelection[i].layer;selectedToken=graphSelection[i].tokenId;rebuild();
            }
        });
    }
    void clearModel() {
        playing=false;play.setText("Play depth");config.reset();report.reset();
        position.collapse();candidate.collapse();position.setItems({});
        layer.setRange(1,1);graph.clearData();
        if(viewport){viewport->setReport({},0,0,-1);viewport->hide();}
    }
    void refreshModels() {
        if(loading.valid()) {message="Wait for the current load before refreshing the model store.";return;}
        try {
            const auto runtime=loadGrimRuntimeAiConfig();
            const auto raw=runtime.at("paths").at("grim_text").at("model_store").get<std::string>();
            if(raw.empty())throw std::runtime_error("Model store path is empty");
            std::filesystem::path store(raw);
            if(store.is_relative())store=std::filesystem::path(getGrimRootDir())/store;
            std::vector<StoredModel> found;
            for(const auto& entry:std::filesystem::directory_iterator(store)) {
                const auto artifact=entry.path()/"model.grimcfg";
                if(entry.is_directory() && std::filesystem::is_regular_file(artifact))
                    found.push_back({entry.path().filename().string(),artifact.lexically_normal()});
            }
            std::sort(found.begin(),found.end(),[](const auto& a,const auto& b){return a.id<b.id;});
            models=std::move(found);
            std::vector<std::string> labels{"Select model"};modelIndex=0;
            for(size_t i=0;i<models.size();++i) {
                labels.push_back(models[i].id);
                if(models[i].configPath==std::filesystem::path(configPath))modelIndex=int(i)+1;
            }
            model.collapse();model.setItems(labels);model.setSelectedIndex(modelIndex);
            if(modelIndex==0) {
                configPath.clear();clearModel();
                message=models.empty()?"No compiled models in the model store. Compile one in Model Config.":"Select a model from the model store to begin.";
            }
        } catch(const std::exception& e) {
            models.clear();modelIndex=0;configPath.clear();clearModel();
            model.collapse();model.setItems({"Store unavailable"});model.setSelectedIndex(0);
            message=std::string("Model store: ")+e.what();
        }
    }
    void beginLoad(bool model) {
        if(loading.valid()) {message="A file is already loading.";return;}
        if(!model && !config) {message="Select a model before opening a capture.";return;}
        const auto path=model?configPath:capturePath;
        if(path.empty()) {message=model?"Select a model from the model store first.":"Enter a capture file path first.";return;}
        if(model)clearModel();
        const auto snapshot=config;
        playing=false;play.setText("Play depth");
        message=model?"Loading and verifying .grimcfg...":"Loading and validating capture...";
        loading=std::async(std::launch::async,[model,path,snapshot] {
            Loaded result;
            if(model) result.config=std::make_shared<GRIM::Config::CompiledModelConfigSnapshot>(GRIM::Config::loadCompiledModelConfig(path));
            else result.report=std::make_shared<Report>(loadReport(path,*snapshot));
            return result;
        });
    }
    void installed() {
        position.collapse();candidate.collapse();
        positionIndex=0;selectedLayer=report->positions.front().layers.front().layer;selectedToken=-1;
        playing=false;play.setText("Play depth");elapsed=0;
        std::vector<std::string> labels;
        for(const auto& p:report->positions) labels.push_back("Position "+std::to_string(p.position));
        position.setItems(labels);position.setSelectedIndex(0);
        layer.setRange(1,float(report->layerCount));
        message=report->synthetic?"SYNTHETIC PREVIEW - fabricated probabilities; no model execution.":"Saved capture loaded. Probabilities are not renormalized to top-k.";
        rebuild();if(viewport)viewport->resetCamera();
    }
    const LayerReadout* readout() const {
        if(!report)return nullptr;
        for(const auto& l:report->positions[positionIndex].layers)if(l.layer==selectedLayer)return &l;
        return nullptr;
    }
    void rebuild() {
        if(!report)return;
        layer.setValue(float(selectedLayer+1));
        const auto* l=readout();
        std::vector<std::string> candidates{"Select candidate"};int chosen=0;
        if(l)for(const auto& c:l->candidates) {
            candidates.push_back("#"+std::to_string(c.tokenId)+"  "+fixed(c.probability*100,2)+"%");
            if(c.tokenId==selectedToken)chosen=int(candidates.size())-1;
        }
        candidate.setItems(candidates);candidate.setSelectedIndex(chosen);
        graphSelection.clear();std::vector<DataPoint> points;
        if(l)for(const auto& c:l->candidates) {
            auto p=DataPoint::xy(float(c.probability),float(l->entropyNats),oneLine(c.text));
            p.color=c.tokenId==selectedToken?0xFFFFFFFF:0xFFFFAA66;
            points.push_back(p);graphSelection.push_back({l->layer,c.tokenId});
        }
        graph.setAxisRange(0,float(std::max(1e-6,std::log(double(report->vocabSize)))));
        graph.setData(points);
        if(viewport)viewport->setReport(report,positionIndex,selectedLayer,selectedToken);
    }
    void layout(const PanelRect& rect,uint64_t panelId) {
        readyLayout=rect.size.x>=660 && rect.size.y>=380;
        if(!readyLayout){if(viewport)viewport->hide();return;}
        const float x=rect.origin.x+10,y=rect.origin.y+4,w=rect.size.x-20;
        auto place=[&](Widget& widget,float px,float py,float width,float height=28.f) {
            widget.setPanelID(panelId);widget.setPosition(px,py);widget.setSize(width,height);
        };
        place(model,x,y-4,w-250,34);place(refresh,x+w-240,y,115);place(loadConfig,x+w-115,y,115);
        place(captureInput,x,y+34,w-125);place(loadCapture,x+w-115,y+34,115);
        place(preview,x,y+68,150);place(view,x+160,y+68,100);place(reset,x+270,y+68,120);
        place(play,x+w-115,y+68,115);
        place(position,x+65,y+146,170);place(layer,x+265,y+146,w-265);
        scene={{x,y+188},{w-255,rect.size.y-204}};
        inspector={{x+w-245,y+188},{245,scene.size.y}};
        place(candidate,inspector.origin.x+75,inspector.origin.y,160);
        place(graph,scene.origin.x,scene.origin.y,scene.size.x,scene.size.y);
    }
};

UIObservatoryView::UIObservatoryView():impl_(std::make_unique<Impl>()){}
UIObservatoryView::~UIObservatoryView()=default;
void UIObservatoryView::refreshModels(){impl_->refreshModels();}
void UIObservatoryView::hide(){
    auto& s=*impl_;
    s.playing=false;s.play.setText("Play depth");
    s.model.collapse();s.position.collapse();s.candidate.collapse();
    s.captureInput.setFocused(false);
    s.layer.setRange(s.layer.getMinValue(),s.layer.getMaxValue());
    if(s.viewport)s.viewport->hide();
}
void UIObservatoryView::update(const InputState& input,float dt,const PanelRect& rect,uint64_t id) {
    auto& s=*impl_;s.layout(rect,id);
    if(s.loading.valid() && s.loading.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
        try {
            auto result=s.loading.get();
            if(result.config) {
                s.position.collapse();s.candidate.collapse();
                s.config=std::move(result.config);s.report.reset();s.position.setItems({});s.layer.setRange(1,1);s.graph.clearData();
                if(s.viewport){s.viewport->setReport({},0,0,-1);s.viewport->hide();}
                s.message="Model configuration verified. Load a saved capture or try the synthetic preview.";
            } else {s.report=std::move(result.report);s.installed();}
        } catch(const std::exception& e){s.message=std::string("Load failed: ")+e.what();}
    }
    if(!s.readyLayout)return;
    bool modelExpanded=s.model.isExpanded();s.model.update(input,dt);
    if(modelExpanded || s.model.isExpanded()) {if(s.viewport)s.viewport->hide();return;}
    bool expanded=s.position.isExpanded();s.position.update(input,dt);
    if(expanded || s.position.isExpanded()) {if(s.viewport)s.viewport->hide();return;}
    if(s.report) {
        expanded=s.candidate.isExpanded();s.candidate.update(input,dt);
        if(expanded || s.candidate.isExpanded()) {if(s.viewport)s.viewport->hide();return;}
    }
    for(Widget* w:std::initializer_list<Widget*>{&s.captureInput,&s.loadConfig,&s.refresh,&s.loadCapture,&s.preview,&s.view,&s.reset,&s.play})w->update(input,dt);
    if(!s.report)return;
    s.layer.update(input,dt);if(s.flat)s.graph.update(input,dt);
    if(s.viewport)if(auto pick=s.viewport->takeSelection()) {
        s.selectedLayer=pick->layer;s.selectedToken=pick->tokenId;s.rebuild();
    }
    if(s.playing && std::isfinite(dt)) {
        s.elapsed+=std::max(0.f,dt);
        if(s.elapsed>=.65f) {
            s.elapsed=0;const auto& layers=s.report->positions[s.positionIndex].layers;
            auto next=std::find_if(layers.begin(),layers.end(),[&](const auto& l){return l.layer>s.selectedLayer;});
            s.selectedLayer=next==layers.end()?layers.front().layer:next->layer;s.rebuild();
        }
    }
}
void UIObservatoryView::draw(OverlayRenderer& r,const PanelRect& rect) {
    auto& s=*impl_;
    if(!s.readyLayout){r.drawText(rect.origin,"Enlarge the panel to at least 660 x 380 content pixels.",0xFFBBBBBB);return;}
    s.model.drawOverlay(r,{0,0});
    for(Widget* w:std::initializer_list<Widget*>{&s.captureInput,&s.loadConfig,&s.refresh,&s.loadCapture,&s.preview,&s.view,&s.reset,&s.play})w->drawOverlay(r,{0,0});
    const float x=rect.origin.x+10,y=rect.origin.y+4;
    r.pushClipRect({x,y+99},{rect.size.x-20,45});
    r.drawText({x,y+101},oneLine(s.message),s.report&&s.report->synthetic?0xFF66CCFF:0xFFCCCCCC);
    if(s.config) {
        const auto& a=s.config->architecture;
        r.drawText({x,y+122},"grimcfg: "+std::to_string(a.num_layers)+" layers | d_model "+std::to_string(a.d_model)+" | heads "+std::to_string(a.num_heads)+" / KV "+std::to_string(a.num_kv_heads)+" | max sequence "+std::to_string(a.max_seq_len),0xFFAAAAAA);
    }
    r.popClipRect();
    if(!s.report) {r.drawText({x,y+190},"Post-run analysis: load an inference or training-example capture.",0xFFBBBBBB);s.model.drawExpandedList(r,{0,0});return;}
    r.drawText({x,y+153},"Token:",0xFFAAAAAA);
    s.position.drawOverlay(r,{0,0});s.layer.drawOverlay(r,{0,0});
    r.pushClipRect(s.scene.origin,s.scene.size);
    if(s.flat || s.model.isExpanded() || s.position.isExpanded() || s.candidate.isExpanded()) {
        if(s.viewport)s.viewport->hide();s.graph.drawOverlay(r,{0,0});
    } else {
        try {
            if(!s.viewport){s.viewport=std::make_unique<ObservatoryViewport>();s.rebuild();}
            s.viewport->draw(r,s.scene.origin,s.scene.size);
            const auto status=s.viewport->status();
            if(!status.empty())r.drawText({s.scene.origin.x+10,s.scene.origin.y+32},status,0xFF66CCFF);
        } catch(const std::exception& e) {
            if(s.viewport)s.viewport->hide();
            s.flat=true;s.view.setText("3D view");s.message=std::string("3D unavailable: ")+e.what();
            s.graph.drawOverlay(r,{0,0});
        }
    }
    r.popClipRect();
    r.pushClipRect(s.inspector.origin,s.inspector.size);
    r.drawText({s.inspector.origin.x,s.inspector.origin.y+7},"Inspect:",0xFFAAAAAA);
    s.candidate.drawOverlay(r,{0,0});
    float iy=s.inspector.origin.y+37;
    auto text=[&](const std::string& value,uint32_t color=0xFFCCCCCC){r.drawText({s.inspector.origin.x,iy},value,color);iy+=21;};
    text(s.report->synthetic?"SYNTHETIC / V = "+std::to_string(s.report->vocabSize):"SAVED MODEL CAPTURE",0xFF66CCFF);
    text(s.report->mode=="inference"?"Mode: Inference":"Mode: Training example replay");
    text("Visibility: "+s.report->visibility);
    text("Input: "+oneLine(s.report->positions[s.positionIndex].inputText),0xFFAAAAAA);
    text("Checkpoint: "+oneLine(s.report->checkpoint),0xFFAAAAAA);
    text("Layer "+std::to_string(s.selectedLayer+1)+" / "+std::to_string(s.report->layerCount));
    if(const auto* l=s.readout()) {
        text("Entropy: "+fixed(l->entropyNats)+" nats");
        double mass=0;for(const auto& c:l->candidates)mass+=c.probability;
        text("Shown mass: "+fixed(mass*100,2)+"%");
        text("Candidates (probability / token ID)",0xFFAAAAAA);
        if(s.selectedToken>=0)for(const auto& c:l->candidates)if(c.tokenId==s.selectedToken) {
            text("Selected: "+oneLine(c.text),0xFFFFFFFF);
            std::ostringstream value;value<<"P(#"<<c.tokenId<<") = "<<std::setprecision(7)<<c.probability;
            text(value.str(),0xFFFFFFFF);
        }
        for(const auto& c:l->candidates) {
            if(iy+22>s.inspector.origin.y+s.inspector.size.y)break;
            text(fixed(c.probability*100,2)+"%  #"+std::to_string(c.tokenId)+"  "+oneLine(c.text),c.tokenId==s.selectedToken?0xFFFFFFFF:0xFFBBBBBB);
        }
    } else text("This layer was not captured.",0xFF66CCFF);
    r.popClipRect();s.position.drawExpandedList(r,{0,0});s.candidate.drawExpandedList(r,{0,0});s.model.drawExpandedList(r,{0,0});
}
bool UIObservatoryView::shouldPassThroughAt(float x,float y) const {
    if(!impl_->viewport)return false;const auto g=impl_->viewport->geometry();
    return g.visible && x>=g.logicalOrigin.x && y>=g.logicalOrigin.y && x<g.logicalOrigin.x+g.logicalSize.x && y<g.logicalOrigin.y+g.logicalSize.y;
}
void UIObservatoryView::collectPassThroughRects(std::vector<PanelRect>& out) const {
    if(impl_->viewport){const auto g=impl_->viewport->geometry();if(g.visible)out.push_back({g.logicalOrigin,g.logicalSize});}
}
