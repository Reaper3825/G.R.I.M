#include "observatory_viewport.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"
#include "ui/ui_orbit_camera.hpp"
#include "ui/ui_root.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/window_manager.hpp"
#include <bgfx/embedded_shader.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/geometric.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>

#if __has_include("vs_observatory.bin.h") && __has_include("fs_observatory.bin.h")
#include "vs_observatory.bin.h"
#include "fs_observatory.bin.h"
#define GRIM_OBSERVATORY_SHADERS 1
#endif

namespace {
using namespace GRIM::Observatory;
std::mutex registryMutex;
std::vector<ObservatoryViewport*> registry;
constexpr const char* passName = "observatory_viewports";
struct Vertex { float x,y,z,r,g,b,a; };
glm::dvec3 location(const Report& r,const LayerReadout& layer,size_t index,size_t ranks,VerticalAxis axis) {
    const auto p=candidateLocation(r,layer,index,ranks,axis);
    return {p[0],p[1],p[2]};
}
glm::vec3 color(int id) {
    const double angle = double(id)*2.399963229728653;
    return {float(.55+.4*std::cos(angle)),float(.55+.4*std::cos(angle+2.094)),
            float(.55+.4*std::cos(angle+4.189))};
}
void vertex(std::vector<Vertex>& out, glm::dvec3 p, glm::vec3 c,float alpha=1.f) {
    out.push_back({float(p.x),float(p.y),float(p.z),c.x,c.y,c.z,alpha});
}
std::string number(double value,int precision=2,bool sign=false) {
    std::ostringstream out;
    if(sign)out<<std::showpos;
    out<<std::fixed<<std::setprecision(precision)<<value;
    return out.str();
}
struct AxisTick { glm::dvec3 point; std::string label; };
struct SceneAxis {
    glm::dvec3 start,end,tickDirection;
    glm::vec3 color;
    uint32_t textColor;
    std::string title;
    Vec2 labelOffset;
    std::vector<AxisTick> ticks;
};
// The rendered marks and the billboard numbers use exactly the same positions.
std::array<SceneAxis,3> sceneAxes(const Report& report,size_t ranks,VerticalAxis vertical,unsigned selectedLayer) {
    const double first=layerDepth(0,report.layerCount),last=layerDepth(report.layerCount-1,report.layerCount);
    std::array<SceneAxis,3> axes{{
        {{-1.2,-1.2,first},{1.2,-1.2,first},{0,.045,0},{1,.7,.4},0xFFFFB366,"X  Probability (%)",{10,8},{}},
        {{-1.2,-1.2,first},{-1.2,1.2,first},{.045,0,0},{.45,.88,.7},0xFF73E0B3,
            vertical==VerticalAxis::Rank?"Y  Candidate rank (#)":"Y  Entropy (nats)",{-10,-30},{}},
        {{-1.2,-1.2,first},{-1.2,-1.2,last+.2},{.045,0,0},{.78,.6,1},0xFFC799FF,"Z  Encoder layer (#)",{-10,22},{}}
    }};
    for(int tick:{0,4,2,1,3})axes[0].ticks.push_back({{-1.+tick*.5,-1.2,first},std::to_string(tick*25)+"%"});
    // Endpoints are considered first so collision thinning preserves the range.
    auto indices=[](size_t count) {
        std::vector<size_t> result{0};
        if(count>1)result.push_back(count-1);
        const size_t intervals=std::min(size_t(5),count-1);
        for(size_t i=1;i<intervals;++i)result.push_back(i*(count-1)/intervals);
        return result;
    };
    if(vertical==VerticalAxis::Rank) {
        for(size_t i:indices(ranks))axes[1].ticks.push_back({{-1.2,ranks>1?1.-2.*i/(ranks-1):0.,first},"#"+std::to_string(i+1)});
    } else {
        const double maximum=std::log(double(report.vocabSize));
        for(int tick:{0,4,2,1,3}) {
            if(maximum==0 && tick!=0)continue;
            axes[1].ticks.push_back({{-1.2,-1.+tick*.5,first},number(maximum*tick/4,2)});
        }
    }
    axes[2].ticks.push_back({{-1.2,-1.2,layerDepth(selectedLayer,report.layerCount)},"L"+std::to_string(selectedLayer+1)});
    for(size_t i:indices(report.layerCount))if(i!=selectedLayer)
        axes[2].ticks.push_back({{-1.2,-1.2,layerDepth(unsigned(i),report.layerCount)},"L"+std::to_string(i+1)});
    return axes;
}
std::string visibleText(const std::string& value) {
    std::string result;
    for(unsigned char c:value) {
        if(c=='\n')result+="\\n";
        else if(c=='\r')result+="\\r";
        else if(c=='\t')result+="\\t";
        else if(c<32)result+='?';
        else result+=char(c);
    }
    return result.empty()?"(empty token)":result;
}
}

struct ObservatoryViewport::State {
    std::mutex mutex;
    std::shared_ptr<UINative3DViewportAttachment> attachment;
    std::shared_ptr<const GRIM::Observatory::Report> report;
    UIOrbitCamera camera;
    std::string owner, message;
    WindowManager::ViewIdBlock views;
    bgfx::ProgramHandle program = BGFX_INVALID_HANDLE;
    bgfx::VertexLayout layout;
    size_t position = 0;
    unsigned layer = 0;
    size_t ranks = 1;
    std::vector<GRIM::Observatory::TokenTrajectory> paths;
    GRIM::Observatory::VerticalAxis axis = GRIM::Observatory::VerticalAxis::Rank;
    int tokenId = -1, lastX=0,lastY=0, travel=0;
    PlatformWindow::ViewportMouseButton drag = PlatformWindow::ViewportMouseButton::None;
    std::optional<Selection> selection;
    std::optional<Selection> hover;
    bool attempted = false;

    void rebuildPaths() {
        paths=report && position<report->positions.size()
            ?GRIM::Observatory::tokenTrajectories(*report,report->positions[position],axis)
            :std::vector<GRIM::Observatory::TokenTrajectory>{};
    }

    std::optional<Selection> pick(double x,double y) const {
        if(!report || position>=report->positions.size())return {};
        const auto g=attachment->lastGeometry();
        const double radius=12.0*g.scale;
        double best=radius*radius,bestDepth=2.0;
        std::optional<Selection> hit;
        for(const auto& l:report->positions[position].layers)for(size_t i=0;i<l.candidates.size();++i) {
            const auto p=camera.projectPoint(location(*report,l,i,ranks,axis));
            if(!p)continue;
            const double distance=(p->x-x)*(p->x-x)+(p->y-y)*(p->y-y);
            if(distance<best || (std::abs(distance-best)<1e-6 && p->z<bestDepth)) {
                best=distance;bestDepth=p->z;hit=Selection{l.layer,l.candidates[i].tokenId};
            }
        }
        return hit;
    }
    void input(const PlatformWindow::ViewportInputEvent& e) {
        std::lock_guard<std::mutex> lock(mutex);
        using namespace PlatformWindow;
        if (e.type == ViewportInputEventType::FocusLost) { drag=ViewportMouseButton::None;hover.reset();return; }
        if (e.type == ViewportInputEventType::MouseWheel) camera.zoomBySteps(double(e.wheelDelta)/120.0);
        if (e.type == ViewportInputEventType::MouseDown) {
            drag=e.button; lastX=e.x;lastY=e.y;travel=0;hover.reset();
        }
        if (e.type == ViewportInputEventType::MouseMove && drag != ViewportMouseButton::None) {
            int dx=e.x-lastX,dy=e.y-lastY;travel+=std::abs(dx)+std::abs(dy);
            if (drag==ViewportMouseButton::Left) camera.orbitByPixels(dx,dy);
            else camera.panByPixels(dx,dy);
            lastX=e.x;lastY=e.y;
        }
        if (e.type == ViewportInputEventType::MouseUp) {
            if (drag==ViewportMouseButton::Left && travel<5)selection=pick(e.x,e.y);
            drag=ViewportMouseButton::None;
        }
    }
};

ObservatoryViewport::ObservatoryViewport() : state_(std::make_shared<State>()) {
    auto& s=*state_;
    s.owner="observatory_"+std::to_string(reinterpret_cast<uintptr_t>(this));
    s.attachment=std::make_shared<UINative3DViewportAttachment>(UIRoot::get().getHWND(),s.owner);
    s.attachment->setInputCallback([weak=std::weak_ptr<State>(state_)](const auto& e) {
        if (auto state=weak.lock()) state->input(e);
    });
    viewport_.attachViewport(s.attachment);
    s.layout.begin().add(bgfx::Attrib::Position,3,bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0,4,bgfx::AttribType::Float).end();
    s.views=WindowManager::reserveViewIds(s.owner,WindowManager::ViewIdRange::PanelViewport,1);
    resetCamera();
    std::lock_guard<std::mutex> lock(registryMutex);
    registry.push_back(this);
    if (registry.size()==1) WindowManager::registerRenderPass(passName,&ObservatoryViewport::renderAll,true);
}
ObservatoryViewport::~ObservatoryViewport() {
    std::lock_guard<std::mutex> lock(registryMutex);
    registry.erase(std::remove(registry.begin(),registry.end(),this),registry.end());
    if (registry.empty()) WindowManager::unregisterRenderPass(passName);
    viewport_.detachViewport();
    if (bgfx::isValid(state_->program)) bgfx::destroy(state_->program);
    WindowManager::releaseViewIds(state_->owner);
}
void ObservatoryViewport::resetCamera() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->camera.setOrbit(.65,.35);
    state_->camera.fitBounds({-1.2,-1.2,-1.2},{1.2,1.2,1.2});
    state_->camera.setDistanceLimits(.1,50);
}
void ObservatoryViewport::setReport(std::shared_ptr<const GRIM::Observatory::Report> report,
                                   size_t position,unsigned layer,int tokenId) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->selection.reset();
    state_->hover.reset();
    const bool changed=state_->report!=report || state_->position!=position;
    state_->report=std::move(report);state_->position=position;state_->layer=layer;state_->tokenId=tokenId;
    state_->ranks=state_->report && position<state_->report->positions.size()
        ?GRIM::Observatory::rankCapacity(state_->report->positions[position]):1;
    if(changed)state_->rebuildPaths();
}
void ObservatoryViewport::setVerticalAxis(GRIM::Observatory::VerticalAxis axis) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const bool changed=state_->axis!=axis;
    state_->axis=axis;state_->hover.reset();state_->selection.reset();
    if(changed)state_->rebuildPaths();
}
void ObservatoryViewport::hide() {
    viewport_.syncViewportGeometry({0,0},false);
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->drag=PlatformWindow::ViewportMouseButton::None;
    state_->selection.reset();
    state_->hover.reset();
}
std::optional<ObservatoryViewport::Selection> ObservatoryViewport::takeSelection() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto selection=state_->selection;state_->selection.reset();return selection;
}
std::string ObservatoryViewport::status() const {
    std::lock_guard<std::mutex> lock(state_->mutex);return state_->message;
}
void ObservatoryViewport::draw(OverlayRenderer& r,Vec2 origin,Vec2 size,Vec2 mousePosition) {
    viewport_.setPosition(0,0);viewport_.setSize(size);
    viewport_.drawOverlay(r,origin);
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto g=viewport_.getGeometry();
    if(g.pixelWidth<=0 || g.pixelHeight<=0)return;
    state_->camera.setViewportSize(g.pixelWidth,g.pixelHeight);
    if (const auto* caps=bgfx::getCaps()) state_->camera.setProjection(.7853981633974483,.01,100,caps->homogeneousDepth);
    auto& s=*state_;
    s.hover.reset();
    if(s.drag==PlatformWindow::ViewportMouseButton::None && viewport_.containsScreenPoint(mousePosition.x,mousePosition.y))
        s.hover=s.pick((mousePosition.x-origin.x)*g.scale,(mousePosition.y-origin.y)*g.scale);
    // Labels and the billboard share the camera's projection with rendered/picked points.
    std::vector<std::array<float,4>> labelBounds;
    auto label=[&](glm::dvec3 p,const std::string& text,uint32_t c,bool badge=false,Vec2 offset=Vec2{}) {
        if (auto q=state_->camera.projectPoint(p)) {
            const Vec2 anchor{origin.x+float(q->x/g.scale),origin.y+float(q->y/g.scale)};
            const float width=r.measureTextWidth(text)+(badge?16.f:0.f),height=badge?26.f:18.f;
            float x=anchor.x+offset.x,y=anchor.y+offset.y;
            auto overlaps=[&] {
                for(const auto& b:labelBounds)if(x<b[2]+4 && x+width+4>b[0] && y<b[3]+3 && y+height+3>b[1])return true;
                return false;
            };
            if(badge) {
                x=std::clamp(x,origin.x+4,origin.x+size.x-width-4);
                y=std::clamp(y,origin.y+27,origin.y+size.y-height-23);
                if(overlaps()) {
                    // A foreshortened axis still gets its title; move its label
                    // into a free row and retain a leader to the real endpoint.
                    x=origin.x+6;y=origin.y+28;
                    while(overlaps() && y+height+30<origin.y+size.y-23)y+=30;
                }
                r.drawLine(anchor,{std::clamp(anchor.x,x,x+width),std::clamp(anchor.y,y,y+height)},c,1.f);
                r.drawRoundedRect({x,y},{width,height},0xF0192435,5);
                r.drawRoundedBorder({x,y},{width,height},c,5);
            } else {
                if(x<origin.x+4 || x+width>origin.x+size.x-4 || y<origin.y+27 || y+height>origin.y+size.y-23 || overlaps())return;
            }
            labelBounds.push_back({x,y,x+width,y+height});
            r.drawText({x+(badge?8.f:0.f),y+(badge?4.f:0.f)},text,c);
        }
    };
    if(s.report && s.position<s.report->positions.size()) {
        const auto& report=*s.report;
        const auto& position=report.positions[s.position];
        const auto axes=sceneAxes(report,s.ranks,s.axis,s.layer);
        for(const auto& axis:axes)label(axis.end,axis.title,axis.textColor,true,axis.labelOffset);
        for(size_t index=0;index<axes.size();++index) {
            const auto& axis=axes[index];
            for(const auto& tick:axis.ticks)
                label(tick.point,tick.label,axis.textColor,false,index==0?Vec2{-8,8}:index==1?Vec2{-35,-8}:Vec2{8,4});
        }
        auto layerLabel=[&](unsigned layer) {
            const auto found=std::lower_bound(position.layers.begin(),position.layers.end(),layer,
                [](const auto& row,unsigned value){return row.layer<value;});
            const bool available=found!=position.layers.end() && found->layer==layer && !found->candidates.empty();
            label({-1.16,-1,GRIM::Observatory::layerDepth(layer,report.layerCount)},
                "L"+std::to_string(layer+1)+(layer==s.layer?" / "+std::to_string(available?found->candidates.size():0)+" points":""),
                layer==s.layer?0xFF8DE6FF:available?0xFF8899B5:0xFF505D73);
        };
        layerLabel(s.layer);
        // Bound label density independently of the actual (fully drawn) layer count.
        const unsigned step=std::max(1u,(report.layerCount+11)/12);
        for(unsigned layer=0;layer<report.layerCount;++layer) {
            if(layer==s.layer)continue;
            if(layer%step!=0 && layer!=s.layer && layer+1!=report.layerCount)continue;
            layerLabel(layer);
        }
        const auto focus=s.hover?s.hover:(s.tokenId>=0?std::optional<Selection>({s.layer,s.tokenId}):std::nullopt);
        if(focus && s.drag==PlatformWindow::ViewportMouseButton::None) {
            const auto layer=std::find_if(position.layers.begin(),position.layers.end(),[&](const auto& l){return l.layer==focus->layer;});
            if(layer!=position.layers.end())for(size_t i=0;i<layer->candidates.size();++i) {
                const auto& candidate=layer->candidates[i];
                if(candidate.tokenId!=focus->tokenId)continue;
                const auto anchor=s.camera.projectPoint(location(report,*layer,i,s.ranks,s.axis));
                if(!anchor)break;
                std::vector<std::string> rows{
                    (report.synthetic?"PREVIEW  |  ":"")+std::string("L")+std::to_string(layer->layer+1)+"  /  rank #"+std::to_string(i+1)+" of "+std::to_string(layer->candidates.size()),
                    "\""+visibleText(candidate.text)+"\"  #"+std::to_string(candidate.tokenId),
                    "Probability  "+number(candidate.probability*100,4)+"%"};
                const auto change=GRIM::Observatory::candidateChange(position,*layer,i);
                if(change.probabilityDelta)rows.push_back("From L"+std::to_string(layer->layer)+": "+number(*change.probabilityDelta*100,3,true)+" pp / rank "+number(*change.rankGain,0,true));
                else rows.push_back(layer->layer==0?"First encoder layer":change.previousLayerAvailable?"Previously outside captured top-k":"Previous layer / readout unavailable");
                rows.push_back("Entropy  "+number(layer->entropyNats,3)+" nats");
                rows.push_back(s.hover?"Click to pin token path":"Pinned token path");
                const float width=std::min(310.f,size.x-16.f);
                const float height=std::min(142.f,size.y-48.f);
                const Vec2 point{origin.x+float(anchor->x/g.scale),origin.y+float(anchor->y/g.scale)};
                float bx=point.x+18;
                if(bx+width>origin.x+size.x-8)bx=point.x-width-18;
                const Vec2 box{std::clamp(bx,origin.x+8,origin.x+size.x-width-8),
                    std::clamp(point.y-height*.5f,origin.y+26,origin.y+size.y-height-22)};
                r.drawLine(point,{std::clamp(point.x,box.x,box.x+width),std::clamp(point.y,box.y,box.y+height)},0xFF8DD8EE,1.5f);
                r.drawRoundedRect({box.x+3,box.y+4},{width,height},0x60000000,8);
                r.drawRoundedRect(box,{width,height},0xF51A2537,8);
                r.drawRoundedBorder(box,{width,height},0xFF52768F,8);
                r.pushClipRect({box.x+10,box.y+8},{width-20,height-16});
                for(size_t row=0;row<rows.size() && 21*float(row)+18<=height-16;++row) {
                    std::string text=rows[row];
                    if(r.measureTextWidth(text)>width-20) {
                        while(!text.empty() && r.measureTextWidth(text+"...")>width-20) {
                            size_t end=text.size()-1;
                            while(end>0 && (static_cast<unsigned char>(text[end])&0xC0)==0x80)--end;
                            text.resize(end);
                        }
                        text+="...";
                    }
                    r.drawText({box.x+10,box.y+9+21*float(row)},text,row==1?0xFFFFFFFF:row==0?0xFF8DE6FF:0xFFB9C8DD);
                }
                r.popClipRect();break;
            }
        }
    }
    r.drawText({origin.x+10,origin.y+8},s.axis==GRIM::Observatory::VerticalAxis::Rank?"X probability / Y shown rank / Z layer":"X probability / Y entropy / Z layer",0xFFB9C8DD);
    r.drawText({origin.x+10,origin.y+size.y-20},"Drag orbit / Right pan / Wheel zoom",0xFF8899B5);
}
void ObservatoryViewport::renderAll(uint32_t) {
    std::lock_guard<std::mutex> guard(registryMutex);
    for (auto* viewport:registry) {
        auto& s=*viewport->state_;
        std::lock_guard<std::mutex> lock(s.mutex);
        auto& a=*s.attachment;
        if (!a.hasFrameBuffer() || !a.lastGeometry().visible) continue;
        auto id=s.views.at(0);
        bgfx::setViewFrameBuffer(id,a.frameBufferHandle());
        bgfx::setViewRect(id,0,0,a.frameBufferWidth(),a.frameBufferHeight());
        bgfx::setViewClear(id,BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH,uint32_t(0x111622FF),1.0f,0);
        bgfx::touch(id);
        if (!s.attempted) {
            s.attempted=true;
#ifdef GRIM_OBSERVATORY_SHADERS
            static const bgfx::EmbeddedShader shaders[]={
                BGFX_EMBEDDED_SHADER(vs_observatory),BGFX_EMBEDDED_SHADER(fs_observatory),BGFX_EMBEDDED_SHADER_END()};
            auto vs=bgfx::createEmbeddedShader(shaders,bgfx::getRendererType(),"vs_observatory");
            auto fs=bgfx::createEmbeddedShader(shaders,bgfx::getRendererType(),"fs_observatory");
            if (bgfx::isValid(vs) && bgfx::isValid(fs)) s.program=bgfx::createProgram(vs,fs,true);
            else { if (bgfx::isValid(vs)) bgfx::destroy(vs);if (bgfx::isValid(fs)) bgfx::destroy(fs); }
#endif
            if (!bgfx::isValid(s.program)) s.message="3D shaders unavailable; use Flat view until the next UI build.";
        }
        if (!bgfx::isValid(s.program) || !s.report || s.position>=s.report->positions.size()) continue;
        const auto frame=s.camera.frame();
        const glm::mat4 view(frame.view),projection(frame.projection);
        bgfx::setViewTransform(id,glm::value_ptr(view),glm::value_ptr(projection));
        std::vector<Vertex> surfaces,lines,ribbons,points;
        auto line=[&](glm::dvec3 p,glm::dvec3 q,glm::vec3 c,float alpha=1.f) {
            vertex(lines,p,c,alpha);vertex(lines,q,c,alpha);
        };
        const auto& position=s.report->positions[s.position];
        const double selectedZ=GRIM::Observatory::layerDepth(s.layer,s.report->layerCount);
        // Only the active plane is filled; the stack remains see-through.
        for(const auto& p:std::initializer_list<glm::dvec3>{{-1,-1,selectedZ},{1,-1,selectedZ},{1,1,selectedZ},
                                                        {-1,-1,selectedZ},{1,1,selectedZ},{-1,1,selectedZ}})
            vertex(surfaces,p,{.22,.48,.65},.12f);
        const double firstZ=GRIM::Observatory::layerDepth(0,s.report->layerCount);
        const double lastZ=GRIM::Observatory::layerDepth(s.report->layerCount-1,s.report->layerCount);
        for(double x:{-1.,1.})for(double y:{-1.,1.})line({x,y,firstZ},{x,y,lastZ},{.32,.40,.53},.6f);
        for(const auto& axis:sceneAxes(*s.report,s.ranks,s.axis,s.layer)) {
            line(axis.start,axis.end,axis.color);
            const auto direction=glm::normalize(axis.end-axis.start);
            line(axis.end,axis.end-direction*.12+axis.tickDirection*1.6,axis.color);
            line(axis.end,axis.end-direction*.12-axis.tickDirection*1.6,axis.color);
            for(const auto& tick:axis.ticks)line(tick.point-axis.tickDirection,tick.point+axis.tickDirection,axis.color);
        }
        for(unsigned layer=0;layer<s.report->layerCount;++layer) {
            const double z=GRIM::Observatory::layerDepth(layer,s.report->layerCount);
            const auto found=std::lower_bound(position.layers.begin(),position.layers.end(),layer,
                [](const auto& row,unsigned value){return row.layer<value;});
            const bool available=found!=position.layers.end() && found->layer==layer && !found->candidates.empty();
            const auto col=layer==s.layer?glm::vec3(.45,.82,.95):available?glm::vec3(.25,.36,.50):glm::vec3(.13,.17,.23);
            line({-1,-1,z},{1,-1,z},col);line({1,-1,z},{1,1,z},col);
            line({1,1,z},{-1,1,z},col);line({-1,1,z},{-1,-1,z},col);
        }
        for(int tick=1;tick<4;++tick) {
            const double v=-1.0+.5*tick;
            line({v,-1,selectedZ},{v,1,selectedZ},{.27,.47,.59},.38f);
            if(s.axis==GRIM::Observatory::VerticalAxis::Entropy)
                line({-1,v,selectedZ},{1,v,selectedZ},{.27,.47,.59},.38f);
        }
        if(s.axis==GRIM::Observatory::VerticalAxis::Rank) {
            const size_t stride=std::max(size_t(1),(s.ranks+15)/16);
            for(size_t rank=0;rank<s.ranks;rank+=stride) {
                const double y=s.ranks>1?1.-2.*rank/(s.ranks-1):0.;
                line({-1,y,selectedZ},{1,y,selectedZ},{.27,.47,.59},.3f);
            }
        }
        const int focusedToken=s.hover?s.hover->tokenId:s.tokenId;
        const auto forward=glm::normalize(frame.target-frame.position);
        for(const auto& path:s.paths) {
            if(path.points.size()<2)continue;
            const bool tracked=path.tokenId==focusedToken;
            const int subdivisions=tracked?16:10;
            std::vector<glm::dvec3> samples;
            samples.reserve((path.points.size()-1)*subdivisions+1);
            samples.emplace_back(path.points.front()[0],path.points.front()[1],path.points.front()[2]);
            for(size_t segment=0;segment+1<path.points.size();++segment)for(int step=1;step<=subdivisions;++step) {
                const auto p=GRIM::Observatory::sampleTrajectory(path,segment,double(step)/subdivisions);
                samples.emplace_back(p[0],p[1],p[2]);
            }
            if(!tracked) {
                const auto col=color(path.tokenId);
                for(size_t i=1;i<samples.size();++i)line(samples[i-1],samples[i],col,focusedToken>=0?.12f:.38f);
                continue;
            }
            // A joined camera-facing strip keeps the focused spline ~3 logical
            // pixels wide without depending on driver-specific wide line support.
            std::vector<glm::dvec3> sides(samples.size());
            for(size_t i=0;i<samples.size();++i) {
                const auto direction=samples[std::min(i+1,samples.size()-1)]-samples[i>0?i-1:0];
                auto side=glm::cross(forward,direction);
                const double length=glm::length(side);
                side=length>1e-10?side/length:frame.right;
                const double depth=std::max(0.,glm::dot(samples[i]-frame.position,forward));
                sides[i]=side*(2*depth*std::tan(.7853981633974483/2)/a.frameBufferHeight()*a.lastGeometry().scale*1.5);
            }
            for(size_t i=1;i<samples.size();++i) {
                if(glm::dot(samples[i-1]-frame.position,forward)<=.01 || glm::dot(samples[i]-frame.position,forward)<=.01)continue;
                const auto left=samples[i-1]-sides[i-1],right=samples[i-1]+sides[i-1];
                const auto nextLeft=samples[i]-sides[i],nextRight=samples[i]+sides[i];
                for(const auto& p:{left,right,nextRight,left,nextRight,nextLeft})vertex(ribbons,p,{.75,.92,1});
            }
        }
        for (const auto& l:position.layers) {
            for (size_t index=0;index<l.candidates.size();++index) {
                const auto& c=l.candidates[index];
                auto p=location(*s.report,l,index,s.ranks,s.axis);
                auto col=color(c.tokenId);
                const bool tracked=c.tokenId==focusedToken;
                const bool selected=(l.layer==s.layer && c.tokenId==s.tokenId)
                    || (s.hover && s.hover->layer==l.layer && s.hover->tokenId==c.tokenId);
                const double depth=glm::dot(p-frame.position,forward);
                if(depth<=0)continue;
                // Camera-facing circular glyphs have a constant screen radius.
                // Size is a selection cue, never a fabricated hidden dimension.
                const double radius=2*depth*std::tan(.7853981633974483/2)/a.frameBufferHeight()
                    *a.lastGeometry().scale*(selected?7.:tracked?5.5:4.);
                if(focusedToken>=0 && !tracked)col*=.4f;
                else if(l.layer!=s.layer && !tracked)col*=.7f;
                constexpr int segments=10;
                for(int segment=0;segment<segments;++segment) {
                    const double a0=segment*6.283185307179586/segments,a1=(segment+1)*6.283185307179586/segments;
                    const auto v0=frame.right*std::cos(a0)+frame.up*std::sin(a0);
                    const auto v1=frame.right*std::cos(a1)+frame.up*std::sin(a1);
                    vertex(points,p,selected?glm::vec3(1):col);
                    vertex(points,p+v0*radius,col*.65f);vertex(points,p+v1*radius,col*.65f);
                    if(selected)line(p+v0*radius*1.6,p+v1*radius*1.6,{.7,.92,1},.9f);
                }
            }
        }
        auto submit=[&](const std::vector<Vertex>& vertices,uint64_t flags) {
            if (vertices.empty()) return;
            uint32_t count=static_cast<uint32_t>(vertices.size());
            if (bgfx::getAvailTransientVertexBuffer(count,s.layout)!=count) {
                s.message="Scene exceeds this frame's vertex budget; use Flat view.";return;
            }
            bgfx::TransientVertexBuffer buffer;
            bgfx::allocTransientVertexBuffer(&buffer,count,s.layout);
            std::memcpy(buffer.data,vertices.data(),vertices.size()*sizeof(Vertex));
            bgfx::setVertexBuffer(0,&buffer);
            bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_DEPTH_TEST_LEQUAL|BGFX_STATE_BLEND_ALPHA|flags);
            bgfx::submit(id,s.program);
        };
        bgfx::setViewMode(id,bgfx::ViewMode::Sequential);
        s.message.clear();submit(surfaces,0);submit(lines,BGFX_STATE_PT_LINES);submit(ribbons,0);submit(points,BGFX_STATE_WRITE_Z);
    }
}
