#include "observatory_viewport.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"
#include "ui/ui_orbit_camera.hpp"
#include "ui/ui_root.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/window_manager.hpp"
#include <bgfx/embedded_shader.h>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>

#if __has_include("vs_observatory.bin.h") && __has_include("fs_observatory.bin.h")
#include "vs_observatory.bin.h"
#include "fs_observatory.bin.h"
#define GRIM_OBSERVATORY_SHADERS 1
#endif

namespace {
std::mutex registryMutex;
std::vector<ObservatoryViewport*> registry;
constexpr const char* passName = "observatory_viewports";
struct Vertex { float x,y,z,r,g,b,a; };
glm::dvec3 location(const GRIM::Observatory::Report& r, unsigned layer, double p, double h) {
    return {2.0*p-1.0, 2.0*h/std::max(1.0e-6,std::log(double(r.vocabSize)))-1.0,
            r.layerCount > 1 ? 2.0*layer/(r.layerCount-1)-1.0 : 0.0};
}
glm::vec3 color(int id) {
    const double angle = double(id)*2.399963229728653;
    return {float(.55+.4*std::cos(angle)),float(.55+.4*std::cos(angle+2.094)),
            float(.55+.4*std::cos(angle+4.189))};
}
void vertex(std::vector<Vertex>& out, glm::dvec3 p, glm::vec3 c) {
    out.push_back({float(p.x),float(p.y),float(p.z),c.x,c.y,c.z,1.f});
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
    int tokenId = -1, lastX=0,lastY=0, travel=0;
    PlatformWindow::ViewportMouseButton drag = PlatformWindow::ViewportMouseButton::None;
    std::optional<Selection> selection;
    bool attempted = false;

    void input(const PlatformWindow::ViewportInputEvent& e) {
        std::lock_guard<std::mutex> lock(mutex);
        using namespace PlatformWindow;
        if (e.type == ViewportInputEventType::FocusLost) { drag=ViewportMouseButton::None; return; }
        if (e.type == ViewportInputEventType::MouseWheel) camera.zoomBySteps(double(e.wheelDelta)/120.0);
        if (e.type == ViewportInputEventType::MouseDown) {
            drag=e.button; lastX=e.x;lastY=e.y;travel=0;
        }
        if (e.type == ViewportInputEventType::MouseMove && drag != ViewportMouseButton::None) {
            int dx=e.x-lastX,dy=e.y-lastY;travel+=std::abs(dx)+std::abs(dy);
            if (drag==ViewportMouseButton::Left) camera.orbitByPixels(dx,dy);
            else camera.panByPixels(dx,dy);
            lastX=e.x;lastY=e.y;
        }
        if (e.type == ViewportInputEventType::MouseUp) {
            if (drag==ViewportMouseButton::Left && travel<5 && report && position<report->positions.size()) {
                double best=144.0, bestDepth=2.0;
                for (const auto& l:report->positions[position].layers) for (const auto& c:l.candidates) {
                    auto p=camera.projectPoint(location(*report,l.layer,c.probability,l.entropyNats));
                    if (!p) continue; 
                    double d=(p->x-e.x)*(p->x-e.x)+(p->y-e.y)*(p->y-e.y);
                    if (d<best || (std::abs(d-best)<1e-6 && p->z<bestDepth)) {
                        best=d;bestDepth=p->z;selection=Selection{l.layer,c.tokenId};
                    }
                }
            }
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
    state_->report=std::move(report);state_->position=position;state_->layer=layer;state_->tokenId=tokenId;
}
void ObservatoryViewport::hide() {
    viewport_.syncViewportGeometry({0,0},false);
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->drag=PlatformWindow::ViewportMouseButton::None;
    state_->selection.reset();
}
std::optional<ObservatoryViewport::Selection> ObservatoryViewport::takeSelection() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto selection=state_->selection;state_->selection.reset();return selection;
}
std::string ObservatoryViewport::status() const {
    std::lock_guard<std::mutex> lock(state_->mutex);return state_->message;
}
void ObservatoryViewport::draw(OverlayRenderer& r,Vec2 origin,Vec2 size) {
    viewport_.setPosition(0,0);viewport_.setSize(size);
    viewport_.drawOverlay(r,origin);
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto g=viewport_.getGeometry();
    state_->camera.setViewportSize(g.pixelWidth,g.pixelHeight);
    if (const auto* caps=bgfx::getCaps()) state_->camera.setProjection(.7853981633974483,.01,100,caps->homogeneousDepth);
    // Labels share the camera's projection, so axis meaning survives orbiting.
    auto label=[&](glm::dvec3 p,const std::string& text,uint32_t c) {
        if (auto q=state_->camera.projectPoint(p))
            r.drawText({origin.x+float(q->x/g.scale),origin.y+float(q->y/g.scale)},text,c);
    };
    label({-1,-1,-1},"0",0xFFAAAAAA);
    label({0,-1,-1},"50%",0xFFFFBB66);
    label({1,-1,-1},"P = 100%",0xFFFFBB66);
    if (state_->report) {
        label({-1,1,-1},"H = "+std::to_string(std::log(double(state_->report->vocabSize)))+" nats",0xFF88DD88);
        label({-1,-1,state_->report->layerCount>1?1.0:0.0},"L"+std::to_string(state_->report->layerCount),0xFFFF99DD);
    }
    r.drawText({origin.x+10,origin.y+8},"X probability  /  Y full-vocabulary entropy  /  Z encoder layer",0xFFCCCCCC);
    r.drawText({origin.x+10,origin.y+size.y-22},"Drag: orbit   Right drag: pan   Wheel: zoom   Click: inspect",0xFFAAAAAA);
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
        std::vector<Vertex> lines,points;
        auto line=[&](glm::dvec3 p,glm::dvec3 q,glm::vec3 c) { vertex(lines,p,c);vertex(lines,q,c); };
        line({-1,-1,-1},{1,-1,-1},{1,.7,.4});
        line({-1,-1,-1},{-1,1,-1},{.5,1,.5});
        line({-1,-1,-1},{-1,-1,1},{1,.5,.8});
        std::map<int,std::pair<unsigned,glm::dvec3>> previous;
        for (const auto& l:s.report->positions[s.position].layers) {
            const double z=location(*s.report,l.layer,0,0).z;
            line({-1,-1,z},{1,-1,z},l.layer==s.layer?glm::vec3(.55f):glm::vec3(.15f));
            if (l.layer==s.layer) {
                line({1,-1,z},{1,1,z},{.35,.35,.45});
                line({1,1,z},{-1,1,z},{.35,.35,.45});
                line({-1,1,z},{-1,-1,z},{.35,.35,.45});
            }
            for (const auto& c:l.candidates) {
                auto p=location(*s.report,l.layer,c.probability,l.entropyNats);
                auto col=color(c.tokenId);
                auto prev=previous.find(c.tokenId);
                if (prev!=previous.end() && prev->second.first+1==l.layer)
                    line(prev->second.second,p,col*.6f);
                previous[c.tokenId]={l.layer,p};
                const bool selected=l.layer==s.layer && c.tokenId==s.tokenId;
                double radius=selected?.032:.015;
                if (selected) col={1,1,1};
                auto right=frame.right*radius,up=frame.up*radius;
                for (auto v:{p-right-up,p+right-up,p+right+up,p-right-up,p+right+up,p-right+up}) vertex(points,v,col);
            }
        }
        auto submit=[&](const std::vector<Vertex>& vertices,uint64_t primitive) {
            if (vertices.empty()) return;
            uint32_t count=static_cast<uint32_t>(vertices.size());
            if (bgfx::getAvailTransientVertexBuffer(count,s.layout)!=count) {
                s.message="Scene exceeds this frame's vertex budget; use Flat view.";return;
            }
            bgfx::TransientVertexBuffer buffer;
            bgfx::allocTransientVertexBuffer(&buffer,count,s.layout);
            std::memcpy(buffer.data,vertices.data(),vertices.size()*sizeof(Vertex));
            bgfx::setVertexBuffer(0,&buffer);
            bgfx::setState(BGFX_STATE_WRITE_RGB|BGFX_STATE_WRITE_A|BGFX_STATE_WRITE_Z|BGFX_STATE_DEPTH_TEST_LEQUAL|primitive);
            bgfx::submit(id,s.program);
        };
        s.message.clear();submit(lines,BGFX_STATE_PT_LINES);submit(points,0);
    }
}
