// Host-only tests against the real widgets/camera; drawing and OS input are sinks.
#include "ui/primitives/ui_graph.hpp"
#include "ui/primitives/ui_slider.hpp"
#include "ui/ui_orbit_camera.hpp"
#include "ui/ui_focus_manager.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include "helpers/key.hpp"
#include "logger.hpp"
#include <glm/geometric.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
bool mousePressed = false, mouseDown = false;
KeyCode pressedKey = KeyCode::Unknown;
struct Rect { Vec2 position, size; };
std::vector<Rect> rectangles;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool nearlyEqual(double a, double b, double tolerance=1e-5) { return std::abs(a-b)<tolerance; }
template<class F> void rejects(F call) {
    bool rejected=false;
    try { call(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"Invalid input was accepted");
}
}

void logDebug(const std::string&,const std::string&) {}
void logError(const std::string&,const std::string&) {}
bool Mouse::wasPressed(MouseButton) { return mousePressed; }
bool Mouse::isDown(MouseButton) { return mouseDown; }
bool Key::wasPressed(KeyCode code) { return code==pressedKey; }
bool Key::isDown(KeyCode) { return false; }
void OverlayRenderer::drawRect(const Vec2& p,const Vec2& s,uint32_t) { rectangles.push_back({p,s}); }
void OverlayRenderer::drawRoundedRect(const Vec2&,const Vec2&,uint32_t,float) {}
void OverlayRenderer::drawRoundedBorder(const Vec2&,const Vec2&,uint32_t,float,float) {}
void OverlayRenderer::drawText(const Vec2&,const std::string&,uint32_t) {}
void OverlayRenderer::drawLine(const Vec2&,const Vec2&,uint32_t,float) {}
float OverlayRenderer::measureTextWidth(const std::string& text) const { return float(text.size()*8); }

void sliderTests() {
    int calls=0;
    UISlider slider("Layer",1,24,20,[&](float){++calls;},1);
    slider.setSize(400,40);
    slider.setRange(1,6);
    check(slider.getValue()==6 && calls==0,"Range did not silently clamp");
    slider.setValue(2.6f);
    check(slider.getValue()==3,"Programmatic value did not snap");
    slider.setRange(1,1);
    mouseDown=true;
    InputState input;input.mousePos={152,20};slider.update(input,.016f);
    check(slider.getValue()==1 && !slider.isEditing() && calls==0,"Locked range accepted input");
    slider.setRange(1,6);slider.setValue(3);
    input.mousePos={360,20};slider.update(input,.016f);
    check(slider.isEditing(),"Text edit did not start");
    slider.setRange(1,2);
    check(!slider.isEditing() && slider.getValue()==2 && !UIFocusManager::getInstance().isWidgetFocused(slider.getFocusID()),"Range switch retained old edit/focus");
    mouseDown=false;
    // Text entry uses the same integer snapping policy as dragging.
    slider.setRange(1,6);slider.setValue(1);
    mouseDown=true;slider.update(input,.016f);mouseDown=false;
    pressedKey=KeyCode::Backspace;
    for(int i=0;i<5;++i)slider.update(input,.016f);
    pressedKey=KeyCode::Unknown;input.textInput="2.6";slider.update(input,.016f);
    input.textInput.clear();pressedKey=KeyCode::Enter;slider.update(input,.016f);pressedKey=KeyCode::Unknown;
    check(slider.getValue()==3 && calls==1,"Text value did not snap or callback failed");
    slider.setRange(1,2);
    rejects([&]{slider.setRange(3,2);});
    rejects([&]{slider.setRange(0,std::numeric_limits<float>::infinity());});
    rejects([&]{slider.setValue(std::numeric_limits<float>::quiet_NaN());});
    check(slider.getMinValue()==1 && slider.getMaxValue()==2,"Failed range update changed state");
}

void graphTests() {
    UIGraph graph("Coordinates",GraphType::Scatter);
    graph.setPosition(100,200);
    graph.setAxisRange(0,1);graph.setXAxisRange(0,1);
    graph.setData({DataPoint::xy(.2f,.7f,"a"),DataPoint::xy(.8f,.3f,"b")});
    const Vec2 first{216,299}; // origin 150,230; area 330x230
    auto hit=graph.hitTest(first);
    check(hit && hit->pointIndex==0 && hit->seriesIndex==-1,"Explicit XY hit failed");
    check(!graph.hitTest({150,299}),"Explicit X silently used index");
    OverlayRenderer renderer;
    rectangles.clear();graph.drawOverlay(renderer,{0,0});
    bool drawn=false;
    for(const auto& r:rectangles) if(nearlyEqual(r.position.x,212)&&nearlyEqual(r.position.y,295)&&nearlyEqual(r.size.x,8))drawn=true;
    check(drawn,"Scatter drawing and picking disagree");
    graph.clearData();graph.setGraphType(GraphType::MultiLine);
    graph.addSeries("a",{DataPoint::xy(.2f,.7f)},0xFFFFFFFF);
    graph.addSeries("b",{DataPoint::xy(.8f,.3f)},0xFFFFFFFF);
    int series=-1,point=-1,legacy=0;
    graph.setOnSeriesPointHover([&](const GraphPointHit& h,const DataPoint&){series=h.seriesIndex;point=h.pointIndex;});
    graph.setOnPointClick([&](int,const DataPoint& p){check(nearlyEqual(p.value,.3),"Legacy callback got wrong series");++legacy;});
    InputState input;input.mousePos=first;graph.update(input,.016f);
    check(series==0&&point==0,"First series not selected");
    input.mousePos={414,391};mousePressed=true;graph.update(input,.016f);mousePressed=false;
    check(series==1&&point==0&&legacy==1,"Series change at same point index was missed");
    graph.setSeriesVisible("b",false);
    check(!graph.hitTest(input.mousePos),"Hidden series remained selectable");
    graph.clearSeries();check(!graph.hitTest(first),"Cleared data remained selectable");
    graph.setGraphType(GraphType::Area);
    graph.addSeries("visible",{DataPoint(.7f),DataPoint(.3f)},0xFFFFFFFF);
    graph.addSeries("hidden",std::vector<DataPoint>(10,DataPoint(.9f)),0xFFFFFFFF);
    graph.setSeriesVisible("hidden",false);
    check(graph.hitTest({480,391})->seriesIndex==0,"Area hidden series changed X layout");
    graph.clearSeries();
    graph.setGraphType(GraphType::Line);graph.setData({DataPoint(.7f),DataPoint(.3f)});
    check(graph.hitTest({150,299}).has_value(),"Legacy indexed X changed");
    graph.getConfig().autoScaleX=true;graph.setData({DataPoint::xy(.4f,.7f)});
    check(graph.getConfig().minX<.4f && graph.getConfig().maxX>.4f,"Singleton X range degenerate");
    graph.getConfig().maxDataPoints=3;
    std::vector<DataPoint> dense;for(int i=0;i<10;++i)dense.push_back(DataPoint::xy(float(i),.5f));
    graph.setData(dense);
    check(graph.getConfig().minX==0 && graph.getConfig().maxX==9,"Downsampling dropped endpoint");
    auto last=graph.hitTest({480,345});check(last&&last->pointIndex==2,"Downsampling exceeded cap");
    rejects([&]{graph.setXAxisRange(1,1);});
    rejects([&]{graph.setAxisRange(1,0);});
    rejects([&]{graph.setData({DataPoint::xy(std::numeric_limits<float>::infinity(),1)});});
}

void cameraTests() {
    UIOrbitCamera camera;
    for(bool homogeneous:{false,true}) {
        camera.setDistanceLimits(.001,1e6);
        camera.setViewportSize(800,600);camera.setProjection(.8,.001,1000,homogeneous);
        camera.setTarget({0,0,0});camera.setOrbit(.45,.3);camera.setDistance(5);
        const auto center=camera.projectPoint({0,0,0});
        check(center&&nearlyEqual(center->x,400)&&nearlyEqual(center->y,300),"Camera target not centered");
        const auto frame=camera.frame();
        check(glm::length(camera.rayFromViewportPixel(400,300).direction-glm::normalize(frame.target-frame.position))<1e-10,"Center pick ray mismatch");
        const glm::dvec3 point(.4,.2,0);
        const auto screen=camera.projectPoint(point);check(screen.has_value(),"Visible point rejected");
        auto ray=camera.rayFromViewportPixel(screen->x,screen->y);
        check(glm::length(glm::cross(glm::normalize(point-ray.origin),ray.direction))<1e-9,"Project/pick mismatch");
        check(!camera.projectPoint(frame.position*2.0),"Behind-camera point visible");
        camera.panByPixels(10,-5);check(glm::length(camera.frame().target-frame.target)>0,"Pan did nothing");
        const double before=camera.distance();camera.zoomBySteps(1);check(camera.distance()<before,"Zoom direction wrong");
        for(auto viewport:{glm::ivec2(800,600),glm::ivec2(320,900)}) {
            camera.setViewportSize(viewport.x,viewport.y);camera.fitBounds({-1,-2,-3},{1,2,3});
            for(int x:{-1,1})for(int y:{-2,2})for(int z:{-3,3})
                check(camera.projectPoint({x,y,z}).has_value(),"Fit clipped a corner");
        }
        camera.setOrbit(0,100);check(std::isfinite(camera.frame().view[0][0]),"Pole orbit degenerate");
        camera.fitBounds({0,0,0},{0,0,0});check(camera.distance()>0,"Singleton bounds failed");
    }
    rejects([&]{camera.setViewportSize(0,1);});
    rejects([&]{camera.setDistanceLimits(2,1);});
    rejects([&]{camera.fitBounds({1,0,0},{0,0,0});});
    rejects([&]{camera.setProjection(4,.1,100,false);});
    rejects([&]{camera.setOrbit(std::numeric_limits<double>::quiet_NaN(),0);});
}

int main() {
    try { sliderTests();graphTests();cameraTests();std::cout<<"PASS: slider ranges, XY rendering/picking, multi-series callbacks, camera projection/picking/fit\n"; }
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
