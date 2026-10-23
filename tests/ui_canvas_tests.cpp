#include "ui/primitives/ui_canvas.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
bool closeEnough(float a, float b) { return std::abs(a-b) < 0.001f; }
bool closeEnough(Vec2 a, Vec2 b) { return closeEnough(a.x,b.x) && closeEnough(a.y,b.y); }
auto& router = GRIM::UI::InputRouter::get();
void frame(UICanvas& canvas, const InputState& input, bool cancel = false) {
    router.beginFrame(cancel); router.currentPanel = 90;
    canvas.update(input, 0); router.endFrame();
}
}
void runCanvasTests() {
    UICanvas canvas;
    canvas.setPosition(100,50); canvas.setSize(400,300);
    canvas.setView({-20,40}, 2);
    require(closeEnough(canvas.screenToWorld(canvas.worldToScreen({35,-55})), {35,-55}), "Canvas transform round trip");
    const Vec2 anchor{220,140};
    const Vec2 world = canvas.screenToWorld(anchor);
    canvas.zoomAt(anchor, 3);
    require(closeEnough(canvas.screenToWorld(anchor), world), "Canvas zoom anchor moved");
    canvas.zoomAt(anchor, 100);
    require(closeEnough(canvas.getZoom(),4) && closeEnough(canvas.screenToWorld(anchor),world), "Canvas zoom clamp moved anchor");
    canvas.setView({0,0}, 1);
    canvas.setView({std::numeric_limits<float>::infinity(),0}, 2);
    require(closeEnough(canvas.getZoom(),1), "Canvas accepted nonfinite view");
    canvas.fitToBounds({{20,30},{100,100}},25);
    require(closeEnough(canvas.worldToScreen({70,80}), {300,200}) && closeEnough(canvas.getZoom(),2.5f), "Canvas fit centering/scale");
    require(canvas.isWorldRectVisible({{20,30},{100,100}}) && !canvas.isWorldRectVisible({{10000,10000},{10,10}}), "Canvas culling");
    canvas.resetView();
    InputState input; input.mousePos = {200,100}; input.mouseWheelDelta = 60;
    frame(canvas,input);
    require(canvas.getZoom() > 1 && canvas.getZoom() < 1.2f, "Canvas fractional wheel lost");
    const float scale = canvas.getZoom();
    input.mousePos = {90,100}; frame(canvas,input);
    require(closeEnough(canvas.getZoom(),scale), "Canvas handled wheel outside viewport");
    input.mousePos = {200,100};
    frame(canvas,input.clipped({300,100},{30,30}));
    require(closeEnough(canvas.getZoom(),scale), "Canvas ignored ancestor input clip");
    input.mouseWheelDelta = 0;
    input.mousePressed[2] = input.mouseDown[2] = true;
    frame(canvas,input);
    require(canvas.isPanning() && canvas.hasPointerCapture(), "Canvas pan did not capture");
    const Vec2 origin = canvas.getViewOrigin();
    input.mousePressed[2] = false; input.mousePos = {600,400};
    frame(canvas,input);
    require(closeEnough(canvas.getViewOrigin(),{origin.x - 400/scale, origin.y - 300/scale}), "Canvas outside pan incorrect");
    input.mouseDown[2] = false; input.mouseReleased[2] = true;
    frame(canvas,input);
    require(!canvas.isPanning() && !canvas.hasPointerCapture(), "Canvas outside release stuck");
    input = {}; input.mousePos = {200,100}; input.mousePressed[2] = input.mouseDown[2] = true;
    frame(canvas,input); frame(canvas,input,true);
    require(!canvas.isPanning() && !canvas.hasPointerCapture(), "Canvas cancel stuck");
    frame(canvas,input); canvas.setVisible(false);
    require(!canvas.isPanning() && !canvas.hasPointerCapture(), "Hidden canvas retained capture");
    canvas.setVisible(true);

    struct ContentCanvas : UICanvas {
        UIButton button{"node", []{}};
        int draws = 0;
        bool updateCanvasContent(const InputState& input, float dt) override {
            button.setPosition(worldToScreen({0,0})); button.setSize(80,30);
            button.update(input,dt);
            return false; // Router capture still protects content.
        }
        void drawCanvasContent(OverlayRenderer& r) override {
            ++draws;
            drawWorldRect(r,{{0,0},{100,40}},0xFFFFFFFF);
            drawWorldLine(r,{0,0},{100,40},0xFFFFFFFF);
            drawWorldText(r,{0,0},"node",0xFFFFFFFF);
        }
    } content;
    content.setSize(400,300);
    input = {}; input.mousePos = {10,10}; input.mousePressed[0] = input.mouseDown[0] = true;
    frame(content,input);
    require(content.button.hasPointerCapture(), "Canvas content did not get first refusal");
    input.mousePressed[2] = input.mouseDown[2] = true;
    input.mouseWheelDelta = 120;
    frame(content,input);
    require(!content.isPanning() && closeEnough(content.getZoom(),1), "Canvas stole content gesture");
    frame(content,{},true);
    OverlayRenderer renderer;
    content.drawOverlay(renderer,{});
    require(content.draws == 1, "Canvas draw extension did not run");
    content.setVisible(false); content.drawOverlay(renderer,{});
    require(content.draws == 1, "Hidden canvas rendered");

    // Nested scrollbox must not scroll after a canvas wheel event at max zoom.
    UIScrollBox parent; parent.setSize(500,400); parent.setContentHeight(800); parent.scrollTo(100);
    auto nested = std::make_shared<UICanvas>(); nested->setSize(300,300); nested->setView({0,0},4);
    parent.addChild(nested);
    input = {}; input.mousePos = {30,30}; input.mouseWheelDelta = 120;
    router.beginFrame(false); router.currentPanel = 90; parent.update(input,0); router.endFrame();
    require(closeEnough(parent.getScrollOffset(),100), "Canvas zoom limit leaked wheel to parent");
    router.beginFrame(true); router.endFrame();
}
