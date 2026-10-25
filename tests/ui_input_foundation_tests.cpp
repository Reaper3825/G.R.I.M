// Focused host-only tests. No renderer, model, Docker, or application startup.
#include "core/input_parser.hpp"
#include "core/platform_window.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_textarea.hpp"
#include "helpers/utf8_edit.hpp"
#include "core/platform_clipboard.hpp"
#include "ui/primitives/ui_slider.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/ui_focus_manager.hpp"
#include "ui/ui_root.hpp"
#include "ui/overlay_renderer.hpp"
#include "ui/ui_renderer.hpp"
#include "helpers/key.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

bool PlatformClipboard::copyText(const std::string&) { return true; }
float PlatformWindow::getMenuBarHeight() { return 0; }
UIRoot::MonitorRect UIRoot::getMonitorRectAt(const Vec2&) const { return {{0,0},{1920,1080}}; }
void OverlayRenderer::drawGlassPanel(const Vec2&, const Vec2&, float, uint32_t, uint32_t, uint32_t, int, float, uintptr_t, bool) {}
void logDebug(const std::string&, const std::string&) {}
void logError(const std::string&, const std::string&) {}
void OverlayRenderer::drawRect(const Vec2&, const Vec2&, uint32_t) {}
void OverlayRenderer::drawRoundedRect(const Vec2&, const Vec2&, uint32_t, float) {}
void OverlayRenderer::drawRoundedBorder(const Vec2&, const Vec2&, uint32_t, float, float) {}
void OverlayRenderer::drawText(const Vec2&, const std::string&, uint32_t, float) {}
float OverlayRenderer::measureTextWidth(const std::string& text, float) const { return static_cast<float>(text.size()) * 8; }
void OverlayRenderer::drawLine(const Vec2&, const Vec2&, uint32_t, float) {}
void OverlayRenderer::pushClipRect(const Vec2&, const Vec2&) {}
void OverlayRenderer::popClipRect() {}
void UIRenderer::drawRect(const Vec2&, const Vec2&, uint32_t, uint16_t) {}
void UIRenderer::drawText(const Vec2&, const std::string&, uint32_t) {}
int Key::toPlatformCode(KeyCode key) {
    switch (key) {
    case KeyCode::Backspace: return VK_BACK;
    case KeyCode::Delete: return VK_DELETE;
    case KeyCode::Escape: return VK_ESCAPE;
    case KeyCode::Enter: return VK_RETURN;
    case KeyCode::Left: return VK_LEFT;
    case KeyCode::Right: return VK_RIGHT;
    case KeyCode::Up: return VK_UP;
    case KeyCode::Down: return VK_DOWN;
    case KeyCode::Home: return VK_HOME;
    case KeyCode::End: return VK_END;
    case KeyCode::A: return 'A';
    default: throw std::runtime_error("Unexpected key in test");
    }
}
static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static auto& router = GRIM::UI::InputRouter::get();
static void begin(bool cancel = false) { router.beginFrame(cancel); router.currentPanel = 1; }
static void end() { router.endFrame(); }

void runCanvasTests();
void runSplitPaneTests();
#ifdef GRIM_WORLD_LAB_TESTS
void runWorldLabTests();
void runWorldAssetTests();
#endif
#ifdef GRIM_TEXTWORLD_INTEGRATION
void runTextWorldClientTests();
#endif
int main() try {
#ifdef GRIM_TEXTWORLD_INTEGRATION
    runTextWorldClientTests();
#endif
#ifdef GRIM_WORLD_LAB_TESTS
    runWorldLabTests();
    runWorldAssetTests();
#endif
    runCanvasTests();
    runSplitPaneTests();
    // Console-style title controls update after panel chrome. They must win
    // capture over dragging, including when released outside the panel.
    struct ToolbarPanel : UIPanel {
        bool isTitleBarControlAt(const Vec2& p) const override {
            return p.x >= 200 && p.x <= 300 && p.y >= 0 && p.y <= 30;
        }
    } panel;
    panel.setPosition(0,0); panel.setSize(400,200);
    int toolbarClicks = 0;
    UIButton toolbar("tab", [&]{ ++toolbarClicks; });
    toolbar.setPosition(200,0); toolbar.setSize(100,30);
    InputState gesture; gesture.mousePos = {250,15};
    gesture.mousePressed[0] = gesture.mouseDown[0] = true;
    begin(); panel.update(gesture,0); toolbar.update(gesture,0); end();
    check(toolbar.hasPointerCapture() && !panel.isDragging(), "Title toolbar lost click to panel drag");
    gesture.mousePressed[0] = gesture.mouseDown[0] = false;
    gesture.mouseReleased[0] = true;
    begin(); panel.update(gesture,0); toolbar.update(gesture,0); end();
    check(toolbarClicks == 1 && !router.capture, "Title toolbar release failed");
    gesture.mouseReleased[0] = false;
    gesture.mousePressed[0] = gesture.mouseDown[0] = true;
    begin(); panel.update(gesture,0); toolbar.update(gesture,0); end();
    gesture.mousePos = {500,300};
    gesture.mousePressed[0] = gesture.mouseDown[0] = false;
    gesture.mouseReleased[0] = true;
    begin(); panel.update(gesture,0); toolbar.update(gesture,0); end();
    check(toolbarClicks == 1 && !router.capture, "Outside toolbar release activated or stuck");
    gesture.mousePos = {150,15}; gesture.mouseReleased[0] = false;
    gesture.mousePressed[0] = gesture.mouseDown[0] = true;
    begin(); panel.update(gesture,0); toolbar.update(gesture,0); end();
    check(panel.isDragging(), "Unoccupied title bar stopped dragging");
    begin(true); panel.update(gesture,0); toolbar.update(gesture,0); end();

    PlatformWindow::ScrollDelta wheel;
    wheel.addWheel(60, 120); wheel.addWheel(60, 60);
    check(wheel.x == 1 && wheel.y == 1.5f, "Wheel events were lost or rounded");
    wheel.addPrecise(4, 8);
    check(std::abs(wheel.x - 1.1f) < .0001f && std::abs(wheel.y - 1.7f) < .0001f, "Precise scroll units");
    InputState source; source.mousePos = {5, 5};
    check(!source.clipped({0,0},{20,20}).clipped({10,10},{20,20}).pointerInClip(), "Nested clip intersection");

    int clicks = 0;
    UIButton first("first", [&]{++clicks;}), second("second", [&]{clicks += 100;});
    first.setSize(100,30); second.setSize(100,30);
    source.mousePressed[0] = source.mouseDown[0] = true;
    begin(); first.update(source,0); second.update(source,0); end();
    check(first.hasPointerCapture() && !second.hasPointerCapture(), "Gesture has multiple owners");
    source.mousePressed[0] = false; source.mouseDown[0] = false; source.mouseReleased[0] = true;
    begin(); first.update(source,0); second.update(source,0); end();
    check(clicks == 1, "Release fell through to sibling");
    source.mouseReleased[0] = false; source.mousePressed[0] = source.mouseDown[0] = true;
    begin(); first.update(source,0); end();
    source.mousePressed[0] = false; source.mouseDown[0] = false; source.mouseReleased[0] = true;
    source.mousePos = {500,500};
    begin(); first.update(source,0); end();
    check(!first.hasPointerCapture() && clicks == 1, "Outside release activated or stuck");
    source.mousePos = {5,5}; source.mouseReleased[0] = false; source.mousePressed[0] = source.mouseDown[0] = true;
    begin(); first.update(source,0); end();
    begin(true); first.update(source,0); end();
    check(!first.hasPointerCapture() && clicks == 1, "Cancellation activated button");
    begin(); first.update(source,0); end();
    first.setVisible(false);
    check(!router.capture, "Hidden widget kept capture");
    first.setVisible(true);

    Widget editor, canvas;
    begin(); editor.routeInput(source); canvas.routeInput(source); editor.setFocused(true);
    source.keyPressed[VK_DELETE] = true; source.textInput = "name";
    check(!canvas.routeInput(source).keyPressed[VK_DELETE] && canvas.routeInput(source).textInput.empty(), "Editor shortcut leaked to canvas");
    check(editor.routeInput(source).keyPressed[VK_DELETE], "Focused editor lost keyboard");
    editor.openInputPopup(); editor.closeInputPopup();
    check(!canvas.routeInput(source).mousePressed[0], "Dismissal click leaked"); end();
    begin(); end();
    check(!router.focus, "Unmounted tab kept focus");

    auto child = std::make_shared<UIButton>("clipped", [&]{++clicks;});
    child->setPosition(0,80); child->setSize(80,40);
    UIScrollBox list; list.setPosition(0,0); list.setSize(100,100); list.addChild(child);
    source = {}; source.mousePos={10,110}; source.mousePressed[0]=source.mouseDown[0]=true;
    begin(); list.update(source,0); end();
    check(!child->hasPointerCapture(), "Partially clipped child received outside click");
    source.mousePos={10,90}; begin(); list.update(source,0); end();
    check(child->hasPointerCapture(), "Visible child region not interactive");
    source.mouseDown[0]=source.mousePressed[0]=false; source.mouseReleased[0]=true;
    begin(); list.update(source,0); end();
    check(clicks==2, "Clipped child release failed");

    UISlider slider("value",0,10,0,[](float){}); slider.setSize(400,40);
    source={}; source.mousePos={150,20}; source.mousePressed[0]=source.mouseDown[0]=true;
    begin(); slider.update(source,0); end();
    source.mousePressed[0]=false; source.mousePos={500,20};
    begin(); slider.update(source,0); end();
    check(slider.getValue()==10, "Slider lost capture outside bounds");
    source.mouseDown[0]=false; source.mouseReleased[0]=true;
    begin(); slider.update(source,0); end();
    check(!slider.hasPointerCapture(), "Slider release stuck");
    begin();
    const std::vector<uint64_t> overlapping{1,2,3};
    check(router.pointerTarget(overlapping, [](auto id){return id;}) == 3, "Topmost routing failed");
    router.claimPointer({999,1});
    check(router.pointerTarget(overlapping, [](auto id){return id;}) == 1, "Capture did not override hover");
    router.forgetPanel(1); end();

    // Real text widgets: valid UTF-8 survives typing, selection and deletion.
    UIInputBox name; name.setSize(300,30);
    source={}; source.mousePos={10,10}; source.mousePressed[0]=source.mouseDown[0]=true;
    begin(); name.update(source,0); end();
    source.mousePressed[0]=source.mouseDown[0]=false; source.mouseReleased[0]=true;
    begin(); name.update(source,0); end();
    // InputState carries UTF-8 bytes in std::string, including under C++20.
    const std::string cafe = "caf\xC3\xA9";
    const std::string cafeEmoji = cafe + "\xF0\x9F\x98\x80";
    source={}; source.textInput=cafeEmoji;
    begin(); name.update(source,0); end();
    check(name.getText()==cafeEmoji, "Unicode text was discarded");
    source.textInput.clear(); source.keyPressed[VK_BACK]=true;
    begin(); name.update(source,0); end();
    check(name.getText()==cafe, "Backspace split a UTF-8 character");
    source.keyPressed[VK_BACK]=false; source.keyPressed[VK_LEFT]=true;
    begin(); name.update(source,0); end();
    source.keyPressed[VK_LEFT]=false; source.keyPressed[VK_DELETE]=true;
    begin(); name.update(source,0); end();
    check(name.getText()=="caf", "Delete split a UTF-8 character");
    name.setVisible(false);
    check(!router.focus, "Hidden editor kept focus");

    // A child consuming scroll blocks its parent; boundary refusal can bubble.
    begin(); router.consumeWheel(10);
    check(!router.allowsWheel(20) && router.allowsWheel(10), "Wheel consumption leaked"); end();
    begin(); check(router.allowsWheel(20), "Wheel consumption survived a frame"); end();
    check(UTF8Edit::printable(std::string("\xc0\xaf",2)).empty(), "Invalid UTF-8 accepted");
    std::cout << "Input foundation tests passed\n";
} catch (const std::exception& e) {
    std::cerr << "Test failure: " << e.what() << std::endl;
    return 1;
}
