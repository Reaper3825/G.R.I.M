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
void UIRoot::updateWindowZOrder() {} // Native window effects are outside these host-only tests.
void OverlayRenderer::drawGlassPanel(const Vec2&, const Vec2&, float, uint32_t, uint32_t, uint32_t, int, float, uintptr_t, bool) {}
void OverlayRenderer::drawRect(const Vec2&, const Vec2&, uint32_t) {}
float lastTestRoundedRadius = 0;
void OverlayRenderer::drawRoundedRect(const Vec2&, const Vec2&, uint32_t, float radius) { lastTestRoundedRadius = radius; }
void OverlayRenderer::drawRoundedBorder(const Vec2&, const Vec2&, uint32_t, float, float) {}
std::string renderedTestText;
Vec2 liveFixtureRoomLabel{}, liveFixtureTileLabel{};
Vec2 lastTestTextPosition{};
std::string lastTestText;
float lastTestTextSize = 0;
void OverlayRenderer::drawText(const Vec2& position, const std::string& text, uint32_t, float size) {
    lastTestTextPosition = position; lastTestText = text; lastTestTextSize = size;
    renderedTestText += text + "\n";
    if (text == "Fixture app 0" && position.y > 350) liveFixtureTileLabel = position;
    if (text == "Live UI fixture" && position.x < 450) liveFixtureRoomLabel = position;
}
float OverlayRenderer::measureTextWidth(const std::string& text, float size) const {
    return static_cast<float>(text.size()) * 8 * size / UITheme::Typography::BodySize;
}
std::vector<std::string> OverlayRenderer::wrapText(const std::string& text, float, float) const { return {text}; }
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
