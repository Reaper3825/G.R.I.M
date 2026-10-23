#include "ui_canvas.hpp"
#include "core/input_parser.hpp"
#include "ui/overlay_renderer.hpp"
#include <algorithm>
#include <cmath>

namespace {
bool finite(Vec2 v) { return std::isfinite(v.x) && std::isfinite(v.y); }
}

Vec2 UICanvas::worldToScreen(Vec2 p) const {
    return {position.x + (p.x - viewOrigin.x) * zoom,
            position.y + (p.y - viewOrigin.y) * zoom};
}
Vec2 UICanvas::screenToWorld(Vec2 p) const {
    return {viewOrigin.x + (p.x - position.x) / zoom,
            viewOrigin.y + (p.y - position.y) / zoom};
}
CanvasRect UICanvas::visibleWorldRect() const {
    return {viewOrigin, {std::max(0.0f, size.x) / zoom, std::max(0.0f, size.y) / zoom}};
}
bool UICanvas::containsScreenPoint(Vec2 p) const {
    return p.x >= position.x && p.y >= position.y &&
           p.x < position.x + size.x && p.y < position.y + size.y;
}
bool UICanvas::isWorldRectVisible(const CanvasRect& r) const {
    if (!finite(r.origin) || !finite(r.size) || r.size.x < 0 || r.size.y < 0 || size.x <= 0 || size.y <= 0) return false;
    const auto v = visibleWorldRect();
    return r.origin.x <= v.origin.x + v.size.x && r.origin.y <= v.origin.y + v.size.y &&
           r.origin.x + r.size.x >= v.origin.x && r.origin.y + r.size.y >= v.origin.y;
}
void UICanvas::setView(Vec2 origin, float scale) {
    if (!finite(origin) || !std::isfinite(scale) || scale <= 0) return;
    viewOrigin = origin;
    zoom = std::clamp(scale, minZoom, maxZoom);
}
void UICanvas::setZoomLimits(float minimum, float maximum) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0.001f || maximum > 1000 || minimum > maximum) return;
    minZoom = minimum; maxZoom = maximum;
    zoomAt({position.x + size.x * 0.5f, position.y + size.y * 0.5f}, zoom);
}
void UICanvas::zoomAt(Vec2 anchor, float scale) {
    if (!finite(anchor) || !std::isfinite(scale) || scale <= 0) return;
    const Vec2 world = screenToWorld(anchor);
    const float next = std::clamp(scale, minZoom, maxZoom);
    setView({world.x - (anchor.x - position.x) / next,
             world.y - (anchor.y - position.y) / next}, next);
}
void UICanvas::panBy(Vec2 delta) {
    if (finite(delta)) setView({viewOrigin.x - delta.x / zoom, viewOrigin.y - delta.y / zoom}, zoom);
}
void UICanvas::resetView() { setView({0,0}, 1); }
void UICanvas::setGridSpacing(float spacing) {
    if (std::isfinite(spacing) && spacing >= 0.001f && spacing <= 1e6f) gridSpacing = spacing;
}
void UICanvas::fitToBounds(const CanvasRect& bounds, float padding) {
    if (!finite(bounds.origin) || !finite(bounds.size) || bounds.size.x < 0 || bounds.size.y < 0 ||
        !std::isfinite(padding) || padding < 0 || size.x <= padding * 2 || size.y <= padding * 2) return;
    const float next = std::clamp(std::min((size.x - 2 * padding) / std::max(bounds.size.x, 0.001f),
                                         (size.y - 2 * padding) / std::max(bounds.size.y, 0.001f)), minZoom, maxZoom);
    setView({bounds.origin.x + bounds.size.x * 0.5f - size.x / (2 * next),
             bounds.origin.y + bounds.size.y * 0.5f - size.y / (2 * next)}, next);
}
void UICanvas::cancelInput() {
    panning = false;
    Widget::cancelInput();
}
void UICanvas::update(const InputState& source, float dt) {
    if (!visible) { cancelInput(); return; }
    const auto clipped = source.clipped(position, size);
    routeInput(clipped); // Register ownership before descendants run.
    if (source.interactionCancelled) { cancelInput(); return; }
    if (!hasPointerCapture()) panning = false;
    // An established pan owns its whole gesture, including release outside.
    const bool contentHandled = !panning && updateCanvasContent(clipped, dt);
    const auto input = routeInput(clipped);
    if (panning) {
        if (!input.mouseInputEnabled || !input.mouseDown[2]) {
            panning = false;
            releasePointer();
        } else {
            panBy({input.mousePos.x - lastPointer.x, input.mousePos.y - lastPointer.y});
            lastPointer = input.mousePos;
        }
        return;
    }
    if (contentHandled || !input.mouseInputEnabled || !containsScreenPoint(input.mousePos)) return;
    if (input.mousePressed[2] && capturePointer()) {
        panning = true;
        lastPointer = input.mousePos;
        return;
    }
    const Vec2 wheel = input.scrollSteps();
    if (wheel.x != 0 || wheel.y != 0) {
        if (input.shift) panBy({-42.0f * (wheel.x + wheel.y), 0});
        else {
            if (wheel.y != 0 && std::isfinite(wheel.y))
                zoomAt(input.mousePos, zoom * std::exp(std::clamp(wheel.y * 0.15f, -10.0f, 10.0f)));
            panBy({-42.0f * wheel.x, 0});
        }
        consumeScroll(); // Zoom limits must not leak wheel to a parent scrollbox.
    }
}
void UICanvas::drawWorldRect(OverlayRenderer& r, const CanvasRect& rect, uint32_t color) const {
    if (isWorldRectVisible(rect)) r.drawRect(worldToScreen(rect.origin), {rect.size.x * zoom, rect.size.y * zoom}, color);
}
void UICanvas::drawWorldLine(OverlayRenderer& r, Vec2 a, Vec2 b, uint32_t color, float width) const {
    if (finite(a) && finite(b) && std::isfinite(width) && width > 0)
        r.drawLine(worldToScreen(a), worldToScreen(b), color, width * zoom);
}
void UICanvas::drawWorldText(OverlayRenderer& r, Vec2 at, const std::string& text, uint32_t color, float fontSize) const {
    if (finite(at) && std::isfinite(fontSize) && fontSize > 0)
        r.drawText(worldToScreen(at), text, color, fontSize * zoom);
}
void UICanvas::drawGrid(OverlayRenderer& r) const {
    // Adaptive spacing avoids dense grids at low zoom; bounded loops remain
    // safe for large views and floating-point coordinates.
    float step = gridSpacing * zoom;
    while (step < 16) step *= 2;
    while (step > 64) step *= 0.5f;
    const double worldStep = step / zoom;
    const float xOffset = static_cast<float>(std::fmod(-static_cast<double>(viewOrigin.x), worldStep) * zoom);
    const float yOffset = static_cast<float>(std::fmod(-static_cast<double>(viewOrigin.y), worldStep) * zoom);
    const float startX = xOffset < 0 ? xOffset + step : xOffset;
    const float startY = yOffset < 0 ? yOffset + step : yOffset;
    for (int i = 0; i < 4096 && startX + i * step < size.x; ++i) {
        const float x = position.x + startX + i * step;
        r.drawLine({x, position.y}, {x, position.y + size.y}, UITheme::Colors::DividerFaint);
    }
    for (int i = 0; i < 4096 && startY + i * step < size.y; ++i) {
        const float y = position.y + startY + i * step;
        r.drawLine({position.x, y}, {position.x + size.x, y}, UITheme::Colors::DividerFaint);
    }
}
void UICanvas::drawOverlay(OverlayRenderer& r, const Vec2&) {
    if (!visible || !finite(size) || size.x <= 0 || size.y <= 0) return;
    r.pushClipRect(position, size);
    // Restore the renderer clip even when a custom content hook throws.
    struct ClipGuard { OverlayRenderer& r; ~ClipGuard() { r.popClipRect(); } } guard{r};
    r.drawRect(position, size, UITheme::Colors::ContentAreaBg);
    if (showGrid) drawGrid(r);
    drawCanvasContent(r);
}
