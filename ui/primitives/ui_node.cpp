#include "ui_node.hpp"
#include "ui/overlay_renderer.hpp"
#include "helpers/utf8_edit.hpp"
#include <algorithm>
#include <cmath>

namespace {
float radius(const UINode& node) {
    return std::clamp(node.style.cornerRadius, 0.f,
                      std::max(0.f, std::min(node.bounds.size.x, node.bounds.size.y) / 2));
}
}

bool UINode::contains(Vec2 p) const {
    const auto& b = bounds;
    if (!(p.x >= b.origin.x && p.y >= b.origin.y &&
          p.x < b.origin.x + b.size.x && p.y < b.origin.y + b.size.y)) return false;
    const float r = radius(*this);
    const float x = std::clamp(p.x, b.origin.x + r, b.origin.x + b.size.x - r);
    const float y = std::clamp(p.y, b.origin.y + r, b.origin.y + b.size.y - r);
    return (p.x-x)*(p.x-x) + (p.y-y)*(p.y-y) <= r*r;
}

Vec2 UINode::anchor(UINodeSide side) const {
    const auto& b = bounds;
    const Vec2 center{b.origin.x + b.size.x/2, b.origin.y + b.size.y/2};
    switch (side) {
    case UINodeSide::North: return {center.x, b.origin.y};
    case UINodeSide::East: return {b.origin.x + b.size.x, center.y};
    case UINodeSide::South: return {center.x, b.origin.y + b.size.y};
    case UINodeSide::West: return {b.origin.x, center.y};
    }
    return center;
}

Vec2 UINode::anchorToward(Vec2 target) const {
    const Vec2 center{bounds.origin.x + bounds.size.x/2, bounds.origin.y + bounds.size.y/2};
    if (bounds.size.x <= 0 || bounds.size.y <= 0) return center;
    const float dx = target.x-center.x, dy = target.y-center.y;
    const float extent = std::max(std::abs(dx)/(bounds.size.x/2), std::abs(dy)/(bounds.size.y/2));
    return extent > 0 ? Vec2{center.x + dx/extent, center.y + dy/extent} : center;
}

void UINode::draw(const UICanvas& canvas, OverlayRenderer& renderer) const {
    if (!canvas.isWorldRectVisible(bounds) || bounds.size.x <= 0 || bounds.size.y <= 0) return;
    const float zoom = canvas.getZoom();
    const auto origin = canvas.worldToScreen(bounds.origin);
    const Vec2 size{bounds.size.x*zoom, bounds.size.y*zoom};
    renderer.drawRoundedRect(origin, size, style.surface, radius(*this)*zoom);
    if (style.borderWidth > 0)
        renderer.drawRoundedBorder(origin, size, style.border, radius(*this)*zoom, style.borderWidth*zoom);
    const float available = std::max(0.f, bounds.size.x - 2*std::max(0.f, style.titlePadding))*zoom;
    if (available <= 0 || title.empty() || style.titleSize <= 0) return;
    const float font = style.titleSize*zoom;
    std::string label = UTF8Edit::printable(title);
    if (renderer.measureTextWidth(label, font) > available) {
        const std::string ellipsis = "...";
        if (renderer.measureTextWidth(ellipsis, font) > available) return;
        while (!label.empty() && renderer.measureTextWidth(label + ellipsis, font) > available)
            label.resize(UTF8Edit::previous(label, static_cast<int>(label.size())));
        label += ellipsis;
    }
    const float textWidth = renderer.measureTextWidth(label, font);
    renderer.pushClipRect(origin, size);
    renderer.drawText({origin.x + (size.x-textWidth)/2, origin.y + style.titleTop*zoom},
                      label, style.titleColor, font);
    renderer.popClipRect();
}
