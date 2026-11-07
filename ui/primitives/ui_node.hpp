#pragma once
#include "ui_canvas.hpp"
#include <string>

class OverlayRenderer;

enum class UINodeSide { North, East, South, West };

struct UINodeStyle {
    uint32_t surface = UITheme::Colors::CardSurface;
    uint32_t border = UITheme::Colors::BorderPrimary;
    uint32_t titleColor = UITheme::Colors::TextPrimary;
    float cornerRadius = 10.f;
    float borderWidth = 1.f;
    float titleSize = 15.f;
    float titlePadding = 12.f;
    float titleTop = 9.f;
};

// A reusable canvas-space node, independent of documents and graph topology.
// The containing canvas owns input routing, selection, dragging and connections.
// All geometry/style lengths are world units. Draw inside the canvas content hook.
class UINode {
public:
    CanvasRect bounds;
    std::string title;
    UINodeStyle style;

    bool contains(Vec2 worldPoint) const;
    Vec2 anchor(UINodeSide side) const;
    // Intersection of a center-to-target ray with the node's bounding rectangle.
    Vec2 anchorToward(Vec2 worldTarget) const;
    void draw(const UICanvas& canvas, OverlayRenderer& renderer) const;
};
