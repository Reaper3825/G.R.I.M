#pragma once
#include "helpers/widget.hpp"
#include "ui/ui_theme.hpp"

struct CanvasRect {
    Vec2 origin;
    Vec2 size;
};

// A viewport, not a graph/document model. World units are pixels at zoom 1.
// Position follows other primitives: absolute overlay coordinates during update/draw.
class UICanvas : public Widget {
public:
    Vec2 worldToScreen(Vec2 world) const;
    Vec2 screenToWorld(Vec2 screen) const;
    CanvasRect visibleWorldRect() const;
    bool containsScreenPoint(Vec2 screen) const;
    bool isWorldRectVisible(const CanvasRect& rect) const;

    Vec2 getViewOrigin() const { return viewOrigin; }
    float getZoom() const { return zoom; }
    bool isPanning() const { return panning; }
    // Invalid/nonfinite arguments leave the view unchanged.
    void setView(Vec2 worldOrigin, float scale);
    void setZoomLimits(float minimum, float maximum);
    void zoomAt(Vec2 screenAnchor, float scale);
    void panBy(Vec2 screenDelta);
    void resetView();
    void fitToBounds(const CanvasRect& bounds, float padding = 32.0f);
    void setGridVisible(bool enabled) { showGrid = enabled; }
    void setGridSpacing(float worldSpacing);

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) override;
    void cancelInput() override;

    // Drawing helpers transform world geometry and scale text/strokes.
    // Use within drawCanvasContent so the viewport clip is active.
    void drawWorldRect(OverlayRenderer&, const CanvasRect&, uint32_t color) const;
    void drawWorldLine(OverlayRenderer&, Vec2 from, Vec2 to, uint32_t color, float width = 1.0f) const;
    void drawWorldText(OverlayRenderer&, Vec2 at, const std::string&, uint32_t color,
                       float fontSize = UITheme::Typography::BodySize) const;

protected:
    // Called first with viewport-clipped screen input. Containers forward this
    // to children, which must routeInput themselves. Return true to reserve this
    // frame's navigation for content. Leaf implementations call routeInput.
    // Node dragging/selection, commands, and document storage belong here.
    virtual bool updateCanvasContent(const InputState&, float dt) { return false; }
    virtual void drawCanvasContent(OverlayRenderer&) {}

private:
    Vec2 viewOrigin{0,0};
    float zoom = 1.0f;
    float minZoom = 0.1f, maxZoom = 4.0f;
    float gridSpacing = 32.0f;
    bool showGrid = true;
    bool panning = false;
    Vec2 lastPointer{};
    void drawGrid(OverlayRenderer&) const;
};
