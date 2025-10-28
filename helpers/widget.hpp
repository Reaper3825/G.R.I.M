#pragma once
#include "helpers/vector2.hpp"
#include <string>
#include <memory>

struct InputState;
class UIRenderer;
class OverlayRenderer;  // ? NEW: Forward declare

class Widget {
public:
    Widget();
    virtual ~Widget() = default;

    virtual void update(const InputState& input, float dt);
    virtual void draw(UIRenderer& renderer);
    
    // ? NEW: Overlay rendering for layered window rendering
    virtual void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos);

    bool isVisible() const { return visible; }
    void setVisible(bool v) { visible = v; }

    Vec2 getPosition() const { return position; }
    void setPosition(float x, float y);
    void setPosition(const Vec2& pos) { position = pos; }

    Vec2 getSize() const { return size; }
    void setSize(float w, float h);
    void setSize(const Vec2& sz) { size = sz; }

protected:
    bool visible = true;
    Vec2 position{0.0f, 0.0f};
    Vec2 size{100.0f, 50.0f};
};
