#pragma once
#include <string>
#include <functional>
#include <memory>
// or define your Vec2 struct { float x, y; }
struct Vec2 {
    float x;
    float y;
};
class InputState;
class UIRenderer;

class Widget {
public:
    Widget();
    virtual ~Widget() = default;

    // Position & layout
    void setPosition(float x, float y);
    void setSize(float w, float h);
    const Vec2& getPosition() const { return position; }
    const Vec2& getSize() const { return size; }

    // Visibility
    void show(bool v) { visible = v; }
    bool isVisible() const { return visible; }

    // Focus & hover
    bool isFocused() const { return focused; }
    bool isHovered() const { return hovered; }

    // Core virtuals
    virtual void update(const InputState& input, float dt);
    virtual void draw(UIRenderer& renderer);

protected:
    Vec2 position{0, 0};
    Vec2 size{0, 0};
    bool visible = true;
    bool hovered = false;
    bool focused = false;
};
