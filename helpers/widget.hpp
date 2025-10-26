#pragma once
#include <string>
#include "helpers/vector2.hpp"

class UIRenderer;
struct InputState; // Forward declaration

class Widget
{
public:
    Widget();
    virtual ~Widget() = default;

    virtual void update(const InputState& input, float dt);
    virtual void draw(UIRenderer& renderer);

    void setPosition(float x, float y);
    void setSize(float w, float h);

    Vec2 getPosition() const { return position; }
    Vec2 getSize() const { return size; }

    bool isVisible() const { return visible; }
    void setVisible(bool v) { visible = v; }

protected:
    Vec2 position{};
    Vec2 size{};
    bool visible = true;
};
