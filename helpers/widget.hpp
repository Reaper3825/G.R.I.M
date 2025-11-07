#pragma once
#include "helpers/vector2.hpp"
#include <string>
#include <memory>
#include <cstdint>

// Include plugin.hpp for GRIM_HOST_API macro
#include "core/plugin.hpp"

struct InputState;
class UIRenderer;
class OverlayRenderer;  // ? NEW: Forward declare

class GRIM_HOST_API Widget {
public:
    Widget();
    virtual ~Widget() = default;
    
    // Focus management
    uint64_t getFocusID() const { return focusID; }
    void setFocusID(uint64_t id) { focusID = id; }
    
    uint64_t getPanelID() const { return panelID; }
    void setPanelID(uint64_t id) { panelID = id; }

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
    uint64_t focusID = 0;  // Unique widget focus identifier
    uint64_t panelID = 0;  // ID of panel this widget belongs to
};
