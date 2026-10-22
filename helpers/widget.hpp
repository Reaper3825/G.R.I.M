#pragma once
#include "vector2.hpp"
#include <string>
#include <memory>
#include <cstdint>

// Include plugin.hpp for GRIM_HOST_API macro
#include "../core/plugin.hpp"

struct InputState;
class UIRenderer;
class OverlayRenderer;

class GRIM_HOST_API Widget {
public:
    Widget();
    virtual ~Widget();
    
    // Focus management
    uint64_t getFocusID() const { return focusID; }
    void setFocusID(uint64_t id) { focusID = id; }
    
    uint64_t getPanelID() const { return panelID; }
    void setPanelID(uint64_t id) { panelID = id; }
    
    // Base focus state management
    virtual bool isFocused() const { return focused; }
    virtual void setFocused(bool focus);
    virtual bool wantsFocus() const { return false; }  // Override in interactive widgets

    virtual void update(const InputState& input, float dt);
    virtual void draw(UIRenderer& renderer);
    
    // Overlay rendering for layered window rendering
    virtual void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos);

    bool isVisible() const { return visible; }
    virtual void setVisible(bool v);
    virtual void cancelInput();

    // Call at the start of a leaf control's update. Containers pass the original
    // input to children, intersecting its clip with their content viewport.
    InputState routeInput(const InputState& input);
    bool capturePointer();
    void releasePointer();
    bool hasPointerCapture() const;
    void openInputPopup();
    void closeInputPopup();
    bool hasInputPopup() const;
    void consumeScroll();

    Vec2 getPosition() const { return position; }
    void setPosition(float x, float y);
    void setPosition(const Vec2& pos) { position = pos; }

    Vec2 getSize() const { return size; }
    void setSize(float w, float h);
    void setSize(const Vec2& sz) { size = sz; }

protected:
    bool visible = true;
    bool focused = false;  // Base focus state
    Vec2 position{0.0f, 0.0f};
    Vec2 size{100.0f, 50.0f};
    uint64_t focusID = 0;  // Unique widget focus identifier
    uint64_t panelID = 0;  // ID of panel this widget belongs to
};
