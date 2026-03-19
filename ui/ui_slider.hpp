#pragma once
#include "widget.hpp"
#include <functional>
#include <string>

class UISlider : public Widget {
public:
    UISlider(const std::string& label, float minVal, float maxVal, float initialVal, 
             std::function<void(float)> onChange);

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    
    void setValue(float val);
    float getValue() const { return value; }
    std::string getLabel() const { return label; }
    
    // For overlay rendering
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);
    
    // Check if this slider is editing
    bool isEditing() const { return editingText; }

private:
    std::string label;
    float minValue;
    float maxValue;
    float value;
    std::function<void(float)> callback;
    
    bool dragging = false;
    Vec2 sliderStart{0, 0};
    Vec2 sliderSize{0, 0};
    
    // Text editing (track itself is the editable area)
    bool editingText = false;
    std::string textBuffer;
    float valueBeforeEdit = 0.0f;
    
    float getNormalizedValue() const;
    float getHandleX() const;
};
