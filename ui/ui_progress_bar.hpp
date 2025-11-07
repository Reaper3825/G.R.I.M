#pragma once
#include "widget.hpp"
#include <string>

class UIProgressBar : public Widget {
public:
    UIProgressBar(const std::string& label, float maxVal = 100.0f);

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);

    // Progress control
    void setValue(float val);
    void setMaxValue(float maxVal);
    float getValue() const { return value; }
    float getMaxValue() const { return maxValue; }
    float getProgress() const; // Returns 0.0 to 1.0

    // Appearance
    void setLabel(const std::string& lbl) { label = lbl; }
    void setShowPercentage(bool show) { showPercentage = show; }
    void setFillColor(uint32_t color) { fillColor = color; }
    void setBackgroundColor(uint32_t color) { bgColor = color; }

private:
    std::string label;
    float value = 0.0f;
    float maxValue = 100.0f;
    bool showPercentage = true;

    // Colors
    uint32_t fillColor = 0xFF00FFFF;      // Cyan fill
    uint32_t bgColor = 0xFF404040;        // Dark gray background
    uint32_t borderColor = 0xFF606060;    // Border
    uint32_t textColor = 0xFFFFFFFF;      // White text
};
