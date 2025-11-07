#include "ui_progress_bar.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

UIProgressBar::UIProgressBar(const std::string& lbl, float maxVal)
    : label(lbl), maxValue(maxVal)
{
}

void UIProgressBar::setValue(float val) {
    value = std::clamp(val, 0.0f, maxValue);
}

void UIProgressBar::setMaxValue(float maxVal) {
    maxValue = std::max(0.0f, maxVal);
    value = std::clamp(value, 0.0f, maxValue);
}

float UIProgressBar::getProgress() const {
    if (maxValue == 0.0f) return 0.0f;
    return std::clamp(value / maxValue, 0.0f, 1.0f);
}

void UIProgressBar::update(const InputState& input, float dt) {
    // Progress bars are not interactive, so no input handling needed
    // Could add hover detection here if needed for tooltips
}

void UIProgressBar::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UIProgressBar::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    // Draw label
    renderer.drawText({position.x, position.y + 10}, label, textColor);

    // Calculate progress bar dimensions
    Vec2 barStart = {position.x + 150, position.y + 10};
    Vec2 barSize = {size.x - 160, 20};

    // Draw background
    renderer.drawRect(barStart, barSize, bgColor);

    // Draw fill
    float fillWidth = barSize.x * getProgress();
    if (fillWidth > 0) {
        renderer.drawRect(barStart, {fillWidth, barSize.y}, fillColor);
    }

    // Draw border (4 rectangles)
    renderer.drawRect(barStart, {barSize.x, 2}, borderColor);
    renderer.drawRect({barStart.x, barStart.y + barSize.y - 2}, {barSize.x, 2}, borderColor);
    renderer.drawRect(barStart, {2, barSize.y}, borderColor);
    renderer.drawRect({barStart.x + barSize.x - 2, barStart.y}, {2, barSize.y}, borderColor);

    // Draw percentage text if enabled
    if (showPercentage) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << (getProgress() * 100.0f) << "%";
        
        // Center text in bar
        float textX = barStart.x + barSize.x + 10;
        float textY = barStart.y + 6;
        renderer.drawText({textX, textY}, oss.str(), textColor);
    }
}
