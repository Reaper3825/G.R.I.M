#include "ui_progress_bar.hpp"
#include "ui_theme.hpp"
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

    // Draw background (rounded)
    float barRadius = barSize.y * 0.5f;
    renderer.drawRoundedRect(barStart, barSize, bgColor, barRadius);

    // Draw fill (rounded)
    float fillWidth = barSize.x * getProgress();
    if (fillWidth > 0) {
        renderer.drawRoundedRect(barStart, {fillWidth, barSize.y}, fillColor, barRadius);
    }

    // Draw border (rounded)
    renderer.drawRoundedBorder(barStart, barSize, borderColor, barRadius);

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
