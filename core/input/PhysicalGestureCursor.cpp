#include "PhysicalGestureCursor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

uint32_t PremultipliedArgb(uint8_t red, uint8_t green, uint8_t blue,
                           float opacity)
{
    const auto alpha = static_cast<uint8_t>(std::clamp(
        static_cast<int>(std::lround(opacity * 255.0f)), 0, 255));
    const auto premultiply = [alpha](uint8_t channel) {
        return static_cast<uint8_t>(
            (static_cast<uint16_t>(channel) * alpha + 127) / 255);
    };
    return (static_cast<uint32_t>(alpha) << 24) |
           (static_cast<uint32_t>(premultiply(red)) << 16) |
           (static_cast<uint32_t>(premultiply(green)) << 8) |
           static_cast<uint32_t>(premultiply(blue));
}

} // namespace

bool PhysicalGestureCursor::SyncArmedState(bool armed, float pinch_openness) {
    constexpr int kApertureLevels = 12;
    const int aperture_level = std::clamp(
        static_cast<int>(std::lround(
            std::clamp(pinch_openness, 0.0f, 1.0f) * kApertureLevels)),
        0, kApertureLevels);

    if (!armed) {
        if (!armed_requested_ && !IsApplied()) return true;
        if (!Restore()) return false;
        armed_requested_ = false;
        applied_aperture_level_ = -1;
        return true;
    }

    requested_aperture_level_ = aperture_level;
    if (armed_requested_ && IsApplied() &&
        applied_aperture_level_ == requested_aperture_level_) return true;
    if (!Apply()) return false;
    armed_requested_ = true;
    applied_aperture_level_ = requested_aperture_level_;
    return true;
}

bool PhysicalGestureCursor::Shutdown() {
    const bool restored = Restore();
    if (restored) {
        armed_requested_ = false;
        applied_aperture_level_ = -1;
    }
    return restored;
}

bool PhysicalGestureCursor::BuildBitmap(
    GRIM::Input::CursorBitmap& bitmap, std::string& error) const
{
    (void)error;
    constexpr int kSize = 48;
    constexpr float kCenter = 24.0f;
    const float aperture = std::clamp(
        static_cast<float>(requested_aperture_level_) / 12.0f, 0.0f, 1.0f);
    const float radius = 5.0f + aperture * 13.0f;
    constexpr float kOutlineHalfWidth = 3.4f;
    constexpr float kColorHalfWidth = 2.0f;

    bitmap.width = kSize;
    bitmap.height = kSize;
    bitmap.hotspot_x = static_cast<int>(kCenter);
    bitmap.hotspot_y = static_cast<int>(kCenter);
    bitmap.argb.assign(static_cast<size_t>(kSize * kSize), 0);

    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const float dx = (static_cast<float>(x) + 0.5f) - kCenter;
            const float dy = (static_cast<float>(y) + 0.5f) - kCenter;
            const float edge_distance = std::abs(std::sqrt(dx * dx + dy * dy) -
                                                 radius);
            const float outline_coverage = std::clamp(
                kOutlineHalfWidth + 0.5f - edge_distance, 0.0f, 1.0f);
            const float color_coverage = std::clamp(
                kColorHalfWidth + 0.5f - edge_distance, 0.0f, 1.0f);
            if (outline_coverage <= 0.0f) continue;

            // A dark outside edge keeps the cursor legible on bright content;
            // the white inner ring remains crisp over dark applications.
            uint8_t red = 18;
            uint8_t green = 20;
            uint8_t blue = 19;
            float opacity = outline_coverage * 0.95f;
            if (color_coverage > 0.0f) {
                red = 255;
                green = 255;
                blue = 255;
                opacity = color_coverage;
            }
            bitmap.argb[static_cast<size_t>(y * kSize + x)] =
                PremultipliedArgb(red, green, blue, opacity);
        }
    }
    return true;
}

}}} // namespace GRIM::Perception::Physical
