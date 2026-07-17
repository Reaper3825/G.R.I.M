#include "PhysicalHandLandmarkMetrics.hpp"

#include <cmath>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

float Distance2D(const PhysicalHandLandmark& a,
                 const PhysicalHandLandmark& b) noexcept
{
    return std::hypot(a.raw_pixel_x - b.raw_pixel_x,
                      a.raw_pixel_y - b.raw_pixel_y);
}

float Distance3D(const PhysicalHandLandmark& a,
                 const PhysicalHandLandmark& b) noexcept
{
    const float dx = a.world_x_m - b.world_x_m;
    const float dy = a.world_y_m - b.world_y_m;
    const float dz = a.world_z_m - b.world_z_m;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool FinitePositive(float value) noexcept {
    return std::isfinite(value) && value > 1.0e-6f;
}

} // namespace

PhysicalPinchMeasurement MeasurePhysicalHandPinch(
    const PhysicalHandObservation& hand) noexcept
{
    PhysicalPinchMeasurement result;
    if (hand.landmark_count <= 17) return result;

    const auto& wrist = hand.landmarks[0];
    const auto& thumb_tip = hand.landmarks[4];
    const auto& index_mcp = hand.landmarks[5];
    const auto& index_tip = hand.landmarks[8];
    const auto& middle_mcp = hand.landmarks[9];
    const auto& pinky_mcp = hand.landmarks[17];

    const bool have_world = wrist.has_world && thumb_tip.has_world &&
        index_mcp.has_world && index_tip.has_world && middle_mcp.has_world &&
        pinky_mcp.has_world;
    if (have_world) {
        result.fingertip_distance = Distance3D(thumb_tip, index_tip);
        const float palm_length = Distance3D(wrist, middle_mcp);
        const float palm_width = Distance3D(index_mcp, pinky_mcp);
        result.palm_scale = 0.5f * (palm_length + palm_width);
        result.used_world_coordinates = true;
    }

    if (!FinitePositive(result.fingertip_distance) ||
        !FinitePositive(result.palm_scale)) {
        result.fingertip_distance = Distance2D(thumb_tip, index_tip);
        const float palm_length = Distance2D(wrist, middle_mcp);
        const float palm_width = Distance2D(index_mcp, pinky_mcp);
        result.palm_scale = 0.5f * (palm_length + palm_width);
        result.used_world_coordinates = false;
    }

    if (!std::isfinite(result.fingertip_distance) ||
        !FinitePositive(result.palm_scale)) return {};
    result.normalized_distance =
        result.fingertip_distance / result.palm_scale;
    result.valid = std::isfinite(result.normalized_distance);
    return result.valid ? result : PhysicalPinchMeasurement{};
}

}}} // namespace GRIM::Perception::Physical
