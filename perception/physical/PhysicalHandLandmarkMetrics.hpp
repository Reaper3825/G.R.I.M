#pragma once

#include "PhysicalHandGestureResult.hpp"

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalPinchMeasurement {
    bool valid = false;
    bool used_world_coordinates = false;
    float fingertip_distance = 0.0f;
    float palm_scale = 0.0f;
    float normalized_distance = 0.0f;
};

// Measures thumb-tip (4) to index-tip (8), normalized by the average of two
// stable palm spans: wrist-to-middle-MCP (0..9) and index-to-pinky MCP (5..17).
// Same-hand 3D world landmarks are preferred; aspect-correct raw pixels are
// the fallback.
PhysicalPinchMeasurement MeasurePhysicalHandPinch(
    const PhysicalHandObservation& hand) noexcept;

}}} // namespace GRIM::Perception::Physical
