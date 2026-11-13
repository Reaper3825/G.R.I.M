#pragma once

#include "PhysicalHandGestureBackend.hpp"

namespace GRIM { namespace Perception { namespace Physical {

// Stage-2 auxiliary human-interaction branch. Tick is non-blocking: it only
// pulls the newest camera frame and replaces a one-frame worker queue.
void TickPhysicalInteraction();
void ShutdownPhysicalInteraction();

PhysicalHandGestureConfig GetPhysicalHandGestureConfig();
void RequestConfigurePhysicalHandGestures(
    const PhysicalHandGestureConfig& config);
void RequestSetPhysicalHandGesturesEnabled(bool enabled);

}}} // namespace GRIM::Perception::Physical
