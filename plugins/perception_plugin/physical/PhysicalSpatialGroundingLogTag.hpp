#pragma once

// Single greppable log tag for the Stage-3 spatial-grounding subsystem.
//   LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG, "...")
// Distinct from PHYSICAL_ENV_LOG_TAG (Stage-1 camera ingest) and
// PHYSICAL_PERC_PRIM_LOG_TAG (Stage-2 image operators) so the three
// pipeline stages can be filtered independently.
namespace GRIM { namespace Perception { namespace Physical {
inline constexpr const char* PHYSICAL_SPATIAL_GROUND_LOG_TAG = "PhysicalSpatialGround";
}}}
