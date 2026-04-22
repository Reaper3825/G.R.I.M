#pragma once

// Single greppable log tag for the Stage-4 world-state layer.
//   LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG, "...")
// Distinct from PHYSICAL_ENV_LOG_TAG (Stage-1 camera ingest),
// PHYSICAL_PERC_PRIM_LOG_TAG (Stage-2 image operators), and
// PHYSICAL_SPATIAL_GROUND_LOG_TAG (Stage-3 depth + grounding) so all
// four pipeline stages can be filtered independently.
namespace GRIM { namespace Perception { namespace Physical {
inline constexpr const char* PHYSICAL_WORLD_STATE_LOG_TAG = "PhysicalWorldState";
}}}
