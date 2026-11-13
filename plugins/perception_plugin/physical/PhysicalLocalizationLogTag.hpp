#pragma once

// Single greppable log tag for the Stage-5 localization / SLAM layer.
//   LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG, "...")
// Distinct from PHYSICAL_ENV_LOG_TAG (Stage-1 camera ingest),
// PHYSICAL_PERC_PRIM_LOG_TAG (Stage-2 image operators),
// PHYSICAL_SPATIAL_GROUND_LOG_TAG (Stage-3 depth + grounding), and
// PHYSICAL_WORLD_STATE_LOG_TAG (Stage-4 fused world state) so all five
// pipeline stages can be filtered independently.
namespace GRIM { namespace Perception { namespace Physical {
inline constexpr const char* PHYSICAL_LOCALIZATION_LOG_TAG = "PhysicalLocalization";
}}}
