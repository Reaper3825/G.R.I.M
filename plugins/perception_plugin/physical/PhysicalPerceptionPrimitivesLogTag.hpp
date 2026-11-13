#pragma once

// Single greppable log tag for the Stage-2 perception-primitives subsystem.
//   LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG, "...")
// Distinct from PHYSICAL_ENV_LOG_TAG so Stage-1 (camera ingest) and
// Stage-2 (image operators) can be filtered independently.
namespace GRIM { namespace Perception { namespace Physical {
inline constexpr const char* PHYSICAL_PERC_PRIM_LOG_TAG = "PhysicalPercPrim";
}}}
