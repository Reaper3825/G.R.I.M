#pragma once

// Single logger tag for every file in perception/physical/.
// Keeps log lines greppable: `LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, ...)`.
namespace GRIM { namespace Perception { namespace Physical {
inline constexpr const char* PHYSICAL_ENV_LOG_TAG = "PhysicalEnv";
}}}
