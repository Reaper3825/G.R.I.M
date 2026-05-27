#pragma once
#include "TelemetryLattice_GPU.hpp"

namespace GRIM::Telemetry {

// Internal-only escape hatch for telemetry subsystems that must read the
// lattice device buffer directly without exposing it on the public API.
inline const LatticeLevelState* latticeLevelsDevicePtr(const TelemetryLattice& lattice) {
	return lattice.levels_;
}

} // namespace GRIM::Telemetry
