#pragma once
/**
 * @file TelemetryLattice_Internal.hpp
 * @brief Internal struct definition for TelemetryLattice
 * 
 * This header exposes the internal structure of TelemetryLattice for
 * use by GPU kernels that need direct memory access.
 * 
 * ONLY include this in .cu files that need direct lattice access!
 * Regular client code should use the opaque API from TelemetryLattice_GPU.hpp
 */

#include "TelemetryState_GPU.hpp"
#include "TelemetryLattice_GPU.hpp"

namespace GRIM::Telemetry {

//=============================================================================
// INTERNAL LATTICE STRUCTURE
//
// MEMORY LAYOUT:
//   levels: [num_levels * num_streams] array of LatticeLevelState
//   Each LatticeLevelState contains TelemetryState + TelemetryVector
//
// ACCESS PATTERN:
//   level_idx = level * num_streams + stream
//   levels[level_idx].vector.mu = ...
//=============================================================================

struct TelemetryLattice {
    // Device memory - flattened 2D array [levels][streams]
    LatticeLevelState* levels = nullptr;  // Was d_levels, renamed for consistency
    
    // Config (CPU-side copy)
    LatticeConfig config;
    
    // Scratch buffers (pre-allocated, no per-call malloc)
    TelemetryVector* d_scratch_vectors = nullptr;
    float* d_observations = nullptr;
    int* d_error_flag = nullptr;
};

} // namespace GRIM::Telemetry
