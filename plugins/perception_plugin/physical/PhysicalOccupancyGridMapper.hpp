#pragma once

#include "PhysicalLocalizationResult.hpp"

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalOccupancyGridMapper
//
//  Maintains a single growing 2D occupancy grid in the world XY plane, in
//  the same log-odds representation Nav2 / SLAM Toolbox use (Thrun §9.2).
//
//  Inputs (per call to RouteCameraPoseToPhysicalOccupancyGridMapper):
//    * The latest camera pose in the world frame (T_world_camera).
//    * The pose's scale state (UnscaledMonocular vs Scaled). The mapper
//      ONLY updates the grid when the scale is known to be metric — a
//      log-odds map populated from unit-norm translations would be
//      meaningless. Rule 20: refuse to silently lie about the geometry.
//
//  What it does:
//    * Marks the cell containing the camera's current world-XY position
//      as "free" by SUBTRACTING `cfg.free_log_odds_increment`.
//    * Marks cells along the line segment from the previous to the
//      current pose as "free" too (Bresenham), so a moving camera carves
//      a corridor of free space.
//    * Cells are clamped to [-cfg.max_abs_log_odds, +cfg.max_abs_log_odds]
//      so a stationary camera does not saturate the map.
//
//  What it deliberately does NOT do (yet):
//    * Mark "occupied" cells. Marking occupancy requires a calibrated
//      depth source (depth map or stereo). When Stage-3 depth is available
//      AND scale_state is metric, a future
//      RouteDepthToPhysicalOccupancyGridMapper() will ray-cast.
//      Until then this mapper produces a free-space-only map — which is
//      already useful for navigation collision-avoidance fallbacks.
//
//  Thread-safety: NOT thread-safe. The loop holds a mutex around every
//  call.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalOccupancyGridMapperConfig {
    // Cell size in world units (meters when scale_state is metric).
    double resolution_meters       = 0.05;     // 5 cm — typical Nav2 default

    // Initial map dimensions. The mapper does NOT auto-grow on this slice
    // (Rule 20: silent reallocation hides bugs). If the camera leaves the
    // grid, RouteCameraPoseToPhysicalOccupancyGridMapper sets `last_error`
    // and skips the update.
    int    initial_cols            = 400;      // 20 m wide @ 5 cm
    int    initial_rows            = 400;
    double world_origin_meters[2]  = {-10.0, -10.0};   // bottom-left cell centre

    // Per-cell free-space update magnitude. Standard Nav2 starts free
    // updates at -0.4 (probability 0.4) and clamps at -2.0 (probability ~0.12).
    float  free_log_odds_increment = 0.4f;
    float  max_abs_log_odds        = 2.0f;
};

class PhysicalOccupancyGridMapper {
public:
    PhysicalOccupancyGridMapper();

    PhysicalOccupancyGridMapper(const PhysicalOccupancyGridMapper&)            = delete;
    PhysicalOccupancyGridMapper& operator=(const PhysicalOccupancyGridMapper&) = delete;

    // Throws on invalid config.
    void ConfigurePhysicalOccupancyGridMapper(const PhysicalOccupancyGridMapperConfig& cfg);

    // Main update. Returns the number of cells whose log-odds was modified
    // by this call (0 when scale_state is non-metric or the camera left the
    // grid; the caller can read `last_error_reason` for the explanation).
    uint64_t RouteCameraPoseToPhysicalOccupancyGridMapper(
        const cv::Mat& T_world_camera_4x4_cv64f,
        PhysicalLocalizationPoseScaleState scale_state);

    // Wipe the map and pose history.
    void ResetPhysicalOccupancyGridMapper();

    // Deep-copy the current grid into `out`.
    void CopyOccupancyGridSnapshot(PhysicalOccupancyGrid2D& out) const;

    PhysicalOccupancyGridMapperConfig
        GetPhysicalOccupancyGridMapperConfigSnapshot() const { return cfg_; }

    std::string GetLastOccupancyGridMapperError() const { return last_error_reason_; }

private:
    PhysicalOccupancyGridMapperConfig cfg_;
    PhysicalOccupancyGrid2D           grid_;
    bool                              has_prior_pose_ = false;
    double                            prior_world_x_meters_ = 0.0;
    double                            prior_world_y_meters_ = 0.0;
    std::string                       last_error_reason_;
};

}}} // namespace GRIM::Perception::Physical
