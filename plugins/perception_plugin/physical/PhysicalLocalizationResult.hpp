#pragma once

#include "PhysicalFrameConditioner.hpp"   // PhysicalSignalRawToModelTransform

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  Stage-5 result types — camera localization + 2D occupancy mapping.
//
//  Why a separate stage?
//    Stages 1-4 describe WHAT is in front of the camera and HOW the
//    camera-frame world is composed (entities, depth, relations).
//    They are inherently per-frame and ego-centric: every coordinate is
//    expressed in the current image. That is sufficient for "what do I
//    see right now", but it is NOT sufficient for any agent that MOVES
//    — robots, drones, AR glasses, phones, laptops being carried around.
//
//    The localization layer gives the model two things the per-frame
//    pipeline cannot:
//      1. WHERE the camera is in a persistent world frame (6-DoF pose).
//      2. A growing 2D map of free / unknown / occupied space (the
//         Nav2-style occupancy grid that downstream navigation reads).
//
//  Numerical-precision contract:
//    * The world frame is anchored at the FIRST successfully-tracked
//      frame: T_world_camera_at_anchor == identity. No magic origin.
//    * Pose is stored as a 4x4 CV_64F matrix (T_world_camera). Rotation
//      AND translation in one place; consumers MUST NOT recompose them.
//    * Quaternion (w,x,y,z) is also stored, derived from R deterministically.
//      Provided as a convenience — never re-derive from translation/twist.
//    * For monocular VO the translation has SCALE AMBIGUITY. The scale
//      state is reported explicitly via `pose_scale_state`; consumers
//      MUST check this before using `position_meters` in any metric way.
//    * The occupancy grid uses log-odds storage (Thrun, Probabilistic
//      Robotics §9.2). Cells store `log(p/(1-p))`; consumers convert to
//      probability with `1 / (1 + exp(-log_odds))`.
//    * source_frame_counter ties this snapshot back to the FrameBus
//      frame it was produced from. Consumers correlating with Stage-3 /
//      Stage-4 MUST verify counters match.
// ─────────────────────────────────────────────────────────────────────────────

enum class PhysicalLocalizationTrackingState : uint8_t {
    Uninitialized = 0,   // no anchor frame yet
    Initializing  = 1,   // anchor set; waiting for sufficient parallax
    Tracking      = 2,   // last frame produced a valid relative pose
    Lost          = 3,   // matches/parallax fell below threshold this frame
    Failed        = 4    // hard error (no calibration, bad input, etc.)
};

inline const char* DescribePhysicalLocalizationTrackingState(
    PhysicalLocalizationTrackingState s)
{
    switch (s) {
        case PhysicalLocalizationTrackingState::Uninitialized: return "Uninitialized";
        case PhysicalLocalizationTrackingState::Initializing:  return "Initializing";
        case PhysicalLocalizationTrackingState::Tracking:      return "Tracking";
        case PhysicalLocalizationTrackingState::Lost:          return "Lost";
        case PhysicalLocalizationTrackingState::Failed:        return "Failed";
    }
    return "InvalidPhysicalLocalizationTrackingState";
}

enum class PhysicalLocalizationPoseScaleState : uint8_t {
    Unknown          = 0,   // no pose yet
    UnscaledMonocular = 1,  // translation has unit norm; not metric
    ScaledByDepthMap  = 2,  // monocular VO + monocular depth → metric (drifts)
    ScaledByStereo    = 3,  // (future) stereo / RGB-D directly metric
    ScaledByImu       = 4   // (future) VIO-fused metric
};

inline const char* DescribePhysicalLocalizationPoseScaleState(
    PhysicalLocalizationPoseScaleState s)
{
    switch (s) {
        case PhysicalLocalizationPoseScaleState::Unknown:           return "Unknown";
        case PhysicalLocalizationPoseScaleState::UnscaledMonocular: return "UnscaledMonocular";
        case PhysicalLocalizationPoseScaleState::ScaledByDepthMap:  return "ScaledByDepthMap";
        case PhysicalLocalizationPoseScaleState::ScaledByStereo:    return "ScaledByStereo";
        case PhysicalLocalizationPoseScaleState::ScaledByImu:       return "ScaledByImu";
    }
    return "InvalidPhysicalLocalizationPoseScaleState";
}

// ── 6-DoF pose ──────────────────────────────────────────────────────────────
//
// All quantities are "T_world_camera" — i.e. the transform that takes a
// point in the camera frame and expresses it in the world frame.
//
//   p_world = R_world_camera * p_camera + t_world_camera
//
struct PhysicalCameraPose {
    // Full 4x4 transform CV_64F. Source of truth.
    cv::Mat   T_world_camera;            // 4x4 CV_64F

    // Convenience views — derived from T_world_camera at write time.
    // NEVER re-derive these on the consumer side.
    double    position_meters[3]         = {0.0, 0.0, 0.0};   // (x,y,z) world
    double    quaternion_world_camera[4] = {1.0, 0.0, 0.0, 0.0}; // (w,x,y,z)
    double    yaw_pitch_roll_radians[3]  = {0.0, 0.0, 0.0};   // ZYX intrinsic

    // Whether `position_meters` is actually metric. Mirrors snapshot-level
    // pose_scale_state but lives on the pose so a consumer that pulls just
    // the pose still has the contract attached.
    PhysicalLocalizationPoseScaleState scale_state =
        PhysicalLocalizationPoseScaleState::Unknown;
};

// ── Velocity (in the world frame) ───────────────────────────────────────────
struct PhysicalCameraVelocity {
    // Linear velocity. Same scale-state contract as the pose.
    double linear_world_per_sec[3]       = {0.0, 0.0, 0.0};

    // Angular velocity as axis*magnitude (rotation vector), radians/sec.
    // Always metric — independent of translation scale.
    double angular_radians_per_sec[3]    = {0.0, 0.0, 0.0};

    // Wall delta-t the velocity was computed over. 0 means "no prior
    // pose to differentiate against" — consumers MUST treat the linear
    // and angular fields as undefined in that case.
    double sample_interval_seconds       = 0.0;
};

// ── Visual landmark (a 2D feature with optional triangulated 3D position) ──
struct PhysicalVisualLandmark {
    uint64_t    landmark_id              = 0;     // monotonic, never reused; 0 = invalid
    cv::Point2f model_image_pixel;                // last observation, model-space
    int32_t     observation_count        = 0;     // total frames it has been seen
    bool        has_world_position       = false; // false until triangulated
    double      world_position[3]        = {0.0, 0.0, 0.0}; // valid iff has_world_position
};

// ── 2D occupancy grid (Nav2 / SLAM Toolbox style) ──────────────────────────
//
// Cells live on a regular grid in the WORLD-XY plane (Z is up). Each cell
// stores log-odds occupancy; convert to probability with sigmoid().
//
// The grid is anchored at `world_origin_meters` — that point lies at the
// CENTRE of cell (0, 0). Cell (col, row) covers world-XY:
//   x ∈ [origin.x + (col-0.5)*res, origin.x + (col+0.5)*res]
//   y ∈ [origin.y + (row-0.5)*res, origin.y + (row+0.5)*res]
//
// Sentinel: log_odds[i] == 0.0 means "Unknown" (sigmoid(0) == 0.5).
//
// `cells_log_odds` is row-major: index = row * cols + col.
struct PhysicalOccupancyGrid2D {
    int32_t  cols                     = 0;
    int32_t  rows                     = 0;
    double   resolution_meters        = 0.0;
    double   world_origin_meters[2]   = {0.0, 0.0};   // (x, y) of cell (0,0) centre
    std::vector<float> cells_log_odds;                 // size = rows * cols
    uint64_t cells_updated_this_frame = 0;
    uint64_t total_cells_observed     = 0;             // any cell with |log_odds| > 0
};

// ── The full snapshot ──────────────────────────────────────────────────────
struct PhysicalLocalizationSnapshot {
    // Provenance — every consumer MUST verify this before correlating with
    // other surfaces (frame bus, world state, etc.).
    uint64_t                            source_frame_counter = 0;

    int32_t                             model_image_width    = 0;
    int32_t                             model_image_height   = 0;
    int32_t                             raw_image_width      = 0;
    int32_t                             raw_image_height     = 0;
    PhysicalSignalRawToModelTransform   raw_to_model{};

    // steady_clock nanoseconds at the moment the loop produced this
    // snapshot. Useful for "localization age" UIs.
    int64_t                             built_at_steady_ns   = 0;

    // ── State machine ──
    PhysicalLocalizationTrackingState   tracking_state =
        PhysicalLocalizationTrackingState::Uninitialized;
    PhysicalLocalizationPoseScaleState  pose_scale_state =
        PhysicalLocalizationPoseScaleState::Unknown;

    // Human-readable reason. Always populated when state ∈ {Lost, Failed,
    // Initializing}. Empty when Tracking.
    std::string                         tracking_reason;

    // ── Pose / velocity ──
    PhysicalCameraPose                  pose_world_camera;
    PhysicalCameraVelocity              velocity_world_camera;

    // The relative pose from the previous published snapshot to this one.
    // Consumers building cumulative trajectories from snapshot diffs use
    // this; otherwise consume `pose_world_camera` directly.
    cv::Mat                             T_prev_to_current;     // 4x4 CV_64F, identity when no prior

    // ── Tracking metrics (every field auditable) ──
    int32_t                             keypoints_detected_current_frame = 0;
    int32_t                             matches_to_prior_frame           = 0;
    int32_t                             essential_inliers                = 0;
    double                              median_parallax_pixels           = 0.0;
    double                              mean_reprojection_error_pixels   = 0.0;
    double                              last_estimation_ms               = 0.0;
    uint64_t                            estimation_count                 = 0;

    // ── Map ──
    // Camera trajectory in the world frame, in publication order. Capped
    // by PhysicalLocalizationBuilderConfig::max_trajectory_samples.
    std::vector<cv::Point3d>            trajectory_world_meters;

    std::vector<PhysicalVisualLandmark> landmarks;
    PhysicalOccupancyGrid2D             occupancy_grid;
};

}}} // namespace GRIM::Perception::Physical
