#pragma once

#include "PhysicalDepthMap.hpp"
#include "PhysicalImageOperatorState.hpp"
#include "PhysicalFrameConditioner.hpp"     // PhysicalSignalRawToModelTransform

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  Stage-3 result types — depth map plus per-track spatial grounding.
//
//  "Grounding" here means turning a 2D detection/track into a structured
//  spatial fact:
//    * how far away   → range_value (with units + confidence)
//    * resting on?    → support_surface  (Floor / Table / Wall / Unknown)
//    * blocks a path? → path_block_score (0..1) + boolean
//    * moved?         → moved_since_last_frame
//    * moving now?    → motion_state (Static / Moving / Coasting)
//
//  Every field is computed from explicit, auditable inputs (depth map +
//  tracker output + raw-to-model transform). No magic constants live in
//  the result struct — the producing PhysicalSpatialGrounder owns them.
// ─────────────────────────────────────────────────────────────────────────────

enum class PhysicalSupportSurfaceClass : uint8_t {
    Unknown = 0,
    Floor   = 1,    // bottom edge sits at floor depth (depth ramps with image-y)
    Table   = 2,    // bottom edge sits on an elevated plateau
    Wall    = 3     // depth is roughly constant across height (vertical surface)
};

inline const char* DescribePhysicalSupportSurfaceClass(PhysicalSupportSurfaceClass c) {
    switch (c) {
        case PhysicalSupportSurfaceClass::Unknown: return "Unknown";
        case PhysicalSupportSurfaceClass::Floor:   return "Floor";
        case PhysicalSupportSurfaceClass::Table:   return "Table";
        case PhysicalSupportSurfaceClass::Wall:    return "Wall";
    }
    return "InvalidPhysicalSupportSurfaceClass";
}

enum class PhysicalEntityMotionState : uint8_t {
    Unknown = 0,    // not enough history yet
    Static  = 1,    // 2D + depth velocity below noise floor
    Moving  = 2,    // measured velocity > moving_threshold this frame
    Coasted = 3     // tracker is coasting — no fresh detection this frame
};

inline const char* DescribePhysicalEntityMotionState(PhysicalEntityMotionState s) {
    switch (s) {
        case PhysicalEntityMotionState::Unknown: return "Unknown";
        case PhysicalEntityMotionState::Static:  return "Static";
        case PhysicalEntityMotionState::Moving:  return "Moving";
        case PhysicalEntityMotionState::Coasted: return "Coasted";
    }
    return "InvalidPhysicalEntityMotionState";
}

// One grounded entity, keyed by its tracker id. The 2D fields are copied
// from the tracker so a UI consumer does not have to cross-reference two
// snapshots that may be a frame apart.
struct PhysicalGroundedEntity {
    uint64_t                     track_id              = 0;
    int32_t                      class_id              = -1;
    std::string                  class_label;

    // ── 2D (copied from the source PhysicalEntityTrack snapshot) ──
    cv::Rect2f                   model_box;
    cv::Rect2f                   raw_box;

    // ── Depth (median over the box's interior, robust to outliers) ──
    // range_value is interpreted using `units`:
    //   Relative → 0.0 = nearest in frame, 1.0 = farthest in frame
    //   Meters   → metres along the camera optical axis
    DepthUnits                   units                 = DepthUnits::Relative;
    float                        range_value           = 0.0f;
    float                        range_value_meters    = 0.0f;   // 0 if units != Meters
    // 0..1: fraction of pixels in the box that yielded a finite, in-range
    // depth sample. Low values mean the box overlaps a depth-map hole.
    float                        range_confidence      = 0.0f;

    // ── Support surface ──
    PhysicalSupportSurfaceClass  support_surface       = PhysicalSupportSurfaceClass::Unknown;
    float                        support_surface_score = 0.0f;   // 0..1 confidence in the class

    // ── Path obstruction ──
    // path_blocked == (path_block_score >= grounder.path_block_threshold).
    // The score is high when the entity is BOTH close (small range_value
    // for Relative; small range_value_meters for Meters) AND projects into
    // the navigation cone in front of the camera (defined by the grounder).
    bool                         path_blocked          = false;
    float                        path_block_score      = 0.0f;

    // ── Motion ──
    PhysicalEntityMotionState    motion_state          = PhysicalEntityMotionState::Unknown;
    bool                         moved_since_last_frame = false;
    // Velocity in MODEL pixel space (x, y) plus depth-axis velocity.
    // depth_velocity_units_per_sec is in `units` per second:
    //   Relative → change in normalised inverse-depth per second
    //   Meters   → metres per second along the optical axis
    float                        velocity_model_px_per_sec_x = 0.0f;
    float                        velocity_model_px_per_sec_y = 0.0f;
    float                        depth_velocity_units_per_sec = 0.0f;
};

struct PhysicalSpatialGroundingResults {
    // Provenance: source frame counter from the FrameBus AND the tracker
    // results counter from the PerceptionPrimitiveBus that fed this output.
    // Consumers MUST verify both before correlating with other surfaces.
    uint64_t                            source_frame_counter           = 0;
    uint64_t                            source_perception_results_counter = 0;

    int32_t                             model_image_width   = 0;
    int32_t                             model_image_height  = 0;
    int32_t                             raw_image_width     = 0;
    int32_t                             raw_image_height    = 0;
    PhysicalSignalRawToModelTransform   raw_to_model{};

    PhysicalImageOperatorState          depth_estimator_state = PhysicalImageOperatorState::NoModelConfigured;
    std::string                         depth_estimator_last_error;
    double                              last_depth_inference_ms = 0.0;
    uint64_t                            depth_inference_count   = 0;

    PhysicalImageOperatorState          grounder_state = PhysicalImageOperatorState::NoModelConfigured;
    std::string                         grounder_last_error;
    double                              last_grounding_ms = 0.0;
    uint64_t                            grounding_count   = 0;

    PhysicalDepthMap                    depth_map;
    std::vector<PhysicalGroundedEntity> grounded_entities;
};

}}} // namespace GRIM::Perception::Physical
