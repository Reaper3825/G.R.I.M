#pragma once

#include "PhysicalDepthMap.hpp"                  // DepthUnits
#include "PhysicalFrameConditioner.hpp"          // PhysicalSignalRawToModelTransform
#include "PhysicalPerceptionPrimitiveResult.hpp" // PhysicalEntityTrackState
#include "PhysicalSpatialGroundingResult.hpp"    // PhysicalSupportSurfaceClass / PhysicalEntityMotionState

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  Stage-4 result types — the structured "world state" the model will read.
//
//  Why a separate layer?
//    Raw perception outputs (detections, masks, OCR, depth) change every
//    frame and are noisy. Tracker output adds identity but no semantics.
//    Spatial grounding adds depth + surface class but no inter-entity
//    relationships and no occlusion / text-on-object association.
//
//    The world-state layer is the SINGLE bridge between raw perception and
//    GRIM's reasoning. It fuses the existing per-frame surfaces into one
//    deterministic, identity-keyed snapshot:
//
//      object_id              == track_id (monotonic, never reused)
//      class / confidence     == latest matched detection
//      position               == smoothed model+raw box AND centre
//      velocity               == 2D model-px/sec + depth-axis units/sec
//      visible / occluded     == derived from depth ordering + box overlap
//      text_on_object         == scene-text lines whose centroid lies in box
//      last_seen_time         == steady_clock ns AND frame counter
//      relation_to_other_objects == top-K relations per entity
//
//  Numerical-precision contract:
//    * Every coordinate is stored in BOTH model and raw space — consumers
//      MUST NOT re-derive one from the other (letterbox padding makes the
//      transform affine-with-offset).
//    * source_frame_counter / source_perception_results_counter /
//      source_grounding_results_counter let the consumer audit which inputs
//      produced this snapshot.
//    * Relations are sorted by other_object_id ASC for stable diffing.
//    * No magic constants live in the result struct — the producing
//      PhysicalWorldStateBuilder owns them via PhysicalWorldStateBuilderConfig.
// ─────────────────────────────────────────────────────────────────────────────

enum class PhysicalEntityVisibility : uint8_t {
    Unknown  = 0,
    Visible  = 1,   // not occluded by any closer entity this frame
    Occluded = 2,   // a closer entity overlaps >= cfg.min_overlap_for_occlusion
    Coasting = 3    // tracker is coasting — no fresh visual evidence this frame
};

inline const char* DescribePhysicalEntityVisibility(PhysicalEntityVisibility v) {
    switch (v) {
        case PhysicalEntityVisibility::Unknown:  return "Unknown";
        case PhysicalEntityVisibility::Visible:  return "Visible";
        case PhysicalEntityVisibility::Occluded: return "Occluded";
        case PhysicalEntityVisibility::Coasting: return "Coasting";
    }
    return "InvalidPhysicalEntityVisibility";
}

enum class PhysicalEntityRelationKind : uint8_t {
    None        = 0,
    Contains    = 1,    // self.box fully encloses other.box
    ContainedBy = 2,    // other.box fully encloses self.box
    Overlaps    = 3,    // boxes intersect, neither contains the other
    LeftOf      = 4,    // self.centre.x < other.centre.x and dx dominates dy
    RightOf     = 5,
    Above       = 6,
    Below       = 7,
    NearerThan  = 8,    // self.range_value < other.range_value by >= depth threshold
    FartherThan = 9
};

inline const char* DescribePhysicalEntityRelationKind(PhysicalEntityRelationKind k) {
    switch (k) {
        case PhysicalEntityRelationKind::None:        return "None";
        case PhysicalEntityRelationKind::Contains:    return "Contains";
        case PhysicalEntityRelationKind::ContainedBy: return "ContainedBy";
        case PhysicalEntityRelationKind::Overlaps:    return "Overlaps";
        case PhysicalEntityRelationKind::LeftOf:      return "LeftOf";
        case PhysicalEntityRelationKind::RightOf:     return "RightOf";
        case PhysicalEntityRelationKind::Above:       return "Above";
        case PhysicalEntityRelationKind::Below:       return "Below";
        case PhysicalEntityRelationKind::NearerThan:  return "NearerThan";
        case PhysicalEntityRelationKind::FartherThan: return "FartherThan";
    }
    return "InvalidPhysicalEntityRelationKind";
}

struct PhysicalEntitySpatialRelation {
    uint64_t                    other_object_id = 0;
    PhysicalEntityRelationKind  kind            = PhysicalEntityRelationKind::None;
    // strength in [0, 1]:
    //   Contains/ContainedBy/Overlaps : intersection_area / min(area_self, area_other)
    //   LeftOf/RightOf/Above/Below    : axis_dominance score (|dx-dy|/(dx+dy)) clamped
    //   NearerThan/FartherThan        : |range_self - range_other| (clamped to [0,1])
    float                       strength        = 0.0f;
};

struct PhysicalWorldEntity {
    // ── Identity ──
    uint64_t                       object_id              = 0;   // == track_id; 0 = invalid
    int32_t                        class_id               = -1;
    std::string                    class_label;
    float                          confidence             = 0.0f;
    PhysicalEntityTrackState       track_state            = PhysicalEntityTrackState::Tentative;

    // ── Position (both spaces; never re-derive) ──
    cv::Rect2f                     model_box;
    cv::Rect2f                     raw_box;
    cv::Point2f                    model_centre;
    cv::Point2f                    raw_centre;

    // ── Velocity ──
    float                          velocity_model_px_per_sec_x  = 0.0f;
    float                          velocity_model_px_per_sec_y  = 0.0f;
    float                          depth_velocity_units_per_sec = 0.0f;
    PhysicalEntityMotionState      motion_state           = PhysicalEntityMotionState::Unknown;
    bool                           moved_since_last_frame = false;

    // ── Visibility / occlusion ──
    PhysicalEntityVisibility       visibility             = PhysicalEntityVisibility::Unknown;
    // 1.0 - (max overlap fraction over closer entities). 1.0 = nothing in front.
    float                          visible_area_fraction  = 1.0f;
    uint64_t                       occluded_by_object_id  = 0;   // 0 = none
    bool                           has_instance_mask      = false;
    int32_t                        instance_mask_pixel_count = 0;

    // ── Depth / surface (copied from grounded entity if present) ──
    bool                           has_depth              = false;
    DepthUnits                     depth_units            = DepthUnits::Relative;
    float                          range_value            = 0.0f;
    float                          range_value_meters     = 0.0f;
    float                          range_confidence       = 0.0f;
    PhysicalSupportSurfaceClass    support_surface        = PhysicalSupportSurfaceClass::Unknown;
    float                          support_surface_score  = 0.0f;
    bool                           path_blocked           = false;
    float                          path_block_score       = 0.0f;

    // ── Text glued to entity (scene-text lines whose quad centroid lies
    //     inside model_box). Empty when the scene-text reader is disabled
    //     or no recogniser is configured. ──
    std::vector<std::string>       text_on_object;

    // ── Time ──
    int64_t                        first_seen_steady_ns       = 0;
    int64_t                        last_seen_steady_ns        = 0;
    uint64_t                       last_seen_frame_counter    = 0;
    uint32_t                       age_in_frames              = 0;
    uint32_t                       hit_streak                 = 0;
    uint32_t                       miss_streak                = 0;

    // ── Relations (sorted by other_object_id ASC; capped to
    //     cfg.max_relations_per_entity by the builder) ──
    std::vector<PhysicalEntitySpatialRelation> relations;
};

struct PhysicalWorldStateSnapshot {
    // Provenance — every consumer MUST verify these before correlating with
    // other surfaces.
    uint64_t                          source_frame_counter              = 0;
    uint64_t                          source_perception_results_counter = 0;
    uint64_t                          source_grounding_results_counter  = 0;

    int32_t                           model_image_width  = 0;
    int32_t                           model_image_height = 0;
    int32_t                           raw_image_width    = 0;
    int32_t                           raw_image_height   = 0;
    PhysicalSignalRawToModelTransform raw_to_model{};

    // steady_clock nanoseconds at the moment BuildPhysicalWorldStateSnapshot
    // returned. Lets a UI show "world-state age" decoupled from frame age.
    int64_t                           built_at_steady_ns = 0;

    // Aggregate counters for the UI / reasoning layer.
    uint32_t                          num_visible_entities  = 0;
    uint32_t                          num_occluded_entities = 0;
    uint32_t                          num_coasting_entities = 0;

    // Sorted by object_id ASC for stable diffing.
    std::vector<PhysicalWorldEntity>  entities;

    // Stage-4 loop-level timing telemetry, in milliseconds.
    double                            tick_total_ms          = 0.0;
    double                            perception_bus_pull_ms = 0.0;
    double                            grounding_bus_pull_ms  = 0.0;
    double                            build_wall_ms          = 0.0;
    double                            publish_bus_ms         = 0.0;
};

}}} // namespace GRIM::Perception::Physical
