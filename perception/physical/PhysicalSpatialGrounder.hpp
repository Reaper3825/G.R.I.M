#pragma once

#include "PhysicalDepthMap.hpp"
#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"     // PhysicalEntityTrackerOutput
#include "PhysicalSpatialGroundingResult.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalSpatialGrounder — Stage-3 fusion of depth map + entity tracks.
//
//  Inputs:
//    * PhysicalDepthMap                — at MODEL pixel resolution
//    * PhysicalEntityTrackerOutput     — current frame's tracks (model space)
//
//  Output:
//    * std::vector<PhysicalGroundedEntity> — one entry per track that
//      survived AT LEAST one valid depth sample inside its box.
//
//  Algorithm (deliberately explicit and auditable — Rule 20):
//
//    1. RANGE — for each track, sample the depth map inside the
//       smoothed_model_box, take the MEDIAN of finite samples (robust to
//       background bleed-through at box edges). Confidence = (#finite /
//       #sampled). If confidence < min_range_confidence, skip the track.
//
//    2. SUPPORT SURFACE — compare the median depth at the box's BOTTOM
//       strip (lower 15%) to the depth at the IMMEDIATE neighbourhood
//       below the box (a strip of equal height extending support_strip_h
//       pixels into the image, clamped to the frame). Three rules:
//
//         a. If the below-strip is BOTH significantly farther in inverse
//            depth (i.e. below-strip inverse < bottom-strip inverse - delta)
//            AND the depth gradient across the box's height ramps in the
//            "near→far as y decreases" direction → Floor.
//         b. If the bottom-strip is ABRUPTLY closer than the strip
//            below it by > table_step → Table (object sits on a plateau
//            with empty space falling away).
//         c. If the depth across the box's interior is approximately
//            CONSTANT (top vs bottom inverse depth differ by less than
//            wall_flatness) AND the box is taller than wall_min_aspect ×
//            wider → Wall.
//         d. Otherwise → Unknown.
//
//       support_surface_score is the absolute magnitude of the rule's
//       discriminator divided by its threshold, clamped to [0,1].
//
//    3. PATH OBSTRUCTION — a track is "in the navigation cone" iff the
//       horizontal centre of its model_box falls within nav_cone_half_width
//       (fraction of model_w) of the frame's horizontal midline AND the
//       box's bottom edge falls below nav_cone_min_y_frac of model_h
//       (i.e. it's on the floor in front of the camera). Score combines
//       cone-membership (binary) with closeness:
//
//         closeness = 1 - clamp(range_value, 0, 1)            [Relative]
//         closeness = clamp(1 - range_meters/max_path_m, 0,1) [Meters]
//         path_block_score = cone_membership ? closeness : 0
//
//       path_blocked = (path_block_score >= path_block_threshold).
//
//    4. MOTION — Coasting tracks ⇒ Coasted. Else compute speed in MODEL
//       pixels per second from the tracker velocity, plus depth velocity
//       from (this frame's median depth) − (last frame's median depth) /
//       dt. moving_threshold_px_per_sec gates 2D; moving_threshold_inv_per_sec
//       gates depth-axis. Either over threshold ⇒ Moving.
//       moved_since_last_frame is true when EITHER component changed by
//       more than its respective per-frame quantum (px_per_frame /
//       inv_per_frame).
//
//  Self-owned, internally managed. Single mainloop entry point is
//  RouteDepthAndTracksToPhysicalSpatialGrounder().
//
//  Per-track depth + timestamp history is kept here (not on the tracker)
//  because the tracker has no notion of depth.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalSpatialGrounderConfig {
    // Sampling.
    float    min_range_confidence            = 0.10f;   // % of finite samples in box

    // Support-surface heuristics. All thresholds are in NORMALISED inverse
    // depth ([0,1] across the frame) — operate identically in Relative and
    // Meters mode because the depth map is normalised either way.
    int      support_strip_height_px         = 24;
    float    floor_below_delta               = 0.06f;
    float    table_step                      = 0.18f;
    float    wall_flatness                   = 0.04f;
    float    wall_min_aspect                 = 1.30f;   // height / width to call Wall

    // Path-blocking heuristic.
    float    nav_cone_half_width_frac        = 0.30f;
    float    nav_cone_min_y_frac             = 0.55f;
    float    path_block_threshold            = 0.50f;
    double   max_path_meters                 = 4.0;     // beyond this ⇒ closeness = 0

    // Motion.
    float    moving_threshold_px_per_sec     = 8.0f;
    float    moving_threshold_inv_per_sec    = 0.05f;   // normalised inv-depth/sec
    float    moved_quantum_px                = 2.0f;    // per-frame minimum to call "moved"
    float    moved_quantum_inv               = 0.01f;
};

class PhysicalSpatialGrounder {
public:
    PhysicalSpatialGrounder();
    ~PhysicalSpatialGrounder();

    // Apply (or re-apply) configuration. Throws on out-of-range values.
    // After this call the grounder transitions to ModelLoaded (no model
    // file to load — its "model" is its parameter set).
    void ConfigurePhysicalSpatialGrounder(const PhysicalSpatialGrounderConfig& cfg);

    // Per-frame entry point. Never throws — failures land in `out_state`/`out_error`.
    // `depth_map` and `tracks` MUST come from the same source frame; the
    // caller (PhysicalSpatialGroundingLoop) guarantees this.
    void RouteDepthAndTracksToPhysicalSpatialGrounder(
        const PhysicalDepthMap&                 depth_map,
        const PhysicalEntityTrackerOutput&      tracks,
        int                                     model_image_width,
        int                                     model_image_height,
        std::vector<PhysicalGroundedEntity>&    out_grounded,
        PhysicalImageOperatorState&             out_state,
        std::string&                            out_error,
        double&                                 out_grounding_ms);

    // Drop history and counters; return to NoModelConfigured.
    void                       ResetPhysicalSpatialGrounder();
    PhysicalImageOperatorState GetPhysicalSpatialGrounderState() const;
    std::string                GetPhysicalSpatialGrounderLastError() const;
    bool                       IsPhysicalSpatialGrounderReady() const;
    uint64_t                   GetPhysicalSpatialGrounderRunCount() const;

private:
    struct DepthHistoryEntry {
        float    last_inverse_depth      = 0.0f;
        float    last_metric_meters      = 0.0f;
        int64_t  last_seen_steady_ns     = 0;
        uint64_t last_seen_frame_counter = 0;
        DepthUnits last_units            = DepthUnits::Relative;
    };

    mutable std::mutex                                   mutex_;
    PhysicalSpatialGrounderConfig                        cfg_{};
    PhysicalImageOperatorState                           state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                                          last_error_reason_;
    uint64_t                                             run_count_ = 0;
    std::unordered_map<uint64_t, DepthHistoryEntry>      depth_history_;
};

}}} // namespace GRIM::Perception::Physical
