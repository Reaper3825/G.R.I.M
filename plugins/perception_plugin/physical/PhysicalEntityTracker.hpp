#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalEntityTracker — Stage-3 identity persistence across frames.
//
//  Input  : per-frame PhysicalObjectDetectorOutput (axis-aligned boxes in
//           MODEL pixel space, plus the affine raw_to_model transform so
//           raw-space boxes can be back-projected coherently).
//  Output : PhysicalEntityTrackerOutput, a snapshot of every track that is
//           alive at the end of this frame (Tentative + Confirmed + Coasting).
//
//  Algorithm (deliberately simple, deterministic, numerically explicit):
//
//    1. PREDICT — every existing track whose last update was on a previous
//       frame has its model-space box translated by velocity * dt, where dt
//       is the wall-time delta in seconds (steady_clock, monotonic). dt is
//       clamped to [0, max_predict_seconds] so a long pause does not
//       teleport a coasted box halfway across the image.
//
//    2. ASSOCIATE — build the cost matrix
//                 cost[t][d] = 1 - IoU(track_t.predicted_box, det_d.model_box)
//       Restrict to same class_id (mismatched classes have cost = +inf).
//       Reject pairings whose IoU < min_iou_for_match (cost > 1 - min_iou).
//       Greedy assignment in ascending order of cost — for the small N we
//       expect (≤ a few dozen tracks/dets) this is provably equivalent to
//       Hungarian under the IoU-gating constraint and is auditable line by
//       line, which Rule 20 prefers over an opaque LAP solver.
//
//    3. UPDATE matched tracks via exponential smoothing in MODEL space:
//          smoothed_box = (1 - alpha) * smoothed_box + alpha * det.model_box
//       Velocity is recomputed from the centre delta divided by dt (or
//       zeroed if dt is below min_velocity_dt_seconds, to avoid noise
//       blowup at high frame rates).
//
//    4. SPAWN one Tentative track per unmatched detection.
//
//    5. AGE unmatched tracks (miss_streak++). Cull when:
//          state == Tentative && miss_streak >= max_tentative_misses
//       OR
//          miss_streak >= max_age_misses
//
//    6. PROMOTE Tentative -> Confirmed when hit_streak >= min_hits_to_confirm.
//
//  Numerical-precision contract:
//    * smoothing alpha is in (0, 1] — bounds-checked at Configure.
//    * raw_box for every track is recomputed by applying raw_to_model to
//      smoothed_model_box on every update. Consumers MUST NOT re-derive it.
//    * IDs are uint64 monotonic; never reused, even after a track is culled.
//
//  Self-owned, internally managed. Mainloop integration point is exactly
//  one call from PhysicalPerceptionPrimitivesLoop:
//
//      RouteDetectionsToPhysicalEntityTracker(detections, raw_to_model,
//                                             raw_w, raw_h, frame_counter,
//                                             out)
//
//  Rule 20 / Rule 3: bad config throws from Configure*. Per-frame routing
//  catches and reports via out.last_error_reason / state==InferenceFailed.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalEntityTrackerConfig {
    // IoU below which a track-detection pair is forbidden from matching.
    float    min_iou_for_match           = 0.30f;

    // Smoothing factor for box position/size (1.0 = pure detection, 0.0 =
    // freeze). Confidence uses the same alpha.
    float    smoothing_alpha             = 0.6f;

    // Promotion / pruning thresholds.
    uint32_t min_hits_to_confirm         = 3;     // Tentative -> Confirmed
    uint32_t max_tentative_misses        = 2;     // cull Tentative early
    uint32_t max_age_misses              = 30;    // cull Confirmed/Coasting

    // Prediction safety bounds.
    double   max_predict_seconds         = 0.5;   // clamp dt for coasting
    double   min_velocity_dt_seconds     = 1e-3;  // below this, treat dt=0

    // If true, only detections whose confidence is >= this floor are fed
    // into the association step. Lets low-quality detections through the
    // detector (where the user can see them in raw form) without polluting
    // the track set.
    float    detection_confidence_floor  = 0.0f;

    // Greedy same-class NMS over the LIVE TRACK set, applied AFTER
    // association and cull. Collapses duplicate tracks that the detector
    // keeps re-spawning over the same physical object (a common YOLO
    // failure mode where one object scores under two anchors / classes
    // close to NMS threshold). Tracks are ranked by:
    //   1. state          (Confirmed > Coasting > Tentative)
    //   2. total_hits     (more evidence wins)
    //   3. smoothed_confidence
    //   4. age_in_frames  (older wins ties)
    // The loser is removed entirely (counted toward total_tracks_culled).
    // 0.0f disables the pass; 1.0f means "only collapse on exact overlap".
    float    cross_track_nms_iou         = 0.70f;
};

class PhysicalEntityTracker {
public:
    PhysicalEntityTracker();
    ~PhysicalEntityTracker();

    // Apply (or re-apply) configuration. Throws on out-of-range values.
    // Calling with the default-constructed config is the canonical "ready"
    // state — there is no model file to load, so it transitions directly
    // to ModelLoaded. NoModelConfigured is reserved for the post-Reset
    // window before the next ConfigurePhysicalEntityTracker call.
    void ConfigurePhysicalEntityTracker(const PhysicalEntityTrackerConfig& cfg);

    // Per-frame entry point. Never throws — failures land in `out`.
    // `detections` MUST come from the same source frame as `raw_to_model`
    // and `source_frame_counter`; the caller (PhysicalPerceptionPrimitives
    // Loop) guarantees this.
    void RouteDetectionsToPhysicalEntityTracker(
        const std::vector<PhysicalObjectDetection>& detections,
        const PhysicalSignalRawToModelTransform& raw_to_model,
        int  raw_image_width,
        int  raw_image_height,
        uint64_t source_frame_counter,
        PhysicalEntityTrackerOutput& out);

    // Drop all tracks and counters. Returns to NoModelConfigured.
    void                       ResetPhysicalEntityTracker();

    PhysicalImageOperatorState GetPhysicalEntityTrackerState() const;
    std::string                GetPhysicalEntityTrackerLastError() const;
    bool                       IsPhysicalEntityTrackerReady() const;

private:
    // Same struct that surfaces in the output, plus internal-only fields
    // required by the predict/update loop.
    struct LiveTrack {
        PhysicalEntityTrack snapshot;          // copied verbatim into output
        cv::Rect2f          predicted_model_box;
        cv::Point2f         last_centre;        // model-space centre at last match
    };

    mutable std::mutex            mutex_;
    PhysicalEntityTrackerConfig   cfg_{};
    PhysicalImageOperatorState    state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                   last_error_reason_;
    uint64_t                      inference_count_       = 0;
    uint64_t                      next_track_id_         = 1;     // 0 is sentinel "no track"
    uint64_t                      total_tracks_spawned_  = 0;
    uint64_t                      total_tracks_confirmed_= 0;
    uint64_t                      total_tracks_culled_   = 0;
    std::vector<LiveTrack>        live_tracks_;
};

}}} // namespace GRIM::Perception::Physical
