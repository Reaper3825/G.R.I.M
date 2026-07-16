#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalFrameConditioner.hpp"   // PhysicalSignalRawToModelTransform
#include "PhysicalFrameBus.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  All five primitive result types live here as plain data structs.
//  Numerical-precision contract:
//    * All bounding boxes / keypoints / quads are stored in BOTH coordinate
//      systems explicitly. Consumers MUST NOT re-derive them from one
//      another via dimensions because letterbox padding makes the
//      raw↔model relationship affine-with-offset, not just a scale.
//    * Confidences are normalised to [0, 1].
//    * model_image_width / model_image_height match the conditioned frame
//      that the operator actually saw.
// ─────────────────────────────────────────────────────────────────────────────

// Helper: a 4-corner polygon (used by scene text and rotated detections).
struct PhysicalQuad {
    cv::Point2f p0;
    cv::Point2f p1;
    cv::Point2f p2;
    cv::Point2f p3;
};

// 1. Object detection — a single axis-aligned box
struct PhysicalObjectDetection {
    int32_t      class_id        = -1;
    std::string  class_label;          // resolved from class names file when available
    float        confidence      = 0.0f;
    cv::Rect2f   model_box;            // in MODEL pixel space (always populated)
    cv::Rect2f   raw_box;              // in RAW   pixel space (back-projected)
};

// 2. Semantic segmentation — a per-pixel class id image at MODEL resolution
struct PhysicalSemanticSegmentation {
    cv::Mat                  class_id_image;   // CV_32SC1, size = (model_w, model_h)
    int32_t                  num_classes = 0;
    std::vector<std::string> class_labels;     // class_labels[i] = name for class i (may be empty)
};

// 2b. Instance segmentation — one binary mask per detected object instance,
//     produced by PhysicalInstanceSegmenter (SAM 2) prompted with the
//     PhysicalObjectDetector's boxes. Each mask is stored compactly as
//     mask_within_bbox (CV_8UC1, values 0 or 255) packed inside its tight
//     mask_model_bbox in MODEL pixel space, so consumers can reason about
//     occlusion via mask intersection without paying for full-frame masks.
//
//     Numerical-precision contract:
//       * mask_within_bbox.size() == mask_model_bbox.size() exactly.
//       * mask_model_bbox is fully contained within [0, model_w) x [0, model_h).
//       * mask_pixel_count == cv::countNonZero(mask_within_bbox / 255).
//       * mask_area_fraction = mask_pixel_count / (model_w * model_h).
//       * source_detection_index points back into PhysicalObjectDetectorOutput
//         .detections so the consumer can recover class / raw-space box.
//         If <0 the prompt did not come from the detector (reserved).
struct PhysicalInstanceMask {
    int32_t      source_detection_index = -1;
    int32_t      class_id               = -1;
    std::string  class_label;
    float        detection_confidence   = 0.0f;   // confidence of source detection
    float        mask_confidence        = 0.0f;   // SAM 2 IoU prediction (mask quality)
    cv::Rect2f   prompt_model_box;                // copy of the source box (traceability)
    cv::Rect2i   mask_model_bbox;                 // tight integer bbox in MODEL space
    cv::Mat      mask_within_bbox;                // CV_8UC1, size = mask_model_bbox.size()
    int32_t      mask_pixel_count       = 0;
    float        mask_area_fraction     = 0.0f;
};

struct PhysicalInstanceSegmentation {
    std::vector<PhysicalInstanceMask> instances;
};

// 3. Image classification — top-K candidates over the whole frame
struct PhysicalImageClassification {
    int32_t      class_id        = -1;
    std::string  class_label;
    float        score           = 0.0f;       // softmax probability in [0, 1]
};

// 4. Pose / keypoint estimation — one detected skeleton (we keep multi-person open)
struct PhysicalPoseKeypoint {
    int32_t       joint_id       = -1;
    std::string   joint_label;
    cv::Point2f   model_xy;
    cv::Point2f   raw_xy;
    float         confidence     = 0.0f;
    bool          visible        = false;      // confidence > min_keypoint_confidence
};

struct PhysicalPoseInstance {
    std::vector<PhysicalPoseKeypoint> keypoints;
    float                              instance_confidence = 0.0f;
    cv::Rect2f                         model_bbox;
    cv::Rect2f                         raw_bbox;
};

// 5. Scene text — one detected, optionally recognised line
struct PhysicalSceneTextLine {
    PhysicalQuad model_quad;
    PhysicalQuad raw_quad;
    std::string  text;            // empty when only detection ran (no recogniser configured)
    float        confidence = 0.0f;
};

// 7. Entity track — one persistent identity carried across frames. Produced
//    by PhysicalEntityTracker by associating PhysicalObjectDetection results
//    over time. The track's box is exponentially smoothed in MODEL space and
//    the raw-space box is recomputed from raw_to_model on each update so the
//    two coordinate spaces are guaranteed coherent.
//
//    Lifecycle:
//      Tentative  — newly spawned, not yet confirmed. Drawn dimly by UI.
//      Confirmed  — survived enough hits in a row to be trusted.
//      Coasting   — confirmed but missed by detector this frame; box predicted
//                   from last velocity. Will revert to Confirmed on next hit
//                   or be culled when miss_streak exceeds max_age_misses.
enum class PhysicalEntityTrackState : uint8_t {
    Tentative = 0,
    Confirmed = 1,
    Coasting  = 2
};

inline const char* DescribePhysicalEntityTrackState(PhysicalEntityTrackState s) {
    switch (s) {
        case PhysicalEntityTrackState::Tentative: return "Tentative";
        case PhysicalEntityTrackState::Confirmed: return "Confirmed";
        case PhysicalEntityTrackState::Coasting:  return "Coasting";
    }
    return "InvalidPhysicalEntityTrackState";
}

struct PhysicalEntityTrack {
    uint64_t                  track_id                   = 0;     // monotonically increasing; never reused
    int32_t                   class_id                   = -1;    // class locked at spawn
    std::string               class_label;
    PhysicalEntityTrackState  state                      = PhysicalEntityTrackState::Tentative;

    // Boxes: model-space is the canonical smoothed estimate; raw-space is
    // back-projected via raw_to_model so downstream consumers do NOT need to
    // re-derive it. Letterbox padding makes the relationship affine.
    cv::Rect2f                smoothed_model_box;
    cv::Rect2f                smoothed_raw_box;

    // Per-frame velocity in MODEL pixel space, computed from the centre
    // delta divided by the wall-time delta in seconds (steady_clock-based,
    // so monotonic). Used by the Coasting prediction step.
    double                    velocity_px_per_sec_x      = 0.0;
    double                    velocity_px_per_sec_y      = 0.0;

    float                     last_detection_confidence  = 0.0f;
    float                     smoothed_confidence        = 0.0f;

    uint32_t                  age_in_frames              = 0;     // frames since spawn (incl. coasted)
    uint32_t                  hit_streak                 = 0;     // consecutive frames matched
    uint32_t                  miss_streak                = 0;     // consecutive frames missed
    uint32_t                  total_hits                 = 0;
    uint32_t                  total_misses               = 0;

    uint64_t                  first_seen_frame_counter   = 0;
    uint64_t                  last_update_frame_counter  = 0;     // matched OR coasted
    uint64_t                  last_matched_frame_counter = 0;     // match only
    int64_t                   first_seen_steady_ns       = 0;
    int64_t                   last_update_steady_ns      = 0;
};

// 8. Facial expression — one detected face plus its top emotion class
struct PhysicalFacialExpression {
    cv::Rect2f               model_bbox;           // MODEL pixel space (always populated)
    cv::Rect2f               raw_bbox;            // RAW   pixel space (back-projected)
    float                    detection_confidence = 0.0f;  // YuNet face score
    int32_t                  expression_id        = -1;    // argmax over class scores
    std::string              expression_label;             // resolved from labels file
    float                    expression_score     = 0.0f;  // softmax probability of top class
    std::vector<float>       all_class_scores;             // full softmax distribution (size == K)
};

// ─────────────────────────────────────────────────────────────────────────────
//  Per-operator output envelope. Each carries its own state + last-error
//  reason so the bus consumer can render BOTH the result AND the lifecycle
//  status in one pull.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalObjectDetectorOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;   // total successful inferences since load
    uint64_t                    last_frame_counter = 0; // PhysicalFrameBus counter that produced this
    double                      last_inference_ms = 0.0;
    PhysicalCacheStatus         cache_status{};
    std::vector<PhysicalObjectDetection> detections;
};

struct PhysicalSemanticSegmenterOutput {
    PhysicalImageOperatorState   state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                  last_error_reason;
    uint64_t                     inference_count  = 0;
    uint64_t                     last_frame_counter = 0;
    double                       last_inference_ms = 0.0;
    PhysicalCacheStatus          cache_status{};
    PhysicalSemanticSegmentation segmentation;
};

// PhysicalInstanceSegmenter (SAM 2) envelope. Reports timing for the
// expensive image-encoder pass separately from the (much cheaper) per-prompt
// decoder pass so the UI can show what dominates wall-time.
//   prompt_count       == number of detector boxes routed THIS frame
//   instances.size()   == number of masks actually decoded (≤ prompt_count;
//                         caps from cfg.max_prompts_per_frame and the
//                         confidence floor are applied before decoding)
struct PhysicalInstanceSegmenterOutput {
    PhysicalImageOperatorState     state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                    last_error_reason;
    uint64_t                       inference_count  = 0;
    uint64_t                       last_frame_counter = 0;
    double                         last_inference_ms = 0.0;   // total: encoder + all decoder calls
    double                         last_encoder_ms   = 0.0;
    double                         last_decoder_total_ms = 0.0;
    int32_t                        prompt_count     = 0;
    PhysicalCacheStatus            cache_status{};
    PhysicalInstanceSegmentation   segmentation;
};

struct PhysicalImageClassifierOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    PhysicalCacheStatus         cache_status{};
    std::vector<PhysicalImageClassification> top_k;
};

struct PhysicalPoseKeypointEstimatorOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    PhysicalCacheStatus         cache_status{};
    std::vector<PhysicalPoseInstance> instances;
};

struct PhysicalSceneTextReaderOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    bool                        recogniser_configured = false; // false → only detection ran
    PhysicalCacheStatus         cache_status{};
    std::vector<PhysicalSceneTextLine> lines;
};

struct PhysicalFacialExpressionDetectorOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    bool                        classifier_configured = false; // false → only face detection ran
    PhysicalCacheStatus         cache_status{};
    std::vector<PhysicalFacialExpression> faces;
};

// PhysicalEntityTracker has no ONNX model — its "model" is its parameter
// set. State semantics:
//   NoModelConfigured — never configured AND never default-initialised
//                        (only happens between Reset and the next Configure).
//   ModelLoaded       — parameter set is in effect; ready to consume detections.
//   ModelLoadFailed   — last Configure rejected its config (e.g. bad alpha).
//   InferenceFailed   — last RouteDetectionsTo... threw.
struct PhysicalEntityTrackerOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;     // total successful Route* calls
    uint64_t                    last_frame_counter = 0;
    double                      last_route_ms     = 0.0;

    // Snapshot of every track that is alive AT THE END of this frame
    // (Tentative + Confirmed + Coasting). Tracks culled this frame are not
    // included.
    std::vector<PhysicalEntityTrack> tracks;

    // Cumulative counters since last Reset. Useful for sanity-checking the
    // tracker's behaviour over a session.
    uint64_t                    total_tracks_spawned   = 0;
    uint64_t                    total_tracks_confirmed = 0;
    uint64_t                    total_tracks_culled    = 0;
};

// 9. Class policy — one ranked, deduplicated row per CANONICAL class. The
//    policy mutates the per-operator results above so the labels match the
//    canonical names; this struct is the "what does the model actually see"
//    summary that downstream consumers (model context matrix builder, UI)
//    can iterate over without re-walking every operator.
//
//    Numerical-precision contract:
//      * priority_rank is taken verbatim from the matched
//        PhysicalClassPriorityRule (or PhysicalClassPolicyConfig::
//        default_priority_rank when no rule matched). Lower = higher
//        priority. Never reinterpreted.
//      * max_confidence is the maximum over EVERY signal that contributed
//        to this class this frame (detection.confidence,
//        track.smoothed_confidence, instance_mask.mask_confidence,
//        classification.score). Always in [0, 1].
//      * Counts are exact post-merge / post-drop counts — they MUST equal
//        what a consumer would get by filtering the per-operator outputs
//        on canonical_label.
struct PhysicalClassPolicyClassSummary {
    std::string  canonical_label;
    int32_t      priority_rank        = 0;
    uint32_t     detection_count      = 0;
    uint32_t     track_count          = 0;
    uint32_t     instance_mask_count  = 0;
    uint32_t     classification_count = 0;
    float        max_confidence       = 0.0f;
};

// PhysicalClassPolicy has no ONNX model — its "model" is its rule set.
// State semantics mirror PhysicalEntityTracker:
//   NoModelConfigured — never configured AND never default-initialised
//                       (only between Reset and the next Configure).
//   ModelLoaded       — rule set is in effect; ready to apply.
//   ModelLoadFailed   — last Configure rejected its config (e.g. an alias
//                       maps to two different canonicals).
//   InferenceFailed   — last ApplyPhysicalClassPolicyToPerceptionResults
//                       threw.
struct PhysicalClassPolicyOutput {
    PhysicalImageOperatorState  state               = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count     = 0;   // total successful Apply* calls
    uint64_t                    last_frame_counter  = 0;
    double                      last_apply_ms       = 0.0;

    // Mutation counters — how many items were touched THIS frame. A
    // pass-through policy (empty rule set) leaves all eight at zero.
    uint32_t                    detections_relabeled       = 0;
    uint32_t                    tracks_relabeled           = 0;
    uint32_t                    instance_masks_relabeled   = 0;
    uint32_t                    classifications_relabeled  = 0;
    uint32_t                    detections_dropped         = 0;
    uint32_t                    tracks_dropped             = 0;
    uint32_t                    instance_masks_dropped     = 0;
    uint32_t                    classifications_dropped    = 0;

    // Ranked summary, sorted ascending by priority_rank, ties broken by
    // canonical_label lexicographic order so the UI is deterministic.
    std::vector<PhysicalClassPolicyClassSummary> ranked_classes;
};

// Stage-2 loop-level timing and cache telemetry for one published primitive
// result. Operator envelopes already carry their inference timings; this
// struct captures integration overhead and cache decisions around them.
struct PhysicalPerceptionPrimitiveTelemetry {
    double   tick_total_ms          = 0.0;
    double   frame_bus_pull_ms      = 0.0;
    double   publish_bus_ms         = 0.0;
    double   operator_wall_ms       = 0.0;
    double   object_detector_wall_ms = 0.0;
    double   semantic_segmenter_wall_ms = 0.0;
    double   image_classifier_wall_ms = 0.0;
    double   pose_estimator_wall_ms = 0.0;
    double   scene_text_reader_wall_ms = 0.0;
    double   facial_expression_wall_ms = 0.0;
    double   entity_tracker_wall_ms = 0.0;
    double   instance_segmenter_wall_ms = 0.0;
    double   class_policy_wall_ms   = 0.0;
    uint32_t cache_hit_count        = 0;
    uint32_t fresh_inference_count  = 0;
    uint32_t no_signal_forced_count = 0;
};

// Aggregate snapshot — one frame's worth of results from all operators
// PLUS the class-policy summary that downstream model-context builders
// will read first.
// Carries the source frame counter so the UI can verify it is rendering a
// coherent set (same input frame for every operator).
struct PhysicalPerceptionPrimitiveResults {
    uint64_t                              source_frame_counter = 0;
    // Pins the exact immutable frame consumed by Stage 2. Asynchronous
    // downstream stages must use this view instead of pulling the newer
    // latest-frame slot and accidentally fusing different frame identities.
    PhysicalFrameBus::FrameView            source_frame;
    int                                   model_image_width    = 0;
    int                                   model_image_height   = 0;
    int                                   raw_image_width      = 0;
    int                                   raw_image_height     = 0;
    PhysicalSignalRawToModelTransform     raw_to_model{};
    PhysicalObjectDetectorOutput          object_detector;
    PhysicalSemanticSegmenterOutput       semantic_segmenter;
    PhysicalInstanceSegmenterOutput       instance_segmenter;
    PhysicalImageClassifierOutput         image_classifier;
    PhysicalPoseKeypointEstimatorOutput   pose_estimator;
    PhysicalSceneTextReaderOutput         scene_text_reader;
    PhysicalFacialExpressionDetectorOutput facial_expression_detector;
    PhysicalEntityTrackerOutput            entity_tracker;
    PhysicalClassPolicyOutput              class_policy;
    PhysicalPerceptionPrimitiveTelemetry    telemetry{};
};

}}} // namespace GRIM::Perception::Physical
