#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalFrameConditioner.hpp"   // PhysicalSignalRawToModelTransform

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
    std::vector<PhysicalObjectDetection> detections;
};

struct PhysicalSemanticSegmenterOutput {
    PhysicalImageOperatorState   state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                  last_error_reason;
    uint64_t                     inference_count  = 0;
    uint64_t                     last_frame_counter = 0;
    double                       last_inference_ms = 0.0;
    PhysicalSemanticSegmentation segmentation;
};

struct PhysicalImageClassifierOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    std::vector<PhysicalImageClassification> top_k;
};

struct PhysicalPoseKeypointEstimatorOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    std::vector<PhysicalPoseInstance> instances;
};

struct PhysicalSceneTextReaderOutput {
    PhysicalImageOperatorState  state            = PhysicalImageOperatorState::NoModelConfigured;
    std::string                 last_error_reason;
    uint64_t                    inference_count  = 0;
    uint64_t                    last_frame_counter = 0;
    double                      last_inference_ms = 0.0;
    bool                        recogniser_configured = false; // false → only detection ran
    std::vector<PhysicalSceneTextLine> lines;
};

// Aggregate snapshot — one frame's worth of results from all five operators.
// Carries the source frame counter so the UI can verify it is rendering a
// coherent set (same input frame for every operator).
struct PhysicalPerceptionPrimitiveResults {
    uint64_t                              source_frame_counter = 0;
    int                                   model_image_width    = 0;
    int                                   model_image_height   = 0;
    int                                   raw_image_width      = 0;
    int                                   raw_image_height     = 0;
    PhysicalSignalRawToModelTransform     raw_to_model{};
    PhysicalObjectDetectorOutput          object_detector;
    PhysicalSemanticSegmenterOutput       semantic_segmenter;
    PhysicalImageClassifierOutput         image_classifier;
    PhysicalPoseKeypointEstimatorOutput   pose_estimator;
    PhysicalSceneTextReaderOutput         scene_text_reader;
};

}}} // namespace GRIM::Perception::Physical
