#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalPoseKeypointEstimator — single-person (top-down) heatmap pose.
//
//  Expected output: [1, num_joints, H, W] heatmaps. Per joint, take the
//  argmax pixel and report its (x, y) in MODEL space and the heatmap peak
//  value as confidence. The bounding box of all visible keypoints (those
//  above min_keypoint_confidence) becomes the instance bbox.
//
//  This is the simplest practical decoder; for multi-person models, swap in
//  a model whose ONNX includes its own decode head.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalPoseKeypointEstimatorConfig {
    enum class OutputFormat : uint8_t {
        // Top-down per-joint heatmap models (HRNet/MoveNet style).
        // Output: [1, J, H, W]; argmax per joint plane.
        Heatmap = 0,
        // YOLOv8-pose style anchor list.
        // Output: [1, 4 + 1 + J*3, A]  (cx,cy,w,h, person_score, kp_x*J, kp_y*J, kp_conf*J)
        // Coordinates are in MODEL pixel space.
        YoloAnchor = 1
    };

    std::string  onnx_model_path;
    std::string  joint_names_path;        // newline-separated joint labels
    int          input_width    = 192;
    int          input_height   = 256;
    float        input_scale    = 1.0f / 255.0f;
    cv::Scalar   input_mean     = cv::Scalar(0.0, 0.0, 0.0);
    bool         swap_rb        = true;
    float        min_keypoint_confidence = 0.30f;
    OutputFormat output_format  = OutputFormat::Heatmap;
    // YoloAnchor-only:
    int          num_keypoints  = 17;       // J
    float        person_confidence_threshold = 0.25f;
    float        nms_iou_threshold           = 0.45f;
    int          dnn_backend_id = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id  = cv::dnn::DNN_TARGET_CPU;
};

class PhysicalPoseKeypointEstimator {
public:
    PhysicalPoseKeypointEstimator();
    ~PhysicalPoseKeypointEstimator();

    void LoadOnnxModelIntoPhysicalPoseKeypointEstimator(const PhysicalPoseKeypointEstimatorConfig& cfg);

    void RouteFrameToPhysicalPoseKeypointEstimator(const cv::Mat& model_image,
                                                   const PhysicalSignalRawToModelTransform& raw_to_model,
                                                   int raw_image_width,
                                                   int raw_image_height,
                                                   uint64_t source_frame_counter,
                                                   PhysicalPoseKeypointEstimatorOutput& out);

    void                       ResetPhysicalPoseKeypointEstimator();
    PhysicalImageOperatorState GetPhysicalPoseKeypointEstimatorState() const;
    std::string                GetPhysicalPoseKeypointEstimatorLastError() const;

private:
    mutable std::mutex                       mutex_;
    PhysicalPoseKeypointEstimatorConfig      cfg_{};
    std::unique_ptr<cv::dnn::Net>            net_;
    std::vector<std::string>                 joint_labels_;
    PhysicalImageOperatorState               state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                              last_error_reason_;
    uint64_t                                 inference_count_ = 0;
};

}}} // namespace
