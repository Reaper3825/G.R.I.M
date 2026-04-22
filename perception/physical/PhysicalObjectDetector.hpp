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
//  PhysicalObjectDetector — Stage-2 axis-aligned object detection.
//
//  Backend: OpenCV cv::dnn::Net loading an ONNX file.
//
//  Expected output tensor layout (Ultralytics YOLOv8 export — the most
//  widely available cross-platform format):
//
//      shape = [1, 4 + num_classes, num_anchors]
//      For each anchor i:
//        cx = out[0,            0, i]   (model-pixel space)
//        cy = out[0,            1, i]
//        w  = out[0,            2, i]
//        h  = out[0,            3, i]
//        cls_scores = out[0, 4..4+C, i]
//
//  Boxes whose max-class score < confidence_threshold are dropped, then
//  non-maximum suppression with iou_threshold yields the final list.
//
//  Self-owned, internally managed. The only entry points used by the
//  Stage-2 loop are:
//
//      LoadOnnxModelIntoPhysicalObjectDetector(cfg)
//      RouteFrameToPhysicalObjectDetector(model_image, raw_to_model,
//                                         raw_w, raw_h, frame_counter, out)
//      ResetPhysicalObjectDetector()
//
//  Rule 20: load throws on missing file / bad ONNX. Inference catches the
//  exception, stores the reason on `out.last_error_reason`, sets
//  state = InferenceFailed, and returns. UI surfaces the reason verbatim.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalObjectDetectorConfig {
    std::string  onnx_model_path;            // empty = NoModelConfigured (intentional)
    std::string  class_names_path;           // optional newline-separated labels
    int          input_width             = 640;
    int          input_height            = 640;
    float        input_scale             = 1.0f / 255.0f;
    cv::Scalar   input_mean              = cv::Scalar(0.0, 0.0, 0.0);
    bool         swap_rb                 = true;       // BGR → RGB inside blob
    float        confidence_threshold    = 0.25f;
    float        iou_threshold           = 0.45f;
    int          max_detections          = 300;
    int          dnn_backend_id          = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id           = cv::dnn::DNN_TARGET_CPU;
};

class PhysicalObjectDetector {
public:
    PhysicalObjectDetector();
    ~PhysicalObjectDetector();

    // Load (or reload) the ONNX weights using `cfg`. Throws on any failure.
    // Empty `cfg.onnx_model_path` is NOT a failure — it explicitly resets
    // the operator to NoModelConfigured (silent fallbacks are forbidden;
    // this is a request to be silent).
    void LoadOnnxModelIntoPhysicalObjectDetector(const PhysicalObjectDetectorConfig& cfg);

    // Run inference on one frame. Never throws — failures land in `out`.
    void RouteFrameToPhysicalObjectDetector(const cv::Mat& model_image,
                                            const PhysicalSignalRawToModelTransform& raw_to_model,
                                            int raw_image_width,
                                            int raw_image_height,
                                            uint64_t source_frame_counter,
                                            PhysicalObjectDetectorOutput& out);

    void                       ResetPhysicalObjectDetector();
    PhysicalImageOperatorState GetPhysicalObjectDetectorState() const;
    std::string                GetPhysicalObjectDetectorLastError() const;
    bool                       IsPhysicalObjectDetectorReady() const;

private:
    mutable std::mutex                 mutex_;
    PhysicalObjectDetectorConfig       cfg_{};
    std::unique_ptr<cv::dnn::Net>      net_;
    std::vector<std::string>           class_labels_;
    PhysicalImageOperatorState         state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                        last_error_reason_;
    uint64_t                           inference_count_ = 0;
};

}}} // namespace
