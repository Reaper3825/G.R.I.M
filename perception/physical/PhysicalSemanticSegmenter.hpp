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
//  PhysicalSemanticSegmenter — per-pixel class labelling at MODEL resolution.
//
//  Backend: cv::dnn::Net loading an ONNX file.
//
//  Expected output: [1, num_classes, H, W] — channel-major logits. We take
//  the per-pixel argmax to produce a CV_32SC1 label image. The label image
//  is then resized (nearest-neighbour, to preserve label identity) back to
//  the model_image dimensions.
//
//  Self-owned, internally managed. Default config (empty path) leaves the
//  operator in NoModelConfigured.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalSemanticSegmenterConfig {
    std::string  onnx_model_path;
    std::string  class_names_path;
    int          input_width    = 512;
    int          input_height   = 512;
    float        input_scale    = 1.0f / 255.0f;
    cv::Scalar   input_mean     = cv::Scalar(0.0, 0.0, 0.0);
    bool         swap_rb        = true;
    int          dnn_backend_id = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id  = cv::dnn::DNN_TARGET_CPU;
};

class PhysicalSemanticSegmenter {
public:
    PhysicalSemanticSegmenter();
    ~PhysicalSemanticSegmenter();

    void LoadOnnxModelIntoPhysicalSemanticSegmenter(const PhysicalSemanticSegmenterConfig& cfg);

    void RouteFrameToPhysicalSemanticSegmenter(const cv::Mat& model_image,
                                               uint64_t source_frame_counter,
                                               PhysicalSemanticSegmenterOutput& out);

    void                       ResetPhysicalSemanticSegmenter();
    PhysicalImageOperatorState GetPhysicalSemanticSegmenterState() const;
    std::string                GetPhysicalSemanticSegmenterLastError() const;

private:
    mutable std::mutex                 mutex_;
    PhysicalSemanticSegmenterConfig    cfg_{};
    std::unique_ptr<cv::dnn::Net>      net_;
    std::vector<std::string>           class_labels_;
    PhysicalImageOperatorState         state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                        last_error_reason_;
    uint64_t                           inference_count_ = 0;
};

}}} // namespace
