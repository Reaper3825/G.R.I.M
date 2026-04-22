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
//  PhysicalImageClassifier — whole-frame top-K classification.
//
//  Expected output: [1, num_classes] logits (raw scores). We softmax inside
//  this operator to give consumers a normalised probability in [0, 1].
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalImageClassifierConfig {
    std::string  onnx_model_path;
    std::string  class_names_path;
    int          input_width    = 224;
    int          input_height   = 224;
    float        input_scale    = 1.0f / 255.0f;
    cv::Scalar   input_mean     = cv::Scalar(0.485, 0.456, 0.406);     // ImageNet mean (×255 if scale==1)
    bool         swap_rb        = true;
    int          top_k          = 5;
    int          dnn_backend_id = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id  = cv::dnn::DNN_TARGET_CPU;
};

class PhysicalImageClassifier {
public:
    PhysicalImageClassifier();
    ~PhysicalImageClassifier();

    void LoadOnnxModelIntoPhysicalImageClassifier(const PhysicalImageClassifierConfig& cfg);

    void RouteFrameToPhysicalImageClassifier(const cv::Mat& model_image,
                                             uint64_t source_frame_counter,
                                             PhysicalImageClassifierOutput& out);

    void                       ResetPhysicalImageClassifier();
    PhysicalImageOperatorState GetPhysicalImageClassifierState() const;
    std::string                GetPhysicalImageClassifierLastError() const;

private:
    mutable std::mutex                 mutex_;
    PhysicalImageClassifierConfig      cfg_{};
    std::unique_ptr<cv::dnn::Net>      net_;
    std::vector<std::string>           class_labels_;
    PhysicalImageOperatorState         state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                        last_error_reason_;
    uint64_t                           inference_count_ = 0;
};

}}} // namespace
