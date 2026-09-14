#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <memory>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// Extracts recognition embeddings from faces already detected by the shared
// YuNet operator. This operator never detects faces and therefore cannot drift
// from the geometry used by expression classification or world-state fusion.
struct PhysicalFaceRecognizerConfig {
    std::string onnx_model_path;
    std::string model_id = "sface";
    int         dnn_backend_id = cv::dnn::DNN_BACKEND_OPENCV;
    int         dnn_target_id  = cv::dnn::DNN_TARGET_CPU;
    int         minimum_face_pixels = 48;
    float       minimum_detection_confidence = 0.70f;
    PhysicalOperatorCadenceConfig cadence{};
};

class PhysicalFaceRecognizer {
public:
    void LoadOnnxModelIntoPhysicalFaceRecognizer(
        const PhysicalFaceRecognizerConfig& cfg);

    void RouteFrameAndFacesToPhysicalFaceRecognizer(
        const cv::Mat& model_image,
        const std::vector<PhysicalFacialExpression>& detected_faces,
        uint64_t source_frame_counter,
        PhysicalFaceRecognizerOutput& out);

    void ResetPhysicalFaceRecognizer();
    PhysicalImageOperatorState GetPhysicalFaceRecognizerState() const;
    std::string GetPhysicalFaceRecognizerLastError() const;

private:
    mutable std::mutex mutex_;
    PhysicalFaceRecognizerConfig cfg_{};
    cv::Ptr<cv::FaceRecognizerSF> recognizer_;
    PhysicalImageOperatorState state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string last_error_reason_;
    uint64_t inference_count_ = 0;
};

}}} // namespace GRIM::Perception::Physical
