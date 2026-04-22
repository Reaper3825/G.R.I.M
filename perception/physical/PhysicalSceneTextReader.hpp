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
//  PhysicalSceneTextReader — text detection (DBNet) + optional recognition
//  (CRNN). Two ONNX models, both loaded if their paths are non-empty.
//
//  Detection only: produces text quadrilaterals (model + raw space).
//  Detection + recognition: each quad is also paired with a UTF-8 string.
//
//  Recognition charset is loaded from `recogniser_charset_path` (one
//  character per line, blank token implicit at index 0). This matches the
//  format expected by cv::dnn::TextRecognitionModel.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalSceneTextReaderConfig {
    // Detector (DB-style)
    std::string  detector_onnx_path;
    int          detector_input_width   = 736;
    int          detector_input_height  = 736;
    float        detector_input_scale   = 1.0f / 255.0f;
    cv::Scalar   detector_input_mean    = cv::Scalar(122.67891434, 116.66876762, 104.00698793);
    float        detector_binary_thresh = 0.30f;
    float        detector_polygon_thresh = 0.60f;
    int          detector_max_candidates = 200;
    double       detector_unclip_ratio   = 2.0;

    // Recogniser (CRNN-style) — optional
    std::string  recogniser_onnx_path;
    std::string  recogniser_charset_path;
    int          recogniser_input_width  = 100;
    int          recogniser_input_height = 32;
    float        recogniser_input_scale  = 1.0f / 127.5f;
    cv::Scalar   recogniser_input_mean   = cv::Scalar(127.5, 127.5, 127.5);

    int          dnn_backend_id          = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id           = cv::dnn::DNN_TARGET_CPU;
};

class PhysicalSceneTextReader {
public:
    PhysicalSceneTextReader();
    ~PhysicalSceneTextReader();

    void LoadOnnxModelsIntoPhysicalSceneTextReader(const PhysicalSceneTextReaderConfig& cfg);

    void RouteFrameToPhysicalSceneTextReader(const cv::Mat& model_image,
                                             const PhysicalSignalRawToModelTransform& raw_to_model,
                                             int raw_image_width,
                                             int raw_image_height,
                                             uint64_t source_frame_counter,
                                             PhysicalSceneTextReaderOutput& out);

    void                       ResetPhysicalSceneTextReader();
    PhysicalImageOperatorState GetPhysicalSceneTextReaderState() const;
    std::string                GetPhysicalSceneTextReaderLastError() const;

private:
    mutable std::mutex                                 mutex_;
    PhysicalSceneTextReaderConfig                      cfg_{};
    std::unique_ptr<cv::dnn::TextDetectionModel_DB>    detector_;
    std::unique_ptr<cv::dnn::TextRecognitionModel>     recogniser_;
    PhysicalImageOperatorState                         state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                                        last_error_reason_;
    uint64_t                                           inference_count_ = 0;
    bool                                               recogniser_configured_ = false;
};

}}} // namespace
