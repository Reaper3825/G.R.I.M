#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>     // cv::FaceDetectorYN

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalFacialExpressionDetector — face detection (YuNet) + per-face
//  facial expression classification (FER+).
//
//  Two ONNX files:
//    detector_onnx_path   → YuNet (cv::FaceDetectorYN) — outputs N face boxes
//                            with score, plus 5 landmarks (unused here).
//    classifier_onnx_path → emotion classifier (e.g. emotion-ferplus-8.onnx).
//                            Per-face crop is letterbox-resized to
//                            classifier_input_(width|height) and converted
//                            to grayscale when classifier_input_grayscale=true
//                            (default — matches FER+).
//
//  Output: a list of (face bbox + top expression label/score + full softmax
//  distribution) for every face that survives the detector's score cutoff.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalFacialExpressionDetectorConfig {
    // Detector (YuNet)
    std::string  detector_onnx_path;
    int          detector_input_width   = 320;
    int          detector_input_height  = 320;
    float        detector_score_threshold = 0.60f;
    float        detector_nms_threshold   = 0.30f;
    int          detector_top_k           = 50;

    // Classifier (FER+ style)
    std::string  classifier_onnx_path;
    std::string  classifier_class_names_path;   // newline-separated, K labels
    int          classifier_input_width   = 64;
    int          classifier_input_height  = 64;
    float        classifier_input_scale   = 1.0f;       // FER+ takes raw [0,255]
    cv::Scalar   classifier_input_mean    = cv::Scalar(0.0, 0.0, 0.0);
    bool         classifier_input_grayscale = true;     // FER+ is 1-channel
    bool         classifier_swap_rb       = false;
    // Pad the detected face bbox before cropping so the classifier sees
    // forehead + chin context. 0.0 = exact bbox, 0.20 = 20% margin.
    float        face_crop_padding_ratio  = 0.10f;

    int          dnn_backend_id = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id  = cv::dnn::DNN_TARGET_CPU;

    // Latency knobs. Defaults preserve every-frame inference.
    PhysicalOperatorCadenceConfig cadence{};
};

class PhysicalFacialExpressionDetector {
public:
    PhysicalFacialExpressionDetector();
    ~PhysicalFacialExpressionDetector();

    void LoadOnnxModelsIntoPhysicalFacialExpressionDetector(
        const PhysicalFacialExpressionDetectorConfig& cfg);

    void RouteFrameToPhysicalFacialExpressionDetector(
        const cv::Mat& model_image,
        const PhysicalSignalRawToModelTransform& raw_to_model,
        int raw_image_width,
        int raw_image_height,
        uint64_t source_frame_counter,
        PhysicalFacialExpressionDetectorOutput& out);

    void                       ResetPhysicalFacialExpressionDetector();
    PhysicalImageOperatorState GetPhysicalFacialExpressionDetectorState() const;
    std::string                GetPhysicalFacialExpressionDetectorLastError() const;

private:
    mutable std::mutex                              mutex_;
    PhysicalFacialExpressionDetectorConfig          cfg_{};
    cv::Ptr<cv::FaceDetectorYN>                     detector_;
    std::unique_ptr<cv::dnn::Net>                   classifier_;
    std::vector<std::string>                        expression_labels_;
    PhysicalImageOperatorState                      state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                                     last_error_reason_;
    uint64_t                                        inference_count_ = 0;
    bool                                            classifier_configured_ = false;
};

}}} // namespace
