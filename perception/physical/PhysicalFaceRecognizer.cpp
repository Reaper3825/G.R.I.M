#include "PhysicalFaceRecognizer.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

float Clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float FaceQuality(const PhysicalFacialExpression& face,
                  const cv::Size& image_size,
                  int minimum_face_pixels) {
    const float shortest = std::min(face.model_bbox.width, face.model_bbox.height);
    const float size_score = Clamp01(shortest / static_cast<float>(
        std::max(1, minimum_face_pixels * 2)));
    const float frame_area = static_cast<float>(std::max(1, image_size.area()));
    const float area_score = Clamp01(
        face.model_bbox.area() / (frame_area * 0.04f));
    return Clamp01(0.55f * face.detection_confidence
                 + 0.30f * size_score
                 + 0.15f * area_score);
}

cv::Mat MakeYuNetFaceRow(const PhysicalFacialExpression& face) {
    cv::Mat row(1, 15, CV_32F, cv::Scalar(0));
    row.at<float>(0, 0) = face.model_bbox.x;
    row.at<float>(0, 1) = face.model_bbox.y;
    row.at<float>(0, 2) = face.model_bbox.width;
    row.at<float>(0, 3) = face.model_bbox.height;
    for (size_t i = 0; i < face.model_landmarks.size(); ++i) {
        row.at<float>(0, 4 + static_cast<int>(i) * 2) = face.model_landmarks[i].x;
        row.at<float>(0, 5 + static_cast<int>(i) * 2) = face.model_landmarks[i].y;
    }
    row.at<float>(0, 14) = face.detection_confidence;
    return row;
}

std::vector<float> NormalizeFeature(const cv::Mat& feature) {
    cv::Mat flat = feature.reshape(1, 1);
    if (flat.empty() || flat.depth() != CV_32F) {
        throw std::runtime_error("face recognizer returned an empty or non-float feature");
    }
    double squared_norm = 0.0;
    for (int i = 0; i < flat.cols; ++i) {
        const float value = flat.at<float>(0, i);
        squared_norm += static_cast<double>(value) * value;
    }
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
        throw std::runtime_error("face recognizer returned a non-normalizable feature");
    }
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    std::vector<float> out(static_cast<size_t>(flat.cols));
    for (int i = 0; i < flat.cols; ++i) {
        out[static_cast<size_t>(i)] = flat.at<float>(0, i) * inverse_norm;
    }
    return out;
}

} // anonymous namespace

void PhysicalFaceRecognizer::LoadOnnxModelIntoPhysicalFaceRecognizer(
    const PhysicalFaceRecognizerConfig& cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_ = cfg;
    recognizer_.reset();
    inference_count_ = 0;
    last_error_reason_.clear();

    if (cfg.onnx_model_path.empty()) {
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        return;
    }
    if (cfg.model_id.empty() || cfg.minimum_face_pixels < 16 ||
        !(cfg.minimum_detection_confidence >= 0.0f &&
          cfg.minimum_detection_confidence <= 1.0f)) {
        state_ = PhysicalImageOperatorState::ModelLoadFailed;
        throw std::invalid_argument("PhysicalFaceRecognizerConfig is invalid");
    }
    try {
        recognizer_ = cv::FaceRecognizerSF::create(
            cfg.onnx_model_path, std::string{}, cfg.dnn_backend_id, cfg.dnn_target_id);
        if (!recognizer_) {
            throw std::runtime_error("cv::FaceRecognizerSF::create returned null");
        }
        state_ = PhysicalImageOperatorState::ModelLoaded;
    } catch (const std::exception& e) {
        recognizer_.reset();
        state_ = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = std::string("PhysicalFaceRecognizer load failed: ") + e.what();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalFaceRecognizer::RouteFrameAndFacesToPhysicalFaceRecognizer(
    const cv::Mat& model_image,
    const std::vector<PhysicalFacialExpression>& detected_faces,
    uint64_t source_frame_counter,
    PhysicalFaceRecognizerOutput& out) {
    out = {};
    out.last_frame_counter = source_frame_counter;
    std::lock_guard<std::mutex> lock(mutex_);
    out.state = state_;
    out.last_error_reason = last_error_reason_;
    out.inference_count = inference_count_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;

    const auto started = std::chrono::steady_clock::now();
    try {
        if (!recognizer_ || model_image.empty() || model_image.type() != CV_8UC3) {
            throw std::runtime_error("recognizer is unavailable or model image is not CV_8UC3");
        }
        for (const auto& face : detected_faces) {
            if (face.detection_confidence < cfg_.minimum_detection_confidence ||
                std::min(face.model_bbox.width, face.model_bbox.height) <
                    static_cast<float>(cfg_.minimum_face_pixels)) {
                continue;
            }
            cv::Mat aligned;
            recognizer_->alignCrop(model_image, MakeYuNetFaceRow(face), aligned);
            cv::Mat feature;
            recognizer_->feature(aligned, feature);

            PhysicalFaceEmbedding item;
            item.model_bbox = face.model_bbox;
            item.raw_bbox = face.raw_bbox;
            item.model_landmarks = face.model_landmarks;
            item.detection_confidence = face.detection_confidence;
            item.quality_score = FaceQuality(face, model_image.size(), cfg_.minimum_face_pixels);
            item.embedding = NormalizeFeature(feature);
            item.embedding_model_id = cfg_.model_id;
            item.source_frame_counter = source_frame_counter;
            out.faces.push_back(std::move(item));
        }
        ++inference_count_;
        out.inference_count = inference_count_;
        out.state = PhysicalImageOperatorState::ModelLoaded;
        out.last_error_reason.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        state_ = PhysicalImageOperatorState::InferenceFailed;
        last_error_reason_ = std::string("PhysicalFaceRecognizer inference failed: ") + e.what();
        out.state = state_;
        out.last_error_reason = last_error_reason_;
        out.faces.clear();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
    out.last_inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

void PhysicalFaceRecognizer::ResetPhysicalFaceRecognizer() {
    std::lock_guard<std::mutex> lock(mutex_);
    recognizer_.reset();
    cfg_ = {};
    state_ = PhysicalImageOperatorState::NoModelConfigured;
    last_error_reason_.clear();
    inference_count_ = 0;
}

PhysicalImageOperatorState PhysicalFaceRecognizer::GetPhysicalFaceRecognizerState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string PhysicalFaceRecognizer::GetPhysicalFaceRecognizerLastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_reason_;
}

}}} // namespace GRIM::Perception::Physical
