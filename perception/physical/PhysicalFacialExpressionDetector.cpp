#include "PhysicalFacialExpressionDetector.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

std::vector<std::string> LoadExpressionLabels(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "PhysicalFacialExpressionDetector: failed to open classifier_class_names_path='"
            + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    if (out.empty()) {
        throw std::runtime_error(
            "PhysicalFacialExpressionDetector: classifier_class_names_path='" + path + "' is empty");
    }
    return out;
}

cv::Point2f BackProjectPoint(const cv::Point2f& m,
                             const PhysicalSignalRawToModelTransform& t)
{
    if (t.scale_x == 0.0 || t.scale_y == 0.0) return cv::Point2f(0.0f, 0.0f);
    return cv::Point2f(
        static_cast<float>((m.x - t.offset_x) / t.scale_x),
        static_cast<float>((m.y - t.offset_y) / t.scale_y));
}

cv::Rect2f BackProjectRectClipped(const cv::Rect2f& mb,
                                  const PhysicalSignalRawToModelTransform& t,
                                  int raw_w, int raw_h)
{
    const cv::Point2f tl = BackProjectPoint(cv::Point2f(mb.x, mb.y), t);
    const cv::Point2f br = BackProjectPoint(cv::Point2f(mb.x + mb.width,
                                                       mb.y + mb.height), t);
    float rx = tl.x, ry = tl.y, rw = br.x - tl.x, rh = br.y - tl.y;
    if (rx < 0) { rw += rx; rx = 0; }
    if (ry < 0) { rh += ry; ry = 0; }
    if (rx + rw > static_cast<float>(raw_w))  rw = static_cast<float>(raw_w)  - rx;
    if (ry + rh > static_cast<float>(raw_h)) rh = static_cast<float>(raw_h) - ry;
    if (rw < 0) rw = 0;
    if (rh < 0) rh = 0;
    return cv::Rect2f(rx, ry, rw, rh);
}

// Numerically stable softmax in-place over a 1xK row of floats.
void SoftmaxInPlace(std::vector<float>& v) {
    if (v.empty()) return;
    const float m = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (auto& x : v) { x = std::exp(x - m); sum += x; }
    if (sum <= 0.0) return;
    const float inv = static_cast<float>(1.0 / sum);
    for (auto& x : v) x *= inv;
}

} // anonymous

PhysicalFacialExpressionDetector::PhysicalFacialExpressionDetector()  = default;
PhysicalFacialExpressionDetector::~PhysicalFacialExpressionDetector() = default;

void PhysicalFacialExpressionDetector::LoadOnnxModelsIntoPhysicalFacialExpressionDetector(
    const PhysicalFacialExpressionDetectorConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_                   = cfg;
    detector_.reset();
    classifier_.reset();
    expression_labels_.clear();
    last_error_reason_.clear();
    inference_count_       = 0;
    classifier_configured_ = false;

    if (cfg.detector_onnx_path.empty()) {
        if (!cfg.classifier_onnx_path.empty()) {
            last_error_reason_ = "PhysicalFacialExpressionDetector: classifier_onnx_path "
                                 "is set but detector_onnx_path is empty \u2014 a face "
                                 "detector is required to produce face crops";
            state_ = PhysicalImageOperatorState::ModelLoadFailed;
            LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
            throw std::runtime_error(last_error_reason_);
        }
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        return;
    }

    try {
        detector_ = cv::FaceDetectorYN::create(
            cfg.detector_onnx_path,
            /*config=*/std::string(),
            cv::Size(cfg.detector_input_width, cfg.detector_input_height),
            cfg.detector_score_threshold,
            cfg.detector_nms_threshold,
            cfg.detector_top_k,
            cfg.dnn_backend_id,
            cfg.dnn_target_id);
        if (!detector_) {
            throw std::runtime_error("cv::FaceDetectorYN::create returned null");
        }

        if (!cfg.classifier_onnx_path.empty()) {
            if (cfg.classifier_class_names_path.empty()) {
                throw std::runtime_error(
                    "classifier_onnx_path is set but classifier_class_names_path is empty");
            }
            expression_labels_ = LoadExpressionLabels(cfg.classifier_class_names_path);
            auto net = std::make_unique<cv::dnn::Net>(
                cv::dnn::readNetFromONNX(cfg.classifier_onnx_path));
            net->setPreferableBackend(static_cast<cv::dnn::Backend>(cfg.dnn_backend_id));
            net->setPreferableTarget(static_cast<cv::dnn::Target>(cfg.dnn_target_id));
            classifier_            = std::move(net);
            classifier_configured_ = true;
        }

        state_ = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalFacialExpressionDetector: ModelLoaded detector='")
                  + cfg.detector_onnx_path + "' classifier='"
                  + (classifier_configured_ ? cfg.classifier_onnx_path : std::string("(none)"))
                  + "' classes=" + std::to_string(expression_labels_.size()));
    } catch (const std::exception& e) {
        detector_.reset();
        classifier_.reset();
        expression_labels_.clear();
        classifier_configured_ = false;
        last_error_reason_     =
            std::string("LoadOnnxModelsIntoPhysicalFacialExpressionDetector failed: ") + e.what();
        state_                 = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalFacialExpressionDetector::RouteFrameToPhysicalFacialExpressionDetector(
    const cv::Mat& model_image,
    const PhysicalSignalRawToModelTransform& raw_to_model,
    int raw_image_width,
    int raw_image_height,
    uint64_t source_frame_counter,
    PhysicalFacialExpressionDetectorOutput& out)
{
    out = PhysicalFacialExpressionDetectorOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state                  = state_;
    out.last_error_reason      = last_error_reason_;
    out.classifier_configured  = classifier_configured_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!detector_) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason =
            "PhysicalFacialExpressionDetector: detector_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason =
            "PhysicalFacialExpressionDetector: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        // YuNet expects the input size set at create() time. If the conditioned
        // model image differs, retarget the detector for this frame.
        detector_->setInputSize(cv::Size(model_image.cols, model_image.rows));

        cv::Mat faces;     // [N, 15] float32: x, y, w, h, lm0_x..lm4_y, score
        detector_->detect(model_image, faces);

        const int N = (faces.empty() ? 0 : faces.rows);
        out.faces.reserve(static_cast<size_t>(N));

        const int K = static_cast<int>(expression_labels_.size());

        for (int i = 0; i < N; ++i) {
            const float* row = faces.ptr<float>(i);
            cv::Rect2f bb(row[0], row[1], row[2], row[3]);
            const float det_score = row[14];

            // Pad the bbox by face_crop_padding_ratio for context.
            if (cfg_.face_crop_padding_ratio > 0.0f) {
                const float px = bb.width  * cfg_.face_crop_padding_ratio;
                const float py = bb.height * cfg_.face_crop_padding_ratio;
                bb.x      -= px;
                bb.y      -= py;
                bb.width  += 2.0f * px;
                bb.height += 2.0f * py;
            }
            // Clip to image bounds.
            cv::Rect bb_int(
                static_cast<int>(std::floor(bb.x)),
                static_cast<int>(std::floor(bb.y)),
                static_cast<int>(std::ceil(bb.width)),
                static_cast<int>(std::ceil(bb.height)));
            bb_int &= cv::Rect(0, 0, model_image.cols, model_image.rows);
            if (bb_int.width < 2 || bb_int.height < 2) continue;

            PhysicalFacialExpression face;
            face.model_bbox = cv::Rect2f(static_cast<float>(bb_int.x),
                                         static_cast<float>(bb_int.y),
                                         static_cast<float>(bb_int.width),
                                         static_cast<float>(bb_int.height));
            face.raw_bbox  = BackProjectRectClipped(face.model_bbox, raw_to_model,
                                                    raw_image_width, raw_image_height);
            face.detection_confidence = det_score;

            if (classifier_configured_ && classifier_ && K > 0) {
                cv::Mat roi = model_image(bb_int);
                cv::Mat prepped;
                if (cfg_.classifier_input_grayscale) {
                    cv::cvtColor(roi, prepped, cv::COLOR_BGR2GRAY);
                } else {
                    prepped = roi;
                }
                cv::Mat blob = cv::dnn::blobFromImage(
                    prepped,
                    cfg_.classifier_input_scale,
                    cv::Size(cfg_.classifier_input_width, cfg_.classifier_input_height),
                    cfg_.classifier_input_mean,
                    cfg_.classifier_swap_rb,
                    /*crop=*/false);
                classifier_->setInput(blob);
                cv::Mat logits = classifier_->forward();   // [1, K] (sometimes [1,K,1,1])

                // Flatten to a [1, K] view.
                cv::Mat flat = logits.reshape(1, 1);
                if (flat.cols < K) {
                    throw std::runtime_error(
                        "facial expression classifier returned " + std::to_string(flat.cols)
                        + " logits but labels file declares " + std::to_string(K) + " classes");
                }
                std::vector<float> scores(static_cast<size_t>(K));
                for (int k = 0; k < K; ++k) {
                    scores[static_cast<size_t>(k)] = flat.at<float>(0, k);
                }
                SoftmaxInPlace(scores);
                int   best_id  = 0;
                float best_val = scores[0];
                for (int k = 1; k < K; ++k) {
                    if (scores[static_cast<size_t>(k)] > best_val) {
                        best_val = scores[static_cast<size_t>(k)];
                        best_id  = k;
                    }
                }
                face.expression_id    = best_id;
                face.expression_score = best_val;
                face.expression_label = expression_labels_[static_cast<size_t>(best_id)];
                face.all_class_scores = std::move(scores);
            }

            out.faces.push_back(std::move(face));
        }

        ++inference_count_;
        out.inference_count   = inference_count_;
        out.last_inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        out.state             = PhysicalImageOperatorState::ModelLoaded;
        out.last_error_reason.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        last_error_reason_ =
            std::string("RouteFrameToPhysicalFacialExpressionDetector failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        out.faces.clear();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalFacialExpressionDetector::ResetPhysicalFacialExpressionDetector() {
    std::lock_guard<std::mutex> lk(mutex_);
    detector_.reset();
    classifier_.reset();
    expression_labels_.clear();
    cfg_ = PhysicalFacialExpressionDetectorConfig{};
    state_ = PhysicalImageOperatorState::NoModelConfigured;
    last_error_reason_.clear();
    inference_count_       = 0;
    classifier_configured_ = false;
}

PhysicalImageOperatorState PhysicalFacialExpressionDetector::GetPhysicalFacialExpressionDetectorState() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_;
}

std::string PhysicalFacialExpressionDetector::GetPhysicalFacialExpressionDetectorLastError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_reason_;
}

}}} // namespace
