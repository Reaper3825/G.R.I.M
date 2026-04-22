#include "PhysicalSceneTextReader.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

std::vector<std::string> LoadCharset(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "PhysicalSceneTextReader: failed to open recogniser_charset_path='" + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r')) line.pop_back();
        out.push_back(line);
    }
    if (out.empty()) {
        throw std::runtime_error(
            "PhysicalSceneTextReader: recogniser_charset_path='" + path + "' is empty");
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

PhysicalQuad QuadFromModelPoints(const std::vector<cv::Point>& q,
                                 const PhysicalSignalRawToModelTransform& /*ignored*/)
{
    PhysicalQuad out;
    out.p0 = q.size() > 0 ? cv::Point2f(static_cast<float>(q[0].x), static_cast<float>(q[0].y)) : cv::Point2f(0,0);
    out.p1 = q.size() > 1 ? cv::Point2f(static_cast<float>(q[1].x), static_cast<float>(q[1].y)) : out.p0;
    out.p2 = q.size() > 2 ? cv::Point2f(static_cast<float>(q[2].x), static_cast<float>(q[2].y)) : out.p1;
    out.p3 = q.size() > 3 ? cv::Point2f(static_cast<float>(q[3].x), static_cast<float>(q[3].y)) : out.p2;
    return out;
}

PhysicalQuad QuadBackProjected(const PhysicalQuad& m,
                               const PhysicalSignalRawToModelTransform& t)
{
    PhysicalQuad r;
    r.p0 = BackProjectPoint(m.p0, t);
    r.p1 = BackProjectPoint(m.p1, t);
    r.p2 = BackProjectPoint(m.p2, t);
    r.p3 = BackProjectPoint(m.p3, t);
    return r;
}

} // anonymous

PhysicalSceneTextReader::PhysicalSceneTextReader()  = default;
PhysicalSceneTextReader::~PhysicalSceneTextReader() = default;

void PhysicalSceneTextReader::LoadOnnxModelsIntoPhysicalSceneTextReader(
    const PhysicalSceneTextReaderConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_                    = cfg;
    detector_.reset();
    recogniser_.reset();
    last_error_reason_.clear();
    inference_count_        = 0;
    recogniser_configured_  = false;

    if (cfg.detector_onnx_path.empty()) {
        // Recogniser without detector is meaningless — treat both empty
        // as NoModelConfigured. Recogniser path set without detector is a
        // user error and we say so loudly.
        if (!cfg.recogniser_onnx_path.empty()) {
            last_error_reason_ = "PhysicalSceneTextReader: recogniser_onnx_path "
                                 "is set but detector_onnx_path is empty — a "
                                 "detector is required to produce text quads";
            state_ = PhysicalImageOperatorState::ModelLoadFailed;
            LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
            throw std::runtime_error(last_error_reason_);
        }
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        return;
    }

    try {
        auto det = std::make_unique<cv::dnn::TextDetectionModel_DB>(cfg.detector_onnx_path);
        det->setBinaryThreshold(cfg.detector_binary_thresh);
        det->setPolygonThreshold(cfg.detector_polygon_thresh);
        det->setMaxCandidates(cfg.detector_max_candidates);
        det->setUnclipRatio(cfg.detector_unclip_ratio);
        det->setInputParams(cfg.detector_input_scale,
                            cv::Size(cfg.detector_input_width, cfg.detector_input_height),
                            cfg.detector_input_mean,
                            /*swapRB=*/true);
        det->setPreferableBackend(static_cast<cv::dnn::Backend>(cfg.dnn_backend_id));
        det->setPreferableTarget(static_cast<cv::dnn::Target>(cfg.dnn_target_id));
        detector_ = std::move(det);

        if (!cfg.recogniser_onnx_path.empty()) {
            if (cfg.recogniser_charset_path.empty()) {
                throw std::runtime_error(
                    "recogniser_onnx_path is set but recogniser_charset_path is empty");
            }
            auto rec = std::make_unique<cv::dnn::TextRecognitionModel>(cfg.recogniser_onnx_path);
            rec->setDecodeType("CTC-greedy");
            rec->setVocabulary(LoadCharset(cfg.recogniser_charset_path));
            rec->setInputParams(cfg.recogniser_input_scale,
                                cv::Size(cfg.recogniser_input_width, cfg.recogniser_input_height),
                                cfg.recogniser_input_mean,
                                /*swapRB=*/false,
                                /*crop=*/false);
            rec->setPreferableBackend(static_cast<cv::dnn::Backend>(cfg.dnn_backend_id));
            rec->setPreferableTarget(static_cast<cv::dnn::Target>(cfg.dnn_target_id));
            recogniser_            = std::move(rec);
            recogniser_configured_ = true;
        }

        state_ = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalSceneTextReader: ModelLoaded detector='")
                  + cfg.detector_onnx_path + "' recogniser='"
                  + (recogniser_configured_ ? cfg.recogniser_onnx_path : std::string("(none)"))
                  + "'");
    } catch (const std::exception& e) {
        detector_.reset();
        recogniser_.reset();
        recogniser_configured_ = false;
        last_error_reason_     = std::string("LoadOnnxModelsIntoPhysicalSceneTextReader failed: ") + e.what();
        state_                 = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalSceneTextReader::RouteFrameToPhysicalSceneTextReader(
    const cv::Mat& model_image,
    const PhysicalSignalRawToModelTransform& raw_to_model,
    int /*raw_image_width*/,
    int /*raw_image_height*/,
    uint64_t source_frame_counter,
    PhysicalSceneTextReaderOutput& out)
{
    out = PhysicalSceneTextReaderOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state                  = state_;
    out.last_error_reason      = last_error_reason_;
    out.recogniser_configured  = recogniser_configured_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!detector_) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalSceneTextReader: detector_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalSceneTextReader: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        std::vector<std::vector<cv::Point>> quads;
        std::vector<float>                  confidences;
        detector_->detect(model_image, quads, confidences);

        out.lines.reserve(quads.size());
        for (size_t i = 0; i < quads.size(); ++i) {
            PhysicalSceneTextLine ln;
            ln.model_quad = QuadFromModelPoints(quads[i], raw_to_model);
            ln.raw_quad   = QuadBackProjected(ln.model_quad, raw_to_model);
            ln.confidence = (i < confidences.size()) ? confidences[i] : 0.0f;

            if (recogniser_configured_ && recogniser_) {
                // Crop a 4-point ROI from the model image and recognise.
                cv::Mat roi;
                std::vector<cv::Point2f> src{
                    ln.model_quad.p0, ln.model_quad.p1, ln.model_quad.p2, ln.model_quad.p3};
                // Bounding box of the quad → axis-aligned crop. A real
                // perspective-warp could be plugged in later; this is the
                // simple correct version that recognises the inscribed text.
                cv::Rect bb = cv::boundingRect(src) & cv::Rect(0, 0, model_image.cols, model_image.rows);
                if (bb.width > 1 && bb.height > 1) {
                    roi = model_image(bb);
                    ln.text = recogniser_->recognize(roi);
                }
            }
            out.lines.push_back(std::move(ln));
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
        last_error_reason_ = std::string("RouteFrameToPhysicalSceneTextReader failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        out.lines.clear();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalSceneTextReader::ResetPhysicalSceneTextReader() {
    std::lock_guard<std::mutex> lk(mutex_);
    detector_.reset(); recogniser_.reset();
    cfg_ = PhysicalSceneTextReaderConfig{};
    last_error_reason_.clear();
    inference_count_ = 0;
    recogniser_configured_ = false;
    state_ = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalSceneTextReader::GetPhysicalSceneTextReaderState() const {
    std::lock_guard<std::mutex> lk(mutex_); return state_;
}
std::string PhysicalSceneTextReader::GetPhysicalSceneTextReaderLastError() const {
    std::lock_guard<std::mutex> lk(mutex_); return last_error_reason_;
}

}}} // namespace
