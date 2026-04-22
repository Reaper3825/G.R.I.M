#include "PhysicalObjectDetector.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Read a newline-separated class names file. Throws on open failure.
// Empty `path` returns an empty vector silently — the caller decides whether
// that's an error.
std::vector<std::string> LoadClassLabelsFromTextFile(const std::string& path) {
    std::vector<std::string> out;
    if (path.empty()) return out;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "PhysicalObjectDetector: failed to open class_names_path='" + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

// Map a model-space rectangle back to raw-sensor space using the affine
// transform stored in metadata. Inverse of: model = raw * scale + offset.
cv::Rect2f BackProjectModelRectToRawRect(const cv::Rect2f& m,
                                         const PhysicalSignalRawToModelTransform& t,
                                         int raw_w, int raw_h)
{
    if (t.scale_x == 0.0 || t.scale_y == 0.0) {
        // Degenerate transform → tell the truth: zero-size box at origin.
        return cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv_sx = static_cast<float>(1.0 / t.scale_x);
    const float inv_sy = static_cast<float>(1.0 / t.scale_y);
    const float ox     = static_cast<float>(t.offset_x);
    const float oy     = static_cast<float>(t.offset_y);
    cv::Rect2f r;
    r.x      = (m.x          - ox) * inv_sx;
    r.y      = (m.y          - oy) * inv_sy;
    r.width  =  m.width                 * inv_sx;
    r.height =  m.height                * inv_sy;
    // Clamp to raw bounds — never report negative coords or out-of-image
    // boxes. Consumers can rely on this contract.
    if (r.x < 0)  { r.width  += r.x; r.x = 0; }
    if (r.y < 0)  { r.height += r.y; r.y = 0; }
    if (r.x + r.width  > static_cast<float>(raw_w)) r.width  = static_cast<float>(raw_w) - r.x;
    if (r.y + r.height > static_cast<float>(raw_h)) r.height = static_cast<float>(raw_h) - r.y;
    if (r.width  < 0) r.width  = 0;
    if (r.height < 0) r.height = 0;
    return r;
}

} // anonymous namespace

PhysicalObjectDetector::PhysicalObjectDetector()  = default;
PhysicalObjectDetector::~PhysicalObjectDetector() = default;

void PhysicalObjectDetector::LoadOnnxModelIntoPhysicalObjectDetector(
    const PhysicalObjectDetectorConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);

    cfg_              = cfg;
    net_.reset();
    class_labels_.clear();
    last_error_reason_.clear();
    inference_count_  = 0;

    if (cfg.onnx_model_path.empty()) {
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  "PhysicalObjectDetector: onnx_model_path empty — staying NoModelConfigured");
        return;
    }
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(cfg.onnx_model_path);
        if (net.empty()) {
            throw std::runtime_error("readNetFromONNX returned empty Net");
        }
        net.setPreferableBackend(cfg.dnn_backend_id);
        net.setPreferableTarget(cfg.dnn_target_id);
        net_ = std::make_unique<cv::dnn::Net>(std::move(net));

        class_labels_ = LoadClassLabelsFromTextFile(cfg.class_names_path);

        state_ = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalObjectDetector: ModelLoaded onnx='") + cfg.onnx_model_path
                  + "' input=" + std::to_string(cfg.input_width) + "x" + std::to_string(cfg.input_height)
                  + " classes_listed=" + std::to_string(class_labels_.size()));
    } catch (const std::exception& e) {
        net_.reset();
        class_labels_.clear();
        last_error_reason_ = std::string("LoadOnnxModelIntoPhysicalObjectDetector failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw; // Rule 20: load failure is loud
    }
}

void PhysicalObjectDetector::RouteFrameToPhysicalObjectDetector(
    const cv::Mat& model_image,
    const PhysicalSignalRawToModelTransform& raw_to_model,
    int raw_image_width,
    int raw_image_height,
    uint64_t source_frame_counter,
    PhysicalObjectDetectorOutput& out)
{
    out = PhysicalObjectDetectorOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state = state_;
    out.last_error_reason = last_error_reason_;

    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!net_) {
        // Defence in depth — should be unreachable while state==ModelLoaded.
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalObjectDetector: net_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalObjectDetector: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        cv::Mat blob = cv::dnn::blobFromImage(
            model_image,
            cfg_.input_scale,
            cv::Size(cfg_.input_width, cfg_.input_height),
            cfg_.input_mean,
            cfg_.swap_rb,
            /*crop=*/false);

        net_->setInput(blob);
        std::vector<cv::Mat> outs;
        net_->forward(outs, net_->getUnconnectedOutLayersNames());
        if (outs.empty()) {
            throw std::runtime_error("forward() returned no output tensors");
        }

        // YOLOv8 layout: [1, 4 + C, N]. We transpose to [N, 4 + C] for decoding.
        cv::Mat raw_out = outs[0];
        if (raw_out.dims != 3 || raw_out.size[0] != 1) {
            throw std::runtime_error(
                "Unexpected output rank/batch (expected [1, 4+C, N]) — got "
                + std::to_string(raw_out.dims) + "D, batch=" + std::to_string(raw_out.size[0]));
        }
        const int num_channels = raw_out.size[1];
        const int num_anchors  = raw_out.size[2];
        if (num_channels < 5) {
            throw std::runtime_error(
                "Output channels=" + std::to_string(num_channels)
                + " < 5; need at least 4 box coords + 1 class score");
        }
        const int num_classes = num_channels - 4;

        // Reshape to 2D, then transpose so each row is one anchor.
        cv::Mat reshaped = raw_out.reshape(1, num_channels);  // [num_channels, num_anchors]
        cv::Mat anchors;                                       // [num_anchors, num_channels]
        cv::transpose(reshaped, anchors);

        const float in_w  = static_cast<float>(cfg_.input_width);
        const float in_h  = static_cast<float>(cfg_.input_height);
        const float mdl_w = static_cast<float>(model_image.cols);
        const float mdl_h = static_cast<float>(model_image.rows);
        const float to_model_x = mdl_w / in_w;
        const float to_model_y = mdl_h / in_h;

        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        class_ids;
        boxes.reserve(static_cast<size_t>(num_anchors));
        scores.reserve(static_cast<size_t>(num_anchors));
        class_ids.reserve(static_cast<size_t>(num_anchors));

        for (int i = 0; i < num_anchors; ++i) {
            const float* row = anchors.ptr<float>(i);
            // class scores live at row[4 .. 4+num_classes]
            int   best_cls   = -1;
            float best_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                const float s = row[4 + c];
                if (s > best_score) { best_score = s; best_cls = c; }
            }
            if (best_score < cfg_.confidence_threshold) continue;
            const float cx = row[0] * to_model_x;
            const float cy = row[1] * to_model_y;
            const float w  = row[2] * to_model_x;
            const float h  = row[3] * to_model_y;
            cv::Rect2f r(cx - w * 0.5f, cy - h * 0.5f, w, h);
            boxes.push_back(r);
            scores.push_back(best_score);
            class_ids.push_back(best_cls);
        }

        std::vector<int> kept;
        if (!boxes.empty()) {
            // cv::dnn::NMSBoxes is overloaded for Rect (int) and Rect2d (double),
            // not Rect2f. Convert to Rect2d preserving sub-pixel precision.
            std::vector<cv::Rect2d> boxes_d;
            boxes_d.reserve(boxes.size());
            for (const auto& r : boxes) {
                boxes_d.emplace_back(static_cast<double>(r.x),
                                     static_cast<double>(r.y),
                                     static_cast<double>(r.width),
                                     static_cast<double>(r.height));
            }
            cv::dnn::NMSBoxes(boxes_d, scores,
                              cfg_.confidence_threshold,
                              cfg_.iou_threshold,
                              kept);
            if (static_cast<int>(kept.size()) > cfg_.max_detections) {
                kept.resize(static_cast<size_t>(cfg_.max_detections));
            }
        }

        out.detections.reserve(kept.size());
        for (int idx : kept) {
            PhysicalObjectDetection d;
            d.class_id    = class_ids[static_cast<size_t>(idx)];
            d.confidence  = scores[static_cast<size_t>(idx)];
            d.model_box   = boxes[static_cast<size_t>(idx)];
            d.raw_box     = BackProjectModelRectToRawRect(
                                d.model_box, raw_to_model, raw_image_width, raw_image_height);
            if (d.class_id >= 0 && d.class_id < static_cast<int>(class_labels_.size())) {
                d.class_label = class_labels_[static_cast<size_t>(d.class_id)];
            } else {
                d.class_label = "class_" + std::to_string(d.class_id);
            }
            out.detections.push_back(std::move(d));
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
        last_error_reason_ = std::string("RouteFrameToPhysicalObjectDetector failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        out.detections.clear();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalObjectDetector::ResetPhysicalObjectDetector() {
    std::lock_guard<std::mutex> lk(mutex_);
    net_.reset();
    class_labels_.clear();
    cfg_              = PhysicalObjectDetectorConfig{};
    last_error_reason_.clear();
    inference_count_  = 0;
    state_            = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalObjectDetector::GetPhysicalObjectDetectorState() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_;
}

std::string PhysicalObjectDetector::GetPhysicalObjectDetectorLastError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_reason_;
}

bool PhysicalObjectDetector::IsPhysicalObjectDetectorReady() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_ == PhysicalImageOperatorState::ModelLoaded && net_ != nullptr;
}

}}} // namespace
