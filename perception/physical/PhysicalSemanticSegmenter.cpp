#include "PhysicalSemanticSegmenter.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {
std::vector<std::string> LoadLabelsFile(const std::string& path) {
    std::vector<std::string> out;
    if (path.empty()) return out;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "PhysicalSemanticSegmenter: failed to open class_names_path='" + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}
} // anonymous

PhysicalSemanticSegmenter::PhysicalSemanticSegmenter()  = default;
PhysicalSemanticSegmenter::~PhysicalSemanticSegmenter() = default;

void PhysicalSemanticSegmenter::LoadOnnxModelIntoPhysicalSemanticSegmenter(
    const PhysicalSemanticSegmenterConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_              = cfg;
    net_.reset();
    class_labels_.clear();
    last_error_reason_.clear();
    inference_count_  = 0;

    if (cfg.onnx_model_path.empty()) {
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        return;
    }
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(cfg.onnx_model_path);
        if (net.empty()) throw std::runtime_error("readNetFromONNX returned empty Net");
        net.setPreferableBackend(cfg.dnn_backend_id);
        net.setPreferableTarget(cfg.dnn_target_id);
        net_           = std::make_unique<cv::dnn::Net>(std::move(net));
        class_labels_  = LoadLabelsFile(cfg.class_names_path);
        state_         = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalSemanticSegmenter: ModelLoaded onnx='") + cfg.onnx_model_path + "'");
    } catch (const std::exception& e) {
        net_.reset();
        last_error_reason_ = std::string("LoadOnnxModelIntoPhysicalSemanticSegmenter failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalSemanticSegmenter::RouteFrameToPhysicalSemanticSegmenter(
    const cv::Mat& model_image,
    uint64_t source_frame_counter,
    PhysicalSemanticSegmenterOutput& out)
{
    out = PhysicalSemanticSegmenterOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state             = state_;
    out.last_error_reason = last_error_reason_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!net_) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalSemanticSegmenter: net_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalSemanticSegmenter: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        cv::Mat blob = cv::dnn::blobFromImage(
            model_image, cfg_.input_scale,
            cv::Size(cfg_.input_width, cfg_.input_height),
            cfg_.input_mean, cfg_.swap_rb, /*crop=*/false);
        net_->setInput(blob);
        cv::Mat raw_out = net_->forward();

        if (raw_out.dims != 4 || raw_out.size[0] != 1) {
            throw std::runtime_error(
                "Unexpected output rank/batch (expected [1, C, H, W]) — got "
                + std::to_string(raw_out.dims) + "D, batch="
                + std::to_string(raw_out.size[0]));
        }
        const int C = raw_out.size[1];
        const int H = raw_out.size[2];
        const int W = raw_out.size[3];
        if (C <= 0 || H <= 0 || W <= 0) {
            throw std::runtime_error("Output has non-positive dim");
        }

        // Per-pixel argmax over channels.
        cv::Mat labels(H, W, CV_32SC1, cv::Scalar(0));
        for (int y = 0; y < H; ++y) {
            int32_t* lbl_row = labels.ptr<int32_t>(y);
            for (int x = 0; x < W; ++x) {
                int   best_c = 0;
                float best_v = raw_out.ptr<float>(0, 0, y)[x];
                for (int c = 1; c < C; ++c) {
                    const float v = raw_out.ptr<float>(0, c, y)[x];
                    if (v > best_v) { best_v = v; best_c = c; }
                }
                lbl_row[x] = best_c;
            }
        }

        // Resize to model image resolution using NEAREST so labels are
        // never interpolated (interpolating class ids would be a bug).
        cv::Mat labels_at_model;
        if (W == model_image.cols && H == model_image.rows) {
            labels_at_model = labels;
        } else {
            cv::resize(labels, labels_at_model,
                       cv::Size(model_image.cols, model_image.rows),
                       0, 0, cv::INTER_NEAREST);
        }

        out.segmentation.class_id_image = labels_at_model;
        out.segmentation.num_classes    = C;
        out.segmentation.class_labels   = class_labels_;

        ++inference_count_;
        out.inference_count   = inference_count_;
        out.last_inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        out.state             = PhysicalImageOperatorState::ModelLoaded;
        out.last_error_reason.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        last_error_reason_ = std::string("RouteFrameToPhysicalSemanticSegmenter failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalSemanticSegmenter::ResetPhysicalSemanticSegmenter() {
    std::lock_guard<std::mutex> lk(mutex_);
    net_.reset();
    class_labels_.clear();
    cfg_              = PhysicalSemanticSegmenterConfig{};
    last_error_reason_.clear();
    inference_count_  = 0;
    state_            = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalSemanticSegmenter::GetPhysicalSemanticSegmenterState() const {
    std::lock_guard<std::mutex> lk(mutex_); return state_;
}
std::string PhysicalSemanticSegmenter::GetPhysicalSemanticSegmenterLastError() const {
    std::lock_guard<std::mutex> lk(mutex_); return last_error_reason_;
}

}}} // namespace
