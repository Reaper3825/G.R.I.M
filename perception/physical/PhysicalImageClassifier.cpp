#include "PhysicalImageClassifier.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
            "PhysicalImageClassifier: failed to open class_names_path='" + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

// Numerically-stable softmax over a flat float row.
void SoftmaxInPlace(float* p, int n) {
    if (n <= 0) return;
    float m = p[0];
    for (int i = 1; i < n; ++i) if (p[i] > m) m = p[i];
    double s = 0.0;
    for (int i = 0; i < n; ++i) { p[i] = std::exp(p[i] - m); s += p[i]; }
    if (s <= 0.0) { for (int i = 0; i < n; ++i) p[i] = 1.0f / static_cast<float>(n); return; }
    const float inv = static_cast<float>(1.0 / s);
    for (int i = 0; i < n; ++i) p[i] *= inv;
}
} // anonymous

PhysicalImageClassifier::PhysicalImageClassifier()  = default;
PhysicalImageClassifier::~PhysicalImageClassifier() = default;

void PhysicalImageClassifier::LoadOnnxModelIntoPhysicalImageClassifier(
    const PhysicalImageClassifierConfig& cfg)
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
    if (cfg.top_k <= 0) {
        last_error_reason_ = "PhysicalImageClassifier: top_k must be > 0";
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
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
                  std::string("PhysicalImageClassifier: ModelLoaded onnx='") + cfg.onnx_model_path + "'");
    } catch (const std::exception& e) {
        net_.reset();
        last_error_reason_ = std::string("LoadOnnxModelIntoPhysicalImageClassifier failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalImageClassifier::RouteFrameToPhysicalImageClassifier(
    const cv::Mat& model_image,
    uint64_t source_frame_counter,
    PhysicalImageClassifierOutput& out)
{
    out = PhysicalImageClassifierOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state             = state_;
    out.last_error_reason = last_error_reason_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!net_) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalImageClassifier: net_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalImageClassifier: model_image is empty or not CV_8UC3";
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

        // Accept either [1, C] or [1, C, 1, 1] (squeeze trailing dims).
        cv::Mat scores = raw_out.reshape(1, 1); // -> 1 x N where N = total elements
        const int N = scores.cols;
        if (N <= 0) throw std::runtime_error("Output has 0 elements");

        std::vector<float> probs(N);
        std::memcpy(probs.data(), scores.ptr<float>(0), sizeof(float) * static_cast<size_t>(N));
        SoftmaxInPlace(probs.data(), N);

        // Top-K via partial sort of indices.
        std::vector<int> idx(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) idx[static_cast<size_t>(i)] = i;
        const int K = std::min(cfg_.top_k, N);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { return probs[static_cast<size_t>(a)]
                                                    > probs[static_cast<size_t>(b)]; });

        out.top_k.reserve(static_cast<size_t>(K));
        for (int i = 0; i < K; ++i) {
            const int cid = idx[static_cast<size_t>(i)];
            PhysicalImageClassification c;
            c.class_id = cid;
            c.score    = probs[static_cast<size_t>(cid)];
            if (cid >= 0 && cid < static_cast<int>(class_labels_.size())) {
                c.class_label = class_labels_[static_cast<size_t>(cid)];
            } else {
                c.class_label = "class_" + std::to_string(cid);
            }
            out.top_k.push_back(std::move(c));
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
        last_error_reason_ = std::string("RouteFrameToPhysicalImageClassifier failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalImageClassifier::ResetPhysicalImageClassifier() {
    std::lock_guard<std::mutex> lk(mutex_);
    net_.reset(); class_labels_.clear();
    cfg_ = PhysicalImageClassifierConfig{};
    last_error_reason_.clear(); inference_count_ = 0;
    state_ = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalImageClassifier::GetPhysicalImageClassifierState() const {
    std::lock_guard<std::mutex> lk(mutex_); return state_;
}
std::string PhysicalImageClassifier::GetPhysicalImageClassifierLastError() const {
    std::lock_guard<std::mutex> lk(mutex_); return last_error_reason_;
}

}}} // namespace
