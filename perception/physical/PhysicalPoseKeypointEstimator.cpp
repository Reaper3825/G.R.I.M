#include "PhysicalPoseKeypointEstimator.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <chrono>
#include <fstream>
#include <limits>
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
            "PhysicalPoseKeypointEstimator: failed to open joint_names_path='" + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

cv::Point2f BackProjectModelPointToRaw(const cv::Point2f& m,
                                       const PhysicalSignalRawToModelTransform& t)
{
    if (t.scale_x == 0.0 || t.scale_y == 0.0) return cv::Point2f(0.0f, 0.0f);
    return cv::Point2f(
        static_cast<float>((m.x - t.offset_x) / t.scale_x),
        static_cast<float>((m.y - t.offset_y) / t.scale_y));
}
} // anonymous

PhysicalPoseKeypointEstimator::PhysicalPoseKeypointEstimator()  = default;
PhysicalPoseKeypointEstimator::~PhysicalPoseKeypointEstimator() = default;

void PhysicalPoseKeypointEstimator::LoadOnnxModelIntoPhysicalPoseKeypointEstimator(
    const PhysicalPoseKeypointEstimatorConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_              = cfg;
    net_.reset();
    joint_labels_.clear();
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
        joint_labels_  = LoadLabelsFile(cfg.joint_names_path);
        state_         = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalPoseKeypointEstimator: ModelLoaded onnx='") + cfg.onnx_model_path + "'");
    } catch (const std::exception& e) {
        net_.reset();
        last_error_reason_ = std::string("LoadOnnxModelIntoPhysicalPoseKeypointEstimator failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalPoseKeypointEstimator::RouteFrameToPhysicalPoseKeypointEstimator(
    const cv::Mat& model_image,
    const PhysicalSignalRawToModelTransform& raw_to_model,
    int raw_image_width,
    int raw_image_height,
    uint64_t source_frame_counter,
    PhysicalPoseKeypointEstimatorOutput& out)
{
    out = PhysicalPoseKeypointEstimatorOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state             = state_;
    out.last_error_reason = last_error_reason_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!net_) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalPoseKeypointEstimator: net_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalPoseKeypointEstimator: model_image is empty or not CV_8UC3";
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
                "Unexpected output rank/batch (expected [1, J, H, W]) — got "
                + std::to_string(raw_out.dims) + "D");
        }
        const int J = raw_out.size[1];
        const int H = raw_out.size[2];
        const int W = raw_out.size[3];
        if (J <= 0 || H <= 0 || W <= 0) throw std::runtime_error("Output has non-positive dim");

        const float to_model_x = static_cast<float>(model_image.cols) / static_cast<float>(W);
        const float to_model_y = static_cast<float>(model_image.rows) / static_cast<float>(H);

        PhysicalPoseInstance inst;
        inst.keypoints.reserve(static_cast<size_t>(J));
        float min_x =  std::numeric_limits<float>::max();
        float min_y =  std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        bool  any_visible = false;
        double conf_sum   = 0.0;
        int    conf_count = 0;

        for (int j = 0; j < J; ++j) {
            int   peak_x = 0, peak_y = 0;
            float peak_v = -std::numeric_limits<float>::max();
            for (int y = 0; y < H; ++y) {
                const float* row = raw_out.ptr<float>(0, j, y);
                for (int x = 0; x < W; ++x) {
                    if (row[x] > peak_v) { peak_v = row[x]; peak_x = x; peak_y = y; }
                }
            }
            PhysicalPoseKeypoint kp;
            kp.joint_id   = j;
            kp.confidence = peak_v;
            kp.visible    = peak_v >= cfg_.min_keypoint_confidence;
            kp.model_xy   = cv::Point2f(static_cast<float>(peak_x) * to_model_x,
                                        static_cast<float>(peak_y) * to_model_y);
            kp.raw_xy     = BackProjectModelPointToRaw(kp.model_xy, raw_to_model);
            if (j >= 0 && j < static_cast<int>(joint_labels_.size())) {
                kp.joint_label = joint_labels_[static_cast<size_t>(j)];
            } else {
                kp.joint_label = "joint_" + std::to_string(j);
            }
            if (kp.visible) {
                any_visible = true;
                if (kp.model_xy.x < min_x) min_x = kp.model_xy.x;
                if (kp.model_xy.y < min_y) min_y = kp.model_xy.y;
                if (kp.model_xy.x > max_x) max_x = kp.model_xy.x;
                if (kp.model_xy.y > max_y) max_y = kp.model_xy.y;
                conf_sum += kp.confidence;
                ++conf_count;
            }
            inst.keypoints.push_back(std::move(kp));
        }

        if (any_visible) {
            inst.model_bbox = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
            // Back-project bbox using the same point transform on its corners.
            const cv::Point2f tl = BackProjectModelPointToRaw(
                cv::Point2f(inst.model_bbox.x, inst.model_bbox.y), raw_to_model);
            const cv::Point2f br = BackProjectModelPointToRaw(
                cv::Point2f(inst.model_bbox.x + inst.model_bbox.width,
                            inst.model_bbox.y + inst.model_bbox.height), raw_to_model);
            float rx = tl.x, ry = tl.y, rw = br.x - tl.x, rh = br.y - tl.y;
            if (rx < 0) { rw += rx; rx = 0; }
            if (ry < 0) { rh += ry; ry = 0; }
            if (rx + rw > static_cast<float>(raw_image_width))  rw = static_cast<float>(raw_image_width)  - rx;
            if (ry + rh > static_cast<float>(raw_image_height)) rh = static_cast<float>(raw_image_height) - ry;
            if (rw < 0) rw = 0;
            if (rh < 0) rh = 0;
            inst.raw_bbox            = cv::Rect2f(rx, ry, rw, rh);
            inst.instance_confidence = static_cast<float>(conf_sum / static_cast<double>(conf_count));
            out.instances.push_back(std::move(inst));
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
        last_error_reason_ = std::string("RouteFrameToPhysicalPoseKeypointEstimator failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        out.instances.clear();
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalPoseKeypointEstimator::ResetPhysicalPoseKeypointEstimator() {
    std::lock_guard<std::mutex> lk(mutex_);
    net_.reset(); joint_labels_.clear();
    cfg_ = PhysicalPoseKeypointEstimatorConfig{};
    last_error_reason_.clear(); inference_count_ = 0;
    state_ = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalPoseKeypointEstimator::GetPhysicalPoseKeypointEstimatorState() const {
    std::lock_guard<std::mutex> lk(mutex_); return state_;
}
std::string PhysicalPoseKeypointEstimator::GetPhysicalPoseKeypointEstimatorLastError() const {
    std::lock_guard<std::mutex> lk(mutex_); return last_error_reason_;
}

}}} // namespace
