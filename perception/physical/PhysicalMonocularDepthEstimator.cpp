#include "PhysicalMonocularDepthEstimator.hpp"

#include "PhysicalSpatialGroundingLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalMonocularDepthEstimator::PhysicalMonocularDepthEstimator()  = default;
PhysicalMonocularDepthEstimator::~PhysicalMonocularDepthEstimator() = default;

void PhysicalMonocularDepthEstimator::LoadOnnxModelIntoPhysicalMonocularDepthEstimator(
    const PhysicalMonocularDepthEstimatorConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);

    cfg_              = cfg;
    net_.reset();
    last_error_reason_.clear();
    inference_count_  = 0;

    // Validate Rule 20: bounds-check inputs that would silently corrupt
    // results downstream.
    if (cfg.input_width <= 0 || cfg.input_height <= 0) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "LoadOnnxModelIntoPhysicalMonocularDepthEstimator: "
                             "input_width/input_height must be > 0 (got "
                             + std::to_string(cfg.input_width) + "x"
                             + std::to_string(cfg.input_height) + ")";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }
    if (cfg.metric_scale_meters < 0.0) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "LoadOnnxModelIntoPhysicalMonocularDepthEstimator: "
                             "metric_scale_meters must be >= 0";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }
    if (cfg.metric_epsilon <= 0.0f) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "LoadOnnxModelIntoPhysicalMonocularDepthEstimator: "
                             "metric_epsilon must be > 0";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }

    if (cfg.onnx_model_path.empty()) {
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG,
                  "PhysicalMonocularDepthEstimator: onnx_model_path empty — staying NoModelConfigured");
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
        state_ = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG,
                  std::string("PhysicalMonocularDepthEstimator: ModelLoaded onnx='")
                  + cfg.onnx_model_path
                  + "' input=" + std::to_string(cfg.input_width) + "x" + std::to_string(cfg.input_height)
                  + " metric_scale_m=" + std::to_string(cfg.metric_scale_meters));
    } catch (const std::exception& e) {
        net_.reset();
        last_error_reason_ = std::string("LoadOnnxModelIntoPhysicalMonocularDepthEstimator failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
        throw; // Rule 20: load failure is loud
    }
}

void PhysicalMonocularDepthEstimator::RouteFrameToPhysicalMonocularDepthEstimator(
    const cv::Mat&  model_image,
    PhysicalDepthMap& out_depth,
    PhysicalImageOperatorState& out_state,
    std::string&    out_error,
    double&         out_inference_ms)
{
    out_depth = PhysicalDepthMap{};
    out_inference_ms = 0.0;

    std::lock_guard<std::mutex> lk(mutex_);
    out_state = state_;
    out_error = last_error_reason_;

    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!net_) {
        out_state = PhysicalImageOperatorState::InferenceFailed;
        out_error = "PhysicalMonocularDepthEstimator: net_ is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, out_error);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out_state = PhysicalImageOperatorState::InferenceFailed;
        out_error = "PhysicalMonocularDepthEstimator: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, out_error);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        // Build a normalised float blob: out = (in*scale - mean) / std.
        // cv::dnn::blobFromImage applies (in - mean) * scale; to get full
        // ImageNet normalisation we must do mean+std manually after the
        // initial scale to [0,1].
        cv::Mat scaled;
        // 1) BGR uint8 → RGB float32 in [0,1]
        cv::Mat rgb;
        if (cfg_.swap_rb) {
            cv::cvtColor(model_image, rgb, cv::COLOR_BGR2RGB);
        } else {
            rgb = model_image;
        }
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(cfg_.input_width, cfg_.input_height), 0, 0, cv::INTER_AREA);
        resized.convertTo(scaled, CV_32FC3, cfg_.input_scale);

        // 2) Per-channel (x - mean/255) / (std/255) — but mean/std were given
        //    in 0..255 ImageNet convention. Combine: x' = (x*scale - mean*scale) / (std*scale).
        const double sx = cfg_.input_scale;
        const cv::Scalar mean_scaled(cfg_.input_mean[0]*sx, cfg_.input_mean[1]*sx, cfg_.input_mean[2]*sx);
        const cv::Scalar std_scaled (cfg_.input_std [0]*sx, cfg_.input_std [1]*sx, cfg_.input_std [2]*sx);
        if (std_scaled[0] <= 0.0 || std_scaled[1] <= 0.0 || std_scaled[2] <= 0.0) {
            throw std::runtime_error("input_std contains a non-positive component — cannot normalise");
        }
        cv::subtract(scaled, mean_scaled, scaled);
        // Per-channel division. cv::divide on a Mat by a Scalar broadcasts.
        std::vector<cv::Mat> ch(3);
        cv::split(scaled, ch);
        ch[0] /= static_cast<float>(std_scaled[0]);
        ch[1] /= static_cast<float>(std_scaled[1]);
        ch[2] /= static_cast<float>(std_scaled[2]);
        cv::merge(ch, scaled);

        cv::Mat blob = cv::dnn::blobFromImage(scaled, /*scalefactor=*/1.0,
                                              cv::Size(cfg_.input_width, cfg_.input_height),
                                              cv::Scalar(),
                                              /*swapRB=*/false, /*crop=*/false);
        net_->setInput(blob);
        cv::Mat raw_out = net_->forward();

        // Accept [1, H, W] (rank 3) and [1, 1, H, W] (rank 4) layouts.
        int H = 0, W = 0;
        cv::Mat depth_2d;
        if (raw_out.dims == 3 && raw_out.size[0] == 1) {
            H = raw_out.size[1];
            W = raw_out.size[2];
            depth_2d = cv::Mat(H, W, CV_32FC1, raw_out.ptr<float>(0)).clone();
        } else if (raw_out.dims == 4 && raw_out.size[0] == 1 && raw_out.size[1] == 1) {
            H = raw_out.size[2];
            W = raw_out.size[3];
            depth_2d = cv::Mat(H, W, CV_32FC1, raw_out.ptr<float>(0, 0)).clone();
        } else {
            throw std::runtime_error(
                "Unexpected depth output shape (expected [1,H,W] or [1,1,H,W]) — got rank="
                + std::to_string(raw_out.dims));
        }
        if (H <= 0 || W <= 0) {
            throw std::runtime_error("Depth output has non-positive dim H="
                                     + std::to_string(H) + " W=" + std::to_string(W));
        }

        // Resize to MODEL resolution (linear is appropriate for depth).
        const int model_w = model_image.cols;
        const int model_h = model_image.rows;
        cv::Mat depth_at_model;
        cv::resize(depth_2d, depth_at_model, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);

        // Compute pre-normalisation min/max across the MODEL-resolution map.
        double mn = 0.0, mx = 0.0;
        cv::minMaxLoc(depth_at_model, &mn, &mx);
        if (!std::isfinite(mn) || !std::isfinite(mx)) {
            throw std::runtime_error("Depth output contains non-finite values (NaN/Inf)");
        }

        // Save raw stats BEFORE normalising so a metric conversion can
        // reverse the scaling.
        cv::Mat raw_inverse = depth_at_model.clone();

        cv::Mat normalised;
        if (mx > mn) {
            normalised = (depth_at_model - mn) / (mx - mn);
        } else {
            normalised = cv::Mat::zeros(model_h, model_w, CV_32FC1);
        }

        out_depth.inverse_depth_image  = normalised;
        out_depth.map_width            = model_w;
        out_depth.map_height           = model_h;
        out_depth.raw_inverse_depth_min = static_cast<float>(mn);
        out_depth.raw_inverse_depth_max = static_cast<float>(mx);

        if (cfg_.metric_scale_meters > 0.0) {
            // depth_m = scale / max(raw_inverse, eps)
            cv::Mat denom;
            cv::max(raw_inverse, cfg_.metric_epsilon, denom);
            cv::Mat metric = cv::Mat::zeros(model_h, model_w, CV_32FC1);
            cv::divide(static_cast<double>(cfg_.metric_scale_meters), denom, metric);
            out_depth.metric_depth_image = metric;
            out_depth.units              = DepthUnits::Meters;
            out_depth.metric_scale_meters = cfg_.metric_scale_meters;
        } else {
            out_depth.units              = DepthUnits::Relative;
            out_depth.metric_scale_meters = 0.0;
        }

        ++inference_count_;
        out_inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        out_state = PhysicalImageOperatorState::ModelLoaded;
        out_error.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        last_error_reason_ = std::string("RouteFrameToPhysicalMonocularDepthEstimator failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out_state          = PhysicalImageOperatorState::InferenceFailed;
        out_error          = last_error_reason_;
        out_depth          = PhysicalDepthMap{};
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
    }
}

void PhysicalMonocularDepthEstimator::ResetPhysicalMonocularDepthEstimator() {
    std::lock_guard<std::mutex> lk(mutex_);
    net_.reset();
    cfg_              = PhysicalMonocularDepthEstimatorConfig{};
    last_error_reason_.clear();
    inference_count_  = 0;
    state_            = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalMonocularDepthEstimator::GetPhysicalMonocularDepthEstimatorState() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_;
}

std::string PhysicalMonocularDepthEstimator::GetPhysicalMonocularDepthEstimatorLastError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_reason_;
}

bool PhysicalMonocularDepthEstimator::IsPhysicalMonocularDepthEstimatorReady() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_ == PhysicalImageOperatorState::ModelLoaded && net_ != nullptr;
}

uint64_t PhysicalMonocularDepthEstimator::GetPhysicalMonocularDepthEstimatorInferenceCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return inference_count_;
}

}}} // namespace GRIM::Perception::Physical
