#include "PhysicalFrameConditioner.hpp"

#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

[[noreturn]] void ThrowInvalidPhysicalSignalConditioningConfig(const std::string& reason) {
    throw std::runtime_error(
        "PhysicalSignalConditioning config invalid: " + reason
        + " [" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]");
}

}

PhysicalSignalConditioningConfig BuildDefaultPhysicalSignalConditioningConfig() {
    return PhysicalSignalConditioningConfig{};
}

PhysicalFrameConditioner::PhysicalFrameConditioner()
    : config_(BuildDefaultPhysicalSignalConditioningConfig()) {}

void PhysicalFrameConditioner::ValidatePhysicalSignalConditioningConfig(
    const PhysicalSignalConditioningConfig& cfg) {
    if (cfg.output_width <= 0 || cfg.output_height <= 0) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "output size must be > 0, got "
            + std::to_string(cfg.output_width) + "x" + std::to_string(cfg.output_height));
    }
    if (cfg.denoise_strength < 0 || cfg.denoise_strength > 30) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "denoise_strength must be in [0,30], got " + std::to_string(cfg.denoise_strength));
    }
    if (!(cfg.manual_exposure_gain > 0.0) || !std::isfinite(cfg.manual_exposure_gain)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "manual_exposure_gain must be finite and > 0, got "
            + std::to_string(cfg.manual_exposure_gain));
    }
    if (!(cfg.target_luma > 0.0) || cfg.target_luma > 255.0 || !std::isfinite(cfg.target_luma)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "target_luma must be finite and in (0,255], got "
            + std::to_string(cfg.target_luma));
    }
    if (!(cfg.deblur_amount >= 0.0) || cfg.deblur_amount > 3.0 || !std::isfinite(cfg.deblur_amount)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "deblur_amount must be finite and in [0,3], got "
            + std::to_string(cfg.deblur_amount));
    }
    if (cfg.flow_max_corners < 16 || cfg.flow_max_corners > 4000) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "flow_max_corners must be in [16,4000], got " + std::to_string(cfg.flow_max_corners));
    }
    if (!(cfg.flow_quality_level > 0.0) || cfg.flow_quality_level > 1.0
        || !std::isfinite(cfg.flow_quality_level)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "flow_quality_level must be finite and in (0,1], got "
            + std::to_string(cfg.flow_quality_level));
    }
    if (!(cfg.flow_min_distance > 0.0) || !std::isfinite(cfg.flow_min_distance)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "flow_min_distance must be finite and > 0, got "
            + std::to_string(cfg.flow_min_distance));
    }
    if (cfg.letterbox_pad_value < 0 || cfg.letterbox_pad_value > 255) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "letterbox_pad_value must be in [0,255], got "
            + std::to_string(cfg.letterbox_pad_value));
    }
    const auto& qg = cfg.quality_gate;
    if (!(qg.min_mean_luma >= 0.0 && qg.max_mean_luma <= 255.0
          && qg.min_mean_luma < qg.max_mean_luma)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "quality_gate luma window invalid: min="
            + std::to_string(qg.min_mean_luma) + " max="
            + std::to_string(qg.max_mean_luma));
    }
    if (!(qg.max_clipped_pixel_ratio >= 0.0 && qg.max_clipped_pixel_ratio <= 1.0)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "quality_gate.max_clipped_pixel_ratio must be in [0,1], got "
            + std::to_string(qg.max_clipped_pixel_ratio));
    }
    if (!(qg.min_laplacian_variance >= 0.0)) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "quality_gate.min_laplacian_variance must be >= 0, got "
            + std::to_string(qg.min_laplacian_variance));
    }
}

void PhysicalFrameConditioner::ConfigurePhysicalSignalConditioning(
    const PhysicalSignalConditioningConfig& cfg) {
    ValidatePhysicalSignalConditioningConfig(cfg);
    const bool geometry_changed = (cfg.output_width != config_.output_width)
                               || (cfg.output_height != config_.output_height);
    config_ = cfg;
    if (geometry_changed || !cfg.enable_stabilization) {
        previous_gray_for_flow_.release();
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "ConfigurePhysicalSignalConditioning: resize="
                  + std::to_string(static_cast<int>(cfg.enable_resize))
                  + " out=" + std::to_string(cfg.output_width) + "x"
                  + std::to_string(cfg.output_height)
                  + " denoise=" + std::to_string(static_cast<int>(cfg.enable_denoise))
                  + " exposure=" + std::to_string(static_cast<int>(cfg.enable_exposure_correction))
                  + " autoExp=" + std::to_string(static_cast<int>(cfg.exposure_auto))
                  + " deblur=" + std::to_string(static_cast<int>(cfg.enable_deblur))
                  + " stabilize=" + std::to_string(static_cast<int>(cfg.enable_stabilization)));
}

void PhysicalFrameConditioner::ResetPhysicalSignalConditioningToDefaults() {
    config_ = BuildDefaultPhysicalSignalConditioningConfig();
    previous_gray_for_flow_.release();
    previous_scene_thumbnail_gray_.release();
    previous_scene_hash_64_    = 0;
    previous_scene_hash_valid_ = false;
    scene_stable_streak_       = 0;
    status_ = {};
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "ResetPhysicalSignalConditioningToDefaults: restored default pipeline settings");
}

void PhysicalFrameConditioner::ResetPhysicalSignalConditioningTemporalState() {
    previous_gray_for_flow_.release();
    previous_scene_thumbnail_gray_.release();
    previous_scene_hash_64_    = 0;
    previous_scene_hash_valid_ = false;
    scene_stable_streak_       = 0;
    status_.last_flow_dx = 0.0;
    status_.last_flow_dy = 0.0;
    status_.last_flow_tracked_points = 0;
    status_.last_scene_stability = {};
}

PhysicalSignalConditioningConfig PhysicalFrameConditioner::GetPhysicalSignalConditioningConfigSnapshot() const {
    return config_;
}

PhysicalSignalConditioningStatus PhysicalFrameConditioner::GetPhysicalSignalConditioningStatusSnapshot() const {
    return status_;
}

double PhysicalFrameConditioner::ComputeMeanLumaFromBgr(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "ComputeMeanLumaFromBgr: expected non-empty CV_8UC3 frame");
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}

namespace {

// Fraction of pixels that are exactly 0 or exactly 255 in any channel.
double ComputeClippedPixelRatio(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "ComputeClippedPixelRatio: expected non-empty CV_8UC3 frame");
    }
    const int total_pixels = bgr.rows * bgr.cols;
    if (total_pixels == 0) return 0.0;
    int clipped = 0;
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b& p = row[x];
            if (p[0] == 0 && p[1] == 0 && p[2] == 0) { ++clipped; continue; }
            if (p[0] == 255 && p[1] == 255 && p[2] == 255) { ++clipped; }
        }
    }
    return static_cast<double>(clipped) / static_cast<double>(total_pixels);
}

// Variance of Laplacian on the gray image — the standard OpenCV blur metric.
double ComputeLaplacianVarianceFromBgr(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "ComputeLaplacianVarianceFromBgr: expected non-empty CV_8UC3 frame");
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F, 3);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    const double s = sigma[0];
    return s * s;
}

// Letterbox raw -> (out_w, out_h) preserving aspect ratio.
// Returns the affine transform raw -> letterboxed model.
PhysicalSignalRawToModelTransform LetterboxResize(const cv::Mat& src,
                                                  int out_w,
                                                  int out_h,
                                                  int pad_value,
                                                  cv::Mat& dst) {
    const double sx = static_cast<double>(out_w) / static_cast<double>(src.cols);
    const double sy = static_cast<double>(out_h) / static_cast<double>(src.rows);
    const double s  = std::min(sx, sy);
    const int new_w = std::max(1, static_cast<int>(std::round(src.cols * s)));
    const int new_h = std::max(1, static_cast<int>(std::round(src.rows * s)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
    const int pad_left   = (out_w - new_w) / 2;
    const int pad_top    = (out_h - new_h) / 2;
    const int pad_right  = out_w - new_w - pad_left;
    const int pad_bottom = out_h - new_h - pad_top;
    cv::copyMakeBorder(resized,
                       dst,
                       pad_top,
                       pad_bottom,
                       pad_left,
                       pad_right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(pad_value, pad_value, pad_value));
    PhysicalSignalRawToModelTransform t;
    t.scale_x  = s;
    t.scale_y  = s;
    t.offset_x = static_cast<double>(pad_left);
    t.offset_y = static_cast<double>(pad_top);
    return t;
}

} // namespace

namespace {

// Compute a 64-bit average-hash bitmap from a small grayscale thumbnail.
// We tile the thumbnail into an 8x8 grid (block-mean) and compare each
// block mean against the overall mean. Bit i (LSB-first) is 1 iff the
// i-th block mean >= overall mean. Robust to small luma drift; sensitive
// to layout changes.
uint64_t ComputeAverageHash64FromGray(const cv::Mat& gray) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        throw std::runtime_error(
            "ComputeAverageHash64FromGray: expected non-empty CV_8UC1 thumbnail");
    }
    if (gray.cols < 8 || gray.rows < 8) {
        throw std::runtime_error(
            "ComputeAverageHash64FromGray: thumbnail must be at least 8x8, got "
            + std::to_string(gray.cols) + "x" + std::to_string(gray.rows));
    }
    cv::Mat block_means(8, 8, CV_32F, cv::Scalar(0.0f));
    const double bw = static_cast<double>(gray.cols) / 8.0;
    const double bh = static_cast<double>(gray.rows) / 8.0;
    double total = 0.0;
    for (int by = 0; by < 8; ++by) {
        const int y0 = static_cast<int>(std::floor(by * bh));
        const int y1 = std::max(y0 + 1, static_cast<int>(std::floor((by + 1) * bh)));
        for (int bx = 0; bx < 8; ++bx) {
            const int x0 = static_cast<int>(std::floor(bx * bw));
            const int x1 = std::max(x0 + 1, static_cast<int>(std::floor((bx + 1) * bw)));
            const cv::Rect roi(x0, y0,
                               std::min(x1 - x0, gray.cols - x0),
                               std::min(y1 - y0, gray.rows - y0));
            const float m = static_cast<float>(cv::mean(gray(roi))[0]);
            block_means.at<float>(by, bx) = m;
            total += m;
        }
    }
    const float overall_mean = static_cast<float>(total / 64.0);
    uint64_t bits = 0;
    for (int i = 0; i < 64; ++i) {
        const int by = i / 8;
        const int bx = i % 8;
        if (block_means.at<float>(by, bx) >= overall_mean) {
            bits |= (uint64_t{1} << i);
        }
    }
    return bits;
}

int Popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int n = 0;
    while (x) { x &= (x - 1); ++n; }
    return n;
#endif
}

} // namespace

PhysicalSignalConditioningResult PhysicalFrameConditioner::ProcessRawFrameToModelSignal(
    const cv::Mat& raw_bgr,
    uint64_t       frame_counter,
    cv::Mat&       out_model_bgr) {
    const auto pass_start = std::chrono::steady_clock::now();
    auto elapsed_ms_since = [](const auto& start) -> double {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };

    if (raw_bgr.empty()) {
        throw std::runtime_error(
            "ProcessRawFrameToModelSignal: raw_bgr is empty");
    }
    if (raw_bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "ProcessRawFrameToModelSignal: expected raw_bgr type CV_8UC3, got "
            + std::to_string(raw_bgr.type()));
    }

    ValidatePhysicalSignalConditioningConfig(config_);

    PhysicalSignalConditioningResult result;
    result.raw_width  = raw_bgr.cols;
    result.raw_height = raw_bgr.rows;

    status_.processed_frame_counter = frame_counter;
    status_.last_input_width        = raw_bgr.cols;
    status_.last_input_height       = raw_bgr.rows;
    status_.last_input_luma         = ComputeMeanLumaFromBgr(raw_bgr);
    status_.last_failure_reason.clear();
    status_.last_applied_exposure_gain = 1.0;
    status_.using_auto_exposure        = config_.exposure_auto;
    status_.stabilization_active       = config_.enable_stabilization;
    status_.last_quality_gate_passed   = true;
    status_.last_quality_gate_reason.clear();
    status_.last_total_ms             = 0.0;
    status_.last_quality_gate_ms      = 0.0;
    status_.last_stabilization_ms     = 0.0;
    status_.last_denoise_ms           = 0.0;
    status_.last_exposure_ms          = 0.0;
    status_.last_deblur_ms            = 0.0;
    status_.last_resize_ms            = 0.0;
    status_.last_color_convert_ms     = 0.0;
    status_.last_scene_stability_ms   = 0.0;

    // ---- Quality gate (computed on RAW input — drop before wasting work) ----
    auto stage_start = std::chrono::steady_clock::now();
    const double clipped_ratio = ComputeClippedPixelRatio(raw_bgr);
    const double lap_var       = ComputeLaplacianVarianceFromBgr(raw_bgr);
    result.quality_gate_ms = elapsed_ms_since(stage_start);
    status_.last_quality_gate_ms = result.quality_gate_ms;
    status_.last_clipped_pixel_ratio = clipped_ratio;
    status_.last_laplacian_variance  = lap_var;

    if (config_.quality_gate.enable) {
        const auto& qg = config_.quality_gate;
        std::string reason;
        if (status_.last_input_luma < qg.min_mean_luma) {
            reason = "luma_too_low(" + std::to_string(status_.last_input_luma)
                   + " < " + std::to_string(qg.min_mean_luma) + ")";
        } else if (status_.last_input_luma > qg.max_mean_luma) {
            reason = "luma_too_high(" + std::to_string(status_.last_input_luma)
                   + " > " + std::to_string(qg.max_mean_luma) + ")";
        } else if (clipped_ratio > qg.max_clipped_pixel_ratio) {
            reason = "clipped_pixels(" + std::to_string(clipped_ratio)
                   + " > " + std::to_string(qg.max_clipped_pixel_ratio) + ")";
        } else if (lap_var < qg.min_laplacian_variance) {
            reason = "too_blurry(lap_var=" + std::to_string(lap_var)
                   + " < " + std::to_string(qg.min_laplacian_variance) + ")";
        }
        if (!reason.empty()) {
            status_.last_quality_gate_passed = false;
            status_.last_quality_gate_reason = reason;
            status_.total_frames_dropped_by_quality_gate += 1;
            LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                      "PhysicalSignalConditioning quality gate dropped frame "
                          + std::to_string(frame_counter) + ": " + reason);
            result.accepted    = false;
            result.drop_reason = reason;
            result.total_ms = elapsed_ms_since(pass_start);
            status_.last_total_ms = result.total_ms;
            return result;
        }
    }

    cv::Mat working = raw_bgr;
    std::ostringstream pipeline;

    if (config_.enable_stabilization) {
        stage_start = std::chrono::steady_clock::now();
        cv::Mat curr_gray;
        cv::cvtColor(working, curr_gray, cv::COLOR_BGR2GRAY);
        status_.last_flow_dx = 0.0;
        status_.last_flow_dy = 0.0;
        status_.last_flow_tracked_points = 0;

        if (!previous_gray_for_flow_.empty()
            && previous_gray_for_flow_.size() == curr_gray.size()) {
            std::vector<cv::Point2f> prev_points;
            cv::goodFeaturesToTrack(previous_gray_for_flow_,
                                    prev_points,
                                    config_.flow_max_corners,
                                    config_.flow_quality_level,
                                    config_.flow_min_distance);
            if (!prev_points.empty()) {
                std::vector<cv::Point2f> curr_points;
                std::vector<unsigned char> track_status;
                std::vector<float> track_error;
                cv::calcOpticalFlowPyrLK(previous_gray_for_flow_,
                                         curr_gray,
                                         prev_points,
                                         curr_points,
                                         track_status,
                                         track_error);

                std::vector<cv::Point2f> in_prev;
                std::vector<cv::Point2f> in_curr;
                in_prev.reserve(prev_points.size());
                in_curr.reserve(prev_points.size());
                for (size_t i = 0; i < prev_points.size(); ++i) {
                    if (track_status[i]) {
                        in_prev.push_back(prev_points[i]);
                        in_curr.push_back(curr_points[i]);
                    }
                }

                status_.last_flow_tracked_points = static_cast<int>(in_prev.size());

                if (in_prev.size() >= 6) {
                    cv::Mat inlier_mask;
                    cv::Mat affine_curr_to_prev = cv::estimateAffinePartial2D(
                        in_curr,
                        in_prev,
                        inlier_mask,
                        cv::RANSAC,
                        3.0);
                    if (!affine_curr_to_prev.empty()
                        && affine_curr_to_prev.rows == 2
                        && affine_curr_to_prev.cols == 3
                        && affine_curr_to_prev.type() == CV_64F) {
                        status_.last_flow_dx = affine_curr_to_prev.at<double>(0, 2);
                        status_.last_flow_dy = affine_curr_to_prev.at<double>(1, 2);
                        cv::Mat stabilized;
                        cv::warpAffine(working,
                                       stabilized,
                                       affine_curr_to_prev,
                                       working.size(),
                                       cv::INTER_LINEAR,
                                       cv::BORDER_REPLICATE);
                        working = stabilized;
                    }
                }
            }
        }
        previous_gray_for_flow_ = curr_gray;
        pipeline << "stabilize ";
        result.stabilization_ms = elapsed_ms_since(stage_start);
        status_.last_stabilization_ms = result.stabilization_ms;
    } else {
        previous_gray_for_flow_.release();
    }

    if (config_.enable_denoise && config_.denoise_strength > 0) {
        stage_start = std::chrono::steady_clock::now();
        cv::Mat denoised;
        cv::fastNlMeansDenoisingColored(working,
                                        denoised,
                                        static_cast<float>(config_.denoise_strength),
                                        static_cast<float>(config_.denoise_strength),
                                        7,
                                        21);
        working = denoised;
        pipeline << "denoise ";
        result.denoise_ms = elapsed_ms_since(stage_start);
        status_.last_denoise_ms = result.denoise_ms;
    }

    if (config_.enable_exposure_correction) {
        stage_start = std::chrono::steady_clock::now();
        const double observed_luma = ComputeMeanLumaFromBgr(working);
        double gain = config_.manual_exposure_gain;
        if (config_.exposure_auto) {
            const double denom = std::max(1.0, observed_luma);
            gain = config_.target_luma / denom;
            gain = std::clamp(gain, 0.50, 2.50);
        }
        cv::Mat corrected;
        working.convertTo(corrected, -1, gain, 0.0);
        working = corrected;
        status_.last_applied_exposure_gain = gain;
        pipeline << "exposure(g=" << gain << ") ";
        result.exposure_ms = elapsed_ms_since(stage_start);
        status_.last_exposure_ms = result.exposure_ms;
    }

    if (config_.enable_deblur && config_.deblur_amount > 0.0) {
        stage_start = std::chrono::steady_clock::now();
        cv::Mat blurred;
        cv::GaussianBlur(working, blurred, cv::Size(0, 0), 1.2, 1.2);
        cv::Mat sharpened;
        cv::addWeighted(working,
                        1.0 + config_.deblur_amount,
                        blurred,
                        -config_.deblur_amount,
                        0.0,
                        sharpened);
        working = sharpened;
        pipeline << "deblur(a=" << config_.deblur_amount << ") ";
        result.deblur_ms = elapsed_ms_since(stage_start);
        status_.last_deblur_ms = result.deblur_ms;
    }

    // ---- Resize stage. Track the raw->model transform regardless of mode. ----
    PhysicalSignalRawToModelTransform raw_to_model;
    if (config_.enable_resize) {
        stage_start = std::chrono::steady_clock::now();
        if (config_.resize_mode == PhysicalSignalResizeMode::Letterbox) {
            cv::Mat resized;
            raw_to_model = LetterboxResize(working,
                                           config_.output_width,
                                           config_.output_height,
                                           config_.letterbox_pad_value,
                                           resized);
            working = resized;
            pipeline << "letterbox(" << config_.output_width << "x"
                     << config_.output_height << ",pad="
                     << config_.letterbox_pad_value << ") ";
        } else {
            cv::Mat resized;
            cv::resize(working,
                       resized,
                       cv::Size(config_.output_width, config_.output_height),
                       0.0,
                       0.0,
                       cv::INTER_AREA);
            raw_to_model.scale_x  = static_cast<double>(config_.output_width)
                                  / static_cast<double>(raw_bgr.cols);
            raw_to_model.scale_y  = static_cast<double>(config_.output_height)
                                  / static_cast<double>(raw_bgr.rows);
            raw_to_model.offset_x = 0.0;
            raw_to_model.offset_y = 0.0;
            working = resized;
            pipeline << "resize(" << config_.output_width << "x"
                     << config_.output_height << ") ";
        }
        result.resize_ms = elapsed_ms_since(stage_start);
        status_.last_resize_ms = result.resize_ms;
    } else {
        raw_to_model.scale_x  = 1.0;
        raw_to_model.scale_y  = 1.0;
        raw_to_model.offset_x = 0.0;
        raw_to_model.offset_y = 0.0;
    }

    if (config_.color_mode == PhysicalSignalColorMode::Gray) {
        stage_start = std::chrono::steady_clock::now();
        cv::Mat gray;
        cv::cvtColor(working, gray, cv::COLOR_BGR2GRAY);
        cv::Mat gray_bgr;
        cv::cvtColor(gray, gray_bgr, cv::COLOR_GRAY2BGR);
        working = gray_bgr;
        pipeline << "gray ";
        result.color_convert_ms = elapsed_ms_since(stage_start);
        status_.last_color_convert_ms = result.color_convert_ms;
    } else {
        pipeline << "bgr ";
    }

    if (working.empty() || working.type() != CV_8UC3) {
        throw std::runtime_error(
            "ProcessRawFrameToModelSignal: pipeline produced invalid output frame");
    }

    working.copyTo(out_model_bgr);
    status_.last_output_width   = out_model_bgr.cols;
    status_.last_output_height  = out_model_bgr.rows;
    status_.last_output_luma    = ComputeMeanLumaFromBgr(out_model_bgr);
    status_.last_pipeline_summary = pipeline.str();
    status_.last_raw_to_model     = raw_to_model;

    // ---- Scene-stability signal. Cheap thumbnail + 64-bit average hash. ----
    // Computed on the CONDITIONED model image so it benefits from any
    // stabilization / denoise that already ran. Skipped (and reported as
    // valid=false) only if the operator was explicitly disabled in config —
    // Rule 20: never silently fall through.
    PhysicalSceneStability stability{};
    const auto& sc = config_.scene_stability;
    if (sc.enable) {
        stage_start = std::chrono::steady_clock::now();
        if (sc.thumbnail_width < 8 || sc.thumbnail_height < 8) {
            throw std::runtime_error(
                "ProcessRawFrameToModelSignal: scene_stability thumbnail must be"
                " at least 8x8, got "
                + std::to_string(sc.thumbnail_width) + "x"
                + std::to_string(sc.thumbnail_height));
        }
        cv::Mat thumb_bgr;
        cv::resize(out_model_bgr,
                   thumb_bgr,
                   cv::Size(sc.thumbnail_width, sc.thumbnail_height),
                   0.0, 0.0, cv::INTER_AREA);
        cv::Mat thumb_gray;
        cv::cvtColor(thumb_bgr, thumb_gray, cv::COLOR_BGR2GRAY);

        const uint64_t curr_hash = ComputeAverageHash64FromGray(thumb_gray);
        const bool have_prev = !previous_scene_thumbnail_gray_.empty()
                               && previous_scene_thumbnail_gray_.size() == thumb_gray.size()
                               && previous_scene_hash_valid_;

        double mad = 0.0;
        int    hamming = 64;
        bool   is_stable = false;
        std::string change_reason;

        if (!have_prev) {
            change_reason = "first_frame";
            scene_stable_streak_ = 0;
        } else {
            cv::Mat diff;
            cv::absdiff(previous_scene_thumbnail_gray_, thumb_gray, diff);
            mad = cv::mean(diff)[0];
            hamming = Popcount64(curr_hash ^ previous_scene_hash_64_);
            const bool motion_ok = (mad <= sc.motion_threshold);
            const bool hash_ok   = (hamming <= sc.hash_hamming_threshold);
            if (!motion_ok && !hash_ok) {
                change_reason = "motion_and_hash(mad=" + std::to_string(mad)
                              + ",hamming=" + std::to_string(hamming) + ")";
            } else if (!motion_ok) {
                change_reason = "motion(mad=" + std::to_string(mad)
                              + ">thr=" + std::to_string(sc.motion_threshold) + ")";
            } else if (!hash_ok) {
                change_reason = "hash(hamming=" + std::to_string(hamming)
                              + ">thr=" + std::to_string(sc.hash_hamming_threshold) + ")";
            } else {
                is_stable = true;
            }
        }

        if (is_stable) {
            if (scene_stable_streak_ >= sc.max_stable_streak_frames) {
                // Force a refresh — break the gate so caches re-prime.
                is_stable = false;
                change_reason = "stable_streak_capped("
                              + std::to_string(scene_stable_streak_) + ")";
                stability.stable_streak_capped = true;
                scene_stable_streak_ = 0;
            } else {
                ++scene_stable_streak_;
            }
        } else {
            scene_stable_streak_ = 0;
        }

        stability.valid                = true;
        stability.motion_magnitude     = mad;
        stability.scene_hash_64        = curr_hash;
        stability.hamming_vs_previous  = hamming;
        stability.is_stable            = is_stable;
        stability.frames_since_change  = scene_stable_streak_;
        stability.change_reason        = change_reason;

        previous_scene_thumbnail_gray_ = thumb_gray;  // shallow share is fine; we own it
        previous_scene_hash_64_        = curr_hash;
        previous_scene_hash_valid_     = true;
        result.scene_stability_ms = elapsed_ms_since(stage_start);
        status_.last_scene_stability_ms = result.scene_stability_ms;
    } else {
        // Disabled by config. Drop temporal state so re-enabling starts fresh.
        stage_start = std::chrono::steady_clock::now();
        previous_scene_thumbnail_gray_.release();
        previous_scene_hash_valid_ = false;
        scene_stable_streak_       = 0;
        stability.valid            = false;
        stability.change_reason    = "disabled";
        result.scene_stability_ms = elapsed_ms_since(stage_start);
        status_.last_scene_stability_ms = result.scene_stability_ms;
    }

    status_.last_scene_stability = stability;

    result.accepted          = true;
    result.raw_to_model      = raw_to_model;
    result.color_space_label = (config_.color_mode == PhysicalSignalColorMode::Gray)
                                   ? std::string("GRAY8_SRGB")
                                   : std::string("BGR8_SRGB");
    result.pipeline_summary  = status_.last_pipeline_summary;
    result.model_width       = out_model_bgr.cols;
    result.model_height      = out_model_bgr.rows;
    result.scene_stability   = stability;
    result.total_ms          = elapsed_ms_since(pass_start);
    status_.last_total_ms    = result.total_ms;
    return result;
}

}}} // namespace GRIM::Perception::Physical
