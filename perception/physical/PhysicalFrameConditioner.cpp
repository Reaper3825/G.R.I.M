#include "PhysicalFrameConditioner.hpp"

#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
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

const char* PhysicalSignalColorModeName(PhysicalSignalColorMode mode) {
    switch (mode) {
        case PhysicalSignalColorMode::Bgr:  return "BGR";
        case PhysicalSignalColorMode::Gray: return "Gray";
        case PhysicalSignalColorMode::Rgb:  return "RGB";
        case PhysicalSignalColorMode::Gbr:  return "GBR";
    }
    return "Invalid";
}

const char* PhysicalSignalColorSpaceLabel(PhysicalSignalColorMode mode) {
    switch (mode) {
        case PhysicalSignalColorMode::Bgr:  return "BGR8_SRGB";
        case PhysicalSignalColorMode::Gray: return "GRAY8_SRGB";
        case PhysicalSignalColorMode::Rgb:  return "RGB8_SRGB";
        case PhysicalSignalColorMode::Gbr:  return "GBR8_SRGB";
    }
    return "INVALID";
}

void ConvertConfiguredSignalToGray(const cv::Mat& signal,
                                   PhysicalSignalColorMode mode,
                                   cv::Mat& gray) {
    switch (mode) {
        case PhysicalSignalColorMode::Bgr:
            cv::cvtColor(signal, gray, cv::COLOR_BGR2GRAY);
            return;
        case PhysicalSignalColorMode::Gray:
            cv::extractChannel(signal, gray, 0);
            return;
        case PhysicalSignalColorMode::Rgb:
            cv::cvtColor(signal, gray, cv::COLOR_RGB2GRAY);
            return;
        case PhysicalSignalColorMode::Gbr: {
            cv::Mat bgr(signal.size(), signal.type());
            const int from_to[] = {1, 0, 0, 1, 2, 2}; // [G,B,R] -> [B,G,R]
            cv::mixChannels(&signal, 1, &bgr, 1, from_to, 3);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            return;
        }
    }
    throw std::runtime_error("ConvertConfiguredSignalToGray: invalid color mode");
}

}

PhysicalSignalConditioningConfig BuildDefaultPhysicalSignalConditioningConfig() {
    return PhysicalSignalConditioningConfig{};
}

PhysicalFrameConditioner::PhysicalFrameConditioner()
    : config_(BuildDefaultPhysicalSignalConditioningConfig()) {}

void PhysicalFrameConditioner::ValidatePhysicalSignalConditioningConfig(
    const PhysicalSignalConditioningConfig& cfg) {
    switch (cfg.color_mode) {
        case PhysicalSignalColorMode::Bgr:
        case PhysicalSignalColorMode::Gray:
        case PhysicalSignalColorMode::Rgb:
        case PhysicalSignalColorMode::Gbr:
            break;
        default:
            ThrowInvalidPhysicalSignalConditioningConfig(
                "color_mode is not a recognized PhysicalSignalColorMode");
    }
    if (cfg.output_width <= 0 || cfg.output_height <= 0) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "output size must be > 0, got "
            + std::to_string(cfg.output_width) + "x" + std::to_string(cfg.output_height));
    }
    if (cfg.denoise_strength < 0 || cfg.denoise_strength > 30) {
        ThrowInvalidPhysicalSignalConditioningConfig(
            "denoise_strength must be in [0,30], got " + std::to_string(cfg.denoise_strength));
    }
    ValidatePhysicalCameraExposureConfig(cfg.exposure);
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
    const bool color_mode_changed = cfg.color_mode != config_.color_mode;
    config_ = cfg;
    exposure_controller_.Configure(config_.exposure);
    // Every explicit conditioning update changes the model-input contract or
    // its pixel values. Ensure cadence-aware consumers sample it at least once.
    force_model_refresh_next_frame_ = true;
    if (geometry_changed || !cfg.enable_stabilization) {
        previous_gray_for_flow_.release();
    }
    if (geometry_changed || color_mode_changed) {
        // A color-layout change is model-significant even when its grayscale
        // scene hash is identical. Force the next published frame to be a
        // cadence miss so every model consumes the newly ordered channels.
        previous_scene_thumbnail_gray_.release();
        previous_scene_hash_64_ = 0;
        previous_scene_hash_valid_ = false;
        scene_stable_streak_ = 0;
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "ConfigurePhysicalSignalConditioning: resize="
                  + std::to_string(static_cast<int>(cfg.enable_resize))
                  + " out=" + std::to_string(cfg.output_width) + "x"
                  + std::to_string(cfg.output_height)
                  + " denoise=" + std::to_string(static_cast<int>(cfg.enable_denoise))
                  + " exposure=" + std::to_string(static_cast<int>(cfg.exposure.enable))
                  + " autoExp=" + std::to_string(static_cast<int>(cfg.exposure.automatic))
                  + " antiFlicker="
                  + std::to_string(static_cast<int>(cfg.exposure.anti_flicker_enable))
                  + " deblur=" + std::to_string(static_cast<int>(cfg.enable_deblur))
                  + " stabilize=" + std::to_string(static_cast<int>(cfg.enable_stabilization))
                  + " color=" + PhysicalSignalColorModeName(cfg.color_mode));
}

void PhysicalFrameConditioner::ResetPhysicalSignalConditioningToDefaults() {
    config_ = BuildDefaultPhysicalSignalConditioningConfig();
    exposure_controller_.Configure(config_.exposure);
    exposure_controller_.ResetTemporalState();
    previous_gray_for_flow_.release();
    previous_scene_thumbnail_gray_.release();
    previous_scene_hash_64_    = 0;
    previous_scene_hash_valid_ = false;
    scene_stable_streak_       = 0;
    force_model_refresh_next_frame_ = true;
    status_ = {};
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "ResetPhysicalSignalConditioningToDefaults: restored default pipeline settings");
}

void PhysicalFrameConditioner::ResetPhysicalSignalConditioningTemporalState() {
    exposure_controller_.ResetTemporalState();
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

void PhysicalFrameConditioner::UpdatePhysicalCameraExposureFeedback(
    bool configured,
    uint64_t apply_counter) {
    exposure_controller_.UpdateHardwareExposureFeedback(
        configured, apply_counter);
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

// Robustly estimate high-frequency sensor noise on a small luminance image.
// The median absolute residual is deliberately used instead of a standard
// deviation so that real edges and textured objects do not make the denoiser
// attack useful detail. Keeping the estimator at <=320 px wide also makes its
// cost effectively independent of the selected camera resolution.
double EstimateSensorNoiseSigmaFromBgr(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "EstimateSensorNoiseSigmaFromBgr: expected non-empty CV_8UC3 frame");
    }

    constexpr int kEstimateWidth = 320;
    constexpr int kEstimateHeight = 240;
    const int sample_width = std::min(bgr.cols, kEstimateWidth);
    const int sample_height = std::min(bgr.rows, kEstimateHeight);
    const int sample_x = (bgr.cols - sample_width) / 2;
    const int sample_y = (bgr.rows - sample_height) / 2;
    const cv::Mat sample = bgr(cv::Rect(
        sample_x, sample_y, sample_width, sample_height));

    cv::Mat gray;
    cv::cvtColor(sample, gray, cv::COLOR_BGR2GRAY);
    cv::Mat low_pass;
    cv::GaussianBlur(gray, low_pass, cv::Size(3, 3), 0.8, 0.8,
                     cv::BORDER_REPLICATE);
    cv::Mat residual;
    cv::absdiff(gray, low_pass, residual);

    std::array<uint32_t, 256> histogram{};
    for (int y = 0; y < residual.rows; ++y) {
        const uint8_t* row = residual.ptr<uint8_t>(y);
        for (int x = 0; x < residual.cols; ++x) {
            ++histogram[row[x]];
        }
    }

    const uint64_t sample_count = residual.total();
    const uint64_t median_rank = sample_count / 2u;
    uint64_t cumulative = 0;
    int median_residual = 0;
    for (int value = 0; value < 256; ++value) {
        cumulative += histogram[static_cast<size_t>(value)];
        if (cumulative > median_rank) {
            median_residual = value;
            break;
        }
    }

    // For zero-mean Gaussian noise, median(abs(x)) = 0.67449 * sigma.
    constexpr double kGaussianMedianAbsoluteDeviation = 0.6744897501960817;
    return static_cast<double>(median_residual)
         / kGaussianMedianAbsoluteDeviation;
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

PhysicalSignalConditioningResult PhysicalFrameConditioner::ProcessCalibratedFrameToModelSignal(
    const cv::Mat& calibrated_bgr,
    uint64_t       frame_counter,
    cv::Mat&       out_model_image) {
    const auto pass_start = std::chrono::steady_clock::now();
    auto elapsed_ms_since = [](const auto& start) -> double {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };

    if (calibrated_bgr.empty()) {
        throw std::runtime_error(
            "ProcessCalibratedFrameToModelSignal: calibrated_bgr is empty");
    }
    if (calibrated_bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "ProcessCalibratedFrameToModelSignal: expected calibrated_bgr type CV_8UC3, got "
            + std::to_string(calibrated_bgr.type()));
    }

    ValidatePhysicalSignalConditioningConfig(config_);

    PhysicalSignalConditioningResult result;
    result.raw_width  = calibrated_bgr.cols;
    result.raw_height = calibrated_bgr.rows;

    status_.processed_frame_counter = frame_counter;
    status_.last_input_width        = calibrated_bgr.cols;
    status_.last_input_height       = calibrated_bgr.rows;
    status_.last_input_luma         = ComputeMeanLumaFromBgr(calibrated_bgr);
    status_.last_failure_reason.clear();
    status_.last_applied_exposure_gain = 1.0;
    status_.last_exposure_low_luma = 0.0;
    status_.last_exposure_meter_luma = 0.0;
    status_.last_exposure_highlight_luma = 0.0;
    status_.last_desired_exposure_gain = 1.0;
    status_.last_flicker_detected = false;
    status_.last_flicker_amplitude = 0.0;
    status_.last_anti_flicker_gain = 1.0;
    status_.last_motion_magnitude = 0.0;
    status_.last_motion_priority = 0.0;
    status_.last_hardware_gain_demand = 0.0;
    status_.last_estimated_noise_sigma = 0.0;
    status_.last_applied_denoise_sigma = 0.0;
    status_.using_auto_exposure        = config_.exposure.automatic;
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

    // ---- Quality gate (computed on calibrated input — drop before wasting work) ----
    auto stage_start = std::chrono::steady_clock::now();
    const double clipped_ratio = ComputeClippedPixelRatio(calibrated_bgr);
    const double lap_var       = ComputeLaplacianVarianceFromBgr(calibrated_bgr);
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

    cv::Mat working = calibrated_bgr;
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
        const double noise_sigma = EstimateSensorNoiseSigmaFromBgr(working);
        status_.last_estimated_noise_sigma = noise_sigma;

        // `denoise_strength` is an operator-set ceiling, not a fixed kernel.
        // Clean frames receive little or no filtering; noisier frames approach
        // the configured ceiling. Bilateral filtering smooths similar pixels
        // while refusing to cross strong intensity/color edges.
        constexpr double kNoiseFloorSigma = 0.75;
        if (noise_sigma > kNoiseFloorSigma) {
            const double aggressiveness =
                static_cast<double>(config_.denoise_strength) / 30.0;
            const double noise_demand = std::clamp(
                (noise_sigma - kNoiseFloorSigma) / 8.0, 0.0, 1.0);
            const double applied_fraction = aggressiveness
                                          * (0.25 + 0.75 * noise_demand);
            const double sigma_color = 4.0 + 22.0 * applied_fraction;
            const double sigma_space = 1.25 + 2.25 * aggressiveness;
            cv::Mat denoised;
            cv::bilateralFilter(working, denoised, 0, sigma_color, sigma_space,
                                cv::BORDER_REPLICATE);
            working = denoised;
            status_.last_applied_denoise_sigma = sigma_color;
        }
        pipeline << "adaptive_edge_denoise(noise=" << noise_sigma
                 << ",sigma=" << status_.last_applied_denoise_sigma << ") ";
        result.denoise_ms = elapsed_ms_since(stage_start);
        status_.last_denoise_ms = result.denoise_ms;
    }

    if (config_.exposure.enable) {
        stage_start = std::chrono::steady_clock::now();
        cv::Mat corrected;
        const PhysicalCameraExposureStatus exposure_status =
            exposure_controller_.ProcessFrame(working, corrected);
        working = corrected;
        status_.last_applied_exposure_gain = exposure_status.applied_gain;
        status_.last_exposure_low_luma = exposure_status.low_luma;
        status_.last_exposure_meter_luma = exposure_status.metered_luma;
        status_.last_exposure_highlight_luma = exposure_status.highlight_luma;
        status_.last_desired_exposure_gain = exposure_status.desired_gain;
        status_.last_flicker_detected = exposure_status.flicker_detected;
        status_.last_flicker_amplitude = exposure_status.flicker_amplitude;
        status_.last_anti_flicker_gain = exposure_status.anti_flicker_gain;
        status_.last_motion_magnitude = exposure_status.motion_magnitude;
        status_.last_motion_priority = exposure_status.motion_priority;
        status_.last_hardware_gain_demand = exposure_status.hardware_gain_demand;
        status_.last_hardware_exposure_active =
            exposure_status.hardware_exposure_active;
        status_.last_hardware_request_pending =
            exposure_status.hardware_request_pending;
        status_.last_hardware_settling = exposure_status.hardware_settling;
        status_.last_hardware_settle_frames_remaining =
            exposure_status.hardware_settle_frames_remaining;
        result.motion_exposure_request = exposure_status.motion_request;
        pipeline << "adaptive_exposure(" << exposure_status.summary << ") ";
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
                                  / static_cast<double>(calibrated_bgr.cols);
            raw_to_model.scale_y  = static_cast<double>(config_.output_height)
                                  / static_cast<double>(calibrated_bgr.rows);
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

    stage_start = std::chrono::steady_clock::now();
    switch (config_.color_mode) {
        case PhysicalSignalColorMode::Bgr:
            pipeline << "bgr ";
            break;
        case PhysicalSignalColorMode::Gray: {
            cv::Mat gray;
            cv::cvtColor(working, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, working, cv::COLOR_GRAY2BGR);
            pipeline << "gray-replicated ";
            break;
        }
        case PhysicalSignalColorMode::Rgb: {
            cv::Mat rgb;
            cv::cvtColor(working, rgb, cv::COLOR_BGR2RGB);
            working = rgb;
            pipeline << "rgb ";
            break;
        }
        case PhysicalSignalColorMode::Gbr: {
            cv::Mat gbr(working.size(), working.type());
            const int from_to[] = {1, 0, 0, 1, 2, 2}; // [B,G,R] -> [G,B,R]
            cv::mixChannels(&working, 1, &gbr, 1, from_to, 3);
            working = gbr;
            pipeline << "gbr ";
            break;
        }
    }
    result.color_convert_ms = elapsed_ms_since(stage_start);
    status_.last_color_convert_ms = result.color_convert_ms;

    if (working.empty() || working.type() != CV_8UC3) {
        throw std::runtime_error(
            "ProcessCalibratedFrameToModelSignal: pipeline produced invalid output frame");
    }

    working.copyTo(out_model_image);
    status_.last_output_width   = out_model_image.cols;
    status_.last_output_height  = out_model_image.rows;
    cv::Mat output_gray;
    ConvertConfiguredSignalToGray(out_model_image, config_.color_mode, output_gray);
    status_.last_output_luma    = cv::mean(output_gray)[0];
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
                "ProcessCalibratedFrameToModelSignal: scene_stability thumbnail must be"
                " at least 8x8, got "
                + std::to_string(sc.thumbnail_width) + "x"
                + std::to_string(sc.thumbnail_height));
        }
        cv::Mat thumb_bgr;
        cv::resize(out_model_image,
                   thumb_bgr,
                   cv::Size(sc.thumbnail_width, sc.thumbnail_height),
                   0.0, 0.0, cv::INTER_AREA);
        cv::Mat thumb_gray;
        ConvertConfiguredSignalToGray(thumb_bgr, config_.color_mode, thumb_gray);

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

    if (force_model_refresh_next_frame_) {
        // `valid=false` is the established signal for every cadence-aware
        // consumer to bypass both stable-scene reuse and its minimum-period
        // gate. This guarantees one fresh inference using the new layout.
        stability.valid = false;
        stability.is_stable = false;
        stability.change_reason = "conditioning_config_changed";
        force_model_refresh_next_frame_ = false;
    }

    status_.last_scene_stability = stability;

    result.accepted          = true;
    result.raw_to_model      = raw_to_model;
    result.color_space_label = PhysicalSignalColorSpaceLabel(config_.color_mode);
    result.pipeline_summary  = status_.last_pipeline_summary;
    result.model_width       = out_model_image.cols;
    result.model_height      = out_model_image.rows;
    result.scene_stability   = stability;
    result.total_ms          = elapsed_ms_since(pass_start);
    status_.last_total_ms    = result.total_ms;
    return result;
}

}}} // namespace GRIM::Perception::Physical
