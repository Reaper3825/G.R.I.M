#include "PhysicalCameraExposureController.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

[[noreturn]] void ThrowInvalidExposureConfig(const std::string& reason) {
    throw std::runtime_error(
        "PhysicalCameraExposure config invalid: " + reason);
}

struct LumaPercentiles {
    double low = 0.0;
    double meter = 0.0;
    double highlight = 0.0;
};

double HistogramPercentile(const std::array<uint32_t, 256>& histogram,
                           uint64_t sample_count,
                           double percentile) {
    const uint64_t rank = static_cast<uint64_t>(std::floor(
        percentile * static_cast<double>(sample_count - 1u)));
    uint64_t cumulative = 0;
    for (int value = 0; value < 256; ++value) {
        cumulative += histogram[static_cast<size_t>(value)];
        if (cumulative > rank) return static_cast<double>(value);
    }
    return 255.0;
}

LumaPercentiles ComputeLumaPercentiles(
    const cv::Mat& bgr,
    const PhysicalCameraExposureConfig& config) {
    cv::Mat sample = bgr;
    constexpr int kMeterWidth = 320;
    if (bgr.cols > kMeterWidth) {
        const double scale = static_cast<double>(kMeterWidth)
                           / static_cast<double>(bgr.cols);
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size{}, scale, scale, cv::INTER_AREA);
        sample = resized;
    }

    cv::Mat gray;
    cv::cvtColor(sample, gray, cv::COLOR_BGR2GRAY);
    std::array<uint32_t, 256> histogram{};
    for (int y = 0; y < gray.rows; ++y) {
        const uint8_t* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) {
            ++histogram[row[x]];
        }
    }

    const uint64_t sample_count = static_cast<uint64_t>(gray.total());
    return {
        HistogramPercentile(histogram, sample_count, config.low_percentile),
        HistogramPercentile(histogram, sample_count, config.meter_percentile),
        HistogramPercentile(histogram, sample_count, config.highlight_percentile)
    };
}

double ComputeMotionMagnitude(const cv::Mat& bgr, cv::Mat& previous_gray) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat motion_gray;
    cv::resize(gray, motion_gray, cv::Size(160, 90), 0.0, 0.0,
               cv::INTER_AREA);
    gray = motion_gray;

    double magnitude = 0.0;
    if (!previous_gray.empty() && previous_gray.size() == gray.size()) {
        cv::Mat current_float;
        cv::Mat previous_float;
        gray.convertTo(current_float, CV_32F);
        previous_gray.convertTo(previous_float, CV_32F);
        current_float -= cv::mean(current_float)[0];
        previous_float -= cv::mean(previous_float)[0];
        cv::Mat difference;
        cv::absdiff(current_float, previous_float, difference);
        magnitude = cv::mean(difference)[0];
    }
    gray.copyTo(previous_gray);
    return magnitude;
}

double Median(std::deque<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2u;
    if ((values.size() & 1u) != 0u) return values[middle];
    return 0.5 * (values[middle - 1u] + values[middle]);
}

struct FlickerObservation {
    bool detected = false;
    double baseline = 0.0;
    double amplitude = 0.0;
};

FlickerObservation ObserveFlicker(
    const std::deque<double>& history,
    const PhysicalCameraExposureConfig& config) {
    FlickerObservation observation;
    if (history.size() < 6u) return observation;

    observation.baseline = Median(history);
    int alternating_steps = 0;
    int comparable_steps = 0;
    double absolute_step_sum = 0.0;
    double previous_step = 0.0;
    bool have_previous_step = false;
    for (size_t i = 1; i < history.size(); ++i) {
        const double step = history[i] - history[i - 1u];
        absolute_step_sum += std::abs(step);
        if (have_previous_step
            && std::abs(step) >= config.anti_flicker_min_amplitude
            && std::abs(previous_step) >= config.anti_flicker_min_amplitude) {
            ++comparable_steps;
            if ((step > 0.0) != (previous_step > 0.0)) {
                ++alternating_steps;
            }
        }
        previous_step = step;
        have_previous_step = true;
    }

    const double average_step = absolute_step_sum
        / static_cast<double>(history.size() - 1u);
    const double net_drift = std::abs(history.back() - history.front());
    const double alternating_ratio = comparable_steps > 0
        ? static_cast<double>(alternating_steps)
            / static_cast<double>(comparable_steps)
        : 0.0;

    observation.amplitude = average_step;
    observation.detected = comparable_steps >= 3
        && alternating_ratio >= 0.65
        && average_step >= config.anti_flicker_min_amplitude
        && net_drift <= average_step * 1.5;
    return observation;
}

} // namespace

void ValidatePhysicalCameraExposureConfig(
    const PhysicalCameraExposureConfig& config) {
    auto finite = [](double value) { return std::isfinite(value); };
    if (!finite(config.manual_gain) || config.manual_gain <= 0.0) {
        ThrowInvalidExposureConfig("manual_gain must be finite and > 0");
    }
    if (!finite(config.target_luma)
        || config.target_luma <= 0.0 || config.target_luma > 255.0) {
        ThrowInvalidExposureConfig("target_luma must be finite and in (0,255]");
    }
    if (!finite(config.low_percentile)
        || !finite(config.meter_percentile)
        || !finite(config.highlight_percentile)
        || config.low_percentile < 0.0
        || config.low_percentile >= config.meter_percentile
        || config.meter_percentile >= config.highlight_percentile
        || config.highlight_percentile > 1.0) {
        ThrowInvalidExposureConfig(
            "percentiles must satisfy 0 <= low < meter < highlight <= 1");
    }
    if (!finite(config.highlight_ceiling)
        || config.highlight_ceiling <= 0.0 || config.highlight_ceiling > 255.0) {
        ThrowInvalidExposureConfig(
            "highlight_ceiling must be finite and in (0,255]");
    }
    if (!finite(config.minimum_gain) || !finite(config.maximum_gain)
        || config.minimum_gain <= 0.0
        || config.minimum_gain > config.maximum_gain) {
        ThrowInvalidExposureConfig(
            "gain bounds must be finite and satisfy 0 < minimum <= maximum");
    }
    auto validate_response = [&](double value, const char* name) {
        if (!finite(value) || value <= 0.0 || value > 1.0) {
            ThrowInvalidExposureConfig(
                std::string(name) + " must be finite and in (0,1]");
        }
    };
    validate_response(config.meter_response, "meter_response");
    validate_response(config.brighten_response, "brighten_response");
    validate_response(config.darken_response, "darken_response");
    if (!finite(config.gain_hysteresis_ratio)
        || config.gain_hysteresis_ratio < 0.0
        || config.gain_hysteresis_ratio > 0.25) {
        ThrowInvalidExposureConfig(
            "gain_hysteresis_ratio must be finite and in [0,0.25]");
    }
    if (config.anti_flicker_window_frames < 6u
        || config.anti_flicker_window_frames > 120u) {
        ThrowInvalidExposureConfig(
            "anti_flicker_window_frames must be in [6,120]");
    }
    if (!finite(config.anti_flicker_min_amplitude)
        || config.anti_flicker_min_amplitude < 0.0) {
        ThrowInvalidExposureConfig(
            "anti_flicker_min_amplitude must be finite and >= 0");
    }
    if (!finite(config.anti_flicker_strength)
        || config.anti_flicker_strength < 0.0
        || config.anti_flicker_strength > 1.0) {
        ThrowInvalidExposureConfig(
            "anti_flicker_strength must be finite and in [0,1]");
    }
    if (!finite(config.anti_flicker_max_correction)
        || config.anti_flicker_max_correction < 0.0
        || config.anti_flicker_max_correction > 0.50) {
        ThrowInvalidExposureConfig(
            "anti_flicker_max_correction must be finite and in [0,0.50]");
    }
    if (!finite(config.motion_low_threshold)
        || !finite(config.motion_high_threshold)
        || config.motion_low_threshold < 0.0
        || config.motion_low_threshold >= config.motion_high_threshold) {
        ThrowInvalidExposureConfig(
            "motion thresholds must be finite and satisfy 0 <= low < high");
    }
    if (!finite(config.motion_gain_compensation)
        || config.motion_gain_compensation < 0.0
        || config.motion_gain_compensation > 1.0) {
        ThrowInvalidExposureConfig(
            "motion_gain_compensation must be finite and in [0,1]");
    }
    if (!finite(config.motion_response)
        || config.motion_response <= 0.0 || config.motion_response > 1.0) {
        ThrowInvalidExposureConfig(
            "motion_response must be finite and in (0,1]");
    }
    if (!finite(config.motion_hysteresis)
        || config.motion_hysteresis < 0.0
        || config.motion_hysteresis > 0.50) {
        ThrowInvalidExposureConfig(
            "motion_hysteresis must be finite and in [0,0.50]");
    }
    validate_response(config.hardware_brighten_response,
                      "hardware_brighten_response");
    validate_response(config.hardware_darken_response,
                      "hardware_darken_response");
    if (!finite(config.residual_gain_minimum)
        || !finite(config.residual_gain_maximum)
        || config.residual_gain_minimum <= 0.0
        || config.residual_gain_minimum > 1.0
        || config.residual_gain_maximum < 1.0
        || config.residual_gain_minimum > config.residual_gain_maximum) {
        ThrowInvalidExposureConfig(
            "residual gain bounds must satisfy 0 < minimum <= 1 <= maximum");
    }
    if (config.hardware_settle_frames > 120u) {
        ThrowInvalidExposureConfig("hardware_settle_frames must be <= 120");
    }
}

PhysicalCameraExposureController::PhysicalCameraExposureController() {
    ValidatePhysicalCameraExposureConfig(config_);
}

void PhysicalCameraExposureController::Configure(
    const PhysicalCameraExposureConfig& config) {
    ValidatePhysicalCameraExposureConfig(config);
    const bool temporal_contract_changed =
        config.enable != config_.enable
        || config.automatic != config_.automatic
        || config.target_luma != config_.target_luma
        || config.anti_flicker_enable != config_.anti_flicker_enable
        || config.anti_flicker_window_frames
            != config_.anti_flicker_window_frames;
    config_ = config;
    if (temporal_contract_changed) ResetTemporalState();
}

void PhysicalCameraExposureController::ResetTemporalState() {
    meter_history_.clear();
    temporal_initialized_ = false;
    temporal_meter_ = 0.0;
    base_gain_ = 1.0;
    hardware_brightness_demand_ = 0.0;
    motion_priority_ = 0.0;
    last_hardware_apply_counter_ = 0;
    hardware_settle_frames_remaining_ = 0;
    hardware_request_pending_ = false;
    last_requested_motion_priority_ = 0.0;
    last_requested_gain_demand_ = 0.0;
    telemetry_frame_counter_ = 0;
    previous_motion_gray_.release();
    status_ = {};
}

void PhysicalCameraExposureController::UpdateHardwareExposureFeedback(
    bool configured,
    uint64_t apply_counter) {
    hardware_exposure_available_ = configured;
    if (!configured) {
        last_hardware_apply_counter_ = 0;
        hardware_settle_frames_remaining_ = 0;
        hardware_brightness_demand_ = 0.0;
        hardware_request_pending_ = false;
        last_requested_motion_priority_ = 0.0;
        last_requested_gain_demand_ = 0.0;
        return;
    }
    if (apply_counter != 0 && apply_counter != last_hardware_apply_counter_) {
        last_hardware_apply_counter_ = apply_counter;
        hardware_request_pending_ = false;
        hardware_settle_frames_remaining_ = config_.hardware_settle_frames;
        // Samples spanning a sensor-property transition are not evidence of
        // mains flicker. Start a fresh window after the camera has settled.
        meter_history_.clear();
    }
}

PhysicalCameraExposureStatus PhysicalCameraExposureController::ProcessFrame(
    const cv::Mat& input_bgr,
    cv::Mat& output_bgr) {
    if (input_bgr.empty() || input_bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "PhysicalCameraExposureController::ProcessFrame expected non-empty CV_8UC3 input");
    }

    PhysicalCameraExposureStatus next;
    next.enabled = config_.enable;
    next.automatic = config_.automatic;
    next.hardware_exposure_active = hardware_exposure_available_
        && config_.motion_aware_enable;
    next.hardware_request_pending = next.hardware_exposure_active
        && hardware_request_pending_;
    next.hardware_settling = next.hardware_exposure_active
        && hardware_settle_frames_remaining_ > 0;
    next.hardware_settle_frames_remaining =
        hardware_settle_frames_remaining_;
    if (!config_.enable) {
        input_bgr.copyTo(output_bgr);
        next.summary = "disabled";
        status_ = next;
        return status_;
    }

    const LumaPercentiles luma = ComputeLumaPercentiles(input_bgr, config_);
    next.low_luma = luma.low;
    next.metered_luma = luma.meter;
    next.highlight_luma = luma.highlight;
    next.motion_magnitude = ComputeMotionMagnitude(
        input_bgr, previous_motion_gray_);
    const bool hardware_control_held = next.hardware_exposure_active
        && (next.hardware_settling || next.hardware_request_pending);
    if (config_.motion_aware_enable) {
        if (!hardware_control_held) {
            const double observed_motion_priority = std::clamp(
                (next.motion_magnitude - config_.motion_low_threshold)
                    / (config_.motion_high_threshold
                       - config_.motion_low_threshold),
                0.0, 1.0);
            if (std::abs(observed_motion_priority - motion_priority_)
                > config_.motion_hysteresis) {
                motion_priority_ += config_.motion_response
                                  * (observed_motion_priority - motion_priority_);
            }
        }
        next.motion_priority = motion_priority_;
    } else {
        motion_priority_ = 0.0;
    }

    if (!hardware_control_held) {
        meter_history_.push_back(luma.meter);
        while (meter_history_.size() > config_.anti_flicker_window_frames) {
            meter_history_.pop_front();
        }
    }
    const FlickerObservation flicker = config_.anti_flicker_enable
        && !hardware_control_held
        ? ObserveFlicker(meter_history_, config_)
        : FlickerObservation{};
    next.flicker_detected = flicker.detected;
    next.flicker_amplitude = flicker.amplitude;

    if (!temporal_initialized_) {
        temporal_meter_ = std::max(1.0, luma.meter);
        base_gain_ = 1.0;
        temporal_initialized_ = true;
    } else if (!hardware_control_held) {
        const double relative_change = std::abs(luma.meter - temporal_meter_)
                                     / std::max(1.0, temporal_meter_);
        const double meter_alpha = relative_change > 0.25
            ? 0.65 : config_.meter_response;
        const double meter_sample = flicker.detected
            ? flicker.baseline : luma.meter;
        temporal_meter_ += meter_alpha * (meter_sample - temporal_meter_);
    }
    next.temporally_metered_luma = temporal_meter_;

    double desired_gain = config_.manual_gain;
    if (config_.automatic) {
        desired_gain = config_.target_luma / std::max(1.0, temporal_meter_);
        const double highlight_safe_gain = config_.highlight_ceiling
                                         / std::max(1.0, luma.highlight);
        desired_gain = std::min(desired_gain, highlight_safe_gain);
        desired_gain = std::clamp(
            desired_gain, config_.minimum_gain, config_.maximum_gain);

        const double software_target = next.hardware_exposure_active
            ? std::clamp(desired_gain,
                         config_.residual_gain_minimum,
                         config_.residual_gain_maximum)
            : desired_gain;
        const double relative_gain_error = std::abs(software_target - base_gain_)
                                         / std::max(0.01, base_gain_);
        if (!hardware_control_held
            && relative_gain_error > config_.gain_hysteresis_ratio) {
            const double alpha = software_target < base_gain_
                ? config_.darken_response : config_.brighten_response;
            base_gain_ += alpha * (software_target - base_gain_);
        }
    } else {
        base_gain_ = std::clamp(
            config_.manual_gain, config_.minimum_gain, config_.maximum_gain);
    }
    next.desired_gain = desired_gain;
    next.base_gain = base_gain_;

    double anti_flicker_gain = 1.0;
    if (flicker.detected && luma.meter > 0.0) {
        const double raw_correction = flicker.baseline / luma.meter;
        const double limited_correction = std::clamp(
            raw_correction,
            1.0 - config_.anti_flicker_max_correction,
            1.0 + config_.anti_flicker_max_correction);
        anti_flicker_gain = 1.0 + config_.anti_flicker_strength
                                  * (limited_correction - 1.0);
    }
    next.anti_flicker_gain = anti_flicker_gain;
    const double applied_gain_minimum =
        next.hardware_exposure_active && config_.automatic
            ? config_.residual_gain_minimum
            : config_.minimum_gain;
    const double applied_gain_maximum =
        next.hardware_exposure_active && config_.automatic
            ? config_.residual_gain_maximum
            : config_.maximum_gain;
    next.applied_gain = std::clamp(
        base_gain_ * anti_flicker_gain,
        applied_gain_minimum,
        applied_gain_maximum);

    if (next.hardware_exposure_active && config_.automatic
        && !hardware_control_held) {
        const double normalized_error = std::clamp(
            (config_.target_luma - temporal_meter_)
                / std::max(1.0, config_.target_luma),
            -1.0, 1.0);
        if (std::abs(normalized_error) > config_.gain_hysteresis_ratio) {
            const double response = normalized_error > 0.0
                ? config_.hardware_brighten_response
                : config_.hardware_darken_response;
            hardware_brightness_demand_ = std::clamp(
                hardware_brightness_demand_ + response * normalized_error,
                0.0, 1.0);
        }
    }
    next.hardware_gain_demand = next.hardware_exposure_active
        ? std::clamp(hardware_brightness_demand_
                + next.motion_priority * config_.motion_gain_compensation,
            0.0, 1.0)
        : 0.0;
    // A valid zeroed request when disabled returns explicitly configured
    // hardware to its slow-exposure/minimum-gain endpoint.
    next.motion_request.valid = true;
    next.motion_request.motion_priority = next.motion_priority;
    next.motion_request.gain_demand = next.hardware_gain_demand;
    constexpr double kHardwareRequestDeadband = 0.02;
    if (next.hardware_exposure_active && !hardware_control_held
        && (std::abs(next.motion_request.motion_priority
                     - last_requested_motion_priority_)
                >= kHardwareRequestDeadband
            || std::abs(next.motion_request.gain_demand
                        - last_requested_gain_demand_)
                >= kHardwareRequestDeadband)) {
        last_requested_motion_priority_ =
            next.motion_request.motion_priority;
        last_requested_gain_demand_ = next.motion_request.gain_demand;
        hardware_request_pending_ = true;
        next.hardware_request_pending = true;
    }

    if (hardware_settle_frames_remaining_ > 0) {
        --hardware_settle_frames_remaining_;
    }

    input_bgr.convertTo(output_bgr, -1, next.applied_gain, 0.0);
    std::ostringstream summary;
    summary << (config_.automatic ? "auto" : "manual")
            << " meter=" << next.metered_luma
            << " hi=" << next.highlight_luma
            << " desired=" << next.desired_gain
            << " applied=" << next.applied_gain
            << " motion=" << next.motion_priority
            << " hwGain=" << next.hardware_gain_demand;
    if (next.hardware_settling) {
        summary << " settling=" << next.hardware_settle_frames_remaining;
    }
    if (next.hardware_request_pending) {
        summary << " pending";
    }
    if (next.flicker_detected) {
        summary << " anti-flicker=" << next.anti_flicker_gain;
    }
    next.summary = summary.str();
    ++telemetry_frame_counter_;
    if (telemetry_frame_counter_ == 1u
        || telemetry_frame_counter_ % 30u == 0u) {
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "Exposure telemetry: raw_p50="
                  + std::to_string(next.metered_luma)
                  + " temporal="
                  + std::to_string(next.temporally_metered_luma)
                  + " digital_gain=" + std::to_string(next.applied_gain)
                  + " motion=" + std::to_string(next.motion_priority)
                  + " hardware_gain_demand="
                  + std::to_string(next.hardware_gain_demand)
                  + " settling="
                  + std::to_string(next.hardware_settle_frames_remaining)
                  + " pending="
                  + std::to_string(next.hardware_request_pending ? 1 : 0)
                  + " flicker="
                  + std::to_string(next.flicker_detected ? 1 : 0));
    }
    status_ = next;
    return status_;
}

PhysicalCameraExposureConfig
PhysicalCameraExposureController::GetConfigSnapshot() const {
    return config_;
}

PhysicalCameraExposureStatus
PhysicalCameraExposureController::GetStatusSnapshot() const {
    return status_;
}

}}} // namespace GRIM::Perception::Physical
