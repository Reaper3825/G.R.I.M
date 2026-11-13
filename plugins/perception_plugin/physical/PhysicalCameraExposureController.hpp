#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// Signal-level exposure controller used by PhysicalFrameConditioner. Camera
// drivers remain free to perform sensor exposure, but this controller replaces
// the former independent per-frame mean-luma multiplier with robust metering,
// temporal adaptation, highlight protection, and global flicker compensation.
struct PhysicalCameraExposureConfig {
    bool   enable                  = true;
    bool   automatic               = true;
    double manual_gain             = 1.0;
    double target_luma             = 112.0;

    double low_percentile          = 0.10;
    double meter_percentile        = 0.50;
    double highlight_percentile    = 0.98;
    double highlight_ceiling       = 245.0;
    double minimum_gain            = 0.50;
    double maximum_gain            = 2.50;

    // Response values are fractions in (0,1]. Darkening is intentionally
    // faster than brightening so newly exposed highlights are protected while
    // shadow noise is not boosted abruptly.
    double meter_response          = 0.15;
    double brighten_response       = 0.18;
    double darken_response         = 0.40;
    double gain_hysteresis_ratio   = 0.015;

    // This compensates frame-wide illumination oscillation. It cannot remove
    // rolling horizontal bands already integrated by the sensor; that needs a
    // backend-specific hardware power-line-frequency control.
    bool        anti_flicker_enable         = true;
    std::size_t anti_flicker_window_frames  = 8;
    double      anti_flicker_min_amplitude  = 1.5;
    double      anti_flicker_strength       = 0.75;
    double      anti_flicker_max_correction = 0.15;

    bool   motion_aware_enable          = true;
    double motion_low_threshold         = 2.0;
    double motion_high_threshold        = 18.0;
    double motion_response              = 0.35;
    double motion_hysteresis            = 0.05;
    double motion_gain_compensation     = 0.50;

    // When native shutter/gain control is available it owns the slow
    // brightness loop. Digital exposure is then only a bounded residual trim
    // so the same luma error is not corrected twice.
    double hardware_brighten_response   = 0.06;
    double hardware_darken_response     = 0.12;
    double residual_gain_minimum        = 0.90;
    double residual_gain_maximum        = 1.10;
    std::size_t hardware_settle_frames  = 8;
};

// Normalized policy output. The capture worker maps these values onto raw
// backend-specific CAP_PROP_EXPOSURE/CAP_PROP_GAIN ranges when compatible
// native discovery succeeds or the local-device URL supplies an override.
struct PhysicalCameraMotionExposureRequest {
    bool   valid            = false;
    double motion_priority  = 0.0; // 0=long/clean, 1=short/freeze motion
    double gain_demand      = 0.0; // 0=min configured gain, 1=max
};

struct PhysicalCameraExposureStatus {
    bool   enabled                  = true;
    bool   automatic               = true;
    double low_luma                = 0.0;
    double metered_luma            = 0.0;
    double highlight_luma          = 0.0;
    double temporally_metered_luma = 0.0;
    double desired_gain            = 1.0;
    double base_gain               = 1.0;
    double anti_flicker_gain       = 1.0;
    double applied_gain            = 1.0;
    bool   flicker_detected        = false;
    double flicker_amplitude       = 0.0;
    double motion_magnitude        = 0.0;
    double motion_priority         = 0.0;
    double hardware_gain_demand    = 0.0;
    bool   hardware_exposure_active = false;
    bool   hardware_request_pending = false;
    bool   hardware_settling        = false;
    std::size_t hardware_settle_frames_remaining = 0;
    PhysicalCameraMotionExposureRequest motion_request{};
    std::string summary;
};

void ValidatePhysicalCameraExposureConfig(
    const PhysicalCameraExposureConfig& config);

class PhysicalCameraExposureController {
public:
    PhysicalCameraExposureController();

    void Configure(const PhysicalCameraExposureConfig& config);
    void ResetTemporalState();
    void UpdateHardwareExposureFeedback(bool configured,
                                        uint64_t apply_counter);

    PhysicalCameraExposureStatus ProcessFrame(
        const cv::Mat& input_bgr,
        cv::Mat& output_bgr);

    PhysicalCameraExposureConfig GetConfigSnapshot() const;
    PhysicalCameraExposureStatus GetStatusSnapshot() const;

private:
    PhysicalCameraExposureConfig config_{};
    PhysicalCameraExposureStatus status_{};
    std::deque<double> meter_history_;
    bool   temporal_initialized_ = false;
    double temporal_meter_ = 0.0;
    double base_gain_ = 1.0;
    double hardware_brightness_demand_ = 0.0;
    double motion_priority_ = 0.0;
    bool hardware_exposure_available_ = false;
    uint64_t last_hardware_apply_counter_ = 0;
    std::size_t hardware_settle_frames_remaining_ = 0;
    bool hardware_request_pending_ = false;
    double last_requested_motion_priority_ = 0.0;
    double last_requested_gain_demand_ = 0.0;
    uint64_t telemetry_frame_counter_ = 0;
    cv::Mat previous_motion_gray_;
};

}}} // namespace GRIM::Perception::Physical
