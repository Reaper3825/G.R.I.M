#pragma once

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

#include "PhysicalSceneStability.hpp"

namespace GRIM { namespace Perception { namespace Physical {

enum class PhysicalSignalColorMode : uint8_t {
    Bgr = 0,
    Gray = 1
};

enum class PhysicalSignalResizeMode : uint8_t {
    Stretch   = 0,   // direct cv::resize (current behavior; distorts aspect)
    Letterbox = 1    // preserve aspect, pad with constant color
};

// Why a quality gate? A vision model wastes compute and gets corrupted
// temporal context if we forward black startup frames, completely saturated
// frames (lens cap pop-off), or motion-smear frames during shake. Better to
// drop them at the producer with an explicit logged reason.
struct PhysicalSignalQualityGateConfig {
    bool   enable                      = true;
    double min_mean_luma               = 4.0;     // < this = "black frame"
    double max_mean_luma               = 252.0;   // > this = "blown out"
    double max_clipped_pixel_ratio     = 0.40;    // fraction of pixels at 0 or 255
    double min_laplacian_variance      = 6.0;     // < this = "too blurry"
};

struct PhysicalSignalConditioningConfig {
    bool   enable_resize               = true;
    int    output_width                = 640;
    int    output_height               = 360;
    PhysicalSignalResizeMode resize_mode = PhysicalSignalResizeMode::Letterbox;
    int    letterbox_pad_value         = 114;     // 0..255; 114 matches YOLO/Ultralytics convention

    bool   enable_denoise              = true; // median blur is cheap and helps stabilize the scene-stability signal (and downstream cache reuse) when sensor noise is high — e.g. in low light or with aggressive digital zoom.
    int    denoise_strength            = 6; // median blur kernel size. Must be odd. 5 is a good default for 640x360 input; adjust up/down for larger/smaller resolutions.
    bool   enable_exposure_correction  = true; // master switch for the exposure-correction subsystem (auto and manual paths both gated on this).
    bool   exposure_auto               = true; // when true, overrides manual_exposure_gain and automatically computes a gain to push the mean luma towards target_luma.
    double manual_exposure_gain        = 1.00; // applied when exposure_auto is false. Multiplier on the input pixel values; 1.0 means no change, <1.0 darkens, >1.0 brightens.
    double target_luma                 = 112.0; // target mean luma for auto exposure. 112 is ~midpoint between pure black and pure white, giving the model the best chance to see texture in either case.

    bool   enable_deblur               = false;
    double deblur_amount               = 0.60;

    bool   enable_stabilization        = false;
    int    flow_max_corners            = 200;
    double flow_quality_level          = 0.01;
    double flow_min_distance           = 8.0;

    PhysicalSignalColorMode color_mode = PhysicalSignalColorMode::Bgr;

    PhysicalSignalQualityGateConfig quality_gate{};

    // Per-frame scene-stability signal computed AFTER the model image is
    // built. Cheap (small thumbnail). Consumed by every cache-aware
    // downstream operator via PhysicalFrameBus::FrameView::scene_stability.
    PhysicalSceneStabilityConfig    scene_stability{};
};

// Affine transform from RAW sensor pixel space to MODEL pixel space:
//   model_x = raw_x * scale_x + offset_x
//   model_y = raw_y * scale_y + offset_y
// To back-project a model-space point to raw coords: invert.
struct PhysicalSignalRawToModelTransform {
    double scale_x  = 1.0;
    double scale_y  = 1.0;
    double offset_x = 0.0;
    double offset_y = 0.0;
};

struct PhysicalSignalConditioningStatus {
    uint64_t processed_frame_counter   = 0;

    int      last_input_width          = 0;
    int      last_input_height         = 0;
    int      last_output_width         = 0;
    int      last_output_height        = 0;

    double   last_input_luma           = 0.0;
    double   last_output_luma          = 0.0;
    double   last_applied_exposure_gain = 1.0;

    int      last_flow_tracked_points  = 0;
    double   last_flow_dx              = 0.0;
    double   last_flow_dy              = 0.0;

    bool     using_auto_exposure       = true;
    bool     stabilization_active      = false;

    // Quality gate diagnostics (computed on raw input every frame)
    double   last_clipped_pixel_ratio  = 0.0;
    double   last_laplacian_variance   = 0.0;
    bool     last_quality_gate_passed  = true;
    std::string last_quality_gate_reason;
    uint64_t total_frames_dropped_by_quality_gate = 0;

    // Geometric provenance: how raw maps into model
    PhysicalSignalRawToModelTransform last_raw_to_model{};

    // Last computed scene-stability signal. Snapshotted into the FrameView
    // metadata at publish time; also surfaced here for diagnostics panels.
    PhysicalSceneStability  last_scene_stability{};

    std::string last_pipeline_summary;
    std::string last_failure_reason;
};

// Result handed back to the producer so it can decide whether to publish
// and what metadata to attach. Self-contained — no globals touched.
struct PhysicalSignalConditioningResult {
    bool                              accepted = false;
    PhysicalSignalRawToModelTransform raw_to_model{};
    std::string                       color_space_label;     // e.g. "BGR8_SRGB", "GRAY8_SRGB"
    std::string                       pipeline_summary;
    std::string                       drop_reason;           // populated iff accepted == false
    int                               raw_width  = 0;
    int                               raw_height = 0;
    int                               model_width  = 0;
    int                               model_height = 0;

    // Scene-stability snapshot for THIS frame. Populated only when accepted
    // is true. Producer (PhysicalEnvironmentLoop) attaches this to the
    // FrameMetadata published on PhysicalFrameBus.
    PhysicalSceneStability            scene_stability{};
};

class PhysicalFrameConditioner {
public:
    PhysicalFrameConditioner();

    void ConfigurePhysicalSignalConditioning(const PhysicalSignalConditioningConfig& cfg);
    void ResetPhysicalSignalConditioningToDefaults();
    void ResetPhysicalSignalConditioningTemporalState();

    PhysicalSignalConditioningConfig GetPhysicalSignalConditioningConfigSnapshot() const;
    PhysicalSignalConditioningStatus GetPhysicalSignalConditioningStatusSnapshot() const;

    // Returns a result describing whether the frame was accepted and, if so,
    // the raw->model transform plus color space label. Caller MUST inspect
    // `accepted`; if false, `out_model_bgr` is left untouched and `drop_reason`
    // is populated. Throws only on programmer error (empty/invalid raw).
    PhysicalSignalConditioningResult ProcessRawFrameToModelSignal(
        const cv::Mat& raw_bgr,
        uint64_t       frame_counter,
        cv::Mat&       out_model_bgr);

private:
    static void ValidatePhysicalSignalConditioningConfig(const PhysicalSignalConditioningConfig& cfg);
    static double ComputeMeanLumaFromBgr(const cv::Mat& bgr);

    PhysicalSignalConditioningConfig config_;
    PhysicalSignalConditioningStatus status_;

    cv::Mat previous_gray_for_flow_;

    // Scene-stability state. previous_scene_thumbnail_gray_ is empty until
    // the first conditioned frame is published; in that case the very first
    // signal is reported as is_stable=false / change_reason="first_frame".
    cv::Mat  previous_scene_thumbnail_gray_;
    uint64_t previous_scene_hash_64_ = 0;
    bool     previous_scene_hash_valid_ = false;
    uint32_t scene_stable_streak_      = 0;
};

PhysicalSignalConditioningConfig BuildDefaultPhysicalSignalConditioningConfig();

}}} // namespace GRIM::Perception::Physical
