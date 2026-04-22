#pragma once

#include "PhysicalDepthMap.hpp"
#include "PhysicalImageOperatorState.hpp"

#include <memory>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalMonocularDepthEstimator — Stage-3 single-image depth predictor.
//
//  Backend: cv::dnn::Net loading an ONNX file. Designed for the MiDaS family
//  (MiDaS small / MiDaS v2.1 / MiDaS v3 / DPT) which all share:
//
//      input  : [1, 3, H, W] float, normalised by mean/std
//      output : [1, H, W] OR [1, 1, H, W] float — INVERSE depth
//               (closer surface ⇒ larger value)
//
//  Self-owned, internally managed. The single mainloop integration entry
//  point is RouteFrameToPhysicalMonocularDepthEstimator(); LoadOnnxModel*
//  is called at config time only.
//
//  Numerical-precision contract:
//    * The estimator NEVER fabricates depth when no model is loaded — it
//      sets state == NoModelConfigured and leaves the output empty.
//    * The output `inverse_depth_image` is min-max normalised to [0, 1]
//      across the frame; the pre-normalisation min/max are preserved on
//      `PhysicalDepthMap.raw_inverse_depth_min/max` so consumers can
//      reverse the scaling.
//    * `metric_scale_meters > 0` ⇒ a metric depth image is produced via
//      depth_m = metric_scale_meters / max(raw_inverse_depth, eps). This
//      is the MiDaS-recommended single-parameter calibration; the value
//      is supplied by config and assumed monocular. Without this calibration
//      the depth is RELATIVE only (units == DepthUnits::Relative). NEVER
//      pretend Relative is Meters — Rule 20.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalMonocularDepthEstimatorConfig {
    std::string  onnx_model_path;                  // empty ⇒ NoModelConfigured (intentional)
    int          input_width             = 256;    // MiDaS-small standard
    int          input_height            = 256;
    float        input_scale             = 1.0f / 255.0f;
    cv::Scalar   input_mean              = cv::Scalar(123.675, 116.28, 103.53); // ImageNet
    cv::Scalar   input_std               = cv::Scalar( 58.395,  57.12,  57.375); // ImageNet
    bool         swap_rb                 = true;   // BGR ⇒ RGB inside blob
    bool         output_is_disparity     = true;   // MiDaS family ⇒ true (inverse depth)

    // Optional metric calibration. When > 0, units = Meters and
    //   depth_m = metric_scale_meters / max(raw_inverse_depth, metric_epsilon)
    // is produced. When 0, units = Relative.
    double       metric_scale_meters     = 0.0;
    float        metric_epsilon          = 1.0e-3f;

    int          dnn_backend_id          = cv::dnn::DNN_BACKEND_OPENCV;
    int          dnn_target_id           = cv::dnn::DNN_TARGET_CPU;
};

class PhysicalMonocularDepthEstimator {
public:
    PhysicalMonocularDepthEstimator();
    ~PhysicalMonocularDepthEstimator();

    // Apply (or re-apply) configuration. Throws on missing file / bad ONNX.
    // Empty `cfg.onnx_model_path` is NOT a failure — explicit "be silent"
    // request that resets the operator to NoModelConfigured.
    void LoadOnnxModelIntoPhysicalMonocularDepthEstimator(
        const PhysicalMonocularDepthEstimatorConfig& cfg);

    // Run inference on one MODEL-space frame. Never throws — failures land
    // in `out_state` / `out_error`. On success `out_depth` is populated and
    // sized to (model_image.cols, model_image.rows).
    void RouteFrameToPhysicalMonocularDepthEstimator(
        const cv::Mat&  model_image,
        PhysicalDepthMap& out_depth,
        PhysicalImageOperatorState& out_state,
        std::string&    out_error,
        double&         out_inference_ms);

    void                       ResetPhysicalMonocularDepthEstimator();
    PhysicalImageOperatorState GetPhysicalMonocularDepthEstimatorState() const;
    std::string                GetPhysicalMonocularDepthEstimatorLastError() const;
    bool                       IsPhysicalMonocularDepthEstimatorReady() const;
    uint64_t                   GetPhysicalMonocularDepthEstimatorInferenceCount() const;

private:
    mutable std::mutex                            mutex_;
    PhysicalMonocularDepthEstimatorConfig         cfg_{};
    std::unique_ptr<cv::dnn::Net>                 net_;
    PhysicalImageOperatorState                    state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                                   last_error_reason_;
    uint64_t                                      inference_count_ = 0;
};

}}} // namespace GRIM::Perception::Physical
