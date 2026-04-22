#pragma once

#include "PhysicalDepthMap.hpp"
#include "PhysicalLocalizationResult.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalVisualScaleFromDepth
//
//  Solves the monocular scale ambiguity for one VO step using a depth map
//  whose source frame == the CURRENT VO frame.
//
//  Geometry:
//    For each inlier 2D-2D correspondence (prev_pt[i] ↔ curr_pt[i]) we
//    triangulate a 3D point in the PREVIOUS camera frame using
//      P_prev = K [I | 0]
//      P_curr = K [R | t̂]            (t̂ unit, ‖t̂‖ = 1)
//    Triangulation yields X_i with depth Z_unit_i along the previous
//    camera optical axis. Z_unit_i is in the same arbitrary "unit" as t̂.
//
//    The depth map gives Z_meters_i at pixel curr_pt[i] (or prev_pt[i] —
//    we sample at PREV because triangulation expressed Z_unit in the
//    PREV camera frame, which matches a depth map captured at the prev
//    timestamp; in practice depth maps lag VO by 0..1 frames so we sample
//    the FIRST viewpoint that matches the depth map's source counter).
//
//    The translation scale that converts t̂ → metric metres is
//      k = median_i  Z_meters_i / Z_unit_i
//
//    Median is chosen over mean for robustness to depth-map outliers
//    (foreground/background bleed, small holes, MiDaS artefacts).
//
//  Rule 20 commitments:
//    * `is_metric == false` whenever the depth map is in DepthUnits::Relative
//      OR metric_scale_meters is not finite & > 0. The numerical
//      `translation_scale` is still returned for diagnostics, but the loop
//      MUST NOT advertise the resulting trajectory as metric.
//    * succeeded == false if fewer than `min_supporting_inliers` valid
//      depth samples could be triangulated. The loop falls back to the
//      caller's chosen default (typically 1.0 → UnscaledMonocular).
//    * Every numeric guard (finite scale, depth > 0, denom > eps) is
//      checked explicitly — no silent NaN propagation.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalVisualScaleResult {
    bool        succeeded         = false;
    bool        is_metric         = false;   // true iff depth_map.units == Meters
                                             //         AND metric_scale_meters > 0
    double      translation_scale = 1.0;     // multiplier for t_unit
    int32_t     supporting_inliers = 0;      // # of (depth_at_pixel, Z_unit) pairs
                                             // that survived all numerical guards
    double      median_z_meters    = 0.0;    // median sampled depth (m, or "units")
    double      median_z_unit      = 0.0;    // median triangulated Z (in t̂-units)
    std::string reason;                      // populated whenever succeeded == false
};

// `pixel_sample_source` tells the helper which 2D track to sample the
// depth map at. The depth map covers ONE physical instant; if the bus
// served us a depth_map whose source_frame_counter == prev VO frame, we
// must sample at prev_pts; if it matches the curr VO frame, we sample at
// curr_pts (in which case Z_unit must be re-expressed in the CURRENT
// camera frame, which we do internally).
enum class PhysicalVisualScaleDepthSampleViewpoint : uint8_t {
    PrevFrame = 0,
    CurrFrame = 1
};

PhysicalVisualScaleResult ComputeTranslationScaleFromDepthMap(
    const cv::Mat& camera_matrix_3x3_cv64f,
    const cv::Mat& R_curr_from_prev_3x3_cv64f,
    const cv::Mat& t_unit_curr_from_prev_3x1_cv64f,
    const std::vector<cv::Point2f>& inlier_prev_pts,
    const std::vector<cv::Point2f>& inlier_curr_pts,
    const PhysicalDepthMap&         depth_map,
    PhysicalVisualScaleDepthSampleViewpoint sample_viewpoint,
    int min_supporting_inliers);

}}} // namespace GRIM::Perception::Physical
