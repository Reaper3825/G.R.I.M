#pragma once

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalDepthMap — per-pixel depth produced by the monocular depth
//  estimator (MiDaS-style) at MODEL pixel resolution.
//
//  The native output of MiDaS-class networks is INVERSE depth (closer = larger
//  value). We carry both spaces explicitly so consumers do not silently
//  conflate them — Rule 20.
//
//  Coordinate convention:
//    * map_width / map_height match the MODEL frame (post-conditioner) the
//      estimator actually saw, so a UI overlay drawn in MODEL space lands on
//      the correct pixels with no extra scaling.
//    * For a pixel (x, y) in the MODEL frame, inverse_depth_image.at<float>(y, x)
//      is the network's raw inverse-depth value AFTER min-max normalisation
//      to [0, 1]. The unnormalised network response is preserved in
//      inverse_depth_min/max so downstream consumers can reverse the scaling.
//    * units == DepthUnits::Relative means there is NO real-world scale —
//      inverse-depth values are useful only for ordering and ratios within a
//      single frame. units == DepthUnits::Meters means metric_scale_meters
//      is calibrated and metric_depth_image carries depth in metres.
// ─────────────────────────────────────────────────────────────────────────────

enum class DepthUnits : uint8_t {
    Relative = 0,   // ordering only (no metric scale supplied)
    Meters   = 1    // metric_depth_image populated in metres
};

inline const char* DescribeDepthUnits(DepthUnits u) {
    switch (u) {
        case DepthUnits::Relative: return "Relative";
        case DepthUnits::Meters:   return "Meters";
    }
    return "InvalidDepthUnits";
}

struct PhysicalDepthMap {
    // Inverse-depth image, CV_32FC1, size = (map_width, map_height).
    // Normalised so that min == 0.0 and max == 1.0 across this frame.
    cv::Mat     inverse_depth_image;

    // Metric depth image, CV_32FC1, only populated when units == Meters.
    // Empty otherwise.
    cv::Mat     metric_depth_image;

    int32_t     map_width      = 0;
    int32_t     map_height     = 0;

    // Raw network response statistics, BEFORE normalisation. Useful for
    // diagnosing scale drift between frames.
    float       raw_inverse_depth_min = 0.0f;
    float       raw_inverse_depth_max = 0.0f;

    DepthUnits  units                 = DepthUnits::Relative;

    // metric_scale_meters: if units == Meters, metric_depth = metric_scale_meters
    //                                          / max(inverse_depth_raw, eps).
    // Configured by the user via PhysicalMonocularDepthEstimatorConfig.
    // Stored here so consumers can reproduce the conversion.
    double      metric_scale_meters   = 0.0;

    bool empty() const { return inverse_depth_image.empty(); }
};

}}} // namespace GRIM::Perception::Physical
