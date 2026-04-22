#include "PhysicalVisualScaleFromDepth.hpp"

#include "PhysicalLocalizationLogTag.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Bilinearly sample a CV_32FC1 depth image at floating-point pixel (u, v).
// Returns NaN if (u, v) is out of bounds OR any of the 4 neighbour samples
// is non-finite / non-positive (depth holes are common in MiDaS output).
float SampleDepthBilinearOrNaN(const cv::Mat& depth_image_32fc1,
                               float u, float v)
{
    if (depth_image_32fc1.empty()) return std::numeric_limits<float>::quiet_NaN();
    if (depth_image_32fc1.type() != CV_32FC1) {
        throw std::runtime_error(
            "SampleDepthBilinearOrNaN: expected CV_32FC1, got type="
            + std::to_string(depth_image_32fc1.type()));
    }
    const int W = depth_image_32fc1.cols;
    const int H = depth_image_32fc1.rows;
    if (!std::isfinite(u) || !std::isfinite(v)) return std::numeric_limits<float>::quiet_NaN();
    if (u < 0.0f || v < 0.0f || u > static_cast<float>(W - 1) || v > static_cast<float>(H - 1)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float fx = u - static_cast<float>(x0);
    const float fy = v - static_cast<float>(y0);
    const float d00 = depth_image_32fc1.at<float>(y0, x0);
    const float d01 = depth_image_32fc1.at<float>(y0, x1);
    const float d10 = depth_image_32fc1.at<float>(y1, x0);
    const float d11 = depth_image_32fc1.at<float>(y1, x1);
    if (!std::isfinite(d00) || !std::isfinite(d01)
        || !std::isfinite(d10) || !std::isfinite(d11)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (d00 <= 0.0f || d01 <= 0.0f || d10 <= 0.0f || d11 <= 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float a = d00 * (1.0f - fx) + d01 * fx;
    const float b = d10 * (1.0f - fx) + d11 * fx;
    return a * (1.0f - fy) + b * fy;
}

double Median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

} // anonymous namespace

PhysicalVisualScaleResult ComputeTranslationScaleFromDepthMap(
    const cv::Mat& K,
    const cv::Mat& R_curr_from_prev,
    const cv::Mat& t_unit_curr_from_prev,
    const std::vector<cv::Point2f>& inlier_prev_pts,
    const std::vector<cv::Point2f>& inlier_curr_pts,
    const PhysicalDepthMap& depth_map,
    PhysicalVisualScaleDepthSampleViewpoint sample_viewpoint,
    int min_supporting_inliers)
{
    PhysicalVisualScaleResult r;

    if (K.empty() || K.rows != 3 || K.cols != 3 || K.type() != CV_64F) {
        throw std::runtime_error(
            "ComputeTranslationScaleFromDepthMap: K must be 3x3 CV_64F");
    }
    if (R_curr_from_prev.empty() || R_curr_from_prev.rows != 3
        || R_curr_from_prev.cols != 3 || R_curr_from_prev.type() != CV_64F) {
        throw std::runtime_error(
            "ComputeTranslationScaleFromDepthMap: R_curr_from_prev must be 3x3 CV_64F");
    }
    if (t_unit_curr_from_prev.empty() || t_unit_curr_from_prev.total() != 3
        || t_unit_curr_from_prev.type() != CV_64F) {
        throw std::runtime_error(
            "ComputeTranslationScaleFromDepthMap: t_unit must be 3-element CV_64F");
    }
    if (inlier_prev_pts.size() != inlier_curr_pts.size()) {
        throw std::runtime_error(
            "ComputeTranslationScaleFromDepthMap: inlier_prev/curr size mismatch ("
            + std::to_string(inlier_prev_pts.size()) + " vs "
            + std::to_string(inlier_curr_pts.size()) + ")");
    }
    if (min_supporting_inliers < 1) {
        throw std::runtime_error(
            "ComputeTranslationScaleFromDepthMap: min_supporting_inliers must be >= 1, got "
            + std::to_string(min_supporting_inliers));
    }
    if (inlier_prev_pts.empty()) {
        r.reason = "No inliers supplied — VO did not produce a Tracking pose.";
        return r;
    }
    if (depth_map.empty()) {
        r.reason = "Depth map is empty (no Stage-3 result published yet for this frame).";
        return r;
    }

    // Pick the depth image to sample. Metric path uses metric_depth_image
    // directly; relative path uses the normalised inverse-depth image and
    // converts to a relative *depth* via depth = 1 / inv_depth_raw.
    const bool depth_is_metric = (depth_map.units == DepthUnits::Meters)
                                 && std::isfinite(depth_map.metric_scale_meters)
                                 && depth_map.metric_scale_meters > 0.0;

    cv::Mat depth_image_for_sampling;
    if (depth_is_metric) {
        if (depth_map.metric_depth_image.empty()
            || depth_map.metric_depth_image.type() != CV_32FC1) {
            throw std::runtime_error(
                "ComputeTranslationScaleFromDepthMap: depth_map.units == Meters but "
                "metric_depth_image is empty or wrong type (CV_32FC1 required)");
        }
        depth_image_for_sampling = depth_map.metric_depth_image;
    } else {
        // Build a relative-depth image on the fly from normalised inverse
        // depth + raw min/max. This is informational ONLY — is_metric stays
        // false so the loop won't promote scale_state to ScaledByDepthMap.
        if (depth_map.inverse_depth_image.empty()
            || depth_map.inverse_depth_image.type() != CV_32FC1) {
            throw std::runtime_error(
                "ComputeTranslationScaleFromDepthMap: inverse_depth_image is empty or "
                "wrong type (CV_32FC1 required)");
        }
        const float raw_min = depth_map.raw_inverse_depth_min;
        const float raw_max = depth_map.raw_inverse_depth_max;
        const float raw_span = raw_max - raw_min;
        if (!std::isfinite(raw_span) || std::abs(raw_span) < 1e-12f) {
            r.reason = "Depth map has degenerate raw inverse-depth range (raw_max == raw_min).";
            return r;
        }
        cv::Mat inv_raw;
        depth_map.inverse_depth_image.convertTo(inv_raw, CV_32FC1, raw_span, raw_min);
        // depth = 1 / inv_raw; protect against zero/negative.
        cv::Mat ones = cv::Mat::ones(inv_raw.size(), CV_32FC1);
        cv::Mat depth_relative;
        cv::divide(ones, inv_raw, depth_relative);
        depth_image_for_sampling = depth_relative;
    }

    // The depth image is at MODEL pixel resolution by contract — see
    // PhysicalDepthMap.hpp ("map_width / map_height match the MODEL frame
    // the estimator actually saw"). The VO inlier points are also in MODEL
    // pixel coordinates by contract (PhysicalVisualOdometer operates on
    // s.frame_view.model_image). So we sample directly with no rescale.
    //
    // Defensive: if the depth-map dims somehow disagree with the depth
    // image's actual size, scale the sample coordinates instead of crashing.
    const int dW = depth_image_for_sampling.cols;
    const int dH = depth_image_for_sampling.rows;
    if (dW <= 0 || dH <= 0) {
        r.reason = "Depth image has non-positive dimensions.";
        return r;
    }
    const float sample_sx = (depth_map.map_width  > 0)
        ? static_cast<float>(dW) / static_cast<float>(depth_map.map_width)
        : 1.0f;
    const float sample_sy = (depth_map.map_height > 0)
        ? static_cast<float>(dH) / static_cast<float>(depth_map.map_height)
        : 1.0f;

    // Triangulate all inliers with P_prev = K[I|0], P_curr = K[R|t̂].
    cv::Mat P_prev = cv::Mat::zeros(3, 4, CV_64F);
    K.copyTo(P_prev(cv::Rect(0, 0, 3, 3)));
    cv::Mat Rt(3, 4, CV_64F);
    R_curr_from_prev.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
    cv::Mat t_col = t_unit_curr_from_prev.reshape(1, 3);
    t_col.copyTo(Rt(cv::Rect(3, 0, 1, 3)));
    cv::Mat P_curr = K * Rt;

    cv::Mat points_4d;
    cv::triangulatePoints(P_prev, P_curr, inlier_prev_pts, inlier_curr_pts, points_4d);
    if (points_4d.cols == 0) {
        r.reason = "cv::triangulatePoints returned 0 points.";
        return r;
    }

    // Pre-extract R rows for camera-frame conversion (only used when
    // sampling at the CURRENT viewpoint).
    const double r00 = R_curr_from_prev.at<double>(0,0);
    const double r01 = R_curr_from_prev.at<double>(0,1);
    const double r02 = R_curr_from_prev.at<double>(0,2);
    const double r10 = R_curr_from_prev.at<double>(1,0);
    const double r11 = R_curr_from_prev.at<double>(1,1);
    const double r12 = R_curr_from_prev.at<double>(1,2);
    const double r20 = R_curr_from_prev.at<double>(2,0);
    const double r21 = R_curr_from_prev.at<double>(2,1);
    const double r22 = R_curr_from_prev.at<double>(2,2);
    const double tx_unit = t_unit_curr_from_prev.at<double>(0);
    const double ty_unit = t_unit_curr_from_prev.at<double>(1);
    const double tz_unit = t_unit_curr_from_prev.at<double>(2);

    std::vector<double> ratios;
    std::vector<double> z_meters_samples;
    std::vector<double> z_unit_samples;
    ratios.reserve(static_cast<size_t>(points_4d.cols));
    z_meters_samples.reserve(static_cast<size_t>(points_4d.cols));
    z_unit_samples.reserve(static_cast<size_t>(points_4d.cols));

    for (int i = 0; i < points_4d.cols; ++i) {
        const float w = points_4d.at<float>(3, i);
        if (!std::isfinite(w) || std::abs(w) < 1e-9f) continue;
        const double X_prev = static_cast<double>(points_4d.at<float>(0, i)) / w;
        const double Y_prev = static_cast<double>(points_4d.at<float>(1, i)) / w;
        const double Z_prev = static_cast<double>(points_4d.at<float>(2, i)) / w;
        if (!std::isfinite(Z_prev) || Z_prev <= 0.0) continue;   // behind prev camera

        // Z_unit in the frame that matches the depth-map sample point.
        double Z_unit_at_sample = 0.0;
        cv::Point2f sample_px;
        if (sample_viewpoint == PhysicalVisualScaleDepthSampleViewpoint::PrevFrame) {
            Z_unit_at_sample = Z_prev;
            sample_px = inlier_prev_pts[static_cast<size_t>(i)];
        } else {
            // Re-express the 3D point in the CURRENT camera frame:
            //   X_curr = R * X_prev + t̂
            const double Z_curr =
                r20 * X_prev + r21 * Y_prev + r22 * Z_prev + tz_unit;
            if (!std::isfinite(Z_curr) || Z_curr <= 0.0) continue;
            Z_unit_at_sample = Z_curr;
            sample_px = inlier_curr_pts[static_cast<size_t>(i)];
        }
        // Avoid unused-variable warnings when we don't transform x/y.
        (void)r00; (void)r01; (void)r02;
        (void)r10; (void)r11; (void)r12;
        (void)tx_unit; (void)ty_unit;

        const float u = sample_px.x * sample_sx;
        const float v = sample_px.y * sample_sy;
        const float depth_sample = SampleDepthBilinearOrNaN(depth_image_for_sampling, u, v);
        if (!std::isfinite(depth_sample) || depth_sample <= 0.0f) continue;

        const double z_meters = static_cast<double>(depth_sample);
        const double ratio    = z_meters / Z_unit_at_sample;
        if (!std::isfinite(ratio) || ratio <= 0.0) continue;

        ratios.push_back(ratio);
        z_meters_samples.push_back(z_meters);
        z_unit_samples.push_back(Z_unit_at_sample);
    }

    r.supporting_inliers = static_cast<int32_t>(ratios.size());
    if (r.supporting_inliers < min_supporting_inliers) {
        r.reason = "Only " + std::to_string(r.supporting_inliers)
                 + " depth-supported inliers (< min_supporting_inliers="
                 + std::to_string(min_supporting_inliers) + ")";
        return r;
    }

    const double k = Median(ratios);
    if (!std::isfinite(k) || k <= 0.0) {
        r.reason = "Median ratio is non-finite or non-positive (k=" + std::to_string(k) + ")";
        return r;
    }

    r.translation_scale = k;
    r.median_z_meters   = Median(z_meters_samples);
    r.median_z_unit     = Median(z_unit_samples);
    r.is_metric         = depth_is_metric;
    r.succeeded         = true;
    return r;
}

}}} // namespace GRIM::Perception::Physical
