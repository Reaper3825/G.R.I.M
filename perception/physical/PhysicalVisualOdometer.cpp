#include "PhysicalVisualOdometer.hpp"

#include "PhysicalLocalizationLogTag.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Build a 4x4 SE(3) transform from a 3x3 rotation and 3x1 translation.
// Always CV_64F. Returns a fresh, owned cv::Mat.
cv::Mat BuildSe3FromRotationAndTranslation(const cv::Mat& R_3x3, const cv::Mat& t_3x1) {
    if (R_3x3.rows != 3 || R_3x3.cols != 3) {
        throw std::runtime_error(
            "BuildSe3FromRotationAndTranslation: R must be 3x3, got "
            + std::to_string(R_3x3.rows) + "x" + std::to_string(R_3x3.cols));
    }
    if (t_3x1.total() != 3) {
        throw std::runtime_error(
            "BuildSe3FromRotationAndTranslation: t must have 3 elements, got "
            + std::to_string(t_3x1.total()));
    }
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R64; R_3x3.convertTo(R64, CV_64F);
    cv::Mat t64; t_3x1.convertTo(t64, CV_64F);
    R64.copyTo(T(cv::Rect(0, 0, 3, 3)));
    // t can come in as 3x1 or 1x3 — normalise to 3x1.
    if (t64.rows == 1 && t64.cols == 3) t64 = t64.t();
    t64.reshape(1, 3).copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

// Inverse of an SE(3) without going through cv::invert (cheaper and
// numerically stable: T_inv = [R^T, -R^T * t]).
cv::Mat InvertSe3(const cv::Mat& T_4x4) {
    if (T_4x4.rows != 4 || T_4x4.cols != 4 || T_4x4.type() != CV_64F) {
        throw std::runtime_error("InvertSe3: expected 4x4 CV_64F");
    }
    cv::Mat R   = T_4x4(cv::Rect(0, 0, 3, 3));
    cv::Mat t   = T_4x4(cv::Rect(3, 0, 1, 3));
    cv::Mat Rt  = R.t();
    cv::Mat T_inv = cv::Mat::eye(4, 4, CV_64F);
    Rt.copyTo(T_inv(cv::Rect(0, 0, 3, 3)));
    cv::Mat tinv = -Rt * t;
    tinv.copyTo(T_inv(cv::Rect(3, 0, 1, 3)));
    return T_inv;
}

double ComputeMedianParallaxPixels(const std::vector<cv::Point2f>& a,
                                   const std::vector<cv::Point2f>& b,
                                   const std::vector<uchar>& inlier_mask)
{
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "ComputeMedianParallaxPixels: a.size() != b.size() ("
            + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")");
    }
    std::vector<double> distances;
    distances.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (i < inlier_mask.size() && !inlier_mask[i]) continue;
        const double dx = a[i].x - b[i].x;
        const double dy = a[i].y - b[i].y;
        distances.push_back(std::sqrt(dx*dx + dy*dy));
    }
    if (distances.empty()) return 0.0;
    const size_t mid = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + mid, distances.end());
    return distances[mid];
}

double ComputeMeanReprojectionErrorPixels(const cv::Mat& K,
                                          const cv::Mat& R,
                                          const cv::Mat& t,
                                          const std::vector<cv::Point2f>& prev_pts,
                                          const std::vector<cv::Point2f>& curr_pts,
                                          const std::vector<uchar>& inlier_mask)
{
    if (prev_pts.size() != curr_pts.size()) {
        throw std::runtime_error(
            "ComputeMeanReprojectionErrorPixels: prev/curr size mismatch");
    }
    if (prev_pts.empty()) return 0.0;

    // Triangulate inlier points using P_prev = K[I|0], P_curr = K[R|t].
    cv::Mat P_prev = cv::Mat::zeros(3, 4, CV_64F);
    K.convertTo(P_prev(cv::Rect(0, 0, 3, 3)), CV_64F);

    cv::Mat Rt(3, 4, CV_64F);
    R.convertTo(Rt(cv::Rect(0, 0, 3, 3)), CV_64F);
    t.convertTo(Rt(cv::Rect(3, 0, 1, 3)), CV_64F);
    cv::Mat P_curr = K * Rt;

    std::vector<cv::Point2f> ip, ic;
    ip.reserve(prev_pts.size()); ic.reserve(curr_pts.size());
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (i < inlier_mask.size() && !inlier_mask[i]) continue;
        ip.push_back(prev_pts[i]);
        ic.push_back(curr_pts[i]);
    }
    if (ip.empty()) return 0.0;

    cv::Mat points_4d;
    cv::triangulatePoints(P_prev, P_curr, ip, ic, points_4d);
    if (points_4d.cols == 0) return 0.0;

    // Project back and accumulate the mean error from BOTH cameras.
    double total_err  = 0.0;
    int    sample_n   = 0;
    for (int i = 0; i < points_4d.cols; ++i) {
        const double w = points_4d.at<float>(3, i);
        if (std::abs(w) < 1e-9) continue;
        const double x = points_4d.at<float>(0, i) / w;
        const double y = points_4d.at<float>(1, i) / w;
        const double z = points_4d.at<float>(2, i) / w;
        if (z <= 0.0) continue;  // behind the prior camera

        // Project into prev: P_prev @ [x y z 1]^T  → already pixel (z = depth)
        const double u_prev = K.at<double>(0,0) * (x / z) + K.at<double>(0,2);
        const double v_prev = K.at<double>(1,1) * (y / z) + K.at<double>(1,2);
        const double du_p = u_prev - ip[i].x;
        const double dv_p = v_prev - ip[i].y;
        total_err += std::sqrt(du_p*du_p + dv_p*dv_p);

        // Project into curr.
        cv::Mat Xh = (cv::Mat_<double>(4,1) << x, y, z, 1.0);
        cv::Mat px = P_curr * Xh;
        const double w_c = px.at<double>(2,0);
        if (std::abs(w_c) < 1e-9) continue;
        const double u_curr = px.at<double>(0,0) / w_c;
        const double v_curr = px.at<double>(1,0) / w_c;
        const double du_c = u_curr - ic[i].x;
        const double dv_c = v_curr - ic[i].y;
        total_err += std::sqrt(du_c*du_c + dv_c*dv_c);

        sample_n += 2;
    }
    return (sample_n == 0) ? 0.0 : (total_err / static_cast<double>(sample_n));
}

void ValidateConfigOrThrow(const PhysicalVisualOdometerConfig& cfg) {
    if (cfg.orb_max_features < 100) {
        throw std::runtime_error("ValidateConfig: orb_max_features < 100 will starve matching");
    }
    if (cfg.orb_scale_factor <= 1.0f) {
        throw std::runtime_error("ValidateConfig: orb_scale_factor must be > 1.0");
    }
    if (cfg.orb_n_levels < 1) {
        throw std::runtime_error("ValidateConfig: orb_n_levels < 1");
    }
    if (cfg.ransac_inlier_threshold_pixels <= 0.0) {
        throw std::runtime_error("ValidateConfig: ransac_inlier_threshold_pixels <= 0");
    }
    if (cfg.ransac_confidence <= 0.5 || cfg.ransac_confidence >= 1.0) {
        throw std::runtime_error("ValidateConfig: ransac_confidence must be in (0.5, 1.0)");
    }
    if (cfg.min_matches_for_pose < 8) {
        throw std::runtime_error("ValidateConfig: min_matches_for_pose < 8 (essential matrix needs >=5)");
    }
    if (cfg.min_inliers_for_pose < 5) {
        throw std::runtime_error("ValidateConfig: min_inliers_for_pose < 5");
    }
    if (cfg.min_median_parallax_pixels < 0.0) {
        throw std::runtime_error("ValidateConfig: min_median_parallax_pixels < 0");
    }
    if (cfg.max_trajectory_samples < 1) {
        throw std::runtime_error("ValidateConfig: max_trajectory_samples < 1");
    }
}

} // anonymous namespace

PhysicalVisualOdometer::PhysicalVisualOdometer() {
    cfg_ = PhysicalVisualOdometerConfig{};
    ValidateConfigOrThrow(cfg_);
    orb_detector_ = cv::ORB::create(cfg_.orb_max_features, cfg_.orb_scale_factor, cfg_.orb_n_levels);
    if (!orb_detector_) {
        throw std::runtime_error("PhysicalVisualOdometer: cv::ORB::create returned null");
    }
    descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, /*crossCheck=*/true);
    if (!descriptor_matcher_) {
        throw std::runtime_error("PhysicalVisualOdometer: cv::BFMatcher::create returned null");
    }
    T_world_camera_ = cv::Mat::eye(4, 4, CV_64F);
}

PhysicalVisualOdometer::~PhysicalVisualOdometer() = default;

void PhysicalVisualOdometer::ConfigurePhysicalVisualOdometer(
    const PhysicalVisualOdometerConfig& cfg)
{
    ValidateConfigOrThrow(cfg);
    cfg_ = cfg;
    orb_detector_ = cv::ORB::create(cfg_.orb_max_features, cfg_.orb_scale_factor, cfg_.orb_n_levels);
    if (!orb_detector_) {
        throw std::runtime_error(
            "ConfigurePhysicalVisualOdometer: cv::ORB::create returned null");
    }
    descriptor_matcher_ = cv::BFMatcher::create(
        cfg_.use_hamming_matcher ? cv::NORM_HAMMING : cv::NORM_HAMMING, /*crossCheck=*/true);
    if (!descriptor_matcher_) {
        throw std::runtime_error(
            "ConfigurePhysicalVisualOdometer: cv::BFMatcher::create returned null");
    }
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "ConfigurePhysicalVisualOdometer: cfg accepted (orb_max_features="
              + std::to_string(cfg_.orb_max_features) + ")");
}

void PhysicalVisualOdometer::SetCameraIntrinsics(const cv::Mat& K, const cv::Mat& dist) {
    if (K.empty() || K.rows != 3 || K.cols != 3) {
        throw std::runtime_error(
            "SetCameraIntrinsics: K must be 3x3 (got "
            + std::to_string(K.rows) + "x" + std::to_string(K.cols) + ")");
    }
    K.convertTo(camera_matrix_, CV_64F);
    if (dist.empty()) {
        dist_coeffs_ = cv::Mat();
    } else {
        dist.convertTo(dist_coeffs_, CV_64F);
    }
    has_intrinsics_ = true;
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "SetCameraIntrinsics: K accepted (fx="
              + std::to_string(camera_matrix_.at<double>(0,0))
              + " fy=" + std::to_string(camera_matrix_.at<double>(1,1))
              + " cx=" + std::to_string(camera_matrix_.at<double>(0,2))
              + " cy=" + std::to_string(camera_matrix_.at<double>(1,2)) + ")");
}

bool PhysicalVisualOdometer::HasCameraIntrinsics() const { return has_intrinsics_; }

void PhysicalVisualOdometer::ResetPhysicalVisualOdometer() {
    has_prior_frame_ = false;
    prior_gray_frame_.release();
    prior_keypoints_.clear();
    prior_descriptors_.release();
    T_world_camera_       = cv::Mat::eye(4, 4, CV_64F);
    frame_count_          = 0;
    anchor_frame_counter_ = 0;
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG, "ResetPhysicalVisualOdometer: state cleared");
}

void PhysicalVisualOdometer::RouteFrameToPhysicalVisualOdometer(
    const cv::Mat& bgr_or_gray, uint64_t frame_counter,
    PhysicalVisualOdometerOutput& out)
{
    const auto t_start = std::chrono::steady_clock::now();
    out = PhysicalVisualOdometerOutput{};
    out.T_world_camera    = T_world_camera_.clone();
    out.T_prev_to_current = cv::Mat::eye(4, 4, CV_64F);

    if (bgr_or_gray.empty()) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Failed;
        out.tracking_reason = "RouteFrameToPhysicalVisualOdometer: input frame is empty";
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, out.tracking_reason);
        throw std::runtime_error(out.tracking_reason);
    }
    if (!has_intrinsics_) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Failed;
        out.tracking_reason = "RouteFrameToPhysicalVisualOdometer: no camera intrinsics — "
                              "calibrate the camera first (UI: Calibration tab → Run)";
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, out.tracking_reason);
        throw std::runtime_error(out.tracking_reason);
    }
    if (!orb_detector_ || !descriptor_matcher_) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Failed;
        out.tracking_reason = "RouteFrameToPhysicalVisualOdometer: ORB / matcher not constructed";
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, out.tracking_reason);
        throw std::runtime_error(out.tracking_reason);
    }

    // ── Convert to grayscale ─────────────────────────────────────────────
    cv::Mat gray;
    if (bgr_or_gray.channels() == 1) {
        gray = bgr_or_gray;
    } else if (bgr_or_gray.channels() == 3) {
        cv::cvtColor(bgr_or_gray, gray, cv::COLOR_BGR2GRAY);
    } else {
        out.tracking_state  = PhysicalLocalizationTrackingState::Failed;
        out.tracking_reason = "RouteFrameToPhysicalVisualOdometer: unsupported channel count "
                              + std::to_string(bgr_or_gray.channels());
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, out.tracking_reason);
        throw std::runtime_error(out.tracking_reason);
    }

    // ── Detect + describe ────────────────────────────────────────────────
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;
    orb_detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    out.keypoints_detected_current_frame = static_cast<int32_t>(keypoints.size());

    // ── Anchor case: no prior frame → register and return Initializing ──
    if (!has_prior_frame_) {
        if (keypoints.size() < static_cast<size_t>(cfg_.min_matches_for_pose)) {
            out.tracking_state  = PhysicalLocalizationTrackingState::Initializing;
            out.tracking_reason = "Anchor candidate: only "
                                  + std::to_string(keypoints.size())
                                  + " keypoints (< min_matches_for_pose="
                                  + std::to_string(cfg_.min_matches_for_pose)
                                  + "); waiting for richer frame";
            const auto t_end = std::chrono::steady_clock::now();
            out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            ++frame_count_;
            return;
        }
        prior_gray_frame_     = gray.clone();
        prior_keypoints_      = keypoints;
        prior_descriptors_    = descriptors.clone();
        has_prior_frame_      = true;
        anchor_frame_counter_ = frame_counter;
        T_world_camera_       = cv::Mat::eye(4, 4, CV_64F);
        out.T_world_camera    = T_world_camera_.clone();
        out.tracking_state    = PhysicalLocalizationTrackingState::Initializing;
        out.tracking_reason   = "Anchor frame registered (frame_counter="
                                + std::to_string(frame_counter)
                                + "); waiting for second frame with parallax";
        const auto t_end = std::chrono::steady_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ++frame_count_;
        LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG, out.tracking_reason);
        return;
    }

    // ── Match against the prior frame ────────────────────────────────────
    if (descriptors.empty() || prior_descriptors_.empty()) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Lost;
        out.tracking_reason = "Empty descriptors (prior_empty="
                              + std::to_string(prior_descriptors_.empty())
                              + " current_empty=" + std::to_string(descriptors.empty()) + ")";
        const auto t_end = std::chrono::steady_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ++frame_count_;
        return;
    }

    std::vector<cv::DMatch> matches;
    descriptor_matcher_->match(prior_descriptors_, descriptors, matches);
    out.matches_to_prior_frame = static_cast<int32_t>(matches.size());

    if (matches.size() < static_cast<size_t>(cfg_.min_matches_for_pose)) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Lost;
        out.tracking_reason = "Only " + std::to_string(matches.size())
                              + " matches (< min_matches_for_pose="
                              + std::to_string(cfg_.min_matches_for_pose) + ")";
        // Replace prior with current frame so we have a fresh chance next tick.
        prior_gray_frame_  = gray.clone();
        prior_keypoints_   = keypoints;
        prior_descriptors_ = descriptors.clone();
        const auto t_end = std::chrono::steady_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ++frame_count_;
        return;
    }

    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> curr_pts;
    prev_pts.reserve(matches.size());
    curr_pts.reserve(matches.size());
    for (const auto& m : matches) {
        prev_pts.push_back(prior_keypoints_[m.queryIdx].pt);
        curr_pts.push_back(keypoints   [m.trainIdx].pt);
    }

    // ── Essential matrix + pose recovery ─────────────────────────────────
    cv::Mat inlier_mask_mat;
    cv::Mat E = cv::findEssentialMat(prev_pts, curr_pts, camera_matrix_,
                                     cv::RANSAC,
                                     cfg_.ransac_confidence,
                                     cfg_.ransac_inlier_threshold_pixels,
                                     cfg_.ransac_max_iterations,
                                     inlier_mask_mat);
    if (E.empty()) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Lost;
        out.tracking_reason = "cv::findEssentialMat returned empty (degenerate geometry?)";
        prior_gray_frame_   = gray.clone();
        prior_keypoints_    = keypoints;
        prior_descriptors_  = descriptors.clone();
        const auto t_end = std::chrono::steady_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ++frame_count_;
        return;
    }

    std::vector<uchar> inlier_mask(inlier_mask_mat.total(), 0);
    for (size_t i = 0; i < inlier_mask_mat.total(); ++i) {
        inlier_mask[i] = inlier_mask_mat.at<uchar>(static_cast<int>(i));
    }

    cv::Mat R, t;
    const int recovered_inliers = cv::recoverPose(E, prev_pts, curr_pts, camera_matrix_,
                                                  R, t, inlier_mask_mat);
    out.essential_inliers     = recovered_inliers;
    out.median_parallax_pixels =
        ComputeMedianParallaxPixels(prev_pts, curr_pts, inlier_mask);

    if (recovered_inliers < cfg_.min_inliers_for_pose) {
        out.tracking_state  = PhysicalLocalizationTrackingState::Lost;
        out.tracking_reason = "recoverPose inliers=" + std::to_string(recovered_inliers)
                              + " < min_inliers_for_pose="
                              + std::to_string(cfg_.min_inliers_for_pose);
        prior_gray_frame_   = gray.clone();
        prior_keypoints_    = keypoints;
        prior_descriptors_  = descriptors.clone();
        const auto t_end = std::chrono::steady_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ++frame_count_;
        return;
    }

    if (out.median_parallax_pixels < cfg_.min_median_parallax_pixels) {
        // Geometrically degenerate (camera essentially still) — keep the
        // anchor, do NOT swap prior frame so we can wait for movement.
        out.tracking_state  = PhysicalLocalizationTrackingState::Initializing;
        out.tracking_reason = "median_parallax="
                              + std::to_string(out.median_parallax_pixels)
                              + "px < min_median_parallax="
                              + std::to_string(cfg_.min_median_parallax_pixels)
                              + "px (camera not moving enough)";
        const auto t_end = std::chrono::steady_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ++frame_count_;
        return;
    }

    // ── Stash recoverPose products for the loop ──────────────────────────
    // We deliberately do NOT compose T_world_camera_ here — the loop owns
    // the choice of translation scale (depth-fusion / IMU / stereo / 1.0).
    // ComposeAndAdvanceWorldPoseUsingScaledTranslation finishes the job.
    {
        cv::Mat R64; R.convertTo(R64, CV_64F);
        cv::Mat t64; t.convertTo(t64, CV_64F);
        if (t64.rows == 1 && t64.cols == 3) t64 = t64.t();
        t64 = t64.reshape(1, 3);
        const double tn = cv::norm(t64);
        if (tn > 1e-12) t64 = t64 / tn;     // recoverPose returns unit, but enforce.
        out.R_curr_from_prev      = R64.clone();
        out.t_unit_curr_from_prev = t64.clone();
    }
    out.inlier_prev_pts.clear();
    out.inlier_curr_pts.clear();
    out.inlier_prev_pts.reserve(static_cast<size_t>(out.essential_inliers));
    out.inlier_curr_pts.reserve(static_cast<size_t>(out.essential_inliers));
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (i < inlier_mask.size() && !inlier_mask[i]) continue;
        out.inlier_prev_pts.push_back(prev_pts[i]);
        out.inlier_curr_pts.push_back(curr_pts[i]);
    }

    out.mean_reprojection_error_pixels =
        ComputeMeanReprojectionErrorPixels(camera_matrix_, R, t,
                                           prev_pts, curr_pts, inlier_mask);

    out.tracking_state  = PhysicalLocalizationTrackingState::Tracking;
    out.tracking_reason.clear();
    // Identity placeholders until the loop calls
    // ComposeAndAdvanceWorldPoseUsingScaledTranslation.
    out.T_prev_to_current = cv::Mat::eye(4, 4, CV_64F);
    out.T_world_camera    = T_world_camera_.clone();
    out.applied_translation_scale = 1.0;

    // Slide prior → current.
    prior_gray_frame_   = gray.clone();
    prior_keypoints_    = keypoints;
    prior_descriptors_  = descriptors.clone();

    const auto t_end = std::chrono::steady_clock::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    ++frame_count_;
}

void PhysicalVisualOdometer::ComposeAndAdvanceWorldPoseUsingScaledTranslation(
    double translation_scale, PhysicalVisualOdometerOutput& inout)
{
    if (inout.tracking_state != PhysicalLocalizationTrackingState::Tracking) {
        throw std::runtime_error(
            "ComposeAndAdvanceWorldPoseUsingScaledTranslation: tracking_state must be "
            "Tracking, got " + std::string(
                DescribePhysicalLocalizationTrackingState(inout.tracking_state)));
    }
    if (inout.R_curr_from_prev.empty()
        || inout.R_curr_from_prev.rows != 3 || inout.R_curr_from_prev.cols != 3
        || inout.R_curr_from_prev.type() != CV_64F)
    {
        throw std::runtime_error(
            "ComposeAndAdvanceWorldPoseUsingScaledTranslation: R_curr_from_prev must "
            "be 3x3 CV_64F (was empty or wrong shape — RouteFrame must run first)");
    }
    if (inout.t_unit_curr_from_prev.empty()
        || inout.t_unit_curr_from_prev.total() != 3
        || inout.t_unit_curr_from_prev.type() != CV_64F)
    {
        throw std::runtime_error(
            "ComposeAndAdvanceWorldPoseUsingScaledTranslation: t_unit_curr_from_prev "
            "must be 3-element CV_64F (was empty or wrong shape)");
    }
    if (!std::isfinite(translation_scale) || translation_scale <= 0.0) {
        throw std::runtime_error(
            "ComposeAndAdvanceWorldPoseUsingScaledTranslation: translation_scale must "
            "be finite and > 0, got " + std::to_string(translation_scale));
    }

    cv::Mat t_scaled = inout.t_unit_curr_from_prev * translation_scale;
    cv::Mat T_curr_prev = BuildSe3FromRotationAndTranslation(
        inout.R_curr_from_prev, t_scaled);
    cv::Mat T_prev_curr = InvertSe3(T_curr_prev);

    T_world_camera_         = cv::Mat(T_world_camera_ * T_prev_curr).clone();
    inout.T_prev_to_current = T_prev_curr.clone();
    inout.T_world_camera    = T_world_camera_.clone();
    inout.applied_translation_scale = translation_scale;
}

}}} // namespace GRIM::Perception::Physical
