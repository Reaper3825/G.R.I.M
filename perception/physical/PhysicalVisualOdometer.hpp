#pragma once

#include "PhysicalLocalizationResult.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalVisualOdometer
//
//  The numerical core of Stage-5. Stateful, single-camera, monocular VO.
//
//  Pipeline (every successful frame):
//    1.  Convert the model-space BGR frame to grayscale.
//    2.  Detect ORB keypoints (rotation+scale invariant, free).
//    3.  Match descriptors to the previous frame via brute-force Hamming
//        with cross-check.
//    4.  Reject outliers via cv::findEssentialMat with RANSAC, using the
//        per-process camera intrinsics K from PhysicalCameraCalibrator.
//    5.  Recover the relative pose [R|t] with cv::recoverPose. Translation
//        is unit-norm — the well-known monocular scale ambiguity is
//        explicit in `pose_scale_state == UnscaledMonocular`.
//    6.  Compose: T_world_curr = T_world_prev * inv([R|t]).
//
//  The class is OWNED by PhysicalLocalizationLoop. No other TU should
//  construct one.
//
//  Thread-safety: NOT thread-safe. The loop holds a mutex around every
//  call. We deliberately do not double-lock here.
//
//  Rule 20: every failure path is loud:
//    * No calibration loaded → state = Failed, reason populated, throw on
//      the first call so the loop reports the issue once and stops.
//    * Empty input image → Failed.
//    * Too few matches / insufficient parallax → state = Lost OR
//      Initializing (still the anchor frame), reason populated. NOT
//      thrown — these are expected transient conditions.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalVisualOdometerConfig {
    int    orb_max_features              = 1500;    // detector budget
    float  orb_scale_factor              = 1.2f;
    int    orb_n_levels                  = 8;

    // Essential-matrix RANSAC. Pixels in MODEL space.
    double ransac_inlier_threshold_pixels = 1.0;
    double ransac_confidence              = 0.999;
    int    ransac_max_iterations          = 1000;

    // Acceptance gates.
    int    min_matches_for_pose          = 30;     // < → Lost
    double min_median_parallax_pixels    = 1.5;    // < while Initializing → stay Initializing
    int    min_inliers_for_pose          = 15;     // < → Lost

    // Trajectory cap (publication side; the loop uses this to size its
    // sliding ring).
    int    max_trajectory_samples        = 4096;

    // ORB feature matching: BFMatcher with cross-check (deterministic).
    // True == HAMMING (the only sane choice for ORB).
    bool   use_hamming_matcher           = true;
};

struct PhysicalVisualOdometerOutput {
    PhysicalLocalizationTrackingState  tracking_state =
        PhysicalLocalizationTrackingState::Uninitialized;
    std::string                        tracking_reason;

    // T_world_camera at this frame. Always 4x4 CV_64F. Identity until the
    // anchor frame is registered. Populated ONLY after the loop calls
    // ComposeAndAdvanceWorldPoseUsingScaledTranslation — until then it
    // mirrors the pose at end-of-previous-frame so callers always have a
    // valid 4x4.
    cv::Mat                            T_world_camera;

    // Relative transform from the PREVIOUS successful frame to this one,
    // already scaled by `applied_translation_scale`. Identity when the
    // frame became the anchor (no prior) or when tracking failed.
    cv::Mat                            T_prev_to_current;

    // ── Raw recoverPose products (Tracking only) ───────────────────────
    // Translation t_unit_curr_from_prev is unit-norm by construction —
    // monocular scale ambiguity is explicit. The loop picks a scale (e.g.
    // from a depth map) and calls ComposeAndAdvanceWorldPoseUsingScaledTranslation
    // to advance T_world_camera_.
    cv::Mat                            R_curr_from_prev;       // 3x3 CV_64F
    cv::Mat                            t_unit_curr_from_prev;  // 3x1 CV_64F, ‖t‖=1

    // Inlier 2D match pairs in MODEL pixel coords, post-RANSAC and
    // post-recoverPose-cheirality. inlier_prev_pts[i] ↔ inlier_curr_pts[i]
    // observe the same 3D scene point. Used by depth-fusion to triangulate
    // and solve for translation magnitude.
    std::vector<cv::Point2f>           inlier_prev_pts;
    std::vector<cv::Point2f>           inlier_curr_pts;

    // Translation scale that ComposeAndAdvanceWorldPoseUsingScaledTranslation
    // applied to t_unit_curr_from_prev. 1.0 when no depth source anchors
    // it (UnscaledMonocular). >0 always.
    double                             applied_translation_scale        = 1.0;

    int32_t                            keypoints_detected_current_frame = 0;
    int32_t                            matches_to_prior_frame           = 0;
    int32_t                            essential_inliers                = 0;
    double                             median_parallax_pixels           = 0.0;
    double                             mean_reprojection_error_pixels   = 0.0;
    double                             elapsed_ms                       = 0.0;
};

class PhysicalVisualOdometer {
public:
    PhysicalVisualOdometer();
    ~PhysicalVisualOdometer();

    PhysicalVisualOdometer(const PhysicalVisualOdometer&)            = delete;
    PhysicalVisualOdometer& operator=(const PhysicalVisualOdometer&) = delete;

    // Hot-swappable. Throws on invalid config.
    void ConfigurePhysicalVisualOdometer(const PhysicalVisualOdometerConfig& cfg);

    // Inject the camera intrinsics (3x3 K and distortion coefficients).
    // Throws if K is not 3x3 CV_64F. Distortion is currently informational
    // — the loop is expected to undistort BEFORE handing the frame in
    // (UndistortBgrFrameUsingPhysicalCalibration), so K here describes the
    // UNDISTORTED frame.
    void SetCameraIntrinsics(const cv::Mat& camera_matrix_3x3_cv64f,
                             const cv::Mat& dist_coeffs_optional);

    // True iff intrinsics have been set at least once.
    bool HasCameraIntrinsics() const;

    // The single processing entry point. `bgr_or_gray_model_frame` may be
    // either CV_8UC3 BGR or CV_8UC1; the operator converts internally.
    //
    // Throws ONLY on programmer error (no intrinsics, empty input). All
    // expected transient failures (no matches, low parallax, RANSAC
    // failure) populate `out.tracking_state` + `out.tracking_reason` and
    // return cleanly so the loop can publish a Lost / Initializing
    // snapshot.
    void RouteFrameToPhysicalVisualOdometer(const cv::Mat& bgr_or_gray_model_frame,
                                            uint64_t frame_counter,
                                            PhysicalVisualOdometerOutput& out);

    // Step 2 of every Tracking frame. RouteFrameToPhysicalVisualOdometer
    // populates `out.R_curr_from_prev` + `out.t_unit_curr_from_prev` (unit
    // norm) but does NOT advance T_world_camera_ — the loop owns the
    // decision of HOW to scale that translation (depth-fusion, IMU,
    // stereo, or just `1.0` for an unscaled monocular trajectory).
    //
    // After RouteFrame returns with tracking_state == Tracking, the loop
    // computes a scale (>0) and calls this. The method:
    //   * Multiplies t_unit by translation_scale.
    //   * Composes T_curr_prev = [R | scale*t_unit].
    //   * Advances T_world_camera_ ← T_world_camera_ * inv(T_curr_prev).
    //   * Writes T_world_camera and T_prev_to_current into `inout`.
    //   * Stamps inout.applied_translation_scale.
    //
    // Throws iff:
    //   * inout.tracking_state != Tracking
    //   * inout.R_curr_from_prev / t_unit_curr_from_prev are missing/wrong
    //     shape
    //   * translation_scale is not finite OR <= 0
    void ComposeAndAdvanceWorldPoseUsingScaledTranslation(
        double translation_scale,
        PhysicalVisualOdometerOutput& inout);

    // Drop the current pose, prior keypoints, and anchor — next frame
    // will become the new anchor.
    void ResetPhysicalVisualOdometer();

    // Diagnostic.
    uint64_t GetPhysicalVisualOdometerFrameCount() const { return frame_count_; }
    PhysicalVisualOdometerConfig GetPhysicalVisualOdometerConfigSnapshot() const { return cfg_; }

private:
    // ── Owned state ──
    PhysicalVisualOdometerConfig         cfg_;
    cv::Ptr<cv::ORB>                     orb_detector_;
    cv::Ptr<cv::BFMatcher>               descriptor_matcher_;

    cv::Mat                              camera_matrix_;          // 3x3 CV_64F
    cv::Mat                              dist_coeffs_;            // 1xN CV_64F or empty
    bool                                 has_intrinsics_ = false;

    // Anchor / previous-frame state.
    bool                                 has_prior_frame_ = false;
    cv::Mat                              prior_gray_frame_;
    std::vector<cv::KeyPoint>            prior_keypoints_;
    cv::Mat                              prior_descriptors_;

    // World pose. Identity until the anchor is registered.
    cv::Mat                              T_world_camera_;

    uint64_t                             frame_count_   = 0;
    uint64_t                             anchor_frame_counter_ = 0;
};

}}} // namespace GRIM::Perception::Physical
