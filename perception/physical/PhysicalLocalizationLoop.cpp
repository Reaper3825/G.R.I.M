#include "PhysicalLocalizationLoop.hpp"

#include "PhysicalCameraCalibrator.hpp"
#include "PhysicalCalibrationStore.hpp"
#include "PhysicalFrameBus.hpp"
#include "PhysicalLocalizationBus.hpp"
#include "PhysicalLocalizationLogTag.hpp"
#include "PhysicalSpatialGroundingBus.hpp"
#include "PhysicalVisualScaleFromDepth.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>

#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// ── Pose helpers ────────────────────────────────────────────────────────────
//
// Quaternion derivation (Shoemake / Eberly). Operates on the 3x3 rotation
// block of T_world_camera. Output convention is (w, x, y, z).
void ExtractQuaternionWxyzFromRotation(const cv::Mat& T_4x4_cv64f, double q_wxyz[4]) {
    const double m00 = T_4x4_cv64f.at<double>(0,0);
    const double m01 = T_4x4_cv64f.at<double>(0,1);
    const double m02 = T_4x4_cv64f.at<double>(0,2);
    const double m10 = T_4x4_cv64f.at<double>(1,0);
    const double m11 = T_4x4_cv64f.at<double>(1,1);
    const double m12 = T_4x4_cv64f.at<double>(1,2);
    const double m20 = T_4x4_cv64f.at<double>(2,0);
    const double m21 = T_4x4_cv64f.at<double>(2,1);
    const double m22 = T_4x4_cv64f.at<double>(2,2);
    const double trace = m00 + m11 + m22;
    if (trace > 0.0) {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        q_wxyz[0] = 0.25 / s;
        q_wxyz[1] = (m21 - m12) * s;
        q_wxyz[2] = (m02 - m20) * s;
        q_wxyz[3] = (m10 - m01) * s;
        return;
    }
    if (m00 > m11 && m00 > m22) {
        const double s = 2.0 * std::sqrt(1.0 + m00 - m11 - m22);
        q_wxyz[0] = (m21 - m12) / s;
        q_wxyz[1] = 0.25 * s;
        q_wxyz[2] = (m01 + m10) / s;
        q_wxyz[3] = (m02 + m20) / s;
    } else if (m11 > m22) {
        const double s = 2.0 * std::sqrt(1.0 + m11 - m00 - m22);
        q_wxyz[0] = (m02 - m20) / s;
        q_wxyz[1] = (m01 + m10) / s;
        q_wxyz[2] = 0.25 * s;
        q_wxyz[3] = (m12 + m21) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + m22 - m00 - m11);
        q_wxyz[0] = (m10 - m01) / s;
        q_wxyz[1] = (m02 + m20) / s;
        q_wxyz[2] = (m12 + m21) / s;
        q_wxyz[3] = 0.25 * s;
    }
}

// ZYX intrinsic Euler decomposition (yaw, pitch, roll).
void ExtractYawPitchRollRadiansFromRotation(const cv::Mat& T_4x4_cv64f, double ypr[3]) {
    const double m00 = T_4x4_cv64f.at<double>(0,0);
    const double m10 = T_4x4_cv64f.at<double>(1,0);
    const double m20 = T_4x4_cv64f.at<double>(2,0);
    const double m21 = T_4x4_cv64f.at<double>(2,1);
    const double m22 = T_4x4_cv64f.at<double>(2,2);
    ypr[0] = std::atan2(m10, m00);                                   // yaw  (Z)
    ypr[1] = std::atan2(-m20, std::sqrt(m21*m21 + m22*m22));          // pitch(Y)
    ypr[2] = std::atan2(m21, m22);                                   // roll (X)
}

void StampPoseConvenienceFields(PhysicalCameraPose& pose,
                                PhysicalLocalizationPoseScaleState scale_state)
{
    if (pose.T_world_camera.empty()
        || pose.T_world_camera.rows != 4 || pose.T_world_camera.cols != 4
        || pose.T_world_camera.type() != CV_64F)
    {
        throw std::runtime_error(
            "StampPoseConvenienceFields: pose.T_world_camera must be 4x4 CV_64F");
    }
    pose.position_meters[0] = pose.T_world_camera.at<double>(0, 3);
    pose.position_meters[1] = pose.T_world_camera.at<double>(1, 3);
    pose.position_meters[2] = pose.T_world_camera.at<double>(2, 3);
    ExtractQuaternionWxyzFromRotation(pose.T_world_camera, pose.quaternion_world_camera);
    ExtractYawPitchRollRadiansFromRotation(pose.T_world_camera, pose.yaw_pitch_roll_radians);
    pose.scale_state = scale_state;
}

// Compute angular velocity (rotation vector / dt) and linear velocity from
// two world poses and the elapsed time.
void ComputeVelocityFromPoseDelta(const cv::Mat& T_prev,
                                  const cv::Mat& T_curr,
                                  double dt_seconds,
                                  PhysicalCameraVelocity& out_velocity)
{
    out_velocity = PhysicalCameraVelocity{};
    out_velocity.sample_interval_seconds = dt_seconds;
    if (dt_seconds <= 0.0) return;
    if (T_prev.empty() || T_curr.empty()) return;

    const double dx = T_curr.at<double>(0, 3) - T_prev.at<double>(0, 3);
    const double dy = T_curr.at<double>(1, 3) - T_prev.at<double>(1, 3);
    const double dz = T_curr.at<double>(2, 3) - T_prev.at<double>(2, 3);
    out_velocity.linear_world_per_sec[0] = dx / dt_seconds;
    out_velocity.linear_world_per_sec[1] = dy / dt_seconds;
    out_velocity.linear_world_per_sec[2] = dz / dt_seconds;

    cv::Mat R_prev = T_prev(cv::Rect(0, 0, 3, 3));
    cv::Mat R_curr = T_curr(cv::Rect(0, 0, 3, 3));
    cv::Mat R_delta = R_curr * R_prev.t();    // rotation from prev to curr
    cv::Mat rvec;
    cv::Rodrigues(R_delta, rvec);
    out_velocity.angular_radians_per_sec[0] = rvec.at<double>(0,0) / dt_seconds;
    out_velocity.angular_radians_per_sec[1] = rvec.at<double>(1,0) / dt_seconds;
    out_velocity.angular_radians_per_sec[2] = rvec.at<double>(2,0) / dt_seconds;
}

// ── Loop state ──────────────────────────────────────────────────────────────

struct PhysicalLocalizationState {
    std::mutex                                          mutex;
    bool                                                initialized   = false;
    bool                                                shutting_down = false;

    std::unique_ptr<PhysicalVisualOdometer>             odometer;
    std::unique_ptr<PhysicalOccupancyGridMapper>        grid_mapper;

    // Pending configs deposited via Request* before lazy init. Applied at
    // the first Tick.
    std::unique_ptr<PhysicalVisualOdometerConfig>       pending_odom_cfg;
    std::unique_ptr<PhysicalOccupancyGridMapperConfig>  pending_grid_cfg;
    bool                                                pending_reset = false;

    // Whether intrinsics have been pushed into the odometer this lifetime.
    bool                                                intrinsics_synced = false;

    std::string                                         last_error_reason;
    uint64_t                                            tick_count          = 0;
    uint64_t                                            processed_count     = 0;
    uint64_t                                            last_seen_frame_ctr = 0;

    PhysicalFrameBus::FrameView                         frame_view;

    // Last grounding-results counter we observed on the spatial-grounding
    // bus. Used to know when a fresh PhysicalDepthMap is available for the
    // CURRENT VO frame.
    uint64_t                                            last_seen_grounding_ctr = 0;
    PhysicalSpatialGroundingBus::ResultsView            grounding_view;
    bool                                                have_any_grounding_view = false;

    // Pose history for velocity differentiation. Stored as (T_world_camera,
    // capture_steady_ns) pairs.
    cv::Mat                                             prev_published_T;
    int64_t                                             prev_published_steady_ns = 0;

    // Sliding trajectory ring (publication-side cap).
    std::deque<cv::Point3d>                             trajectory_ring;
};

PhysicalLocalizationState& GetState() {
    static PhysicalLocalizationState s;
    return s;
}

void LazyInitLocked(PhysicalLocalizationState& s) {
    if (s.initialized) return;
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "TickPhysicalLocalization: first call — running lazy init");
    s.odometer    = std::make_unique<PhysicalVisualOdometer>();
    s.grid_mapper = std::make_unique<PhysicalOccupancyGridMapper>();

    if (s.pending_odom_cfg) {
        try {
            s.odometer->ConfigurePhysicalVisualOdometer(*s.pending_odom_cfg);
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: pending odometer cfg failed: ") + e.what();
            LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, s.last_error_reason);
        }
        s.pending_odom_cfg.reset();
    }
    if (s.pending_grid_cfg) {
        try {
            s.grid_mapper->ConfigurePhysicalOccupancyGridMapper(*s.pending_grid_cfg);
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("LazyInit: pending grid cfg failed: ") + e.what();
            LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, s.last_error_reason);
        }
        s.pending_grid_cfg.reset();
    }
    s.initialized = true;
}

void TrySyncIntrinsicsLocked(PhysicalLocalizationState& s) {
    if (s.intrinsics_synced) return;
    if (!IsPhysicalCalibrationDataAvailable()) return;
    PhysicalCalibrationData calib;
    try {
        GetPhysicalCalibrationData(calib);
    } catch (const std::exception& e) {
        s.last_error_reason =
            std::string("TrySyncIntrinsics: GetPhysicalCalibrationData threw: ") + e.what();
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, s.last_error_reason);
        return;
    }
    try {
        s.odometer->SetCameraIntrinsics(calib.camera_matrix, calib.dist_coeffs);
        s.intrinsics_synced = true;
        LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
                  "TrySyncIntrinsics: pushed K into PhysicalVisualOdometer");
    } catch (const std::exception& e) {
        s.last_error_reason =
            std::string("TrySyncIntrinsics: SetCameraIntrinsics threw: ") + e.what();
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, s.last_error_reason);
    }
}

void PublishUninitializedSnapshotLocked(const PhysicalLocalizationState& s,
                                        const std::string& reason)
{
    PhysicalLocalizationSnapshot snap;
    snap.tracking_state    = PhysicalLocalizationTrackingState::Uninitialized;
    snap.pose_scale_state  = PhysicalLocalizationPoseScaleState::Unknown;
    snap.tracking_reason   = reason;
    snap.pose_world_camera.T_world_camera = cv::Mat::eye(4, 4, CV_64F);
    snap.T_prev_to_current                = cv::Mat::eye(4, 4, CV_64F);
    snap.built_at_steady_ns = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    StampPoseConvenienceFields(snap.pose_world_camera, snap.pose_scale_state);
    try {
        PhysicalLocalizationBus::Instance().PublishPhysicalLocalizationSnapshotToBus(snap);
    } catch (const std::exception& e) {
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG,
                  std::string("PublishUninitializedSnapshot threw: ") + e.what());
    }
    (void)s;
}

void PublishFailedSnapshotLocked(const std::string& reason, uint64_t source_frame_counter)
{
    PhysicalLocalizationSnapshot snap;
    snap.source_frame_counter = source_frame_counter;
    snap.tracking_state       = PhysicalLocalizationTrackingState::Failed;
    snap.pose_scale_state     = PhysicalLocalizationPoseScaleState::Unknown;
    snap.tracking_reason      = reason;
    snap.pose_world_camera.T_world_camera = cv::Mat::eye(4, 4, CV_64F);
    snap.T_prev_to_current                = cv::Mat::eye(4, 4, CV_64F);
    snap.built_at_steady_ns   = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    StampPoseConvenienceFields(snap.pose_world_camera, snap.pose_scale_state);
    try {
        PhysicalLocalizationBus::Instance().PublishPhysicalLocalizationSnapshotToBus(snap);
    } catch (const std::exception& e) {
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG,
                  std::string("PublishFailedSnapshot threw: ") + e.what());
    }
}

} // anonymous namespace

void TickPhysicalLocalization() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.shutting_down) return;
    LazyInitLocked(s);
    ++s.tick_count;

    if (s.pending_reset) {
        s.odometer->ResetPhysicalVisualOdometer();
        s.grid_mapper->ResetPhysicalOccupancyGridMapper();
        s.prev_published_T.release();
        s.prev_published_steady_ns = 0;
        s.trajectory_ring.clear();
        s.last_seen_grounding_ctr = 0;
        s.have_any_grounding_view = false;
        s.pending_reset = false;
        LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG, "TickPhysicalLocalization: reset applied");
    }

    // Try to pick up intrinsics every tick until we have them.
    TrySyncIntrinsicsLocked(s);

    if (!s.odometer->HasCameraIntrinsics()) {
        // Publish a Failed snapshot so the UI has something to show. Once
        // calibration is loaded TrySyncIntrinsicsLocked will succeed and
        // we'll start publishing real snapshots.
        const std::string reason =
            "TickPhysicalLocalization: no camera intrinsics — calibrate the camera "
            "(UI: Physical Environment → Calibration tab → Run intrinsic calibration)";
        if (s.last_error_reason != reason) {
            s.last_error_reason = reason;
            LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, reason);
        }
        PublishFailedSnapshotLocked(reason, /*source_frame_counter=*/0);
        return;
    }

    const bool frame_advanced = PhysicalFrameBus::Instance().PullLatestFrameView(
        s.frame_view, s.last_seen_frame_ctr);
    if (!frame_advanced) return;

    if (s.frame_view.model_image.empty()) {
        const std::string reason =
            "TickPhysicalLocalization: pulled frame has empty model_image";
        s.last_error_reason = reason;
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, reason);
        PublishFailedSnapshotLocked(reason, s.frame_view.frame_counter);
        return;
    }

    // ── Run odometer ──────────────────────────────────────────────────────
    PhysicalVisualOdometerOutput vo_out;
    try {
        s.odometer->RouteFrameToPhysicalVisualOdometer(
            s.frame_view.model_image, s.frame_view.frame_counter, vo_out);
    } catch (const std::exception& e) {
        const std::string reason =
            std::string("RouteFrameToPhysicalVisualOdometer threw: ") + e.what();
        s.last_error_reason = reason;
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, reason);
        PublishFailedSnapshotLocked(reason, s.frame_view.frame_counter);
        return;
    }

    // ── Resolve translation scale (depth fusion when possible) ────────────
    // Default: leave translation at unit norm — the trajectory is then
    // self-consistent only step-to-step (true monocular ambiguity).
    double translation_scale = 1.0;
    PhysicalLocalizationPoseScaleState resolved_scale_state =
        PhysicalLocalizationPoseScaleState::Unknown;
    std::string scale_diagnostic;

    if (vo_out.tracking_state == PhysicalLocalizationTrackingState::Tracking) {
        // Pull the latest grounding view; only use the depth map if it was
        // produced from the same source frame as our current VO frame
        // (or — fallback — from the *previous* VO anchor frame, since the
        // depth network can lag the VO pipeline by one frame on slow hosts).
        const bool grounding_advanced =
            PhysicalSpatialGroundingBus::Instance()
                .PullLatestPhysicalSpatialGroundingResultsView(
                    s.grounding_view, s.last_seen_grounding_ctr);
        if (grounding_advanced) s.have_any_grounding_view = true;

        const PhysicalDepthMap* depth_for_fusion = nullptr;
        PhysicalVisualScaleDepthSampleViewpoint sample_vp =
            PhysicalVisualScaleDepthSampleViewpoint::CurrFrame;
        if (s.have_any_grounding_view
            && !s.grounding_view.results.depth_map.empty())
        {
            const uint64_t depth_src = s.grounding_view.results.source_frame_counter;
            if (depth_src == s.frame_view.frame_counter) {
                depth_for_fusion = &s.grounding_view.results.depth_map;
                sample_vp = PhysicalVisualScaleDepthSampleViewpoint::CurrFrame;
            } else if (depth_src == s.frame_view.frame_counter - 1) {
                depth_for_fusion = &s.grounding_view.results.depth_map;
                sample_vp = PhysicalVisualScaleDepthSampleViewpoint::PrevFrame;
            } else {
                scale_diagnostic = "depth-fusion: no matching depth map "
                    "(depth_src=" + std::to_string(depth_src)
                    + ", vo_curr=" + std::to_string(s.frame_view.frame_counter) + ")";
            }
        } else {
            scale_diagnostic = "depth-fusion: no grounding view published yet";
        }

        if (depth_for_fusion) {
            try {
                PhysicalCalibrationData calib_for_fusion;
                GetPhysicalCalibrationData(calib_for_fusion);
                constexpr int kMinSupportingInliers = 10;
                PhysicalVisualScaleResult sr =
                    ComputeTranslationScaleFromDepthMap(
                        calib_for_fusion.camera_matrix,
                        vo_out.R_curr_from_prev,
                        vo_out.t_unit_curr_from_prev,
                        vo_out.inlier_prev_pts,
                        vo_out.inlier_curr_pts,
                        *depth_for_fusion,
                        sample_vp,
                        kMinSupportingInliers);
                if (sr.succeeded) {
                    translation_scale = sr.translation_scale;
                    resolved_scale_state = sr.is_metric
                        ? PhysicalLocalizationPoseScaleState::ScaledByDepthMap
                        : PhysicalLocalizationPoseScaleState::UnscaledMonocular;
                    scale_diagnostic =
                        "depth-fusion: k=" + std::to_string(sr.translation_scale)
                        + " metric=" + std::to_string(sr.is_metric)
                        + " inliers=" + std::to_string(sr.supporting_inliers)
                        + " median_z_meters=" + std::to_string(sr.median_z_meters)
                        + " median_z_unit=" + std::to_string(sr.median_z_unit);
                } else {
                    scale_diagnostic = "depth-fusion: " + sr.reason;
                }
            } catch (const std::exception& e) {
                scale_diagnostic =
                    std::string("ComputeTranslationScaleFromDepthMap threw: ") + e.what();
                LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, scale_diagnostic);
            }
        }

        if (resolved_scale_state == PhysicalLocalizationPoseScaleState::Unknown) {
            // No metric anchor available — keep monocular convention
            // (unit translation, scale_state = UnscaledMonocular). The
            // grid mapper will refuse to write under this state.
            resolved_scale_state = PhysicalLocalizationPoseScaleState::UnscaledMonocular;
            translation_scale = 1.0;
        }

        // Compose pose with the resolved scale.
        try {
            s.odometer->ComposeAndAdvanceWorldPoseUsingScaledTranslation(
                translation_scale, vo_out);
        } catch (const std::exception& e) {
            const std::string reason =
                std::string("ComposeAndAdvanceWorldPoseUsingScaledTranslation threw: ")
                + e.what();
            s.last_error_reason = reason;
            LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, reason);
            PublishFailedSnapshotLocked(reason, s.frame_view.frame_counter);
            return;
        }
    }

    // ── Build snapshot ────────────────────────────────────────────────────
    PhysicalLocalizationSnapshot snap;
    snap.source_frame_counter = s.frame_view.frame_counter;
    snap.model_image_width    = s.frame_view.metadata.model_width;
    snap.model_image_height   = s.frame_view.metadata.model_height;
    snap.raw_image_width      = s.frame_view.metadata.raw_width;
    snap.raw_image_height     = s.frame_view.metadata.raw_height;
    snap.raw_to_model         = s.frame_view.metadata.raw_to_model;
    if (snap.model_image_width  <= 0) snap.model_image_width  = s.frame_view.model_image.cols;
    if (snap.model_image_height <= 0) snap.model_image_height = s.frame_view.model_image.rows;

    snap.tracking_state                   = vo_out.tracking_state;
    snap.tracking_reason                  = vo_out.tracking_reason;
    snap.pose_world_camera.T_world_camera = vo_out.T_world_camera;
    snap.T_prev_to_current                = vo_out.T_prev_to_current;
    snap.keypoints_detected_current_frame = vo_out.keypoints_detected_current_frame;
    snap.matches_to_prior_frame           = vo_out.matches_to_prior_frame;
    snap.essential_inliers                = vo_out.essential_inliers;
    snap.median_parallax_pixels           = vo_out.median_parallax_pixels;
    snap.mean_reprojection_error_pixels   = vo_out.mean_reprojection_error_pixels;
    snap.last_estimation_ms               = vo_out.elapsed_ms;
    snap.estimation_count                 = s.odometer->GetPhysicalVisualOdometerFrameCount();

    // Monocular VO: scale unknown until a depth source anchors it. The
    // mapper consults this and refuses to update with non-metric poses.
    snap.pose_scale_state =
        (snap.tracking_state == PhysicalLocalizationTrackingState::Tracking)
            ? resolved_scale_state
            : PhysicalLocalizationPoseScaleState::Unknown;

    // Surface depth-fusion diagnostic — visible in tracking_reason when
    // tracking succeeded but scale stayed unscaled (e.g. depth model is
    // relative, or no depth map yet). Never overwrites a real failure
    // reason because that path didn't enter this branch.
    if (snap.tracking_state == PhysicalLocalizationTrackingState::Tracking
        && !scale_diagnostic.empty()
        && snap.pose_scale_state != PhysicalLocalizationPoseScaleState::ScaledByDepthMap)
    {
        snap.tracking_reason = scale_diagnostic;
    }

    StampPoseConvenienceFields(snap.pose_world_camera, snap.pose_scale_state);

    // Velocity (only when we have two consecutive published Tracking poses).
    const int64_t now_ns = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    if (snap.tracking_state == PhysicalLocalizationTrackingState::Tracking
        && !s.prev_published_T.empty() && s.prev_published_steady_ns > 0)
    {
        const double dt_s = (now_ns - s.prev_published_steady_ns) / 1e9;
        ComputeVelocityFromPoseDelta(s.prev_published_T,
                                     snap.pose_world_camera.T_world_camera,
                                     dt_s,
                                     snap.velocity_world_camera);
    }

    // ── Update occupancy grid (no-op when scale is non-metric) ────────────
    try {
        s.grid_mapper->RouteCameraPoseToPhysicalOccupancyGridMapper(
            snap.pose_world_camera.T_world_camera, snap.pose_scale_state);
    } catch (const std::exception& e) {
        // Non-fatal — log, leave the previous grid in place.
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG,
                  std::string("RouteCameraPoseToPhysicalOccupancyGridMapper threw: ") + e.what());
    }
    s.grid_mapper->CopyOccupancyGridSnapshot(snap.occupancy_grid);

    // ── Trajectory ring ───────────────────────────────────────────────────
    if (snap.tracking_state == PhysicalLocalizationTrackingState::Tracking) {
        s.trajectory_ring.emplace_back(
            snap.pose_world_camera.position_meters[0],
            snap.pose_world_camera.position_meters[1],
            snap.pose_world_camera.position_meters[2]);
        const auto cap = s.odometer->GetPhysicalVisualOdometerConfigSnapshot().max_trajectory_samples;
        while (static_cast<int>(s.trajectory_ring.size()) > cap) {
            s.trajectory_ring.pop_front();
        }
    }
    snap.trajectory_world_meters.assign(s.trajectory_ring.begin(), s.trajectory_ring.end());

    snap.built_at_steady_ns = now_ns;

    // ── Publish ───────────────────────────────────────────────────────────
    try {
        PhysicalLocalizationBus::Instance().PublishPhysicalLocalizationSnapshotToBus(snap);
        ++s.processed_count;
        if (snap.tracking_state == PhysicalLocalizationTrackingState::Tracking) {
            s.last_error_reason.clear();
            s.prev_published_T          = snap.pose_world_camera.T_world_camera.clone();
            s.prev_published_steady_ns  = now_ns;
        } else if (!snap.tracking_reason.empty()) {
            s.last_error_reason = snap.tracking_reason;
        }
    } catch (const std::exception& e) {
        s.last_error_reason =
            std::string("PublishPhysicalLocalizationSnapshotToBus failed: ") + e.what();
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, s.last_error_reason);
    }
}

void ShutdownPhysicalLocalization() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.shutting_down = true;
    if (s.odometer)    s.odometer->ResetPhysicalVisualOdometer();
    if (s.grid_mapper) s.grid_mapper->ResetPhysicalOccupancyGridMapper();
    s.odometer.reset();
    s.grid_mapper.reset();
    s.pending_odom_cfg.reset();
    s.pending_grid_cfg.reset();
    s.trajectory_ring.clear();
    s.prev_published_T.release();
    s.prev_published_steady_ns = 0;
    PhysicalLocalizationBus::Instance().ResetPhysicalLocalizationBus();
    s.initialized          = false;
    s.tick_count           = 0;
    s.processed_count      = 0;
    s.last_seen_frame_ctr  = 0;
    s.intrinsics_synced    = false;
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG, "ShutdownPhysicalLocalization: complete");
}

bool IsPhysicalLocalizationRunning() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.initialized && !s.shutting_down;
}

std::string GetLastPhysicalLocalizationError() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.last_error_reason;
}

uint64_t GetPhysicalLocalizationTickCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.tick_count;
}

uint64_t GetPhysicalLocalizationProcessedCount() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.processed_count;
}

PhysicalVisualOdometerConfig GetPhysicalLocalizationOdometerConfigSnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.odometer) return s.odometer->GetPhysicalVisualOdometerConfigSnapshot();
    if (s.pending_odom_cfg) return *s.pending_odom_cfg;
    return PhysicalVisualOdometerConfig{};
}

PhysicalOccupancyGridMapperConfig GetPhysicalLocalizationGridConfigSnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.grid_mapper) return s.grid_mapper->GetPhysicalOccupancyGridMapperConfigSnapshot();
    if (s.pending_grid_cfg) return *s.pending_grid_cfg;
    return PhysicalOccupancyGridMapperConfig{};
}

void RequestConfigurePhysicalLocalizationOdometer(const PhysicalVisualOdometerConfig& cfg) {
    // Validate OUTSIDE the lock by attempting to construct a throw-away
    // odometer with this cfg — guarantees the lock never wraps a throw.
    PhysicalVisualOdometer probe;
    probe.ConfigurePhysicalVisualOdometer(cfg);

    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.odometer) {
        s.odometer->ConfigurePhysicalVisualOdometer(cfg);
    } else {
        s.pending_odom_cfg = std::make_unique<PhysicalVisualOdometerConfig>(cfg);
    }
    s.last_error_reason.clear();
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "RequestConfigurePhysicalLocalizationOdometer: cfg accepted");
}

void RequestConfigurePhysicalLocalizationGrid(const PhysicalOccupancyGridMapperConfig& cfg) {
    PhysicalOccupancyGridMapper probe;
    probe.ConfigurePhysicalOccupancyGridMapper(cfg);

    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.grid_mapper) {
        s.grid_mapper->ConfigurePhysicalOccupancyGridMapper(cfg);
    } else {
        s.pending_grid_cfg = std::make_unique<PhysicalOccupancyGridMapperConfig>(cfg);
    }
    s.last_error_reason.clear();
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "RequestConfigurePhysicalLocalizationGrid: cfg accepted");
}

void RequestResetPhysicalLocalization() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.pending_reset = true;
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "RequestResetPhysicalLocalization: queued for next Tick");
}

}}} // namespace GRIM::Perception::Physical
