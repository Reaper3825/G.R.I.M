#include "PhysicalCameraCalibrator.hpp"

#include "PhysicalEnvironmentLogTag.hpp"
#include "PhysicalFrameBus.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

[[noreturn]] void ThrowWithLocation(const char* fn, const std::string& msg) {
    std::ostringstream ss;
    ss << fn << ": " << msg
       << " [" << __FILE__ << ":" << __LINE__ << "]";
    throw std::runtime_error(ss.str());
}

std::string FormatTimestampUtcIso() {
    const auto t  = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(t);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &tt);
#else
    gmtime_r(&tt, &tm_buf);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// Auto-capture cadence: we accept at most one sample per N milliseconds, and
// only if the centroid lands in a coverage cell that hasn't been filled yet.
constexpr int kAutoCaptureMinIntervalMs = 700;
constexpr int kCoverageGridCols          = 8;
constexpr int kCoverageGridRows          = 6;
constexpr int kMinSamplesRequired        = 10;
constexpr int kMaxSamplesPerCell         = 1;

struct PhysicalCalibrationModuleState {
    std::mutex                                          mutex;
    bool                                                initialized   = false;

    // Pattern config (mutable from UI)
    int                                                 pattern_inner_cols    = 9;
    int                                                 pattern_inner_rows    = 6;
    float                                               pattern_square_meters = 0.025f;

    // Sample pool
    std::vector<std::vector<cv::Point2f>>               accepted_image_points;
    cv::Size                                            samples_image_size {0, 0};
    std::vector<int>                                    coverage_cell_counts;  // resized lazily
    std::chrono::steady_clock::time_point               last_accept_time {};
    bool                                                capturing = false;

    // Last frame analysis (kept for UI live readout even when not capturing)
    DetectedCalibrationPattern                          last_detection;
    bool                                                last_frame_present = false;
    int                                                 last_frame_width   = 0;
    int                                                 last_frame_height  = 0;

    // Latest pulled FrameBus content
    PhysicalFrameBus::FrameView                         pull_view;
    uint64_t                                            last_seen_counter  = 0;

    // Calibration result
    PhysicalCalibrationData                             calib;
    bool                                                calib_valid        = false;
    PhysicalCalibrationStage                            stage = PhysicalCalibrationStage::Uncalibrated;
    std::string                                         status_reason;
};

PhysicalCalibrationModuleState& GetModule() {
    static PhysicalCalibrationModuleState s;
    return s;
}

void EnsureCoverageGridSizedLocked(PhysicalCalibrationModuleState& s) {
    const size_t want = static_cast<size_t>(kCoverageGridCols) * kCoverageGridRows;
    if (s.coverage_cell_counts.size() != want) {
        s.coverage_cell_counts.assign(want, 0);
    }
}

int CoverageCellIndexForPointLocked(const PhysicalCalibrationModuleState& s,
                                    const cv::Point2f& p,
                                    int                img_w,
                                    int                img_h) {
    if (img_w <= 0 || img_h <= 0) return -1;
    int cx = static_cast<int>((p.x / static_cast<float>(img_w)) * kCoverageGridCols);
    int cy = static_cast<int>((p.y / static_cast<float>(img_h)) * kCoverageGridRows);
    if (cx < 0) cx = 0; else if (cx >= kCoverageGridCols) cx = kCoverageGridCols - 1;
    if (cy < 0) cy = 0; else if (cy >= kCoverageGridRows) cy = kCoverageGridRows - 1;
    return cy * kCoverageGridCols + cx;
}

void LazyInitLocked(PhysicalCalibrationModuleState& s) {
    if (s.initialized) return;
    s.initialized = true;
    EnsureCoverageGridSizedLocked(s);

    PhysicalCalibrationData on_disk;
    bool loaded = false;
    try {
        loaded = LoadPhysicalCalibrationDataFromPath(GetPhysicalCalibrationStorePath(), on_disk);
    } catch (const std::exception& e) {
        s.stage         = PhysicalCalibrationStage::Failed;
        s.status_reason = std::string("LazyInit: load failed: ") + e.what();
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG, s.status_reason);
        return;
    }

    if (loaded) {
        s.calib                  = std::move(on_disk);
        s.calib_valid            = true;
        s.stage                  = PhysicalCalibrationStage::LoadedFromDisk;
        s.pattern_inner_cols     = s.calib.pattern_inner_cols;
        s.pattern_inner_rows     = s.calib.pattern_inner_rows;
        s.pattern_square_meters  = s.calib.pattern_square_meters;
        s.status_reason          = "loaded calibration from disk";
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraCalibrator: lazy-init loaded prior calibration "
                  "(rms=" + std::to_string(s.calib.rms_reprojection_error)
                  + ", samples=" + std::to_string(s.calib.sample_count) + ")");
    } else {
        s.stage         = PhysicalCalibrationStage::Uncalibrated;
        s.status_reason = "no prior calibration on disk; capture samples to begin";
    }
}

// Pull a fresh frame view from the FrameBus (under module lock).
bool PullLatestFrameLocked(PhysicalCalibrationModuleState& s) {
    PhysicalFrameBus::FrameView v;
    if (PhysicalFrameBus::Instance().PullLatestFrameView(v, s.last_seen_counter)) {
        s.pull_view          = std::move(v);
        s.last_frame_present = !s.pull_view.raw_image.empty();
        if (s.last_frame_present) {
            s.last_frame_width  = s.pull_view.raw_image.cols;
            s.last_frame_height = s.pull_view.raw_image.rows;
        }
        return true;
    }
    return false;
}

void RunDetectionOnPulledFrameLocked(PhysicalCalibrationModuleState& s) {
    if (!s.last_frame_present || s.pull_view.raw_image.empty()) return;
    try {
        DetectChessboardCornersInBgrFrame(
            s.pull_view.raw_image,
            s.pattern_inner_cols, s.pattern_inner_rows,
            s.last_detection);
    } catch (const std::exception& e) {
        s.last_detection.found          = false;
        s.last_detection.failure_reason = std::string("detection threw: ") + e.what();
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                  std::string("RunDetectionOnPulledFrame: ") + e.what());
    }
}

bool TryAddSampleLocked(PhysicalCalibrationModuleState& s,
                        bool                            allow_duplicate_cell,
                        std::string&                    out_reason) {
    if (!s.last_detection.found || s.last_detection.image_points.empty()) {
        out_reason = "no chessboard pattern in current frame";
        return false;
    }
    if (!s.last_frame_present) {
        out_reason = "no frame available";
        return false;
    }

    const cv::Size img_sz(s.last_frame_width, s.last_frame_height);
    if (!s.accepted_image_points.empty() && img_sz != s.samples_image_size) {
        out_reason = "frame size " + std::to_string(img_sz.width) + "x"
                   + std::to_string(img_sz.height)
                   + " does not match prior samples "
                   + std::to_string(s.samples_image_size.width) + "x"
                   + std::to_string(s.samples_image_size.height)
                   + " — clear samples to switch sources";
        return false;
    }

    EnsureCoverageGridSizedLocked(s);
    const int cell = CoverageCellIndexForPointLocked(
        s, s.last_detection.centroid_px, img_sz.width, img_sz.height);
    if (cell < 0 || cell >= static_cast<int>(s.coverage_cell_counts.size())) {
        out_reason = "computed coverage cell out of range";
        return false;
    }
    if (!allow_duplicate_cell && s.coverage_cell_counts[cell] >= kMaxSamplesPerCell) {
        out_reason = "coverage cell already filled (move the board to another image region)";
        return false;
    }

    s.accepted_image_points.push_back(s.last_detection.image_points);
    s.samples_image_size = img_sz;
    s.coverage_cell_counts[cell] += 1;
    s.last_accept_time = std::chrono::steady_clock::now();
    out_reason = "accepted (count=" + std::to_string(s.accepted_image_points.size()) + ")";
    return true;
}

void AutoCaptureIfDueLocked(PhysicalCalibrationModuleState& s) {
    if (!s.capturing) return;
    if (!s.last_detection.found) return;
    const auto now = std::chrono::steady_clock::now();
    const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - s.last_accept_time).count();
    if (ms < kAutoCaptureMinIntervalMs) return;
    std::string reason;
    if (TryAddSampleLocked(s, /*allow_duplicate_cell=*/false, reason)) {
        s.status_reason = "auto-captured: " + reason;
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraCalibrator: " + s.status_reason);
    }
}

} // anonymous

// ─────────────────────────────────────────────────────────────────────────────

void TickPhysicalCameraCalibration() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    LazyInitLocked(s);
    PullLatestFrameLocked(s);
    RunDetectionOnPulledFrameLocked(s);
    AutoCaptureIfDueLocked(s);
}

PhysicalCalibrationStatus GetPhysicalCalibrationStatusSnapshot() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    PhysicalCalibrationStatus out;
    out.stage                  = s.stage;
    out.pattern_inner_cols     = s.pattern_inner_cols;
    out.pattern_inner_rows     = s.pattern_inner_rows;
    out.pattern_square_meters  = s.pattern_square_meters;
    out.accepted_sample_count  = static_cast<int>(s.accepted_image_points.size());
    out.coverage_grid_cols     = kCoverageGridCols;
    out.coverage_grid_rows     = kCoverageGridRows;
    out.coverage_cell_counts   = s.coverage_cell_counts;
    out.coverage_cells_filled  = 0;
    for (int c : out.coverage_cell_counts) if (c > 0) ++out.coverage_cells_filled;

    out.last_frame_present     = s.last_frame_present;
    out.last_pattern_found     = s.last_detection.found;
    out.last_detector_used     = s.last_detection.detector_used;
    out.last_preprocess_path   = s.last_detection.preprocess_path;
    out.last_frame_brightness  = s.last_detection.gray_mean;
    out.last_frame_width       = s.last_frame_width;
    out.last_frame_height      = s.last_frame_height;
    out.last_pattern_centroid_px = s.last_detection.centroid_px;
    out.last_failure_reason    = s.last_detection.failure_reason;

    out.has_calibration_data   = s.calib_valid;
    if (s.calib_valid) {
        out.camera_matrix          = s.calib.camera_matrix.clone();
        out.dist_coeffs            = s.calib.dist_coeffs.clone();
        out.calibrated_image_size  = s.calib.image_size;
        out.rms_reprojection_error = s.calib.rms_reprojection_error;
    }
    out.status_reason = s.status_reason;
    return out;
}

bool IsPhysicalCalibrationDataAvailable() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.calib_valid;
}

void GetPhysicalCalibrationData(PhysicalCalibrationData& out) {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.calib_valid) {
        ThrowWithLocation(__FUNCTION__,
            "no calibration data is loaded — caller MUST check "
            "IsPhysicalCalibrationDataAvailable() first");
    }
    out = s.calib;
    out.camera_matrix = s.calib.camera_matrix.clone();
    out.dist_coeffs   = s.calib.dist_coeffs.clone();
}

void UndistortBgrFrameUsingPhysicalCalibration(const cv::Mat& bgr_in, cv::Mat& bgr_out) {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.calib_valid) {
        ThrowWithLocation(__FUNCTION__, "no calibration data loaded");
    }
    if (bgr_in.empty()) {
        ThrowWithLocation(__FUNCTION__, "bgr_in is empty");
    }
    if (bgr_in.type() != CV_8UC3) {
        ThrowWithLocation(__FUNCTION__,
            "expected CV_8UC3, got type=" + std::to_string(bgr_in.type()));
    }
    cv::undistort(bgr_in, bgr_out, s.calib.camera_matrix, s.calib.dist_coeffs);
}

// ── Requests ────────────────────────────────────────────────────────────────

void RequestStartPhysicalCameraCalibrationCapture() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    LazyInitLocked(s);
    s.capturing     = true;
    s.stage         = PhysicalCalibrationStage::Capturing;
    s.status_reason = "capture started — point the chessboard at the camera";
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, "RequestStartPhysicalCameraCalibrationCapture");
}

void RequestStopPhysicalCameraCalibrationCapture() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.capturing = false;
    if (s.calib_valid) {
        s.stage = PhysicalCalibrationStage::Calibrated;
    } else {
        s.stage = PhysicalCalibrationStage::Uncalibrated;
    }
    s.status_reason = "capture stopped (samples retained)";
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, "RequestStopPhysicalCameraCalibrationCapture");
}

void RequestCapturePhysicalCalibrationSampleNow() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    LazyInitLocked(s);
    PullLatestFrameLocked(s);
    RunDetectionOnPulledFrameLocked(s);
    std::string reason;
    const bool ok = TryAddSampleLocked(s, /*allow_duplicate_cell=*/true, reason);
    s.status_reason = (ok ? "manual capture: " : "manual capture rejected: ") + reason;
    if (ok) {
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "RequestCapturePhysicalCalibrationSampleNow: " + s.status_reason);
    } else {
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                  "RequestCapturePhysicalCalibrationSampleNow: " + s.status_reason);
    }
}

void RequestClearPhysicalCalibrationSamples() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.accepted_image_points.clear();
    s.samples_image_size = cv::Size(0, 0);
    EnsureCoverageGridSizedLocked(s);
    std::fill(s.coverage_cell_counts.begin(), s.coverage_cell_counts.end(), 0);
    s.last_accept_time = std::chrono::steady_clock::time_point{};
    s.status_reason    = "samples cleared";
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, "RequestClearPhysicalCalibrationSamples");
}

void RequestRunIntrinsicCalibrationFromSamples() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    LazyInitLocked(s);

    if (static_cast<int>(s.accepted_image_points.size()) < kMinSamplesRequired) {
        ThrowWithLocation(__FUNCTION__,
            "need at least " + std::to_string(kMinSamplesRequired)
            + " samples; have " + std::to_string(s.accepted_image_points.size()));
    }
    if (s.samples_image_size.width <= 0 || s.samples_image_size.height <= 0) {
        ThrowWithLocation(__FUNCTION__,
            "samples_image_size is invalid: "
            + std::to_string(s.samples_image_size.width) + "x"
            + std::to_string(s.samples_image_size.height));
    }

    const auto object_points_one = BuildObjectPointsForChessboard(
        s.pattern_inner_cols, s.pattern_inner_rows, s.pattern_square_meters);

    std::vector<std::vector<cv::Point3f>> object_points;
    object_points.reserve(s.accepted_image_points.size());
    for (size_t i = 0; i < s.accepted_image_points.size(); ++i) {
        object_points.push_back(object_points_one);
    }

    cv::Mat K, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = 0.0;
    try {
        rms = cv::calibrateCamera(
            object_points, s.accepted_image_points, s.samples_image_size,
            K, dist, rvecs, tvecs,
            cv::CALIB_RATIONAL_MODEL); // 8-coef distortion model
    } catch (const cv::Exception& e) {
        s.stage         = PhysicalCalibrationStage::Failed;
        s.status_reason = std::string("calibrateCamera threw: ") + e.what();
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG, s.status_reason);
        ThrowWithLocation(__FUNCTION__, std::string("OpenCV calibrateCamera failed: ") + e.what());
    }

    if (K.empty() || K.rows != 3 || K.cols != 3 || dist.empty()) {
        s.stage         = PhysicalCalibrationStage::Failed;
        s.status_reason = "calibrateCamera returned malformed K or dist";
        ThrowWithLocation(__FUNCTION__, s.status_reason);
    }
    if (!std::isfinite(rms) || rms <= 0.0 || rms > 5.0) {
        // Soft warning, not a throw — but record it. RMS > 5 px is suspicious
        // for any reasonable calibration; we still store the result so the UI
        // can show what we got.
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                  "RequestRunIntrinsicCalibrationFromSamples: suspicious RMS="
                  + std::to_string(rms) + " (expected < 1.0 for a good calibration)");
    }

    K.convertTo(K,    CV_64F);
    dist.convertTo(dist, CV_64F);

    s.calib                          = {};
    s.calib.camera_matrix            = K;
    s.calib.dist_coeffs              = dist;
    s.calib.image_size               = s.samples_image_size;
    s.calib.rms_reprojection_error   = rms;
    s.calib.sample_count             = static_cast<int>(s.accepted_image_points.size());
    s.calib.pattern_inner_cols       = s.pattern_inner_cols;
    s.calib.pattern_inner_rows       = s.pattern_inner_rows;
    s.calib.pattern_square_meters    = s.pattern_square_meters;
    s.calib.created_at_iso           = FormatTimestampUtcIso();
    s.calib.source_url_at_capture    = s.pull_view.source_url;
    s.calib.source_label_at_capture  = s.pull_view.source_label;
    s.calib_valid                    = true;
    s.stage                          = PhysicalCalibrationStage::Calibrated;
    s.status_reason                  = "calibration complete (rms="
                                     + std::to_string(rms) + ")";
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "RequestRunIntrinsicCalibrationFromSamples: " + s.status_reason);
}

void RequestSavePhysicalCalibrationToDisk() {
    PhysicalCalibrationData snapshot;
    {
        auto& s = GetModule();
        std::lock_guard<std::mutex> lk(s.mutex);
        if (!s.calib_valid) {
            ThrowWithLocation(__FUNCTION__,
                "no calibration available — run RequestRunIntrinsicCalibrationFromSamples first");
        }
        snapshot = s.calib;
        snapshot.camera_matrix = s.calib.camera_matrix.clone();
        snapshot.dist_coeffs   = s.calib.dist_coeffs.clone();
    }
    SavePhysicalCalibrationDataToPath(snapshot, GetPhysicalCalibrationStorePath());
    {
        auto& s = GetModule();
        std::lock_guard<std::mutex> lk(s.mutex);
        s.status_reason = "calibration saved to '" + GetPhysicalCalibrationStorePath() + "'";
    }
}

bool RequestLoadPhysicalCalibrationFromDisk() {
    PhysicalCalibrationData data;
    const bool loaded = LoadPhysicalCalibrationDataFromPath(
        GetPhysicalCalibrationStorePath(), data);
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (loaded) {
        s.calib                  = std::move(data);
        s.calib_valid            = true;
        s.pattern_inner_cols     = s.calib.pattern_inner_cols;
        s.pattern_inner_rows     = s.calib.pattern_inner_rows;
        s.pattern_square_meters  = s.calib.pattern_square_meters;
        s.stage                  = PhysicalCalibrationStage::LoadedFromDisk;
        s.status_reason          = "reloaded calibration from disk";
    } else {
        s.status_reason = "no calibration file on disk at '"
                        + GetPhysicalCalibrationStorePath() + "'";
    }
    return loaded;
}

void RequestReconfigurePhysicalCalibrationPattern(int   inner_cols,
                                                  int   inner_rows,
                                                  float square_meters) {
    if (inner_cols < 2 || inner_rows < 2 || !(square_meters > 0.0f)) {
        ThrowWithLocation(__FUNCTION__,
            "invalid pattern (cols=" + std::to_string(inner_cols)
            + ", rows=" + std::to_string(inner_rows)
            + ", square_m=" + std::to_string(square_meters) + ")");
    }
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.pattern_inner_cols    = inner_cols;
    s.pattern_inner_rows    = inner_rows;
    s.pattern_square_meters = square_meters;
    s.accepted_image_points.clear();
    s.samples_image_size    = cv::Size(0, 0);
    EnsureCoverageGridSizedLocked(s);
    std::fill(s.coverage_cell_counts.begin(), s.coverage_cell_counts.end(), 0);
    s.status_reason = "pattern reconfigured (samples cleared)";
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "RequestReconfigurePhysicalCalibrationPattern: cols=" + std::to_string(inner_cols)
              + " rows=" + std::to_string(inner_rows)
              + " square_m=" + std::to_string(square_meters));
}

void ResetPhysicalCalibrationState() {
    auto& s = GetModule();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.accepted_image_points.clear();
    s.samples_image_size = cv::Size(0, 0);
    s.coverage_cell_counts.clear();
    s.calib       = {};
    s.calib_valid = false;
    s.capturing   = false;
    s.stage       = PhysicalCalibrationStage::Uncalibrated;
    s.status_reason.clear();
    s.last_detection = {};
    s.last_frame_present = false;
    s.last_seen_counter  = 0;
    s.initialized        = false; // next tick will lazy-init again
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, "ResetPhysicalCalibrationState");
}

}}} // namespace
