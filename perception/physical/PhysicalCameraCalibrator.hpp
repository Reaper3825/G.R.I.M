#pragma once

#include "PhysicalCalibrationPattern.hpp"
#include "PhysicalCalibrationStore.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalCameraCalibrator
//
//  Owns the per-process calibration state machine. Responsibilities:
//    - On first tick (lazy init), attempt to load any saved calibration from
//      the bootstrap path. If present and valid, the subsystem starts in
//      "Calibrated" state and downstream code can immediately call
//      UndistortBgrFrameUsingPhysicalCalibration().
//    - When capture is active, every TickPhysicalCameraCalibration() pulls
//      the latest frame from PhysicalFrameBus, runs lighting-adaptive
//      detection, and (if a pattern is found AND coverage policy permits)
//      adds an accepted sample to the pool.
//    - Exposes Request* mutators for the UI:
//        Start/Stop capture, Capture-Now, Run intrinsic calibration, Save,
//        Clear samples, Reconfigure pattern.
//    - All public functions are thread-safe.
// ─────────────────────────────────────────────────────────────────────────────

enum class PhysicalCalibrationStage : uint8_t {
    Uncalibrated     = 0,  // nothing loaded, no samples
    LoadedFromDisk   = 1,  // calibration loaded at startup; ready to undistort
    Capturing        = 2,  // collecting samples from the FrameBus
    Calibrated       = 3,  // ran calibrateCamera() at least once this session
    Failed           = 4   // last operation failed; see status_reason
};

// Snapshot the UI reads each frame.
struct PhysicalCalibrationStatus {
    PhysicalCalibrationStage stage = PhysicalCalibrationStage::Uncalibrated;

    // Pattern config
    int     pattern_inner_cols    = 9;
    int     pattern_inner_rows    = 6;
    float   pattern_square_meters = 0.025f;

    // Sample pool
    int     accepted_sample_count = 0;
    int     coverage_grid_cols    = 8;
    int     coverage_grid_rows    = 6;
    std::vector<int>           coverage_cell_counts;     // size = grid_cols*grid_rows
    int     coverage_cells_filled = 0;

    // Latest frame analysis (live, even when not capturing)
    bool        last_frame_present       = false;
    bool        last_pattern_found       = false;
    std::string last_detector_used;
    int         last_preprocess_path     = 0;
    double      last_frame_brightness    = 0.0;          // 0..255 grayscale mean
    int         last_frame_width         = 0;
    int         last_frame_height        = 0;
    cv::Point2f last_pattern_centroid_px = {0, 0};
    std::string last_failure_reason;

    // Calibration result (valid when stage is LoadedFromDisk or Calibrated)
    bool        has_calibration_data     = false;
    cv::Mat     camera_matrix;       // 3x3 CV_64F
    cv::Mat     dist_coeffs;         // 1xN CV_64F
    cv::Size    calibrated_image_size {0, 0};
    double      rms_reprojection_error  = 0.0;

    // Last error / status string for UI
    std::string status_reason;
};

// Single mainloop entry point for this module — called from inside
// TickPhysicalEnvironment(). Cheap; safe to call every frame. Lazy-inits.
void TickPhysicalCameraCalibration();

// Read the live status snapshot (deep copies the cv::Mats).
PhysicalCalibrationStatus GetPhysicalCalibrationStatusSnapshot();

// Returns true once we have a usable K matrix (loaded from disk or computed).
bool IsPhysicalCalibrationDataAvailable();

// Copies the current calibration into `out`. Throws if not available
// (Rule 20 — caller MUST check IsPhysicalCalibrationDataAvailable first).
void GetPhysicalCalibrationData(PhysicalCalibrationData& out);

// Service for downstream consumers (later: model context matrix). Throws if
// no calibration is loaded, or if `bgr_in` is empty/wrong type.
void UndistortBgrFrameUsingPhysicalCalibration(const cv::Mat& bgr_in, cv::Mat& bgr_out);

// ── UI-facing requests ──────────────────────────────────────────────────────

// Switch to the Capturing stage. Idempotent.
void RequestStartPhysicalCameraCalibrationCapture();

// Stop capture (samples are kept). Idempotent.
void RequestStopPhysicalCameraCalibrationCapture();

// Force-add the most recent FrameBus frame as a sample (if a pattern is
// currently visible). Throws on hard validation failures; soft "no pattern
// in current frame" sets status_reason and returns without throwing.
void RequestCapturePhysicalCalibrationSampleNow();

// Clear all collected samples and the coverage grid. Does NOT clear the
// already-loaded calibration (call Reset for that).
void RequestClearPhysicalCalibrationSamples();

// Run cv::calibrateCamera over the accepted samples. Throws if there are
// fewer than `min_samples_required` (=10) samples, or if calibration fails.
void RequestRunIntrinsicCalibrationFromSamples();

// Persist the current calibration to the bootstrap path. Throws if no
// calibration is available (nothing to save).
void RequestSavePhysicalCalibrationToDisk();

// Reload calibration from the bootstrap path. Returns true if a file was
// found and loaded. Throws on parse / shape errors.
bool RequestLoadPhysicalCalibrationFromDisk();

// Reconfigure the chessboard pattern. Throws on invalid dims. Clearing
// samples is the caller's responsibility — pattern dims must match all
// samples in the pool, so this also clears the sample pool to be safe.
void RequestReconfigurePhysicalCalibrationPattern(int   inner_cols,
                                                  int   inner_rows,
                                                  float square_meters);

// Drop everything (samples + loaded calibration). Used by tests / shutdown.
void ResetPhysicalCalibrationState();

}}} // namespace GRIM::Perception::Physical
