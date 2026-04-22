#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalCalibrationStore
//
//  On-disk persistence of intrinsic camera calibration. Single canonical
//  location under the GRIM root: <GRIM_ROOT>/data/perception/physical/
//  camera_calibration.json
//
//  The format is OpenCV FileStorage (JSON dialect). It captures camera
//  matrix K, distortion coefficients, the image size that was calibrated
//  against, the RMS reprojection error, the sample count used, and the
//  pattern definition (so we know what was used).
//
//  Save and Load both fail loud — they throw std::runtime_error with the
//  full path and the underlying error. Load returns false (and leaves out
//  untouched) when the file simply does not exist; that is a normal
//  startup case and not an error.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalCalibrationData {
    cv::Mat       camera_matrix;            // 3x3 CV_64F
    cv::Mat       dist_coeffs;              // 1xN CV_64F (5 or 8)
    cv::Size      image_size{0, 0};
    double        rms_reprojection_error = 0.0;
    int           sample_count           = 0;

    // Pattern provenance — what physical board produced this calibration.
    int           pattern_inner_cols     = 0;
    int           pattern_inner_rows     = 0;
    float         pattern_square_meters  = 0.0f;

    std::string   created_at_iso;          // e.g. "2026-04-21T18:32:11Z"
    std::string   source_url_at_capture;   // RTSP/HTTP URL used during capture
    std::string   source_label_at_capture;
};

// Returns the canonical absolute path used by Save/Load. Creates the parent
// directory chain if it does not yet exist (so that Save can succeed on a
// fresh checkout). Throws on filesystem errors.
std::string GetPhysicalCalibrationStorePath();

// Persist `data` to `path`. Throws on validation failure (e.g. K is wrong
// shape) or I/O failure.
void SavePhysicalCalibrationDataToPath(const PhysicalCalibrationData& data,
                                       const std::string&             path);

// Load `data` from `path`. Returns false if the file does not exist (caller
// treats that as "no calibration on disk yet"). Throws on parse / shape
// errors so we never silently load garbage.
bool LoadPhysicalCalibrationDataFromPath(const std::string&        path,
                                         PhysicalCalibrationData&  data);

}}} // namespace GRIM::Perception::Physical
