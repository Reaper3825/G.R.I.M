#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalCalibrationPattern
//
//  Lighting-adaptive chessboard detection used by PhysicalCameraCalibrator.
//
//  The detector is required to work across the full lighting envelope
//  (brightness 0 → 100% as exposed in the UI; gray-mean 0..255 in pixels).
//  We achieve that with a per-frame preprocessing pipeline:
//
//    1. Convert to grayscale and measure gray-mean.
//    2. Apply gamma correction toward mid-gray (brighten very dark frames,
//       darken very bright ones) — pure pixel math, no auto-exposure
//       feedback to the camera.
//    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
//       with a clip-limit chosen from the brightness bucket.
//    4. Run cv::findChessboardCornersSB which is the modern, lighting-
//       robust detector. SB internally normalizes; our preprocessing only
//       helps the truly extreme cases (gray-mean < 20 or > 235).
//    5. If SB fails, fall back to legacy cv::findChessboardCorners with
//       adaptive threshold + cv::cornerSubPix refinement.
//
//  Every public function fails loud (throws std::runtime_error with a
//  message that names the function, the input shape/type, and the OpenCV
//  status). Callers are expected to log+rethrow or surface to UI; this
//  module never silently swallows.
// ─────────────────────────────────────────────────────────────────────────────

// One detected pattern instance.
struct DetectedCalibrationPattern {
    bool                       found            = false;   // overall detection result
    std::vector<cv::Point2f>   image_points;               // size == cols*rows when found
    cv::Point2f                centroid_px      = {0,0};   // mean of image_points
    double                     gray_mean        = 0.0;     // per-frame brightness used for adapt
    int                        preprocess_path  = 0;       // 0=raw, 1=clahe, 2=gamma+clahe, 3=legacy fallback
    std::string                detector_used;              // "SB" or "legacy"
    std::string                failure_reason;             // populated when found==false
};

// Compute the mean grayscale value of a BGR8 image. Throws if `bgr` is empty
// or not 3-channel (Rule 20).
double ComputeFrameBrightnessFromBgr(const cv::Mat& bgr);

// Run the lighting-adaptive preprocessing pipeline. `bgr_in` MUST be CV_8UC3.
// Writes a CV_8UC1 image into `gray_out`. Throws if input is invalid.
// `out_brightness` and `out_path` describe what was done (for diagnostics).
void PreprocessFrameForLightingRobustDetection(const cv::Mat& bgr_in,
                                               cv::Mat&        gray_out,
                                               double&         out_brightness,
                                               int&            out_path);

// Build the canonical 3D object-point list for a `cols x rows` inner-corner
// chessboard with squares of `square_meters`. Order matches OpenCV's
// row-major corner ordering. Throws if cols/rows < 2 or square_meters <= 0.
std::vector<cv::Point3f> BuildObjectPointsForChessboard(int   cols,
                                                        int   rows,
                                                        float square_meters);

// Detect a chessboard in `bgr` with the inner-corner layout `cols x rows`.
// Always populates `out` (sets `found=false` + `failure_reason` when no
// pattern is visible). Throws ONLY for programmer/usage errors (empty input,
// wrong type, illegal pattern dims) — a "no chessboard in this frame" is
// not an error and does not throw.
void DetectChessboardCornersInBgrFrame(const cv::Mat&               bgr,
                                       int                          cols,
                                       int                          rows,
                                       DetectedCalibrationPattern&  out);

// Render the detected corners onto a BGR8 copy for UI overlay. `bgr_in` is
// not modified; `bgr_out` is set to a fresh clone with corners drawn.
// Throws on input validity errors.
void DrawDetectedCalibrationPatternOnBgr(const cv::Mat&                       bgr_in,
                                         const DetectedCalibrationPattern&    pattern,
                                         int                                  cols,
                                         int                                  rows,
                                         cv::Mat&                             bgr_out);

}}} // namespace GRIM::Perception::Physical
