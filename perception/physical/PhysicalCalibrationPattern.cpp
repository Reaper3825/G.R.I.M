#include "PhysicalCalibrationPattern.hpp"

#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
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

// Apply gamma in-place on a CV_8UC1 image using a 256-entry LUT.
// gamma > 1.0 darkens, gamma < 1.0 brightens.
void ApplyGammaCorrectionInPlace(cv::Mat& gray, double gamma) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        ThrowWithLocation(__FUNCTION__,
            "input must be non-empty CV_8UC1 (got type=" + std::to_string(gray.type())
            + ", empty=" + std::to_string(gray.empty()) + ")");
    }
    if (!(gamma > 0.0)) {
        ThrowWithLocation(__FUNCTION__,
            "gamma must be > 0 (got " + std::to_string(gamma) + ")");
    }
    cv::Mat lut(1, 256, CV_8UC1);
    auto*   p = lut.ptr<uint8_t>();
    for (int i = 0; i < 256; ++i) {
        const double v = std::pow(static_cast<double>(i) / 255.0, gamma) * 255.0;
        const int    c = static_cast<int>(v + 0.5);
        p[i] = static_cast<uint8_t>(c < 0 ? 0 : (c > 255 ? 255 : c));
    }
    cv::LUT(gray, lut, gray);
}

} // anonymous

double ComputeFrameBrightnessFromBgr(const cv::Mat& bgr) {
    if (bgr.empty()) {
        ThrowWithLocation(__FUNCTION__, "bgr is empty");
    }
    if (bgr.type() != CV_8UC3) {
        ThrowWithLocation(__FUNCTION__,
            "expected CV_8UC3, got type=" + std::to_string(bgr.type()));
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    const cv::Scalar m = cv::mean(gray);
    return m[0];
}

void PreprocessFrameForLightingRobustDetection(const cv::Mat& bgr_in,
                                               cv::Mat&        gray_out,
                                               double&         out_brightness,
                                               int&            out_path) {
    if (bgr_in.empty()) {
        ThrowWithLocation(__FUNCTION__, "bgr_in is empty");
    }
    if (bgr_in.type() != CV_8UC3) {
        ThrowWithLocation(__FUNCTION__,
            "expected CV_8UC3, got type=" + std::to_string(bgr_in.type()));
    }

    cv::cvtColor(bgr_in, gray_out, cv::COLOR_BGR2GRAY);
    out_brightness = cv::mean(gray_out)[0];

    // Bucketed adaptation. Numbers are picked so that brightness=0 and
    // brightness=255 both end up in the working range (~80..160) before CLAHE.
    if (out_brightness < 20.0) {
        // Very dark — pull mid-gray up hard, then strong CLAHE.
        ApplyGammaCorrectionInPlace(gray_out, 0.45);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8,8));
        clahe->apply(gray_out, gray_out);
        out_path = 2;
    } else if (out_brightness < 40.0) {
        ApplyGammaCorrectionInPlace(gray_out, 0.65);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8,8));
        clahe->apply(gray_out, gray_out);
        out_path = 2;
    } else if (out_brightness > 215.0) {
        // Very bright — pull mid-gray down hard, then mild CLAHE.
        ApplyGammaCorrectionInPlace(gray_out, 1.8);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        clahe->apply(gray_out, gray_out);
        out_path = 2;
    } else if (out_brightness > 195.0) {
        ApplyGammaCorrectionInPlace(gray_out, 1.4);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        clahe->apply(gray_out, gray_out);
        out_path = 2;
    } else {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        clahe->apply(gray_out, gray_out);
        out_path = 1;
    }
}

std::vector<cv::Point3f> BuildObjectPointsForChessboard(int   cols,
                                                        int   rows,
                                                        float square_meters) {
    if (cols < 2 || rows < 2) {
        ThrowWithLocation(__FUNCTION__,
            "cols and rows must each be >= 2 (got cols=" + std::to_string(cols)
            + ", rows=" + std::to_string(rows) + ")");
    }
    if (!(square_meters > 0.0f)) {
        ThrowWithLocation(__FUNCTION__,
            "square_meters must be > 0 (got " + std::to_string(square_meters) + ")");
    }
    std::vector<cv::Point3f> pts;
    pts.reserve(static_cast<size_t>(cols) * rows);
    // Z=0 plane, row-major to match OpenCV's corner output order.
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            pts.emplace_back(static_cast<float>(c) * square_meters,
                             static_cast<float>(r) * square_meters,
                             0.0f);
        }
    }
    return pts;
}

void DetectChessboardCornersInBgrFrame(const cv::Mat&               bgr,
                                       int                          cols,
                                       int                          rows,
                                       DetectedCalibrationPattern&  out) {
    out = {};
    if (bgr.empty()) {
        ThrowWithLocation(__FUNCTION__, "bgr is empty");
    }
    if (bgr.type() != CV_8UC3) {
        ThrowWithLocation(__FUNCTION__,
            "expected CV_8UC3, got type=" + std::to_string(bgr.type()));
    }
    if (cols < 2 || rows < 2) {
        ThrowWithLocation(__FUNCTION__,
            "cols and rows must each be >= 2 (got cols=" + std::to_string(cols)
            + ", rows=" + std::to_string(rows) + ")");
    }

    cv::Mat raw_gray;
    cv::cvtColor(bgr, raw_gray, cv::COLOR_BGR2GRAY);
    out.gray_mean = cv::mean(raw_gray)[0];
    const cv::Size pattern(cols, rows);
    std::vector<cv::Point2f> corners;

    const int sb_flags = cv::CALIB_CB_NORMALIZE_IMAGE
                       | cv::CALIB_CB_EXHAUSTIVE
                       | cv::CALIB_CB_ACCURACY;
    auto try_sb = [&](const cv::Mat& gray, int path,
                      float coordinate_scale = 1.0f) {
        corners.clear();
        bool detected = cv::findChessboardCornersSB(
            gray, pattern, corners, sb_flags);
        if (detected && coordinate_scale != 1.0f) {
            for (auto& corner : corners) corner *= coordinate_scale;
        }
        if (detected) {
            out.detector_used = path == 3 ? "SB-upscaled" : "SB";
            out.preprocess_path = path;
        }
        return detected;
    };

    // Preserve printed edge contrast on the primary path. SB already performs
    // its own normalization, so unconditional CLAHE is counterproductive.
    bool found = false;
    try {
        found = try_sb(raw_gray, 0);
    } catch (const cv::Exception& e) {
        // Treat as a non-detection but log it once — SB shouldn't throw on
        // valid input, so this is interesting.
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  std::string("DetectChessboardCornersInBgrFrame: SB threw: ") + e.what());
        found = false;
    }

    cv::Mat enhanced_gray;
    int enhanced_path = 0;
    if (!found) {
        double measured_brightness = 0.0;
        PreprocessFrameForLightingRobustDetection(
            bgr, enhanced_gray, measured_brightness, enhanced_path);
        try {
            found = try_sb(enhanced_gray, enhanced_path);
        } catch (const cv::Exception& e) {
            LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                std::string("DetectChessboardCornersInBgrFrame: enhanced SB threw: ")
                + e.what());
            found = false;
        }
    }

    // At VGA-like resolutions a distant printed board can have corners too
    // small for stable localization. A 1.5x cubic retry improves sampling but
    // returns points in the original raw-frame coordinate system.
    if (!found && bgr.cols <= 1280 && bgr.rows <= 720) {
        try {
            cv::Mat enlarged;
            cv::resize(raw_gray, enlarged, cv::Size(), 1.5, 1.5,
                       cv::INTER_CUBIC);
            found = try_sb(enlarged, 3, 1.0f / 1.5f);
        } catch (const cv::Exception& e) {
            LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                std::string("DetectChessboardCornersInBgrFrame: upscaled SB threw: ")
                + e.what());
            found = false;
        }
    }

    auto try_legacy = [&](const cv::Mat& gray, int path) {
        corners.clear();
        const int legacy_flags = cv::CALIB_CB_ADAPTIVE_THRESH
                               | cv::CALIB_CB_NORMALIZE_IMAGE;
        const bool detected = cv::findChessboardCorners(
            gray, pattern, corners, legacy_flags);
        if (detected) {
            cv::cornerSubPix(
                gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 1e-3));
            out.detector_used   = "legacy";
            out.preprocess_path = path;
        }
        return detected;
    };
    if (!found) found = try_legacy(raw_gray, 4);
    if (!found && !enhanced_gray.empty()) {
        found = try_legacy(enhanced_gray, 5);
    }

    if (!found) {
        out.found          = false;
        out.failure_reason = "no complete " + std::to_string(cols) + "x"
                           + std::to_string(rows) + " inner-corner pattern found"
                             " (printed board must be " + std::to_string(cols + 1)
                           + "x" + std::to_string(rows + 1)
                           + " squares, brightness=" + std::to_string(out.gray_mean) + ")";
        return;
    }

    if (corners.size() != static_cast<size_t>(cols) * rows) {
        // Detector returned success but wrong corner count — fail loud.
        ThrowWithLocation(__FUNCTION__,
            "detector returned wrong corner count: got "
            + std::to_string(corners.size())
            + ", expected " + std::to_string(static_cast<size_t>(cols) * rows));
    }

    out.image_points = std::move(corners);
    cv::Point2f sum(0, 0);
    for (const auto& p : out.image_points) sum += p;
    out.centroid_px = sum * (1.0f / static_cast<float>(out.image_points.size()));
    out.found       = true;
}

void DrawDetectedCalibrationPatternOnBgr(const cv::Mat&                       bgr_in,
                                         const DetectedCalibrationPattern&    pattern,
                                         int                                  cols,
                                         int                                  rows,
                                         cv::Mat&                             bgr_out) {
    if (bgr_in.empty() || bgr_in.type() != CV_8UC3) {
        ThrowWithLocation(__FUNCTION__,
            "bgr_in must be non-empty CV_8UC3");
    }
    bgr_out = bgr_in.clone();
    if (!pattern.found || pattern.image_points.empty()) return;
    cv::drawChessboardCorners(bgr_out, cv::Size(cols, rows), pattern.image_points, true);
}

}}} // namespace
