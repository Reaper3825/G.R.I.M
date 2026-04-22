#include "PhysicalCalibrationStore.hpp"

#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <opencv2/core/persistence.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifndef GRIM_ROOT_DIR
#error "GRIM_ROOT_DIR must be defined by the build system"
#endif

namespace fs = std::filesystem;

namespace GRIM { namespace Perception { namespace Physical {

namespace {

[[noreturn]] void ThrowWithLocation(const char* fn, const std::string& msg) {
    std::ostringstream ss;
    ss << fn << ": " << msg
       << " [" << __FILE__ << ":" << __LINE__ << "]";
    throw std::runtime_error(ss.str());
}

void ValidateCalibrationDataForPersistence(const PhysicalCalibrationData& d, const char* fn) {
    if (d.camera_matrix.empty() || d.camera_matrix.rows != 3 || d.camera_matrix.cols != 3
        || d.camera_matrix.type() != CV_64F) {
        ThrowWithLocation(fn,
            "camera_matrix must be 3x3 CV_64F (got "
            + std::to_string(d.camera_matrix.rows) + "x"
            + std::to_string(d.camera_matrix.cols)
            + " type=" + std::to_string(d.camera_matrix.type())
            + " empty=" + std::to_string(d.camera_matrix.empty()) + ")");
    }
    if (d.dist_coeffs.empty() || d.dist_coeffs.type() != CV_64F
        || (d.dist_coeffs.rows != 1 && d.dist_coeffs.cols != 1)) {
        ThrowWithLocation(fn,
            "dist_coeffs must be a non-empty CV_64F vector (got "
            + std::to_string(d.dist_coeffs.rows) + "x"
            + std::to_string(d.dist_coeffs.cols)
            + " type=" + std::to_string(d.dist_coeffs.type()) + ")");
    }
    if (d.image_size.width <= 0 || d.image_size.height <= 0) {
        ThrowWithLocation(fn,
            "image_size must be positive (got "
            + std::to_string(d.image_size.width) + "x"
            + std::to_string(d.image_size.height) + ")");
    }
    if (d.pattern_inner_cols < 2 || d.pattern_inner_rows < 2 || !(d.pattern_square_meters > 0.0f)) {
        ThrowWithLocation(fn,
            "pattern provenance is invalid (cols=" + std::to_string(d.pattern_inner_cols)
            + " rows=" + std::to_string(d.pattern_inner_rows)
            + " square_m=" + std::to_string(d.pattern_square_meters) + ")");
    }
}

} // anonymous

std::string GetPhysicalCalibrationStorePath() {
    fs::path base = fs::path(GRIM_ROOT_DIR) / "data" / "perception" / "physical";
    std::error_code ec;
    fs::create_directories(base, ec);
    if (ec) {
        ThrowWithLocation(__FUNCTION__,
            "failed to create directory '" + base.string() + "': " + ec.message());
    }
    return (base / "camera_calibration.json").string();
}

void SavePhysicalCalibrationDataToPath(const PhysicalCalibrationData& data,
                                       const std::string&             path) {
    ValidateCalibrationDataForPersistence(data, __FUNCTION__);

    // OpenCV's FileStorage chooses format from the file extension.
    cv::FileStorage fs_out(path, cv::FileStorage::WRITE);
    if (!fs_out.isOpened()) {
        ThrowWithLocation(__FUNCTION__,
            "cv::FileStorage failed to open for write: '" + path + "'");
    }

    fs_out << "schema_version"          << 1;
    fs_out << "camera_matrix"           << data.camera_matrix;
    fs_out << "dist_coeffs"             << data.dist_coeffs;
    fs_out << "image_width"             << data.image_size.width;
    fs_out << "image_height"            << data.image_size.height;
    fs_out << "rms_reprojection_error"  << data.rms_reprojection_error;
    fs_out << "sample_count"            << data.sample_count;
    fs_out << "pattern_inner_cols"      << data.pattern_inner_cols;
    fs_out << "pattern_inner_rows"      << data.pattern_inner_rows;
    fs_out << "pattern_square_meters"   << static_cast<double>(data.pattern_square_meters);
    fs_out << "created_at_iso"          << data.created_at_iso;
    fs_out << "source_url_at_capture"   << data.source_url_at_capture;
    fs_out << "source_label_at_capture" << data.source_label_at_capture;
    fs_out.release();

    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
        "SavePhysicalCalibrationDataToPath: wrote '" + path
        + "' (rms=" + std::to_string(data.rms_reprojection_error)
        + ", samples=" + std::to_string(data.sample_count) + ")");
}

bool LoadPhysicalCalibrationDataFromPath(const std::string&        path,
                                         PhysicalCalibrationData&  data) {
    std::error_code ec;
    if (!fs::exists(fs::path(path), ec) || ec) {
        return false; // legitimate "no calibration yet"
    }

    cv::FileStorage fs_in(path, cv::FileStorage::READ);
    if (!fs_in.isOpened()) {
        ThrowWithLocation(__FUNCTION__,
            "cv::FileStorage failed to open for read: '" + path + "'");
    }

    PhysicalCalibrationData tmp;

    int schema_version = 0;
    fs_in["schema_version"] >> schema_version;
    if (schema_version != 1) {
        ThrowWithLocation(__FUNCTION__,
            "unsupported schema_version " + std::to_string(schema_version)
            + " in '" + path + "' (expected 1)");
    }

    fs_in["camera_matrix"]  >> tmp.camera_matrix;
    fs_in["dist_coeffs"]    >> tmp.dist_coeffs;

    int w = 0, h = 0;
    fs_in["image_width"]  >> w;
    fs_in["image_height"] >> h;
    tmp.image_size = cv::Size(w, h);

    fs_in["rms_reprojection_error"] >> tmp.rms_reprojection_error;
    fs_in["sample_count"]           >> tmp.sample_count;
    fs_in["pattern_inner_cols"]     >> tmp.pattern_inner_cols;
    fs_in["pattern_inner_rows"]     >> tmp.pattern_inner_rows;

    double sq = 0.0;
    fs_in["pattern_square_meters"]  >> sq;
    tmp.pattern_square_meters = static_cast<float>(sq);

    fs_in["created_at_iso"]          >> tmp.created_at_iso;
    fs_in["source_url_at_capture"]   >> tmp.source_url_at_capture;
    fs_in["source_label_at_capture"] >> tmp.source_label_at_capture;
    fs_in.release();

    ValidateCalibrationDataForPersistence(tmp, __FUNCTION__);

    data = std::move(tmp);
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
        "LoadPhysicalCalibrationDataFromPath: loaded '" + path
        + "' (rms=" + std::to_string(data.rms_reprojection_error)
        + ", samples=" + std::to_string(data.sample_count)
        + ", " + std::to_string(data.image_size.width) + "x"
        + std::to_string(data.image_size.height) + ")");
    return true;
}

}}} // namespace
