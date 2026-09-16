#include "PhysicalCameraFocusController.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

double ComputeFocusScore(const cv::Mat& raw_bgr) {
    if (raw_bgr.empty()) {
        throw std::runtime_error(
            "ComputeFocusScore: raw camera frame is empty");
    }

    cv::Mat small;
    constexpr int kScoreWidth = 320;
    if (raw_bgr.cols > kScoreWidth) {
        const double scale = static_cast<double>(kScoreWidth)
                           / static_cast<double>(raw_bgr.cols);
        cv::resize(raw_bgr, small, cv::Size{}, scale, scale, cv::INTER_AREA);
    } else {
        small = raw_bgr;
    }

    cv::Mat gray;
    if (small.channels() == 3) {
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    } else if (small.channels() == 4) {
        cv::cvtColor(small, gray, cv::COLOR_BGRA2GRAY);
    } else if (small.channels() == 1) {
        gray = small;
    } else {
        throw std::runtime_error(
            "ComputeFocusScore: camera frame must have 1, 3, or 4 channels");
    }

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(laplacian, mean, deviation);
    return deviation[0] * deviation[0];
}

std::string ComposeSummary(const PhysicalCameraFocusStatus& status) {
    std::ostringstream out;
    if (!status.configuration_attempted) {
        out << "driver defaults";
    } else if (status.manual_focus_requested) {
        out << "manual focus=" << status.requested_focus
            << (status.manual_focus_set_accepted ? " accepted" : " rejected");
    } else if (status.autofocus_requested) {
        out << "autofocus " << (status.autofocus_enabled ? "on" : "off")
            << (status.autofocus_set_accepted ? " accepted" : " rejected");
    }
    out << " negotiated(auto=" << status.negotiated_autofocus
        << ",focus=" << status.negotiated_focus << ')';
    if (status.calibration_focus_locked) {
        out << (status.calibration_lock_accepted
                    ? " calibration-lock"
                    : " calibration-lock-rejected");
    }
    return out.str();
}

} // namespace

void PhysicalCameraFocusController::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = PhysicalCameraFocusStatus{};
    configured_focus_ = PhysicalCameraFocusConfig{};
    have_configuration_ = false;
    autofocus_before_calibration_lock_ = 0.0;
}

PhysicalCameraFocusStatus
PhysicalCameraFocusController::ApplyPhysicalCameraFocus(
    cv::VideoCapture& capture,
    const PhysicalCameraFocusConfig& config)
{
    if (!capture.isOpened()) {
        throw std::runtime_error(
            "ApplyPhysicalCameraFocus: capture is not open");
    }
    if (config.configure_focus && !std::isfinite(config.manual_focus)) {
        throw std::runtime_error(
            "ApplyPhysicalCameraFocus: manual focus must be finite");
    }

    PhysicalCameraFocusStatus next;
    next.configuration_attempted = config.configure_autofocus
                                || config.configure_focus;
    next.autofocus_requested = config.configure_autofocus
                            || config.configure_focus;
    next.autofocus_enabled = config.configure_focus
                           ? false
                           : config.autofocus;
    next.manual_focus_requested = config.configure_focus;
    next.requested_focus = config.manual_focus;

    if (next.autofocus_requested) {
        next.autofocus_set_accepted = capture.set(
            cv::CAP_PROP_AUTOFOCUS, next.autofocus_enabled ? 1.0 : 0.0);
    }
    if (config.configure_focus) {
        next.manual_focus_set_accepted = capture.set(
            cv::CAP_PROP_FOCUS, config.manual_focus);
    }

    next.negotiated_autofocus = capture.get(cv::CAP_PROP_AUTOFOCUS);
    next.negotiated_focus = capture.get(cv::CAP_PROP_FOCUS);
    next.summary = ComposeSummary(next);

    std::lock_guard<std::mutex> lock(mutex_);
    configured_focus_ = config;
    have_configuration_ = true;
    status_ = next;
    return status_;
}

PhysicalCameraFocusStatus
PhysicalCameraFocusController::SetPhysicalCameraCalibrationFocusLock(
    cv::VideoCapture& capture,
    bool locked)
{
    if (!capture.isOpened()) {
        throw std::runtime_error(
            "SetPhysicalCameraCalibrationFocusLock: capture is not open");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (status_.calibration_focus_locked == locked) return status_;

    if (locked) {
        autofocus_before_calibration_lock_ = capture.get(cv::CAP_PROP_AUTOFOCUS);
        const double lens_position = capture.get(cv::CAP_PROP_FOCUS);
        const bool auto_off = capture.set(cv::CAP_PROP_AUTOFOCUS, 0.0);
        const bool lens_held = !std::isfinite(lens_position)
                            || capture.set(cv::CAP_PROP_FOCUS, lens_position);
        status_.calibration_lock_accepted = auto_off && lens_held;
        status_.calibration_focus_locked = true;
    } else {
        bool restored = true;
        if (have_configuration_ && configured_focus_.configure_focus) {
            restored = capture.set(cv::CAP_PROP_AUTOFOCUS, 0.0);
            restored = capture.set(
                cv::CAP_PROP_FOCUS, configured_focus_.manual_focus) && restored;
        } else {
            const double restore_auto =
                have_configuration_ && configured_focus_.configure_autofocus
                    ? (configured_focus_.autofocus ? 1.0 : 0.0)
                    : autofocus_before_calibration_lock_;
            restored = capture.set(cv::CAP_PROP_AUTOFOCUS, restore_auto);
        }
        status_.calibration_lock_accepted = restored;
        status_.calibration_focus_locked = false;
    }

    status_.negotiated_autofocus = capture.get(cv::CAP_PROP_AUTOFOCUS);
    status_.negotiated_focus = capture.get(cv::CAP_PROP_FOCUS);
    status_.summary = ComposeSummary(status_);
    return status_;
}

void PhysicalCameraFocusController::ObserveRawFrameForFocus(
    const cv::Mat& raw_bgr,
    uint64_t frame_counter)
{
    if (frame_counter == 0 || (frame_counter % 8u) != 0u) return;
    double score = 0.0;
    try {
        score = ComputeFocusScore(raw_bgr);
    } catch (...) {
        // Focus telemetry is diagnostic only. It must never interrupt the
        // capture worker if a backend supplies an unexpected frame format.
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    status_.focus_score = score;
    status_.scored_frame_counter = frame_counter;
}

PhysicalCameraFocusStatus
PhysicalCameraFocusController::GetPhysicalCameraFocusStatusSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

}}} // namespace GRIM::Perception::Physical
