#include "PhysicalStereoCapture.hpp"

#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <cmath>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalStereoCapture::~PhysicalStereoCapture() {
    ClosePhysicalStereoCapture();
}

void PhysicalStereoCapture::OpenPhysicalStereoCapture(
    const PhysicalStereoCaptureConfig& config)
{
    if (config.left_url.empty() || config.right_url.empty()) {
        throw std::runtime_error(
            "PhysicalStereoCapture::OpenPhysicalStereoCapture: both camera URLs are required");
    }
    if (config.left_url == config.right_url) {
        throw std::runtime_error(
            "PhysicalStereoCapture::OpenPhysicalStereoCapture: left and right URLs must identify different cameras");
    }
    if (!std::isfinite(config.maximum_pair_skew_ms)
        || config.maximum_pair_skew_ms <= 0.0
        || config.maximum_pair_skew_ms > 1000.0) {
        throw std::runtime_error(
            "PhysicalStereoCapture::OpenPhysicalStereoCapture: maximum_pair_skew_ms must be finite and in (0,1000]");
    }

    ClosePhysicalStereoCapture();
    config_ = config;
    left_stream_  = std::make_unique<PhysicalCameraStream>();
    right_stream_ = std::make_unique<PhysicalCameraStream>();
    try {
        left_stream_->OpenPhysicalCameraStream(config_.left_url);
        right_stream_->OpenPhysicalCameraStream(config_.right_url);
    } catch (...) {
        ClosePhysicalStereoCapture();
        throw;
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalStereoCapture: opening left='" + config_.left_url
              + "' right='" + config_.right_url
              + "' max_skew_ms=" + std::to_string(config_.maximum_pair_skew_ms));
}

void PhysicalStereoCapture::ClosePhysicalStereoCapture() {
    if (left_stream_) left_stream_->ClosePhysicalCameraStream();
    if (right_stream_) right_stream_->ClosePhysicalCameraStream();
    left_stream_.reset();
    right_stream_.reset();
    left_pending_.clear();
    right_pending_.clear();
    last_seen_left_counter_  = 0;
    last_seen_right_counter_ = 0;
    pair_counter_            = 0;
    rejected_left_count_     = 0;
    rejected_right_count_    = 0;
    last_signed_skew_ms_     = 0.0;
    last_error_reason_.clear();
}

void PhysicalStereoCapture::DrainRetainedFrames() {
    if (!left_stream_ || !right_stream_) return;

    PhysicalCapturedCameraFrame frame;
    while (left_stream_->PullNextCapturedFrameInto(frame, last_seen_left_counter_)) {
        left_pending_.push_back(std::move(frame));
        frame = PhysicalCapturedCameraFrame{};
    }
    while (right_stream_->PullNextCapturedFrameInto(frame, last_seen_right_counter_)) {
        right_pending_.push_back(std::move(frame));
        frame = PhysicalCapturedCameraFrame{};
    }

    constexpr size_t kPendingQueueCapacity = 16;
    while (left_pending_.size() > kPendingQueueCapacity) {
        left_pending_.pop_front();
        ++rejected_left_count_;
    }
    while (right_pending_.size() > kPendingQueueCapacity) {
        right_pending_.pop_front();
        ++rejected_right_count_;
    }
}

bool PhysicalStereoCapture::PullLatestSynchronizedPairInto(
    PhysicalStereoFramePair& out_pair)
{
    out_pair = PhysicalStereoFramePair{};
    if (!left_stream_ || !right_stream_) return false;

    const auto left_state  = left_stream_->GetState();
    const auto right_state = right_stream_->GetState();
    if (left_state == PhysicalCameraStreamState::Failed) {
        last_error_reason_ = "left stream failed: " + left_stream_->GetLastErrorReason();
        return false;
    }
    if (right_state == PhysicalCameraStreamState::Failed) {
        last_error_reason_ = "right stream failed: " + right_stream_->GetLastErrorReason();
        return false;
    }

    DrainRetainedFrames();
    bool paired = false;
    while (!left_pending_.empty() && !right_pending_.empty()) {
        const auto& left  = left_pending_.front();
        const auto& right = right_pending_.front();
        const int64_t signed_skew_ns = static_cast<int64_t>(left.capture_steady_ns)
                                     - static_cast<int64_t>(right.capture_steady_ns);
        const double signed_skew_ms = static_cast<double>(signed_skew_ns) / 1.0e6;
        if (std::abs(signed_skew_ms) <= config_.maximum_pair_skew_ms) {
            ++pair_counter_;
            left.image.copyTo(out_pair.left_image);
            right.image.copyTo(out_pair.right_image);
            out_pair.pair_counter            = pair_counter_;
            out_pair.left_frame_counter      = left.frame_counter;
            out_pair.right_frame_counter     = right.frame_counter;
            out_pair.left_capture_steady_ns  = left.capture_steady_ns;
            out_pair.right_capture_steady_ns = right.capture_steady_ns;
            out_pair.signed_skew_ms          = signed_skew_ms;
            last_signed_skew_ms_             = signed_skew_ms;
            last_error_reason_.clear();
            left_pending_.pop_front();
            right_pending_.pop_front();
            paired = true;
            continue;
        }
        if (signed_skew_ns < 0) {
            left_pending_.pop_front();
            ++rejected_left_count_;
        } else {
            right_pending_.pop_front();
            ++rejected_right_count_;
        }
    }
    return paired;
}

PhysicalStereoCaptureStatus PhysicalStereoCapture::GetPhysicalStereoCaptureStatus() const {
    PhysicalStereoCaptureStatus status;
    status.left_url  = config_.left_url;
    status.left_label = config_.left_label;
    status.right_url = config_.right_url;
    status.right_label = config_.right_label;
    status.maximum_pair_skew_ms = config_.maximum_pair_skew_ms;
    status.last_signed_skew_ms = last_signed_skew_ms_;
    status.synchronized_pair_count = pair_counter_;
    status.rejected_left_count = rejected_left_count_;
    status.rejected_right_count = rejected_right_count_;
    status.last_error_reason = last_error_reason_;
    if (left_stream_) {
        status.left_state = left_stream_->GetState();
        status.left_fps = left_stream_->GetMeasuredFps();
        status.left_frame_counter = left_stream_->GetFrameCounter();
    }
    if (right_stream_) {
        status.right_state = right_stream_->GetState();
        status.right_fps = right_stream_->GetMeasuredFps();
        status.right_frame_counter = right_stream_->GetFrameCounter();
    }
    return status;
}

}}} // namespace GRIM::Perception::Physical