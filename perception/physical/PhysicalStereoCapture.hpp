#pragma once

#include "PhysicalCameraStream.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalStereoCaptureConfig {
    std::string left_url;
    std::string left_label;
    std::string right_url;
    std::string right_label;
    double      maximum_pair_skew_ms = 15.0;
};

struct PhysicalStereoFramePair {
    cv::Mat      left_image;
    cv::Mat      right_image;
    uint64_t     pair_counter           = 0;
    uint64_t     left_frame_counter     = 0;
    uint64_t     right_frame_counter    = 0;
    uint64_t     left_capture_steady_ns = 0;
    uint64_t     right_capture_steady_ns = 0;
    double       signed_skew_ms         = 0.0;
};

struct PhysicalStereoCaptureStatus {
    PhysicalCameraStreamState left_state  = PhysicalCameraStreamState::Idle;
    PhysicalCameraStreamState right_state = PhysicalCameraStreamState::Idle;
    std::string left_url;
    std::string left_label;
    std::string right_url;
    std::string right_label;
    std::string last_error_reason;
    double      maximum_pair_skew_ms = 15.0;
    double      last_signed_skew_ms  = 0.0;
    double      left_fps             = 0.0;
    double      right_fps            = 0.0;
    uint64_t    left_frame_counter   = 0;
    uint64_t    right_frame_counter  = 0;
    uint64_t    synchronized_pair_count = 0;
    uint64_t    rejected_left_count     = 0;
    uint64_t    rejected_right_count    = 0;
};

class PhysicalStereoCapture {
public:
    PhysicalStereoCapture() = default;
    ~PhysicalStereoCapture();

    PhysicalStereoCapture(const PhysicalStereoCapture&) = delete;
    PhysicalStereoCapture& operator=(const PhysicalStereoCapture&) = delete;

    void OpenPhysicalStereoCapture(const PhysicalStereoCaptureConfig& config);
    void ClosePhysicalStereoCapture();

    // Drains retained frames and returns the newest pair accepted during this
    // call. Older unmatched frames are rejected explicitly by timestamp.
    bool PullLatestSynchronizedPairInto(PhysicalStereoFramePair& out_pair);

    PhysicalStereoCaptureStatus GetPhysicalStereoCaptureStatus() const;

private:
    void DrainRetainedFrames();

    PhysicalStereoCaptureConfig                    config_{};
    std::unique_ptr<PhysicalCameraStream>           left_stream_;
    std::unique_ptr<PhysicalCameraStream>           right_stream_;
    std::deque<PhysicalCapturedCameraFrame>         left_pending_;
    std::deque<PhysicalCapturedCameraFrame>         right_pending_;
    uint64_t                                        last_seen_left_counter_  = 0;
    uint64_t                                        last_seen_right_counter_ = 0;
    uint64_t                                        pair_counter_            = 0;
    uint64_t                                        rejected_left_count_     = 0;
    uint64_t                                        rejected_right_count_    = 0;
    double                                          last_signed_skew_ms_     = 0.0;
    std::string                                     last_error_reason_;
};

}}} // namespace GRIM::Perception::Physical