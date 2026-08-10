#pragma once

#include "PhysicalStereoCapture.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalStereoFramePacket {
    cv::Mat      left_image;
    cv::Mat      right_image;
    uint64_t     pair_counter            = 0;
    uint64_t     left_frame_counter      = 0;
    uint64_t     right_frame_counter     = 0;
    uint64_t     left_capture_steady_ns  = 0;
    uint64_t     right_capture_steady_ns = 0;
    double       signed_skew_ms          = 0.0;
    std::string  left_url;
    std::string  left_label;
    std::string  right_url;
    std::string  right_label;
    std::chrono::steady_clock::time_point published_at{};
};

class PhysicalStereoFrameBus {
public:
    struct FrameView {
        std::shared_ptr<const PhysicalStereoFramePacket> packet;
        cv::Mat      left_image;
        cv::Mat      right_image;
        uint64_t     pair_counter            = 0;
        uint64_t     left_frame_counter      = 0;
        uint64_t     right_frame_counter     = 0;
        uint64_t     left_capture_steady_ns  = 0;
        uint64_t     right_capture_steady_ns = 0;
        double       signed_skew_ms          = 0.0;
        std::string  left_url;
        std::string  left_label;
        std::string  right_url;
        std::string  right_label;
    };

    static PhysicalStereoFrameBus& Instance();

    void PublishPhysicalStereoFramePairToBus(
        const PhysicalStereoFramePair& pair,
        const PhysicalStereoCaptureConfig& config);
    bool PullLatestPhysicalStereoFrameView(
        FrameView& out,
        uint64_t& last_seen_pair_counter) const;
    bool HasEverPublishedStereoFrame() const;
    void ResetPhysicalStereoFrameBus();

private:
    PhysicalStereoFrameBus() = default;

    mutable std::mutex mutex_;
    std::shared_ptr<const PhysicalStereoFramePacket> latest_packet_;
    bool ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical