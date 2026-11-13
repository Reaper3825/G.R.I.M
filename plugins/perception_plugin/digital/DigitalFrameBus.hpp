#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>

#include <opencv2/core.hpp>

#include "DigitalCaptureTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

struct DigitalFramePacket {
    cv::Mat image;
    std::uint64_t frame_counter = 0;
    std::chrono::steady_clock::time_point published_at{};
    DigitalCaptureMetadata metadata{};
};

class DigitalFrameBus {
public:
    struct FrameView {
        std::shared_ptr<const DigitalFramePacket> packet;
        cv::Mat image;
        std::uint64_t frame_counter = 0;
        std::chrono::steady_clock::time_point published_at{};
        DigitalCaptureMetadata metadata{};
    };

    static DigitalFrameBus& Instance();

    // Publishes successful frames and failed capture attempts. Successful image
    // storage is cloned exactly once here; consumers receive immutable shared
    // cv::Mat views and must clone before mutation.
    void Publish(const DigitalCaptureResult& result, std::uint64_t frame_counter);
    bool PullLatest(FrameView& out, std::uint64_t& last_seen_counter) const;
    bool HasEverPublished() const;
    void Reset();

private:
    DigitalFrameBus() = default;
    DigitalFrameBus(const DigitalFrameBus&) = delete;
    DigitalFrameBus& operator=(const DigitalFrameBus&) = delete;

    mutable std::mutex mutex_;
    std::shared_ptr<const DigitalFramePacket> latest_;
};

}}} // namespace GRIM::Perception::Digital
