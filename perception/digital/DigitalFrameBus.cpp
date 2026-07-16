#include "DigitalFrameBus.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Digital {

DigitalFrameBus& DigitalFrameBus::Instance() {
    static DigitalFrameBus bus;
    return bus;
}

void DigitalFrameBus::Publish(const DigitalCaptureResult& result,
                              std::uint64_t frame_counter) {
    if (frame_counter == 0) {
        throw std::invalid_argument("DigitalFrameBus::Publish frame_counter must be non-zero");
    }
    if (result.metadata.status == DigitalCaptureStatus::Ok && result.image.empty()) {
        throw std::invalid_argument("DigitalFrameBus::Publish successful result has an empty image");
    }

    auto packet = std::make_shared<DigitalFramePacket>();
    packet->frame_counter = frame_counter;
    packet->published_at = std::chrono::steady_clock::now();
    packet->metadata = result.metadata;
    if (!result.image.empty()) {
        packet->image = result.image.clone();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = std::move(packet);
}

bool DigitalFrameBus::PullLatest(FrameView& out,
                                 std::uint64_t& last_seen_counter) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!latest_ || latest_->frame_counter <= last_seen_counter) {
        return false;
    }

    out.packet = latest_;
    out.image = latest_->image;
    out.frame_counter = latest_->frame_counter;
    out.published_at = latest_->published_at;
    out.metadata = latest_->metadata;
    last_seen_counter = latest_->frame_counter;
    return true;
}

bool DigitalFrameBus::HasEverPublished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<bool>(latest_);
}

void DigitalFrameBus::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_.reset();
}

}}} // namespace GRIM::Perception::Digital
