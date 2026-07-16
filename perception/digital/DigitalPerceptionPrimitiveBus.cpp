#include "DigitalPerceptionPrimitiveBus.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Digital {

DigitalPerceptionPrimitiveBus& DigitalPerceptionPrimitiveBus::Instance() {
    static DigitalPerceptionPrimitiveBus bus;
    return bus;
}

void DigitalPerceptionPrimitiveBus::Publish(
    const DigitalPerceptionPrimitiveSnapshot& snapshot) {
    if (snapshot.source_frame_counter == 0) {
        throw std::invalid_argument(
            "DigitalPerceptionPrimitiveBus::Publish requires a source frame counter");
    }
    auto immutable = std::make_shared<const DigitalPerceptionPrimitiveSnapshot>(snapshot);
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = std::move(immutable);
    published_at_ = std::chrono::steady_clock::now();
}

bool DigitalPerceptionPrimitiveBus::PullLatest(
    SnapshotView& out, std::uint64_t& last_seen_frame_counter) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!latest_ || latest_->source_frame_counter <= last_seen_frame_counter) return false;
    out.snapshot = latest_;
    out.published_at = published_at_;
    last_seen_frame_counter = latest_->source_frame_counter;
    return true;
}

bool DigitalPerceptionPrimitiveBus::HasEverPublished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<bool>(latest_);
}

void DigitalPerceptionPrimitiveBus::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_.reset();
    published_at_ = {};
}

}}} // namespace GRIM::Perception::Digital
