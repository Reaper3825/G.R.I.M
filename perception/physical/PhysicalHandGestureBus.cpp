#include "PhysicalHandGestureBus.hpp"

#include <chrono>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {
uint64_t SteadyNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
}

PhysicalHandGestureBus& PhysicalHandGestureBus::Instance() {
    static PhysicalHandGestureBus instance;
    return instance;
}

void PhysicalHandGestureBus::PublishPhysicalHandGestureSnapshot(
    const PhysicalHandGestureSnapshot& snapshot)
{
    if (snapshot.source_frame_counter != 0 &&
        (!snapshot.source_frame.packet ||
         snapshot.source_frame.frame_counter != snapshot.source_frame_counter)) {
        throw std::runtime_error(
            "PhysicalHandGestureBus::PublishPhysicalHandGestureSnapshot: "
            "source_frame is missing or does not match source_frame_counter");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = snapshot;
    latest_.publish_sequence = next_publish_sequence_++;
    latest_.published_steady_ns = SteadyNowNs();
    latest_time_ = std::chrono::steady_clock::now();
    ever_published_ = true;
}

bool PhysicalHandGestureBus::PullLatestPhysicalHandGestureSnapshot(
    SnapshotView& out, uint64_t& last_seen_publish_sequence) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ever_published_ || latest_.publish_sequence == last_seen_publish_sequence) {
        return false;
    }
    out.snapshot = latest_;
    out.published_at = latest_time_;
    last_seen_publish_sequence = latest_.publish_sequence;
    return true;
}

bool PhysicalHandGestureBus::HasEverPublishedSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ever_published_;
}

void PhysicalHandGestureBus::ResetPhysicalHandGestureBus() {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = {};
    latest_time_ = {};
    next_publish_sequence_ = 1;
    ever_published_ = false;
}

}}} // namespace GRIM::Perception::Physical
