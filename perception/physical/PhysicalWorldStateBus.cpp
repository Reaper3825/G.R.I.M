#include "PhysicalWorldStateBus.hpp"

#include "PhysicalWorldStateLogTag.hpp"
#include "logger.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalWorldStateBus& PhysicalWorldStateBus::Instance() {
    static PhysicalWorldStateBus s_instance;
    return s_instance;
}

void PhysicalWorldStateBus::PublishPhysicalWorldStateSnapshotToBus(
    const PhysicalWorldStateSnapshot& snapshot)
{
    if (snapshot.source_frame_counter == 0) {
        const std::string reason =
            "PublishPhysicalWorldStateSnapshotToBus: source_frame_counter==0 (uninitialised)";
        LOG_ERROR(PHYSICAL_WORLD_STATE_LOG_TAG, reason);
        throw std::runtime_error(reason);
    }
    std::lock_guard<std::mutex> lk(mutex_);
    latest_snapshot_ = snapshot;
    latest_time_     = std::chrono::steady_clock::now();
    ever_published_  = true;
}

bool PhysicalWorldStateBus::PullLatestPhysicalWorldStateSnapshotView(
    SnapshotView& out, uint64_t& last_seen_frame_counter) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (latest_snapshot_.source_frame_counter <= last_seen_frame_counter) return false;
    out.snapshot     = latest_snapshot_;
    out.published_at = latest_time_;
    last_seen_frame_counter = latest_snapshot_.source_frame_counter;
    return true;
}

bool PhysicalWorldStateBus::HasEverPublishedWorldStateSnapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalWorldStateBus::ResetPhysicalWorldStateBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_snapshot_ = PhysicalWorldStateSnapshot{};
    latest_time_     = {};
    ever_published_  = false;
}

}}} // namespace GRIM::Perception::Physical
