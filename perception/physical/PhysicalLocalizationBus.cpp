#include "PhysicalLocalizationBus.hpp"

#include "PhysicalLocalizationLogTag.hpp"
#include "logger.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalLocalizationBus& PhysicalLocalizationBus::Instance() {
    static PhysicalLocalizationBus s_instance;
    return s_instance;
}

void PhysicalLocalizationBus::PublishPhysicalLocalizationSnapshotToBus(
    const PhysicalLocalizationSnapshot& snapshot)
{
    if (snapshot.tracking_state == PhysicalLocalizationTrackingState::Tracking
        && snapshot.source_frame_counter == 0)
    {
        const std::string reason =
            "PublishPhysicalLocalizationSnapshotToBus: tracking_state==Tracking but "
            "source_frame_counter==0 — impossible combination";
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, reason);
        throw std::runtime_error(reason);
    }
    std::lock_guard<std::mutex> lk(mutex_);
    latest_snapshot_ = snapshot;
    latest_time_     = std::chrono::steady_clock::now();
    ++publish_sequence_;
    ever_published_  = true;
}

bool PhysicalLocalizationBus::PullLatestPhysicalLocalizationSnapshotView(
    SnapshotView& out, uint64_t& last_seen_publish_sequence) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (publish_sequence_ <= last_seen_publish_sequence) return false;
    out.snapshot     = latest_snapshot_;
    out.published_at = latest_time_;
    last_seen_publish_sequence = publish_sequence_;
    return true;
}

bool PhysicalLocalizationBus::HasEverPublishedLocalizationSnapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalLocalizationBus::ResetPhysicalLocalizationBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_snapshot_  = PhysicalLocalizationSnapshot{};
    latest_time_      = {};
    publish_sequence_ = 0;
    ever_published_   = false;
}

}}} // namespace GRIM::Perception::Physical
