#include "PhysicalSpatialGroundingBus.hpp"

#include "PhysicalSpatialGroundingLogTag.hpp"
#include "logger.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalSpatialGroundingBus& PhysicalSpatialGroundingBus::Instance() {
    static PhysicalSpatialGroundingBus s_instance;
    return s_instance;
}

void PhysicalSpatialGroundingBus::PublishPhysicalSpatialGroundingResultsToBus(
    const PhysicalSpatialGroundingResults& results)
{
    if (results.source_frame_counter == 0) {
        // Rule 20: counters start at 1. Zero means uninitialised data slipped
        // through the loop — refuse to publish so consumers do not see a
        // ghost result.
        const std::string reason =
            "PublishPhysicalSpatialGroundingResultsToBus: source_frame_counter==0 (uninitialised)";
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, reason);
        throw std::runtime_error(reason);
    }
    std::lock_guard<std::mutex> lk(mutex_);
    latest_results_ = results;
    latest_time_    = std::chrono::steady_clock::now();
    ever_published_ = true;
}

bool PhysicalSpatialGroundingBus::PullLatestPhysicalSpatialGroundingResultsView(
    ResultsView& out, uint64_t& last_seen_frame_counter) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (latest_results_.source_frame_counter <= last_seen_frame_counter) return false;
    out.results       = latest_results_;
    out.published_at  = latest_time_;
    last_seen_frame_counter = latest_results_.source_frame_counter;
    return true;
}

bool PhysicalSpatialGroundingBus::HasEverPublishedSpatialGroundingResults() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalSpatialGroundingBus::ResetPhysicalSpatialGroundingBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_results_ = PhysicalSpatialGroundingResults{};
    latest_time_    = {};
    ever_published_ = false;
}

}}} // namespace GRIM::Perception::Physical
