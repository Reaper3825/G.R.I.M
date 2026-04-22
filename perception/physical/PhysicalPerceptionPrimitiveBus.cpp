#include "PhysicalPerceptionPrimitiveBus.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

PhysicalPerceptionPrimitiveBus& PhysicalPerceptionPrimitiveBus::Instance() {
    static PhysicalPerceptionPrimitiveBus inst;
    return inst;
}

void PhysicalPerceptionPrimitiveBus::PublishPhysicalPerceptionResultsToBus(
    const PhysicalPerceptionPrimitiveResults& results)
{
    if (results.source_frame_counter == 0) {
        throw std::runtime_error(
            "PhysicalPerceptionPrimitiveBus::PublishPhysicalPerceptionResultsToBus: "
            "source_frame_counter is 0 — producer MUST stamp the originating "
            "PhysicalFrameBus counter (which starts at 1)");
    }
    if (results.model_image_width <= 0 || results.model_image_height <= 0) {
        throw std::runtime_error(
            "PhysicalPerceptionPrimitiveBus::PublishPhysicalPerceptionResultsToBus: "
            "model_image_width/height must be > 0");
    }
    std::lock_guard<std::mutex> lk(mutex_);
    latest_results_ = results;
    latest_time_    = std::chrono::steady_clock::now();
    ever_published_ = true;
}

bool PhysicalPerceptionPrimitiveBus::PullLatestPhysicalPerceptionResultsView(
    ResultsView& out, uint64_t& last_seen_frame_counter) const
{
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_published_) return false;
    if (latest_results_.source_frame_counter == last_seen_frame_counter) return false;
    out.results       = latest_results_;
    out.published_at  = latest_time_;
    last_seen_frame_counter = latest_results_.source_frame_counter;
    return true;
}

bool PhysicalPerceptionPrimitiveBus::HasEverPublishedResults() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return ever_published_;
}

void PhysicalPerceptionPrimitiveBus::ResetPhysicalPerceptionPrimitiveBus() {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_results_ = PhysicalPerceptionPrimitiveResults{};
    latest_time_    = {};
    ever_published_ = false;
}

}}} // namespace
