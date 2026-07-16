#include "PhysicalGestureEventBus.hpp"

namespace GRIM { namespace Perception { namespace Physical {

PhysicalGestureEventBus& PhysicalGestureEventBus::Instance() {
    static PhysicalGestureEventBus instance;
    return instance;
}

PhysicalGestureEvent PhysicalGestureEventBus::PublishPhysicalGestureEvent(
    const PhysicalGestureEvent& event)
{
    std::lock_guard<std::mutex> lock(mutex_);
    PhysicalGestureEvent published = event;
    published.sequence = next_sequence_++;
    events_.push_back(published);
    while (events_.size() > kMaxBufferedEvents) events_.pop_front();
    return published;
}

std::vector<PhysicalGestureEvent>
PhysicalGestureEventBus::PullPhysicalGestureEventsAfter(
    uint64_t& last_seen_sequence) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PhysicalGestureEvent> result;
    for (const auto& event : events_) {
        if (event.sequence > last_seen_sequence) result.push_back(event);
    }
    if (!result.empty()) last_seen_sequence = result.back().sequence;
    return result;
}

bool PhysicalGestureEventBus::PullLatestPhysicalGestureEvent(
    PhysicalGestureEvent& out, uint64_t& last_seen_sequence) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.empty() || events_.back().sequence == last_seen_sequence)
        return false;
    out = events_.back();
    last_seen_sequence = out.sequence;
    return true;
}

void PhysicalGestureEventBus::ResetPhysicalGestureEventBus() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
    next_sequence_ = 1;
}

}}} // namespace GRIM::Perception::Physical
