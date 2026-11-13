#pragma once

#include "PhysicalGestureControlResult.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// Bounded event history. Unlike latest-frame buses, discrete Started/Released
// edges must not be overwritten before consumers have a chance to observe them.
class PhysicalGestureEventBus {
public:
    static PhysicalGestureEventBus& Instance();

    PhysicalGestureEvent PublishPhysicalGestureEvent(
        const PhysicalGestureEvent& event);
    std::vector<PhysicalGestureEvent> PullPhysicalGestureEventsAfter(
        uint64_t& last_seen_sequence) const;
    bool PullLatestPhysicalGestureEvent(
        PhysicalGestureEvent& out, uint64_t& last_seen_sequence) const;
    void ResetPhysicalGestureEventBus();

private:
    PhysicalGestureEventBus() = default;
    static constexpr size_t kMaxBufferedEvents = 128;

    mutable std::mutex mutex_;
    std::deque<PhysicalGestureEvent> events_;
    uint64_t next_sequence_ = 1;
};

}}} // namespace GRIM::Perception::Physical
