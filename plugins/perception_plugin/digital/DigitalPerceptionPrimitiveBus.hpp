#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>

#include "DigitalPerceptionPrimitiveTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

class DigitalPerceptionPrimitiveBus {
public:
    struct SnapshotView {
        std::shared_ptr<const DigitalPerceptionPrimitiveSnapshot> snapshot;
        std::chrono::steady_clock::time_point published_at{};
    };

    static DigitalPerceptionPrimitiveBus& Instance();

    void Publish(const DigitalPerceptionPrimitiveSnapshot& snapshot);
    bool PullLatest(SnapshotView& out, std::uint64_t& last_seen_frame_counter) const;
    bool HasEverPublished() const;
    void Reset();

private:
    DigitalPerceptionPrimitiveBus() = default;
    DigitalPerceptionPrimitiveBus(const DigitalPerceptionPrimitiveBus&) = delete;
    DigitalPerceptionPrimitiveBus& operator=(const DigitalPerceptionPrimitiveBus&) = delete;

    mutable std::mutex mutex_;
    std::shared_ptr<const DigitalPerceptionPrimitiveSnapshot> latest_;
    std::chrono::steady_clock::time_point published_at_{};
};

}}} // namespace GRIM::Perception::Digital
