#pragma once
#include "plugins/perception_plugin/digital/DigitalPerceptionPrimitiveTypes.hpp"
#include <functional>

namespace GRIM::Perception {
enum class ClickTargetKind { Text, Object };
struct ClickTargetResult {
    bool success = false;
    std::string error, message;
    int x = 0, y = 0; // Physical virtual-desktop coordinates on success.
};

// Exact normalized labels only; no substring, first-match, or screen-center fallback.
ClickTargetResult ResolveClickTarget(ClickTargetKind kind, const std::string& target,
    const Digital::DigitalPerceptionPrimitiveSnapshot& observation, uint64_t nowSteadyNs);

struct ClickTargetServices {
    std::function<Digital::DigitalPerceptionPrimitiveSnapshot()> observe;
    std::function<uint64_t()> nowSteadyNs;
    std::function<bool()> sourceStillCurrent;
    std::function<bool(int, int)> dispatch;
};
ClickTargetResult ExecuteClickTarget(ClickTargetKind kind, const std::string& target,
                                    const ClickTargetServices& services);
// Local host adapter: fresh capture and bounds providers; no model generation.
ClickTargetResult ClickOnObservedTarget(ClickTargetKind kind, const std::string& target,
                                      const std::string& button = "left");
}
