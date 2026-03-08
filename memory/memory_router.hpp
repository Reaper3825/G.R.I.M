#pragma once
#include "unified_memory.hpp"
#include <vector>
#include <string>
#include <optional>

namespace GRIM {

enum class MemoryRouteTarget {
    ShortTerm,
    LongTerm,
    CommandExec,
    Ignore
};

struct RouteDecision {
    MemoryRouteTarget target;
    std::string reason;
    float priority;   // 0.0–1.0
};

class MemoryRouter {
public:
    // Route a MemoryObject based on its metadata
    static RouteDecision evaluate(const UnifiedMemoryObject& obj);

    // Optional: actually perform routing (stubbed for now)
    static void dispatch(const UnifiedMemoryObject& obj);

private:
    // Internal helpers
    static bool isFact(const UnifiedMemoryObject& obj);
    static bool isCommand(const UnifiedMemoryObject& obj);
    static bool isEvent(const UnifiedMemoryObject& obj);
    static bool isLowConfidence(const UnifiedMemoryObject& obj);
};

} // namespace GRIM
