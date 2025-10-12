#pragma once
#include "memory_manager.hpp"
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
    static RouteDecision evaluate(const MemoryObject& obj);

    // Optional: actually perform routing (stubbed for now)
    static void dispatch(const MemoryObject& obj);

private:
    // Internal helpers
    static bool isFact(const MemoryObject& obj);
    static bool isCommand(const MemoryObject& obj);
    static bool isEvent(const MemoryObject& obj);
    static bool isLowConfidence(const MemoryObject& obj);
};

} // namespace GRIM
