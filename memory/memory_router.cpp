#include "memory_router.hpp"
#include "Logger.hpp"
#include <iostream>

namespace GRIM {

bool MemoryRouter::isFact(const MemoryObject& obj) {
    return obj.type == TypeTag::Fact || obj.intent == IntentTag::SetPref;
}

bool MemoryRouter::isCommand(const MemoryObject& obj) {
    return obj.type == TypeTag::Command || obj.intent == IntentTag::Query;
}

bool MemoryRouter::isEvent(const MemoryObject& obj) {
    return obj.type == TypeTag::Event || obj.intent == IntentTag::StatusUpdate;
}

bool MemoryRouter::isLowConfidence(const MemoryObject& obj) {
    return obj.confidence < 0.55f;
}

// ===============================================
// Evaluate where to send this MemoryObject
// ===============================================

RouteDecision MemoryRouter::evaluate(const MemoryObject& obj) {
    RouteDecision result;
    result.priority = obj.confidence;

    // 1. Drop uncertain noise unless user-verified
    if (isLowConfidence(obj)) {
        result.target = MemoryRouteTarget::Ignore;
        result.reason = "Low confidence input.";
        return result;
    }

    // 2. Commands take priority — immediate execution
    if (isCommand(obj)) {
        result.target = MemoryRouteTarget::CommandExec;
        result.reason = "Identified as actionable command.";
        result.priority = 1.0f;
        return result;
    }

    // 3. Facts or preferences → long-term
    if (isFact(obj)) {
        result.target = MemoryRouteTarget::LongTerm;
        result.reason = "Stable fact or preference.";
        return result;
    }

    // 4. Events → short-term (may feed summary system)
    if (isEvent(obj)) {
        result.target = MemoryRouteTarget::ShortTerm;
        result.reason = "Runtime or environmental event.";
        return result;
    }

    // 5. Default catch-all
    result.target = MemoryRouteTarget::Ignore;
    result.reason = "Unclassified input.";
    return result;
}

// ===============================================
// Dispatch (placeholder for integration later)
// ===============================================

void MemoryRouter::dispatch(const MemoryObject& obj) {
    RouteDecision decision = evaluate(obj);

    switch (decision.target) {
        case MemoryRouteTarget::ShortTerm:
            std::cout << "[Router] -> ShortTerm (" << decision.reason << ")\n";
            break;

        case MemoryRouteTarget::LongTerm:
            std::cout << "[Router] -> LongTerm (" << decision.reason << ")\n";
            break;

        case MemoryRouteTarget::CommandExec:
            std::cout << "[Router] -> CommandExec (" << decision.reason << ")\n";
            break;

        case MemoryRouteTarget::Ignore:
        default:
            std::cout << "[Router] -> Ignored (" << decision.reason << ")\n";
            break;
    }
}

} // namespace GRIM
