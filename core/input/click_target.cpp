#include "click_target.hpp"
#include <algorithm>
#include <cctype>
#include <limits>

namespace GRIM::Perception {
namespace {
using namespace Digital;
ClickTargetResult Failure(const char* error, const std::string& message) {
    return {false, error, message};
}
std::string Normalize(const std::string& text) {
    std::string result;
    bool space = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) { space = !result.empty(); continue; }
        if (space) result += ' ';
        space = false;
        result += static_cast<char>(std::tolower(c));
    }
    return result;
}
bool Inside(const DigitalRect& box, const DigitalRect& frame) {
    return box.IsValid() && frame.IsValid() && box.x >= frame.x && box.y >= frame.y &&
        int64_t(box.x) + box.width <= int64_t(frame.x) + frame.width &&
        int64_t(box.y) + box.height <= int64_t(frame.y) + frame.height;
}
}
ClickTargetResult ResolveClickTarget(ClickTargetKind kind, const std::string& target,
    const Digital::DigitalPerceptionPrimitiveSnapshot& observation, uint64_t nowSteadyNs) {
    using namespace Digital;
    const auto query = Normalize(target);
    if (query.empty() || target.find('\0') != std::string::npos)
        return Failure("ERR_INVALID_ARGS", "A nonempty target label is required");
    const auto& source = observation.source_metadata;
    if (source.status != DigitalCaptureStatus::Ok || !source.source_rect.IsValid() ||
        source.source_device_id != "local" || source.source_transport != "native" || source.source_platform != "windows")
        return Failure("ERR_INPUT_OBSERVATION_UNAVAILABLE", "Targeting requires a successful local Windows capture");
    constexpr uint64_t maxAge = 5'000'000'000ULL;
    if (!source.capture_steady_ns || nowSteadyNs < source.capture_steady_ns ||
        nowSteadyNs - source.capture_steady_ns > maxAge)
        return Failure("ERR_INPUT_STALE_TARGET", "The captured target is too old; retry with a fresh observation");

    std::vector<DigitalRect> matches;
    bool invalid = false, blocked = false;
    if (kind == ClickTargetKind::Text) {
        if (observation.ocr.status != DigitalPrimitiveStatus::Ok)
            return Failure("ERR_INPUT_OBSERVATION_UNAVAILABLE", "OCR bounds are unavailable");
        for (const auto& region : observation.ocr.regions) {
            if (Normalize(region.text) != query) continue;
            const auto& box = region.frame_rect;
            if (!(region.confidence >= 0.5f && region.confidence <= 1.0f) ||
                !Inside(box, {0, 0, source.source_rect.width, source.source_rect.height})) {
                invalid = true; continue;
            }
            const int64_t x = int64_t(source.source_rect.x) + box.x;
            const int64_t y = int64_t(source.source_rect.y) + box.y;
            if (x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max() ||
                y < std::numeric_limits<int>::min() || y > std::numeric_limits<int>::max()) {
                invalid = true; continue;
            }
            matches.push_back({static_cast<int>(x), static_cast<int>(y), box.width, box.height});
        }
    } else if (kind == ClickTargetKind::Object) {
        const auto& native = observation.automation;
        if (native.status != DigitalPrimitiveStatus::Ok || !native.target_matches_capture ||
            native.target_changed_since_capture || native.truncated)
            return Failure("ERR_INPUT_OBSERVATION_UNAVAILABLE", "A complete, capture-matched native control observation is required");
        for (const auto& element : native.elements) {
            if (Normalize(element.name) != query && Normalize(element.automation_id) != query && Normalize(element.role) != query)
                continue;
            if (!element.enabled || element.offscreen || element.password) { blocked = true; continue; }
            if (!Inside(element.desktop_rect, source.source_rect)) { invalid = true; continue; }
            matches.push_back(element.desktop_rect);
        }
    } else return Failure("ERR_INVALID_ARGS", "Unknown target kind");

    // Invalid/blocked duplicate labels also prevent proving a unique actionable target.
    if (invalid || blocked)
        return Failure("ERR_INPUT_TARGET_NOT_ACTIONABLE", "Matching target has invalid bounds, low confidence, or is not actionable");
    if (matches.empty()) return Failure("ERR_INPUT_TARGET_NOT_FOUND", "No exact bounded target matches the supplied label");
    if (matches.size() != 1) return Failure("ERR_INPUT_TARGET_AMBIGUOUS", "Multiple targets match; supply a more specific label or use explicit coordinates");
    const auto& box = matches.front();
    const int64_t x = int64_t(box.x) + box.width / 2, y = int64_t(box.y) + box.height / 2;
    if (x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max() ||
        y < std::numeric_limits<int>::min() || y > std::numeric_limits<int>::max())
        return Failure("ERR_INPUT_TARGET_NOT_ACTIONABLE", "Target coordinates overflow the desktop coordinate range");
    return {true, "", "Located target", static_cast<int>(x), static_cast<int>(y)};
}

ClickTargetResult ExecuteClickTarget(ClickTargetKind kind, const std::string& target,
                                    const ClickTargetServices& services) {
    try {
        // Reject empty input before any observation or external action.
        if (Normalize(target).empty() || target.find('\0') != std::string::npos)
            return Failure("ERR_INVALID_ARGS", "A nonempty target label is required");
        const auto observation = services.observe();
        auto result = ResolveClickTarget(kind, target, observation, services.nowSteadyNs());
        if (!result.success) return result;
        if (!services.sourceStillCurrent())
            return Failure("ERR_INPUT_STALE_TARGET", "The foreground window or its geometry changed during targeting");
        if (!services.dispatch(result.x, result.y))
            return Failure("ERR_INPUT_DISPATCH_FAILED", "The operating system did not accept the complete click input");
        result.message = "Click input sent at (" + std::to_string(result.x) + ", " + std::to_string(result.y) +
            "); the application response has not been verified";
        return result;
    } catch (const std::exception& error) {
        return Failure("ERR_INPUT_TARGETING_FAILED", std::string("Targeting failed: ") + error.what());
    } catch (...) { return Failure("ERR_INPUT_TARGETING_FAILED", "Targeting failed with an unknown error"); }
}
}
