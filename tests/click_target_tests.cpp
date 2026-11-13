#include "core/input/click_target.hpp"
#include "core/input/checked_click.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace {
using namespace GRIM::Perception;
using namespace GRIM::Perception::Digital;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
ClickTargetResult commandResult;
std::string commandTarget;
ClickTargetKind commandKind;
int commandCalls = 0;
int unrelatedInputs = 0;
int movedX = 0, movedY = 0, typingDelay = 0;
std::string rawTyped, buttonUsed, keyUsed;
std::vector<std::string> comboUsed;
}
void logDebug(const std::string&, const std::string&) {}
void logError(const std::string&, const std::string&) {}
void logTrace(const std::string&, const std::string&) {}
int main() try {
    using namespace GRIM::Perception;
    using namespace GRIM::Perception::Digital;
    DigitalPerceptionPrimitiveSnapshot snapshot;
    auto& source = snapshot.source_metadata;
    source.status = DigitalCaptureStatus::Ok;
    source.source_device_id = "local"; source.source_transport = "native"; source.source_platform = "windows";
    source.source_rect = {-1920, -200, 1280, 720}; // Negative desktop origin, no DPI rescaling.
    source.scale_factor = 1.5f;
    source.capture_steady_ns = 10'000'000'000ULL;
    snapshot.ocr.status = DigitalPrimitiveStatus::Ok;
    snapshot.ocr.regions.push_back({{100, 40, 80, 20}, "Submit", 0.95f});
    uint64_t now = source.capture_steady_ns + 1;
    int dispatches = 0, observations = 0, dispatchedX = 0, dispatchedY = 0;
    bool current = true, delivered = true;
    ClickTargetServices services{
        [&] { ++observations; return snapshot; }, [&] { return now; }, [&] { return current; },
        [&](int x, int y) { ++dispatches; dispatchedX = x; dispatchedY = y; return delivered; }};
    auto click = [&](ClickTargetKind kind, const std::string& text) { return ExecuteClickTarget(kind, text, services); };
    auto result = click(ClickTargetKind::Text, "  SUBMIT \t");
    check(result.success && dispatchedX == -1780 && dispatchedY == -150 && dispatches == 1,
          "OCR bounds were not translated to physical desktop coordinates");
    auto reject = [&](ClickTargetKind kind, const std::string& text, const char* error) {
        const auto before = dispatches;
        const auto outcome = click(kind, text);
        check(!outcome.success && outcome.error == error && before == dispatches, "Invalid target sent input or returned wrong error");
    };
    const auto beforeObservation = observations;
    reject(ClickTargetKind::Text, " \t", "ERR_INVALID_ARGS");
    check(observations == beforeObservation, "Empty target triggered capture");
    reject(ClickTargetKind::Text, "Sub", "ERR_INPUT_TARGET_NOT_FOUND");
    snapshot.ocr.regions.push_back(snapshot.ocr.regions.front());
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_TARGET_AMBIGUOUS");
    snapshot.ocr.regions.pop_back();
    snapshot.ocr.regions[0].frame_rect.x = 1300;
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_TARGET_NOT_ACTIONABLE");
    snapshot.ocr.regions[0].frame_rect.x = 100;
    snapshot.ocr.regions[0].confidence = std::numeric_limits<float>::quiet_NaN();
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_TARGET_NOT_ACTIONABLE");
    snapshot.ocr.regions[0].confidence = 0.95f;
    snapshot.ocr.status = DigitalPrimitiveStatus::Unavailable;
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_OBSERVATION_UNAVAILABLE");
    snapshot.ocr.status = DigitalPrimitiveStatus::Ok;
    source.source_device_id = "remote";
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_OBSERVATION_UNAVAILABLE");
    source.source_device_id = "local";
    source.status = DigitalCaptureStatus::CaptureFailed;
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_OBSERVATION_UNAVAILABLE");
    source.status = DigitalCaptureStatus::Ok;
    now += 6'000'000'000ULL;
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_STALE_TARGET");
    now = source.capture_steady_ns - 1;
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_STALE_TARGET");
    now = source.capture_steady_ns + 1;
    current = false;
    reject(ClickTargetKind::Text, "Submit", "ERR_INPUT_STALE_TARGET");
    current = true;
    delivered = false;
    result = click(ClickTargetKind::Text, "Submit");
    check(!result.success && result.error == "ERR_INPUT_DISPATCH_FAILED", "Delivery failure became success");
    delivered = true;
    auto broken = services;
    broken.observe = []() -> DigitalPerceptionPrimitiveSnapshot { throw std::runtime_error("capture failed"); };
    check(ExecuteClickTarget(ClickTargetKind::Text, "Submit", broken).error == "ERR_INPUT_TARGETING_FAILED",
          "Capture exception escaped targeting");

    auto& native = snapshot.automation;
    native.status = DigitalPrimitiveStatus::Ok; native.target_matches_capture = true;
    native.elements.push_back({{-1800, -100, 100, 40}, "Save", "save-button", "button", true, false, false});
    check(click(ClickTargetKind::Object, "SAVE-BUTTON").success && dispatchedX == -1750 && dispatchedY == -80,
          "Native desktop bounds were scaled or offset twice");
    check(click(ClickTargetKind::Object, "button").success, "Unique control role did not resolve");
    native.elements.push_back({{-1600, -100, 100, 40}, "Cancel", "cancel-button", "button", true, false, false});
    reject(ClickTargetKind::Object, "button", "ERR_INPUT_TARGET_AMBIGUOUS");
    native.elements.pop_back();
    native.elements[0].enabled = false;
    reject(ClickTargetKind::Object, "Save", "ERR_INPUT_TARGET_NOT_ACTIONABLE");
    native.elements[0].enabled = true; native.elements[0].offscreen = true;
    reject(ClickTargetKind::Object, "Save", "ERR_INPUT_TARGET_NOT_ACTIONABLE");
    native.elements[0].offscreen = false; native.elements[0].password = true;
    reject(ClickTargetKind::Object, "Save", "ERR_INPUT_TARGET_NOT_ACTIONABLE");
    native.elements[0].password = false; native.target_changed_since_capture = true;
    reject(ClickTargetKind::Object, "Save", "ERR_INPUT_OBSERVATION_UNAVAILABLE");
    native.target_changed_since_capture = false; native.truncated = true;
    reject(ClickTargetKind::Object, "Save", "ERR_INPUT_OBSERVATION_UNAVAILABLE");

    // Checked delivery: no button events after failed movement, and partial input is failure.
    int sends = 0, releases = 0;
    bool moved = true, positioned = true;
    unsigned int sent = 2;
    GRIM::Input::CheckedClickServices input{
        [&](int, int) { return moved; }, [&](int, int) { return positioned; },
        [&] { ++sends; return sent; }, [&] { ++releases; }};
    check(GRIM::Input::SendCheckedClick(-1200, 10, input), "Complete input was rejected");
    moved = false;
    check(!GRIM::Input::SendCheckedClick(10, 10, input) && sends == 1, "Failed movement still sent button input");
    moved = true; positioned = false;
    check(!GRIM::Input::SendCheckedClick(10, 10, input) && sends == 1, "Cursor mismatch still sent button input");
    positioned = true; sent = 1;
    check(!GRIM::Input::SendCheckedClick(10, 10, input) && releases == 1, "Partial delivery did not fail and release");
    sent = 0;
    check(!GRIM::Input::SendCheckedClick(10, 10, input), "Rejected input became success");

    std::cout << "Click targeting, geometry, ambiguity, delivery and command-result tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
