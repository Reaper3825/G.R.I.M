#include "click_target.hpp"
#include "perception_context.hpp"
#include "digital/DigitalCaptureSource.hpp"
#include "digital/DigitalOcrProvider.hpp"
#include "digital/DigitalAutomationProvider.hpp"
#include "core/input/InputController.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>

namespace GRIM::Perception {
ClickTargetResult ClickOnObservedTarget(ClickTargetKind kind, const std::string& target, const std::string& button) {
    std::string normalized = button;
    for (auto& c : normalized) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    GRIM::MouseButton mapped;
    if (normalized == "left") mapped = GRIM::MouseButton::Left;
    else if (normalized == "right") mapped = GRIM::MouseButton::Right;
    else if (normalized == "middle") mapped = GRIM::MouseButton::Middle;
    else return {false, "ERR_INVALID_ARGS", "Unknown mouse button"};
#ifndef _WIN32
    return {false, "ERR_INPUT_UNSUPPORTED", "Click targeting is not implemented for this input platform"};
#else
    HWND foreground = nullptr;
    Digital::DigitalCaptureMetadata metadata;
    const auto now = [] {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    ClickTargetServices services;
    services.nowSteadyNs = now;
    services.observe = [&] {
        Digital::DigitalPerceptionPrimitiveSnapshot result;
        foreground = GetForegroundWindow();
        auto source = Digital::CreatePlatformDigitalCaptureSource();
        if (!source || !foreground) return result;
        Digital::DigitalCaptureRequest request;
        request.mode = Digital::DigitalCaptureMode::ActiveWindow;
        const auto capture = source->Capture(request);
        metadata = result.source_metadata = capture.metadata;
        if (!capture.Succeeded() || capture.image.cols != metadata.source_rect.width ||
            capture.image.rows != metadata.source_rect.height) {
            result.source_metadata.status = Digital::DigitalCaptureStatus::CaptureFailed;
            return result;
        }
        if (kind == ClickTargetKind::Text) {
            auto ocr = Digital::CreateDefaultDigitalOcrProvider();
            if (ocr) result.ocr = ocr->Recognize(capture.image, metadata.dpi_x, metadata.dpi_y);
        } else {
            auto native = Digital::CreatePlatformDigitalAutomationProvider();
            if (native) result.automation = native->InspectForeground(metadata, 2048);
        }
        return result;
    };
    services.sourceStillCurrent = [&] {
        if (!foreground || GetForegroundWindow() != foreground) return false;
        RECT rect{};
        if (!GetWindowRect(foreground, &rect)) return false;
        const auto& expected = metadata.active_window_rect;
        return rect.left == expected.x && rect.top == expected.y &&
            int64_t(rect.right) - rect.left == expected.width && int64_t(rect.bottom) - rect.top == expected.height;
    };
    services.dispatch = [mapped](int x, int y) { return InputController::clickAtChecked(x, y, mapped); };
    return ExecuteClickTarget(kind, target, services);
#endif
}

ClickTargetResult PerceptionContextManager::clickOnText(const std::string& text, const std::string& button) {
    return ClickOnObservedTarget(ClickTargetKind::Text, text, button);
}
ClickTargetResult PerceptionContextManager::clickOnObject(const std::string& label, const std::string& button) {
    return ClickOnObservedTarget(ClickTargetKind::Object, label, button);
}
}
