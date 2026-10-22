#pragma once
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <functional>

// UI-thread ownership only. No OS polling, rendering, or domain dependencies.
// Frame ownership remains latched after release so the release/click cannot
// fall through to a later-updated sibling.
namespace GRIM::UI {
struct InputOwner {
    uint64_t widget = 0;
    uint64_t panel = 0;
    explicit operator bool() const { return widget != 0; }
};

class InputRouter {
public:
    static InputRouter& get() {
        // Widgets can be members of other process-lifetime singletons.
        static auto* instance = new InputRouter;
        return *instance;
    }
    void beginFrame(bool cancel = false) {
        seen.clear();
        cancelled = cancel;
        if (cancel) {
            for (const auto owner : {capture, popup, focus}) cancelOwner(owner);
            capture = {}; popup = {}; focus = {};
        }
        frameCapture = capture;
        framePopup = popup;
        wheelOwner = 0;
    }
    void endFrame() {
        for (const auto owner : {capture, popup, focus})
            if (owner && !seen.count(owner.widget)) cancelOwner(owner);
        currentPanel = 0;
    }
    void touch(uint64_t widget, std::function<void()> cancel = {}) {
        seen.insert(widget);
        if (cancel) cancellation[widget] = std::move(cancel);
    }
    bool claimPointer(InputOwner owner) {
        if (!owner || cancelled || (frameCapture && frameCapture.widget != owner.widget) ||
            (framePopup && framePopup.widget != owner.widget)) return false;
        capture = frameCapture = owner;
        if (focus.widget != owner.widget) focus = {};
        return true;
    }
    void releasePointer(uint64_t widget) { if (capture.widget == widget) capture = {}; }
    void openPopup(InputOwner owner) { popup = framePopup = owner; }
    void closePopup(uint64_t widget) { if (popup.widget == widget) popup = {}; }
    bool allowsPointer(uint64_t widget) const {
        return !cancelled && (!frameCapture || frameCapture.widget == widget) &&
               (!framePopup || framePopup.widget == widget);
    }
    bool ownsPointer(uint64_t widget) const { return capture.widget == widget && widget; }
    bool ownsPopup(uint64_t widget) const { return popup.widget == widget && widget; }
    bool allowsWheel(uint64_t widget) const {
        return allowsPointer(widget) && (!wheelOwner || wheelOwner == widget);
    }
    void consumeWheel(uint64_t widget) { wheelOwner = widget; }
    void forget(uint64_t widget) {
        for (auto* owner : {&capture, &popup, &focus})
            if (owner->widget == widget) *owner = {};
        cancellation.erase(widget);
    }
    void forgetPanel(uint64_t panel) {
        for (const auto owner : {capture, popup, focus})
            if (owner.panel == panel) cancelOwner(owner);
    }
    uint64_t pointerPanel() const {
        return frameCapture ? frameCapture.panel : framePopup.panel;
    }
    template<class Range, class HitTest>
    uint64_t pointerTarget(const Range& backToFront, HitTest hit) const {
        if (pointerPanel()) return pointerPanel();
        for (auto it = backToFront.rbegin(); it != backToFront.rend(); ++it)
            if (const auto id = hit(*it)) return id;
        return 0;
    }
    InputOwner capture, popup, focus;
    uint64_t currentPanel = 0;
    bool cancelled = false;
private:
    void cancelOwner(InputOwner owner) {
        const auto found = cancellation.find(owner.widget);
        const auto callback = found == cancellation.end() ? std::function<void()>{} : found->second;
        forget(owner.widget);
        if (callback) callback();
    }
    InputOwner frameCapture, framePopup;
    uint64_t wheelOwner = 0;
    std::unordered_set<uint64_t> seen;
    std::unordered_map<uint64_t, std::function<void()>> cancellation;
};
}
