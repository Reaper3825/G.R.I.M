// mouse.cpp
#include "mouse.hpp"
#include "logger.hpp"
#include "input_parser.hpp"  // ✅ For InputState definition

std::unordered_map<MouseButton, MouseState> Mouse::buttonStates;
std::mutex Mouse::stateMutex;

// ✅ REMOVED: Hook-based system replaced with direct polling
// This eliminates threading issues and improves reliability

void Mouse::initialize() {
    LOG_DEBUG("Mouse", "Mouse system initialized (direct polling mode)");
}

void Mouse::shutdown() {
    std::lock_guard<std::mutex> lock(stateMutex);
    buttonStates.clear();
    LOG_DEBUG("Mouse", "Mouse system shutdown");
}

// ✅ NEW: Update mouse state from InputState (called from main loop)
void Mouse::updateFromInput(const struct InputState& input) {
    std::lock_guard<std::mutex> lock(stateMutex);
    
    // Use position from InputState (already in screen coordinates)
    POINT pos;
    pos.x = static_cast<LONG>(input.mousePos.x);
    pos.y = static_cast<LONG>(input.mousePos.y);
    
    // Map InputState buttons to MouseButton enum
    struct ButtonMapping { MouseButton btn; int index; };
    ButtonMapping mappings[] = {
        {MouseButton::Left, 0},
        {MouseButton::Right, 1},
        {MouseButton::Middle, 2}
    };
    
    for (const auto& mapping : mappings) {
        auto& state = buttonStates[mapping.btn];
        state.position = pos;
        
        bool down = input.mouseDown[mapping.index];
        bool pressed = input.mousePressed[mapping.index];
        bool released = input.mouseReleased[mapping.index];
        
        // Update state
        state.down = down;
        state.pressed = pressed;
        state.released = released;
        
        // Fire callbacks
        if (pressed) {
            for (auto& cb : state.pressCallbacks) {
                cb(mapping.btn);
            }
        }
        if (released) {
            for (auto& cb : state.releaseCallbacks) {
                cb(mapping.btn);
            }
        }
    }
}

void Mouse::setDown(MouseButton btn) {
    // ✅ DEPRECATED: Now updated via updateFromInput()
    std::lock_guard<std::mutex> lock(stateMutex);
    auto& s = buttonStates[btn];
    if (!s.down) {
        s.pressed = true;
        for (auto& cb : s.pressCallbacks) cb(btn);
    }
    s.down = true;
}

void Mouse::setUp(MouseButton btn) {
    // ✅ DEPRECATED: Now updated via updateFromInput()
    std::lock_guard<std::mutex> lock(stateMutex);
    auto& s = buttonStates[btn];
    if (s.down) {
        s.released = true;
        for (auto& cb : s.releaseCallbacks) cb(btn);
    }
    s.down = false;
}

bool Mouse::isDown(MouseButton btn) {
    std::lock_guard<std::mutex> lock(stateMutex);
    return buttonStates[btn].down;
}

bool Mouse::wasPressed(MouseButton btn) {
    std::lock_guard<std::mutex> lock(stateMutex);
    return buttonStates[btn].pressed;
}

bool Mouse::wasReleased(MouseButton btn) {
    std::lock_guard<std::mutex> lock(stateMutex);
    return buttonStates[btn].released;
}

POINT Mouse::getPosition() {
    POINT pt;
#ifdef _WIN32
    GetCursorPos(&pt);
#else
    pt.x = 0;
    pt.y = 0;
#endif
    return pt;
}

void Mouse::endFrame() {
    std::lock_guard<std::mutex> lock(stateMutex);
    for (auto& [_, s] : buttonStates) {
        s.pressed = false;
        s.released = false;
    }
}

void Mouse::onPress(MouseButton btn, std::function<void(MouseButton)> cb) {
    std::lock_guard<std::mutex> lock(stateMutex);
    buttonStates[btn].pressCallbacks.push_back(std::move(cb));
}
void Mouse::onRelease(MouseButton btn, std::function<void(MouseButton)> cb) {
    buttonStates[btn].releaseCallbacks.push_back(std::move(cb));
}
