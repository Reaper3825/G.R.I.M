// mouse.hpp
#pragma once
#include <unordered_map>
#include <functional>
#include <mutex>
#include <vector>
#include "core/grim_platform.h"

// Forward declaration
struct InputState;

enum class MouseButton {
    Left, Right, Middle, X1, X2, Unknown
};

struct MouseState {
    bool down = false;
    bool pressed = false;
    bool released = false;
    POINT position = {0,0};
    std::vector<std::function<void(MouseButton)>> pressCallbacks;
    std::vector<std::function<void(MouseButton)>> releaseCallbacks;
};

class Mouse {
public:
    static void initialize();
    static void shutdown();
    static void updatePosition();
    static POINT getPosition();
    static bool isDown(MouseButton btn);
    static bool wasPressed(MouseButton btn);
    static bool wasReleased(MouseButton btn);
    static void endFrame();
    static void onPress(MouseButton btn, std::function<void(MouseButton)> cb);
    static void onRelease(MouseButton btn, std::function<void(MouseButton)> cb);
    
    // ✅ NEW: Update from InputState instead of using hooks
    static void updateFromInput(const InputState& input);

private:
    static std::unordered_map<MouseButton, MouseState> buttonStates;
    static std::mutex stateMutex;
    
    // ✅ DEPRECATED: Hook-based methods kept for compatibility but not used
    static void setDown(MouseButton btn);
    static void setUp(MouseButton btn);
};
