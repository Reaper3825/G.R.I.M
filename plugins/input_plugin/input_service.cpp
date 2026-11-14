#include "input_service.hpp"
#include "InputController.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <stdexcept>
namespace GRIM::Input {
static MouseButton stringToMouseButton(const std::string& button) {
    std::string lower = button;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "right") return MouseButton::Right;
    if (lower == "middle") return MouseButton::Middle;
    if (lower == "left") return MouseButton::Left;
    throw std::invalid_argument("Unknown mouse button");
}

// Helper: Convert string key name to virtual key code
static WORD stringToVirtualKey(const std::string& key) {
    std::string lower = key;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Common keys
    if (lower == "enter" || lower == "return") return VK_RETURN;
    if (lower == "esc" || lower == "escape") return VK_ESCAPE;
    if (lower == "tab") return VK_TAB;
    if (lower == "space") return VK_SPACE;
    if (lower == "backspace") return VK_BACK;
    if (lower == "delete" || lower == "del") return VK_DELETE;
    if (lower == "home") return VK_HOME;
    if (lower == "end") return VK_END;
    if (lower == "pageup" || lower == "pgup") return VK_PRIOR;
    if (lower == "pagedown" || lower == "pgdn") return VK_NEXT;
    
    // Arrow keys
    if (lower == "left") return VK_LEFT;
    if (lower == "right") return VK_RIGHT;
    if (lower == "up") return VK_UP;
    if (lower == "down") return VK_DOWN;
    
    // Modifier keys
    if (lower == "shift") return VK_SHIFT;
    if (lower == "ctrl" || lower == "control") return VK_CONTROL;
    if (lower == "alt") return VK_MENU;
    if (lower == "win" || lower == "windows") return VK_LWIN;
    
    // Function keys
    if (lower == "f1") return VK_F1;
    if (lower == "f2") return VK_F2;
    if (lower == "f3") return VK_F3;
    if (lower == "f4") return VK_F4;
    if (lower == "f5") return VK_F5;
    if (lower == "f6") return VK_F6;
    if (lower == "f7") return VK_F7;
    if (lower == "f8") return VK_F8;
    if (lower == "f9") return VK_F9;
    if (lower == "f10") return VK_F10;
    if (lower == "f11") return VK_F11;
    if (lower == "f12") return VK_F12;
    
    // Single character keys
    if (key.length() == 1) {
        char c = std::toupper(key[0]);
        if (c >= 'A' && c <= 'Z') return c;
        if (c >= '0' && c <= '9') return c;
    }
    
    return 0; // Unknown key
}


bool ExecuteInputPrimitive(const nlohmann::json& request) {
    const auto op = request.at("op").get<std::string>();
#ifndef _WIN32
    if (op == "available") return false;
    throw std::runtime_error("Native input is unavailable on this platform");
#else
    if (op == "available") return true;
#endif
    if (op == "move") InputController::moveMouse(request.at("x").get<int>(), request.at("y").get<int>());
    else if (op == "click") InputController::click(stringToMouseButton(request.at("button").get<std::string>()));
    else if (op == "type") InputController::typeText(request.at("text").get<std::string>(), 15);
    else if (op == "key") {
        const auto vk = stringToVirtualKey(request.at("key").get<std::string>());
        if (!vk) throw std::invalid_argument("Unknown key");
        if (vk) { InputController::keyEvent(vk, KeyAction::Press);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            InputController::keyEvent(vk, KeyAction::Release); }
    } else if (op == "combo") {
        std::vector<WORD> keys;
        for (const auto& name : request.at("keys")) {
            const auto vk = stringToVirtualKey(name.get<std::string>());
            if (!vk) throw std::invalid_argument("Unknown key in combination");
            keys.push_back(vk);
        }
        if (keys.empty()) throw std::invalid_argument("Empty key combination");
        InputController::combo(keys);
    } else throw std::invalid_argument("Unknown input primitive");
    return true;
}
}
