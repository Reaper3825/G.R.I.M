#pragma once
#include "commands/command_result.hpp"
#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace inputCommands {
inline const GrimPluginAPI* host = nullptr;
struct ServiceUnavailable {};
struct UnknownException {};
inline CommandResult unavailable() {
    return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
}
inline bool primitive(const nlohmann::json& request) {
    const auto json = request.dump();
    const char* raw = host->input_primitive(host, json.data(), json.size());
    if (!raw) throw ServiceUnavailable{};
    struct Release { const char* raw; ~Release() { host->string_free(host, raw); } } release{raw};
    const auto response = nlohmann::json::parse(raw);
    if (response.at("format") != "grim-runtime-service" || response.at("version") != 1)
        throw std::runtime_error("Unsupported input primitive response");
    if (!response.at("error").is_null()) {
        const auto& error = response.at("error");
        if (error.at("kind") == "exception") throw std::runtime_error(error.at("message").get<std::string>());
        throw UnknownException{};
    }
    return response.at("value").get<bool>();
}
inline CommandResult cmdInputMoveMouse(const std::string& arg) {
    // Expected format: "x,y" or "x y"

    if (!primitive({{"op", "available"}})) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }

    // Parse coordinates
    std::string coords = arg;
    std::replace(coords.begin(), coords.end(), ',', ' ');
    std::istringstream iss(coords);
    int x, y;

    if (!(iss >> x >> y)) {
        return {false, "[Input] Invalid coordinates format. Use: move mouse 100 200", "ERR_INVALID_ARGS", "error", "Invalid coordinates", Colors::Red};
    }

    if (!primitive({{"op", "move"}, {"x", x}, {"y", y}})) return unavailable();

    return {true, "Mouse moved to (" + std::to_string(x) + ", " + std::to_string(y) + ")", "ERR_NONE", "routine", "Mouse moved", Colors::Green};
}

inline CommandResult cmdInputClick(const std::string& arg) {
    // Expected format: "left" | "right" | "middle" | ""

    if (!primitive({{"op", "available"}})) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }

    std::string button = arg.empty() ? "left" : arg;
    if (!primitive({{"op", "click"}, {"button", button}})) return unavailable();

    return {true, "Clicked " + button + " mouse button", "ERR_NONE", "routine", "Clicked", Colors::Green};
}

inline CommandResult cmdInputType(const std::string& arg) {
    // Expected format: "text to type"

    if (!primitive({{"op", "available"}})) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }

    if (arg.empty()) {
        return {false, "[Input] No text provided. Use: type hello world", "ERR_INVALID_ARGS", "error", "No text to type", Colors::Red};
    }

    if (!primitive({{"op", "type"}, {"text", arg}})) return unavailable(); // 15ms delay for natural typing

    return {true, "Typed: " + arg, "ERR_NONE", "routine", "Text typed", Colors::Green};
}

inline CommandResult cmdInputKey(const std::string& arg) {
    // Expected format: "enter" | "ctrl+c" | "alt+f4"

    if (!primitive({{"op", "available"}})) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }

    if (arg.empty()) {
        return {false, "[Input] No key provided. Use: key enter or key ctrl+c", "ERR_INVALID_ARGS", "error", "No key specified", Colors::Red};
    }

    // Check if it's a combo (contains + or space)
    if (arg.find('+') != std::string::npos) {
        // Parse combo
        std::vector<std::string> keys;
        std::istringstream iss(arg);
        std::string key;
        while (std::getline(iss, key, '+')) {
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            if (!key.empty()) keys.push_back(key);
        }

        if (!primitive({{"op", "combo"}, {"keys", keys}})) return unavailable();
        return {true, "Pressed key combo: " + arg, "ERR_NONE", "routine", "Key combo pressed", Colors::Green};
    } else {
        // Single key
        if (!primitive({{"op", "key"}, {"key", arg}})) return unavailable();
        return {true, "Pressed key: " + arg, "ERR_NONE", "routine", "Key pressed", Colors::Green};
    }
}

}
