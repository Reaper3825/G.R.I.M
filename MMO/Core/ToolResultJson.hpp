#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>

namespace GRIM::MMO {

inline nlohmann::json makeToolResult(bool success, std::string_view message,
                                     std::string_view errorCode, std::string_view category,
                                     nlohmann::json payload = nlohmann::json::object()) {
    const auto* status = success ? "completed" :
        (errorCode == "ERR_TOOL_COMPLETION_UNKNOWN" ? "unknown" : "failed");
    return {{"status", status}, {"error_code", errorCode}, {"message", message},
            {"category", category}, {"payload", std::move(payload)}};
}

std::string formatToolResult(const nlohmann::json& payload);

}
