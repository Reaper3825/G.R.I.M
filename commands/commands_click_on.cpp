#include "commands_perception.hpp"
#include "perception/perception_context.hpp"
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

namespace {
std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}
}

CommandResult cmdInputClickOn(const std::string& arg) {
    using namespace GRIM::Perception;
    if (!g_contextManager)
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED",
                "error", "Input system not available", Colors::Red};
    auto target = trim(arg);
    auto kind = ClickTargetKind::Text;
    const auto separator = target.find_first_of(" \t\r\n");
    auto prefix = target.substr(0, separator);
    for (auto& c : prefix) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (prefix == "text" || prefix == "object") {
        kind = prefix == "object" ? ClickTargetKind::Object : ClickTargetKind::Text;
        target = separator == std::string::npos ? "" : trim(target.substr(separator + 1));
    }
    if (target.empty() || target.find('\0') != std::string::npos)
        return {false, "[Input] Supply an exact text label or native control name, ID, or role",
                "ERR_INVALID_ARGS", "error", "No valid click target specified", Colors::Red};
    const auto result = kind == ClickTargetKind::Text ? g_contextManager->clickOnText(target)
                                                     : g_contextManager->clickOnObject(target);
    CommandResult response{result.success, "[Input] " + result.message,
        result.success ? "ERR_NONE" : result.error, result.success ? "routine" : "error",
        result.success ? "Click input sent" : "Click target failed", result.success ? Colors::Green : Colors::Red};
    if (result.success)
        response.structuredOutputJson = nlohmann::json{{"input_sent", true}, {"application_response_verified", false},
            {"target", target}, {"target_kind", kind == ClickTargetKind::Text ? "text" : "object"},
            {"desktop_x", result.x}, {"desktop_y", result.y}}.dump();
    return response;
}
