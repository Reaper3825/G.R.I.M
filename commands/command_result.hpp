#pragma once
#include <functional>
#include <string>
#include <nlohmann/json.hpp>
#include "helpers/color.hpp"

// ====================================================
// Command result structure
// ====================================================
struct CommandResult {
    bool success = true;
    std::string message;
    std::string errorCode;
    std::string category;
    std::string voice;
    Color color = Colors::Default;
    std::string structuredOutputJson;
    // Provider-authored data. The registry supplies an empty object when omitted.
    nlohmann::json structuredOutput;
};

// ====================================================
// Command function type
// ====================================================
using CommandFunc = std::function<CommandResult(const std::string&)>;

