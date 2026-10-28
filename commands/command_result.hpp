#pragma once
#include <functional>
#include <string>
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
};

// ====================================================
// Command function type
// ====================================================
using CommandFunc = std::function<CommandResult(const std::string&)>;

