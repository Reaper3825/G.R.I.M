#pragma once
#include "commands/commands_core.hpp"
#include <string>

namespace GRIM {
namespace ActionExecutor {

// Intelligent action execution - understands natural commands and executes them
// Integrates with perception to see + interact with the screen
// Called automatically when GRIM's AI detects an action intent

CommandResult executeAction(const std::string& action);

// Check if a command is an action-type command (input control)
bool isActionCommand(const std::string& input);

// Extract action parameters from natural language
struct ActionParams {
    std::string type;      // "click", "type", "move", "press", etc.
    std::string target;    // What to click on, what to type, etc.
    int x = -1, y = -1;    // Coordinates if specified
    std::string modifier;  // "left", "right", "ctrl", etc.
};

ActionParams parseAction(const std::string& input);

} // namespace ActionExecutor
} // namespace GRIM
