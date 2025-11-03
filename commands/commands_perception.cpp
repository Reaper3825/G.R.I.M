#include "commands_perception.hpp"
#include "perception/perception.hpp"
#include "perception/perception_context.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "logger.hpp"
#include <sstream>
#include <algorithm>

// =============================================================
// Perception/Vision Commands
// =============================================================

CommandResult cmdPerceptionWhat(const std::string& arg) {
    LOG_TRACE("Perception", "cmdPerceptionWhat called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::isAvailable()) {
        return {
          false,
   "[Perception] System not initialized",
            "ERR_PERCEPTION_NOT_INITIALIZED",
            "error",
         "Perception system not available",
        Colors::Red
        };
    }
    
    std::string result = GRIM::Perception::analyzeScreen();
    
    return {
        true,
    result,
        "ERR_NONE",
        "routine",
      result,
 Colors::Cyan
    };
}

CommandResult cmdPerceptionRead(const std::string& arg) {
    LOG_TRACE("Perception", "cmdPerceptionRead called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::isAvailable()) {
        return {
      false,
 "[Perception] System not initialized",
    "ERR_PERCEPTION_NOT_INITIALIZED",
     "error",
   "Perception system not available",
  Colors::Red
        };
    }
    
    std::string result = GRIM::Perception::readText();
    
    return {
      true,
    result,
        "ERR_NONE",
        "routine",
        result,
        Colors::Cyan
    };
}

CommandResult cmdPerceptionDetect(const std::string& arg) {
    LOG_TRACE("Perception", "cmdPerceptionDetect called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::isAvailable()) {
        return {
            false,
       "[Perception] System not initialized",
       "ERR_PERCEPTION_NOT_INITIALIZED",
 "error",
    "Perception system not available",
     Colors::Red
        };
    }
    
    std::string result = GRIM::Perception::detectObjects();
    
    return {
  true,
 result,
        "ERR_NONE",
    "routine",
        result,
        Colors::Cyan
    };
}


// =============================================================
// Input Control Commands
// =============================================================

CommandResult cmdInputMoveMouse(const std::string& arg) {
    // Expected format: "x,y" or "x y"
    LOG_TRACE("Input", "cmdInputMoveMouse called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::g_contextManager) {
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
    
    GRIM::Perception::g_contextManager->moveMouseTo(x, y);
    
    return {true, "Mouse moved to (" + std::to_string(x) + ", " + std::to_string(y) + ")", "ERR_NONE", "routine", "Mouse moved", Colors::Green};
}

CommandResult cmdInputClick(const std::string& arg) {
    // Expected format: "left" | "right" | "middle" | ""
    LOG_TRACE("Input", "cmdInputClick called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::g_contextManager) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }
    
    std::string button = arg.empty() ? "left" : arg;
    GRIM::Perception::g_contextManager->clickMouse(button);
    
    return {true, "Clicked " + button + " mouse button", "ERR_NONE", "routine", "Clicked", Colors::Green};
}

CommandResult cmdInputType(const std::string& arg) {
    // Expected format: "text to type"
    LOG_TRACE("Input", "cmdInputType called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::g_contextManager) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }
    
    if (arg.empty()) {
        return {false, "[Input] No text provided. Use: type hello world", "ERR_INVALID_ARGS", "error", "No text to type", Colors::Red};
    }
    
    GRIM::Perception::g_contextManager->typeText(arg, 15); // 15ms delay for natural typing
    
    return {true, "Typed: " + arg, "ERR_NONE", "routine", "Text typed", Colors::Green};
}

CommandResult cmdInputKey(const std::string& arg) {
    // Expected format: "enter" | "ctrl+c" | "alt+f4"
    LOG_TRACE("Input", "cmdInputKey called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::g_contextManager) {
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
        
        GRIM::Perception::g_contextManager->pressKeyCombo(keys);
        return {true, "Pressed key combo: " + arg, "ERR_NONE", "routine", "Key combo pressed", Colors::Green};
    } else {
        // Single key
        GRIM::Perception::g_contextManager->tapKey(arg);
        return {true, "Pressed key: " + arg, "ERR_NONE", "routine", "Key pressed", Colors::Green};
    }
}

CommandResult cmdInputClickOn(const std::string& arg) {
    // Expected format: "text Button Name" or "object button"
    LOG_TRACE("Input", "cmdInputClickOn called with arg=\"" + arg + "\"");
    
    if (!GRIM::Perception::g_contextManager) {
        return {false, "[Input] Perception system not initialized", "ERR_INPUT_NOT_INITIALIZED", "error", "Input system not available", Colors::Red};
    }
    
    if (arg.empty()) {
        return {false, "[Input] No target provided. Use: click on text Submit or click on object button", "ERR_INVALID_ARGS", "error", "No target specified", Colors::Red};
    }
    
    // Parse: "text ..." or "object ..."
    std::string lower = arg;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower.find("text ") == 0) {
        std::string target = arg.substr(5); // Remove "text "
        GRIM::Perception::g_contextManager->clickOnText(target);
        return {true, "Attempting to click on text: " + target, "ERR_NONE", "routine", "Clicking on text", Colors::Green};
    } else if (lower.find("object ") == 0) {
        std::string target = arg.substr(7); // Remove "object "
        GRIM::Perception::g_contextManager->clickOnObject(target);
        return {true, "Attempting to click on object: " + target, "ERR_NONE", "routine", "Clicking on object", Colors::Green};
    } else {
        // Try as text by default
        GRIM::Perception::g_contextManager->clickOnText(arg);
        return {true, "Attempting to click on: " + arg, "ERR_NONE", "routine", "Clicking", Colors::Green};
    }
}
