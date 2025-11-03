#include "action_executor.hpp"
#include "perception/perception_context.hpp"
#include "logger.hpp"
#include <algorithm>
#include <sstream>
#include <regex>

namespace GRIM {
namespace ActionExecutor {

bool isActionCommand(const std::string& input) {
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Keywords that indicate an action command
    const std::vector<std::string> actionKeywords = {
        "click", "press", "type", "write", "enter", "hit", "tap",
        "move", "scroll", "open", "close", "submit"
    };
    
    for (const auto& keyword : actionKeywords) {
        if (lower.find(keyword) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

ActionParams parseAction(const std::string& input) {
    ActionParams params;
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Match: "click [on] [the] <target>"
    std::regex clickPattern(R"((click|press|tap)(?:\s+on)?(?:\s+the)?\s+(.+))");
    std::smatch match;
    if (std::regex_search(lower, match, clickPattern)) {
        params.type = "click";
        params.target = match[2].str();
        return params;
    }
    
    // Match: "type <text>" or "write <text>" or "enter <text>"
    std::regex typePattern(R"((type|write|enter)\s+(.+))");
    if (std::regex_search(lower, match, typePattern)) {
        params.type = "type";
        // Use original input to preserve case
        size_t pos = input.find(match[2].str());
        if (pos != std::string::npos) {
            params.target = input.substr(pos);
        } else {
            params.target = match[2].str();
        }
        return params;
    }
    
    // Match: "press <key>" or "hit <key>"
    std::regex keyPattern(R"((press|hit|tap)\s+(.+))");
    if (std::regex_search(lower, match, keyPattern)) {
        params.type = "key";
        params.target = match[2].str();
        return params;
    }
    
    // Match: "move mouse to <x> <y>"
    std::regex movePattern(R"(move(?:\s+mouse)?\s+to\s+(\d+)\s+(\d+))");
    if (std::regex_search(lower, match, movePattern)) {
        params.type = "move";
        params.x = std::stoi(match[1].str());
        params.y = std::stoi(match[2].str());
        return params;
    }
    
    // Match: "scroll [up|down] [amount]"
    std::regex scrollPattern(R"(scroll(?:\s+(up|down))?(?:\s+(\d+))?)");
    if (std::regex_search(lower, match, scrollPattern)) {
        params.type = "scroll";
        params.modifier = match[1].matched ? match[1].str() : "down";
        params.target = match[2].matched ? match[2].str() : "120";
        return params;
    }
    
    // Match: "open <app>" or "launch <app>"
    std::regex openPattern(R"((open|launch|start|run)(?:\s+up)?\s+(.+))");
    if (std::regex_search(lower, match, openPattern)) {
        params.type = "open";
        params.target = match[2].str();
        return params;
    }
    
    // Match: "submit [form]" or "close [window]"
    if (lower.find("submit") != std::string::npos) {
        params.type = "submit";
        return params;
    }
    if (lower.find("close") != std::string::npos) {
        params.type = "close";
        return params;
    }
    
    return params;
}

CommandResult executeAction(const std::string& action) {
    logDebug("ActionExecutor", "Executing action: " + action);
    
    if (!Perception::g_contextManager) {
        return {false, "[Action] Perception system not initialized", "ERR_ACTION_NO_PERCEPTION", 
                "error", "Cannot execute action", Colors::Red};
    }
    
    ActionParams params = parseAction(action);
    
    if (params.type.empty()) {
        logDebug("ActionExecutor", "Could not parse action type from: " + action);
        return {false, "[Action] Could not understand action: " + action, "ERR_ACTION_UNKNOWN",
                "error", "I don't understand that action", Colors::Red};
    }
    
    try {
        if (params.type == "click") {
            // Intelligent clicking - try to find target on screen
            auto ctx = Perception::g_contextManager->getCurrentContext(true);
            
            if (!ctx.isValid) {
                return {false, "[Action] Cannot see screen", "ERR_ACTION_NO_VISION",
                        "error", "I can't see the screen", Colors::Red};
            }
            
            // Try to click on the target text or object
            Perception::g_contextManager->clickOnText(params.target);
            
            return {true, "Clicked on: " + params.target, "ERR_NONE",
                    "action", "Clicked on " + params.target, Colors::Green};
        }
        else if (params.type == "type") {
            Perception::g_contextManager->typeText(params.target, 15);
            
            return {true, "Typed: " + params.target, "ERR_NONE",
                    "action", "Typed text", Colors::Green};
        }
        else if (params.type == "key") {
            // Check if it's a combo (contains + or space between modifiers)
            if (params.target.find('+') != std::string::npos) {
                std::vector<std::string> keys;
                std::istringstream iss(params.target);
                std::string key;
                while (std::getline(iss, key, '+')) {
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    if (!key.empty()) keys.push_back(key);
                }
                Perception::g_contextManager->pressKeyCombo(keys);
                return {true, "Pressed: " + params.target, "ERR_NONE",
                        "action", "Pressed " + params.target, Colors::Green};
            } else {
                Perception::g_contextManager->tapKey(params.target);
                return {true, "Pressed: " + params.target, "ERR_NONE",
                        "action", "Pressed " + params.target, Colors::Green};
            }
        }
        else if (params.type == "move") {
            if (params.x >= 0 && params.y >= 0) {
                Perception::g_contextManager->moveMouseTo(params.x, params.y);
                return {true, "Moved mouse to (" + std::to_string(params.x) + ", " + 
                        std::to_string(params.y) + ")", "ERR_NONE",
                        "action", "Mouse moved", Colors::Green};
            }
        }
        else if (params.type == "scroll") {
            int delta = std::stoi(params.target);
            if (params.modifier == "up") delta = std::abs(delta);
            else delta = -std::abs(delta);
            
            Perception::g_contextManager->scrollMouse(delta);
            return {true, "Scrolled " + params.modifier, "ERR_NONE",
                    "action", "Scrolled", Colors::Green};
        }
        else if (params.type == "submit") {
            // Try common submit button text
            auto ctx = Perception::g_contextManager->getCurrentContext(true);
            if (ctx.hasText) {
                // Try to find and click common submit button text
                const std::vector<std::string> submitTexts = {
                    "Submit", "OK", "Send", "Continue", "Next", "Confirm"
                };
                
                for (const auto& text : submitTexts) {
                    if (ctx.screenText.find(text) != std::string::npos) {
                        Perception::g_contextManager->clickOnText(text);
                        return {true, "Submitted by clicking: " + text, "ERR_NONE",
                                "action", "Submitted", Colors::Green};
                    }
                }
            }
            
            // Fallback: press Enter
            Perception::g_contextManager->tapKey("enter");
            return {true, "Submitted (pressed Enter)", "ERR_NONE",
                    "action", "Submitted", Colors::Green};
        }
        else if (params.type == "close") {
            // Alt+F4 to close window
            Perception::g_contextManager->pressKeyCombo({"alt", "f4"});
            return {true, "Closed window", "ERR_NONE",
                    "action", "Window closed", Colors::Green};
        }
        else if (params.type == "open") {
            // Use Win+R to open run dialog, then type the app name
            Perception::g_contextManager->pressKeyCombo({"win", "r"});
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            Perception::g_contextManager->typeText(params.target);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            Perception::g_contextManager->tapKey("enter");
            
            return {true, "Opened: " + params.target, "ERR_NONE",
                    "action", "Opened " + params.target, Colors::Green};
        }
        
    } catch (const std::exception& e) {
        logError("ActionExecutor", "Exception executing action: " + std::string(e.what()));
        return {false, "[Action] Failed: " + std::string(e.what()), "ERR_ACTION_EXCEPTION",
                "error", "Action failed", Colors::Red};
    }
    
    return {false, "[Action] Unknown action type: " + params.type, "ERR_ACTION_UNKNOWN",
            "error", "Unknown action", Colors::Red};
}

} // namespace ActionExecutor
} // namespace GRIM
