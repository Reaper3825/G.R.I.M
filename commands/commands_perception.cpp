#include "commands_perception.hpp"
#include "perception/perception.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "logger.hpp"

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
