#pragma once
#include <ctime>
#include <string>
#include <vector>

namespace GRIM {

struct ContextSnapshot {
    std::vector<std::string> recentIntents;
    std::vector<std::string> recentCommands;
    std::string currentMood;
    int conversationDepth = 0;

    std::string lastNlpCategory;
    int consecutiveCommands = 0;
    std::time_t lastCommandTime = 0;
};

} // namespace GRIM
