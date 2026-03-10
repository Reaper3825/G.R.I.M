#pragma once
#include <string>
#include <unordered_map>
#include "commands/commands_core.hpp"    // for CommandResult

struct RewardBreakdown {
    float baseReward = 0.0f;
    float timeBonus = 0.0f;
    float sentimentBonus = 0.0f;
    float categoryWeight = 0.0f;
    float diversityBonus = 0.0f;
    float total() const {
        return baseReward + timeBonus + sentimentBonus + categoryWeight + diversityBonus;
    }
};

class RewardEngine {
public:
    RewardEngine();

    // Main entry point
    RewardBreakdown computeReward(
        const CommandResult& result,
        const std::string& lastCommand,
        float execTime,
        float sentimentScore,
        float diversityFactor,
        const std::string& mood
    );

private:
    std::unordered_map<std::string, float> categoryWeights;
    float successReward;
    float failurePenalty;

    float getCategoryWeight(const std::string& category) const;
};
