#include "ai_reward.hpp"
#include "logger.hpp"
#include <sstream>
#include <iomanip>

RewardEngine::RewardEngine() {
    // Tunable parameters
    successReward = 1.0f;
    failurePenalty = -1.0f;

    // Weighted categories
    categoryWeights = {
        {"user_feedback_positive", 1.0f},
        {"user_feedback_negative", -1.0f},
        {"system_stable", 0.2f},
        {"critical_error", -2.0f},
        {"novelty_discovery", 0.5f},
        {"default", 0.0f}
    };
}

float RewardEngine::getCategoryWeight(const std::string& cat) const {
    auto it = categoryWeights.find(cat);
    return (it != categoryWeights.end()) ? it->second : categoryWeights.at("default");
}

RewardBreakdown RewardEngine::computeReward(
    const CommandResult& result,
    const std::string& lastCommand,
    float execTime,
    float sentimentScore,
    float diversityFactor,
    const std::string& mood
) {
    RewardBreakdown r;

    // Base success/failure reward
    r.baseReward = result.success ? successReward : failurePenalty;

    // Execution time bonus (faster = better)
    r.timeBonus = (1.0f - std::clamp(execTime / 5.0f, 0.0f, 1.0f)) * 0.2f;

    // Sentiment bonus (-1 to +1)
    r.sentimentBonus = sentimentScore * 0.5f;

    // Category weighting
    r.categoryWeight = getCategoryWeight(result.category);

    // Diversity / novelty encouragement
    r.diversityBonus = diversityFactor * 0.3f;

    // Optional mood adjustment (example: curious mood gives exploration bonus)
    if (mood == "Curious") r.diversityBonus += 0.2f;
    if (mood == "Irritated") r.sentimentBonus *= 0.5f;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "Reward breakdown => Base: " << r.baseReward
        << " | Time: " << r.timeBonus
        << " | Sentiment: " << r.sentimentBonus
        << " | Category: " << r.categoryWeight
        << " | Diversity: " << r.diversityBonus
        << " | Total: " << r.total();
    LOG_DEBUG("Reward", oss.str());

    return r;
}
