#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "commands_core.hpp"
#include "ai_reward.hpp"

namespace GRIM::RL
{
    // =====================================================
    // RL System Initialization
    // =====================================================

    // Start Python bridge (rl_bridge.py)
    bool init();

    // Stop Python bridge and flush pending transitions
    void shutdown();

    // =====================================================
    // RL Action Selection
    // =====================================================

    // Request RL model to choose next action given observation
    // Returns JSON containing { "action": idx, "suggested_command": "cmd" }
    nlohmann::json getAction(const nlohmann::json& obs);

    // =====================================================
    // RL Reward System
    // =====================================================

    // Feed result of command execution into reward system and buffer
    void processCommandResult(const CommandResult& result,
                              const std::string& lastCommand,
                              float execTime,
                              float sentimentScore,
                              float diversityFactor,
                              const std::string& mood);

    // Flush pending transitions to Python for PPO update
    void flushTransitions();

    // =====================================================
    // Helpers (Internal)
    // =====================================================

    // Build list of all live commands from commandMap
    std::vector<std::string> buildDynamicActionSpace();

    // Generate deterministic float embeddings for commands
    nlohmann::json embedCommands(const std::vector<std::string>& commands);
}
