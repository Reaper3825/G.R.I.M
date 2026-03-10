#include "ai_rl.hpp"
#include "logger.hpp"
#include "bridge_manager.hpp"
#include "ai_reward.hpp"
#include "commands/commands_core.hpp"
#include <nlohmann/json.hpp>
#include <mutex>

namespace GRIM::RL
{

// =====================================================
// Constants / Globals
// =====================================================
static constexpr const char* RL_BRIDGE_ID = "rl";
static constexpr const char* RL_SCRIPT_PATH = "resources/python/rl_bridge.py";
static constexpr size_t BATCH_SIZE = 8;

static RewardEngine g_rewardEngine;
static std::vector<nlohmann::json> g_transitionBuffer;
static std::mutex g_batchMutex;

// =====================================================
// Helpers
// =====================================================

// Build a dynamic action space directly from GRIM's command registry
static std::vector<std::string> buildDynamicActionSpace()
{
    std::vector<std::string> actions;
    actions.reserve(commandMap.size());
    for (const auto& [cmd, _] : commandMap)
        actions.push_back(cmd);
    return actions;
}

// Generate lightweight deterministic embedding values per command
// (Placeholder — replace with NLP embeddings later)
static nlohmann::json embedCommands(const std::vector<std::string>& commands)
{
    nlohmann::json embeddings;
    for (const auto& cmd : commands)
    {
        uint64_t h = std::hash<std::string>{}(cmd);
        float vec = static_cast<float>((h % 1000000) / 1000000.0);
        embeddings[cmd] = vec;
    }
    return embeddings;
}

// =====================================================
// Init / Shutdown
// =====================================================
bool init()
{
    try
    {
        if (!BridgeManager::start(RL_BRIDGE_ID, RL_SCRIPT_PATH))
        {
            LOG_ERROR("RL", "Failed to start rl_bridge.py");
            return false;
        }
        LOG_DEBUG("RL", "rl_bridge.py started via BridgeManager.");
        g_transitionBuffer.clear();
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("RL", std::string("Init exception: ") + e.what());
        return false;
    }
}

void shutdown()
{
    try
    {
        flushTransitions();
        BridgeManager::stop(RL_BRIDGE_ID);
        LOG_DEBUG("RL", "rl_bridge.py stopped.");
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("RL", std::string("Shutdown exception: ") + e.what());
    }
}

// =====================================================
// Action Selection (Dynamic Command Space)
// =====================================================
nlohmann::json getAction(const nlohmann::json& obs)
{
    try
    {
        auto actions = buildDynamicActionSpace();
        auto cmdEmbeddings = embedCommands(actions);

        nlohmann::json req = {
            {"obs", obs},
            {"commands", actions},
            {"embeddings", cmdEmbeddings}
        };

        nlohmann::json res = BridgeManager::send(RL_BRIDGE_ID, req);

        // Interpret response
        if (res.contains("suggested_command"))
        {
            LOG_DEBUG("RL", "Python suggested command: " +
                res["suggested_command"].get<std::string>());
        }
        else if (res.contains("action"))
        {
            int idx = res["action"].get<int>();
            if (idx >= 0 && idx < static_cast<int>(actions.size()))
            {
                res["suggested_command"] = actions[idx];
                LOG_DEBUG("RL", "RL suggested index " + std::to_string(idx) +
                                " → " + actions[idx]);
            }
        }

        return res;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("RL", std::string("getAction exception: ") + e.what());
        return {{"error", e.what()}};
    }
}

// =====================================================
// Reward Feedback Loop
// =====================================================
void processCommandResult(const CommandResult& result,
                          const std::string& lastCommand,
                          float execTime,
                          float sentimentScore,
                          float diversityFactor,
                          const std::string& mood)
{
    try
    {
        RewardBreakdown rb = g_rewardEngine.computeReward(
            result, lastCommand, execTime, sentimentScore, diversityFactor, mood);

        float rewardValue = rb.total();

        LOG_REWARD(rb.baseReward, rb.timeBonus, rb.sentimentBonus,
                   rb.categoryWeight, rb.diversityBonus, rewardValue);

        nlohmann::json transition = {
            {"state", {
                {"command", lastCommand},
                {"category", result.category},
                {"mood", mood},
                {"success", result.success}
            }},
            {"reward", rewardValue},
            {"result", {
                {"message", result.message},
                {"errorCode", result.errorCode}
            }}
        };

        {
            std::lock_guard<std::mutex> lock(g_batchMutex);
            g_transitionBuffer.push_back(transition);

            if (g_transitionBuffer.size() >= BATCH_SIZE)
                flushTransitions();
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("RL", std::string("processCommandResult exception: ") + e.what());
    }
}

// =====================================================
// Batch Flushing
// =====================================================
void flushTransitions()
{
    std::lock_guard<std::mutex> lock(g_batchMutex);
    if (g_transitionBuffer.empty())
        return;

    try
    {
        nlohmann::json batch = {{"batch", g_transitionBuffer}};
        BridgeManager::send(RL_BRIDGE_ID, batch);
        LOG_DEBUG("RL", "Sent " + std::to_string(g_transitionBuffer.size()) +
                         " RL transitions (batch flush).");
        g_transitionBuffer.clear();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("RL", std::string("flushTransitions exception: ") + e.what());
    }
}

} // namespace GRIM::RL
