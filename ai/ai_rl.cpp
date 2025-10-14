#include "ai_rl.hpp"
#include "logger.hpp"
#include "bridge_manager.hpp"  // same one used for Coqui
#include <nlohmann/json.hpp>

namespace GRIM::RL {

static constexpr const char* RL_BRIDGE_ID = "rl";
static constexpr const char* RL_SCRIPT_PATH =
    "resources/python/rl_bridge.py";

bool init() {
    try {
        if (!BridgeManager::start(RL_BRIDGE_ID, RL_SCRIPT_PATH)) {
            LOG_ERROR("RL", "Failed to start rl_bridge.py");
            return false;
        }
        LOG_DEBUG("RL", "rl_bridge.py started via BridgeManager.");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Init exception: ") + e.what());
        return false;
    }
}

nlohmann::json getAction(const nlohmann::json& obs) {
    try {
        nlohmann::json req = {{"obs", obs}};
        nlohmann::json res = BridgeManager::send(RL_BRIDGE_ID, req);
        return res;
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("getAction exception: ") + e.what());
        return {{"error", e.what()}};
    }
}

void shutdown() {
    try {
        BridgeManager::stop(RL_BRIDGE_ID);
        LOG_DEBUG("RL", "rl_bridge.py stopped.");
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Shutdown exception: ") + e.what());
    }
}

} // namespace GRIM::RL
