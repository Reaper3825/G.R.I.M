#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace BridgeManager {
    bool start(const std::string& id, const std::string& scriptPath);
    nlohmann::json send(const std::string& id, const nlohmann::json& message);
    void stop(const std::string& id);
}
