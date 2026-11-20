#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace GRIM::MMO {

std::string formatToolResult(const nlohmann::json& payload);

}
