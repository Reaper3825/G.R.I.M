#pragma once

#include <nlohmann/json.hpp>
#include <string_view>

namespace GRIM::MMO {

nlohmann::json formatToolResult(std::string_view payload);

}
