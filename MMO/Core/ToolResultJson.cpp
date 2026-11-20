#include "ToolResultJson.hpp"

namespace GRIM::MMO {

std::string formatToolResult(const nlohmann::json& payload) {
    return payload.is_object() ? payload.dump() : "{}";
}

}
