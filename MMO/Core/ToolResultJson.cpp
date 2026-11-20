#include "ToolResultJson.hpp"

namespace GRIM::MMO {

nlohmann::json formatToolResult(std::string_view payload) {
    if (payload.empty()) return nullptr;

    auto parsed = nlohmann::json::parse(payload, nullptr, false);
    // The plugin bridge rejects malformed JSON. Native handlers keep their
    // original command outcome when their payload cannot be parsed.
    if (parsed.is_discarded()) return nullptr;
    return parsed;
}

}
