#pragma once
#include <cstdint>

namespace GRIM::MMO {

// Per-invocation feedback/execution source; independent of tool metadata.
enum class ToolExecutionMode : std::uint8_t {
    Production = 0,
    Sandbox = 1, // Reserved; no simulator is implemented yet.
};

} // namespace GRIM::MMO
