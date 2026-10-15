#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace GRIM::Batching {

// Rows retain their compiled context width, including padding. A token budget
// limits the rectangle; unused budget smaller than one row cannot be packed.
inline int rowsForTokenBudget(int token_budget, int context_window) {
    if (context_window <= 0 || token_budget < context_window) {
        throw std::runtime_error(
            "token budget must fit at least one complete context row");
    }
    return token_budget / context_window;
}

inline int tokenBudgetForRows(int rows, int context_window) {
    const auto tokens = static_cast<std::int64_t>(rows) * context_window;
    if (rows <= 0 || context_window <= 0 ||
        tokens > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid or overflowing context-row token budget");
    }
    return static_cast<int>(tokens);
}

} // namespace GRIM::Batching
