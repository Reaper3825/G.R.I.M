#pragma once

#include <cstdint>

namespace GRIM {

// Half-open [begin, end) token range produced by an invisible logical
// delimiter pair. Delimiter text is never inserted into model input_ids.
struct GoalTokenSpan {
    std::int32_t begin = -1;
    std::int32_t end = -1;

    bool valid() const noexcept { return begin >= 0 && end > begin; }
    std::int32_t length() const noexcept { return valid() ? end - begin : 0; }
};

} // namespace GRIM
