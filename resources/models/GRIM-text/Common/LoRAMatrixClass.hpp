#pragma once

#include <cstdint>

namespace GRIM {

enum class LoRAMatrixClass : std::uint8_t {
    QKV,
    ATTENTION_OUTPUT,
    FFN_GATE,
    FFN_UP,
    FFN_DOWN
};

} // namespace GRIM