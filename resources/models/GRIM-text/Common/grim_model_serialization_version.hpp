#pragma once

#include <cstdint>

namespace GRIM {

// Bump when checkpoint format or baked training semantics change in a way that
// requires explicit compatibility enforcement.
inline constexpr std::uint32_t GRIM_MODEL_VERSION = 7;  // Issue #33: Added final_rms_gamma

} // namespace GRIM
