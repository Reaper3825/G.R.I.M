#pragma once

#include <cstdint>

namespace GRIM {

// Bump when checkpoint format or baked training semantics change in a way that
// requires explicit compatibility enforcement.
// Version history:
//   v1: Initial schema (2025)
//   v2: Added ScratchBlock reasoning layer (atom embeddings, projection)
//   v3: Added numeric head for number prediction
//   v4: Separated positional encodings (ALiBi vs RoPE vs learned vs HYBRID)
//   v5: Added GQA support (num_kv_heads)
//   v6: Added FlatBuffer schema upgrades (checksum, timestamps)
//   v7: Added final_rms_gamma for Issue #33 (encoder output normalization)
//   v8: Added LayerScale weights (Issue #109), QK-norm alpha scales, learned loss weighting
//   v9: Align checkpoint version with GRMT data v9 (single <NUM> atom, current schema)
inline constexpr std::uint32_t GRIM_MODEL_VERSION = 9;

// GRMT training tensor stream may advance without bumping checkpoint MODEL_VERSION.
// v10: After per-token atom length-prefixed strings, append int32 token_exec_slots[len]
//      (ExecutionBlock bootstrap / token_to_slot_map).
inline constexpr std::uint32_t GRMT_FORMAT_VERSION = 10;

} // namespace GRIM
