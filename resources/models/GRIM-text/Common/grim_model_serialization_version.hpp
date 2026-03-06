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

// Same version used for GRMT training data format (read/write in DataLoader and training_data_loader).
inline constexpr std::uint32_t GRMT_FORMAT_VERSION = GRIM_MODEL_VERSION;

} // namespace GRIM
