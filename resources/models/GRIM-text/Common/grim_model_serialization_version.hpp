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
//   v10: Current GRIM-text checkpoint contract before per-channel LayerScale
//   v11: LayerScale weights are per-channel gamma vectors [d_model], not one scalar
//   v12: Removed ScratchBlock text-feature projection from the checkpoint schema
//   v13: Added NumberEncoder checkpoint weights
inline constexpr std::uint32_t GRIM_MODEL_VERSION = 13;

// GRMT training tensor stream may advance without bumping checkpoint MODEL_VERSION.
// v10: After per-token atom length-prefixed strings, append int32 token_exec_slots[len]
//      (ExecutionBlock bootstrap / token_to_slot_map).
// v11: After per-token atom strings, append compiled structured-execution payload:
//      uint8  execution_payload_active
//      int32  token_exec_slots[len]
//      uint32 compiled_bootstrap_binding_count
//      CompiledBootstrapBinding[count]  (binding_id, token_pos, slot_id — 12 bytes each)
//      uint32 teacher_step_count
//      TeacherStep[count]               (op_id, arg1_slot, arg2_slot, write_slot, expected_value — 20 bytes each)
// v12: Removed the per-token text feature side-channel from the GRMT stream. Atom metadata is
//      token_numeric_values + token_atom_mask + token_atom_flags + atom strings.
// v13: Removed the slot_selection_target stream (uint32 count + per-entry kind/slot_id). The
//      execution-entangled decode-time slot selector was deleted; numeric-meaning selector
//      supervision will arrive as a new channel (see docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md).
// v14: Replaced per-token atom-text reconstruction with direct per-sequence AtomTable persistence:
//      uint32 atom_entry_ids[len]
//      uint8  has_atom_table
//      AtomTable binary payload (entries + exact numeric payload arrays + string pool)
inline constexpr std::uint32_t GRMT_FORMAT_VERSION = 14;

} // namespace GRIM
