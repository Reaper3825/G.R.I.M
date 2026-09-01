//======================================================//
//  BatchDeviceBindings.hpp
//
//  Explicit, first-class struct of device pointers for
//  ONE training/eval step. Replaces the old `mutable d_*`
//  fields on BatchPayload, which were the only mechanism
//  by which the upload path stashed device addresses for
//  downstream forward/loss code to read back.
//
//  RATIONALE
//  =========
//  - BatchPayload is host-only, immutable, POD-like. After
//    buildBatchPayload() returns, no field may be written
//    by Phase2, the loss path, or the upload path.
//  - The forward/loss path still needs *device* addresses
//    for this step (slot map, atom mask, etc.). Naming them on
//    a struct that is passed alongside `BatchPayload` keeps the
//    host-vs-device split explicit (no hidden state on payload,
//    no implicit "current batch" on TrainingState).
//
//  OWNERSHIP
//  =========
//  BatchDeviceBindings owns NOTHING. The underlying device memory is borrowed
//  from the BatchPayload-attached BatchDeviceStorage owner for the active
//  upload boundary. Fixed-shape training/eval geometry is authored by
//  HyperParameters/HyperparameterGroupings and enforced at the upload
//  boundary; this struct is only the borrowed device-address view.
//
//  LIFETIME
//  ========
//  Valid only between the upload's final stream sync and the next upload that
//  reuses the same BatchDeviceStorage owner. Phase2 must not cache one beyond
//  the step that produced it.
//======================================================//

#pragma once

#include <cstdint>

namespace GRIM {
namespace Batching {

// Forward declaration.
struct BatchPayload;

// Device pointers for a single batch step. Filled by
// Batching::uploadBatchToDevice() after H2D copies complete;
// consumed by autogradTrainingStep / executeStep
// without ever writing back through this struct.
//
struct BatchDeviceBindings {
    int*      d_input_ids       = nullptr;  // [payload.total_tokens]
    const int* d_sequence_lengths = nullptr; // [payload.batch_size], valid tokens through EOS
    int*      d_target_ids      = nullptr;  // [payload.total_tokens] for ordinary LM training
    uint8_t*  d_atom_insertion_gap_targets = nullptr; // [gap_rows, kAtomDecisionClassCount]
    uint8_t*  d_atom_insertion_valid_gap_mask = nullptr; // [gap_rows]
    float*    d_numeric_values  = nullptr;  // [payload.total_tokens]
    uint8_t*  d_atom_mask       = nullptr;  // [payload.total_tokens] (nullable when atom mask not used)
    uint32_t* d_atom_flags      = nullptr;  // [payload.total_tokens] (nullable when not allocated)
    uint32_t* d_atom_entry_ids  = nullptr;  // [payload.total_tokens], row-local AtomTable entry id
    uint32_t* d_token_local_atom_indices = nullptr; // [payload.total_tokens], nullable when retrieval is disabled
    int32_t*  d_token_to_slot_index_map = nullptr; // [payload.total_tokens]
    int*      d_atom_positions  = nullptr;  // [payload.authoredAtomCount()] compact authored atom token positions
    int*      d_atom_types      = nullptr;  // [payload.authoredAtomCount()] compact authored atom types aligned with d_atom_positions

    // Compact sequence-local selector metadata. Candidate banks are segmented
    // by [row, AtomType] and ordered by local_index. Pointers are null when the
    // compiled retrieval feature is disabled; counts are zero when the active
    // payload has no corresponding rows/data.
    int* d_local_atom_query_positions = nullptr; // [local_atom_query_count]
    int* d_local_atom_query_types = nullptr;     // [local_atom_query_count]
    int* d_local_atom_query_targets = nullptr;   // [local_atom_query_count], 0 or local_index + 1
    int* d_local_atom_row_type_candidate_offsets = nullptr; // [batch_size * kAtomTypeCount + 1]
    int* d_local_atom_candidate_first_close_positions = nullptr; // [local_atom_candidate_count]
    int* d_local_atom_candidate_content_offsets = nullptr; // [local_atom_candidate_count + 1]
    int* d_local_atom_candidate_content_positions = nullptr; // [local_atom_content_position_count]
    int local_atom_query_count = 0;
    int local_atom_candidate_count = 0;
    int local_atom_content_position_count = 0;

    // Candidate atom-entry pool (arg/option selector). Batch-global "menu" of
    // options the selector scores; row r's window is
    // [d_row_atom_offset[r], d_row_atom_offset[r+1]). Nullable when the
    // selector is disabled (num_pool_atoms == 0).
    float*    d_pool_numeric_values = nullptr; // [num_pool_atoms]
    double*   d_pool_numeric_float_values = nullptr; // [num_pool_atoms], exact float payload
    int64_t*  d_pool_numeric_int_values = nullptr; // [num_pool_atoms], exact integer payload
    uint8_t*  d_pool_numeric_kinds = nullptr; // [num_pool_atoms], NumericPayloadKind
    int*      d_pool_atom_types     = nullptr; // [num_pool_atoms]
    int*      d_row_atom_offset     = nullptr; // [batch_size + 1]
    int       num_pool_atoms        = 0;

    // Arg/option selector supervision: per-token batch-global target pool index
    // (or -1). Nullable when the selector is disabled.
    int*      d_arg_select_targets = nullptr; // [total_tokens]
};

}  // namespace Batching
}  // namespace GRIM
