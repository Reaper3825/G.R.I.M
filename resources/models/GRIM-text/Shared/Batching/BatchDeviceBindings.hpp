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
//    for this step (slot map, atom mask, etc.). Those come
//    from TrainingState's reusable cache buffers AFTER the
//    H2D copies for this batch land. Naming them on a
//    struct that is passed alongside `BatchPayload` keeps
//    the host-vs-device split explicit (no hidden state on
//    payload, no implicit "current batch" on TrainingState).
//
//  OWNERSHIP
//  =========
//  BatchDeviceBindings owns NOTHING. The underlying device
//  memory is owned by TrainingState (cached_token_ids_tensor,
//  cached_token_to_slot_map, etc.). BatchDeviceBindings is a
//  view: pointers + the geometry used to interpret them.
//
//  LIFETIME
//  ========
//  Valid only between the upload's final stream sync and the
//  next H2D for a different batch. Phase2 must not cache one
//  beyond the step that produced it.
//======================================================//

#pragma once

#include <cstdint>

namespace GRIM {
namespace Batching {

// Forward declaration.
struct BatchPayload;

// Device pointers for a single batch step. Filled by
// LanguageModel::uploadBatchToDevice() after H2D copies complete;
// consumed by autogradTrainingStep / executeStep
// without ever writing back through this struct.
//
// Geometry (batch_size, max_seq_len) is duplicated from BatchPayload
// purely as a row-stride hint for kernels that index bindings without
// the payload (e.g. inference decode). The authoritative geometry
// always lives on BatchPayload.
struct BatchDeviceBindings {
    int*      d_input_ids       = nullptr;  // [batch_size * max_seq_len]
    int*      d_target_ids      = nullptr;  // [batch_size * max_seq_len]
    int*      d_seq_lengths     = nullptr;  // [batch_size] real token count per padded row
    float*    d_numeric_values  = nullptr;  // [batch_size * max_seq_len]
    uint8_t*  d_atom_mask       = nullptr;  // [batch_size * max_seq_len] (nullable when atom mask not used)
    uint32_t* d_atom_flags      = nullptr;  // [batch_size * max_seq_len] (nullable when not allocated)
    int32_t*  d_token_to_slot_map = nullptr; // [batch_size * max_seq_len]
    int*      d_mtp_shifted_targets = nullptr; // [payload.mtp_shifted_targets.size() * batch_size * max_seq_len], head-major; nullable when MTP disabled

    int batch_size  = 0;
    int max_seq_len = 0;
};

}  // namespace Batching
}  // namespace GRIM
