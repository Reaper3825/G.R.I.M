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
    int*      d_target_ids      = nullptr;  // [payload.total_tokens] for training payloads
    float*    d_numeric_values  = nullptr;  // [payload.total_tokens]
    uint8_t*  d_atom_mask       = nullptr;  // [payload.total_tokens] (nullable when atom mask not used)
    uint32_t* d_atom_flags      = nullptr;  // [payload.total_tokens] (nullable when not allocated)
    int32_t*  d_token_to_slot_map = nullptr; // [payload.total_tokens]
    int*      d_atom_positions  = nullptr;  // [payload.authoredAtomCount()] compact authored atom token positions
    int*      d_atom_types      = nullptr;  // [payload.authoredAtomCount()] compact authored atom types aligned with d_atom_positions
    int*      d_mtp_shifted_targets = nullptr; // [payload.mtp_shifted_targets.size() * payload.total_tokens], head-major; nullable when MTP disabled
};

}  // namespace Batching
}  // namespace GRIM
