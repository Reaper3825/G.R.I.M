#pragma once
//======================================================//
//  DecodeTimeNumPolicy — decode-time <NUM> selector ops
//
//  Exposes:
//    - Live candidate set L construction from ExecutionMemory
//    - Fixed deterministic slot_features[s] assembly (non-trainable)
//    - Null / Ambiguity / Selected policy decision from selector scores
//    - Mask-or-bind interface consumed by generation path
//    - resolveDecodeTimeNumSlotSelectionOrMask(): shared entry-point
//      for decode-time selector evaluation consumed by Phase2 generation.
//
//  Does NOT own:
//    - Trainable selector tensors (DecodeTimeSlotSelector owner)
//    - Learned slot-state encoding (DecodeTimeSlotSelector ops)
//    - Sampler mask application (generation path)
//    - ExecutionMemory allocation/lifecycle (ExecutionBlockLayer)
//    - Runtime candidate/scratch buffers (owned by DecodeTimeSelectorRuntime)
//======================================================//

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct cublasContext;
using cublasHandle_t = cublasContext*;
#endif

#include "DecodeTimeResolveResult.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

namespace Forward {
struct DecodeTimeSelectorRuntime;
}

namespace Batching {
struct BatchPayload;
struct BatchDeviceBindings;
}

struct ExecutionMemory;

//──────────────────────────────────────────────────────
// Slot feature layout for fixed feature assembly
//──────────────────────────────────────────────────────

// Fixed per-slot feature vector assembled from ExecutionMemory.
// Deterministic, non-trainable. Order matters for W_k_select.
//
// [0] slot_id         (float cast of integer slot index)
// [1] numeric_value   (from ExecutionMemory::values)
// [2] valid_bit       (from ExecutionMemory::valid_mask)
// [3] recent_write    (from ExecutionMemory::recent_write_mask)
// [4] usage_scalar    (from ExecutionMemory::usage)
// Static implementation capacity for the fixed feature assembler. The authored
// config value hp.d_slot_features must match this exactly; the constructor
// fails loud otherwise instead of silently changing feature layout semantics.
static constexpr int kSlotFeatureDim = 5;

//──────────────────────────────────────────────────────
// PolicyCandidateSet — assembled live slot data
//──────────────────────────────────────────────────────

// Validate the grouped selector construction contract that all decode-time
// selector ops rely on.
void validateDecodeTimeNumPolicyConfig(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp);

// Build the live candidate set L for one active row.
// BatchPayload owns row runtime semantics, BatchDeviceBindings owns the
// uploaded GPU addresses for that row, and ExecutionMemory owns the final
// slot state. Candidate feature packing remains GPU-only inside the caller-
// owned runtime workspace; host-side code receives only num_live_slots.
void buildDecodeTimeCandidateSet(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const ExecutionMemory& exec_memory,
    Forward::DecodeTimeSelectorRuntime& runtime,
    int batch_row,
    cudaStream_t stream);

// Evaluate selector scores to produce a selection decision.
//
//   d_scores:        device pointer [1 + num_live_slots] from selector
//   num_live_slots:  must match runtime.num_live_slots
//   stream:          CUDA stream
GRIM::SlotSelectionResult evaluateDecodeTimeSelectionScores(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const float* d_scores,
    Forward::DecodeTimeSelectorRuntime& runtime,
    int num_live_slots,
    cudaStream_t stream);

// Resolve a host-authored real slot id to the selector CE class index using
// the device compact candidate list. Returns -1 if the slot is absent.
int resolveDecodeTimeTargetIndexForSlot(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    int32_t target_slot,
    Forward::DecodeTimeSelectorRuntime& runtime,
    cudaStream_t stream);

//──────────────────────────────────────────────────────
// resolveDecodeTimeNumSlotSelectionOrMask
//
// Shared entry-point for decode-time selector evaluation.
// Called by Phase2 generation paths to produce a selection result.
//
// Semantics:
//   - Empty prompt_token_to_slot_map → all tokens mapped to -1
//     (non-state-bearing).
//   - slot_id == -1 → this token is non-state-bearing
//     (no slot selected).
//   - When selector resolves status == Selected, the
//     specific live slot's value is bound to the generated
//     <NUM> token. Otherwise <NUM> is masked (cannot be
//     generated).
//
// Writes result to out_* parameters. Sets out_valid=true
// when evaluation completes (even if result is Null).
//──────────────────────────────────────────────────────

// Forward declarations for types used by the resolver
struct DecodeTimeSlotSelector;

/// Evaluate the decode-time slot selector against the current execution memory.
///
/// @param selector       Trainable selector layer (nullable — returns invalid if null)
/// @param selector_hp    Grouped selector construction/read-view payload
/// @param payload        Active caller-authored payload for this forward step
/// @param bindings       Uploaded device bindings for the same payload
/// @param batch_row      Row whose execution state is being resolved
/// @param selector_enabled  Config flag controlling selector
/// @param exec_block_active Whether execution block was active for this sequence
/// @param has_exec_memory   Whether inference exec memory is populated
/// @param exec_memory       Current ExecutionMemory state (read-only)
/// @param d_hidden_state    Device pointer to hidden state [1, d_model]
/// @param stream            CUDA stream
/// @param cublas_handle     cuBLAS handle from the caller's runtime payload
///
/// @return DecodeTimeResolveResult with valid/status/selected_slot/selected_value
DecodeTimeResolveResult resolveDecodeTimeNumSlotSelectionOrMask(
    const DecodeTimeSlotSelector* selector,
    const HyperParameters::DecodeTimeSelectorConstructionHP& selector_hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    Forward::DecodeTimeSelectorRuntime& runtime,
    int batch_row,
    bool selector_enabled,
    bool exec_block_active,
    bool has_exec_memory,
    const ExecutionMemory& exec_memory,
    const float* d_hidden_state,
    cudaStream_t stream,
    cublasHandle_t cublas_handle);

} // namespace GRIM
