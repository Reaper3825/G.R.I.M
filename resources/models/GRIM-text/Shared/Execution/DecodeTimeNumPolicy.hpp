#pragma once
//======================================================//
//  DecodeTimeNumPolicy — decode-time <NUM> policy
//
//  Owns:
//    - Live candidate set L construction from ExecutionMemory
//    - Fixed deterministic slot_features[s] assembly (non-trainable)
//    - Null / Ambiguity / Selected policy decision from selector scores
//    - Mask-or-bind interface consumed by generation path
//    - resolveDecodeTimeNumSlotSelectionOrMask(): shared entry-point
//      for decode-time selector evaluation, consumed by both
//      inference (executeDecodeForward_) and generation paths.
//
//  Does NOT own:
//    - Trainable selector tensors (DecodeTimeSlotSelectorLayer)
//    - Learned slot-state encoding (DecodeTimeSlotSelectorLayer)
//    - Sampler mask application (generation path)
//    - ExecutionMemory allocation/lifecycle (ExecutionBlockLayer)
//======================================================//

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

#include <cstdint>
#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

//──────────────────────────────────────────────────────
// Selector policy result types
//──────────────────────────────────────────────────────

enum class SlotSelectionStatus : uint8_t {
    Selected,   // Exactly one legal slot resolved
    Null,       // Explicit null selection (no slot needed)
    Ambiguous   // Legal candidates exist but no unique winner
};

struct SlotSelectionResult {
    SlotSelectionStatus status;
    int32_t selected_slot;   // Valid only when status == Selected; actual slot index in L
    float confidence;        // top1 - top2 margin when applicable
};

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
static constexpr int kSlotFeatureDim = HyperParameters::DECODE_TIME_SLOT_FEATURE_DIM;

// Maximum supported live slots (matches DecodeTimeSlotSelectorLayer::kMaxSlots)
static constexpr int kPolicyMaxSlots = 16;

//──────────────────────────────────────────────────────
// PolicyCandidateSet — assembled live slot data
//──────────────────────────────────────────────────────

struct PolicyCandidateSet {
    int num_live_slots;                            // |L|
    int32_t live_slot_ids[kPolicyMaxSlots];         // Actual slot indices for each candidate
    float slot_features[kPolicyMaxSlots * kSlotFeatureDim]; // Host-side packed features
    float* d_slot_features;                         // Device pointer [num_live, kSlotFeatureDim]
};

//──────────────────────────────────────────────────────
// DecodeTimeNumPolicy
//──────────────────────────────────────────────────────

class DecodeTimeNumPolicy {
public:
    explicit DecodeTimeNumPolicy(const HyperParameters::DecodeTimeSelectorConstructionHP& config);
    ~DecodeTimeNumPolicy();

    DecodeTimeNumPolicy(DecodeTimeNumPolicy&& other) noexcept;
    DecodeTimeNumPolicy& operator=(DecodeTimeNumPolicy&& other) noexcept;

    DecodeTimeNumPolicy(const DecodeTimeNumPolicy&) = delete;
    DecodeTimeNumPolicy& operator=(const DecodeTimeNumPolicy&) = delete;

    // Build live candidate set L from ExecutionMemory fields.
    // All inputs are device pointers to ExecutionMemory tensors.
    //
    //   d_valid_mask:       [V]   float valid bits
    //   d_values:           [V,1] float scalar values
    //   d_recent_write:     [V]   float recent-write one-hot
    //   d_usage:            [V]   float usage scalars
    //   V:                  total slots
    //   scratch_slots:      S — slots [0..S-1] are scratch-only (excluded from L)
    //   stream:             CUDA stream for H2D copies
    //
    // Populates internal candidate set and uploads features to device.
    void buildCandidateSet(const float* d_valid_mask,
                           const float* d_values,
                           const float* d_recent_write,
                           const float* d_usage,
                           int V,
                           int scratch_slots,
                           cudaStream_t stream);

    // Evaluate selector scores to produce a selection decision.
    //
    //   d_scores:        device pointer [1 + num_live_slots] from selector
    //   num_live_slots:  must match candidates_.num_live_slots
    //   stream:          CUDA stream
    //
    // Returns SlotSelectionResult with status, selected_slot (real slot index), confidence.
    SlotSelectionResult evaluateScores(const float* d_scores,
                                       int num_live_slots,
                                       cudaStream_t stream);

    // Access candidate set after buildCandidateSet()
    const PolicyCandidateSet& candidates() const { return candidates_; }

    const HyperParameters::DecodeTimeSelectorConstructionHP& config() const { return config_; }

private:
    HyperParameters::DecodeTimeSelectorConstructionHP config_;
    PolicyCandidateSet candidates_;

    // Host staging buffers for H2D reads during candidate construction
    float* h_valid_mask_ = nullptr;
    float* h_values_ = nullptr;
    float* h_recent_write_ = nullptr;
    float* h_usage_ = nullptr;
    float* h_scores_ = nullptr;
};

//──────────────────────────────────────────────────────
// resolveDecodeTimeNumSlotSelectionOrMask
//
// Shared entry-point for decode-time selector evaluation.
// Called by both inference (executeDecodeForward_) and
// generation paths to produce a SlotSelectionResult.
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
class DecodeTimeSlotSelectorLayer;
struct ExecutionMemory;

struct DecodeTimeResolveResult {
    bool valid = false;
    SlotSelectionStatus status = SlotSelectionStatus::Null;
    int32_t selected_slot = -1;
    float selected_value = 0.0f;
};

/// Evaluate the decode-time slot selector against the current execution memory.
///
/// @param selector       Trainable selector layer (nullable — returns invalid if null)
/// @param policy         Decode-time policy (nullable — returns invalid if null)
/// @param selector_enabled  Config flag controlling selector
/// @param exec_block_active Whether execution block was active for this sequence
/// @param has_exec_memory   Whether inference exec memory is populated
/// @param exec_memory       Current ExecutionMemory state (read-only)
/// @param d_hidden_state    Device pointer to hidden state [1, d_model]
/// @param stream            CUDA stream
///
/// @return DecodeTimeResolveResult with valid/status/selected_slot/selected_value
DecodeTimeResolveResult resolveDecodeTimeNumSlotSelectionOrMask(
    DecodeTimeSlotSelectorLayer* selector,
    DecodeTimeNumPolicy* policy,
    bool selector_enabled,
    bool exec_block_active,
    bool has_exec_memory,
    const ExecutionMemory& exec_memory,
    const float* d_hidden_state,
    cudaStream_t stream);

} // namespace GRIM
