//======================================================//
//  ExecutionMetadata.hpp
//  Single cross-layer definition site for semantic
//  execution metadata types.
//
//  This header defines the canonical structured execution
//  record and all compiled payload types derived from it.
//  It is the ONLY legal definition site for these types.
//
//  Must NOT own: JSON parsing, tokenization, padding/remap,
//  validation logic, CUDA/autograd code, GRMT IO.
//======================================================//

#pragma once

#include "SlotIdentity.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Execution {

// =============================================================================
// ExecutionGateTarget — supervised activation label
//
// IGNORE is intentionally distinct from NOOP. Ordinary plaintext rows often
// have no authored execution trace, but absence of annotation is not evidence
// that execution is unnecessary. Only explicitly-labelled negatives may use
// NOOP; verified structured programs use EXECUTE.
// =============================================================================
enum class ExecutionGateTarget : int8_t {
    UNSUPERVISED = -1,
    NOOP = 0,
    EXECUTE = 1
};

inline bool isValidExecutionGateTarget(ExecutionGateTarget target) {
    return target == ExecutionGateTarget::UNSUPERVISED
        || target == ExecutionGateTarget::NOOP
        || target == ExecutionGateTarget::EXECUTE;
}

// =============================================================================
// TeacherStep — per-step ground truth for execution supervision
//
// This is the execution supervision projection of the canonical
// StructuredExecutionRecord. It answers only:
//   - which slot should be read as arg1
//   - which slot should be read as arg2
//   - which op should fire
//   - which slot should receive the write
//   - what scalar result is expected
//
// compiled_slot_bindings and teacher_steps are paired projections of one
// canonical record and must never be authored independently.
// =============================================================================
struct TeacherStep {
    int op_id = -1;
    SlotId arg1_slot;
    SlotId arg2_slot;
    SlotId write_slot;
    float expected_value = 0.0f;
};

// =============================================================================
// CompiledSlotBinding — explicit semantic/runtime lowering for one row
//
// SlotId remains stable for the semantic episode. SlotIndex is a dense register
// address scoped to this compiled row payload. Neither value may be inferred
// from the other; every lowering and trace materialization crosses this table.
// =============================================================================
struct CompiledSlotBinding {
    SlotId slot_id;
    SlotIndex slot_index;
};

// =============================================================================
// CompiledBootstrapBinding — compiled provenance for one bootstrap literal
//
// Each binding identifies which token position initializes which slot.
// The binding set for a row must be injective in both dimensions:
//   - no two bindings target the same token_pos
//   - no two bindings initialize the same slot_id
//
// Only bootstrap literal bindings create state-bearing token positions.
// Not every numeric token is state-bearing — only those explicitly
// named in bootstrap_bindings[].
// =============================================================================
struct CompiledBootstrapBinding {
    int32_t binding_id = -1; // Ordinal within the row's bootstrap binding set
    int32_t token_pos = -1;  // Token position that initializes this slot
    SlotIndex slot_index;   // Dense runtime slot initialized at this position
};

// =============================================================================
// BootstrapLiteralBinding — builder-side semantic binding
//
// Each binding identifies a semantic literal in the canonical structured
// record, the slot it must initialize, and its role.
// This is the builder-side source; CompiledBootstrapBinding is the
// compiled form after tokenization.
// =============================================================================
struct BootstrapLiteralBinding {
    int32_t literal_id;         // Ordinal within the structured record
    SlotId slot_id;             // Semantic slot this literal initializes
    int32_t occurrence_role;    // Semantic role identifier
    int32_t rendered_span_id;   // Identifier for the rendered text span
};

// =============================================================================
// StructuredExecutionRecord — canonical semantic source of truth
//
// This is the ACTUAL truth for execution-active rows. It exists BEFORE
// tokenization and is the only place where execution meaning originates.
//
// Runtime slot indices and teacher slot identities are derived projections.
// bootstrap_bindings defines the only literals allowed to seed registers.
//
// Rules:
//   - bootstrap_bindings must be non-empty for execution-active rows
//   - each bootstrap binding has unique slot_id (no duplicate initialization)
//   - slot identity is stable for the entire row lifetime
//   - teacher-step writes may target non-bootstrapped slots if in D_row
//   - teacher steps may not invent or rebind slot identity mid-row
// =============================================================================
struct StructuredExecutionRecord {
    bool execution_active = false;
    ExecutionGateTarget execution_gate_target = ExecutionGateTarget::UNSUPERVISED;

    // Bootstrap literal bindings — define which literals seed which registers
    std::vector<BootstrapLiteralBinding> bootstrap_bindings;

    // Ordered execution steps with expected outputs
    struct ExecutionStep {
        int op_id;
        SlotId arg1_slot;
        SlotId arg2_slot;
        SlotId write_slot;
        float expected_value;
    };
    std::vector<ExecutionStep> steps;

    // Explicit row-local slot domain D_row:
    // The unique set of slot ids referenced by this row's execution program.
    // Includes bootstrap binding slot ids + teacher-step read/write slot ids.
    // Fixed before tokenization, does not change during execution.
    // Compiled indices for D_row lie within [0, V); identities have no range semantics.
    std::vector<SlotId> slot_domain;
};

// =============================================================================
// CompiledStructuredExecutionPayload — compiled runtime/supervision payload
//
// Derived from StructuredExecutionRecord by the canonical builder.
// This is what GrmtSequence, GRMT, and BatchPayload carry.
//
// execution_active is the AUTHORITATIVE activation bit.
// Non-empty teacher_steps is a supervised-training payload validity
// requirement, NOT the activation source.
//
// Runtime D_row is reconstructed as:
//   D_row = { b.slot_id | b ∈ compiled_slot_bindings }
//           ∪ { step.arg1_slot, step.arg2_slot, step.write_slot | step ∈ teacher_steps }
// It is NOT serialized separately.
//
// token_exec_slot_indices is compiled ONLY from bootstrap_bindings[], not from
// arbitrary numeric tokens in rendered text.
// =============================================================================
struct CompiledStructuredExecutionPayload {
    bool execution_active = false;
    ExecutionGateTarget execution_gate_target = ExecutionGateTarget::UNSUPERVISED;

    // Prefix-only planner observation boundary. Positions are row-relative.
    // The gate reads the final token of the complete prompt. A supervised
    // target requires execution_prompt_length > 0 and
    // execution_prompt_end_pos == execution_prompt_length - 1.
    int32_t execution_prompt_end_pos = -1;
    int32_t execution_prompt_length = 0;

    // Runtime binding projection: per-token slot assignment
    // token_exec_slot_indices[pos] >= 0 means state-bearing, -1 means non-state-bearing.
    // These are temporary dense addresses, never semantic identities.
    std::vector<int32_t> token_exec_slot_indices;

    // Compiled bootstrap provenance
    std::vector<CompiledBootstrapBinding> compiled_bootstrap_bindings;

    // Explicit episode-local identity -> dense runtime address lowering
    std::vector<CompiledSlotBinding> compiled_slot_bindings;

    // Execution supervision projection
    std::vector<TeacherStep> teacher_steps;
};

// =============================================================================
// Utilities: Resolve the explicit semantic/runtime slot boundary
//
// D_row = { b.slot_id | b ∈ compiled_slot_bindings }
//                   ∪ { step.arg1_slot, step.arg2_slot, step.write_slot | step ∈ teacher_steps }
//
// Configured slot ranges [S, V) remain outer bounds only.
// =============================================================================
inline std::optional<SlotIndex> findSlotIndex(
    const std::vector<CompiledSlotBinding>& bindings,
    SlotId id)
{
    if (!id.valid()) {
        return std::nullopt;
    }
    for (const auto& binding : bindings) {
        if (binding.slot_id == id) {
            return binding.slot_index;
        }
    }
    return std::nullopt;
}

inline std::optional<SlotId> findSlotId(
    const std::vector<CompiledSlotBinding>& bindings,
    SlotIndex index)
{
    if (!index.valid()) {
        return std::nullopt;
    }
    for (const auto& binding : bindings) {
        if (binding.slot_index == index) {
            return binding.slot_id;
        }
    }
    return std::nullopt;
}

inline SlotIndex requireSlotIndex(
    const std::vector<CompiledSlotBinding>& bindings,
    SlotId id,
    const std::string& context)
{
    const auto resolved = findSlotIndex(bindings, id);
    if (!resolved.has_value()) {
        throw std::runtime_error(
            context + ": semantic slot " + describeSlotId(id)
            + " has no compiled runtime binding");
    }
    return *resolved;
}

inline SlotId requireSlotId(
    const std::vector<CompiledSlotBinding>& bindings,
    SlotIndex index,
    const std::string& context)
{
    const auto resolved = findSlotId(bindings, index);
    if (!resolved.has_value()) {
        throw std::runtime_error(
            context + ": runtime slot index " + describeSlotIndex(index)
            + " has no semantic identity binding");
    }
    return *resolved;
}

inline std::vector<SlotId> reconstructSlotDomain(
    const std::vector<CompiledSlotBinding>& bindings,
    const std::vector<CompiledBootstrapBinding>& bootstrap_bindings,
    const std::vector<TeacherStep>& teacher_steps)
{
    std::vector<SlotId> domain;
    domain.reserve(bootstrap_bindings.size() + teacher_steps.size() * 3);

    for (const auto& binding : bootstrap_bindings) {
        domain.push_back(requireSlotId(
            bindings,
            binding.slot_index,
            "reconstructSlotDomain bootstrap"));
    }
    for (const auto& step : teacher_steps) {
        domain.push_back(step.arg1_slot);
        domain.push_back(step.arg2_slot);
        domain.push_back(step.write_slot);
    }

    std::sort(domain.begin(), domain.end());
    domain.erase(std::unique(domain.begin(), domain.end()), domain.end());

    return domain;
}

}  // namespace Execution
}  // namespace GRIM
