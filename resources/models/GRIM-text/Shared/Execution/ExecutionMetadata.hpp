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
#include "TransitionIdentity.hpp"

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
// TransitionInvocation — per-step transition target or realized trace
//
// This is the domain-neutral execution projection of the canonical
// StructuredExecutionRecord. It names an executable transition and the
// ordered slot identities supplied to and produced by that transition.
// It deliberately carries neither slot values nor assumptions about what a
// slot contains. AtomTable/runtime state own values, types, objects, and
// internal composition.
//
// Compiled slot/transition bindings and transition_targets are paired
// projections of one canonical record and must never be authored independently.
// =============================================================================
struct TransitionInvocation {
    TransitionId transition_id;
    std::vector<SlotId> arguments;
    std::vector<SlotId> results;
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

// TransitionId remains stable for the semantic episode. TransitionIndex is a
// dense class/dispatcher address scoped to this compiled row payload.
struct CompiledTransitionBinding {
    TransitionId transition_id;
    TransitionIndex transition_index;
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
// tokenization and is the only source of authored execution structure.
// Its slot and transition identities remain opaque references.
//
// Runtime indices and transition targets are derived projections.
// bootstrap_bindings defines the only literals allowed to seed registers.
//
// Rules:
//   - bootstrap_bindings must be non-empty for execution-active rows
//   - each bootstrap binding has unique slot_id (no duplicate initialization)
//   - slot identity is stable for the entire row lifetime
//   - transition results may target non-bootstrapped slots if in D_row
//   - transitions may not invent or rebind slot identity mid-row
// =============================================================================
struct StructuredExecutionRecord {
    bool execution_active = false;
    ExecutionGateTarget execution_gate_target = ExecutionGateTarget::UNSUPERVISED;

    // Bootstrap literal bindings — define which literals seed which registers
    std::vector<BootstrapLiteralBinding> bootstrap_bindings;

    // Ordered transition applications. Transition signatures own arity; slot
    // payload values and composition remain AtomTable/runtime concerns.
    std::vector<TransitionInvocation> transitions;

    // Explicit row-local slot domain D_row:
    // The unique set of slot ids referenced by this row's execution program.
    // Includes bootstrap binding slot ids and every transition argument/result.
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
// Non-empty transition_targets is a supervised-training payload validity
// requirement, NOT the activation source.
//
// Runtime D_row is reconstructed as:
//   D_row = { requireSlotId(b.slot_index) | b in compiled_bootstrap_bindings }
//           union { slot | slot in invocation.arguments or invocation.results }
// It is NOT serialized separately.
//
// token_exec_slot_indices is compiled ONLY from bootstrap_bindings[], not from
// arbitrary numeric tokens in rendered text.
// =============================================================================
struct CompiledStructuredExecutionPayload {
    bool execution_active = false;
    ExecutionGateTarget execution_gate_target = ExecutionGateTarget::UNSUPERVISED;

    // Functional prompt span. In SFT this includes every model-visible token
    // before the answer, including goal/context fields. Positions are
    // row-relative and the start is derived as
    // prompt_end_pos - prompt_length + 1. The gate reads the final token of
    // the complete functional prompt.
    int32_t prompt_end_pos = -1;
    int32_t prompt_length = 0;

    // Runtime binding projection: per-token slot assignment
    // token_exec_slot_indices[pos] >= 0 means state-bearing, -1 means non-state-bearing.
    // These are temporary dense addresses, never semantic identities.
    std::vector<int32_t> token_exec_slot_indices;

    // Compiled bootstrap provenance
    std::vector<CompiledBootstrapBinding> compiled_bootstrap_bindings;

    // Explicit episode-local identity -> dense runtime address lowering
    std::vector<CompiledSlotBinding> compiled_slot_bindings;

    // Explicit semantic transition -> dense model-head/dispatcher lowering
    std::vector<CompiledTransitionBinding> compiled_transition_bindings;

    // Execution supervision projection
    std::vector<TransitionInvocation> transition_targets;
};

// =============================================================================
// Utilities: Resolve the explicit semantic/runtime slot boundary
//
// D_row = { requireSlotId(b.slot_index) | b in compiled_bootstrap_bindings }
//         union { slot | slot in invocation.arguments or invocation.results }
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

inline std::optional<TransitionIndex> findTransitionIndex(
    const std::vector<CompiledTransitionBinding>& bindings,
    TransitionId id)
{
    if (!id.valid()) {
        return std::nullopt;
    }
    for (const auto& binding : bindings) {
        if (binding.transition_id == id) {
            return binding.transition_index;
        }
    }
    return std::nullopt;
}

inline std::optional<TransitionId> findTransitionId(
    const std::vector<CompiledTransitionBinding>& bindings,
    TransitionIndex index)
{
    if (!index.valid()) {
        return std::nullopt;
    }
    for (const auto& binding : bindings) {
        if (binding.transition_index == index) {
            return binding.transition_id;
        }
    }
    return std::nullopt;
}

inline TransitionIndex requireTransitionIndex(
    const std::vector<CompiledTransitionBinding>& bindings,
    TransitionId id,
    const std::string& context)
{
    const auto resolved = findTransitionIndex(bindings, id);
    if (!resolved.has_value()) {
        throw std::runtime_error(
            context + ": semantic transition " + describeTransitionId(id)
            + " has no compiled runtime binding");
    }
    return *resolved;
}

inline TransitionId requireTransitionId(
    const std::vector<CompiledTransitionBinding>& bindings,
    TransitionIndex index,
    const std::string& context)
{
    const auto resolved = findTransitionId(bindings, index);
    if (!resolved.has_value()) {
        throw std::runtime_error(
            context + ": runtime transition index "
            + describeTransitionIndex(index)
            + " has no semantic identity binding");
    }
    return *resolved;
}

inline std::vector<SlotId> reconstructSlotDomain(
    const std::vector<CompiledSlotBinding>& bindings,
    const std::vector<CompiledBootstrapBinding>& bootstrap_bindings,
    const std::vector<TransitionInvocation>& transition_targets)
{
    std::vector<SlotId> domain;
    std::size_t transition_slot_count = 0;
    for (const auto& invocation : transition_targets) {
        transition_slot_count += invocation.arguments.size();
        transition_slot_count += invocation.results.size();
    }
    domain.reserve(bootstrap_bindings.size() + transition_slot_count);

    for (const auto& binding : bootstrap_bindings) {
        domain.push_back(requireSlotId(
            bindings,
            binding.slot_index,
            "reconstructSlotDomain bootstrap"));
    }
    for (const auto& invocation : transition_targets) {
        domain.insert(
            domain.end(),
            invocation.arguments.begin(),
            invocation.arguments.end());
        domain.insert(
            domain.end(),
            invocation.results.begin(),
            invocation.results.end());
    }

    std::sort(domain.begin(), domain.end());
    domain.erase(std::unique(domain.begin(), domain.end()), domain.end());

    return domain;
}

}  // namespace Execution
}  // namespace GRIM
