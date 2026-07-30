//======================================================//
//  ExecutionPayloadValidation.cu
//  Implementation of the single shared execution
//  payload validator (Workstream 4).
//
//  Rule 20: Every violation throws with detailed context.
//  No fallbacks, no silent defaults, no partial validation.
//======================================================//

#include "ExecutionPayloadValidation.hpp"
#include "../Batching/BatchPayload.hpp"
#include "ExecutionMetadata.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace GRIM {
namespace Execution {

void validateExecutionPayload(
    const Batching::BatchPayload& payload,
    const char* caller,
    int num_slots,
    int num_ops,
    int num_steps)
{
    const int B = payload.batch_size;
    const int S = payload.max_seq_len;
    const std::string tag(caller);

    // ═════════════════════════════════════════════════════════════════════════
    // BATCH-LEVEL: dimension arrays must match batch_size when present
    // ═════════════════════════════════════════════════════════════════════════
    if (!payload.execution_active.empty() &&
        static_cast<int>(payload.execution_active.size()) != B) {
        throw std::runtime_error(
            tag + ": execution_active.size()=" +
            std::to_string(payload.execution_active.size()) +
            " != batch_size=" + std::to_string(B));
    }
    if (!payload.execution_gate_targets.empty() &&
        static_cast<int>(payload.execution_gate_targets.size()) != B) {
        throw std::runtime_error(
            tag + ": execution_gate_targets.size() must equal batch_size");
    }
    if (!payload.execution_prompt_end_positions.empty() &&
        static_cast<int>(payload.execution_prompt_end_positions.size()) != B) {
        throw std::runtime_error(
            tag + ": execution_prompt_end_positions.size() must equal batch_size");
    }
    if (!payload.execution_prompt_lengths.empty() &&
        static_cast<int>(payload.execution_prompt_lengths.size()) != B) {
        throw std::runtime_error(
            tag + ": execution_prompt_lengths.size() must equal batch_size");
    }
    if (!payload.compiled_slot_bindings.empty() &&
        static_cast<int>(payload.compiled_slot_bindings.size()) != B) {
        throw std::runtime_error(
            tag + ": compiled_slot_bindings.size()=" +
            std::to_string(payload.compiled_slot_bindings.size()) +
            " != batch_size=" + std::to_string(B));
    }
    if (!payload.compiled_transition_bindings.empty() &&
        static_cast<int>(payload.compiled_transition_bindings.size()) != B) {
        throw std::runtime_error(
            tag + ": compiled_transition_bindings.size()=" +
            std::to_string(payload.compiled_transition_bindings.size()) +
            " != batch_size=" + std::to_string(B));
    }
    if (!payload.compiled_bootstrap_bindings.empty() &&
        static_cast<int>(payload.compiled_bootstrap_bindings.size()) != B) {
        throw std::runtime_error(
            tag + ": compiled_bootstrap_bindings.size()=" +
            std::to_string(payload.compiled_bootstrap_bindings.size()) +
            " != batch_size=" + std::to_string(B));
    }
    if (!payload.transition_targets.empty() &&
        static_cast<int>(payload.transition_targets.size()) != B) {
        throw std::runtime_error(
            tag + ": transition_targets.size()=" +
            std::to_string(payload.transition_targets.size()) +
            " != batch_size=" + std::to_string(B));
    }

    // If none of the execution arrays are populated, nothing to validate.
    if (payload.execution_active.empty()) {
        return;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PER-ROW VALIDATION
    // ═════════════════════════════════════════════════════════════════════════
    for (int b = 0; b < B; ++b) {
        const bool active = payload.execution_active[b];
        const size_t row_offset = static_cast<size_t>(b) * S;
        const int seq_len = payload.seq_lengths[b];

        // Helper lambdas for row-scoped error messages
        auto row_tag = [&](const std::string& detail) {
            return tag + ": row " + std::to_string(b) + " " + detail;
        };

        const auto gate_target = payload.execution_gate_targets.empty()
            ? ExecutionGateTarget::UNSUPERVISED
            : payload.execution_gate_targets[b];
        if (!isValidExecutionGateTarget(gate_target)) {
            throw std::runtime_error(row_tag("has invalid execution gate target"));
        }
        const int prompt_length = payload.execution_prompt_lengths.empty()
            ? 0 : payload.execution_prompt_lengths[b];
        const int prompt_end_pos = payload.execution_prompt_end_positions.empty()
            ? -1 : payload.execution_prompt_end_positions[b];
        if (gate_target != ExecutionGateTarget::UNSUPERVISED || active) {
            if (prompt_length <= 0 || prompt_length > payload.seq_lengths[b]) {
                throw std::runtime_error(row_tag(
                    "has invalid execution_prompt_length=" +
                    std::to_string(prompt_length)));
            }
            if (prompt_end_pos != prompt_length - 1) {
                throw std::runtime_error(row_tag(
                    "execution_prompt_end_pos must equal execution_prompt_length - 1"));
            }
        }
        if (active && gate_target != ExecutionGateTarget::EXECUTE) {
            throw std::runtime_error(row_tag(
                "execution_active=true requires EXECUTE gate target"));
        }
        if (!active && gate_target == ExecutionGateTarget::EXECUTE) {
            throw std::runtime_error(row_tag(
                "EXECUTE gate target requires execution_active=true"));
        }
        // Access per-row data (safe — sizes checked above)
        const auto& cbb = payload.compiled_bootstrap_bindings.empty()
            ? std::vector<CompiledBootstrapBinding>{}
            : payload.compiled_bootstrap_bindings[b];
        const auto& csb = payload.compiled_slot_bindings.empty()
            ? std::vector<CompiledSlotBinding>{}
            : payload.compiled_slot_bindings[b];
        const auto& ctb = payload.compiled_transition_bindings.empty()
            ? std::vector<CompiledTransitionBinding>{}
            : payload.compiled_transition_bindings[b];
        const auto& targets = payload.transition_targets.empty()
            ? std::vector<TransitionInvocation>{}
            : payload.transition_targets[b];

        if (!active) {
            // ─────────────────────────────────────────────────────────────────
            // NON-EXECUTION ROW
            // ─────────────────────────────────────────────────────────────────

            // transition_targets must be empty
            if (!targets.empty()) {
                throw std::runtime_error(row_tag(
                    "has execution_active=false but non-empty transition_targets ("
                    + std::to_string(targets.size()) + " steps) — "
                    "transition_targets alone do not activate execution"));
            }

            // compiled_bootstrap_bindings must be empty
            if (!cbb.empty()) {
                throw std::runtime_error(row_tag(
                    "has execution_active=false but non-empty compiled_bootstrap_bindings ("
                    + std::to_string(cbb.size()) + " bindings)"));
            }
            if (!csb.empty()) {
                throw std::runtime_error(row_tag(
                    "has execution_active=false but non-empty compiled_slot_bindings ("
                    + std::to_string(csb.size()) + " bindings)"));
            }
            if (!ctb.empty()) {
                throw std::runtime_error(row_tag(
                    "has execution_active=false but non-empty "
                    "compiled_transition_bindings (" +
                    std::to_string(ctb.size()) + " bindings)"));
            }

            // all token_exec_slot_indices in this row must be -1
            for (int t = 0; t < seq_len; ++t) {
                const int32_t slot = payload.token_to_slot_index_map[row_offset + t];
                if (slot != -1) {
                    throw std::runtime_error(row_tag(
                        "has execution_active=false but token_to_slot_index_map[" +
                        std::to_string(t) + "]=" + std::to_string(slot) +
                        " (must be -1 for non-execution rows)"));
                }
            }

        } else {
            // ─────────────────────────────────────────────────────────────────
            // EXECUTION-ACTIVE ROW
            // ─────────────────────────────────────────────────────────────────

            // compiled_bootstrap_bindings must be non-empty
            if (cbb.empty()) {
                throw std::runtime_error(row_tag(
                    "has execution_active=true but empty compiled_bootstrap_bindings"));
            }
            if (static_cast<int>(csb.size()) != num_slots) {
                throw std::runtime_error(row_tag(
                    "compiled_slot_bindings.size()=" + std::to_string(csb.size()) +
                    " != execution_block_num_slots=" + std::to_string(num_slots)));
            }
            if (static_cast<int>(ctb.size()) != num_ops) {
                throw std::runtime_error(row_tag(
                    "compiled_transition_bindings.size()=" +
                    std::to_string(ctb.size()) +
                    " != execution_block_num_ops=" + std::to_string(num_ops)));
            }

            // transition_targets must have exactly num_steps entries
            if (static_cast<int>(targets.size()) != num_steps) {
                throw std::runtime_error(row_tag(
                    "transition_targets.size()=" + std::to_string(targets.size()) +
                    " != execution_block_num_steps=" + std::to_string(num_steps)));
            }

            // ── Bootstrap binding injectivity and range checks ──

            // Semantic identities and dense runtime addresses form a complete,
            // row-lifetime bijection. No identity is derived from an index.
            std::unordered_set<std::uint64_t> semantic_ids;
            std::unordered_set<int32_t> runtime_indices;
            for (size_t i = 0; i < csb.size(); ++i) {
                const auto& binding = csb[i];
                if (!binding.slot_id.valid()) {
                    throw std::runtime_error(row_tag(
                        "compiled_slot_bindings[" + std::to_string(i) +
                        "] has invalid semantic SlotId"));
                }
                if (!binding.slot_index.valid() ||
                    binding.slot_index.dense() >= num_slots) {
                    throw std::runtime_error(row_tag(
                        "compiled_slot_bindings[" + std::to_string(i) +
                        "].slot_index=" + describeSlotIndex(binding.slot_index) +
                        " out of range [0, " + std::to_string(num_slots) + ")"));
                }
                if (!semantic_ids.insert(binding.slot_id.serialized()).second) {
                    throw std::runtime_error(row_tag(
                        "compiled_slot_bindings has duplicate semantic SlotId=" +
                        describeSlotId(binding.slot_id)));
                }
                if (!runtime_indices.insert(binding.slot_index.dense()).second) {
                    throw std::runtime_error(row_tag(
                        "compiled_slot_bindings has duplicate runtime SlotIndex=" +
                        describeSlotIndex(binding.slot_index)));
                }
            }

            std::unordered_set<std::uint64_t> transition_ids;
            std::unordered_set<int32_t> transition_indices;
            for (size_t i = 0; i < ctb.size(); ++i) {
                const auto& binding = ctb[i];
                if (!binding.transition_id.valid()) {
                    throw std::runtime_error(row_tag(
                        "compiled_transition_bindings[" + std::to_string(i) +
                        "] has invalid semantic TransitionId"));
                }
                if (!binding.transition_index.valid() ||
                    binding.transition_index.dense() >= num_ops) {
                    throw std::runtime_error(row_tag(
                        "compiled_transition_bindings[" + std::to_string(i) +
                        "].transition_index=" +
                        describeTransitionIndex(binding.transition_index) +
                        " out of range [0, " + std::to_string(num_ops) + ")"));
                }
                if (!transition_ids.insert(
                        binding.transition_id.serialized()).second) {
                    throw std::runtime_error(row_tag(
                        "compiled_transition_bindings has duplicate "
                        "TransitionId=" +
                        describeTransitionId(binding.transition_id)));
                }
                if (!transition_indices.insert(
                        binding.transition_index.dense()).second) {
                    throw std::runtime_error(row_tag(
                        "compiled_transition_bindings has duplicate "
                        "TransitionIndex=" +
                        describeTransitionIndex(binding.transition_index)));
                }
            }

            std::unordered_set<int32_t> bound_positions;
            std::unordered_set<int32_t> bound_slot_indices;

            for (size_t i = 0; i < cbb.size(); ++i) {
                const auto& binding = cbb[i];

                if (binding.binding_id != static_cast<int32_t>(i)) {
                    throw std::runtime_error(row_tag(
                        "compiled_bootstrap_bindings[" + std::to_string(i) +
                        "].binding_id=" + std::to_string(binding.binding_id) +
                        " must equal its row-local ordinal"));
                }

                // token_pos in valid range for this row
                if (binding.token_pos < 0 || binding.token_pos >= seq_len) {
                    throw std::runtime_error(row_tag(
                        "compiled_bootstrap_bindings[" + std::to_string(i) +
                        "].token_pos=" + std::to_string(binding.token_pos) +
                        " out of range [0, " + std::to_string(seq_len) + ")"));
                }

                // Bootstrap provenance may bind only numeric atom tokens.
                // Non-bootstrap numeric atoms remain deliberately unbound;
                // conversely, a compiled binding targeting ordinary text is
                // malformed metadata and must fail before device execution.
                if (payload.atom_mask[row_offset + binding.token_pos] == 0) {
                    throw std::runtime_error(row_tag(
                        "compiled_bootstrap_bindings[" + std::to_string(i) +
                        "].token_pos=" + std::to_string(binding.token_pos) +
                        " does not identify a numeric atom token"));
                }

                if (!binding.slot_index.valid() ||
                    binding.slot_index.dense() >= num_slots) {
                    throw std::runtime_error(row_tag(
                        "compiled_bootstrap_bindings[" + std::to_string(i) +
                        "].slot_index=" + describeSlotIndex(binding.slot_index) +
                        " out of range [0, " + std::to_string(num_slots) + ")"));
                }
                (void)requireSlotId(
                    csb,
                    binding.slot_index,
                    row_tag("compiled_bootstrap_bindings resolution"));

                // Injective in token_pos
                if (!bound_positions.insert(binding.token_pos).second) {
                    throw std::runtime_error(row_tag(
                        "compiled_bootstrap_bindings has duplicate token_pos=" +
                        std::to_string(binding.token_pos)));
                }

                // Injective in compiled runtime address
                if (!bound_slot_indices.insert(binding.slot_index.dense()).second) {
                    throw std::runtime_error(row_tag(
                        "compiled_bootstrap_bindings has duplicate slot_index=" +
                        describeSlotIndex(binding.slot_index)));
                }
            }

            // ── Transition target resolution and current executor arity ──

            for (int k = 0; k < static_cast<int>(targets.size()); ++k) {
                const auto& invocation = targets[k];
                (void)requireTransitionIndex(
                    ctb,
                    invocation.transition_id,
                    row_tag(
                        "transition_targets[" + std::to_string(k) +
                        "].transition_id"));
                if (invocation.arguments.size() != 2) {
                    throw std::runtime_error(row_tag(
                        "transition_targets[" + std::to_string(k) +
                        "].arguments.size()=" +
                        std::to_string(invocation.arguments.size()) +
                        " but the current executor requires 2"));
                }
                if (invocation.results.size() != 1) {
                    throw std::runtime_error(row_tag(
                        "transition_targets[" + std::to_string(k) +
                        "].results.size()=" +
                        std::to_string(invocation.results.size()) +
                        " but the current executor requires 1"));
                }
                for (std::size_t argument = 0;
                     argument < invocation.arguments.size();
                     ++argument) {
                    (void)requireSlotIndex(
                        csb,
                        invocation.arguments[argument],
                        row_tag(
                            "transition_targets[" + std::to_string(k) +
                            "].arguments[" + std::to_string(argument) + "]"));
                }
                for (std::size_t result = 0;
                     result < invocation.results.size();
                     ++result) {
                    (void)requireSlotIndex(
                        csb,
                        invocation.results[result],
                        row_tag(
                            "transition_targets[" + std::to_string(k) +
                            "].results[" + std::to_string(result) + "]"));
                }
            }

            // ── Reconstruct D_row from compiled_bootstrap_bindings ∪ transition_targets ──

            std::vector<SlotId> d_row = reconstructSlotDomain(csb, cbb, targets);

            // Every semantic domain member must lower through the row binding table.
            for (SlotId slot : d_row) {
                if (!findSlotIndex(csb, slot).has_value()) {
                    throw std::runtime_error(row_tag(
                        "reconstructed D_row contains slot_id=" +
                        describeSlotId(slot) + " without a compiled binding"));
                }
            }

            // ── token_exec_slot_indices ↔ compiled_bootstrap_bindings consistency ──
            // R = { pos | token_exec_slot_indices[pos] != -1 } must match bootstrap bindings exactly.
            // For every binding (token_pos, slot_index):
            // token_exec_slot_indices[token_pos] == slot_index.
            // For every pos NOT in a binding: token_exec_slot_indices[pos] == -1.

            // Build expected slot map from bootstrap bindings
            std::vector<int32_t> expected_slots(seq_len, -1);
            for (const auto& binding : cbb) {
                expected_slots[binding.token_pos] = binding.slot_index.dense();
            }

            for (int t = 0; t < seq_len; ++t) {
                const int32_t actual = payload.token_to_slot_index_map[row_offset + t];
                const int32_t expected = expected_slots[t];
                if (actual != expected) {
                    throw std::runtime_error(row_tag(
                        "token_to_slot_index_map[" + std::to_string(t) + "]=" +
                        std::to_string(actual) + " but expected " +
                        std::to_string(expected) +
                        " from compiled_bootstrap_bindings"));
                }
            }

            // At least one slot-bearing token exists
            if (bound_positions.empty()) {
                throw std::runtime_error(row_tag(
                    "execution_active=true but zero slot-bearing token positions"));
            }
        }
    }
}

}  // namespace Execution
}  // namespace GRIM
