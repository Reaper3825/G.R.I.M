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
    if (!payload.compiled_bootstrap_bindings.empty() &&
        static_cast<int>(payload.compiled_bootstrap_bindings.size()) != B) {
        throw std::runtime_error(
            tag + ": compiled_bootstrap_bindings.size()=" +
            std::to_string(payload.compiled_bootstrap_bindings.size()) +
            " != batch_size=" + std::to_string(B));
    }
    if (!payload.teacher_steps.empty() &&
        static_cast<int>(payload.teacher_steps.size()) != B) {
        throw std::runtime_error(
            tag + ": teacher_steps.size()=" +
            std::to_string(payload.teacher_steps.size()) +
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
        const auto& ts = payload.teacher_steps.empty()
            ? std::vector<TeacherStep>{}
            : payload.teacher_steps[b];

        if (!active) {
            // ─────────────────────────────────────────────────────────────────
            // NON-EXECUTION ROW
            // ─────────────────────────────────────────────────────────────────

            // teacher_steps must be empty
            if (!ts.empty()) {
                throw std::runtime_error(row_tag(
                    "has execution_active=false but non-empty teacher_steps ("
                    + std::to_string(ts.size()) + " steps) — "
                    "teacher_steps alone do not activate execution"));
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

            // teacher_steps must have exactly num_steps entries
            if (static_cast<int>(ts.size()) != num_steps) {
                throw std::runtime_error(row_tag(
                    "teacher_steps.size()=" + std::to_string(ts.size()) +
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

            // ── Teacher step range checks ──

            for (int k = 0; k < static_cast<int>(ts.size()); ++k) {
                const auto& step = ts[k];

                if (step.op_id < 0 || step.op_id >= num_ops) {
                    throw std::runtime_error(row_tag(
                        "teacher_steps[" + std::to_string(k) + "].op_id=" +
                        std::to_string(step.op_id) + " out of range [0, " +
                        std::to_string(num_ops) + ")"));
                }
                (void)requireSlotIndex(
                    csb, step.arg1_slot,
                    row_tag("teacher_steps[" + std::to_string(k) + "].arg1_slot"));
                (void)requireSlotIndex(
                    csb, step.arg2_slot,
                    row_tag("teacher_steps[" + std::to_string(k) + "].arg2_slot"));
                (void)requireSlotIndex(
                    csb, step.write_slot,
                    row_tag("teacher_steps[" + std::to_string(k) + "].write_slot"));
            }

            // ── Reconstruct D_row from compiled_bootstrap_bindings ∪ teacher_steps ──

            std::vector<SlotId> d_row = reconstructSlotDomain(csb, cbb, ts);

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
