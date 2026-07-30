//======================================================//
//  ExecutionPayloadValidation.hpp
//  Single shared execution payload validator.
//
//  One function validates row-level and batch-level
//  execution metadata invariants. Called from
//  buildBatchPayload() and Phase2 processBatch() — no other call site is
//  permitted to perform semantic execution validation.
//
//  Rule 20: crash with detailed error on ANY violation.
//======================================================//

#pragma once

namespace GRIM {
namespace Batching { struct BatchPayload; }

namespace Execution {

// Validate all execution payload invariants for the given batch.
//
// Enforces:
//   Row-level (per batch row b):
//     - Non-execution rows: execution_active==false, transition_targets empty,
//       compiled_bootstrap_bindings empty, all slots==-1
//     - Execution-active rows: execution_active==true, bootstrap bindings
//       non-empty and injective in token_pos and SlotIndex, a complete
//       SlotId/SlotIndex and TransitionId/TransitionIndex bijections,
//       transition invocations whose slot identities resolve
//       targets, and token_exec_slot_indices consistent with bootstrap bindings
//
//   Batch-level:
//     - Dimension arrays match batch_size
//     - No "half execution-active" rows
//     - R = { pos | token_exec_slot_indices[pos] != -1 } matches bootstrap bindings exactly
//     - D_row reconstructed from bootstrap bindings and transition invocations
//     - token_exec_slot_indices and compiled_bootstrap_bindings are mutually consistent
//
// @param payload  The fully-built BatchPayload to validate
// @param caller   Identifying string for error messages (e.g. "buildBatchPayload")
// @param num_slots  Configured execution_block_num_slots (upper bound for slot IDs)
// @param num_ops    Current executor transition-class count (TransitionIndex bound)
// @param num_steps  Configured execution_block_num_steps (expected transition target count per active row)
void validateExecutionPayload(
    const Batching::BatchPayload& payload,
    const char* caller,
    int num_slots,
    int num_ops,
    int num_steps);

}  // namespace Execution
}  // namespace GRIM
