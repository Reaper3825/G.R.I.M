//======================================================//
//  ExecutionResultEmission.hpp
//  Strict host-side contract for exposing one completed execution result.
//======================================================//

#pragma once

#include "../Forward/GeneratedSequence.hpp"
#include "../UnigramByte/AtomTable.hpp"

#include <cmath>
#include <stdexcept>

namespace GRIM::Execution {

struct ExecutionResultEmission {
    bool available = false;
    SlotId slot;
    float value = 0.0f;
    Tokenizer::AtomType atom_type = Tokenizer::AtomType::ATOM_INT;
};

// A result becomes emit-able only when the learned stop controller explicitly
// terminates after a completed step. Reaching max steps is not an implicit
// result choice, and no first/last-valid-slot or recency fallback is permitted.
inline ExecutionResultEmission resolveTerminalExecutionResult(
    bool execution_ran,
    bool stopped_by_model,
    bool stopped_at_max_steps,
    const std::vector<ExecutionStepControlTelemetry>& steps,
    const std::vector<CompiledSlotBinding>& slot_bindings,
    const std::vector<float>& final_slot_values,
    const std::vector<uint8_t>& final_slot_valid)
{
    ExecutionResultEmission result;
    if (!execution_ran || !stopped_by_model) {
        return result;
    }
    if (stopped_at_max_steps) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: conflicting stop states");
    }
    if (steps.empty()) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: model-stopped execution has no completed step");
    }
    if (final_slot_values.size() != final_slot_valid.size()) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: final slot value/validity length mismatch");
    }

    const auto& terminal_step = steps.back();
    if (terminal_step.predicted_class != 1) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: terminal step was not classified STOP");
    }
    if (!terminal_step.invocation.transition_id.valid()) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: terminal step has no transition identity");
    }
    if (terminal_step.invocation.results.size() != 1) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: terminal transition must expose "
            "exactly one result for scalar emission");
    }
    const SlotId slot = terminal_step.invocation.results.front();
    const int slot_index = requireSlotIndex(
        slot_bindings,
        slot,
        "resolveTerminalExecutionResult terminal result").dense();
    if (static_cast<size_t>(slot_index) >= final_slot_values.size()) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: terminal result slot is out of range");
    }
    if (final_slot_valid[static_cast<size_t>(slot_index)] == 0) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: terminal result slot is not valid");
    }

    const float value = final_slot_values[static_cast<size_t>(slot_index)];
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "resolveTerminalExecutionResult: terminal result is not finite");
    }

    result.available = true;
    result.slot = slot;
    result.value = value;
    result.atom_type = Tokenizer::numericAtomTypeForValue(value);
    return result;
}

} // namespace GRIM::Execution
