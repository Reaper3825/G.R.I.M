#pragma once

#include "../UnigramByte/AtomTable.hpp"
#include "../Execution/ExecutionMetadata.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GRIM::TokenizerArtifacts {

struct GrmtSequence {
    std::vector<int> token_ids;
    std::vector<int> targets;
    std::vector<float> token_numeric_values;
    std::vector<std::uint8_t> token_atom_mask;
    std::vector<std::uint32_t> token_atom_flags;
    std::shared_ptr<GRIM::Tokenizer::AtomTable> atom_table;
    std::vector<std::uint32_t> atom_entry_ids;
    // Compiled dense runtime indices only. Semantic identities are carried by
    // compiled_slot_bindings and never inferred from these tensor addresses.
    std::vector<std::int32_t> token_exec_slot_indices;

    bool execution_active = false;
    GRIM::Execution::ExecutionGateTarget execution_gate_target =
        GRIM::Execution::ExecutionGateTarget::UNSUPERVISED;
    std::int32_t execution_prompt_end_pos = -1;
    std::int32_t execution_prompt_length = 0;
    std::vector<GRIM::Execution::CompiledSlotBinding> compiled_slot_bindings;
    std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings;
    std::vector<GRIM::Execution::TeacherStep> teacher_steps;

    bool hasAnyValidTarget() const;
    void validateForWrite(const std::string& source) const;
};

} // namespace GRIM::TokenizerArtifacts
