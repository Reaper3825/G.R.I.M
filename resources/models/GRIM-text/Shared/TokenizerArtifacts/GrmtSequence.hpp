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
    std::vector<std::int32_t> token_exec_slots;

    bool execution_active = false;
    std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings;
    std::vector<GRIM::Execution::TeacherStep> teacher_steps;
    std::vector<GRIM::Execution::SlotSelectionTarget> slot_selection_targets;

    bool hasAnyValidTarget() const;
    void validateForWrite(const std::string& source) const;
};

} // namespace GRIM::TokenizerArtifacts
