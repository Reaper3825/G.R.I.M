//======================================================//
//  GeneratedSequence.hpp
//  Phase2 inference output payload
//======================================================//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace GRIM {
namespace Tokenizer {
class AtomTable;
}

struct ExecutionStepControlTelemetry {
    int step_index = -1;
    int predicted_class = -1;  // 0=CONTINUE, 1=STOP
    float continue_probability = 0.0f;
    float stop_probability = 0.0f;
};

struct ExecutionControlTelemetry {
    bool gate_evaluated = false;
    int gate_predicted_class = -1;  // 0=NOOP, 1=EXECUTE
    float noop_probability = 0.0f;
    float execute_probability = 0.0f;
    bool execution_ran = false;
    bool execution_suppressed_no_bootstrap = false;
    bool stopped_by_model = false;
    bool stopped_at_max_steps = false;
    std::vector<ExecutionStepControlTelemetry> steps;
};

struct GeneratedSequence {
    std::vector<int> token_ids;
    std::vector<float> token_scores;
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;
    /// Per-token execution slot id (-1 = non-state-bearing); mirrors BatchPayload::token_to_slot_map
    std::vector<int32_t> token_to_slot_map;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> context_atom_table;  // Atom registry from context (null for generated tokens)
    std::vector<uint32_t> atom_entry_ids;  // Per-token atom entry IDs (kAtomEntryNone = no atom)
    ExecutionControlTelemetry execution_control;
    float score = 0.0f;
    bool finished = false;
};

} // namespace GRIM
