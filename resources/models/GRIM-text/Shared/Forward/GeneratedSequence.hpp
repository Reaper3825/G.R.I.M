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

struct GeneratedSequence {
    std::vector<int> token_ids;
    std::vector<float> token_scores;
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;
    /// Per-token execution slot id (-1 = non-state-bearing); mirrors BatchPayload::token_to_slot_map
    std::vector<int32_t> token_to_slot_map;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> context_atom_table;  // Atom registry from context (null for generated tokens)
    std::vector<uint32_t> atom_entry_ids;  // Per-token atom entry IDs (kAtomEntryNone = no atom)
    float score = 0.0f;
    bool finished = false;
};

} // namespace GRIM