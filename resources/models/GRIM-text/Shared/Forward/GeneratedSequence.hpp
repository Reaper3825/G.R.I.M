//======================================================//
//  GeneratedSequence.hpp
//  Phase2 inference output payload
//======================================================//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace GRIM {
namespace Lenses { struct LensSnapshot; struct LensCaptureResult; }
namespace Tokenizer {
class AtomTable;
class SequenceLocalAtomTable;
}

// Provenance belongs to the generated trajectory, including sampled markers
// and runtime-forced retrieval tokens. Score values cannot establish origin.
enum class TokenOrigin : std::uint8_t {
    SuppliedContext,
    SampledAction,
    RuntimeContext,
    ForcedToken,
};

struct GeneratedSequence {
    std::shared_ptr<const Lenses::LensSnapshot> prefill_lens_snapshot;
    std::shared_ptr<const Lenses::LensCaptureResult> prefill_lens_capture_result;
    std::vector<int> token_ids;
    std::vector<TokenOrigin> token_origins; // Aligned with token_ids.
    // Continuation scores only: log probabilities for sampled tokens and zero
    // placeholders for forced tokens. Use token_origins to select policy actions.
    std::vector<float> token_scores;
    // Aligned with token_scores; absolute offsets into token_ids. Supplied and
    // inserted context have no score entry. Positions are not prediction rows.
    std::vector<std::size_t> score_token_positions;
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> context_atom_table;  // Atom registry from context (null for generated tokens)
    std::vector<uint32_t> atom_entry_ids;  // Per-token atom entry IDs (kAtomEntryNone = no atom)
    std::shared_ptr<GRIM::Tokenizer::SequenceLocalAtomTable>
        context_local_atom_table;
    std::vector<uint32_t> token_local_atom_indices;
    float score = 0.0f;
    bool finished = false;
};

} // namespace GRIM
