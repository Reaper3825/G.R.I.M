#pragma once
#include "GeneratedSequence.hpp"
#include "../UnigramByte/UniByte.hpp"
#include <algorithm>

namespace GRIM::Forward {
// Verify the realized input IDs, rather than comparing rendered and decoded
// text. Normalization and supplied typed boundaries cannot expose prefill text.
inline Tokenizer::DecodeRequest continuationDecodeRequest(
    const GeneratedSequence& sequence, const std::vector<int>& prefill_ids) {
    if (sequence.token_ids.size() < prefill_ids.size() ||
        !std::equal(prefill_ids.begin(), prefill_ids.end(), sequence.token_ids.begin()))
        throw std::runtime_error("Inference generation changed the realized prefill token prefix");
    return Tokenizer::DecodeRequest(sequence.token_ids, sequence.atom_entry_ids,
        sequence.context_atom_table.get(), sequence.token_numeric_values,
        sequence.token_atom_mask, prefill_ids.size());
}
} // namespace GRIM::Forward
