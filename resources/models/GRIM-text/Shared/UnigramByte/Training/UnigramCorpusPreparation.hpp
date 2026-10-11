// Normalized corpus and exclusion geometry shared by mining and EM.
#pragma once
#include "../Unigram.hpp"

namespace GRIM::Tokenizer {
struct AtomSpanValidationTotals {
    size_t span_count = 0;
    size_t byte_count = 0;
};

struct UnigramPreparedCorpus {
    std::vector<std::string> normalized_texts;
    std::vector<std::vector<AtomSpan>> normalized_exclusion_spans;
    size_t max_segment_length = 0;
};

AtomSpanValidationTotals validateOriginalAtomSpansBeforeNormalization(
    const std::vector<std::string>& texts,
    const std::vector<std::vector<AtomSpan>>& atom_spans);
UnigramPreparedCorpus prepareUnigramTrainingCorpus(
    const std::vector<std::string>& texts,
    const std::vector<std::vector<AtomSpan>>& effective_atom_spans,
    const std::vector<UnigramPiece>& manual_pieces);
} // namespace GRIM::Tokenizer
