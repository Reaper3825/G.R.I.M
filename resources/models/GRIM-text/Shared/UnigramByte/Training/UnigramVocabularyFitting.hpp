// Initial EM, likelihood pruning with refits, and final EM.
#pragma once
#include "../VocabWriteOp.hpp"

namespace GRIM::Tokenizer {
void fitUnigramVocabulary(UnigramVocabWriteTarget target,
    const std::vector<std::string>& norm_texts,
    const std::vector<std::vector<AtomSpan>>& norm_training_exclusion_spans,
    bool enable_byte_fallback_, int total_piece_target,
    const ::GRIM::HyperParameters::TokenizerHP& hp);
} // namespace GRIM::Tokenizer
