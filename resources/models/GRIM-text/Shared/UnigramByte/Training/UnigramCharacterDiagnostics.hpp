// Character coverage diagnostics and the initial learned-vocabulary reset.
#pragma once
#include "../VocabWriteOp.hpp"

namespace GRIM::Tokenizer {
// Diagnostic seeds do not participate in candidate admission or pruning.
void runUnigramCharacterDiagnostics(
    const std::vector<std::string>& norm_texts,
    const std::vector<std::vector<AtomSpan>>& norm_training_exclusion_spans,
    float character_coverage, UnigramVocabWriteTarget target);
} // namespace GRIM::Tokenizer
