#pragma once

#include "Unigram.hpp"

#include <string>
#include <vector>

namespace GRIM::Tokenizer {

// Read authored exact pieces for a fresh vocabulary training pass.
// An empty path supplies no additional pieces.
std::vector<UnigramPiece> loadManualVocabPieces(const std::string& path);

// Reserve the authored entries in the learned token-ID range. They are emitted
// by the exact pre-Viterbi matcher, excluded from EM, and protected from pruning.
// Subsequent mined admission skips exact duplicate text.
void appendManualVocabPieces(UnigramLM& unigram,
                             const std::vector<UnigramPiece>& pieces,
                             int target_vocab_size);

} // namespace GRIM::Tokenizer
