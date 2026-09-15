#pragma once

#include "Unigram.hpp"

#include <string>
#include <vector>

namespace GRIM::Tokenizer {

// Read authored ordinary pieces for a fresh vocabulary training pass.
// An empty path supplies no additional pieces.
std::vector<UnigramPiece> loadManualVocabPieces(const std::string& path);

// Reserve the authored entries in the initial learned-piece vector. Subsequent
// mined candidate admission uses UnigramLM::hasPiece() to skip exact duplicates.
void appendManualVocabPieces(UnigramLM& unigram,
                             const std::vector<UnigramPiece>& pieces,
                             int target_vocab_size);

} // namespace GRIM::Tokenizer
