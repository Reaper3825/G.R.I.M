// Mined candidate ranking, filtering, structural dedup, and initial scores.
#pragma once
#include "SubwordMining.hpp"
#include "../Unigram.hpp"

namespace GRIM::Tokenizer {
// Appends manual pieces and all qualifying mined candidates to the reset vocab.
void admitUnigramVocabularyCandidates(UnigramLM& unigram,
    const UnigramSubwordMiningResult& mining_result, const ::GRIM::HyperParameters::TokenizerHP& hp, const std::vector<UnigramPiece>& manual_pieces);
} // namespace GRIM::Tokenizer
