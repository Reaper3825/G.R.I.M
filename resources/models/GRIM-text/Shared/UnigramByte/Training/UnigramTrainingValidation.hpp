// Preconditions and posterior guards shared by vocabulary training stages.
#pragma once
#include "../Unigram.hpp"

namespace GRIM::Tokenizer {
void validateTrainFromCorpusParameters(const std::vector<std::string>& texts,
    int target_vocab_size, float character_coverage, int min_subword_freq,
    int subword_mining_workers);
void requireUnigramFinalCleanupLeavesLearnedPiece(size_t piece_count,
    size_t dead_count, const char* caller);
void requireUnigramAcceptedCandidateSetIsScorable(size_t accepted_count,
    double total_accepted_count, const char* caller);
void requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
    double expected_learned_piece_tokens,
    double expected_fixed_penalty_byte_fallback_tokens,
    const char* phase_label, const char* caller);
} // namespace GRIM::Tokenizer
