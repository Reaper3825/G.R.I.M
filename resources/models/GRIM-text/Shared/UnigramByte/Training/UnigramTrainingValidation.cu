// Training preconditions and learned-posterior guards.

#include "UnigramTrainingValidation.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

void validateTrainFromCorpusParameters(const std::vector<std::string>& texts,
                                              const ::GRIM::HyperParameters::TokenizerHP& hp) {
    const auto target_vocab_size = hp.target_vocab_size;
    const auto character_coverage = hp.character_coverage;
    const auto min_subword_freq = hp.min_subword_freq;
    const auto subword_mining_workers = hp.subword_mining_workers;
    validateTokenizerTrainingHP(hp, "UnigramLM::trainFromCorpus");
    if (texts.empty()) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: texts is empty - caller MUST provide at least one training text");
    }
    if (target_vocab_size <= 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: target_vocab_size must be > 0, got " +
                                 std::to_string(target_vocab_size));
    }
    if (!std::isfinite(static_cast<double>(character_coverage)) ||
        character_coverage <= 0.0f || character_coverage > 1.0f) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: character_coverage must be finite and in (0, 1], got " +
                                 std::to_string(character_coverage));
    }
    if (min_subword_freq <= 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: min_subword_freq must be > 0, got " +
                                 std::to_string(min_subword_freq));
    }
    if (subword_mining_workers < 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: subword_mining_workers must be >= 0, got " +
                                 std::to_string(subword_mining_workers));
    }
}

void requireUnigramFinalCleanupLeavesLearnedPiece(size_t piece_count,
                                                  size_t dead_count,
                                                  const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("requireUnigramFinalCleanupLeavesLearnedPiece: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (piece_count == 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned vocab is already empty before final cleanup; caller MUST preserve at least one learned piece");
    }
    if (dead_count > piece_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": dead token count exceeds learned piece count, dead_count=" +
                                 std::to_string(dead_count) + ", piece_count=" + std::to_string(piece_count));
    }
    if (dead_count == piece_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": final dead-token cleanup would delete every learned piece (piece_count=" +
                                 std::to_string(piece_count) +
                                 "); byte fallback dominated all posterior mass, so training MUST fail before Phase-D lattice construction");
    }
}

void requireUnigramAcceptedCandidateSetIsScorable(size_t accepted_count,
                                                  double total_accepted_count,
                                                  const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("requireUnigramAcceptedCandidateSetIsScorable: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (accepted_count == 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": candidate admission produced zero accepted learned pieces before initial score normalization; frequency/validity/dedup filters eliminated every candidate");
    }
    if (!std::isfinite(total_accepted_count) || total_accepted_count <= 0.0) {
        throw std::runtime_error(std::string(caller) +
                                 ": accepted candidate set is not scorable; accepted_count=" +
                                 std::to_string(accepted_count) +
                                 ", total_accepted_count=" +
                                 std::to_string(total_accepted_count) +
                                 ". Learned candidate counts MUST sum to a positive finite value before initial score normalization");
    }
}

void requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
    double expected_learned_piece_tokens,
    double expected_fixed_penalty_byte_fallback_tokens,
    double minimum_learned_to_fallback_ratio,
    const char* phase_label,
    const char* caller) {
    if (!std::isfinite(minimum_learned_to_fallback_ratio) || minimum_learned_to_fallback_ratio <= 0) {
        throw std::runtime_error("minimum_learned_to_fallback_ratio must be positive and finite");
    }

    if (phase_label == nullptr || phase_label[0] == '\0') {
        throw std::runtime_error("requireUnigramLearnedPosteriorMassNotByteFallbackDominated: phase label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("requireUnigramLearnedPosteriorMassNotByteFallbackDominated: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!std::isfinite(expected_learned_piece_tokens) || expected_learned_piece_tokens < 0.0) {
        throw std::runtime_error(std::string(caller) + ": " + phase_label +
                                 " produced invalid expected_learned_piece_tokens=" +
                                 std::to_string(expected_learned_piece_tokens));
    }
    if (!std::isfinite(expected_fixed_penalty_byte_fallback_tokens) ||
        expected_fixed_penalty_byte_fallback_tokens < 0.0) {
        throw std::runtime_error(std::string(caller) + ": " + phase_label +
                                 " produced invalid expected_fixed_penalty_byte_fallback_tokens=" +
                                 std::to_string(expected_fixed_penalty_byte_fallback_tokens));
    }
    if (expected_fixed_penalty_byte_fallback_tokens == 0.0) {
        return;
    }

    const double learned_to_fallback_ratio =
        expected_learned_piece_tokens / expected_fixed_penalty_byte_fallback_tokens;
    if (learned_to_fallback_ratio < minimum_learned_to_fallback_ratio) {
        throw std::runtime_error(std::string(caller) + ": " + phase_label +
                                 " byte fallback dominated posterior mass after EM convergence; "
                                 "expected_learned_piece_tokens=" +
                                 std::to_string(expected_learned_piece_tokens) +
                                 ", expected_fixed_penalty_byte_fallback_tokens=" +
                                 std::to_string(expected_fixed_penalty_byte_fallback_tokens) +
                                 ", learned_to_fallback_ratio=" +
                                 std::to_string(learned_to_fallback_ratio) +
                                 ", required_ratio>=" + std::to_string(minimum_learned_to_fallback_ratio) + ". Training MUST fail instead of treating fallback-dominated evidence as a learned-piece distribution");
    }
}

} // namespace Tokenizer
} // namespace GRIM
