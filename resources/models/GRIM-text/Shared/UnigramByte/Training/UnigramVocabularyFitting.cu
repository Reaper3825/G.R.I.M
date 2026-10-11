// Initial EM, likelihood pruning with refits, and final EM.

#include "UnigramVocabularyFitting.hpp"
#include "UnigramEMConvergence.hpp"
#include "UnigramMaximizationStep.hpp"
#include "UnigramLikelihoodPruning.hpp"

#include <iostream>
#include <utility>

namespace GRIM {
namespace Tokenizer {

void fitUnigramVocabulary(
    UnigramVocabWriteTarget target,
    const std::vector<std::string>& norm_texts,
    const std::vector<std::vector<AtomSpan>>& norm_training_exclusion_spans,
    bool enable_byte_fallback_, int total_piece_target, int subword_mining_workers) {
    auto& pieces_ = target.pieces;
    auto& piece_to_id_ = target.piece_to_id;
    // Step 5: Soft EM + likelihood-loss prune-to-target (SentencePiece-style)
    //
    // Starting from the full data-selected candidate vocab, we:
    //   1. Run true Unigram forward-backward EM to convergence (Phase-A).
    //   2. Iteratively prune the lowest-likelihood-contribution pieces toward
    //      ordinary target + manual allowance, re-fitting with a few EM sub-iterations each
    //      round, always protecting single-codepoint / user-defined / no-alt
    //      coverage pieces (Prune rounds).
    //   3. Re-converge EM on the final vocab (Final).
    //
    // Raw frequency cannot separate a real word from its dead prefixes
    // (they share the same substring count); only the EM likelihood signal
    // can, which is why pruning is posterior/likelihood-driven, not a
    // frequency cap. Single-char coverage is provided by single-char unigram
    // pieces; byte fallback is never a protected vocab entry.
    constexpr double SMOOTHING = kUnigramPieceCountSmoothing;
    // SentencePiece-style EM + likelihood-loss prune-to-target controls.
    constexpr double PRUNE_SHRINK_FACTOR   = 0.75;  // max fraction kept per prune round
    constexpr int    PRUNE_SUB_ITERATIONS  = 2;     // EM re-fit iterations after each prune round

    const UnigramExpectationStepRequest expectation_request{
        pieces_, norm_texts, norm_training_exclusion_spans, enable_byte_fallback_};

    // ---- Phase A: EM to convergence on the full data-selected candidate vocab ----
    std::cout << "[UnigramLM] Phase-A: forward-backward EM on data-selected candidate vocab ("
              << pieces_.size() << " pieces; pruning toward total_piece_target=" << total_piece_target
              << ")" << std::endl;
    UnigramEMConvergenceResult em_result = runUnigramEMToConvergence(pieces_, expectation_request, "Phase-A");
    requireUnigramConvergedPhaseNotFallbackDominated(em_result, "Phase-A");

    // ---- Prune rounds: drop lowest-likelihood-contribution pieces toward target ----
    // Each round computes per-piece likelihood loss from the current EM
    // posterior, prunes toward max(target, shrink*current) while protecting
    // single-codepoint / user-defined / no-alternative coverage pieces, then
    // re-fits scores with a few EM sub-iterations.
    int prune_round = 0;
    while (static_cast<int>(pieces_.size()) > total_piece_target) {
        std::vector<double> freq_by_index(pieces_.size(), 0.0);
        for (const auto& token_count : em_result.learned_token_counts) {
            const int idx = indexForTokenId(token_count.first);
            if (idx >= 0 && idx < static_cast<int>(freq_by_index.size())) {
                freq_by_index[static_cast<size_t>(idx)] = token_count.second;
            }
        }

        const int before = static_cast<int>(pieces_.size());
        UnigramLikelihoodPruningResult prune = runUnigramLikelihoodPruning(
            pieces_,
            freq_by_index,
            total_piece_target,
            PRUNE_SHRINK_FACTOR,
            subword_mining_workers,
            "UnigramLM::trainFromCorpus likelihood-loss prune");

        if (prune.removed <= 0) {
            std::cout << "[UnigramLM] Likelihood-loss prune: no further prunable pieces (protected="
                      << prune.protected_count << ", irreplaceable_no_alt=" << prune.irreplaceable_no_alt
                      << "); stopping at " << before << " pieces, above total_piece_target="
                      << total_piece_target << " (coverage pieces cannot be pruned)" << std::endl;
            break;
        }

        rewriteUnigramVocab(
            UnigramVocabWriteTarget{pieces_, piece_to_id_},
            std::move(prune.survivors),
            "UnigramLM::trainFromCorpus likelihood-loss prune compaction");
        ++prune_round;
        std::cout << "[UnigramLM] Likelihood-loss prune round " << prune_round << ": "
                  << before << " -> " << pieces_.size() << " pieces (removed " << prune.removed
                  << ", protected=" << prune.protected_count
                  << ", irreplaceable_no_alt=" << prune.irreplaceable_no_alt << ")" << std::endl;

        // Re-fit scores on the smaller vocab before the next loss estimate.
        UnigramExpectationStepResult sub_e_step;
        for (int sub_iter = 0; sub_iter < PRUNE_SUB_ITERATIONS; ++sub_iter) {
            sub_e_step = runUnigramExpectationStep(expectation_request);
            runUnigramMaximizationStep(UnigramMaximizationStepRequest{
                pieces_, sub_e_step.learned_token_counts, sub_e_step.expected_learned_piece_tokens, SMOOTHING});
        }
        em_result.learned_token_counts = std::move(sub_e_step.learned_token_counts);
        em_result.expected_learned_piece_tokens = sub_e_step.expected_learned_piece_tokens;
        em_result.expected_fixed_penalty_byte_fallback_tokens =
            sub_e_step.expected_fixed_penalty_byte_fallback_tokens;
    }

    // ---- Final: re-converge EM on the pruned vocab ----
    std::cout << "[UnigramLM] Final: re-converging EM after " << prune_round
              << " prune round(s) (vocab now " << pieces_.size() << " pieces)" << std::endl;
    UnigramEMConvergenceResult final_result = runUnigramEMToConvergence(pieces_, expectation_request, "Final");
    requireUnigramConvergedPhaseNotFallbackDominated(final_result, "Final");

}

} // namespace Tokenizer
} // namespace GRIM
