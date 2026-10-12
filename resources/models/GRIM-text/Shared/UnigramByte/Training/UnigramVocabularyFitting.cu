// Initial EM, likelihood pruning with refits, and final EM.

#include "UnigramVocabularyFitting.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"
#include "UnigramEMConvergence.hpp"
#include "UnigramMaximizationStep.hpp"
#include "UnigramLikelihoodPruning.hpp"
#include "../VocabWriteOp.hpp"

#include <iostream>
#include <utility>

namespace GRIM {
namespace Tokenizer {

void fitUnigramVocabulary(UnigramTrainingSession& session) {
    refreshUnigramTrainingSessionVocabulary(session);
    auto& pieces_ = *session.pieces;
    auto& piece_to_id_ = *session.piece_to_id;
    const auto& hp = *session.hp;
    const int total_piece_target = session.total_piece_target;
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
    // SentencePiece-style EM + likelihood-loss prune-to-target controls.
    const int PRUNE_SUB_ITERATIONS = hp.prune_em_iterations;     // EM re-fit iterations after each prune round

    // ---- Phase A: EM to convergence on the full data-selected candidate vocab ----
    std::cout << "[UnigramLM] Phase-A: forward-backward EM on data-selected candidate vocab ("
              << pieces_.size() << " pieces; pruning toward total_piece_target=" << total_piece_target
              << ")" << std::endl;
    runUnigramEMToConvergence(session, "Phase-A");
    requireUnigramConvergedPhaseNotFallbackDominated(session, "Phase-A");

    // ---- Prune rounds: drop lowest-likelihood-contribution pieces toward target ----
    // Each round computes per-piece likelihood loss from the current EM
    // posterior, prunes toward max(target, shrink*current) while protecting
    // single-codepoint / user-defined / no-alternative coverage pieces, then
    // re-fits scores with a few EM sub-iterations.
    int prune_round = 0;
    while (static_cast<int>(pieces_.size()) > total_piece_target) {
        const int before = static_cast<int>(pieces_.size());
        UnigramLikelihoodPruningResult prune = runUnigramLikelihoodPruning(
            session, "UnigramLM::trainFromCorpus likelihood-loss prune");

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
        refreshUnigramTrainingSessionVocabulary(session);
        ++prune_round;
        std::cout << "[UnigramLM] Likelihood-loss prune round " << prune_round << ": "
                  << before << " -> " << pieces_.size() << " pieces (removed " << prune.removed
                  << ", protected=" << prune.protected_count
                  << ", irreplaceable_no_alt=" << prune.irreplaceable_no_alt << ")" << std::endl;

        // Re-fit scores on the smaller vocab before the next loss estimate.
        for (int sub_iter = 0; sub_iter < PRUNE_SUB_ITERATIONS; ++sub_iter) {
            runUnigramEMIteration(session);
        }
    }

    // ---- Final: re-converge EM on the pruned vocab ----
    std::cout << "[UnigramLM] Final: re-converging EM after " << prune_round
              << " prune round(s) (vocab now " << pieces_.size() << " pieces)" << std::endl;
    runUnigramEMToConvergence(session, "Final");
    requireUnigramConvergedPhaseNotFallbackDominated(session, "Final");

}

} // namespace Tokenizer
} // namespace GRIM
