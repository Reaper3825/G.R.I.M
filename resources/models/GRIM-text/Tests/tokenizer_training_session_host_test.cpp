// Focused host unit tests; no tokenizer trainer target or GPU execution.
#include "HyperParameters/HyperparameterGroupings.hpp"
#include "UnigramByte/Training/UnigramExpectationStep.hpp"
#include "UnigramByte/Training/UnigramForwardBackward.hpp"
#include "UnigramByte/Training/UnigramMaximizationStep.hpp"
#include "UnigramByte/Training/UnigramLikelihoodPruning.hpp"
#include "UnigramByte/Training/UnigramVocabularyFitting.hpp"
#include "UnigramByte/VocabWriteOp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace GRIM::Tokenizer;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected) {
    check(std::abs(actual - expected) <= 1e-10 * std::max(1.0, std::abs(expected)),
          "cached expectation differs from reference forward/backward");
}
template <typename Fn> void requiresFailure(Fn fn) {
    bool failed = false;
    try { fn(); } catch (const std::runtime_error&) { failed = true; }
    check(failed, "expected a session validation failure");
}

int main() try {
    static_assert(std::is_aggregate_v<UnigramTrainingSession>);
    static_assert(std::is_trivially_copyable_v<UnigramTrainingSession>);
    GRIM::HyperParameters::TokenizerHP hp{};
    hp.piece_count_smoothing = 0.1;
    hp.prune_shrink_factor = 0.75;
    hp.prune_em_iterations = 2;
    hp.subword_mining_workers = 2;
    hp.em_max_iterations = 6;
    hp.em_convergence_tolerance = 0.0001;
    hp.minimum_learned_to_fallback_ratio = 0.1;
    hp.em_fail_on_iteration_limit = false;
    std::vector<UnigramPiece> pieces{
        {"a", -1.2f, false}, {"b", -1.3f, false},
        {"ab", -0.8f, false}, {"ba", -1.1f, false},
        {"aba", -0.9f, false}, {"MANUAL", 0.0f, true}};
    std::unordered_map<std::string, int> index;
    UnigramTrainingStorage storage;
    UnigramTrainingSession session;
    session.hp = &hp;
    session.pieces = &pieces;
    session.piece_to_id = &index;
    session.storage = &storage;
    session.total_piece_target = 3;
    session.enable_byte_fallback = true;

    // Leading/trailing exclusions, adjacent spans, empty texts and consecutive
    // newlines must produce the exact same eligible segment boundaries.
    const std::vector<std::string> texts{
        "aba\n\nabXXba\n", "XXabYY", "", "\xc3\xa9", "aXXYYb"};
    const std::vector<std::vector<AtomSpan>> exclusions{
        {{7, 9}}, {{0, 2}, {4, 6}}, {}, {}, {{1, 3}, {3, 5}}};
    prepareUnigramTrainingSessionCorpus(session, texts, exclusions);
    check(session.segment_count == 7, "incorrect segment boundaries");
    requiresFailure([&] { prepareUnigramTrainingSessionCorpus(session, texts, exclusions); });
    refreshUnigramTrainingSessionVocabulary(session);
    auto compareReference = [&] {
        UnigramForwardBackwardLattice reference(pieces, session.enable_byte_fallback, "reference");
        UnigramForwardBackwardStats stats(pieces.size());
        for (const std::string& segment : std::vector<std::string>{
                "aba", "ab", "ba", "ab", "\xc3\xa9", "a", "b"}) {
            reference.accumulateSegment(segment, stats, "reference");
        }
        runUnigramExpectationStep(session);
        close(session.log_likelihood, stats.log_likelihood);
        close(session.expected_learned_piece_tokens, stats.expected_learned_piece_tokens);
        close(session.expected_fixed_penalty_byte_fallback_tokens,
              stats.expected_fixed_penalty_byte_fallback_tokens);
        for (size_t i = 0; i < pieces.size(); ++i) close(session.expected_counts[i], stats.piece_expected_counts[i]);
        close(session.expected_counts[session.piece_count - 1], 0.0); // manual pieces are excluded
    };
    compareReference();
    const auto* match_address = session.matches;
    auto* alpha_address = session.alpha;
    const auto generation = session.vocabulary_generation;
    runUnigramMaximizationStep(session);
    compareReference();
    // External score-only mutation must use current scores without new matches.
    pieces[2].score -= 0.7f;
    compareReference();
    check(session.matches == match_address && session.alpha == alpha_address &&
          session.vocabulary_generation == generation, "EM rebuilt matching or scratch storage");

    session.matches_generation = 0;
    requiresFailure([&] { runUnigramExpectationStep(session); });
    session.matches_generation = generation;
    session.expectation_generation = 0;
    requiresFailure([&] { runUnigramMaximizationStep(session); });
    session.address_space = UnigramTrainingAddressSpace::Device;
    requiresFailure([&] { runUnigramExpectationStep(session); });
    session.address_space = UnigramTrainingAddressSpace::Host;
    storage.alpha.reserve(storage.alpha.capacity() + 100);
    requiresFailure([&] { runUnigramExpectationStep(session); });
    refreshUnigramTrainingSessionVocabulary(session); // rebind all addresses
    alpha_address = session.alpha;
    compareReference();

    auto prune = runUnigramLikelihoodPruning(session, "test prune");
    check(prune.removed > 0, "expected ordinary pieces to be pruned");
    requiresFailure([&] { runUnigramExpectationStep(session); });
    rewriteUnigramVocab({pieces, index}, std::move(prune.survivors), "test rewrite");
    refreshUnigramTrainingSessionVocabulary(session);
    compareReference();
    check(session.vocabulary_generation > generation, "prune did not refresh generation");
    for (const auto& match : storage.matches) {
        check(match.piece_index >= 0 && static_cast<size_t>(match.piece_index) < pieces.size(),
              "pruned match retained an obsolete index");
    }

    // Convergence and pruning refits exercise the same real session operation.
    fitUnigramVocabulary(session);
    check(session.segment_count == 7 && session.alpha == alpha_address,
          "fitting changed corpus geometry or scratch capacity");
    check(index.size() == pieces.size(), "vocabulary rewrite lost text/index consistency");

    // With all remaining pieces protected, pruning must not move their text out.
    pieces = {{"a", -0.5f, false}, {"b", -0.5f, false}, {"MANUAL", 0.0f, true}};
    refreshUnigramTrainingSessionVocabulary(session);
    runUnigramExpectationStep(session);
    session.total_piece_target = 1;
    const auto no_progress = runUnigramLikelihoodPruning(session, "protected prune");
    check(no_progress.removed == 0 && pieces[0].text == "a" && pieces[2].text == "MANUAL",
          "no-progress pruning damaged protected pieces");
    session.enable_byte_fallback = false;
    requiresFailure([&] { runUnigramExpectationStep(session); }); // uncovered UTF-8 bytes
    std::cout << "Tokenizer training session host tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
