// Learned-piece normalization; fixed byte fallback is telemetry only.
#include "UnigramMaximizationStep.hpp"
#include "UnigramExpectationStep.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace GRIM::Tokenizer {
int runUnigramMaximizationStep(UnigramTrainingSession& session) {
    requireUnigramTrainingSessionReady(session, "Unigram maximization");
    if (session.expectation_generation != session.vocabulary_generation) {
        throw std::runtime_error("Unigram maximization: counts do not belong to the current vocabulary");
    }
    auto& pieces = *session.pieces;
    const size_t ordinary_count = static_cast<size_t>(std::count_if(
        pieces.begin(), pieces.end(), [](const UnigramPiece& piece) { return !piece.is_user_defined; }));
    const double total = session.expected_learned_piece_tokens +
        session.hp->piece_count_smoothing * static_cast<double>(ordinary_count);
    int unused = 0;
    for (size_t i = 0; i < session.piece_count; ++i) {
        if (pieces[i].is_user_defined) continue;
        const double expected = session.expected_counts[i];
        const float score = static_cast<float>(std::log(
            (expected + session.hp->piece_count_smoothing) / total));
        pieces[i].score = score;
        session.piece_scores[i] = score;
        if (expected == 0.0) ++unused;
    }
    return unused;
}
int runUnigramEMIteration(UnigramTrainingSession& session) {
    runUnigramExpectationStep(session);
    return runUnigramMaximizationStep(session);
}
} // namespace GRIM::Tokenizer
