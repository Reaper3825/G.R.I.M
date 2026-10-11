// Repeated expectation/maximization passes and convergence telemetry.
#pragma once
#include "UnigramExpectationStep.hpp"

namespace GRIM::Tokenizer {
inline constexpr double kUnigramPieceCountSmoothing = 0.1;

struct UnigramEMConvergenceResult {
    std::unordered_map<int, double> learned_token_counts;
    int iterations = 0;
    double expected_learned_piece_tokens = 0.0;
    double expected_fixed_penalty_byte_fallback_tokens = 0.0;
    double log_likelihood = 0.0;
};

UnigramEMConvergenceResult runUnigramEMToConvergence(
    std::vector<UnigramPiece>& pieces_,
    const UnigramExpectationStepRequest& expectation_request,
    const char* phase_label);
void requireUnigramConvergedPhaseNotFallbackDominated(
    const UnigramEMConvergenceResult& result, const char* phase_label);
} // namespace GRIM::Tokenizer
