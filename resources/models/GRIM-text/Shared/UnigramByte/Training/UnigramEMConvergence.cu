// EM convergence control and iteration telemetry.

#include "UnigramEMConvergence.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"
#include "UnigramMaximizationStep.hpp"
#include "UnigramTrainingValidation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIM {
namespace Tokenizer {

void requireUnigramConvergedPhaseNotFallbackDominated(
    const UnigramEMConvergenceResult& result, const ::GRIM::HyperParameters::TokenizerHP& hp, const char* phase_label) {
    requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
        result.expected_learned_piece_tokens,
        result.expected_fixed_penalty_byte_fallback_tokens,
        hp.minimum_learned_to_fallback_ratio,
        phase_label,
        "UnigramLM::trainFromCorpus converged forward-backward phase");
}

UnigramEMConvergenceResult runUnigramEMToConvergence(
    std::vector<UnigramPiece>& pieces_,
    const UnigramExpectationStepRequest& expectation_request,
    const ::GRIM::HyperParameters::TokenizerHP& hp, const char* phase_label) {
    const int EM_MAX_ITERATIONS = hp.em_max_iterations;
    const double EM_CONVERGENCE_THRESH = hp.em_convergence_tolerance;
    const double SMOOTHING = hp.piece_count_smoothing;
    double prev_ll = -1e30;
    UnigramExpectationStepResult last_e_step;
    int iter = 0;
    bool reached_convergence = false;
    for (; iter < EM_MAX_ITERATIONS; ++iter) {
        UnigramExpectationStepResult e_step = runUnigramExpectationStep(expectation_request);
        int unused = runUnigramMaximizationStep(UnigramMaximizationStepRequest{
            pieces_, e_step.learned_token_counts, e_step.expected_learned_piece_tokens, SMOOTHING});

        double relative_change = 1.0;
        if (prev_ll >= -1e20) {
            relative_change = std::abs(
                (e_step.log_likelihood - prev_ll) /
                std::min(std::abs(prev_ll), std::abs(e_step.log_likelihood)));
        }

        std::cout << "[UnigramLM] " << phase_label << " iter " << (iter + 1)
                  << ": LL=" << std::fixed << std::setprecision(2) << e_step.log_likelihood
                  << ", expected_learned_piece_tokens=" << e_step.expected_learned_piece_tokens
                  << ", expected_fixed_penalty_byte_fallback_tokens=" << e_step.expected_fixed_penalty_byte_fallback_tokens
                  << ", unused=" << unused
                  << ", delta=" << std::scientific << std::setprecision(4) << relative_change
                  << std::defaultfloat << std::endl;

        last_e_step = std::move(e_step);
        bool converged = (iter > 0 && relative_change < EM_CONVERGENCE_THRESH);
        prev_ll = last_e_step.log_likelihood;
        if (converged) {
            reached_convergence = true;
            std::cout << "[UnigramLM] " << phase_label << " converged after " << (iter + 1)
                      << " iterations (delta=" << std::scientific << std::setprecision(4)
                      << relative_change << std::defaultfloat << ")" << std::endl;
            ++iter;
            break;
        }
    }
    if (!reached_convergence) {
        if (hp.em_fail_on_iteration_limit) {
            throw std::runtime_error(std::string("UnigramLM: ") + phase_label + " reached tokenizer_em_max_iterations without convergence");
        }
        std::cout << "[UnigramLM] " << phase_label << " hit max iterations ("
                  << EM_MAX_ITERATIONS << ") without full convergence" << std::endl;
    }
    return UnigramEMConvergenceResult{std::move(last_e_step.learned_token_counts),
                                     iter,
                                     last_e_step.expected_learned_piece_tokens,
                                     last_e_step.expected_fixed_penalty_byte_fallback_tokens,
                                     last_e_step.log_likelihood};
}

} // namespace Tokenizer
} // namespace GRIM
