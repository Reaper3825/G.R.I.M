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
    const UnigramTrainingSession& session, const char* phase_label) {
    requireUnigramTrainingSessionReady(session, "Unigram converged phase");
    requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
        session.expected_learned_piece_tokens,
        session.expected_fixed_penalty_byte_fallback_tokens,
        session.hp->minimum_learned_to_fallback_ratio,
        phase_label,
        "UnigramLM::trainFromCorpus converged forward-backward phase");
}

UnigramEMConvergenceResult runUnigramEMToConvergence(
    UnigramTrainingSession& session, const char* phase_label) {
    requireUnigramTrainingSessionReady(session, "Unigram convergence");
    if (!phase_label || !phase_label[0]) {
        throw std::runtime_error("Unigram convergence: phase label is empty");
    }
    const auto& hp = *session.hp;
    const int EM_MAX_ITERATIONS = hp.em_max_iterations;
    const double EM_CONVERGENCE_THRESH = hp.em_convergence_tolerance;
    double prev_ll = -1e30;
    int iter = 0;
    bool reached_convergence = false;
    for (; iter < EM_MAX_ITERATIONS; ++iter) {
        int unused = runUnigramEMIteration(session);

        double relative_change = 1.0;
        if (prev_ll >= -1e20) {
            relative_change = std::abs(
                (session.log_likelihood - prev_ll) /
                std::min(std::abs(prev_ll), std::abs(session.log_likelihood)));
        }

        std::cout << "[UnigramLM] " << phase_label << " iter " << (iter + 1)
                  << ": LL=" << std::fixed << std::setprecision(2) << session.log_likelihood
                  << ", expected_learned_piece_tokens=" << session.expected_learned_piece_tokens
                  << ", expected_fixed_penalty_byte_fallback_tokens=" << session.expected_fixed_penalty_byte_fallback_tokens
                  << ", unused=" << unused
                  << ", delta=" << std::scientific << std::setprecision(4) << relative_change
                  << std::defaultfloat << std::endl;

        bool converged = (iter > 0 && relative_change < EM_CONVERGENCE_THRESH);
        prev_ll = session.log_likelihood;
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
    return UnigramEMConvergenceResult{iter, reached_convergence};

}

} // namespace Tokenizer
} // namespace GRIM
