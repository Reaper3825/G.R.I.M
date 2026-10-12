// Convergence policy over the shared session E/M operation.
#pragma once
#include "UnigramTrainingSession.hpp"
namespace GRIM::Tokenizer {
struct UnigramEMConvergenceResult {
    int iterations = 0;
    bool converged = false;
};
UnigramEMConvergenceResult runUnigramEMToConvergence(
    UnigramTrainingSession& session, const char* phase_label);
void requireUnigramConvergedPhaseNotFallbackDominated(
    const UnigramTrainingSession& session, const char* phase_label);
}
