// Learned score updates through the training session's dense count addresses.
#pragma once
#include "UnigramTrainingSession.hpp"
namespace GRIM::Tokenizer {
int runUnigramMaximizationStep(UnigramTrainingSession& session);
// Shared E/M operation used by convergence and pruning refits.
int runUnigramEMIteration(UnigramTrainingSession& session);
}
