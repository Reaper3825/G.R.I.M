// Corpus posterior aggregation through persistent session addresses.
#pragma once
#include "UnigramTrainingSession.hpp"
namespace GRIM::Tokenizer {
void runUnigramExpectationStep(UnigramTrainingSession& session);
}
