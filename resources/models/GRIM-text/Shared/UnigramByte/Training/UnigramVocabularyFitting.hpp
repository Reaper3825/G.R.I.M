// Initial EM, pruning/refits and final EM share one persistent session.
#pragma once
#include "UnigramTrainingSession.hpp"
namespace GRIM::Tokenizer {
void fitUnigramVocabulary(UnigramTrainingSession& session);
}
