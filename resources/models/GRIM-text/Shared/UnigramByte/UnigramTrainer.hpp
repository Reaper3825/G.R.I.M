//======================================================//
//  UnigramTrainer.hpp
//  Training Pipeline Declaration for UnigramLM
//
//  The trainFromCorpus() method is declared on UnigramLM
//  in Unigram.hpp. This header exists solely as a forward
//  reference for build systems that want to express the
//  training dependency explicitly. The implementation is
//  in UnigramTrainer.cu (split compilation unit).
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

// Training implementation lives in UnigramTrainer.cu.
// trainFromCorpus() is a method on UnigramLM declared in Unigram.hpp.
// This header serves as documentation — include Unigram.hpp for the API.

#include "Unigram.hpp"
#include "TextUtils.hpp"
