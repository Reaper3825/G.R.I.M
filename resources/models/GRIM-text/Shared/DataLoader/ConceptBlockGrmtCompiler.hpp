#pragma once

#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

// Compiles the selected ConceptBlock curriculum into a GRMT tokenizer artifact.
// Existing compatible artifacts are reused according to TokenizerHP.
bool PrepareTrainingDataFromCache(
    const HyperParameters::TokenizerHP& tokenizer_hp);

} // namespace GRIM
