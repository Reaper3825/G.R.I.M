//======================================================//
//  NumericTokens.hpp
//  Fixed numeric sub-vocabulary declarations
//======================================================//

#pragma once

#include "TokenLayout.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

#include <string>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct NumericTokenDefinition {
    int token_id = -1;
    std::string text;
};

struct NumericTokenAppendTarget {
    std::vector<NumericTokenDefinition>& entries;
};

void AppendNumericTokens(
    const ::GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    NumericTokenAppendTarget target);

} // namespace Tokenizer
} // namespace GRIM
