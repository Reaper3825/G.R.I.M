//======================================================//
//  NumericTokens.hpp
//  Fixed numeric sub-vocabulary matching and encoding
//======================================================//

#pragma once

#include "TokenLayout.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct NumericTokenSpan {
    std::size_t start = 0;
    std::size_t end = 0;
};

// Returns the byte length of the numeric literal beginning at offset, or zero
// when offset is not the start of a supported decimal literal.
std::size_t numericLiteralLengthAt(std::string_view text, std::size_t offset);

// Finds supported decimal literals that do not overlap the sorted, non-
// overlapping excluded spans. This is shared by tokenizer training and runtime
// corpus emission so fixed numeric tokens cannot drift from learned-vocab
// boundaries.
std::vector<NumericTokenSpan> findNumericTokenSpans(
    std::string_view text,
    const std::vector<NumericTokenSpan>& excluded_spans = {});

// Encodes one complete supported decimal literal into the minimum number of
// fixed numeric tokens. Repeated equal digits use the longest available token.
void appendNumericLiteralTokenIds(
    std::string_view literal,
    std::vector<int>& token_ids);

std::string_view numericTokenTextOrThrow(int token_id, const char* caller);

} // namespace Tokenizer
} // namespace GRIM
