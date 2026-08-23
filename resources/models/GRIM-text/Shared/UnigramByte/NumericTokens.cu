//======================================================//
//  NumericTokens.cu
//  Fixed numeric sub-vocabulary construction
//======================================================//

#include "NumericTokens.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM {
namespace Tokenizer {
namespace {

inline constexpr std::array<std::string_view, NUMERIC_VOCAB_SIZE> kNumericTokenText = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "00", "11", "22", "33", "44", "55", "66", "77", "88", "99",
    "000", "111", "222", "333", "444", "555", "666", "777", "888", "999",
    "0000", "1111", "2222", "3333", "4444",
    "5555", "6666", "7777", "8888", "9999",
    ".", ",", "-", "+", "e", "E"
};

static_assert(NUMERIC_TOKEN_OFFSET == 260,
              "Numeric token range must begin immediately after byte tokens");
static_assert(NUMERIC_TOKEN_END == 306,
              "Numeric token range must occupy IDs [260, 305]");
static_assert(kNumericTokenText.size() == static_cast<std::size_t>(NUMERIC_VOCAB_SIZE),
              "Numeric token table size must match NUMERIC_VOCAB_SIZE");

} // namespace

void AppendNumericTokens(
    const ::GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    NumericTokenAppendTarget target) {
    if (tokenizer_hp.target_vocab_size <= 0) {
        throw std::runtime_error(
            "AppendNumericTokens: tokenizer_hp.target_vocab_size must be > 0");
    }
    if (tokenizer_hp.target_vocab_size >
        std::numeric_limits<int>::max() - NUMERIC_TOKEN_END) {
        throw std::runtime_error(
            "AppendNumericTokens: configured learned-vocabulary size would overflow "
            "the token ID space after the numeric sub-vocabulary");
    }

    for (const NumericTokenDefinition& existing : target.entries) {
        if (existing.token_id >= NUMERIC_TOKEN_OFFSET &&
            existing.token_id < NUMERIC_TOKEN_END) {
            throw std::runtime_error(
                "AppendNumericTokens: target already contains token ID " +
                std::to_string(existing.token_id) + " from the numeric range");
        }
    }

    std::vector<NumericTokenDefinition> staged = target.entries;
    staged.reserve(staged.size() + kNumericTokenText.size());
    for (std::size_t index = 0; index < kNumericTokenText.size(); ++index) {
        staged.push_back(NumericTokenDefinition{
            NUMERIC_TOKEN_OFFSET + static_cast<int>(index),
            std::string(kNumericTokenText[index])});
    }

    target.entries.swap(staged);
}

} // namespace Tokenizer
} // namespace GRIM
