//======================================================//
//  NumericTokens.cu
//  Fixed numeric sub-vocabulary matching and encoding
//======================================================//

#include "NumericTokens.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
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

bool isAsciiDigit(char value) {
    return value >= '0' && value <= '9';
}

std::size_t consumeDigits(std::string_view text, std::size_t offset) {
    while (offset < text.size() && isAsciiDigit(text[offset])) {
        ++offset;
    }
    return offset;
}

} // namespace

std::size_t numericLiteralLengthAt(std::string_view text, std::size_t offset) {
    if (offset >= text.size()) {
        return 0;
    }

    const std::size_t start = offset;
    if (text[offset] == '+' || text[offset] == '-') {
        ++offset;
        if (offset >= text.size()) {
            return 0;
        }
    }

    bool has_mantissa_digit = false;
    if (isAsciiDigit(text[offset])) {
        const std::size_t integer_start = offset;
        offset = consumeDigits(text, offset);
        has_mantissa_digit = true;

        const std::size_t first_group_size = offset - integer_start;
        if (first_group_size <= 3) {
            while (offset < text.size() && text[offset] == ',') {
                const std::size_t group_start = offset + 1;
                const std::size_t group_end = consumeDigits(text, group_start);
                if (group_end - group_start != 3) {
                    break;
                }
                offset = group_end;
            }
        }
    }

    if (offset < text.size() && text[offset] == '.' &&
        offset + 1 < text.size() && isAsciiDigit(text[offset + 1])) {
        offset = consumeDigits(text, offset + 1);
        has_mantissa_digit = true;
    }
    if (!has_mantissa_digit) {
        return 0;
    }

    if (offset < text.size() && (text[offset] == 'e' || text[offset] == 'E')) {
        std::size_t exponent_end = offset + 1;
        if (exponent_end < text.size() &&
            (text[exponent_end] == '+' || text[exponent_end] == '-')) {
            ++exponent_end;
        }
        const std::size_t exponent_digits_start = exponent_end;
        exponent_end = consumeDigits(text, exponent_end);
        if (exponent_end > exponent_digits_start) {
            offset = exponent_end;
        }
    }

    return offset - start;
}

std::vector<NumericTokenSpan> findNumericTokenSpans(
    std::string_view text,
    const std::vector<NumericTokenSpan>& excluded_spans) {
    std::size_t previous_excluded_end = 0;
    for (const NumericTokenSpan& span : excluded_spans) {
        if (span.start > span.end || span.end > text.size() ||
            span.start < previous_excluded_end) {
            throw std::runtime_error(
                "findNumericTokenSpans: excluded spans must be sorted, non-overlapping, and in bounds");
        }
        previous_excluded_end = span.end;
    }

    std::vector<NumericTokenSpan> spans;
    std::size_t excluded_index = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        while (excluded_index < excluded_spans.size() &&
               excluded_spans[excluded_index].end <= offset) {
            ++excluded_index;
        }
        if (excluded_index < excluded_spans.size() &&
            offset >= excluded_spans[excluded_index].start) {
            offset = excluded_spans[excluded_index].end;
            continue;
        }

        const std::size_t length = numericLiteralLengthAt(text, offset);
        if (length == 0) {
            ++offset;
            continue;
        }
        const std::size_t end = offset + length;
        if (excluded_index < excluded_spans.size() &&
            end > excluded_spans[excluded_index].start) {
            offset = excluded_spans[excluded_index].end;
            continue;
        }
        spans.push_back(NumericTokenSpan{offset, end});
        offset = end;
    }
    return spans;
}

void appendNumericLiteralTokenIds(
    std::string_view literal,
    std::vector<int>& token_ids) {
    if (literal.empty() || numericLiteralLengthAt(literal, 0) != literal.size()) {
        throw std::runtime_error(
            "appendNumericLiteralTokenIds: input is not one complete supported decimal literal: '" +
            std::string(literal) + "'");
    }

    std::size_t offset = 0;
    while (offset < literal.size()) {
        const char value = literal[offset];
        if (isAsciiDigit(value)) {
            std::size_t run_end = offset + 1;
            while (run_end < literal.size() && literal[run_end] == value) {
                ++run_end;
            }
            std::size_t remaining = run_end - offset;
            while (remaining > 0) {
                const std::size_t token_length = std::min<std::size_t>(remaining, 4);
                const int digit = value - '0';
                token_ids.push_back(
                    NUMERIC_TOKEN_OFFSET + static_cast<int>((token_length - 1) * 10) + digit);
                remaining -= token_length;
            }
            offset = run_end;
            continue;
        }

        int punctuation_index = -1;
        switch (value) {
            case '.': punctuation_index = 40; break;
            case ',': punctuation_index = 41; break;
            case '-': punctuation_index = 42; break;
            case '+': punctuation_index = 43; break;
            case 'e': punctuation_index = 44; break;
            case 'E': punctuation_index = 45; break;
            default:
                throw std::runtime_error(
                    "appendNumericLiteralTokenIds: unsupported byte in validated numeric literal");
        }
        token_ids.push_back(NUMERIC_TOKEN_OFFSET + punctuation_index);
        ++offset;
    }
}

std::string_view numericTokenTextOrThrow(int token_id, const char* caller) {
    if (!caller) {
        throw std::runtime_error("numericTokenTextOrThrow: caller is NULL");
    }
    if (!isNumericTokenId(token_id)) {
        throw std::runtime_error(std::string(caller) + ": token_id=" +
                                 std::to_string(token_id) +
                                 " is outside the fixed numeric token range");
    }
    return kNumericTokenText[static_cast<std::size_t>(token_id - NUMERIC_TOKEN_OFFSET)];
}

} // namespace Tokenizer
} // namespace GRIM
