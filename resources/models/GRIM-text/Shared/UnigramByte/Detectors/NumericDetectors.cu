//======================================================//
//  NumericDetectors.cu
//  Raw-text numeric detector implementations
//======================================================//

#include "NumericDetectors.hpp"

#include <cctype>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

namespace {

bool isDigit(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

} // namespace

std::optional<RawTextDetection> IntegerDetector::detect(std::string_view text,
                                                        size_t pos) const {
    if (pos >= text.size()) return std::nullopt;

    size_t i = pos;

    // Optional sign
    if (text[i] == '+' || text[i] == '-') {
        if (i + 1 >= text.size() || !isDigit(text[i + 1])) {
            return std::nullopt;
        }
        ++i;
    }

    // Must have at least one digit
    if (!isDigit(text[i])) return std::nullopt;

    // Consume digits
    while (i < text.size() && isDigit(text[i])) {
        ++i;
    }

    // Reject if followed by '.' (could be float like 5.3)
    if (i < text.size() && text[i] == '.') {
        return std::nullopt;
    }

    // Reject if followed by e/E + (digit or sign) — scientific notation like 5e3, 5E+2.
    // Do NOT reject "5english", "5em", etc. — FloatDetector already failed for those,
    // so deferring would create a gap where BOTH detectors reject the digit.
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        if (i + 1 < text.size()) {
            char next = text[i + 1];
            if (isDigit(next) || next == '+' || next == '-') {
                return std::nullopt;
            }
        }
        // "5english", "5em", "5E_something" — not scientific notation, accept as integer.
    }

    // Digits followed by alpha (e.g. "5th", "100ms", "3D") remain integer atoms;
    // the alpha suffix is tokenized separately through the raw-text registry/unigram path.
    return RawTextDetection(pos, i, AtomType::ATOM_INT, name());
}

std::optional<RawTextDetection> FloatDetector::detect(std::string_view text,
                                                      size_t pos) const {
    if (pos >= text.size()) return std::nullopt;

    size_t i = pos;
    bool has_dot = false;
    bool has_exp = false;
    bool has_digit = false;

    // Optional sign
    if (text[i] == '+' || text[i] == '-') {
        ++i;
    }

    // Integer part or leading dot
    while (i < text.size() && isDigit(text[i])) {
        has_digit = true;
        ++i;
    }

    // Decimal point
    if (i < text.size() && text[i] == '.') {
        has_dot = true;
        ++i;

        // Fractional part
        while (i < text.size() && isDigit(text[i])) {
            has_digit = true;
            ++i;
        }
    }

    // Exponent
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        has_exp = true;
        ++i;

        // Optional exponent sign
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
            ++i;
        }

        // Exponent digits
        bool has_exp_digit = false;
        while (i < text.size() && isDigit(text[i])) {
            has_exp_digit = true;
            ++i;
        }

        if (!has_exp_digit) return std::nullopt;
    }

    // Must have dot or exponent to be float (not just integer)
    if (!has_digit || (!has_dot && !has_exp)) return std::nullopt;

    return RawTextDetection(pos, i, AtomType::ATOM_FLOAT, name());
}

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM