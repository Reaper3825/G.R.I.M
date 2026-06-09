//======================================================//
//  NumericDetectors.cu
//  Raw-text numeric detector implementations
//======================================================//

#include "NumericDetectors.hpp"

namespace GRIM {
namespace Tokenizer {
namespace Detector {

namespace {

// Locale-independent ASCII digit check. std::isdigit consults the process
// locale and can classify bytes >= 128 differently across environments,
// which makes tokenization nondeterministic. Tokenization must be a pure
// function of the input bytes.
bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

// A '+'/'-' at pos is a numeric sign only when it is NOT in operator/suffix
// context. When the preceding byte is an ASCII alphanumeric, '.', ')' or ']',
// the byte is arithmetic or a range/connector ("5-3", "x-1", "f(x)-2",
// "a[i]-1"), and binding it to the following number would silently rewrite
// subtraction as a negative literal.
bool signIsOperatorContext(std::string_view text, size_t pos) {
    if (pos == 0) return false;
    const char prev = text[pos - 1];
    const bool prev_alnum = (prev >= '0' && prev <= '9') ||
                            (prev >= 'a' && prev <= 'z') ||
                            (prev >= 'A' && prev <= 'Z');
    return prev_alnum || prev == '.' || prev == ')' || prev == ']';
}

} // namespace

std::optional<RawTextDetection> IntegerDetector::detect(std::string_view text,
                                                        size_t pos) const {
    if (pos >= text.size()) return std::nullopt;

    size_t i = pos;

    // Optional sign — but never bind an operator to the number ("5-3" must
    // stay [5][-][3], not [5][-3]).
    if (text[i] == '+' || text[i] == '-') {
        if (signIsOperatorContext(text, i)) {
            return std::nullopt;
        }
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

    // Defer to FloatDetector only when '.' actually starts a fraction
    // ('.' followed by a digit, e.g. "5.3"). A bare trailing dot ("42.")
    // is sentence punctuation: claim the digits and leave the '.' as text.
    // Without the digit check, "42." is rejected by BOTH detectors and the
    // number silently degrades to unigram pieces.
    if (i + 1 < text.size() && text[i] == '.' && isDigit(text[i + 1])) {
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

    // Optional sign — but never bind an operator to the number ("1.5-2" must
    // stay [1.5][-][2], not [1.5][-2]).
    if (text[i] == '+' || text[i] == '-') {
        if (signIsOperatorContext(text, i)) {
            return std::nullopt;
        }
        ++i;
    }

    // Integer part or leading dot
    while (i < text.size() && isDigit(text[i])) {
        has_digit = true;
        ++i;
    }

    // Decimal point counts only when at least one fractional digit follows.
    // A bare trailing dot ("The answer is 42.") is sentence punctuation —
    // consuming it would swallow the period into the atom and reclassify the
    // integer as a float. IntegerDetector claims the digits in that case.
    if (i + 1 < text.size() && text[i] == '.' && isDigit(text[i + 1])) {
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