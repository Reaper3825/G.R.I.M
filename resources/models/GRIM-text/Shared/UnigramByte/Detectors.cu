//======================================================//
//  Detectors.cu
//  Structural Pattern Detector Implementations
//
//  Only active detectors: integer, float, hex, binary.
//  Dead detectors (path, date, time, IP, string, identifier)
//  have been deleted per Rule 26.
//======================================================//

#include "Detectors.hpp"

#include <cctype>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

bool detectInteger(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    size_t i = pos;
    
    // Optional sign
    if (text[i] == '+' || text[i] == '-') {
        if (i + 1 >= text.size() || !std::isdigit(text[i + 1])) {
            return false;
        }
        ++i;
    }
    
    // Must have at least one digit
    if (!std::isdigit(text[i])) return false;
    
    // Consume digits
    while (i < text.size() && std::isdigit(text[i])) {
        ++i;
    }
    
    // Reject if followed by '.' (could be float like 5.3)
    if (i < text.size() && text[i] == '.') {
        return false;
    }
    
    // Reject if followed by e/E + (digit or sign) — scientific notation like 5e3, 5E+2
    // Do NOT reject "5english", "5em", etc. — detectFloat already failed for those,
    // so deferring would create a gap where BOTH detectors reject the digit.
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        if (i + 1 < text.size()) {
            char next = text[i + 1];
            if (std::isdigit(next) || next == '+' || next == '-') {
                return false;  // Genuine scientific notation — defer to detectFloat
            }
        }
        // "5english", "5em", "5E_something" — not scientific notation, accept as integer
    }
    
    // NOTE: We intentionally do NOT reject digits followed by alpha (e.g. "5th", "100ms", "3D").
    // The digit run is the integer atom; the alpha suffix gets tokenized separately via Viterbi.
    // The old guard `if (isalpha(text[i])) return false` caused digits in ordinals, units, and
    // version strings to bypass atom detection entirely, leaking raw byte tokens into training.
    
    end = i;
    return i > pos;
}

bool detectFloat(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    size_t i = pos;
    bool has_dot = false;
    bool has_exp = false;
    bool has_digit = false;
    
    // Optional sign
    if (text[i] == '+' || text[i] == '-') {
        ++i;
    }
    
    // Integer part or leading dot
    while (i < text.size() && std::isdigit(text[i])) {
        has_digit = true;
        ++i;
    }
    
    // Decimal point
    if (i < text.size() && text[i] == '.') {
        has_dot = true;
        ++i;
        
        // Fractional part
        while (i < text.size() && std::isdigit(text[i])) {
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
        while (i < text.size() && std::isdigit(text[i])) {
            has_exp_digit = true;
            ++i;
        }
        
        if (!has_exp_digit) return false;
    }
    
    // Must have dot or exponent to be float (not just integer)
    if (!has_digit || (!has_dot && !has_exp)) return false;
    
    end = i;
    return true;
}

bool detectHex(const std::string& text, size_t pos, size_t& end) {
    if (pos + 2 >= text.size()) return false;
    
    if (text[pos] != '0' || (text[pos + 1] != 'x' && text[pos + 1] != 'X')) {
        return false;
    }
    
    size_t i = pos + 2;
    
    // Must have at least one hex digit
    auto isHexDigit = [](char c) {
        return std::isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    };
    
    if (!isHexDigit(text[i])) return false;
    
    while (i < text.size() && isHexDigit(text[i])) {
        ++i;
    }
    
    end = i;
    return true;
}

bool detectBinary(const std::string& text, size_t pos, size_t& end) {
    if (pos + 2 >= text.size()) return false;
    
    if (text[pos] != '0' || (text[pos + 1] != 'b' && text[pos + 1] != 'B')) {
        return false;
    }
    
    size_t i = pos + 2;
    
    // Must have at least one binary digit
    if (text[i] != '0' && text[i] != '1') return false;
    
    while (i < text.size() && (text[i] == '0' || text[i] == '1')) {
        ++i;
    }
    
    end = i;
    return true;
}

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM
