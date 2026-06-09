//======================================================//
//  TextUtils.cu
//  Implementation of UTF-8 utilities and SentencePiece normalization
//======================================================//

#include "TextUtils.hpp"
#include "TokenLayout.hpp"  // For AtomSpan (via forward decl is enough but we need the struct def)
#include "Unigram.hpp"      // For AtomSpan struct definition

#include <algorithm>
#include <stdexcept>

namespace GRIM {
namespace Tokenizer {

namespace {

size_t spieceRewriteSourceLength(const std::string& text, size_t pos) {
    if (pos >= text.size()) {
        throw std::runtime_error("spieceRewriteSourceLength: pos is outside text");
    }
    if (text[pos] == '\r') {
        if (pos + 1 < text.size() && text[pos + 1] == '\n') {
            return 2;
        }
        return 1;
    }
    if (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n') {
        return 1;
    }
    return 0;
}

} // namespace

//======================================================//
//  SentencePiece ▁ Marker (definition)
//======================================================//
const char SPIECE_UNDERLINE[] = "\xE2\x96\x81";

//======================================================//
//  UTF-8 Helpers
//======================================================//

size_t utf8SequenceLength(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

bool utf8DecodeAt(const std::string& s, size_t pos, uint32_t* out_cp, size_t* out_len) {
    if (pos >= s.size()) return false;
    unsigned char b0 = static_cast<unsigned char>(s[pos]);
    if ((b0 & 0x80) == 0) {
        *out_cp = b0;
        *out_len = 1;
        return true;
    }
    const size_t n = utf8SequenceLength(b0);
    if (n < 2 || pos + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k) {
        if ((static_cast<unsigned char>(s[pos + k]) & 0xC0) != 0x80) return false;
    }
    uint32_t cp = 0;
    if (n == 2) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
        if (cp < 0x80u) return false;
    } else if (n == 3) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
        if (cp < 0x800u) return false;
        if (cp >= 0xD800u && cp <= 0xDFFFu) return false;
    } else if (n == 4) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        unsigned char b3 = static_cast<unsigned char>(s[pos + 3]);
        cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
        if (cp < 0x10000u || cp > 0x10FFFFu) return false;
    } else {
        return false;
    }
    *out_cp = cp;
    *out_len = n;
    return true;
}

//======================================================//
//  Character Classification
//======================================================//

bool isWhitespaceASCII(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool isPunct(char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

bool isStructuralEdgeWhitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return cp == 0x09u || cp == 0x0Au || cp == 0x0Bu || cp == 0x0Cu || cp == 0x0Du || cp == 0x20u;
    }
    if (cp == 0x85u) return true;   // NEL
    if (cp == 0xA0u) return true;  // NBSP
    if (cp == 0x1680u) return true; // Ogham space mark
    if (cp >= 0x2000u && cp <= 0x200Au) return true; // General punctuation spaces
    if (cp >= 0x200Bu && cp <= 0x200Fu) return true; // ZWSP, ZWNJ, ZWJ, marks
    if (cp == 0x2028u || cp == 0x2029u) return true; // Line / paragraph separator
    if (cp == 0x202Fu) return true; // Narrow no-break space
    if (cp == 0x205Fu) return true; // Medium mathematical space
    if (cp == 0x2060u) return true;  // Word joiner
    if (cp == 0x3000u) return true;  // Ideographic space
    if (cp == 0xFEFFu) return true;  // BOM / ZWNBSP
    return false;
}

//======================================================//
//  SentencePiece Normalization
//======================================================//

std::string normalizeSpaces(const std::string& text, bool prepend_space) {
    if (text.empty()) return text;

    size_t rewrite_count = 0;
    for (size_t i = 0; i < text.size(); ) {
        const size_t rewrite_source_len = spieceRewriteSourceLength(text, i);
        if (rewrite_source_len != 0) {
            ++rewrite_count;
            i += rewrite_source_len;
        } else {
            ++i;
        }
    }

    std::string result;
    result.reserve(text.size() + (prepend_space ? SPIECE_UNDERLINE_LEN : 0) + rewrite_count * 2);
    if (prepend_space) {
        result.append(SPIECE_UNDERLINE, SPIECE_UNDERLINE_LEN);
    }

    for (size_t i = 0; i < text.size(); ) {
        const size_t rewrite_source_len = spieceRewriteSourceLength(text, i);
        if (rewrite_source_len != 0) {
            result.append(SPIECE_UNDERLINE, SPIECE_UNDERLINE_LEN);
            i += rewrite_source_len;
        } else {
            result.push_back(text[i]);
            ++i;
        }
    }
    return result;
}

std::string denormalizeSpaces(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size(); ) {
        if (i + SPIECE_UNDERLINE_LEN <= text.size() &&
            static_cast<unsigned char>(text[i])   == 0xE2 &&
            static_cast<unsigned char>(text[i+1]) == 0x96 &&
            static_cast<unsigned char>(text[i+2]) == 0x81) {
            result.push_back(' ');
            i += SPIECE_UNDERLINE_LEN;
        } else {
            result.push_back(text[i]);
            ++i;
        }
    }

    // Strip leading space (from the prepended ▁). All source ASCII spacing bytes
    // are intentionally rewritten through the same ▁ marker, so decode returns
    // plain spaces rather than reconstructing tabs/newlines/carriage returns.
    if (!result.empty() && result[0] == ' ') {
        result.erase(0, 1);
    }
    return result;
}

std::string normalizeWithSpans(const std::string& text, std::vector<AtomSpan>& spans) {
    if (text.empty()) {
        for (const auto& span : spans) {
            if (span.start > span.end || span.end > text.size()) {
                throw std::runtime_error("[normalizeWithSpans] Invalid AtomSpan byte range");
            }
        }
        return text;
    }

    // Build orig→norm byte offset mapping
    std::vector<size_t> orig_to_norm(text.size() + 1);
    size_t norm_pos = SPIECE_UNDERLINE_LEN; // 3-byte prepend

    for (size_t i = 0; i < text.size(); ) {
        orig_to_norm[i] = norm_pos;
        const size_t rewrite_source_len = spieceRewriteSourceLength(text, i);
        if (rewrite_source_len != 0) {
            if (rewrite_source_len == 2) {
                orig_to_norm[i + 1] = norm_pos;
            }
            norm_pos += SPIECE_UNDERLINE_LEN;
            i += rewrite_source_len;
        } else {
            ++norm_pos;
            ++i;
        }
    }
    orig_to_norm[text.size()] = norm_pos;

    for (auto& span : spans) {
        if (span.start > span.end || span.end > text.size()) {
            throw std::runtime_error("[normalizeWithSpans] Invalid AtomSpan byte range");
        }

        span.start = orig_to_norm[span.start];
        span.end   = orig_to_norm[span.end];
    }

    return normalizeSpaces(text, true);
}

} // namespace Tokenizer
} // namespace GRIM
