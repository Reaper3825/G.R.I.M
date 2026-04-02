//======================================================//
//  TextUtils.hpp
//  UTF-8 Utilities and SentencePiece Normalization
//
//  Pure utility functions used across training and inference.
//  No state, no classes, no CUDA — just string processing.
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM {
namespace Tokenizer {

// Forward declaration for training pipeline
struct AtomSpan;

//======================================================//
//  SentencePiece ▁ Marker
//======================================================//
// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece word-boundary marker
extern const char SPIECE_UNDERLINE[];
constexpr size_t SPIECE_UNDERLINE_LEN = 3;

//======================================================//
//  UTF-8 Helpers
//======================================================//

// Returns the byte length of the UTF-8 sequence starting at byte c.
size_t utf8SequenceLength(unsigned char c);

// Decode one UTF-8 codepoint at byte index pos. Returns false if truncated or ill-formed.
bool utf8DecodeAt(const std::string& s, size_t pos, uint32_t* out_cp, size_t* out_len);

//======================================================//
//  Character Classification
//======================================================//

// ASCII whitespace check (space, tab, newline, carriage return)
bool isWhitespaceASCII(unsigned char c);

// Punctuation check (ASCII subset for speed)
bool isPunct(char c);

// Codepoints stripped only at substring edges for structural vocab dedup (training).
bool isStructuralEdgeWhitespace(uint32_t cp);

//======================================================//
//  SentencePiece-style Whitespace Normalization
//  Replaces spaces with ▁ (U+2581) and prepends ▁ at start.
//======================================================//

// "Hello World" → "▁Hello▁World"
std::string normalizeSpaces(const std::string& text);

// "▁Hello▁World" → "Hello World"
std::string denormalizeSpaces(const std::string& text);

// Normalize and adjust atom span byte offsets to match expansion.
std::string normalizeWithSpans(const std::string& text, std::vector<AtomSpan>& spans);

} // namespace Tokenizer
} // namespace GRIM
