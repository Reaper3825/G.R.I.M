//======================================================//
//  TokenLayout.hpp
//  Token ID Layout Constants and AtomType Enum for GRIM Tokenizer
//
//  This is the foundational header that defines the token ID
//  numbering scheme shared across all tokenizer components.
//  Every other tokenizer header includes this instead of
//  reaching into Unigram.hpp for constants.
//
//  Token ID layout:
//    [0-3]                    = Special tokens (<unk>, <pad>, <s>, </s>)
//    [4-259]                  = Byte tokens (fallback)
//    [ATOM_TOKEN_OFFSET..UNIGRAM_VOCAB_OFFSET-1] = Atom tokens (structural placeholders)
//    [UNIGRAM_VOCAB_OFFSET+]  = Unigram vocabulary (regular pieces only)
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cstdint>
#include <string>

#include "Byte.hpp"  // For BYTE_TOKEN_OFFSET, BYTE_VOCAB_SIZE, special token IDs

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Atom Types — distinct tokens per numeric sub-type
//======================================================//
enum class AtomType : int {
    ATOM_NONE  = 0,
    ATOM_INT   = 1,   // Integer literals: 42, -17, +5
    ATOM_FLOAT = 2,   // Float literals: 3.14, -2.5e10, .5
    ATOM_ACTIVE_COUNT,
    ATOM_TYPE_COUNT
};

constexpr int kAtomTypeCount = static_cast<int>(AtomType::ATOM_ACTIVE_COUNT);

//======================================================//
//  Token ID Layout Constants
//======================================================//
constexpr int ATOM_TOKEN_OFFSET = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;  // Atoms start after byte tokens (260)
inline int ATOM_VOCAB_SIZE = kAtomTypeCount;  // Atom slots derived from AtomType count
inline int UNIGRAM_VOCAB_OFFSET = ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE;
inline uint32_t ATOM_TOKEN_BASE = static_cast<uint32_t>(ATOM_TOKEN_OFFSET);
inline uint32_t ATOM_TOKEN_MAX = static_cast<uint32_t>(UNIGRAM_VOCAB_OFFSET);
// Sentinel: position has no registered AtomTable entry (0 is a valid AtomTable ID)
constexpr uint32_t kAtomEntryNone = UINT32_MAX;
constexpr int MAX_PIECE_LENGTH = 32;           // Maximum token length in bytes
constexpr float UNKNOWN_SCORE = -100.0f;       // Score for unknown pieces

//======================================================//
//  Layout Configuration
//======================================================//
inline void configureTokenLayout(int /*atom_vocab_size*/) {
    // Atom tokens are reserved for type-only placeholders; size derived from AtomType.
    ATOM_VOCAB_SIZE = kAtomTypeCount;
    UNIGRAM_VOCAB_OFFSET = ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE;
    ATOM_TOKEN_MAX = static_cast<uint32_t>(UNIGRAM_VOCAB_OFFSET);
}

//======================================================//
//  Token ID ↔ Index Conversion
//======================================================//
inline int tokenIdForIndex(int index) { return UNIGRAM_VOCAB_OFFSET + index; }
inline int indexForTokenId(int token_id) { return token_id - UNIGRAM_VOCAB_OFFSET; }

//======================================================//
//  Atom Token Helpers
//======================================================//
inline int atomTypeToTokenId(AtomType type) {
    return ATOM_TOKEN_OFFSET + static_cast<int>(type);
}

inline AtomType tokenIdToAtomType(int token_id) {
    if (token_id < ATOM_TOKEN_OFFSET || token_id >= UNIGRAM_VOCAB_OFFSET) {
        return AtomType::ATOM_NONE;
    }
    return static_cast<AtomType>(token_id - ATOM_TOKEN_OFFSET);
}

inline bool isAtomToken(int token_id) {
    return token_id >= ATOM_TOKEN_OFFSET && token_id < UNIGRAM_VOCAB_OFFSET;
}

inline const char* atomTypeName(AtomType type) {
    switch (type) {
        case AtomType::ATOM_NONE:  return "NONE";
        case AtomType::ATOM_INT:   return "INT";
        case AtomType::ATOM_FLOAT: return "FLOAT";
        default: return "UNKNOWN";
    }
}

inline bool isNumericAtom(AtomType type) {
    return type == AtomType::ATOM_INT || type == AtomType::ATOM_FLOAT;
}

} // namespace Tokenizer
} // namespace GRIM
