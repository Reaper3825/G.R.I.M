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
//    [260-305]                = Fixed numeric tokens
//    [ATOM_TOKEN_OFFSET..UNIGRAM_VOCAB_OFFSET-1] = Atom span boundary tokens
//    [UNIGRAM_VOCAB_OFFSET+]  = Unigram vocabulary (regular pieces only)
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Reserved + Byte Token Constants
//======================================================//
constexpr int NUM_SPECIAL_TOKENS = 4;          // <unk>=0, <pad>=1, <s>=2, </s>=3
constexpr int SPECIAL_TOKEN_OFFSET = 0;        // Special tokens start at ID 0
constexpr int BYTE_VOCAB_SIZE = 256;           // 0x00 - 0xFF
constexpr int BYTE_TOKEN_OFFSET = NUM_SPECIAL_TOKENS;  // Byte tokens start at ID 4

// Absolute special token IDs
constexpr int UNK_TOKEN_ID = 0;
constexpr int PAD_TOKEN_ID = 1;
constexpr int BOS_TOKEN_ID = 2;
constexpr int EOS_TOKEN_ID = 3;

inline bool isByteTokenId(int token_id) {
    return token_id >= BYTE_TOKEN_OFFSET && token_id < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;
}

inline int byteToTokenId(uint8_t byte_value) {
    return static_cast<int>(byte_value) + BYTE_TOKEN_OFFSET;
}

inline uint8_t tokenIdToByteOrThrow(int token_id, const char* caller) {
    if (!isByteTokenId(token_id)) {
        throw std::runtime_error(std::string(caller) + ": token_id=" +
                                 std::to_string(token_id) +
                                 " is not a byte token");
    }
    return static_cast<uint8_t>(token_id - BYTE_TOKEN_OFFSET);
}

//======================================================//
//  Fixed Numeric Token Constants
//======================================================//
inline constexpr int NUMERIC_TOKEN_OFFSET = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;
inline constexpr int NUMERIC_VOCAB_SIZE = 46;
inline constexpr int NUMERIC_TOKEN_END = NUMERIC_TOKEN_OFFSET + NUMERIC_VOCAB_SIZE;

inline bool isNumericTokenId(int token_id) {
    return token_id >= NUMERIC_TOKEN_OFFSET && token_id < NUMERIC_TOKEN_END;
}

//======================================================//
//  Atom Types — distinct tokens per authored value type
//======================================================//
enum class AtomType : int {
    ATOM_INT    = 0,   // Integer literals: 42, -17, +5
    ATOM_FLOAT  = 1,   // Float literals: 3.14, -2.5e10, .5
    ATOM_STRING = 2,   // Authored string values
    ATOM_BOOL   = 3,   // Authored boolean values: true, false
    ATOM_ENTITY = 4,   // Authored named entities in their exact UTF-8 byte form
    ATOM_ACTIVE_COUNT,
    ATOM_TYPE_COUNT = ATOM_ACTIVE_COUNT
};

constexpr int kAtomTypeCount = static_cast<int>(AtomType::ATOM_ACTIVE_COUNT);

// Each active atom type owns an opening and closing boundary token. Opening
// tokens begin after the fixed numeric sub-vocabulary: <INT>=306,
// <FLOAT>=307, <STRING>=308, <BOOL>=309, and <ENTITY>=310. Closing tokens
// follow as a second type-indexed block beginning at 311.
enum class AtomBoundaryKind : uint8_t {
    OPEN = 0,
    CLOSE = 1
};

//======================================================//
//  Token ID Layout Constants
//======================================================//
constexpr int ATOM_TOKEN_OFFSET = NUMERIC_TOKEN_END;  // Atoms start after numeric tokens (306)
inline constexpr int ATOM_OPEN_TOKEN_OFFSET = ATOM_TOKEN_OFFSET;
inline constexpr int ATOM_CLOSE_TOKEN_OFFSET = ATOM_OPEN_TOKEN_OFFSET + kAtomTypeCount;
inline constexpr int ATOM_VOCAB_SIZE = kAtomTypeCount * 2;
inline constexpr int UNIGRAM_VOCAB_OFFSET = ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE;
inline constexpr uint32_t ATOM_TOKEN_BASE = static_cast<uint32_t>(ATOM_TOKEN_OFFSET);
inline constexpr uint32_t ATOM_TOKEN_MAX = static_cast<uint32_t>(UNIGRAM_VOCAB_OFFSET);
static_assert(NUMERIC_TOKEN_OFFSET == 260, "Numeric token range must begin at ID 260");
static_assert(ATOM_TOKEN_OFFSET == 306, "Atom token range must begin at ID 306");
static_assert(UNIGRAM_VOCAB_OFFSET == 316, "Learned unigram range must begin at ID 316");
// Sentinel: position has no registered AtomTable entry (0 is a valid AtomTable ID)
constexpr uint32_t kAtomEntryNone = UINT32_MAX;
// Sequence-local atom addresses use a separate typed index space and never
// alias durable AtomTable entry IDs.
constexpr uint32_t kLocalAtomIndexNone = UINT32_MAX;
constexpr int MAX_PIECE_LENGTH = 32;           // Maximum token length in bytes
constexpr float UNKNOWN_SCORE = -100.0f;       // Unnormalized fixed per-byte fallback penalty

//======================================================//
//  Layout Special Token Metadata
//======================================================//
struct SpecialTokenDefinition {
    int id;
    const char* text;
    const char* name;
};

inline constexpr SpecialTokenDefinition SPECIAL_TOKEN_DEFINITIONS[NUM_SPECIAL_TOKENS] = {
    {UNK_TOKEN_ID, "<unk>", "UNK"},
    {PAD_TOKEN_ID, "<pad>", "PAD"},
    {BOS_TOKEN_ID, "<s>",   "BOS"},
    {EOS_TOKEN_ID, "</s>",  "EOS"}
};

inline bool isSpecialTokenId(int token_id) {
    return token_id >= SPECIAL_TOKEN_OFFSET &&
           token_id < SPECIAL_TOKEN_OFFSET + NUM_SPECIAL_TOKENS;
}

inline const char* specialTokenText(int token_id) {
    for (const auto& def : SPECIAL_TOKEN_DEFINITIONS) {
        if (def.id == token_id) return def.text;
    }
    throw std::runtime_error("specialTokenText: token_id=" + std::to_string(token_id) +
                             " is not a registered special token");
}

inline const char* specialTokenName(int token_id) {
    for (const auto& def : SPECIAL_TOKEN_DEFINITIONS) {
        if (def.id == token_id) return def.name;
    }
    throw std::runtime_error("specialTokenName: token_id=" + std::to_string(token_id) +
                             " is not a registered special token");
}

inline bool isNeverTargetSpecialTokenId(int token_id) {
    return token_id == UNK_TOKEN_ID || token_id == PAD_TOKEN_ID || token_id == BOS_TOKEN_ID;
}

//======================================================//
//  TokenLayout — runtime-queried token ID ranges
//======================================================//
struct TokenLayout {
    int num_special  = 0;
    int num_bytes    = 0;
    int num_numeric  = 0;
    int num_atoms    = 0;
    int num_unigram  = 0;

    int special_offset() const { return SPECIAL_TOKEN_OFFSET; }
    int byte_offset()    const { return num_special; }
    int numeric_offset() const { return num_special + num_bytes; }
    int atom_offset()    const { return num_special + num_bytes + num_numeric; }
    int unigram_offset() const { return num_special + num_bytes + num_numeric + num_atoms; }
    int total_vocab()    const { return num_special + num_bytes + num_numeric + num_atoms + num_unigram; }

    bool isSpecial(int id) const { return id >= special_offset() && id < byte_offset(); }
    bool isByte(int id)    const { return id >= byte_offset()    && id < numeric_offset(); }
    bool isNumeric(int id) const { return id >= numeric_offset() && id < atom_offset(); }
    bool isAtom(int id)    const { return id >= atom_offset()    && id < unigram_offset(); }
    bool isUnigram(int id) const { return id >= unigram_offset() && id < total_vocab(); }

    int firstContentTokenId() const { return num_special; }
};

inline TokenLayout tokenLayoutFromActualVocabOrThrow(
    std::uint32_t actual_vocab_size,
    const char* caller)
{
    if (actual_vocab_size < static_cast<std::uint32_t>(UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error(std::string(caller) +
            ": actual_vocab_size must include special+byte+numeric+atom ranges (>= " +
            std::to_string(UNIGRAM_VOCAB_OFFSET) + "), got " +
            std::to_string(actual_vocab_size));
    }
    if (actual_vocab_size > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(caller) +
            ": actual_vocab_size=" + std::to_string(actual_vocab_size) +
            " exceeds int capacity for TokenLayout");
    }

    TokenLayout layout;
    layout.num_special = NUM_SPECIAL_TOKENS;
    layout.num_bytes = BYTE_VOCAB_SIZE;
    layout.num_numeric = NUMERIC_VOCAB_SIZE;
    layout.num_atoms = ATOM_VOCAB_SIZE;
    layout.num_unigram = static_cast<int>(actual_vocab_size) - UNIGRAM_VOCAB_OFFSET;
    if (layout.num_unigram < 0) {
        throw std::runtime_error(std::string(caller) +
            ": derived num_unigram is negative for actual_vocab_size=" +
            std::to_string(actual_vocab_size));
    }
    if (layout.total_vocab() != static_cast<int>(actual_vocab_size)) {
        throw std::runtime_error(std::string(caller) +
            ": TokenLayout total_vocab=" + std::to_string(layout.total_vocab()) +
            " != actual_vocab_size=" + std::to_string(actual_vocab_size));
    }
    return layout;
}

//======================================================//
//  Token ID ↔ Index Conversion
//======================================================//
inline int tokenIdForIndex(int index) { return UNIGRAM_VOCAB_OFFSET + index; }
inline int indexForTokenId(int token_id) { return token_id - UNIGRAM_VOCAB_OFFSET; }
inline bool isAtomTokenId(int token_id) {
    return token_id >= ATOM_TOKEN_OFFSET && token_id < UNIGRAM_VOCAB_OFFSET;
}

inline bool isAtomOpenTokenId(int token_id) {
    return token_id >= ATOM_OPEN_TOKEN_OFFSET && token_id < ATOM_CLOSE_TOKEN_OFFSET;
}

inline bool isAtomCloseTokenId(int token_id) {
    return token_id >= ATOM_CLOSE_TOKEN_OFFSET && token_id < UNIGRAM_VOCAB_OFFSET;
}

//======================================================//
//  Atom Token Helpers
//======================================================//
inline int atomTypeIndexOrThrow(AtomType type, const char* caller) {
    switch (type) {
        case AtomType::ATOM_INT:
        case AtomType::ATOM_FLOAT:
        case AtomType::ATOM_STRING:
        case AtomType::ATOM_BOOL:
        case AtomType::ATOM_ENTITY:
            return static_cast<int>(type);
        default:
            throw std::runtime_error(std::string(caller) +
                                     ": invalid atom type cannot be mapped into the atom token range");
    }
}

inline int atomTypeToOpenTokenId(AtomType type) {
    return ATOM_OPEN_TOKEN_OFFSET + atomTypeIndexOrThrow(type, "atomTypeToOpenTokenId");
}

inline int atomTypeToCloseTokenId(AtomType type) {
    return ATOM_CLOSE_TOKEN_OFFSET + atomTypeIndexOrThrow(type, "atomTypeToCloseTokenId");
}

inline AtomBoundaryKind atomTokenBoundaryKind(int token_id) {
    if (isAtomOpenTokenId(token_id)) {
        return AtomBoundaryKind::OPEN;
    }
    if (isAtomCloseTokenId(token_id)) {
        return AtomBoundaryKind::CLOSE;
    }
    throw std::runtime_error("atomTokenBoundaryKind: token_id=" + std::to_string(token_id) +
                             " is outside the atom token range");
}

inline AtomType tokenIdToAtomType(int token_id) {
    if (!isAtomTokenId(token_id)) {
        throw std::runtime_error("tokenIdToAtomType: token_id=" + std::to_string(token_id) +
                                 " is outside the atom token range");
    }

    const int type_index = isAtomOpenTokenId(token_id)
        ? token_id - ATOM_OPEN_TOKEN_OFFSET
        : token_id - ATOM_CLOSE_TOKEN_OFFSET;
    switch (type_index) {
        case static_cast<int>(AtomType::ATOM_INT):
            return AtomType::ATOM_INT;
        case static_cast<int>(AtomType::ATOM_FLOAT):
            return AtomType::ATOM_FLOAT;
        case static_cast<int>(AtomType::ATOM_STRING):
            return AtomType::ATOM_STRING;
        case static_cast<int>(AtomType::ATOM_BOOL):
            return AtomType::ATOM_BOOL;
        case static_cast<int>(AtomType::ATOM_ENTITY):
            return AtomType::ATOM_ENTITY;
        default:
            throw std::runtime_error("tokenIdToAtomType: token_id=" + std::to_string(token_id) +
                                     " does not map to a live atom type");
    }
}

inline const char* atomTypeName(AtomType type) {
    switch (type) {
        case AtomType::ATOM_INT:   return "INT";
        case AtomType::ATOM_FLOAT: return "FLOAT";
        case AtomType::ATOM_STRING: return "STRING";
        case AtomType::ATOM_BOOL:   return "BOOL";
        case AtomType::ATOM_ENTITY: return "ENTITY";
        default: return "UNKNOWN";
    }
}

inline std::string atomTokenText(int token_id) {
    const AtomType type = tokenIdToAtomType(token_id);
    const bool is_close = atomTokenBoundaryKind(token_id) == AtomBoundaryKind::CLOSE;
    return std::string(is_close ? "</" : "<") + atomTypeName(type) + ">";
}

inline bool isNumericAtom(AtomType type) {
    return type == AtomType::ATOM_INT || type == AtomType::ATOM_FLOAT;
}

} // namespace Tokenizer
} // namespace GRIM
