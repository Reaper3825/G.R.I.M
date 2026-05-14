//======================================================//
//  Byte.hpp
//  Byte-level fallback tokenizer for GRIM
//  
//  When Unigram LM fails to segment a piece, we fall back
//  to raw byte encoding. This ensures 100% coverage of any
//  UTF-8 input including unknown characters, emojis, etc.
//  
//  Token ID layout:
//    [0-3]     = Special tokens (<unk>, <pad>, <s>, </s>)
//    [4-259]   = Raw byte tokens (0x00 to 0xFF)
//    [260+]    = Reserved for Atom + Unigram vocabulary
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Constants
//======================================================//
constexpr int NUM_SPECIAL_TOKENS = 4;          // <unk>=0, <pad>=1, <s>=2, </s>=3
constexpr int SPECIAL_TOKEN_OFFSET = 0;        // Special tokens start at ID 0
constexpr int BYTE_VOCAB_SIZE = 256;           // 0x00 - 0xFF
constexpr int BYTE_TOKEN_OFFSET = NUM_SPECIAL_TOKENS;  // Byte tokens start at ID 4 (after specials)

// Absolute special token IDs
constexpr int UNK_TOKEN_ID = 0;
constexpr int PAD_TOKEN_ID = 1;
constexpr int BOS_TOKEN_ID = 2;
constexpr int EOS_TOKEN_ID = 3;

//======================================================//
//  Byte Token Info
//======================================================//
struct ByteToken {
    uint8_t byte_value;
    int token_id;
    bool is_continuation;  // UTF-8 continuation byte (10xxxxxx)
    bool is_printable;
};

//======================================================//
//  ByteEncoder - byte-level tokenizer primitive
//======================================================//
class ByteEncoder {
public:
    ByteEncoder();
    ~ByteEncoder();

    // Disable copy
    ByteEncoder(const ByteEncoder&) = delete;
    ByteEncoder& operator=(const ByteEncoder&) = delete;

    // Move support
    ByteEncoder(ByteEncoder&&) noexcept;
    ByteEncoder& operator=(ByteEncoder&&) noexcept;

    //--------------------------------------------------//
    // CPU Interface
    //--------------------------------------------------//
    
    // Encode raw bytes to token IDs
    std::vector<int> encode(const std::string& text) const;
    
    // Decode token IDs back to bytes
    std::string decode(const std::vector<int>& token_ids) const;
    
    // Check if a token ID is a byte token
    bool isByteToken(int token_id) const;
    
    // Get byte value from token ID
    uint8_t tokenToByte(int token_id) const;
    
    // Get token ID from byte value
    int byteToToken(uint8_t byte_value) const;
    
    // Get info about a byte
    ByteToken getByteInfo(uint8_t byte_value) const;

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    int byteTokenCount() const { return BYTE_VOCAB_SIZE; }
    int tokenOffset() const { return BYTE_TOKEN_OFFSET; }
    
    // Get display string for a byte token (for debugging)
    std::string tokenToString(int token_id) const;

    // No private device-side lookup tables: encode is a pure offset add and
    // decode is a pure offset subtract, so the kernels in Byte.cu need no
    // precomputed table. (Earlier versions allocated token<->byte tables that
    // were never read by any kernel; they were removed as dead state per the
    // project's delete-dead-code rule.)
};

//======================================================//
//  UTF-8 Utilities
//======================================================//
namespace UTF8 {
    // Check if byte is UTF-8 start byte
    inline bool isStartByte(uint8_t b) {
        return (b & 0xC0) != 0x80;  // Not 10xxxxxx
    }
    
    // Check if byte is UTF-8 continuation byte
    inline bool isContinuation(uint8_t b) {
        return (b & 0xC0) == 0x80;  // 10xxxxxx
    }
    
    // Get expected sequence length from start byte
    inline int sequenceLength(uint8_t start_byte) {
        if ((start_byte & 0x80) == 0x00) return 1;      // 0xxxxxxx
        if ((start_byte & 0xE0) == 0xC0) return 2;      // 110xxxxx
        if ((start_byte & 0xF0) == 0xE0) return 3;      // 1110xxxx
        if ((start_byte & 0xF8) == 0xF0) return 4;      // 11110xxx
        return 1;  // Invalid, treat as single byte
    }
    
    // Check if byte is printable ASCII
    inline bool isPrintable(uint8_t b) {
        return b >= 0x20 && b < 0x7F;
    }
}

} // namespace Tokenizer
} // namespace GRIM
