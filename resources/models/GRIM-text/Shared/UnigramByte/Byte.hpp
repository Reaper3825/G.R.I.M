//======================================================//
//  Byte.hpp
//  Byte-level fallback tokenizer for GRIM
//  
//  When Unigram LM fails to segment a piece, we fall back
//  to raw byte encoding. This ensures 100% coverage of any
//  UTF-8 input including unknown characters, emojis, etc.
//  
//  Token ID layout:
//    [0-255]   = Raw byte tokens (0x00 to 0xFF)
//    [256+]    = Reserved for Unigram vocabulary
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Constants
//======================================================//
constexpr int BYTE_VOCAB_SIZE = 256;           // 0x00 - 0xFF
constexpr int BYTE_TOKEN_OFFSET = 0;           // Byte tokens start at ID 0
constexpr int UNIGRAM_TOKEN_OFFSET = 256;      // Unigram tokens start after bytes

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
//  ByteEncoder - CPU/GPU byte-level tokenizer
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
    std::vector<int> encode(const uint8_t* data, size_t length) const;
    
    // Decode token IDs back to bytes
    std::string decode(const std::vector<int>& token_ids) const;
    std::string decode(const int* token_ids, size_t count) const;
    
    // Check if a token ID is a byte token
    bool isByteToken(int token_id) const;
    
    // Get byte value from token ID
    uint8_t tokenToByte(int token_id) const;
    
    // Get token ID from byte value
    int byteToToken(uint8_t byte_value) const;
    
    // Get info about a byte
    ByteToken getByteInfo(uint8_t byte_value) const;

    //--------------------------------------------------//
    // GPU Interface
    //--------------------------------------------------//
    
    // Encode bytes on GPU (text must already be in device memory)
    // d_input: input bytes on device
    // d_output: output token IDs on device (must be pre-allocated)
    // length: number of bytes
    // stream: CUDA stream for async execution
    bool encodeGPU(const uint8_t* d_input, 
                   int* d_output, 
                   size_t length,
                   cudaStream_t stream = nullptr);
    
    // Decode tokens on GPU
    bool decodeGPU(const int* d_input,
                   uint8_t* d_output,
                   size_t count,
                   cudaStream_t stream = nullptr);
    
    // Batch encode multiple sequences
    bool encodeBatchGPU(const uint8_t* const* d_inputs,
                        const size_t* lengths,
                        int** d_outputs,
                        size_t batch_size,
                        cudaStream_t stream = nullptr);

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    int vocabSize() const { return BYTE_VOCAB_SIZE; }
    int tokenOffset() const { return BYTE_TOKEN_OFFSET; }
    
    // Get display string for a byte token (for debugging)
    std::string tokenToString(int token_id) const;

private:
    // Lookup tables (device pointers)
    uint8_t* d_token_to_byte_;    // [256] token_id -> byte value
    int* d_byte_to_token_;        // [256] byte value -> token_id
    bool* d_is_continuation_;     // [256] is UTF-8 continuation byte
    
    bool gpu_initialized_;
    
    void initGPU();
    void releaseGPU();
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
