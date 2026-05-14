//======================================================//
//  Byte.cu
//  CUDA implementation of byte-level fallback tokenizer
//======================================================//

#include "Byte.hpp"

#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  ByteEncoder Implementation
//======================================================//

ByteEncoder::ByteEncoder() = default;
ByteEncoder::~ByteEncoder() = default;
ByteEncoder::ByteEncoder(ByteEncoder&&) noexcept = default;
ByteEncoder& ByteEncoder::operator=(ByteEncoder&&) noexcept = default;

//--------------------------------------------------//
// CPU Interface
//--------------------------------------------------//

std::vector<int> ByteEncoder::encode(const std::string& text) const {
    std::vector<int> result(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        result[i] = static_cast<int>(static_cast<uint8_t>(text[i])) + BYTE_TOKEN_OFFSET;
    }
    return result;
}

std::string ByteEncoder::decode(const std::vector<int>& token_ids) const {
    std::string result;
    result.reserve(token_ids.size());
    for (int tid : token_ids) {
        if (!isByteToken(tid)) {
            throw std::runtime_error("ByteEncoder::decode: token_id=" + std::to_string(tid) +
                                     " is not a byte token");
        }
        result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
    }
    return result;
}

bool ByteEncoder::isByteToken(int token_id) const {
    return token_id >= BYTE_TOKEN_OFFSET && token_id < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;
}

uint8_t ByteEncoder::tokenToByte(int token_id) const {
    if (!isByteToken(token_id)) return 0;
    return static_cast<uint8_t>(token_id - BYTE_TOKEN_OFFSET);
}

int ByteEncoder::byteToToken(uint8_t byte_value) const {
    return static_cast<int>(byte_value) + BYTE_TOKEN_OFFSET;
}

ByteToken ByteEncoder::getByteInfo(uint8_t byte_value) const {
    ByteToken info;
    info.byte_value = byte_value;
    info.token_id = byteToToken(byte_value);
    info.is_continuation = UTF8::isContinuation(byte_value);
    info.is_printable = UTF8::isPrintable(byte_value);
    return info;
}

std::string ByteEncoder::tokenToString(int token_id) const {
    if (!isByteToken(token_id)) {
        return "<non-byte>";
    }
    
    uint8_t byte_val = tokenToByte(token_id);
    std::ostringstream oss;
    
    // Whitespace specials must come before the printable branch: ' ' is
    // printable per UTF8::isPrintable, and the other whitespace bytes are not
    // printable but reach here as control bytes.
    if (byte_val == ' ') {
        oss << "<SP>";
    } else if (byte_val == '\n') {
        oss << "<LF>";
    } else if (byte_val == '\r') {
        oss << "<CR>";
    } else if (byte_val == '\t') {
        oss << "<TAB>";
    } else if (UTF8::isPrintable(byte_val)) {
        oss << "'" << static_cast<char>(byte_val) << "'";
    } else {
        oss << "<0x" << std::hex << std::setw(2) << std::setfill('0') 
            << static_cast<int>(byte_val) << ">";
    }
    
    return oss.str();
}

} // namespace Tokenizer
} // namespace GRIM
