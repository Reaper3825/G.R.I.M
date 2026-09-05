#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace GRIM::GRMT {

// v25 row prefix: uint32 byte length followed by opaque source-ID bytes.
// Bound allocations before reading an untrusted/corrupt artifact.
inline constexpr std::uint32_t kMaxConceptBlockIdBytes = 65536;

inline void validateConceptBlockId(const std::string& id, const std::string& source) {
    if (id.empty() || id.size() > kMaxConceptBlockIdBytes)
        throw std::runtime_error("[GRMT] missing or oversized concept_block_id in " + source);
}

inline void writeConceptBlockId(std::ostream& output, const std::string& id, const std::string& sink) {
    validateConceptBlockId(id, sink);
    const auto length = static_cast<std::uint32_t>(id.size());
    output.write(reinterpret_cast<const char*>(&length), sizeof(length));
    output.write(id.data(), static_cast<std::streamsize>(length));
    if (!output) throw std::runtime_error("[GRMT] failed writing concept_block_id to " + sink);
}

inline std::string readConceptBlockId(std::istream& input, const std::string& source) {
    std::uint32_t length = 0;
    input.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (!input || length == 0 || length > kMaxConceptBlockIdBytes)
        throw std::runtime_error("[GRMT] invalid concept_block_id length in " + source);
    std::string id(length, '\0');
    input.read(id.data(), static_cast<std::streamsize>(length));
    if (!input) throw std::runtime_error("[GRMT] truncated concept_block_id in " + source);
    return id;
}

} // namespace GRIM::GRMT
