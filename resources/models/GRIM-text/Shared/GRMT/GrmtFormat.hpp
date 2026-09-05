#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM {

// Current GRMT training tensor stream version. Version 24 invalidates SFT
// artifacts authored before answer-only target masking.
//
// v23 adds top-level ConceptBlock known/unknown token IDs and one invisible
// logical token span per ordered entry. These collections remain independent
// of row-level Goal metadata.
//
// v22 adds the independent sequence-local typed atom table and the opening-only
// local index channel. Local addresses are (AtomType, local_index) and do not
// reuse durable AtomTable entry IDs.
//
// v21 changes numeric atoms from one metadata-bearing placeholder token to a
// typed boundary span. The serialized row shape is unchanged: atom metadata is
// carried only at the opening boundary, while content and closing boundaries
// have empty side channels.
//
// v20 adds ordered constraint token IDs and one invisible logical token span
// per constraint. There is intentionally no collection-wide constraint span.
//
// v19 adds invisible logical token spans for target state, the full criteria
// collection, and each ordered criterion/evidence pair. Evidence spans may be
// absent for criteria awaiting evidence generation. The prompt is pinned ahead
// of the goal decomposition; delimiter strings never enter model-visible tokens.
//
// v18 adds row-level Goal metadata: target-state tokens and ordered,
// evidence-paired success-criterion tokens.
//
// v17 replaced arithmetic-specific teacher records with variable-arity
// TransitionInvocation targets. Opaque uint64 TransitionId values are lowered
// through per-row CompiledTransitionBinding tables. Argument and result payload
// values remain outside transition metadata.
inline constexpr std::uint32_t GRMT_FORMAT_VERSION = 24;

} // namespace GRIM

namespace GRIM::GRMT {

inline constexpr std::uint32_t kMagic = 0x474D5254u; // ASCII bytes: GRMT
inline constexpr std::size_t kHeaderSizeBytes = sizeof(std::uint32_t) * 4;

struct Header {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t num_sequences = 0;
    std::uint32_t vocab_size = 0;
};

static_assert(sizeof(Header) == kHeaderSizeBytes,
              "GRMT header must remain four contiguous uint32 fields");

struct HeaderValidation {
    bool require_current_version = true;
    bool require_nonzero_sequences = true;
    bool require_nonzero_vocab_size = true;
};

struct HeaderReadStatus {
    bool ok = false;
    Header header{};
    std::string error;
};

inline std::string hex32(std::uint32_t value) {
    std::ostringstream oss;
    oss << "0x" << std::uppercase << std::hex
        << std::setw(8) << std::setfill('0') << value;
    return oss.str();
}

inline HeaderReadStatus readRawHeaderStatus(std::istream& input, const std::string& source) {
    HeaderReadStatus status{};
    input.read(reinterpret_cast<char*>(&status.header), static_cast<std::streamsize>(sizeof(Header)));
    if (!input) {
        status.error = "[GRMT] header read failed or truncated: " + source +
                       " (expected " + std::to_string(kHeaderSizeBytes) + " bytes)";
        return status;
    }
    status.ok = true;
    return status;
}

inline Header readRawHeaderOrThrow(std::istream& input, const std::string& source) {
    const HeaderReadStatus status = readRawHeaderStatus(input, source);
    if (!status.ok) {
        throw std::runtime_error(status.error);
    }
    return status.header;
}

inline std::string headerValidationError(
    const Header& header,
    const std::string& source,
    const HeaderValidation& validation = HeaderValidation{})
{
    if (header.magic != kMagic) {
        return "[GRMT] invalid magic in " + source + ": actual=" + hex32(header.magic) +
               " expected='GRMT'/" + hex32(kMagic);
    }
    if (validation.require_current_version && header.version != GRIM::GRMT_FORMAT_VERSION) {
        return "[GRMT] version mismatch in " + source + ": file=" +
               std::to_string(header.version) + " expected=" +
               std::to_string(GRIM::GRMT_FORMAT_VERSION);
    }
    if (validation.require_nonzero_sequences && header.num_sequences == 0) {
        return "[GRMT] header reports num_sequences=0: " + source;
    }
    if (validation.require_nonzero_vocab_size && header.vocab_size == 0) {
        return "[GRMT] header reports vocab_size=0: " + source;
    }
    return {};
}

inline void validateHeaderOrThrow(
    const Header& header,
    const std::string& source,
    const HeaderValidation& validation = HeaderValidation{})
{
    const std::string error = headerValidationError(header, source, validation);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

inline HeaderReadStatus readHeaderStatus(
    const std::string& path,
    const HeaderValidation& validation = HeaderValidation{})
{
    HeaderReadStatus status{};
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        status.error = "[GRMT] cannot open file for header read: " + path;
        return status;
    }

    const HeaderReadStatus raw_status = readRawHeaderStatus(input, path);
    if (!raw_status.ok) {
        return raw_status;
    }
    status.header = raw_status.header;

    status.error = headerValidationError(status.header, path, validation);
    if (!status.error.empty()) {
        return status;
    }
    status.ok = true;
    return status;
}

inline Header readHeaderOrThrow(
    std::istream& input,
    const std::string& source,
    const HeaderValidation& validation = HeaderValidation{})
{
    Header header = readRawHeaderOrThrow(input, source);
    validateHeaderOrThrow(header, source, validation);
    return header;
}

inline Header readHeaderOrThrow(
    const std::string& path,
    const HeaderValidation& validation = HeaderValidation{})
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("[GRMT] cannot open file for header read: " + path);
    }
    return readHeaderOrThrow(input, path, validation);
}

inline Header makeCurrentHeader(std::uint32_t num_sequences, std::uint32_t vocab_size) {
    Header header{};
    header.magic = kMagic;
    header.version = GRIM::GRMT_FORMAT_VERSION;
    header.num_sequences = num_sequences;
    header.vocab_size = vocab_size;
    validateHeaderOrThrow(header, "new GRMT header");
    return header;
}

inline void writeHeaderOrThrow(
    std::ostream& output,
    const Header& header,
    const std::string& sink,
    const HeaderValidation& validation = HeaderValidation{})
{
    validateHeaderOrThrow(header, sink, validation);
    output.write(reinterpret_cast<const char*>(&header), static_cast<std::streamsize>(sizeof(Header)));
    if (!output) {
        throw std::runtime_error("[GRMT] failed to write header: " + sink);
    }
}

} // namespace GRIM::GRMT
