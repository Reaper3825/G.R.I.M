//======================================================//
//  TokenizerDetector.hpp
//  Parent interface for raw-text tokenizer detectors
//
//  Detectors operate on raw text byte offsets only. They do not inspect,
//  create, or classify token IDs.
//======================================================//

#pragma once

#include "../TokenLayout.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Registry-owned structural detection result
//======================================================//
struct StructuralSpan {
    size_t start;           // Start position in text (may include leading whitespace)
    size_t end;             // End position (exclusive)
    AtomType atom_type;     // Type of structure detected
    uint32_t atom_entry_id = kAtomEntryNone; // Per-sequence AtomTable entry ID once registered

    // Zero-copy buffer reference (NO std::string allocation!)
    const char* buffer_ptr; // Pointer to original text buffer
    uint32_t offset;        // Offset in buffer
    uint32_t length;        // Length of span (end - start)

    // Content bounds (same as offset/length since no widening)
    uint32_t content_offset; // Offset to content
    uint32_t content_length; // Length of content

    int placeholder_id;     // Token ID of placeholder

    // Helper: get string_view of full span (may include leading whitespace)
    std::string_view view() const {
        return std::string_view(buffer_ptr + offset, length);
    }

    // Helper: get string_view of just the atom content (no whitespace)
    std::string_view contentView() const {
        return std::string_view(buffer_ptr + content_offset, content_length);
    }
};

namespace Detector {

enum class RawTextFeature : uint8_t {
    ATOM,
    WHITESPACE,
    UPPERCASE_RUN
};
 
struct RawTextDetectorOptions {
    bool detect_numbers;
    bool detect_whitespace;
    bool detect_uppercase;

    RawTextDetectorOptions(bool numbers, bool whitespace, bool uppercase) noexcept
        : detect_numbers(numbers)
        , detect_whitespace(whitespace)
        , detect_uppercase(uppercase) {}
};

struct RawTextDetection {
    size_t start;
    size_t end;
    RawTextFeature feature;
    AtomType atom_type;
    const char* detector_name;

    RawTextDetection(size_t start_in,
                     size_t end_in,
                     RawTextFeature feature_in,
                     AtomType atom_type_in,
                     const char* detector_name_in) noexcept
        : start(start_in)
        , end(end_in)
        , feature(feature_in)
        , atom_type(atom_type_in)
        , detector_name(detector_name_in) {}

    bool emitsAtom() const noexcept { return atom_type != AtomType::ATOM_NONE; }
    size_t length() const noexcept { return end - start; }
};

class RawTextDetector {
public:
    virtual ~RawTextDetector() = default;

    virtual const char* name() const noexcept = 0;
    virtual int priority() const noexcept = 0;
    virtual bool enabled(const RawTextDetectorOptions& options) const noexcept = 0;
    virtual std::optional<RawTextDetection> detect(std::string_view text,
                                                   size_t pos) const = 0;
};

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM