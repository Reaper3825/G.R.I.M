//======================================================//
//  TokenizerDetector.hpp
//  Parent interface for raw-text tokenizer detectors
//
//  Detectors operate on raw text byte offsets only. They do not inspect,
//  create, or classify token IDs.
//======================================================//

#pragma once

#include "StructuralSpan.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace GRIM {
namespace Tokenizer {

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

// Detector-only metadata decorating the canonical StructuralSpan shape.
// Span geometry and atom content are never mirrored here.
struct RawTextDetection : StructuralSpan {
    RawTextFeature feature;
    const char* detector_name;

    RawTextDetection(size_t start_in,
                     size_t end_in,
                     AtomType atom_type_in,
                     const char* detector_name_in) noexcept
        : feature(RawTextFeature::ATOM)
        , detector_name(detector_name_in) {
        start = start_in;
        end = end_in;
        atom_type = atom_type_in;
        offset = static_cast<uint32_t>(start_in);
        StructuralSpan::length = static_cast<uint32_t>(end_in - start_in);
        content_offset = offset;
        content_length = StructuralSpan::length;
    }

    RawTextDetection(const StructuralSpan& span,
                     const char* detector_name_in) noexcept
        : StructuralSpan(span)
        , feature(RawTextFeature::ATOM)
        , detector_name(detector_name_in) {}

    RawTextDetection(size_t start_in,
                     size_t end_in,
                     RawTextFeature feature_in,
                     const char* detector_name_in) noexcept
        : feature(feature_in)
        , detector_name(detector_name_in) {
        start = start_in;
        end = end_in;
        offset = static_cast<uint32_t>(start_in);
        StructuralSpan::length = static_cast<uint32_t>(end_in - start_in);
        content_offset = offset;
        content_length = StructuralSpan::length;
    }

    bool emitsAtom() const noexcept { return feature == RawTextFeature::ATOM; }
    size_t byteLength() const noexcept { return end - start; }
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
