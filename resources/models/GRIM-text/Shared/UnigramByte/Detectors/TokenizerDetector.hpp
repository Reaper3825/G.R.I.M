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