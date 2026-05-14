//======================================================//
//  TextFeatureDetectors.hpp
//  Raw-text feature detectors that do not emit atom tokens
//======================================================//

#pragma once

#include "TokenizerDetector.hpp"

namespace GRIM {
namespace Tokenizer {
namespace Detector {

class WhitespaceDetector final : public RawTextDetector {
public:
    const char* name() const noexcept override { return "whitespace"; }
    int priority() const noexcept override { return 20; }
    bool enabled(const RawTextDetectorOptions& options) const noexcept override {
        return options.detect_whitespace;
    }
    std::optional<RawTextDetection> detect(std::string_view text,
                                           size_t pos) const override;
};

class UppercaseRunDetector final : public RawTextDetector {
public:
    const char* name() const noexcept override { return "uppercase_run"; }
    int priority() const noexcept override { return 10; }
    bool enabled(const RawTextDetectorOptions& options) const noexcept override {
        return options.detect_uppercase;
    }
    std::optional<RawTextDetection> detect(std::string_view text,
                                           size_t pos) const override;
};

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM