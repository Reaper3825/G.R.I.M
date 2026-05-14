//======================================================//
//  NumericDetectors.hpp
//  Raw-text numeric detectors for tokenizer atom spans
//======================================================//

#pragma once

#include "TokenizerDetector.hpp"

namespace GRIM {
namespace Tokenizer {
namespace Detector {

class FloatDetector final : public RawTextDetector {
public:
    const char* name() const noexcept override { return "float"; }
    int priority() const noexcept override { return 200; }
    bool enabled(const RawTextDetectorOptions& options) const noexcept override {
        return options.detect_numbers;
    }
    std::optional<RawTextDetection> detect(std::string_view text,
                                           size_t pos) const override;
};

class IntegerDetector final : public RawTextDetector {
public:
    const char* name() const noexcept override { return "integer"; }
    int priority() const noexcept override { return 100; }
    bool enabled(const RawTextDetectorOptions& options) const noexcept override {
        return options.detect_numbers;
    }
    std::optional<RawTextDetection> detect(std::string_view text,
                                           size_t pos) const override;
};

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM