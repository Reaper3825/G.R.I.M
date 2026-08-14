//======================================================//
//  AtomDelimiterDetector.hpp
//  Detector for authored typed atom spans
//======================================================//

#pragma once

#include "TokenizerDetector.hpp"

namespace GRIM {
namespace Tokenizer {
namespace Detector {

class AtomDelimiterDetector final : public RawTextDetector {
public:
    const char* name() const noexcept override { return "atom_delimiter"; }
    int priority() const noexcept override { return 300; }
    bool enabled(const RawTextDetectorOptions&) const noexcept override {
        return true;
    }
    std::optional<RawTextDetection> detect(std::string_view text,
                                           size_t pos) const override;
};

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM
