//======================================================//
//  DetectorRegistry.hpp
//  Registry + scanner for raw-text tokenizer detectors
//======================================================//

#pragma once

#include "TokenizerDetector.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

class DetectorRegistry {
public:
    DetectorRegistry() = default;
    DetectorRegistry(DetectorRegistry&&) noexcept = default;
    DetectorRegistry& operator=(DetectorRegistry&&) noexcept = default;

    DetectorRegistry(const DetectorRegistry&) = delete;
    DetectorRegistry& operator=(const DetectorRegistry&) = delete;

    void registerDetector(std::unique_ptr<RawTextDetector> detector);

    std::vector<RawTextDetection> scan(std::string_view text,
                                       const RawTextDetectorOptions& options) const;

    std::vector<StructuralSpan> detectStructures(std::string_view text,
                                                 const RawTextDetectorOptions& options) const;

    size_t size() const noexcept { return detectors_.size(); }
    bool empty() const noexcept { return detectors_.empty(); }

private:
    std::vector<std::unique_ptr<RawTextDetector>> detectors_;
};

DetectorRegistry makeDefaultRawTextDetectorRegistry();

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM