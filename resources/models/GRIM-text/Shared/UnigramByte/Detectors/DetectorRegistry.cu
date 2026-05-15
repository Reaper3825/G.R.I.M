//======================================================//
//  DetectorRegistry.cu
//  Raw-text detector registry implementation
//======================================================//

#include "DetectorRegistry.hpp"

#include "NumericDetectors.hpp"
#include "TextFeatureDetectors.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

void DetectorRegistry::registerDetector(std::unique_ptr<RawTextDetector> detector) {
    if (!detector) {
        throw std::runtime_error("DetectorRegistry::registerDetector received NULL detector");
    }
    const char* name = detector->name();
    if (!name || name[0] == '\0') {
        throw std::runtime_error("DetectorRegistry::registerDetector received detector with empty name");
    }
    for (const auto& existing : detectors_) {
        if (std::string(existing->name()) == name) {
            throw std::runtime_error(
                "DetectorRegistry::registerDetector duplicate detector name: " +
                std::string(name));
        }
    }

    detectors_.push_back(std::move(detector));
    std::sort(detectors_.begin(), detectors_.end(),
              [](const auto& a, const auto& b) {
                  return a->priority() > b->priority();
              });
}

std::vector<RawTextDetection> DetectorRegistry::scan(
    std::string_view text,
    const RawTextDetectorOptions& options) const
{
    if (detectors_.empty()) {
        throw std::runtime_error("DetectorRegistry::scan called with zero registered detectors");
    }

    std::vector<RawTextDetection> detections;
    detections.reserve(32);

    for (size_t pos = 0; pos < text.size(); ) {
        std::optional<RawTextDetection> best;
        int best_priority = 0;

        for (const auto& detector : detectors_) {
            if (!detector->enabled(options)) {
                continue;
            }

            auto candidate = detector->detect(text, pos);
            if (!candidate.has_value()) {
                continue;
            }

            if (candidate->start != pos) {
                throw std::runtime_error(
                    std::string("Detector '") + detector->name() +
                    "' returned start=" + std::to_string(candidate->start) +
                    " for scan position=" + std::to_string(pos));
            }
            if (candidate->end <= pos || candidate->end > text.size()) {
                throw std::runtime_error(
                    std::string("Detector '") + detector->name() +
                    "' returned invalid end=" + std::to_string(candidate->end) +
                    " for text size=" + std::to_string(text.size()));
            }
            if (!candidate->detector_name || candidate->detector_name[0] == '\0') {
                throw std::runtime_error(
                    std::string("Detector '") + detector->name() +
                    "' returned detection with empty detector_name");
            }

            const size_t candidate_len = candidate->length();
            const size_t best_len = best.has_value() ? best->length() : 0;
            const int candidate_priority = detector->priority();
            if (!best.has_value() ||
                candidate_len > best_len ||
                (candidate_len == best_len && candidate_priority > best_priority)) {
                best = candidate;
                best_priority = candidate_priority;
            }
        }

        if (best.has_value()) {
            detections.push_back(*best);
            pos = best->end;
            continue;
        }

        ++pos;
    }

    return detections;
}

std::vector<StructuralSpan> DetectorRegistry::detectStructures(
    std::string_view text,
    const RawTextDetectorOptions& options) const
{
    auto makeSpan = [text](size_t start, size_t end, AtomType type) -> StructuralSpan {
        StructuralSpan span;
        span.start = start;
        span.end = end;
        span.atom_type = type;
        span.buffer_ptr = text.data();
        span.offset = static_cast<uint32_t>(start);
        span.length = static_cast<uint32_t>(end - start);
        span.content_offset = static_cast<uint32_t>(start);
        span.content_length = static_cast<uint32_t>(end - start);
        span.placeholder_id = atomTypeToTokenId(type);
        return span;
    };

    std::vector<StructuralSpan> spans;
    spans.reserve(32);

    const auto detections = scan(text, options);
    for (const auto& detection : detections) {
        if (!detection.emitsAtom()) {
            continue;
        }
        spans.push_back(makeSpan(detection.start, detection.end, detection.atom_type));
    }

    return spans;
}

DetectorRegistry makeDefaultRawTextDetectorRegistry() {
    DetectorRegistry registry;
    registry.registerDetector(std::make_unique<FloatDetector>());
    registry.registerDetector(std::make_unique<IntegerDetector>());
    registry.registerDetector(std::make_unique<WhitespaceDetector>());
    registry.registerDetector(std::make_unique<UppercaseRunDetector>());
    return registry;
}

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM