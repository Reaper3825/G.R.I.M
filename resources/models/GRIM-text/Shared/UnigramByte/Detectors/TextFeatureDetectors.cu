//======================================================//
//  TextFeatureDetectors.cu
//  Raw-text feature detector implementations
//======================================================//

#include "TextFeatureDetectors.hpp"

#include "../TextUtils.hpp"

#include <cctype>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

std::optional<RawTextDetection> WhitespaceDetector::detect(std::string_view text,
                                                           size_t pos) const {
    if (pos >= text.size()) return std::nullopt;
    if (!isWhitespaceASCII(static_cast<unsigned char>(text[pos]))) {
        return std::nullopt;
    }

    size_t end = pos + 1;
    while (end < text.size() && isWhitespaceASCII(static_cast<unsigned char>(text[end]))) {
        ++end;
    }

    return RawTextDetection(pos, end, RawTextFeature::WHITESPACE, name());
}

std::optional<RawTextDetection> UppercaseRunDetector::detect(std::string_view text,
                                                             size_t pos) const {
    if (pos >= text.size()) return std::nullopt;
    if (std::isupper(static_cast<unsigned char>(text[pos])) == 0) {
        return std::nullopt;
    }

    size_t end = pos + 1;
    while (end < text.size() && std::isupper(static_cast<unsigned char>(text[end])) != 0) {
        ++end;
    }

    return RawTextDetection(pos, end, RawTextFeature::UPPERCASE_RUN, name());
}

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM