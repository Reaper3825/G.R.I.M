//======================================================//
//  TextFeatureDetectors.cu
//  Raw-text feature detector implementations
//======================================================//

#include "TextFeatureDetectors.hpp"

#include "../TextUtils.hpp"

namespace GRIM {
namespace Tokenizer {
namespace Detector {

namespace {

// Locale-independent ASCII uppercase check. std::isupper consults the process
// locale and can classify bytes >= 128 as uppercase under non-C locales,
// changing scan consumption (and therefore tokenization) across environments.
bool isAsciiUpper(char c) {
    return c >= 'A' && c <= 'Z';
}

} // namespace

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
    if (!isAsciiUpper(text[pos])) {
        return std::nullopt;
    }

    size_t end = pos + 1;
    while (end < text.size() && isAsciiUpper(text[end])) {
        ++end;
    }

    return RawTextDetection(pos, end, RawTextFeature::UPPERCASE_RUN, name());
}

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM