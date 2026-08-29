//======================================================//
//  AtomDelimiterDetector.cu
//  Detector for authored typed atom spans
//======================================================//

#include "AtomDelimiterDetector.hpp"

#include "../TextUtils.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

namespace {

bool startsWithAt(std::string_view text, size_t pos, std::string_view value) {
    return pos <= text.size() && value.size() <= text.size() - pos &&
           text.compare(pos, value.size(), value) == 0;
}

} // namespace

std::optional<RawTextDetection> AtomDelimiterDetector::detect(
    std::string_view text,
    size_t pos) const {
    if (pos >= text.size() || text[pos] != '<') {
        return std::nullopt;
    }

    std::string_view open;
    std::string_view close;
    AtomType atom_type = AtomType::ATOM_INT;
    if (startsWithAt(text, pos, "<INT>")) {
        open = "<INT>";
        close = "</INT>";
        atom_type = AtomType::ATOM_INT;
    } else if (startsWithAt(text, pos, "<FLOAT>")) {
        open = "<FLOAT>";
        close = "</FLOAT>";
        atom_type = AtomType::ATOM_FLOAT;
    } else if (startsWithAt(text, pos, "<STRING>")) {
        open = "<STRING>";
        close = "</STRING>";
        atom_type = AtomType::ATOM_STRING;
    } else if (startsWithAt(text, pos, "<BOOL>")) {
        open = "<BOOL>";
        close = "</BOOL>";
        atom_type = AtomType::ATOM_BOOL;
    } else if (startsWithAt(text, pos, "<ENTITY>")) {
        open = "<ENTITY>";
        close = "</ENTITY>";
        atom_type = AtomType::ATOM_ENTITY;
    } else {
        return std::nullopt;
    }

    const size_t inner_begin = pos + open.size();
    const size_t close_begin = text.find(close, inner_begin);
    if (close_begin == std::string_view::npos) {
        throw std::runtime_error(
            std::string("AtomDelimiterDetector: authored ") +
            std::string(open) + " span at byte " +
            std::to_string(pos) + " has no matching " +
            std::string(close));
    }

    size_t content_begin = inner_begin;
    size_t content_end = close_begin;
    if (atom_type != AtomType::ATOM_STRING &&
        atom_type != AtomType::ATOM_ENTITY) {
        while (content_begin < content_end &&
               isWhitespaceASCII(static_cast<unsigned char>(text[content_begin]))) {
            ++content_begin;
        }
        while (content_end > content_begin &&
               isWhitespaceASCII(static_cast<unsigned char>(text[content_end - 1]))) {
            --content_end;
        }
    }
    if (content_begin == content_end && atom_type != AtomType::ATOM_STRING) {
        throw std::runtime_error(
            std::string("AtomDelimiterDetector: authored ") +
            std::string(open) + " span at byte " +
            std::to_string(pos) + " has empty content");
    }

    const size_t span_end = close_begin + close.size();
    if (pos > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        span_end - pos > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        content_begin > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        content_end - content_begin > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(
            "AtomDelimiterDetector: authored atom span exceeds StructuralSpan capacity");
    }

    StructuralSpan span;
    span.start = pos;
    span.end = span_end;
    span.atom_type = atom_type;
    span.buffer_ptr = text.data();
    span.offset = static_cast<uint32_t>(pos);
    span.length = static_cast<uint32_t>(span_end - pos);
    span.content_offset = static_cast<uint32_t>(content_begin);
    span.content_length = static_cast<uint32_t>(content_end - content_begin);
    return RawTextDetection(span, name());
}

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM
