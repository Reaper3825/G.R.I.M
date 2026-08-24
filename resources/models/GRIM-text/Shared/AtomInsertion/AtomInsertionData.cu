//======================================================//
//  AtomInsertionData.cu
//  Byte-gap supervision for the inference atom inserter
//======================================================//

#include "AtomInsertionData.hpp"

#include "../UnigramByte/AtomTable.hpp"
#include "../UnigramByte/Detectors/DetectorRegistry.hpp"
#include "../UnigramByte/TextUtils.hpp"

#include <stdexcept>
#include <string>

namespace GRIM::AtomInsertion {

namespace {

std::string requireCaller(const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error(
            "AtomInsertionData requires a non-empty caller label");
    }
    return std::string(caller) + ": AtomInsertionData";
}

std::vector<std::uint8_t> buildValidUtf8GapMask(
    const std::string& bytes,
    const std::string& prefix) {
    std::vector<std::uint8_t> mask(bytes.size() + 1, 0);
    mask[0] = 1;

    std::size_t position = 0;
    while (position < bytes.size()) {
        std::uint32_t codepoint = 0;
        std::size_t sequence_length = 0;
        if (!Tokenizer::utf8DecodeAt(
                bytes, position, &codepoint, &sequence_length) ||
            sequence_length == 0 || sequence_length > bytes.size() - position) {
            throw std::runtime_error(
                prefix + ": de-annotated input is not well-formed UTF-8 at byte=" +
                std::to_string(position));
        }
        position += sequence_length;
        mask[position] = 1;
    }
    return mask;
}

void appendExactBytes(std::string_view source,
                      std::size_t begin,
                      std::size_t end,
                      std::string& destination,
                      const std::string& prefix) {
    if (begin > end || end > source.size()) {
        throw std::runtime_error(
            prefix + ": source byte range [" + std::to_string(begin) + "," +
            std::to_string(end) + ") is out of bounds for size=" +
            std::to_string(source.size()));
    }
    destination.append(source.data() + begin, end - begin);
}

void setDelimiterTarget(std::vector<GapDelimiterTargets>& targets,
                        std::size_t gap,
                        int delimiter_token_id,
                        const std::string& prefix) {
    if (gap >= targets.size()) {
        throw std::runtime_error(
            prefix + ": delimiter target gap=" + std::to_string(gap) +
            " is outside gap count=" + std::to_string(targets.size()));
    }
    if (!Tokenizer::isAtomTokenId(delimiter_token_id)) {
        throw std::runtime_error(
            prefix + ": token_id=" + std::to_string(delimiter_token_id) +
            " is not an atom delimiter token");
    }
    const std::size_t column = static_cast<std::size_t>(
        delimiter_token_id - Tokenizer::ATOM_TOKEN_OFFSET);
    if (targets[gap][column] != 0) {
        throw std::runtime_error(
            prefix + ": duplicate delimiter target token_id=" +
            std::to_string(delimiter_token_id) + " at gap=" +
            std::to_string(gap));
    }
    targets[gap][column] = 1;
}

void validateSpanContent(std::string_view annotated_source,
                         const Tokenizer::Detector::RawTextDetection& detection,
                         const std::string& prefix) {
    const std::size_t content_begin =
        static_cast<std::size_t>(detection.content_offset);
    const std::size_t content_end = content_begin +
        static_cast<std::size_t>(detection.content_length);
    if (content_begin > content_end || content_end > annotated_source.size()) {
        throw std::runtime_error(
            prefix + ": detector returned out-of-range atom content [" +
            std::to_string(content_begin) + "," +
            std::to_string(content_end) + ")");
    }

    const Tokenizer::ParseResult parsed = Tokenizer::AtomTable::parseAtom(
        detection.atom_type,
        annotated_source.substr(content_begin, content_end - content_begin));
    if (!parsed.success) {
        throw std::runtime_error(
            prefix + ": authored " +
            std::string(Tokenizer::atomTypeName(detection.atom_type)) +
            " content at [" + std::to_string(content_begin) + "," +
            std::to_string(content_end) + ") is invalid: " +
            parsed.error_message);
    }
}

} // namespace

std::size_t delimiterClassIndexOrThrow(int delimiter_token_id,
                                       const char* caller) {
    const std::string prefix = requireCaller(caller);
    if (!Tokenizer::isAtomTokenId(delimiter_token_id)) {
        throw std::runtime_error(
            prefix + ": token_id=" + std::to_string(delimiter_token_id) +
            " is not an atom delimiter token");
    }
    return static_cast<std::size_t>(
        delimiter_token_id - Tokenizer::ATOM_TOKEN_OFFSET);
}

int delimiterTokenIdForClassOrThrow(std::size_t delimiter_class,
                                    const char* caller) {
    const std::string prefix = requireCaller(caller);
    if (delimiter_class >= kDelimiterClassCount) {
        throw std::runtime_error(
            prefix + ": delimiter_class=" +
            std::to_string(delimiter_class) + " is outside [0," +
            std::to_string(kDelimiterClassCount) + ")");
    }
    return Tokenizer::ATOM_TOKEN_OFFSET +
        static_cast<int>(delimiter_class);
}

bool AtomInsertionExample::hasDelimiterTarget(
    std::size_t gap,
    int delimiter_token_id) const {
    if (gap >= gap_delimiter_targets.size()) {
        throw std::runtime_error(
            "AtomInsertionExample::hasDelimiterTarget: gap=" +
            std::to_string(gap) + " is outside gap count=" +
            std::to_string(gap_delimiter_targets.size()));
    }
    const std::size_t column = delimiterClassIndexOrThrow(
        delimiter_token_id,
        "AtomInsertionExample::hasDelimiterTarget");
    return gap_delimiter_targets[gap][column] != 0;
}

void AtomInsertionExample::validate(const char* caller) const {
    const std::string prefix = requireCaller(caller);
    const std::size_t byte_count = plain_text_bytes.size();
    const std::size_t expected_gap_count = byte_count + 1;

    if (byte_token_ids.size() != byte_count) {
        throw std::runtime_error(
            prefix + ": byte_token_ids.size()=" +
            std::to_string(byte_token_ids.size()) +
            " != plain_text_bytes.size()=" + std::to_string(byte_count));
    }
    if (gap_delimiter_targets.size() != expected_gap_count) {
        throw std::runtime_error(
            prefix + ": gap_delimiter_targets.size()=" +
            std::to_string(gap_delimiter_targets.size()) +
            " != byte count + 1=" + std::to_string(expected_gap_count));
    }
    if (valid_utf8_gaps.size() != expected_gap_count) {
        throw std::runtime_error(
            prefix + ": valid_utf8_gaps.size()=" +
            std::to_string(valid_utf8_gaps.size()) +
            " != byte count + 1=" + std::to_string(expected_gap_count));
    }

    for (std::size_t index = 0; index < byte_count; ++index) {
        const auto byte_value = static_cast<std::uint8_t>(
            static_cast<unsigned char>(plain_text_bytes[index]));
        const int expected_id = Tokenizer::byteToTokenId(byte_value);
        if (byte_token_ids[index] != expected_id) {
            throw std::runtime_error(
                prefix + ": byte_token_ids[" + std::to_string(index) +
                "]=" + std::to_string(byte_token_ids[index]) +
                " does not encode source byte=" +
                std::to_string(static_cast<unsigned int>(byte_value)));
        }
    }

    const std::vector<std::uint8_t> expected_utf8_gaps =
        buildValidUtf8GapMask(plain_text_bytes, prefix);
    if (valid_utf8_gaps != expected_utf8_gaps) {
        throw std::runtime_error(
            prefix + ": valid_utf8_gaps does not match exact UTF-8 boundaries");
    }

    for (std::size_t gap = 0; gap < expected_gap_count; ++gap) {
        for (std::size_t column = 0; column < kDelimiterClassCount; ++column) {
            if (gap_delimiter_targets[gap][column] > 1) {
                throw std::runtime_error(
                    prefix + ": gap target must be binary at gap=" +
                    std::to_string(gap) + " column=" +
                    std::to_string(column));
            }
        }
    }

    std::vector<GapDelimiterTargets> expected_targets(expected_gap_count);
    std::size_t previous_end = 0;
    for (std::size_t occurrence = 0; occurrence < spans.size(); ++occurrence) {
        const AtomInsertionSpanLabel& span = spans[occurrence];
        if (span.begin_gap > span.end_gap ||
            span.end_gap > byte_count) {
            throw std::runtime_error(
                prefix + ": invalid span occurrence=" +
                std::to_string(occurrence) + " gaps=[" +
                std::to_string(span.begin_gap) + "," +
                std::to_string(span.end_gap) + ")");
        }
        if (occurrence > 0 && span.begin_gap < previous_end) {
            throw std::runtime_error(
                prefix + ": overlapping span occurrence=" +
                std::to_string(occurrence));
        }
        if (valid_utf8_gaps[span.begin_gap] == 0 ||
            valid_utf8_gaps[span.end_gap] == 0) {
            throw std::runtime_error(
                prefix + ": atom boundary falls inside a UTF-8 code point at occurrence=" +
                std::to_string(occurrence));
        }

        const int open_token_id =
            Tokenizer::atomTypeToOpenTokenId(span.type);
        const int close_token_id =
            Tokenizer::atomTypeToCloseTokenId(span.type);
        setDelimiterTarget(
            expected_targets, span.begin_gap, open_token_id, prefix);
        setDelimiterTarget(
            expected_targets, span.end_gap, close_token_id, prefix);
        previous_end = span.end_gap;
    }

    if (gap_delimiter_targets != expected_targets) {
        throw std::runtime_error(
            prefix + ": dense delimiter targets do not exactly match span labels");
    }
}

AtomInsertionExample buildAtomInsertionExample(
    std::string_view annotated_source,
    const char* caller) {
    const std::string prefix = requireCaller(caller);

    const Tokenizer::Detector::DetectorRegistry registry =
        Tokenizer::Detector::makeDefaultRawTextDetectorRegistry();
    const std::vector<Tokenizer::Detector::RawTextDetection> detections =
        registry.scan(
            annotated_source,
            Tokenizer::Detector::RawTextDetectorOptions(false, false));

    AtomInsertionExample example;
    example.plain_text_bytes.reserve(annotated_source.size());
    example.spans.reserve(detections.size());

    std::size_t source_cursor = 0;
    for (const Tokenizer::Detector::RawTextDetection& detection : detections) {
        if (!detection.emitsAtom()) {
            continue;
        }
        if (detection.start < source_cursor ||
            detection.start > detection.end ||
            detection.end > annotated_source.size()) {
            throw std::runtime_error(
                prefix + ": atom detections must be sorted, non-overlapping, and in bounds");
        }

        const int open_token_id =
            Tokenizer::atomTypeToOpenTokenId(detection.atom_type);
        const int close_token_id =
            Tokenizer::atomTypeToCloseTokenId(detection.atom_type);
        const std::string open_text = Tokenizer::atomTokenText(open_token_id);
        const std::string close_text = Tokenizer::atomTokenText(close_token_id);

        if (detection.end - detection.start <
            open_text.size() + close_text.size()) {
            throw std::runtime_error(
                prefix + ": atom detection is shorter than its typed delimiters");
        }
        const std::size_t inner_begin = detection.start + open_text.size();
        const std::size_t inner_end = detection.end - close_text.size();
        if (annotated_source.substr(detection.start, open_text.size()) !=
                open_text ||
            annotated_source.substr(inner_end, close_text.size()) !=
                close_text) {
            throw std::runtime_error(
                prefix + ": detector span does not match its canonical typed delimiters");
        }

        const std::size_t content_begin =
            static_cast<std::size_t>(detection.content_offset);
        const std::size_t content_end = content_begin +
            static_cast<std::size_t>(detection.content_length);
        if (content_begin < inner_begin || content_end < content_begin ||
            content_end > inner_end) {
            throw std::runtime_error(
                prefix + ": atom content is not contained between its delimiters");
        }
        validateSpanContent(annotated_source, detection, prefix);

        appendExactBytes(
            annotated_source,
            source_cursor,
            detection.start,
            example.plain_text_bytes,
            prefix);

        const std::size_t plain_inner_begin = example.plain_text_bytes.size();
        appendExactBytes(
            annotated_source,
            inner_begin,
            inner_end,
            example.plain_text_bytes,
            prefix);

        AtomInsertionSpanLabel span;
        span.begin_gap = plain_inner_begin + (content_begin - inner_begin);
        span.end_gap = plain_inner_begin + (content_end - inner_begin);
        span.type = detection.atom_type;
        example.spans.push_back(span);

        source_cursor = detection.end;
    }

    appendExactBytes(
        annotated_source,
        source_cursor,
        annotated_source.size(),
        example.plain_text_bytes,
        prefix);

    example.byte_token_ids.reserve(example.plain_text_bytes.size());
    for (const char value : example.plain_text_bytes) {
        const auto byte_value = static_cast<std::uint8_t>(
            static_cast<unsigned char>(value));
        example.byte_token_ids.push_back(Tokenizer::byteToTokenId(byte_value));
    }

    example.gap_delimiter_targets.resize(
        example.plain_text_bytes.size() + 1);
    for (const AtomInsertionSpanLabel& span : example.spans) {
        setDelimiterTarget(
            example.gap_delimiter_targets,
            span.begin_gap,
            Tokenizer::atomTypeToOpenTokenId(span.type),
            prefix);
        setDelimiterTarget(
            example.gap_delimiter_targets,
            span.end_gap,
            Tokenizer::atomTypeToCloseTokenId(span.type),
            prefix);
    }

    example.valid_utf8_gaps =
        buildValidUtf8GapMask(example.plain_text_bytes, prefix);
    example.validate(caller);
    return example;
}

} // namespace GRIM::AtomInsertion
