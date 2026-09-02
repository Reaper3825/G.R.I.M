//======================================================//
//  AtomInsertionData.cu
//  Byte-gap supervision for the inference atom inserter
//======================================================//

#include "AtomInsertionData.hpp"

#include "../Batching/BatchPayload.hpp"
#include "../UnigramByte/AtomTable.hpp"
#include "../UnigramByte/Detectors/DetectorRegistry.hpp"
#include "../UnigramByte/TextUtils.hpp"

#include <stdexcept>
#include <string>
#include <limits>

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

void setDecisionTarget(std::vector<GapAtomDecisionTargets>& targets,
                       std::size_t gap,
                       int decision_class,
                       const std::string& prefix) {
    if (gap >= targets.size()) {
        throw std::runtime_error(
            prefix + ": atom decision target gap=" + std::to_string(gap) +
            " is outside gap count=" + std::to_string(targets.size()));
    }
    if (decision_class < 0 || decision_class >= kAtomDecisionClassCount) {
        throw std::runtime_error(
            prefix + ": atom decision class=" +
            std::to_string(decision_class) + " is outside [0," +
            std::to_string(kAtomDecisionClassCount) + ")");
    }
    const std::size_t column = static_cast<std::size_t>(decision_class);
    if (targets[gap][column] != 0) {
        throw std::runtime_error(
            prefix + ": duplicate atom decision class=" +
            std::to_string(decision_class) + " at gap=" +
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

bool AtomInsertionExample::hasDecisionTarget(
    std::size_t gap,
    int decision_class) const {
    if (gap >= gap_decision_targets.size()) {
        throw std::runtime_error(
            "AtomInsertionExample::hasDecisionTarget: gap=" +
            std::to_string(gap) + " is outside gap count=" +
            std::to_string(gap_decision_targets.size()));
    }
    (void)decisionVocabColumnOrThrow(
        decision_class, "AtomInsertionExample::hasDecisionTarget");
    return gap_decision_targets[gap][
        static_cast<std::size_t>(decision_class)] != 0;
}

void AtomInsertionExample::validate(const char* caller) const {
    const std::string prefix = requireCaller(caller);
    const std::size_t byte_count = plain_text_bytes.size();
    const std::size_t expected_gap_count = byte_count + 1;
    const std::size_t expected_input_count = byte_count + 2;

    if (!EnableAtomIdentification) {
        throw std::runtime_error(
            prefix + ": EnableAtomIdentification is false for an "
            "atom-identification example");
    }
    if (transformer_input_ids.size() != expected_input_count) {
        throw std::runtime_error(
            prefix + ": transformer_input_ids.size()=" +
            std::to_string(transformer_input_ids.size()) +
            " != byte count + BOS + EOS=" +
            std::to_string(expected_input_count));
    }
    if (transformer_input_ids.front() != Tokenizer::BOS_TOKEN_ID ||
        transformer_input_ids.back() != Tokenizer::EOS_TOKEN_ID) {
        throw std::runtime_error(
            prefix + ": transformer input must begin with BOS and end with EOS");
    }
    if (gap_decision_targets.size() != expected_gap_count) {
        throw std::runtime_error(
            prefix + ": gap_decision_targets.size()=" +
            std::to_string(gap_decision_targets.size()) +
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
        const int actual_id = transformer_input_ids[index + 1];
        if (actual_id != expected_id) {
            throw std::runtime_error(
                prefix + ": transformer_input_ids[" +
                std::to_string(index + 1) + "]=" +
                std::to_string(actual_id) +
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
        for (int decision_class = 0;
             decision_class < kAtomDecisionClassCount;
             ++decision_class) {
            if (gap_decision_targets[gap][
                    static_cast<std::size_t>(decision_class)] > 1) {
                throw std::runtime_error(
                    prefix + ": gap target must be binary at gap=" +
                    std::to_string(gap) + " column=" +
                    std::to_string(decision_class));
            }
        }
    }

    std::vector<GapAtomDecisionTargets> expected_targets(expected_gap_count);
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

        setDecisionTarget(
            expected_targets,
            span.begin_gap,
            openDecisionClassIndexOrThrow(span.type, caller),
            prefix);
        setDecisionTarget(
            expected_targets,
            span.end_gap,
            kExitDecisionClassIndex,
            prefix);
        previous_end = span.end_gap;
    }

    if (gap_decision_targets != expected_targets) {
        throw std::runtime_error(
            prefix + ": dense decision targets do not exactly match span labels");
    }
}

AtomInsertionExample buildAtomInsertionExample(
    std::string_view annotated_source,
    bool EnableAtomIdentification,
    const char* caller) {
    const std::string prefix = requireCaller(caller);
    if (!EnableAtomIdentification) {
        throw std::runtime_error(
            prefix + ": EnableAtomIdentification is false; the "
            "atom-identification data path must not run");
    }

    const Tokenizer::Detector::DetectorRegistry registry =
        Tokenizer::Detector::makeDefaultRawTextDetectorRegistry();
    const std::vector<Tokenizer::Detector::RawTextDetection> detections =
        registry.scan(
            annotated_source,
            Tokenizer::Detector::RawTextDetectorOptions(false, false));

    AtomInsertionExample example;
    example.EnableAtomIdentification = EnableAtomIdentification;
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
        if (detection.atom_type == Tokenizer::AtomType::ATOM_STRING ||
            detection.atom_type == Tokenizer::AtomType::ATOM_ENTITY) {
            if (content_begin != inner_begin || content_end != inner_end) {
                throw std::runtime_error(
                    prefix + ": STRING/ENTITY content must exactly equal its delimiter interior");
            }
        } else {
            for (std::size_t index = inner_begin; index < content_begin; ++index) {
                if (!Tokenizer::isWhitespaceASCII(
                        static_cast<unsigned char>(annotated_source[index]))) {
                    throw std::runtime_error(
                        prefix + ": non-whitespace bytes precede trimmed atom content");
                }
            }
            for (std::size_t index = content_end; index < inner_end; ++index) {
                if (!Tokenizer::isWhitespaceASCII(
                        static_cast<unsigned char>(annotated_source[index]))) {
                    throw std::runtime_error(
                        prefix + ": non-whitespace bytes follow trimmed atom content");
                }
            }
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

    example.transformer_input_ids.reserve(example.plain_text_bytes.size() + 2);
    example.transformer_input_ids.push_back(Tokenizer::BOS_TOKEN_ID);
    for (const char value : example.plain_text_bytes) {
        const auto byte_value = static_cast<std::uint8_t>(
            static_cast<unsigned char>(value));
        example.transformer_input_ids.push_back(
            Tokenizer::byteToTokenId(byte_value));
    }
    example.transformer_input_ids.push_back(Tokenizer::EOS_TOKEN_ID);

    example.gap_decision_targets.resize(
        example.plain_text_bytes.size() + 1);
    for (const AtomInsertionSpanLabel& span : example.spans) {
        setDecisionTarget(
            example.gap_decision_targets,
            span.begin_gap,
            openDecisionClassIndexOrThrow(span.type, caller),
            prefix);
        setDecisionTarget(
            example.gap_decision_targets,
            span.end_gap,
            kExitDecisionClassIndex,
            prefix);
    }

    example.valid_utf8_gaps =
        buildValidUtf8GapMask(example.plain_text_bytes, prefix);
    example.validate(caller);
    return example;
}

Batching::BatchPayload buildAtomInsertionBatchPayload(
    const std::vector<AtomInsertionExample>& examples,
    int vocab_size,
    int max_seq_len,
    bool EnableAtomIdentification,
    const char* caller) {
    const std::string prefix = requireCaller(caller);
    if (!EnableAtomIdentification) {
        throw std::runtime_error(
            prefix + ": EnableAtomIdentification is false; the atom "
            "batch path must not run");
    }
    if (examples.empty()) {
        throw std::runtime_error(prefix + ": atom batch is empty");
    }
    if (examples.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(prefix + ": atom batch size exceeds int range");
    }
    if (max_seq_len < 2) {
        throw std::runtime_error(
            prefix + ": max_seq_len must include at least BOS and EOS");
    }
    if (vocab_size < Tokenizer::UNIGRAM_VOCAB_OFFSET) {
        throw std::runtime_error(
            prefix + ": vocab_size does not contain the atom delimiter range");
    }

    const int batch_size = static_cast<int>(examples.size());
    if (batch_size > std::numeric_limits<int>::max() / max_seq_len) {
        throw std::runtime_error(prefix + ": atom batch token count overflows int");
    }
    const int total_tokens = batch_size * max_seq_len;
    const int gap_rows_per_sequence = max_seq_len - 1;
    const int total_gap_rows = batch_size * gap_rows_per_sequence;

    Batching::BatchPayload payload;
    payload.mode = Batching::BatchPayloadMode::Training;
    payload.EnableAtomIdentification = EnableAtomIdentification;
    payload.batch_size = batch_size;
    payload.max_seq_len = max_seq_len;
    payload.total_tokens = total_tokens;
    payload.vocab_size = vocab_size;
    payload.seq_ids.resize(static_cast<std::size_t>(batch_size));
    payload.seq_lengths.resize(static_cast<std::size_t>(batch_size));
    payload.valid_target_counts.resize(static_cast<std::size_t>(batch_size));
    // Prompt coordinates belong to the future gap-aware pooling design and are
    // intentionally absent for the standalone atom task.
    payload.prompt_lengths.clear();
    payload.prompt_end_positions.clear();
    payload.goals.assign(static_cast<std::size_t>(batch_size), nullptr);
    payload.concept_block_spans.assign(
        static_cast<std::size_t>(batch_size), nullptr);
    payload.seq_atom_tables.assign(static_cast<std::size_t>(batch_size), nullptr);

    payload.input_ids.assign(
        static_cast<std::size_t>(total_tokens),
        Tokenizer::PAD_TOKEN_ID);
    payload.target_ids.clear();
    payload.numeric_values.assign(static_cast<std::size_t>(total_tokens), 0.0f);
    payload.atom_mask.assign(static_cast<std::size_t>(total_tokens), 0);
    payload.atom_aux_target_mask.assign(
        static_cast<std::size_t>(total_tokens), 0);
    payload.atom_flags.assign(static_cast<std::size_t>(total_tokens), 0);
    payload.atom_entry_ids.assign(
        static_cast<std::size_t>(total_tokens),
        Tokenizer::kAtomEntryNone);
    payload.token_to_slot_index_map.assign(
        static_cast<std::size_t>(total_tokens), -1);

    payload.atom_insertion_gap_rows_per_sequence = gap_rows_per_sequence;
    payload.atom_insertion_gap_targets.assign(
        static_cast<std::size_t>(total_gap_rows) * kAtomDecisionClassCount,
        0);
    payload.atom_insertion_valid_gap_mask.assign(
        static_cast<std::size_t>(total_gap_rows), 0);

    for (int row = 0; row < batch_size; ++row) {
        const AtomInsertionExample& example =
            examples[static_cast<std::size_t>(row)];
        example.validate(caller);
        if (!example.EnableAtomIdentification) {
            throw std::runtime_error(
                prefix + ": disabled atom example at row=" +
                std::to_string(row));
        }
        if (example.transformerInputSize() >
            static_cast<std::size_t>(max_seq_len)) {
            throw std::runtime_error(
                prefix + ": transformer input exceeds max_seq_len at row=" +
                std::to_string(row));
        }

        const int sequence_length =
            static_cast<int>(example.transformerInputSize());
        const int real_gap_count = static_cast<int>(example.gapSize());
        if (real_gap_count != sequence_length - 1) {
            throw std::runtime_error(
                prefix + ": example input/gap geometry mismatch at row=" +
                std::to_string(row));
        }

        payload.seq_ids[static_cast<std::size_t>(row)] =
            static_cast<std::uint32_t>(row);
        payload.seq_lengths[static_cast<std::size_t>(row)] = sequence_length;
        payload.actual_tokens += sequence_length;

        const std::size_t token_row_offset =
            static_cast<std::size_t>(row) * max_seq_len;
        for (int token = 0; token < sequence_length; ++token) {
            payload.input_ids[token_row_offset + token] =
                example.transformer_input_ids[static_cast<std::size_t>(token)];
        }

        const std::size_t gap_row_offset =
            static_cast<std::size_t>(row) * gap_rows_per_sequence;
        int row_valid_gap_count = 0;
        for (int gap = 0; gap < real_gap_count; ++gap) {
            const std::size_t flat_gap = gap_row_offset + gap;
            const uint8_t valid =
                example.valid_utf8_gaps[static_cast<std::size_t>(gap)];
            payload.atom_insertion_valid_gap_mask[flat_gap] = valid;
            if (valid != 0) {
                ++row_valid_gap_count;
                ++payload.atom_insertion_valid_gap_count;
            }
            for (int decision_class = 0;
                 decision_class < kAtomDecisionClassCount;
                 ++decision_class) {
                const uint8_t target = example.gap_decision_targets[
                    static_cast<std::size_t>(gap)][
                    static_cast<std::size_t>(decision_class)];
                payload.atom_insertion_gap_targets[
                    flat_gap * kAtomDecisionClassCount +
                    static_cast<std::size_t>(decision_class)] = target;
                if (target != 0) {
                    ++payload.atom_insertion_positive_label_count;
                }
            }
        }
        payload.valid_target_counts[static_cast<std::size_t>(row)] =
            row_valid_gap_count;
    }

    payload.padding_tokens = payload.total_tokens - payload.actual_tokens;
    payload.valid_tokens = payload.atom_insertion_valid_gap_count;
    payload.lm_valid_tokens = 0;
    payload.fits_in_cache = true;
    payload.validate(caller);
    return payload;
}

Batching::BatchPayload buildAtomInsertionInferencePayload(
    std::string_view plain_text_bytes,
    int vocab_size,
    int max_sequence_capacity,
    const char* caller) {
    const std::string prefix = requireCaller(caller);
    if (max_sequence_capacity < 2) {
        throw std::runtime_error(
            prefix + ": max_sequence_capacity must include BOS and EOS");
    }
    if (vocab_size < Tokenizer::UNIGRAM_VOCAB_OFFSET) {
        throw std::runtime_error(
            prefix + ": vocab_size does not contain the atom delimiter range");
    }
    if (plain_text_bytes.size() >
        static_cast<std::size_t>(max_sequence_capacity - 2)) {
        throw std::runtime_error(
            prefix + ": input byte count=" +
            std::to_string(plain_text_bytes.size()) +
            " exceeds model byte capacity=" +
            std::to_string(max_sequence_capacity - 2));
    }
    if (plain_text_bytes.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() - 2)) {
        throw std::runtime_error(prefix + ": input byte count exceeds int range");
    }

    const std::string bytes(plain_text_bytes);
    const int sequence_length = static_cast<int>(bytes.size()) + 2;
    const int gap_count = sequence_length - 1;

    Batching::BatchPayload payload;
    payload.mode = Batching::BatchPayloadMode::InferencePrefill;
    payload.EnableAtomIdentification = true;
    payload.batch_size = 1;
    payload.max_seq_len = sequence_length;
    payload.total_tokens = sequence_length;
    payload.actual_tokens = sequence_length;
    payload.padding_tokens = 0;
    payload.valid_tokens = 0;
    payload.lm_valid_tokens = 0;
    payload.vocab_size = vocab_size;
    payload.seq_ids.assign(1, 0);
    payload.seq_lengths.assign(1, sequence_length);
    payload.valid_target_counts.assign(1, 0);
    payload.fits_in_cache = true;

    payload.input_ids.reserve(static_cast<std::size_t>(sequence_length));
    payload.input_ids.push_back(Tokenizer::BOS_TOKEN_ID);
    for (const char value : bytes) {
        payload.input_ids.push_back(Tokenizer::byteToTokenId(
            static_cast<std::uint8_t>(static_cast<unsigned char>(value))));
    }
    payload.input_ids.push_back(Tokenizer::EOS_TOKEN_ID);

    payload.numeric_values.assign(
        static_cast<std::size_t>(sequence_length), 0.0f);
    payload.atom_mask.assign(
        static_cast<std::size_t>(sequence_length), 0);
    payload.atom_flags.assign(
        static_cast<std::size_t>(sequence_length), 0);
    payload.atom_entry_ids.assign(
        static_cast<std::size_t>(sequence_length), Tokenizer::kAtomEntryNone);
    payload.token_to_slot_index_map.assign(
        static_cast<std::size_t>(sequence_length), -1);
    payload.seq_atom_tables.assign(1, nullptr);

    payload.atom_insertion_gap_rows_per_sequence = gap_count;
    payload.atom_insertion_valid_gap_mask =
        buildValidUtf8GapMask(bytes, prefix);
    for (const std::uint8_t valid :
         payload.atom_insertion_valid_gap_mask) {
        if (valid != 0) {
            ++payload.atom_insertion_valid_gap_count;
        }
    }
    payload.atom_insertion_positive_label_count = 0;
    payload.atom_insertion_gap_targets.clear();

    payload.validate(caller);
    return payload;
}

} // namespace GRIM::AtomInsertion
