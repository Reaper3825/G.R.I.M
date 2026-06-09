//======================================================//
//  AtomTable.cu
//  CUDA implementation of Atom Registry
//======================================================//

#include "AtomTable.hpp"
#include "Detectors/StructuralSpan.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

// NOTE: NO std::regex - banned for performance reasons

namespace GRIM {
namespace Tokenizer {

namespace {

const char* atomCategoryName(AtomCategory category) {
    switch (category) {
        case AtomCategory::NUMERIC: return "NUMERIC";
        case AtomCategory::SYSTEM: return "SYSTEM";
        default: return "UNKNOWN";
    }
}

const char* atomOriginName(AtomOrigin origin) {
    switch (origin) {
        case AtomOrigin::USER_INPUT: return "USER_INPUT";
        case AtomOrigin::MODEL_GENERATED: return "MODEL_GENERATED";
        case AtomOrigin::SYSTEM: return "SYSTEM";
        case AtomOrigin::TOOL_DERIVED: return "TOOL_DERIVED";
        case AtomOrigin::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

std::string escapeForTsv(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\t': out += "\\t"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

template <typename T>
void writeBinary(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool readBinary(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return stream.good();
}

void writeTextSpan32(std::ostream& stream, const TextSpan32& span) {
    writeBinary(stream, span.offset);
    writeBinary(stream, span.length);
}

bool readTextSpan32(std::istream& stream, TextSpan32& span) {
    return readBinary(stream, span.offset) && readBinary(stream, span.length);
}

void writeDigitBinding(std::ostream& stream, const DigitBinding& digit) {
    writeBinary(stream, digit.digit);
    writeBinary(stream, digit.pow10);
    writeBinary(stream, digit.index_left);
    writeBinary(stream, digit.index_right);
    writeTextSpan32(stream, digit.digit_span);
}

bool readDigitBinding(std::istream& stream, DigitBinding& digit) {
    return readBinary(stream, digit.digit) &&
           readBinary(stream, digit.pow10) &&
           readBinary(stream, digit.index_left) &&
           readBinary(stream, digit.index_right) &&
           readTextSpan32(stream, digit.digit_span);
}

void writeArgNumber(std::ostream& stream, const ArgNumber& number) {
    writeTextSpan32(stream, number.raw_span);
    writeTextSpan32(stream, number.content_span);
    writeTextSpan32(stream, number.mantissa_span);
    writeTextSpan32(stream, number.sign_span);
    writeTextSpan32(stream, number.decimal_point_span);
    writeTextSpan32(stream, number.exponent_marker_span);
    writeTextSpan32(stream, number.exponent_sign_span);
    writeTextSpan32(stream, number.exponent_digits_span);
    writeBinary(stream, number.base);
    writeBinary(stream, number.has_sign);
    writeBinary(stream, number.sign_negative);
    writeBinary(stream, number.has_decimal_point);
    writeBinary(stream, number.has_exponent);
    writeBinary(stream, number.exponent_negative);
    writeBinary(stream, number.integer_digit_count);
    writeBinary(stream, number.fractional_digit_count);
    writeBinary(stream, number.exponent_value);
    writeBinary(stream, number.confidence);

    const uint32_t digit_count = static_cast<uint32_t>(number.digits.size());
    writeBinary(stream, digit_count);
    for (const DigitBinding& digit : number.digits) {
        writeDigitBinding(stream, digit);
    }
}

bool readArgNumber(std::istream& stream, uint32_t atom_entry_id, ArgNumber& number) {
    number = ArgNumber{};
    number.number_atom_id = atom_entry_id;

    if (!readTextSpan32(stream, number.raw_span) ||
        !readTextSpan32(stream, number.content_span) ||
        !readTextSpan32(stream, number.mantissa_span) ||
        !readTextSpan32(stream, number.sign_span) ||
        !readTextSpan32(stream, number.decimal_point_span) ||
        !readTextSpan32(stream, number.exponent_marker_span) ||
        !readTextSpan32(stream, number.exponent_sign_span) ||
        !readTextSpan32(stream, number.exponent_digits_span) ||
        !readBinary(stream, number.base) ||
        !readBinary(stream, number.has_sign) ||
        !readBinary(stream, number.sign_negative) ||
        !readBinary(stream, number.has_decimal_point) ||
        !readBinary(stream, number.has_exponent) ||
        !readBinary(stream, number.exponent_negative) ||
        !readBinary(stream, number.integer_digit_count) ||
        !readBinary(stream, number.fractional_digit_count) ||
        !readBinary(stream, number.exponent_value) ||
        !readBinary(stream, number.confidence)) {
        return false;
    }

    uint32_t digit_count = 0;
    if (!readBinary(stream, digit_count)) {
        return false;
    }
    number.digits.resize(digit_count);
    for (uint32_t i = 0; i < digit_count; ++i) {
        if (!readDigitBinding(stream, number.digits[i])) {
            return false;
        }
    }
    return true;
}

std::string formatArgNumberForText(const std::optional<ArgNumber>& number) {
    if (!number.has_value()) {
        return "<none>";
    }

    std::ostringstream oss;
    oss << "raw={offset=" << number->raw_span.offset
        << ",length=" << number->raw_span.length
        << "};content={offset=" << number->content_span.offset
        << ",length=" << number->content_span.length
        << "};mantissa={offset=" << number->mantissa_span.offset
        << ",length=" << number->mantissa_span.length
        << "};base=" << static_cast<int>(number->base)
        << ";has_sign=" << static_cast<int>(number->has_sign)
        << ";sign_negative=" << static_cast<int>(number->sign_negative)
        << ";has_decimal_point=" << static_cast<int>(number->has_decimal_point)
        << ";has_exponent=" << static_cast<int>(number->has_exponent)
        << ";exponent_negative=" << static_cast<int>(number->exponent_negative)
        << ";exponent_value=" << number->exponent_value
        << ";integer_digit_count=" << number->integer_digit_count
        << ";fractional_digit_count=" << number->fractional_digit_count
        << ";confidence=" << number->confidence
        << ";digits=[";
    for (size_t i = 0; i < number->digits.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        const DigitBinding& digit = number->digits[i];
        oss << "{digit=" << static_cast<int>(digit.digit)
            << ",pow10=" << digit.pow10
            << ",index_left=" << digit.index_left
            << ",index_right=" << digit.index_right
            << ",span={offset=" << digit.digit_span.offset
            << ",length=" << digit.digit_span.length
            << "}}";
    }
    oss << "]";
    return oss.str();
}

bool stringRefInBounds(const StringRef& ref, size_t pool_size) {
    if (ref.offset > pool_size) {
        return false;
    }
    const size_t remaining = pool_size - ref.offset;
    return static_cast<size_t>(ref.length) <= remaining;
}

bool atomTypeIsPersistable(AtomType type) {
    return type == AtomType::ATOM_INT || type == AtomType::ATOM_FLOAT;
}

bool containsWhitespace(std::string_view text) {
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            return true;
        }
    }
    return false;
}

bool atomValueMatchesType(AtomType type, const AtomValue& value) {
    if (type == AtomType::ATOM_INT) {
        return std::holds_alternative<AtomInteger>(value);
    }
    if (type == AtomType::ATOM_FLOAT) {
        return std::holds_alternative<AtomFloat>(value);
    }
    return false;
}

bool atomValuesEquivalent(const AtomValue& a, const AtomValue& b) {
    if (a.index() != b.index()) {
        return false;
    }
    if (const auto* av = std::get_if<AtomInteger>(&a)) {
        const auto& bv = std::get<AtomInteger>(b);
        return av->value == bv.value && av->base == bv.base && av->has_sign == bv.has_sign;
    }
    if (const auto* av = std::get_if<AtomFloat>(&a)) {
        const auto& bv = std::get<AtomFloat>(b);
        return av->value == bv.value && av->has_exponent == bv.has_exponent && av->exponent == bv.exponent;
    }
    return false;
}

void requireAtomTableCreationCaller(const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("createAtomTableFromRawTextDetections: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void requireAtomTablePayloadApplyCaller(const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("applyAtomTokenizationPayloadToTokenSideChannels: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

void validateRawTextDetectionForAtomTableCreation(
    const Detector::RawTextDetection& detection,
    size_t detection_index,
    size_t source_size,
    const char* caller) {
    if (detection.detector_name == nullptr || detection.detector_name[0] == '\0') {
        throw std::runtime_error(std::string(caller) +
                                 ": raw text detection index=" + std::to_string(detection_index) +
                                 " has empty detector_name");
    }
    if (detection.end < detection.start) {
        throw std::runtime_error(std::string(caller) +
                                 ": raw text detection index=" + std::to_string(detection_index) +
                                 " has end < start, start=" + std::to_string(detection.start) +
                                 ", end=" + std::to_string(detection.end));
    }
    if (detection.end > source_size) {
        throw std::runtime_error(std::string(caller) +
                                 ": raw text detection index=" + std::to_string(detection_index) +
                                 " exceeds source text size, end=" + std::to_string(detection.end) +
                                 ", source_size=" + std::to_string(source_size));
    }
    if (detection.emitsAtom() && detection.start == detection.end) {
        throw std::runtime_error(std::string(caller) +
                                 ": atom-emitting raw text detection index=" + std::to_string(detection_index) +
                                 " is empty for detector '" + std::string(detection.detector_name) + "'");
    }
    if (detection.start > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        detection.length() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": raw text detection index=" + std::to_string(detection_index) +
                                 " exceeds StructuralSpan uint32 offset/length capacity");
    }
}

struct PendingMantissaDigit {
    uint32_t offset = 0;
    uint8_t digit = 0;
    uint32_t mantissa_index = 0;
};

uint16_t requireUint16ForArgNumber(
    uint32_t value,
    const std::string& field_name,
    const char* caller) {
    if (value > static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": arg_number " + field_name +
                                 " exceeds uint16 capacity, value=" + std::to_string(value));
    }
    return static_cast<uint16_t>(value);
}

int16_t requireInt16Pow10ForArgNumber(
    int32_t value,
    const std::string& raw_text,
    const char* caller) {
    if (value < static_cast<int32_t>(std::numeric_limits<int16_t>::min()) ||
        value > static_cast<int32_t>(std::numeric_limits<int16_t>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": arg_number pow10 exceeds int16 capacity for numeric atom text='" +
                                 raw_text + "', pow10=" + std::to_string(value));
    }
    return static_cast<int16_t>(value);
}

[[noreturn]] void throwDetectorNumericAtomContractFailure(
    const char* caller,
    const Detector::RawTextDetection& detection,
    size_t detection_index,
    AtomType atom_type,
    std::string_view atom_text,
    const std::string& reason) {
    throw std::runtime_error(std::string(caller) +
                             ": detector-emitted atom span is not parseable; detection_index=" +
                             std::to_string(detection_index) +
                             ", detector='" + std::string(detection.detector_name) +
                             "', atom_type=" + atomTypeName(atom_type) +
                             ", span=[" + std::to_string(detection.start) + ", " +
                             std::to_string(detection.end) + "), raw_text='" +
                             std::string(atom_text) + "', reason='" + reason +
                             "'; upstream detector/data pipeline bug: detector-emitted numeric spans must not fall back to text");
}

[[noreturn]] void throwArgNumberPopulationFailure(
    const char* caller,
    AtomType atom_type,
    uint32_t atom_entry_id,
    const TextSpan32& raw_span,
    std::string_view atom_text,
    const std::string& reason) {
    throw std::runtime_error(std::string(caller) +
                             ": failed to populate arg_number for numeric atom entry id=" +
                             std::to_string(atom_entry_id) +
                             ", atom_type=" + atomTypeName(atom_type) +
                             ", raw_span={offset=" + std::to_string(raw_span.offset) +
                             ", length=" + std::to_string(raw_span.length) +
                             "}, raw_text='" + std::string(atom_text) +
                             "', reason='" + reason + "'");
}

ArgNumber buildArgNumberFromContentText(
    std::string_view content_text,
    const TextSpan32& raw_span,
    const TextSpan32& content_span,
    uint32_t atom_entry_id,
    float confidence,
    AtomType atom_type,
    const char* caller,
    const Detector::RawTextDetection* detection,
    size_t detection_index) {
    if (!isNumericAtom(atom_type)) {
        throw std::runtime_error(std::string(caller) +
                                 ": buildArgNumberFromContentText requires numeric atom type, got " +
                                 atomTypeName(atom_type));
    }
    auto fail = [&](const std::string& reason) -> void {
        if (detection != nullptr) {
            throwDetectorNumericAtomContractFailure(
                caller,
                *detection,
                detection_index,
                atom_type,
                content_text,
                "failed to populate required atom entry arg_number metadata: " + reason);
        }
        throwArgNumberPopulationFailure(caller, atom_type, atom_entry_id, raw_span, content_text, reason);
    };

    if (content_text.size() != static_cast<size_t>(content_span.length)) {
        throw std::runtime_error(std::string(caller) +
                                 ": arg_number content_text size mismatch for atom_entry_id=" +
                                 std::to_string(atom_entry_id) +
                                 ", content_text.size()=" + std::to_string(content_text.size()) +
                                 ", content_span.length=" + std::to_string(content_span.length));
    }
    if (content_span.length == 0) {
        fail("empty numeric atom content span");
    }

    const uint32_t content_end = content_span.length;
    const std::string atom_text(content_text.data(), content_text.size());

    ArgNumber number{};
    number.number_atom_id = atom_entry_id;
    number.raw_span = raw_span;
    number.content_span = content_span;
    number.base = 10;
    number.confidence = confidence;

    uint32_t pos = 0;
    if (pos < content_end && (content_text[pos] == '+' || content_text[pos] == '-')) {
        number.has_sign = 1;
        number.sign_negative = content_text[pos] == '-' ? 1 : 0;
        number.sign_span.offset = content_span.offset + pos;
        number.sign_span.length = 1;
        ++pos;
    }

    std::vector<PendingMantissaDigit> mantissa_digits;
    mantissa_digits.reserve(content_span.length);
    uint32_t integer_digit_count = 0;
    uint32_t fractional_digit_count = 0;
    uint32_t first_mantissa_offset = 0;
    uint32_t last_mantissa_offset = 0;
    bool saw_mantissa_digit = false;
    bool saw_decimal_point = false;

    while (pos < content_end) {
        const unsigned char c = static_cast<unsigned char>(content_text[pos]);
        if (c >= static_cast<unsigned char>('0') && c <= static_cast<unsigned char>('9')) {
            const uint32_t absolute_offset = content_span.offset + pos;
            if (!saw_mantissa_digit) {
                first_mantissa_offset = absolute_offset;
            }
            saw_mantissa_digit = true;
            last_mantissa_offset = absolute_offset;

            PendingMantissaDigit pending{};
            pending.offset = absolute_offset;
            pending.digit = static_cast<uint8_t>(c - static_cast<unsigned char>('0'));
            pending.mantissa_index = static_cast<uint32_t>(mantissa_digits.size());
            mantissa_digits.push_back(pending);

            if (saw_decimal_point) {
                ++fractional_digit_count;
            } else {
                ++integer_digit_count;
            }
            ++pos;
            continue;
        }

        if (c == static_cast<unsigned char>('.') && !saw_decimal_point) {
            saw_decimal_point = true;
            number.has_decimal_point = 1;
            number.decimal_point_span.offset = content_span.offset + pos;
            number.decimal_point_span.length = 1;
            ++pos;
            continue;
        }

        break;
    }

    if (!saw_mantissa_digit) {
        fail("numeric atom has no mantissa digits");
    }

    number.mantissa_span.offset = first_mantissa_offset;
    number.mantissa_span.length = last_mantissa_offset - first_mantissa_offset + 1U;

    int32_t exponent_value = 0;
    if (pos < content_end && (content_text[pos] == 'e' || content_text[pos] == 'E')) {
        number.has_exponent = 1;
        number.exponent_marker_span.offset = content_span.offset + pos;
        number.exponent_marker_span.length = 1;
        ++pos;

        if (pos < content_end && (content_text[pos] == '+' || content_text[pos] == '-')) {
            number.exponent_negative = content_text[pos] == '-' ? 1 : 0;
            number.exponent_sign_span.offset = content_span.offset + pos;
            number.exponent_sign_span.length = 1;
            ++pos;
        }

        const uint32_t exponent_digits_begin = pos;
        int64_t exponent_abs = 0;
        uint32_t exponent_digit_count = 0;
        while (pos < content_end) {
            const unsigned char c = static_cast<unsigned char>(content_text[pos]);
            if (c < static_cast<unsigned char>('0') || c > static_cast<unsigned char>('9')) {
                break;
            }
            exponent_abs = exponent_abs * 10 + static_cast<int64_t>(c - static_cast<unsigned char>('0'));
            if (exponent_abs > static_cast<int64_t>(std::numeric_limits<int16_t>::max())) {
                fail("exponent magnitude exceeds arg_number pow10 capacity");
            }
            ++exponent_digit_count;
            ++pos;
        }
        if (exponent_digit_count == 0) {
            fail("exponent marker has no exponent digits");
        }
        number.exponent_digits_span.offset = content_span.offset + exponent_digits_begin;
        number.exponent_digits_span.length = exponent_digit_count;
        exponent_value = static_cast<int32_t>(exponent_abs);
        if (number.exponent_negative) {
            exponent_value = -exponent_value;
        }
        number.exponent_value = exponent_value;
    }

    if (pos != content_end) {
        fail("unexpected non-numeric byte inside numeric atom at relative_offset=" + std::to_string(pos));
    }

    number.integer_digit_count = requireUint16ForArgNumber(integer_digit_count, "integer_digit_count", caller);
    number.fractional_digit_count = requireUint16ForArgNumber(fractional_digit_count, "fractional_digit_count", caller);
    number.digits.reserve(mantissa_digits.size());

    for (const PendingMantissaDigit& pending : mantissa_digits) {
        const uint32_t index_left = pending.mantissa_index;
        const uint32_t index_right = static_cast<uint32_t>(mantissa_digits.size() - 1U) - index_left;
        const int32_t pow10 = static_cast<int32_t>(integer_digit_count) - 1 -
                              static_cast<int32_t>(index_left) + exponent_value;

        DigitBinding binding{};
        binding.digit = pending.digit;
        binding.index_left = requireUint16ForArgNumber(index_left, "digit.index_left", caller);
        binding.index_right = requireUint16ForArgNumber(index_right, "digit.index_right", caller);
        binding.pow10 = requireInt16Pow10ForArgNumber(pow10, atom_text, caller);
        binding.digit_span.offset = pending.offset;
        binding.digit_span.length = 1;
        number.digits.push_back(binding);
    }

    if (number.digits.empty()) {
        fail("numeric atom produced zero digit bindings");
    }

    return number;
}

void ensureAtomEntryHasArgNumber(
    AtomEntry& entry,
    std::string_view content_text,
    const TextSpan32& raw_span,
    const TextSpan32& content_span,
    const char* caller,
    const Detector::RawTextDetection* detection = nullptr,
    size_t detection_index = 0) {
    if (!isNumericAtom(entry.type)) {
        return;
    }
    if (entry.arg_number.has_value()) {
        if (entry.arg_number->number_atom_id != entry.id) {
            throw std::runtime_error(std::string(caller) +
                                     ": arg_number.number_atom_id mismatch for atom entry id=" +
                                     std::to_string(entry.id) +
                                     ", stored=" + std::to_string(entry.arg_number->number_atom_id));
        }
        return;
    }
    entry.arg_number = buildArgNumberFromContentText(
        content_text,
        raw_span,
        content_span,
        entry.id,
        entry.confidence,
        entry.type,
        caller,
        detection,
        detection_index);
}

void recordAtomEntryArgNumberSummary(
    const AtomEntry& entry,
    ArgNumberPopulationPayload& payload,
    const char* caller) {
    if (!isNumericAtom(entry.type)) {
        ++payload.skipped_atoms;
        return;
    }
    if (!entry.arg_number.has_value()) {
        throw std::runtime_error(std::string(caller) +
                                 ": numeric atom entry id=" + std::to_string(entry.id) +
                                 " is missing required arg_number metadata");
    }
    ++payload.total_numbers;
    payload.total_digits += static_cast<uint32_t>(entry.arg_number->digits.size());
}

void dumpAtomTableCreationBreakdown(
    const AtomTableFromDetectionsResult& result,
    const char* caller) {
    if (!result.atom_table) {
        throw std::runtime_error(std::string(caller) +
                                 ": dumpAtomTableCreationBreakdown requires a valid atom_table");
    }

    std::cerr << "[ATOMTABLE_ENTRY_BREAKDOWN_DUMP] caller=" << caller
              << " total_atom_tokens=" << result.atom_tokens.size()
              << " total_numbers=" << result.arg_number_payload.total_numbers
              << " total_digits=" << result.arg_number_payload.total_digits
              << " skipped_atoms=" << result.arg_number_payload.skipped_atoms
              << " malformed_numbers=" << result.arg_number_payload.malformed_numbers
              << "\n";

    for (size_t atom_index = 0; atom_index < result.atom_tokens.size(); ++atom_index) {
        const AtomTokenizationPayload& atom_payload = result.atom_tokens[atom_index];
        const std::optional<AtomEntry> entry = result.atom_table->getAtom(atom_payload.atom_entry_id);
        if (!entry.has_value()) {
            throw std::runtime_error(std::string(caller) +
                                     ": dumpAtomTableCreationBreakdown could not reload atom entry id=" +
                                     std::to_string(atom_payload.atom_entry_id));
        }

        const std::string_view raw_text = result.atom_table->getString(entry->raw_text_ref);
        const std::optional<NumericPayload> numeric_payload = result.atom_table->getNumericValue(entry->id);
        const std::optional<ArgNumber>& arg_number = entry->arg_number;

        std::cerr << "  [ATOM_ENTRY] atom_index=" << atom_index
                  << " atom_entry_id=" << entry->id
                  << " token_id=" << atom_payload.token_id
                  << " atom_type=" << atomTypeName(entry->type)
                  << " category=" << atomCategoryName(entry->category)
                  << " origin=" << atomOriginName(entry->origin)
                  << " confidence=" << entry->confidence
                  << " created_at=" << entry->created_at
                  << " hash=0x" << std::hex << entry->hash << std::dec
                  << "\n";

        std::cerr << "    raw_text='" << std::string(raw_text) << "'"
                  << " raw_text_ref={offset=" << entry->raw_text_ref.offset
                  << ", length=" << entry->raw_text_ref.length << "}"
                  << " source_span=[" << entry->source_start << ", " << entry->source_end << ")"
                  << " numeric_value=" << entry->numeric_value
                  << " flags=" << entry->flags
                  << "\n";

        std::cerr << "    token_payload: atom_mask=" << static_cast<int>(atom_payload.token_atom_mask)
                  << " is_byte_fallback=" << (atom_payload.is_byte_fallback ? "true" : "false")
                  << " token_numeric_value=" << atom_payload.token_numeric_value
                  << " token_atom_flags=" << atom_payload.token_atom_flags
                  << " span={offset=" << atom_payload.span.offset
                  << ", length=" << atom_payload.span.length
                  << ", content_offset=" << atom_payload.span.content_offset
                  << ", content_length=" << atom_payload.span.content_length
                  << ", placeholder_id=" << atom_payload.span.placeholder_id
                  << "}"
                  << "\n";

        if (numeric_payload.has_value()) {
            std::cerr << "    numeric_payload: kind=" << static_cast<int>(numeric_payload->kind)
                      << " float_value=" << numeric_payload->float_value
                      << " int_value=" << numeric_payload->int_value
                      << "\n";
        } else {
            std::cerr << "    numeric_payload: <none>\n";
        }

        if (!arg_number.has_value()) {
            throw std::runtime_error(std::string(caller) +
                                     ": dumpAtomTableCreationBreakdown found atom entry id=" +
                                     std::to_string(entry->id) +
                                     " without required arg_number metadata; atom_index=" +
                                     std::to_string(atom_index) +
                                     ", atom_type=" + atomTypeName(entry->type) +
                                     ", raw_text='" + std::string(raw_text) + "'");
        }

        const ArgNumber& number = *arg_number;
        std::cerr << "    arg_number: raw_span={offset=" << number.raw_span.offset
                  << ", length=" << number.raw_span.length
                  << "} content_span={offset=" << number.content_span.offset
                  << ", length=" << number.content_span.length
                  << "} mantissa_span={offset=" << number.mantissa_span.offset
                  << ", length=" << number.mantissa_span.length
                  << "} base=" << static_cast<int>(number.base)
                  << " has_sign=" << static_cast<int>(number.has_sign)
                  << " sign_negative=" << static_cast<int>(number.sign_negative)
                  << " sign_span={offset=" << number.sign_span.offset
                  << ", length=" << number.sign_span.length
                  << "} has_decimal_point=" << static_cast<int>(number.has_decimal_point)
                  << " decimal_point_span={offset=" << number.decimal_point_span.offset
                  << ", length=" << number.decimal_point_span.length
                  << "} has_exponent=" << static_cast<int>(number.has_exponent)
                  << " exponent_negative=" << static_cast<int>(number.exponent_negative)
                  << " exponent_value=" << number.exponent_value
                  << " exponent_marker_span={offset=" << number.exponent_marker_span.offset
                  << ", length=" << number.exponent_marker_span.length
                  << "} exponent_sign_span={offset=" << number.exponent_sign_span.offset
                  << ", length=" << number.exponent_sign_span.length
                  << "} exponent_digits_span={offset=" << number.exponent_digits_span.offset
                  << ", length=" << number.exponent_digits_span.length
                  << "} integer_digit_count=" << number.integer_digit_count
                  << " fractional_digit_count=" << number.fractional_digit_count
                  << " confidence=" << number.confidence
                  << " digit_count=" << number.digits.size()
                  << "\n";

        for (size_t digit_index = 0; digit_index < number.digits.size(); ++digit_index) {
            const DigitBinding& digit = number.digits[digit_index];
            std::cerr << "      [DIGIT_BINDING] digit_index=" << digit_index
                      << " digit=" << static_cast<int>(digit.digit)
                      << " pow10=" << digit.pow10
                      << " index_left=" << digit.index_left
                      << " index_right=" << digit.index_right
                      << " digit_span={offset=" << digit.digit_span.offset
                      << ", length=" << digit.digit_span.length
                      << "}"
                      << "\n";
        }
    }
}

} // namespace

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Unpack atom numeric values for computation
__global__ void kernelUnpackAtomNumerics(
    const double* __restrict__ numeric_float_values,
    const int64_t* __restrict__ numeric_int_values,
    const uint8_t* __restrict__ numeric_kind,
    const uint32_t* __restrict__ types,
    size_t num_atoms,
    float* __restrict__ output_floats,
    int64_t* __restrict__ output_ints,
    bool* __restrict__ is_integer
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;
    
    (void)types;
    const uint8_t kind = numeric_kind[idx];
    const bool is_int = (kind == static_cast<uint8_t>(NumericPayloadKind::INTEGER));
    
    is_integer[idx] = is_int;
    output_floats[idx] = static_cast<float>(numeric_float_values[idx]);
    output_ints[idx] = numeric_int_values[idx];
}

// Kernel: Pack atom data for embedding lookup
__global__ void kernelPackAtomEmbeddings(
    const double* __restrict__ numeric_float_values,
    const uint32_t* __restrict__ types,
    const uint32_t* __restrict__ flags,
    size_t num_atoms,
    int embedding_dim,
    float* __restrict__ output_embeddings  // [num_atoms, embedding_dim]
) {
    const size_t atom_idx = blockIdx.x;
    const size_t dim_idx = threadIdx.x;
    
    if (atom_idx >= num_atoms || dim_idx >= embedding_dim) return;
    
    float value = 0.0f;
    
    // First few dimensions encode the atom type
    if (dim_idx < 16) {
        uint32_t type = types[atom_idx];
        value = (type == dim_idx) ? 1.0f : 0.0f;
    }
    // Next dimensions encode numeric value (log-scaled)
    else if (dim_idx < 32) {
        float num_val = static_cast<float>(numeric_float_values[atom_idx]);
        int bit = dim_idx - 16;
        
        if (num_val != 0.0f) {
            // Encode sign
            if (bit == 0) {
                value = (num_val < 0) ? 1.0f : 0.0f;
            }
            // Encode magnitude in log scale
            else {
                float log_val = log2f(fabsf(num_val) + 1.0f);
                value = fmodf(log_val / (float)bit, 1.0f);
            }
        }
    }
    // Remaining dimensions from flags
    else if (dim_idx < 64) {
        uint32_t flag_bit = dim_idx - 32;
        value = ((flags[atom_idx] >> flag_bit) & 1) ? 1.0f : 0.0f;
    }
    
    output_embeddings[atom_idx * embedding_dim + dim_idx] = value;
}

//======================================================//
//  AtomTable Implementation
//======================================================//

AtomTable::AtomTable() {
    entries_.reserve(64);
    numeric_float_values_.reserve(64);
    numeric_int_values_.reserve(64);
    numeric_kinds_.reserve(64);
    string_pool_.reserve(4096);  // Start with 4KB string pool
    hash_to_ids_.reserve(64);
}

AtomTable::~AtomTable() {
    freeGPUData(gpu_data_);
}

AtomTable::AtomTable(AtomTable&& other) noexcept
    : entries_(std::move(other.entries_))
    , numeric_float_values_(std::move(other.numeric_float_values_))
    , numeric_int_values_(std::move(other.numeric_int_values_))
    , numeric_kinds_(std::move(other.numeric_kinds_))
    , string_pool_(std::move(other.string_pool_))
    , hash_to_ids_(std::move(other.hash_to_ids_))
    , type_index_(std::move(other.type_index_))
    , next_id_(other.next_id_)
    , dedup_hits_(other.dedup_hits_)
    , total_queries_(other.total_queries_)
    , pending_gpu_upload_(std::move(other.pending_gpu_upload_))
    , gpu_dirty_(other.gpu_dirty_)
    , gpu_data_(other.gpu_data_)
{
    other.gpu_data_ = GPUAtomData{};
    other.next_id_ = 0;
    other.dedup_hits_ = 0;
    other.total_queries_ = 0;
    other.gpu_dirty_ = false;
}

AtomTable& AtomTable::operator=(AtomTable&& other) noexcept {
    if (this != &other) {
        freeGPUData(gpu_data_);
        entries_ = std::move(other.entries_);
        numeric_float_values_ = std::move(other.numeric_float_values_);
        numeric_int_values_ = std::move(other.numeric_int_values_);
        numeric_kinds_ = std::move(other.numeric_kinds_);
        string_pool_ = std::move(other.string_pool_);
        hash_to_ids_ = std::move(other.hash_to_ids_);
        type_index_ = std::move(other.type_index_);
        pending_gpu_upload_ = std::move(other.pending_gpu_upload_);
        gpu_data_ = other.gpu_data_;
        next_id_ = other.next_id_;
        dedup_hits_ = other.dedup_hits_;
        total_queries_ = other.total_queries_;
        gpu_dirty_ = other.gpu_dirty_;
        
        other.gpu_data_ = GPUAtomData{};
        other.next_id_ = 0;
        other.dedup_hits_ = 0;
        other.total_queries_ = 0;
        other.gpu_dirty_ = false;
    }
    return *this;
}

AtomTableFromDetectionsResult createAtomTableFromRawTextDetections(
    std::string_view source_text,
    const std::vector<Detector::RawTextDetection>& detections,
    const char* caller) {
    requireAtomTableCreationCaller(caller);

    AtomTableFromDetectionsResult result;
    result.atom_table = std::make_shared<AtomTable>();
    result.atom_tokens.reserve(detections.size());

    for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        const Detector::RawTextDetection& detection = detections[detection_index];
        validateRawTextDetectionForAtomTableCreation(
            detection,
            detection_index,
            source_text.size(),
            caller);

        if (!detection.emitsAtom()) {
            continue;
        }

        const size_t detection_length = detection.end - detection.start;
        StructuralSpan span{};
        span.start = detection.start;
        span.end = detection.end;
        span.atom_type = detection.atom_type;
        span.buffer_ptr = source_text.data();
        span.offset = static_cast<uint32_t>(detection.start);
        span.length = static_cast<uint32_t>(detection_length);
        span.content_offset = static_cast<uint32_t>(detection.start);
        span.content_length = static_cast<uint32_t>(detection_length);
        span.placeholder_id = atomTypeToTokenId(detection.atom_type);

        const std::string_view atom_text(source_text.data() + detection.start, detection_length);
        const ParseResult parse_check = AtomTable::parseAtom(detection.atom_type, std::string(atom_text));
        if (!parse_check.success) {
            throwDetectorNumericAtomContractFailure(
                caller,
                detection,
                detection_index,
                detection.atom_type,
                atom_text,
                parse_check.error_message);
        }

        try {
            span.atom_entry_id = result.atom_table->registerSpan(span);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(caller) +
                                     ": failed to register detector-emitted atom span after parse precheck; detection_index=" +
                                     std::to_string(detection_index) +
                                     ", detector='" + std::string(detection.detector_name) +
                                     "', atom_type=" + atomTypeName(detection.atom_type) +
                                     ", span=[" + std::to_string(detection.start) + ", " +
                                     std::to_string(detection.end) + "), text='" +
                                     std::string(atom_text) + "', error='" + e.what() + "'");
        }

        if (span.atom_entry_id == kAtomEntryNone) {
            throw std::runtime_error(std::string(caller) +
                                     ": registerSpan returned kAtomEntryNone for detector-emitted atom span at detection_index=" +
                                     std::to_string(detection_index));
        }

        const std::optional<AtomEntry> entry = result.atom_table->getAtom(span.atom_entry_id);
        if (!entry.has_value()) {
            throw std::runtime_error(std::string(caller) +
                                     ": registered atom_entry_id is not retrievable, atom_entry_id=" +
                                     std::to_string(span.atom_entry_id));
        }
        AtomTokenizationPayload payload{};
        payload.span = span;
        payload.token_id = span.placeholder_id;
        payload.is_byte_fallback = false;
        payload.token_numeric_value = entry->numeric_value;
        payload.token_atom_flags = entry->flags;
        payload.token_atom_mask = 1;
        payload.atom_entry_id = span.atom_entry_id;
        result.atom_tokens.push_back(payload);
        recordAtomEntryArgNumberSummary(*entry, result.arg_number_payload, caller);
    }

    const bool dump_atom_entry_breakdown = true;
    if (dump_atom_entry_breakdown) {
        dumpAtomTableCreationBreakdown(result, caller);
    }

    return result;
}

static void applyAtomTokenizationPayloadToTokenSideChannelsInternal(
    const AtomTableFromDetectionsResult& atom_table_build,
    const std::vector<uint32_t>& atom_token_indices,
    const std::vector<int>& token_ids,
    const std::vector<float>& token_numeric_values,
    const std::vector<uint8_t>& token_atom_mask,
    const std::vector<uint32_t>& token_atom_flags,
    std::vector<uint32_t>& atom_entry_ids,
    const char* caller) {
    requireAtomTablePayloadApplyCaller(caller);

    const size_t token_count = token_ids.size();
    if (token_numeric_values.size() != token_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": token_numeric_values.size()=" + std::to_string(token_numeric_values.size()) +
                                 " != token_ids.size()=" + std::to_string(token_count));
    }
    if (token_atom_mask.size() != token_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": token_atom_mask.size()=" + std::to_string(token_atom_mask.size()) +
                                 " != token_ids.size()=" + std::to_string(token_count));
    }
    if (token_atom_flags.size() != token_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": token_atom_flags.size()=" + std::to_string(token_atom_flags.size()) +
                                 " != token_ids.size()=" + std::to_string(token_count));
    }
    if (atom_entry_ids.size() != token_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": atom_entry_ids.size()=" + std::to_string(atom_entry_ids.size()) +
                                 " != token_ids.size()=" + std::to_string(token_count));
    }
    if (atom_table_build.atom_tokens.size() != atom_token_indices.size()) {
        throw std::runtime_error(std::string(caller) +
                                 ": atom token payload count mismatch: atom_tokens=" +
                                 std::to_string(atom_table_build.atom_tokens.size()) +
                                 ", atom_token_indices=" + std::to_string(atom_token_indices.size()));
    }

    for (std::size_t atom_index = 0; atom_index < atom_table_build.atom_tokens.size(); ++atom_index) {
        const uint32_t token_index_u32 = atom_token_indices[atom_index];
        const size_t token_index = static_cast<size_t>(token_index_u32);
        if (token_index >= token_count) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom token index out of range at atom_index=" +
                                     std::to_string(atom_index) +
                                     ", token_index=" + std::to_string(token_index) +
                                     ", token_count=" + std::to_string(token_count));
        }

        const AtomTokenizationPayload& atom_payload = atom_table_build.atom_tokens[atom_index];
        if (token_ids[token_index] != atom_payload.token_id) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom token_id mismatch at token_index=" + std::to_string(token_index) +
                                     ": stored=" + std::to_string(token_ids[token_index]) +
                                     ", payload=" + std::to_string(atom_payload.token_id));
        }
        if (token_atom_mask[token_index] != atom_payload.token_atom_mask) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom mask mismatch at token_index=" + std::to_string(token_index) +
                                     ": stored=" + std::to_string(static_cast<int>(token_atom_mask[token_index])) +
                                     ", payload=" + std::to_string(static_cast<int>(atom_payload.token_atom_mask)));
        }
        if (token_atom_flags[token_index] != atom_payload.token_atom_flags) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom flags mismatch at token_index=" + std::to_string(token_index) +
                                     ": stored=" + std::to_string(token_atom_flags[token_index]) +
                                     ", payload=" + std::to_string(atom_payload.token_atom_flags));
        }
        if (token_numeric_values[token_index] != atom_payload.token_numeric_value) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom numeric payload mismatch at token_index=" + std::to_string(token_index) +
                                     ": stored=" + std::to_string(token_numeric_values[token_index]) +
                                     ", payload=" + std::to_string(atom_payload.token_numeric_value));
        }
        if (atom_payload.atom_entry_id == kAtomEntryNone) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom payload carries kAtomEntryNone at atom_index=" +
                                     std::to_string(atom_index));
        }

        atom_entry_ids[token_index] = atom_payload.atom_entry_id;
    }
}

std::shared_ptr<AtomTable> createAtomTableFromRawTextDetectionsForTokenSideChannels(
    std::string_view source_text,
    const std::vector<Detector::RawTextDetection>& detections,
    const std::vector<uint32_t>& atom_token_indices,
    const std::vector<int>& token_ids,
    const std::vector<float>& token_numeric_values,
    const std::vector<uint8_t>& token_atom_mask,
    const std::vector<uint32_t>& token_atom_flags,
    std::vector<uint32_t>& atom_entry_ids,
    const char* caller) {
    AtomTableFromDetectionsResult atom_table_build = createAtomTableFromRawTextDetections(
        source_text,
        detections,
        caller);
    applyAtomTokenizationPayloadToTokenSideChannelsInternal(
        atom_table_build,
        atom_token_indices,
        token_ids,
        token_numeric_values,
        token_atom_mask,
        token_atom_flags,
        atom_entry_ids,
        caller);
    return std::move(atom_table_build.atom_table);
}

void AtomTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    freeGPUData(gpu_data_);
    entries_.clear();
    numeric_float_values_.clear();
    numeric_int_values_.clear();
    numeric_kinds_.clear();
    string_pool_.clear();
    hash_to_ids_.clear();
    type_index_.clear();
    pending_gpu_upload_.clear();
    next_id_ = 0;
    dedup_hits_ = 0;
    total_queries_ = 0;
    gpu_dirty_ = false;
}

void AtomTable::reserve(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.reserve(count);
    numeric_float_values_.reserve(count);
    numeric_int_values_.reserve(count);
    numeric_kinds_.reserve(count);
    hash_to_ids_.reserve(count);
}

bool AtomTable::saveToFile(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    const char magic[4] = {'A', 'T', 'M', 'B'};
    const uint32_t version = 2;
    const uint32_t entry_count = static_cast<uint32_t>(entries_.size());
    const uint32_t pool_size = static_cast<uint32_t>(string_pool_.size());

    file.write(magic, sizeof(magic));
    writeBinary(file, version);
    writeBinary(file, entry_count);
    writeBinary(file, pool_size);

    for (const AtomEntry& entry : entries_) {
        const uint32_t type_value = static_cast<uint32_t>(entry.type);
        const uint8_t category_value = static_cast<uint8_t>(entry.category);
        const uint8_t origin_value = static_cast<uint8_t>(entry.origin);
        const uint8_t has_arg_number = entry.arg_number.has_value() ? 1 : 0;

        writeBinary(file, entry.hash);
        writeBinary(file, entry.id);
        writeBinary(file, type_value);
        writeBinary(file, category_value);
        writeBinary(file, origin_value);
        file.write(reinterpret_cast<const char*>(entry.padding1), sizeof(entry.padding1));
        writeBinary(file, entry.raw_text_ref.offset);
        writeBinary(file, entry.raw_text_ref.length);
        writeBinary(file, entry.confidence);
        writeBinary(file, entry.created_at);
        writeBinary(file, entry.source_start);
        writeBinary(file, entry.source_end);
        writeBinary(file, entry.numeric_value);
        writeBinary(file, entry.flags);
        writeBinary(file, entry.reserved_zero);
        writeBinary(file, has_arg_number);
        if (entry.arg_number.has_value()) {
            writeArgNumber(file, *entry.arg_number);
        }
    }
    if (pool_size > 0) {
        file.write(reinterpret_cast<const char*>(string_pool_.data()),
                   pool_size * sizeof(char));
    }

    return file.good();
}

bool AtomTable::saveToTextFile(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "id\t"
         << "type\t"
         << "category\t"
         << "origin\t"
         << "raw_len\t"
         << "raw_text\t"
         << "numeric_value\t"
         << "flags\t"
         << "confidence\t"
         << "created_at\t"
         << "source_start\t"
         << "source_end\t"
         << "hash\t"
         << "arg_number"
         << "\n";

    for (const auto& entry : entries_) {
        std::string_view raw = getString(entry.raw_text_ref);

        std::ostringstream hash_stream;
        hash_stream << "0x" << std::hex << entry.hash;

        file << entry.id << "\t"
             << atomTypeName(entry.type) << "\t"
             << atomCategoryName(entry.category) << "\t"
             << atomOriginName(entry.origin) << "\t"
             << entry.raw_text_ref.length << "\t"
             << escapeForTsv(raw) << "\t"
             << entry.numeric_value << "\t"
             << entry.flags << "\t"
             << entry.confidence << "\t"
             << entry.created_at << "\t"
             << entry.source_start << "\t"
             << entry.source_end << "\t"
             << hash_stream.str() << "\t"
             << escapeForTsv(formatArgNumberForText(entry.arg_number))
             << "\n";
    }

    return file.good();
}

bool AtomTable::loadFromFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4] = {0};
    file.read(magic, sizeof(magic));
    if (file.gcount() != sizeof(magic) ||
        magic[0] != 'A' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'B') {
        return false;
    }

    uint32_t version = 0;
    uint32_t entry_count = 0;
    uint32_t pool_size = 0;
    if (!readBinary(file, version) ||
        !readBinary(file, entry_count) ||
        !readBinary(file, pool_size)) {
        return false;
    }

    if (!file.good() || version != 2) {
        return false;
    }
    // Sanity check: reject obviously corrupt files (> 10M entries)
    if (entry_count > 10000000) {
        return false;
    }

    std::vector<AtomEntry> entries;
    entries.reserve(entry_count);
    std::vector<char> pool(pool_size);

    for (uint32_t i = 0; i < entry_count; ++i) {
        AtomEntry entry{};
        uint32_t type_value = 0;
        uint8_t category_value = 0;
        uint8_t origin_value = 0;
        uint8_t has_arg_number = 0;

        if (!readBinary(file, entry.hash) ||
            !readBinary(file, entry.id) ||
            !readBinary(file, type_value) ||
            !readBinary(file, category_value) ||
            !readBinary(file, origin_value)) {
            return false;
        }
        file.read(reinterpret_cast<char*>(entry.padding1), sizeof(entry.padding1));
        if (!file.good() ||
            !readBinary(file, entry.raw_text_ref.offset) ||
            !readBinary(file, entry.raw_text_ref.length) ||
            !readBinary(file, entry.confidence) ||
            !readBinary(file, entry.created_at) ||
            !readBinary(file, entry.source_start) ||
            !readBinary(file, entry.source_end) ||
            !readBinary(file, entry.numeric_value) ||
            !readBinary(file, entry.flags) ||
            !readBinary(file, entry.reserved_zero) ||
            !readBinary(file, has_arg_number)) {
            return false;
        }

        entry.type = static_cast<AtomType>(type_value);
        entry.category = static_cast<AtomCategory>(category_value);
        entry.origin = static_cast<AtomOrigin>(origin_value);
        if (has_arg_number != 0) {
            ArgNumber number{};
            if (!readArgNumber(file, entry.id, number)) {
                return false;
            }
            entry.arg_number = std::move(number);
        }

        entries.push_back(std::move(entry));
    }
    if (pool_size > 0) {
        file.read(reinterpret_cast<char*>(pool.data()),
                  pool_size * sizeof(char));
    }
    if (!file.good()) {
        return false;
    }

    for (uint32_t i = 0; i < entry_count; ++i) {
        AtomEntry& entry = entries[i];
        if (entry.id < ATOM_TOKEN_BASE || entry.id >= ATOM_TOKEN_MAX) {
            return false;
        }
        if (!atomTypeIsPersistable(entry.type)) {
            return false;
        }
        const uint32_t expected_idx = entry.id - ATOM_TOKEN_BASE;
        if (expected_idx != i) {
            return false;
        }
        if (!stringRefInBounds(entry.raw_text_ref, pool.size())) {
            return false;
        }
        if (entry.category != AtomCategory::NUMERIC && entry.category != AtomCategory::SYSTEM) {
            return false;
        }
        if (entry.reserved_zero != 0) {
            return false;
        }
        if (!entry.arg_number.has_value()) {
            return false;
        }
        if (entry.arg_number->number_atom_id != entry.id) {
            return false;
        }
        if (entry.arg_number->digits.empty()) {
            return false;
        }
    }

    std::vector<double> numeric_float_values(entry_count, 0.0);
    std::vector<int64_t> numeric_int_values(entry_count, 0);
    std::vector<uint8_t> numeric_kinds(entry_count, static_cast<uint8_t>(NumericPayloadKind::NONE));
    for (uint32_t i = 0; i < entry_count; ++i) {
        AtomEntry& entry = entries[i];
        std::string raw_text;
        if (entry.raw_text_ref.length > 0) {
            raw_text.assign(pool.data() + entry.raw_text_ref.offset, entry.raw_text_ref.length);
        }
        ParseResult result = parseAtom(entry.type, raw_text);
        if (!result.success) {
            return false;
        }
        packNumericValue(entry, result.value, numeric_float_values[i], numeric_int_values[i], numeric_kinds[i]);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        freeGPUData(gpu_data_);

        entries_ = std::move(entries);
        numeric_float_values_ = std::move(numeric_float_values);
        numeric_int_values_ = std::move(numeric_int_values);
        numeric_kinds_ = std::move(numeric_kinds);
        string_pool_ = std::move(pool);
        hash_to_ids_.clear();
        type_index_.clear();
        pending_gpu_upload_.clear();
        dedup_hits_ = 0;
        total_queries_ = 0;

        hash_to_ids_.reserve(entries_.size());
        pending_gpu_upload_.reserve(entries_.size());
        for (const auto& entry : entries_) {
            hash_to_ids_[entry.hash].push_back(entry.id);
            type_index_[entry.type].push_back(entry.id);
            pending_gpu_upload_.push_back(entry.id);
        }

        next_id_ = static_cast<uint32_t>(entries_.size());
        gpu_dirty_ = !entries_.empty();
    }

    return true;
}

bool AtomTable::computeFileHash(const std::string& path, uint64_t& out_hash) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    out_hash = kFnvOffset;

    std::vector<char> buffer(64 * 1024);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        std::streamsize count = file.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            out_hash ^= static_cast<uint64_t>(static_cast<uint8_t>(buffer[i]));
            out_hash *= kFnvPrime;
        }
    }

    return !file.bad();
}

//--------------------------------------------------//
// String Pool Implementation
//--------------------------------------------------//

StringRef AtomTable::internString(const std::string& str) {
    return internString(str.data(), str.size());
}

StringRef AtomTable::internString(std::string_view sv) {
    return internString(sv.data(), sv.size());
}

StringRef AtomTable::internString(const char* data, size_t length) {
    if (length == 0) return StringRef(0, 0);
    
    uint32_t offset = static_cast<uint32_t>(string_pool_.size());
    uint32_t len = static_cast<uint32_t>(length);
    
    // Append to pool
    string_pool_.insert(string_pool_.end(), data, data + length);
    
    return StringRef(offset, len);
}

std::string_view AtomTable::getString(const StringRef& ref) const {
    if (ref.length == 0) return std::string_view();
    if (!stringRefInBounds(ref, string_pool_.size())) {
        throw std::runtime_error(
            "AtomTable::getString StringRef out of bounds: offset=" +
            std::to_string(ref.offset) + ", length=" + std::to_string(ref.length) +
            ", pool_size=" + std::to_string(string_pool_.size()));
    }
    return std::string_view(string_pool_.data() + ref.offset, ref.length);
}

size_t AtomTable::getDeduplicationHitRate() const {
    if (total_queries_ == 0) return 0;
    return (dedup_hits_ * 100) / total_queries_;
}

//--------------------------------------------------//
// Deduplication
//--------------------------------------------------//

uint32_t AtomTable::findExisting(AtomType type, uint64_t hash, std::string_view raw_text) {
    total_queries_++;
    
    auto it = hash_to_ids_.find(hash);
    if (it == hash_to_ids_.end()) {
        return UINT32_MAX;  // Not found
    }

    for (uint32_t token_id : it->second) {
        if (token_id < ATOM_TOKEN_BASE) {
            continue;
        }
        const uint32_t entry_idx = token_id - ATOM_TOKEN_BASE;
        if (entry_idx >= entries_.size()) {
            continue;
        }

        const AtomEntry& existing = entries_[entry_idx];
        std::string_view existing_text = getString(existing.raw_text_ref);
        if (existing.type == type && existing_text == raw_text) {
            dedup_hits_++;
            return token_id;  // Return the token ID (ATOM_TOKEN_BASE+), not the array index
        }
    }
    
    return UINT32_MAX;  // Hash collision but different content
}



//--------------------------------------------------//
// Registration
//--------------------------------------------------//

bool AtomTable::tryRegisterSpan(const StructuralSpan& span, uint32_t& out_id) {
    // Zero-copy: pass buffer pointer + length directly to string pool!
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Use contentView for the atom content text.
    std::string_view raw_text(span.buffer_ptr + span.content_offset, span.content_length);
    const TextSpan32 raw_span{span.offset, span.length};
    const TextSpan32 content_span{span.content_offset, span.content_length};
    
    // Compute hash for deduplication
    uint64_t hash = computeHash(span.atom_type, raw_text);
    
    // Check if this atom already exists
    uint32_t existing_id = findExisting(span.atom_type, hash, raw_text);
    if (existing_id != UINT32_MAX) {
        const uint32_t existing_idx = existing_id - ATOM_TOKEN_BASE;
        if (existing_idx >= entries_.size()) {
            throw std::runtime_error("AtomTable::tryRegisterSpan dedup returned out-of-range id=" +
                                     std::to_string(existing_id));
        }
        ensureAtomEntryHasArgNumber(
            entries_[existing_idx],
            raw_text,
            raw_span,
            content_span,
            "AtomTable::tryRegisterSpan");
        out_id = existing_id;  // Return existing ID (deduplication hit!)
        return true;
    }
    

    
    // Parse the atom (still needs std::string temporarily)
    std::string raw_text_str(raw_text);
    auto result = parseAtom(span.atom_type, raw_text_str);
    
    AtomValue parsed;
    if (result.success) {
        parsed = result.value;
    } else {
        out_id = UINT32_MAX;
        return false;
    }
    
    AtomEntry entry{};
    entry.id = ATOM_TOKEN_BASE + next_id_;
    next_id_++;
    entry.type = span.atom_type;
    entry.hash = hash;
    entry.source_start = static_cast<uint32_t>(span.start);
    entry.source_end = static_cast<uint32_t>(span.end);
    
    // Initialize metadata
    entry.origin = AtomOrigin::USER_INPUT;
    entry.category = getCategoryForType(span.atom_type);
    entry.confidence = 1.0f;
    entry.created_at = getCurrentTimestamp();
    
    // Intern the atom content
    entry.raw_text_ref = internString(span.buffer_ptr + span.content_offset, span.content_length);
    
    // Pack numeric value
    double numeric_float_value = 0.0;
    int64_t numeric_int_value = 0;
    uint8_t numeric_kind = static_cast<uint8_t>(NumericPayloadKind::NONE);
    packNumericValue(entry, parsed, numeric_float_value, numeric_int_value, numeric_kind);
    ensureAtomEntryHasArgNumber(entry, raw_text, raw_span, content_span, "AtomTable::tryRegisterSpan");
    
    // Index for fast lookup
    uint32_t new_id = entry.id;
    entries_.push_back(std::move(entry));
    numeric_float_values_.push_back(numeric_float_value);
    numeric_int_values_.push_back(numeric_int_value);
    numeric_kinds_.push_back(numeric_kind);
    hash_to_ids_[hash].push_back(new_id);
    type_index_[span.atom_type].push_back(new_id);
    
    // Mark for GPU upload
    pending_gpu_upload_.push_back(new_id);
    gpu_dirty_ = true;
    
    out_id = new_id;
    return true;
}

uint32_t AtomTable::registerSpan(const StructuralSpan& span) {
    uint32_t out_id = UINT32_MAX;
    if (!tryRegisterSpan(span, out_id)) {
        throw std::runtime_error(
            "AtomTable::registerSpan failed for atom type " +
            std::to_string(static_cast<int>(span.atom_type)));
    }
    return out_id;
}

//--------------------------------------------------//
// Lookup
//--------------------------------------------------//

std::optional<AtomEntry> AtomTable::getAtom(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Token IDs are offset by ATOM_TOKEN_BASE, convert to array index
    if (id < ATOM_TOKEN_BASE) {
        return std::nullopt;  // Not a valid atom token
    }
    
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx >= entries_.size()) {
        return std::nullopt;
    }
    
    // Verify the entry actually has this ID (sanity check)
    const AtomEntry& entry = entries_[idx];
    if (entry.id == id) {
        return entry;
    }
    
    // Fallback: Linear search (shouldn't happen with proper indexing)
    for (const auto& e : entries_) {
        if (e.id == id) {
            return e;
        }
    }
    return std::nullopt;
}

std::vector<AtomEntry> AtomTable::getAtomsByType(AtomType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<AtomEntry> result;
    
    auto it = type_index_.find(type);
    if (it != type_index_.end()) {
        result.reserve(it->second.size());
        for (uint32_t token_id : it->second) {
            // Convert token ID to array index
            if (token_id >= ATOM_TOKEN_BASE) {
                uint32_t idx = token_id - ATOM_TOKEN_BASE;
                if (idx < entries_.size()) {
                    result.push_back(entries_[idx]);
                }
            }
        }
    }
    
    return result;
}

bool AtomTable::hasAtom(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Convert token ID to array index and check bounds
    if (id < ATOM_TOKEN_BASE) return false;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    return idx < entries_.size();
}

//--------------------------------------------------//
// Main Parse Entry Point
//--------------------------------------------------//

ParseResult AtomTable::parseAtom(AtomType type, const std::string& text) {
    if (type == AtomType::ATOM_INT) {
        return parseInteger(text);
    }
    if (type == AtomType::ATOM_FLOAT) {
        return parseFloat(text);
    }
    return ParseResult{
        false,
        AtomInteger{},
        "Unsupported AtomTable atom type " + std::to_string(static_cast<int>(type)) +
            "; only ATOM_INT and ATOM_FLOAT are supported"
    };
}

//--------------------------------------------------//
// Integer Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseInteger(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty input";
        return result;
    }
    if (containsWhitespace(text)) {
        result.error_message = "Whitespace inside numeric atom text";
        return result;
    }
    
    try {
        AtomInteger atom;
        atom.base = 10;
        atom.has_sign = (text[0] == '+' || text[0] == '-');
        
        size_t pos = 0;
        atom.value = std::stoll(text, &pos, 10);
        
        if (pos != text.size()) {
            result.error_message = "Invalid integer format";
            return result;
        }
        
        result.success = true;
        result.value = atom;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

//--------------------------------------------------//
// Float Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseFloat(const std::string& text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty input";
        return result;
    }
    if (containsWhitespace(text)) {
        result.error_message = "Whitespace inside numeric atom text";
        return result;
    }
    
    try {
        AtomFloat atom;
        size_t pos = 0;
        atom.value = std::stod(text, &pos);
        
        if (pos != text.size()) {
            result.error_message = "Invalid float format";
            return result;
        }

        if (!std::isfinite(atom.value)) {
            result.error_message = "Non-finite float";
            return result;
        }
        
        // Check for exponent
        atom.has_exponent = (text.find('e') != std::string::npos || 
                            text.find('E') != std::string::npos);
        atom.exponent = 0;
        
        if (atom.has_exponent) {
            size_t e_pos = text.find_first_of("eE");
            if (e_pos != std::string::npos && e_pos + 1 < text.size()) {
                atom.exponent = std::stoi(text.substr(e_pos + 1));
            }
        }
        
        result.success = true;
        result.value = atom;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

//--------------------------------------------------//
// Serialization
//--------------------------------------------------//

std::string AtomTable::atomToString(const AtomEntry& entry) const {
    // Atom decode is source round-trip only. Canonicalization belongs to data-quality tooling,
    // not tokenizer storage or decode.
    return std::string(getString(entry.raw_text_ref));
}

// Serialize atom value directly to buffer (ZERO HEAP ALLOCATION!)
size_t AtomTable::atomValueSerialize(AtomType type, const AtomValue& value, char* out, size_t max) {
    (void)type;
    if (!out || max == 0) return 0;
    
    size_t written = 0;
    
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        
        if constexpr (std::is_same_v<T, AtomInteger>) {
            if (arg.base == 16) {
                written = snprintf(out, max, "0x%llx", (long long)arg.value);
            } else if (arg.base == 2) {
                // Binary
                written = snprintf(out, max, "0b");
                if (written < max) {
                    int64_t v = arg.value;
                    char temp[65];
                    int pos = 0;
                    do {
                        temp[pos++] = '0' + (v & 1);
                        v >>= 1;
                    } while (v > 0 && pos < 64);
                    for (int i = pos - 1; i >= 0 && written < max; --i) {
                        out[written++] = temp[i];
                    }
                }
            } else {
                written = snprintf(out, max, "%lld", (long long)arg.value);
            }
        }
        else if constexpr (std::is_same_v<T, AtomFloat>) {
            written = snprintf(out, max, "%g", arg.value);
        }
    }, value);
    
    return (written < max) ? written : 0;  // Return 0 if truncated
}



std::optional<NumericPayload> AtomTable::getNumericValue(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (id < ATOM_TOKEN_BASE) {
        return std::nullopt;
    }
    const uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx >= entries_.size()) {
        return std::nullopt;
    }
    if (idx >= numeric_float_values_.size() ||
        idx >= numeric_int_values_.size() ||
        idx >= numeric_kinds_.size()) {
        throw std::runtime_error("AtomTable::getNumericValue numeric side-channel size mismatch for atom id=" +
                                 std::to_string(id));
    }

    const AtomEntry& entry = entries_[idx];
    if (entry.id != id) {
        throw std::runtime_error("AtomTable::getNumericValue id/index mismatch for atom id=" +
                                 std::to_string(id) + ", entry.id=" + std::to_string(entry.id));
    }
    if (!isNumericAtom(entry.type)) {
        return std::nullopt;
    }

    const NumericPayloadKind kind = static_cast<NumericPayloadKind>(numeric_kinds_[idx]);
    if (kind == NumericPayloadKind::NONE) {
        return std::nullopt;
    }
    if (kind != NumericPayloadKind::INTEGER && kind != NumericPayloadKind::FLOAT) {
        throw std::runtime_error("AtomTable::getNumericValue invalid numeric kind=" +
                                 std::to_string(static_cast<int>(numeric_kinds_[idx])) +
                                 " for atom id=" + std::to_string(id));
    }

    NumericPayload payload{};
    payload.kind = kind;
    payload.float_value = numeric_float_values_[idx];
    payload.int_value = numeric_int_values_[idx];
    return payload;
}

bool AtomTable::hasNumericValue(AtomType type) {
    return isNumericAtom(type);
}

//--------------------------------------------------//
// Metadata Helpers
//--------------------------------------------------//

AtomCategory AtomTable::getCategoryForType(AtomType type) {
    if (isNumericAtom(type)) return AtomCategory::NUMERIC;
    throw std::runtime_error("AtomTable::getCategoryForType unsupported atom type " +
                             std::to_string(static_cast<int>(type)) +
                             "; only ATOM_INT and ATOM_FLOAT are supported");
}

uint64_t AtomTable::computeHash(const AtomEntry& entry) const {
    std::string_view raw_text = getString(entry.raw_text_ref);
    return computeHash(entry.type, raw_text);
}

uint64_t AtomTable::computeHash(AtomType type, std::string_view raw_text) {
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    
    // Hash the type
    hash ^= static_cast<uint64_t>(type);
    hash *= 1099511628211ULL;
    
    // Hash the raw text
    for (char c : raw_text) {
        hash ^= static_cast<uint64_t>(static_cast<uint8_t>(c));
        hash *= 1099511628211ULL;
    }
    
    return hash;
}

uint64_t AtomTable::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<uint64_t>(micros.count());
}

void AtomTable::setOrigin(uint32_t id, AtomOrigin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id < ATOM_TOKEN_BASE) return;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx < entries_.size()) {
        entries_[idx].origin = origin;
    }
}

void AtomTable::setConfidence(uint32_t id, float confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id < ATOM_TOKEN_BASE) return;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx < entries_.size()) {
        entries_[idx].confidence = std::clamp(confidence, 0.0f, 1.0f);
        if (entries_[idx].arg_number.has_value()) {
            entries_[idx].arg_number->confidence = entries_[idx].confidence;
        }
    }
}

void AtomTable::setCategory(uint32_t id, AtomCategory category) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id < ATOM_TOKEN_BASE) return;
    uint32_t idx = id - ATOM_TOKEN_BASE;
    if (idx < entries_.size()) {
        entries_[idx].category = category;
    }
}

//--------------------------------------------------//
// GPU Packing (Numeric Values Only)
//--------------------------------------------------//

void AtomTable::packNumericValue(AtomEntry& entry,
                                 const AtomValue& parsed,
                                 double& numeric_float_value,
                                 int64_t& numeric_int_value,
                                 uint8_t& numeric_kind) {
    entry.numeric_value = 0.0f;
    entry.flags = 0;
    numeric_float_value = 0.0;
    numeric_int_value = 0;
    numeric_kind = static_cast<uint8_t>(NumericPayloadKind::NONE);
    
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        
        if constexpr (std::is_same_v<T, AtomInteger>) {
            entry.numeric_value = static_cast<float>(arg.value);
            numeric_float_value = static_cast<double>(arg.value);
            numeric_int_value = arg.value;
            numeric_kind = static_cast<uint8_t>(NumericPayloadKind::INTEGER);
            entry.flags = (arg.base << 8) | (arg.has_sign ? 1 : 0);
        }
        else if constexpr (std::is_same_v<T, AtomFloat>) {
            entry.numeric_value = static_cast<float>(arg.value);
            numeric_float_value = arg.value;
            numeric_kind = static_cast<uint8_t>(NumericPayloadKind::FLOAT);
            entry.flags = (arg.has_exponent ? 1 : 0) | (arg.exponent << 8);
        }
    }, parsed);
}

//--------------------------------------------------//
// Batch GPU Upload - Only uploads pending atoms
//--------------------------------------------------//

bool AtomTable::uploadToGPU(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Only upload if there are pending changes
    if (!gpu_dirty_ || pending_gpu_upload_.empty()) {
        gpu_data_.num_atoms = entries_.size();
        return true;
    }
    
    size_t num_atoms = entries_.size();
    if (num_atoms == 0) {
        freeGPUData(gpu_data_);
        gpu_data_.num_atoms = 0;
        gpu_dirty_ = false;
        pending_gpu_upload_.clear();
        return true;
    }

    if (numeric_float_values_.size() != num_atoms ||
        numeric_int_values_.size() != num_atoms ||
        numeric_kinds_.size() != num_atoms) {
        throw std::runtime_error("AtomTable::uploadToGPU numeric side-channel size mismatch");
    }
    
    cudaError_t err;
    GPUAtomData temp{};
    
    // Allocate device memory (numeric values, types, flags only - no aux buffer)
    err = cudaMalloc(&temp.d_numeric_values, num_atoms * sizeof(float));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&temp.d_numeric_float_values, num_atoms * sizeof(double));
    if (err != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }
    
    err = cudaMalloc(&temp.d_numeric_int_values, num_atoms * sizeof(int64_t));
    if (err != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }

    err = cudaMalloc(&temp.d_numeric_kind, num_atoms * sizeof(uint8_t));
    if (err != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }

    err = cudaMalloc(&temp.d_flags, num_atoms * sizeof(uint32_t));
    if (err != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }
    
    err = cudaMalloc(&temp.d_types, num_atoms * sizeof(uint32_t));
    if (err != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }
    
    // Pack data into host arrays (cache-aligned for transfer efficiency)
    std::vector<float> h_numeric(num_atoms);
    std::vector<uint32_t> h_flags(num_atoms);
    std::vector<uint32_t> h_types(num_atoms);
    
    for (size_t i = 0; i < num_atoms; ++i) {
        const AtomEntry& entry = entries_[i];
        h_numeric[i] = entry.numeric_value;
        h_flags[i] = entry.flags;
        h_types[i] = static_cast<uint32_t>(entry.type);
    }
    
    // Batch upload to GPU. The stream is synchronized before returning because
    // these host vectors are local transaction buffers.
    const cudaError_t e1 = cudaMemcpyAsync(temp.d_numeric_values, h_numeric.data(),
                                           num_atoms * sizeof(float), cudaMemcpyHostToDevice, stream);
    const cudaError_t e2 = cudaMemcpyAsync(temp.d_numeric_float_values, numeric_float_values_.data(),
                                           num_atoms * sizeof(double), cudaMemcpyHostToDevice, stream);
    const cudaError_t e3 = cudaMemcpyAsync(temp.d_numeric_int_values, numeric_int_values_.data(),
                                           num_atoms * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    const cudaError_t e4 = cudaMemcpyAsync(temp.d_numeric_kind, numeric_kinds_.data(),
                                           num_atoms * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    const cudaError_t e5 = cudaMemcpyAsync(temp.d_flags, h_flags.data(),
                                           num_atoms * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    const cudaError_t e6 = cudaMemcpyAsync(temp.d_types, h_types.data(),
                                           num_atoms * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess ||
        e4 != cudaSuccess || e5 != cudaSuccess || e6 != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }

    const cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        freeGPUData(temp);
        return false;
    }
    
    temp.num_atoms = num_atoms;
    
    // Clear pending upload queue and mark clean
    freeGPUData(gpu_data_);
    gpu_data_ = temp;
    pending_gpu_upload_.clear();
    gpu_dirty_ = false;
    
    return true;
}

void AtomTable::freeGPUData(AtomTable::GPUAtomData& data) {
    if (data.d_numeric_values) cudaFree(data.d_numeric_values);
    if (data.d_numeric_float_values) cudaFree(data.d_numeric_float_values);
    if (data.d_numeric_int_values) cudaFree(data.d_numeric_int_values);
    if (data.d_numeric_kind) cudaFree(data.d_numeric_kind);
    if (data.d_flags) cudaFree(data.d_flags);
    if (data.d_types) cudaFree(data.d_types);
    
    data = AtomTable::GPUAtomData{};
}

} // namespace Tokenizer
} // namespace GRIM

