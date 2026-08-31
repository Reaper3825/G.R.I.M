//======================================================//
//  AtomTable.cu
//  CUDA implementation of Atom Registry
//======================================================//

#include "AtomTable.hpp"
#include "SequenceLocalAtomTable.hpp"
#include "Detectors/StructuralSpan.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

// NOTE: NO std::regex - banned for performance reasons

namespace GRIM {
namespace Tokenizer {

namespace {

constexpr char kAtomTableStreamMagic[4] = {'A', 'T', 'M', 'B'};

const char* atomCategoryName(AtomCategory category) {
    switch (category) {
        case AtomCategory::NUMERIC: return "NUMERIC";
        case AtomCategory::SYSTEM: return "SYSTEM";
        case AtomCategory::TEXT: return "TEXT";
        case AtomCategory::LOGICAL: return "LOGICAL";
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

void writeExactOrThrow(std::ostream& stream,
                       const void* data,
                       std::size_t bytes,
                       const char* sink,
                       const char* field) {
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error(std::string("AtomTable::serializeToStreamOrThrow failed writing ") +
                                 field + " to " + sink + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

void readExactOrThrow(std::istream& stream,
                      void* data,
                      std::size_t bytes,
                      const char* source,
                      const char* field) {
    stream.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error(std::string("AtomTable::deserializeFromStreamOrThrow failed reading ") +
                                 field + " from " + source + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

void writeTextSpan32(std::ostream& stream, const TextSpan32& span) {
    writeBinary(stream, span.offset);
    writeBinary(stream, span.length);
}

bool readTextSpan32(std::istream& stream, TextSpan32& span) {
    return readBinary(stream, span.offset) && readBinary(stream, span.length);
}

// Single field-order traversal for AtomNumber binary I/O. Write and read share
// this definition so the on-disk layout cannot desynchronize between the save
// and load paths.
template <bool kWrite, typename Stream>
bool transferDigitBinding(Stream& stream, DigitBinding& digit) {
    auto io = [&](auto& field) {
        if constexpr (kWrite) {
            writeBinary(stream, field);
            return true;
        } else {
            return readBinary(stream, field);
        }
    };
    return io(digit.digit) && io(digit.pow10) && io(digit.index_left) &&
           io(digit.index_right) && io(digit.digit_span.offset) && io(digit.digit_span.length);
}

template <bool kWrite, typename Stream>
bool transferArgNumber(Stream& stream, AtomNumber& number) {
    auto io = [&](auto& field) {
        if constexpr (kWrite) {
            writeBinary(stream, field);
            return true;
        } else {
            return readBinary(stream, field);
        }
    };
    auto ioSpan = [&](TextSpan32& span) { return io(span.offset) && io(span.length); };

    if (!(ioSpan(number.raw_span) &&
          ioSpan(number.content_span) &&
          ioSpan(number.mantissa_span) &&
          ioSpan(number.sign_span) &&
          ioSpan(number.decimal_point_span) &&
          ioSpan(number.exponent_marker_span) &&
          ioSpan(number.exponent_sign_span) &&
          ioSpan(number.exponent_digits_span) &&
          io(number.base) &&
          io(number.has_sign) &&
          io(number.sign_negative) &&
          io(number.has_decimal_point) &&
          io(number.has_exponent) &&
          io(number.exponent_negative) &&
          io(number.integer_digit_count) &&
          io(number.fractional_digit_count) &&
          io(number.exponent_value) &&
          io(number.confidence))) {
        return false;
    }

    uint32_t digit_count = static_cast<uint32_t>(number.digits.size());
    if (!io(digit_count)) {
        return false;
    }
    if constexpr (!kWrite) {
        number.digits.resize(digit_count);
    }
    for (uint32_t i = 0; i < digit_count; ++i) {
        if (!transferDigitBinding<kWrite>(stream, number.digits[i])) {
            return false;
        }
    }
    return true;
}

void writeArgNumber(std::ostream& stream, const AtomNumber& number) {
    // transferArgNumber takes a mutable reference so write/read share one
    // traversal; the kWrite=true instantiation never mutates the fields.
    transferArgNumber<true>(stream, const_cast<AtomNumber&>(number));
}

bool readArgNumber(std::istream& stream, uint32_t atom_entry_id, AtomNumber& number) {
    number = AtomNumber{};
    number.number_atom_id = atom_entry_id;
    return transferArgNumber<false>(stream, number);
}

std::string formatArgNumberForText(const std::optional<AtomNumber>& number) {
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
    return type == AtomType::ATOM_INT || type == AtomType::ATOM_FLOAT ||
           type == AtomType::ATOM_STRING || type == AtomType::ATOM_BOOL ||
           type == AtomType::ATOM_ENTITY;
}

bool atomEntryIdInRange(uint32_t id, size_t entry_count) {
    return id != kAtomEntryNone && static_cast<size_t>(id) < entry_count;
}

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void hashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
}

template <typename T>
void hashValue(uint64_t& hash, const T& value) {
    hashBytes(hash, &value, sizeof(value));
}

void hashStringView(uint64_t& hash, std::string_view value) {
    hashBytes(hash, value.data(), value.size());
}

void hashArgNumberForDedup(uint64_t& hash, const AtomNumber& number) {
    hashValue(hash, number.base);
    hashValue(hash, number.has_sign);
    hashValue(hash, number.sign_negative);
    hashValue(hash, number.has_decimal_point);
    hashValue(hash, number.has_exponent);
    hashValue(hash, number.exponent_negative);
    hashValue(hash, number.integer_digit_count);
    hashValue(hash, number.fractional_digit_count);
    hashValue(hash, number.exponent_value);
    const uint32_t digit_count = static_cast<uint32_t>(number.digits.size());
    hashValue(hash, digit_count);
    for (const DigitBinding& digit : number.digits) {
        hashValue(hash, digit.digit);
        hashValue(hash, digit.pow10);
        hashValue(hash, digit.index_left);
        hashValue(hash, digit.index_right);
        hashValue(hash, digit.digit_span.length);
    }
}

bool argNumbersEqualForDedup(const AtomNumber& lhs, const AtomNumber& rhs) {
    if (lhs.base != rhs.base ||
        lhs.has_sign != rhs.has_sign ||
        lhs.sign_negative != rhs.sign_negative ||
        lhs.has_decimal_point != rhs.has_decimal_point ||
        lhs.has_exponent != rhs.has_exponent ||
        lhs.exponent_negative != rhs.exponent_negative ||
        lhs.integer_digit_count != rhs.integer_digit_count ||
        lhs.fractional_digit_count != rhs.fractional_digit_count ||
        lhs.exponent_value != rhs.exponent_value ||
        lhs.digits.size() != rhs.digits.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.digits.size(); ++i) {
        const DigitBinding& lhs_digit = lhs.digits[i];
        const DigitBinding& rhs_digit = rhs.digits[i];
        if (lhs_digit.digit != rhs_digit.digit ||
            lhs_digit.pow10 != rhs_digit.pow10 ||
            lhs_digit.index_left != rhs_digit.index_left ||
            lhs_digit.index_right != rhs_digit.index_right ||
            lhs_digit.digit_span.length != rhs_digit.digit_span.length) {
            return false;
        }
    }
    return true;
}

[[noreturn]] void throwPersistedEntryValidationFailure(
    const char* boundary,
    uint32_t index,
    const char* field,
    const std::string& detail,
    const char* location_label) {
    throw std::runtime_error(
        std::string(boundary) + ": persisted entry validation failed at index=" +
        std::to_string(index) + " field=" + field + " detail=" + detail +
        " in " + location_label);
}

void validatePersistedAtomEntryOrThrow(
    const AtomEntry& entry,
    uint32_t index,
    size_t pool_size,
    const char* boundary,
    const char* location_label) {
    auto failValidation = [&](const char* field, const std::string& detail) {
        throwPersistedEntryValidationFailure(boundary, index, field, detail, location_label);
    };

    if (entry.id == kAtomEntryNone) {
        failValidation("entry.id", "id must not be kAtomEntryNone");
    }
    if (entry.id != index) {
        failValidation("entry.id", "expected contiguous atom entry id " + std::to_string(index) +
                                       " but found " + std::to_string(entry.id));
    }
    if (!atomTypeIsPersistable(entry.type)) {
        failValidation("entry.type", std::string("non-persistable type ") + atomTypeName(entry.type));
    }
    if (!stringRefInBounds(entry.raw_text_ref, pool_size)) {
        failValidation("entry.raw_text_ref",
                       "out of bounds (offset=" + std::to_string(entry.raw_text_ref.offset) +
                       ", length=" + std::to_string(entry.raw_text_ref.length) +
                       ", pool_size=" + std::to_string(pool_size) + ")");
    }
    if (entry.raw_text_ref.length == 0 && entry.type != AtomType::ATOM_STRING) {
        failValidation("entry.raw_text_ref.length", "raw text length is zero");
    }
    const AtomCategory expected_category =
        entry.type == AtomType::ATOM_STRING || entry.type == AtomType::ATOM_ENTITY
        ? AtomCategory::TEXT
        : entry.type == AtomType::ATOM_BOOL
            ? AtomCategory::LOGICAL
            : AtomCategory::NUMERIC;
    if (entry.category != expected_category && entry.category != AtomCategory::SYSTEM) {
        failValidation("entry.category", std::string("unexpected category ") +
                                             atomCategoryName(entry.category));
    }
    if (entry.reserved_zero != 0) {
        failValidation("entry.reserved_zero", "expected 0 but found " +
                                              std::to_string(entry.reserved_zero));
    }
    if (entry.arg_number.has_value()) {
        if (!isNumericAtom(entry.type)) {
            failValidation("entry.arg_number", "non-numeric atom must not carry arg_number payload");
        }
        if (entry.arg_number->number_atom_id != entry.id) {
            failValidation("entry.arg_number.number_atom_id",
                           "expected " + std::to_string(entry.id) +
                           " but found " + std::to_string(entry.arg_number->number_atom_id));
        }
        if (entry.arg_number->digits.empty()) {
            failValidation("entry.arg_number.digits", "digit binding list is empty");
        }
    }
}

void validatePersistedNumericKindOrThrow(
    const AtomEntry& entry,
    uint8_t numeric_kind,
    uint32_t index,
    const char* boundary,
    const char* location_label) {
    const uint8_t expected_kind = entry.type == AtomType::ATOM_INT
        ? static_cast<uint8_t>(NumericPayloadKind::INTEGER)
        : entry.type == AtomType::ATOM_FLOAT
            ? static_cast<uint8_t>(NumericPayloadKind::FLOAT)
            : static_cast<uint8_t>(NumericPayloadKind::NONE);
    if (numeric_kind != expected_kind) {
        throw std::runtime_error(std::string(boundary) +
                                 ": numeric kind mismatch for entry id=" +
                                 std::to_string(entry.id) +
                                 " at index=" + std::to_string(index) +
                                 " expected=" + std::to_string(expected_kind) +
                                 " actual=" + std::to_string(numeric_kind) +
                                 " in " + location_label);
    }
}

void requireCallerLabel(const char* caller, const char* boundary) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error(std::string(boundary) + ": caller label is empty at " +
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
        detection.byteLength() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": raw text detection index=" + std::to_string(detection_index) +
                                  " exceeds StructuralSpan uint32 offset/length capacity");
    }
    if (detection.offset != static_cast<uint32_t>(detection.start) ||
        detection.length != static_cast<uint32_t>(detection.byteLength())) {
        throw std::runtime_error(std::string(caller) +
                                 ": raw text detection index=" + std::to_string(detection_index) +
                                 " has StructuralSpan offset/length inconsistent with start/end");
    }
    if (detection.emitsAtom()) {
        const size_t content_start = static_cast<size_t>(detection.content_offset);
        const size_t content_end = content_start + static_cast<size_t>(detection.content_length);
        if (content_start < detection.start || content_end > detection.end ||
            content_end < content_start) {
            throw std::runtime_error(std::string(caller) +
                                     ": raw text detection index=" + std::to_string(detection_index) +
                                     " has content outside its StructuralSpan");
        }
        if (detection.content_length == 0 &&
            detection.atom_type != AtomType::ATOM_STRING) {
            throw std::runtime_error(std::string(caller) +
                                     ": atom-emitting raw text detection index=" +
                                     std::to_string(detection_index) + " has empty content");
        }
    }
}

[[noreturn]] void throwDetectorAtomContractFailure(
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
                             "'; upstream detector/data pipeline bug: detector-emitted atom spans must not fall back to text");
}

} // namespace

//======================================================//
//  CUDA Kernels
//======================================================//

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

AtomTable::AtomTable(AtomTable&& other) noexcept : AtomTable() {
    *this = std::move(other);
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
    requireCallerLabel(caller, "createAtomTableFromRawTextDetections");

    AtomTableFromDetectionsResult result;
    result.atom_table = std::make_shared<AtomTable>();
    result.local_atom_table = std::make_shared<SequenceLocalAtomTable>();
    result.atom_tokens.reserve(detections.size());

    size_t previous_detection_end = 0;
    for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        const Detector::RawTextDetection& detection = detections[detection_index];
        validateRawTextDetectionForAtomTableCreation(
            detection,
            detection_index,
            source_text.size(),
            caller);

        // Detections MUST be sorted and non-overlapping (DetectorRegistry::scan
        // guarantees this by construction). The UniByte merge loop walks atom
        // spans monotonically; an unsorted or overlapping span would move the
        // cursor backwards and duplicate or loop over source bytes. Fail loud.
        if (detection.start < previous_detection_end) {
            throw std::runtime_error(std::string(caller) +
                                     ": raw text detection index=" + std::to_string(detection_index) +
                                     " (detector='" + std::string(detection.detector_name) +
                                     "') starts at " + std::to_string(detection.start) +
                                     " before previous detection end=" +
                                     std::to_string(previous_detection_end) +
                                     "; detections must be sorted and non-overlapping");
        }
        previous_detection_end = detection.end;

        if (!detection.emitsAtom()) {
            continue;
        }

        StructuralSpan span = detection;
        span.buffer_ptr = source_text.data();
        if (span.open_token_id < 0) {
            span.open_token_id = atomTypeToOpenTokenId(detection.atom_type);
        }
        if (span.close_token_id < 0) {
            span.close_token_id = atomTypeToCloseTokenId(detection.atom_type);
        }

        const std::string_view atom_text(
            source_text.data() + span.content_offset,
            span.content_length);

        // Local references form their own typed, sequence-scoped address space.
        // They are assigned directly from the detected value and never resolve
        // through AtomTable entry IDs.
        span.local_atom_index =
            result.local_atom_table->ticket(span.atom_type, atom_text).local_index;

        bool registered = false;
        try {
            registered = result.atom_table->tryRegisterSpan(span, span.atom_entry_id);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(caller) +
                                     ": failed to register detector-emitted atom span; detection_index=" +
                                     std::to_string(detection_index) +
                                     ", detector='" + std::string(detection.detector_name) +
                                     "', atom_type=" + atomTypeName(detection.atom_type) +
                                     ", span=[" + std::to_string(detection.start) + ", " +
                                     std::to_string(detection.end) + "), text='" +
                                     std::string(atom_text) + "', error='" + e.what() + "'");
        }
        if (!registered) {
            // Cold path: tryRegisterSpan only returns false when the span text
            // does not parse. Re-parse purely to recover the precise reason for
            // the detector-contract error; the hot path parses exactly once.
            const ParseResult parse_check = AtomTable::parseAtom(detection.atom_type, atom_text);
            throwDetectorAtomContractFailure(
                caller,
                detection,
                detection_index,
                detection.atom_type,
                atom_text,
                parse_check.error_message);
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
        payload.token_numeric_value = entry->numeric_value;
        payload.token_atom_flags = entry->flags;
        result.atom_tokens.push_back(payload);
    }

    return result;
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

void AtomTable::serializeToStreamOrThrow(std::ostream& stream, const char* sink) const {
    requireCallerLabel(sink, "AtomTable::serializeToStreamOrThrow");

    std::lock_guard<std::mutex> lock(mutex_);
    if (numeric_float_values_.size() != entries_.size() ||
        numeric_int_values_.size() != entries_.size() ||
        numeric_kinds_.size() != entries_.size()) {
        throw std::runtime_error(std::string("AtomTable::serializeToStreamOrThrow: exact numeric payload sizes do not match entry count for ") +
                                 sink + " (entries=" + std::to_string(entries_.size()) +
                                 ", float_payloads=" + std::to_string(numeric_float_values_.size()) +
                                 ", int_payloads=" + std::to_string(numeric_int_values_.size()) +
                                 ", kinds=" + std::to_string(numeric_kinds_.size()) + ")");
    }

    writeExactOrThrow(stream, kAtomTableStreamMagic, sizeof(kAtomTableStreamMagic), sink, "magic");

    const uint32_t entry_count = static_cast<uint32_t>(entries_.size());
    const uint32_t pool_size = static_cast<uint32_t>(string_pool_.size());
    writeExactOrThrow(stream, &entry_count, sizeof(entry_count), sink, "entry_count");
    writeExactOrThrow(stream, &pool_size, sizeof(pool_size), sink, "pool_size");

    for (uint32_t i = 0; i < entry_count; ++i) {
        const AtomEntry& entry = entries_[i];
        validatePersistedAtomEntryOrThrow(entry,
                                          i,
                                          string_pool_.size(),
                                          "AtomTable::serializeToStreamOrThrow",
                                          sink);
        validatePersistedNumericKindOrThrow(entry,
                                            numeric_kinds_[i],
                                            i,
                                            "AtomTable::serializeToStreamOrThrow",
                                            sink);

        const uint32_t type_value = static_cast<uint32_t>(entry.type);
        const uint8_t category_value = static_cast<uint8_t>(entry.category);
        const uint8_t origin_value = static_cast<uint8_t>(entry.origin);
        const uint8_t has_arg_number = entry.arg_number.has_value() ? 1 : 0;

        writeExactOrThrow(stream, &entry.hash, sizeof(entry.hash), sink, "entry.hash");
        writeExactOrThrow(stream, &entry.id, sizeof(entry.id), sink, "entry.id");
        writeExactOrThrow(stream, &type_value, sizeof(type_value), sink, "entry.type");
        writeExactOrThrow(stream, &category_value, sizeof(category_value), sink, "entry.category");
        writeExactOrThrow(stream, &origin_value, sizeof(origin_value), sink, "entry.origin");
        writeExactOrThrow(stream, entry.padding1, sizeof(entry.padding1), sink, "entry.padding1");
        writeExactOrThrow(stream, &entry.raw_text_ref.offset, sizeof(entry.raw_text_ref.offset), sink, "entry.raw_text_ref.offset");
        writeExactOrThrow(stream, &entry.raw_text_ref.length, sizeof(entry.raw_text_ref.length), sink, "entry.raw_text_ref.length");
        writeExactOrThrow(stream, &entry.confidence, sizeof(entry.confidence), sink, "entry.confidence");
        writeExactOrThrow(stream, &entry.created_at, sizeof(entry.created_at), sink, "entry.created_at");
        writeExactOrThrow(stream, &entry.source_start, sizeof(entry.source_start), sink, "entry.source_start");
        writeExactOrThrow(stream, &entry.source_end, sizeof(entry.source_end), sink, "entry.source_end");
        writeExactOrThrow(stream, &entry.numeric_value, sizeof(entry.numeric_value), sink, "entry.numeric_value");
        writeExactOrThrow(stream, &entry.flags, sizeof(entry.flags), sink, "entry.flags");
        writeExactOrThrow(stream, &entry.reserved_zero, sizeof(entry.reserved_zero), sink, "entry.reserved_zero");
        writeExactOrThrow(stream, &has_arg_number, sizeof(has_arg_number), sink, "entry.has_arg_number");
        if (entry.arg_number.has_value()) {
            AtomNumber arg_number = *entry.arg_number;
            if (!transferArgNumber<true>(stream, arg_number) || !stream) {
                throw std::runtime_error(std::string("AtomTable::serializeToStreamOrThrow failed writing arg_number to ") + sink);
            }
        }
    }

    if (entry_count > 0) {
        writeExactOrThrow(stream,
                          numeric_float_values_.data(),
                          static_cast<std::size_t>(entry_count) * sizeof(double),
                          sink,
                          "numeric_float_values");
        writeExactOrThrow(stream,
                          numeric_int_values_.data(),
                          static_cast<std::size_t>(entry_count) * sizeof(int64_t),
                          sink,
                          "numeric_int_values");
        writeExactOrThrow(stream,
                          numeric_kinds_.data(),
                          static_cast<std::size_t>(entry_count) * sizeof(uint8_t),
                          sink,
                          "numeric_kinds");
    }
    if (pool_size > 0) {
        writeExactOrThrow(stream,
                          string_pool_.data(),
                          static_cast<std::size_t>(pool_size),
                          sink,
                          "string_pool");
    }
}

void AtomTable::deserializeFromStreamOrThrow(std::istream& stream, const char* source) {
    requireCallerLabel(source, "AtomTable::deserializeFromStreamOrThrow");

    char magic[4] = {0, 0, 0, 0};
    readExactOrThrow(stream, magic, sizeof(magic), source, "magic");
    if (std::memcmp(magic, kAtomTableStreamMagic, sizeof(magic)) != 0) {
        throw std::runtime_error(std::string("AtomTable::deserializeFromStreamOrThrow: invalid magic in ") +
                                 source + " (expected 'ATMB')");
    }

    uint32_t entry_count = 0;
    uint32_t pool_size = 0;
    readExactOrThrow(stream, &entry_count, sizeof(entry_count), source, "entry_count");
    readExactOrThrow(stream, &pool_size, sizeof(pool_size), source, "pool_size");
    if (entry_count > 10000000) {
        throw std::runtime_error(std::string("AtomTable::deserializeFromStreamOrThrow: implausible entry_count in ") +
                                 source + ": " + std::to_string(entry_count));
    }

    std::vector<AtomEntry> entries;
    entries.reserve(entry_count);
    for (uint32_t i = 0; i < entry_count; ++i) {
        AtomEntry entry{};
        uint32_t type_value = 0;
        uint8_t category_value = 0;
        uint8_t origin_value = 0;
        uint8_t has_arg_number = 0;

        readExactOrThrow(stream, &entry.hash, sizeof(entry.hash), source, "entry.hash");
        readExactOrThrow(stream, &entry.id, sizeof(entry.id), source, "entry.id");
        readExactOrThrow(stream, &type_value, sizeof(type_value), source, "entry.type");
        readExactOrThrow(stream, &category_value, sizeof(category_value), source, "entry.category");
        readExactOrThrow(stream, &origin_value, sizeof(origin_value), source, "entry.origin");
        readExactOrThrow(stream, entry.padding1, sizeof(entry.padding1), source, "entry.padding1");
        readExactOrThrow(stream, &entry.raw_text_ref.offset, sizeof(entry.raw_text_ref.offset), source, "entry.raw_text_ref.offset");
        readExactOrThrow(stream, &entry.raw_text_ref.length, sizeof(entry.raw_text_ref.length), source, "entry.raw_text_ref.length");
        readExactOrThrow(stream, &entry.confidence, sizeof(entry.confidence), source, "entry.confidence");
        readExactOrThrow(stream, &entry.created_at, sizeof(entry.created_at), source, "entry.created_at");
        readExactOrThrow(stream, &entry.source_start, sizeof(entry.source_start), source, "entry.source_start");
        readExactOrThrow(stream, &entry.source_end, sizeof(entry.source_end), source, "entry.source_end");
        readExactOrThrow(stream, &entry.numeric_value, sizeof(entry.numeric_value), source, "entry.numeric_value");
        readExactOrThrow(stream, &entry.flags, sizeof(entry.flags), source, "entry.flags");
        readExactOrThrow(stream, &entry.reserved_zero, sizeof(entry.reserved_zero), source, "entry.reserved_zero");
        readExactOrThrow(stream, &has_arg_number, sizeof(has_arg_number), source, "entry.has_arg_number");

        entry.type = static_cast<AtomType>(type_value);
        entry.category = static_cast<AtomCategory>(category_value);
        entry.origin = static_cast<AtomOrigin>(origin_value);
        if (has_arg_number != 0) {
            AtomNumber arg_number{};
            if (!readArgNumber(stream, entry.id, arg_number)) {
                throw std::runtime_error(std::string("AtomTable::deserializeFromStreamOrThrow failed reading arg_number from ") + source);
            }
            entry.arg_number = std::move(arg_number);
        }

        entries.push_back(std::move(entry));
    }

    std::vector<double> numeric_float_values(entry_count, 0.0);
    std::vector<int64_t> numeric_int_values(entry_count, 0);
    std::vector<uint8_t> numeric_kinds(entry_count, static_cast<uint8_t>(NumericPayloadKind::NONE));
    if (entry_count > 0) {
        readExactOrThrow(stream,
                         numeric_float_values.data(),
                         static_cast<std::size_t>(entry_count) * sizeof(double),
                         source,
                         "numeric_float_values");
        readExactOrThrow(stream,
                         numeric_int_values.data(),
                         static_cast<std::size_t>(entry_count) * sizeof(int64_t),
                         source,
                         "numeric_int_values");
        readExactOrThrow(stream,
                         numeric_kinds.data(),
                         static_cast<std::size_t>(entry_count) * sizeof(uint8_t),
                         source,
                         "numeric_kinds");
    }

    std::vector<char> pool(pool_size);
    if (pool_size > 0) {
        readExactOrThrow(stream,
                         pool.data(),
                         static_cast<std::size_t>(pool_size),
                         source,
                         "string_pool");
    }

    for (uint32_t i = 0; i < entry_count; ++i) {
        const AtomEntry& entry = entries[i];
        validatePersistedAtomEntryOrThrow(entry,
                                          i,
                                          pool.size(),
                                          "AtomTable::deserializeFromStreamOrThrow",
                                          source);
        validatePersistedNumericKindOrThrow(entry,
                                            numeric_kinds[i],
                                            i,
                                            "AtomTable::deserializeFromStreamOrThrow",
                                            source);
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
        for (size_t i = 0; i < entries_.size(); ++i) {
            AtomEntry& entry = entries_[i];
            entry.hash = computeHash(entry);
            hash_to_ids_[entry.hash].push_back(entry.id);
            type_index_[entry.type].push_back(entry.id);
            pending_gpu_upload_.push_back(entry.id);
        }

        next_id_ = static_cast<uint32_t>(entries_.size());
        gpu_dirty_ = !entries_.empty();
    }
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

uint32_t AtomTable::findExisting(uint64_t hash,
                                 const AtomEntry& candidate,
                                 std::string_view candidate_raw_text,
                                 double numeric_float_value,
                                 int64_t numeric_int_value,
                                 uint8_t numeric_kind) {
    total_queries_++;
    
    auto it = hash_to_ids_.find(hash);
    if (it == hash_to_ids_.end()) {
        return UINT32_MAX;  // Not found
    }

    for (uint32_t entry_id : it->second) {
        if (!atomEntryIdInRange(entry_id, entries_.size())) {
            throw std::runtime_error("AtomTable::findExisting hash bucket contains corrupt atom entry id=" +
                                     std::to_string(entry_id) + ", entries=" +
                                     std::to_string(entries_.size()));
        }
        const AtomEntry& existing = entries_[entry_id];
        const std::string_view existing_raw_text = getString(existing.raw_text_ref);
        if (existing.type == candidate.type &&
            existing.numeric_value == candidate.numeric_value &&
            existing.flags == candidate.flags &&
            existing.reserved_zero == candidate.reserved_zero &&
            existing_raw_text == candidate_raw_text &&
            numeric_kinds_[entry_id] == numeric_kind &&
            numeric_float_values_[entry_id] == numeric_float_value &&
            numeric_int_values_[entry_id] == numeric_int_value &&
            existing.arg_number.has_value() == candidate.arg_number.has_value() &&
            (!existing.arg_number.has_value() ||
             argNumbersEqualForDedup(*existing.arg_number, *candidate.arg_number))) {
            dedup_hits_++;
            return entry_id;
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
    // Parse once; this is the only parse on the new-atom path.
    const ParseResult result = parseAtom(span.atom_type, raw_text);
    if (!result.success) {
        out_id = UINT32_MAX;
        return false;
    }
    const AtomValue& parsed = result.value;
    
    AtomEntry entry{};
    entry.id = next_id_;
    entry.type = span.atom_type;
    entry.source_start = static_cast<uint32_t>(span.start);
    entry.source_end = static_cast<uint32_t>(span.end);
    
    // Initialize metadata
    entry.origin = AtomOrigin::USER_INPUT;
    entry.category = getCategoryForType(span.atom_type);
    entry.confidence = 1.0f;
    entry.created_at = getCurrentTimestamp();
    
    // Pack numeric value
    double numeric_float_value = 0.0;
    int64_t numeric_int_value = 0;
    uint8_t numeric_kind = static_cast<uint8_t>(NumericPayloadKind::NONE);
    packValue(entry, parsed, numeric_float_value, numeric_int_value, numeric_kind);

    const uint64_t hash = computeHash(entry,
                                      raw_text,
                                      numeric_float_value,
                                      numeric_int_value,
                                      numeric_kind);

    uint32_t existing_id = findExisting(hash,
                                        entry,
                                        raw_text,
                                        numeric_float_value,
                                        numeric_int_value,
                                        numeric_kind);
    if (existing_id != UINT32_MAX) {
        if (!atomEntryIdInRange(existing_id, entries_.size())) {
            throw std::runtime_error("AtomTable::tryRegisterSpan dedup returned out-of-range id=" +
                                     std::to_string(existing_id));
        }
        out_id = existing_id;  // Return existing ID (deduplication hit!)
        return true;
    }

    entry.hash = hash;
    entry.raw_text_ref = internString(span.buffer_ptr + span.content_offset, span.content_length);
    next_id_++;
    
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

std::shared_ptr<AtomTable> AtomTable::cloneHostForGeneration() const {
    auto clone = std::make_shared<AtomTable>();
    std::lock_guard<std::mutex> lock(mutex_);

    clone->entries_ = entries_;
    clone->numeric_float_values_ = numeric_float_values_;
    clone->numeric_int_values_ = numeric_int_values_;
    clone->numeric_kinds_ = numeric_kinds_;
    clone->string_pool_ = string_pool_;
    clone->hash_to_ids_ = hash_to_ids_;
    clone->type_index_ = type_index_;
    clone->next_id_ = next_id_;
    clone->dedup_hits_ = dedup_hits_;
    clone->total_queries_ = total_queries_;

    // GPUAtomData owns allocations and is never shared with the session copy.
    clone->pending_gpu_upload_.clear();
    clone->pending_gpu_upload_.reserve(clone->entries_.size());
    for (uint32_t id = 0; id < clone->entries_.size(); ++id) {
        clone->pending_gpu_upload_.push_back(id);
    }
    clone->gpu_dirty_ = !clone->entries_.empty();
    clone->gpu_data_ = GPUAtomData();
    return clone;
}

uint32_t AtomTable::registerGeneratedNumericValue(float value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "AtomTable::registerGeneratedNumericValue requires a finite value");
    }

    const std::string rendered = formatNumericValue(value);
    StructuralSpan span{};
    span.start = 0;
    span.end = rendered.size();
    span.atom_type = numericAtomTypeForValue(value);
    span.buffer_ptr = rendered.data();
    span.offset = 0;
    span.length = static_cast<uint32_t>(rendered.size());
    span.content_offset = 0;
    span.content_length = static_cast<uint32_t>(rendered.size());
    span.open_token_id = atomTypeToOpenTokenId(span.atom_type);
    span.close_token_id = atomTypeToCloseTokenId(span.atom_type);

    const size_t size_before = size();
    const uint32_t id = registerSpan(span);
    if (static_cast<size_t>(id) >= size_before) {
        setOrigin(id, AtomOrigin::MODEL_GENERATED);
    }
    return id;
}

//--------------------------------------------------//
// Lookup
//--------------------------------------------------//

std::optional<AtomEntry> AtomTable::getAtom(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!atomEntryIdInRange(id, entries_.size())) {
        return std::nullopt;
    }
    
    const AtomEntry& entry = entries_[id];
    if (entry.id != id) {
        throw std::runtime_error("AtomTable::getAtom id/index mismatch for atom id=" +
                                 std::to_string(id) + ", entry.id=" + std::to_string(entry.id) +
                                 "; entries_ indexing is corrupt");
    }
    return entry;
}

std::vector<AtomEntry> AtomTable::getAtomsByType(AtomType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<AtomEntry> result;
    auto it = type_index_.find(type);
    if (it == type_index_.end()) {
        return result;
    }
    result.reserve(it->second.size());
    for (uint32_t entry_id : it->second) {
        if (!atomEntryIdInRange(entry_id, entries_.size())) {
            throw std::runtime_error("AtomTable::getAtomsByType type index contains corrupt atom entry id=" +
                                     std::to_string(entry_id) + ", entries=" +
                                     std::to_string(entries_.size()));
        }
        result.push_back(entries_[entry_id]);
    }
    return result;
}


//--------------------------------------------------//
// Main Parse Entry Point
//--------------------------------------------------//

ParseResult AtomTable::parseAtom(AtomType type, std::string_view text) {
    if (type == AtomType::ATOM_INT) {
        return parseInteger(text);
    }
    if (type == AtomType::ATOM_FLOAT) {
        return parseFloat(text);
    }
    if (type == AtomType::ATOM_STRING) {
        return parseString(text);
    }
    if (type == AtomType::ATOM_BOOL) {
        return parseBoolean(text);
    }
    if (type == AtomType::ATOM_ENTITY) {
        return parseEntity(text);
    }
    return ParseResult{
        false,
        AtomInteger{},
        "Unsupported AtomTable atom type " + std::to_string(static_cast<int>(type)) +
            "; supported types are ATOM_INT, ATOM_FLOAT, ATOM_STRING, ATOM_BOOL, and ATOM_ENTITY"
    };
}

ParseResult AtomTable::parseString(std::string_view) {
    return ParseResult{true, AtomString{}, {}};
}

ParseResult AtomTable::parseEntity(std::string_view text) {
    if (text.empty()) {
        return ParseResult{false, AtomEntity{}, "Entity content must not be empty"};
    }
    return ParseResult{true, AtomEntity{}, {}};
}

ParseResult AtomTable::parseBoolean(std::string_view text) {
    if (text == "true") {
        return ParseResult{true, AtomBoolean{true}, {}};
    }
    if (text == "false") {
        return ParseResult{true, AtomBoolean{false}, {}};
    }
    return ParseResult{
        false,
        AtomBoolean{},
        "Invalid boolean format; expected 'true' or 'false'"
    };
}

//--------------------------------------------------//
// Integer Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseInteger(std::string_view text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty input";
        return result;
    }

    AtomInteger atom;
    atom.base = 10;
    atom.has_sign = (text[0] == '+' || text[0] == '-');

    // std::from_chars enforces the strict grammar directly: no whitespace, no
    // locale, no allocation, no partial-consumption surprises. It accepts a
    // leading '-' but not '+', so skip an explicit '+' (a digit must follow).
    size_t begin = 0;
    if (text[0] == '+') {
        begin = 1;
        if (text.size() == 1 || text[1] < '0' || text[1] > '9') {
            result.error_message = "Invalid integer format";
            return result;
        }
    }

    const char* last = text.data() + text.size();
    const std::from_chars_result parsed = std::from_chars(text.data() + begin, last, atom.value, 10);
    if (parsed.ec == std::errc::result_out_of_range) {
        result.error_message = "Integer out of range";
        return result;
    }
    if (parsed.ec != std::errc() || parsed.ptr != last) {
        result.error_message = "Invalid integer format";
        return result;
    }

    result.success = true;
    result.value = atom;
    return result;
}

//--------------------------------------------------//
// Float Parsing
//--------------------------------------------------//

ParseResult AtomTable::parseFloat(std::string_view text) {
    ParseResult result;
    result.success = false;
    
    if (text.empty()) {
        result.error_message = "Empty input";
        return result;
    }

    // Strict shape scan: [+-]? digits [. digits]? ([eE] [+-]? digits)? with at
    // least one mantissa digit. Rejects whitespace, hex floats, inf/nan
    // spellings, and trailing junk in one pass, and recovers the exponent
    // without a second scan or substring allocation.
    size_t pos = 0;
    if (text[pos] == '+' || text[pos] == '-') {
        ++pos;
    }
    size_t mantissa_digits = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        ++pos;
        ++mantissa_digits;
    }
    if (pos < text.size() && text[pos] == '.') {
        ++pos;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            ++pos;
            ++mantissa_digits;
        }
    }
    if (mantissa_digits == 0) {
        result.error_message = "Invalid float format";
        return result;
    }

    AtomFloat atom;
    atom.has_exponent = false;
    atom.exponent = 0;
    if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
        atom.has_exponent = true;
        ++pos;
        bool exponent_negative = false;
        if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) {
            exponent_negative = (text[pos] == '-');
            ++pos;
        }
        size_t exponent_digit_count = 0;
        int64_t exponent_value = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            exponent_value = exponent_value * 10 + (text[pos] - '0');
            if (exponent_value > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                result.error_message = "Float exponent out of range";
                return result;
            }
            ++pos;
            ++exponent_digit_count;
        }
        if (exponent_digit_count == 0) {
            result.error_message = "Invalid float format";
            return result;
        }
        if (exponent_negative) {
            exponent_value = -exponent_value;
        }
        atom.exponent = static_cast<int>(exponent_value);
    }
    if (pos != text.size()) {
        result.error_message = "Invalid float format";
        return result;
    }

    // Value parse (shape is already validated above) goes through a stream
    // imbued with the classic "C" locale. std::stod honors the process-global
    // LC_NUMERIC, so under e.g. a comma-decimal locale "3.14" mis-parses and
    // tokenization becomes environment-dependent. from_chars for double is not
    // reliably available on every toolchain we build with. Cold path: runs
    // once per unique atom, dedup hits skip it.
    {
        std::istringstream value_stream{std::string(text)};
        value_stream.imbue(std::locale::classic());
        value_stream >> atom.value;
        if (value_stream.fail() || !value_stream.eof()) {
            result.error_message = "Float value parse failed (or out of range)";
            return result;
        }
    }

    if (!std::isfinite(atom.value)) {
        result.error_message = "Non-finite float";
        return result;
    }

    result.success = true;
    result.value = atom;
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

std::optional<NumericPayload> AtomTable::getNumericValue(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!atomEntryIdInRange(id, entries_.size())) {
        return std::nullopt;
    }
    const uint32_t idx = id;
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
    if (type == AtomType::ATOM_STRING || type == AtomType::ATOM_ENTITY) {
        return AtomCategory::TEXT;
    }
    if (type == AtomType::ATOM_BOOL) return AtomCategory::LOGICAL;
    throw std::runtime_error("AtomTable::getCategoryForType unsupported atom type " +
                             std::to_string(static_cast<int>(type)));
}

uint64_t AtomTable::computeHash(const AtomEntry& entry) const {
    if (!atomEntryIdInRange(entry.id, entries_.size()) ||
        static_cast<size_t>(entry.id) >= numeric_float_values_.size() ||
        static_cast<size_t>(entry.id) >= numeric_int_values_.size() ||
        static_cast<size_t>(entry.id) >= numeric_kinds_.size()) {
        throw std::runtime_error("AtomTable::computeHash cannot resolve numeric side channels for atom entry id=" +
                                 std::to_string(entry.id));
    }
    std::string_view raw_text = getString(entry.raw_text_ref);
    return computeHash(entry,
                       raw_text,
                       numeric_float_values_[entry.id],
                       numeric_int_values_[entry.id],
                       numeric_kinds_[entry.id]);
}

uint64_t AtomTable::computeHash(const AtomEntry& entry,
                                std::string_view raw_text,
                                double numeric_float_value,
                                int64_t numeric_int_value,
                                uint8_t numeric_kind) const {
    uint64_t hash = kFnvOffset;
    hashValue(hash, static_cast<int>(entry.type));
    hashStringView(hash, raw_text);
    hashValue(hash, entry.numeric_value);
    hashValue(hash, entry.flags);
    hashValue(hash, entry.reserved_zero);
    hashValue(hash, numeric_kind);
    hashValue(hash, numeric_float_value);
    hashValue(hash, numeric_int_value);
    const uint8_t has_arg_number = entry.arg_number.has_value() ? 1 : 0;
    hashValue(hash, has_arg_number);
    if (entry.arg_number.has_value()) {
        hashArgNumberForDedup(hash, *entry.arg_number);
    }
    return hash;
}

uint64_t AtomTable::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<uint64_t>(micros.count());
}

AtomEntry& AtomTable::entryForIdLocked(uint32_t id, const char* caller) {
    if (!atomEntryIdInRange(id, entries_.size())) {
        throw std::runtime_error(std::string(caller) + ": invalid atom id=" + std::to_string(id) +
                                 ", entries=" + std::to_string(entries_.size()));
    }
    AtomEntry& entry = entries_[id];
    if (entry.id != id) {
        throw std::runtime_error(std::string(caller) + ": id/index mismatch for atom id=" +
                                 std::to_string(id) + ", entry.id=" + std::to_string(entry.id) +
                                 "; entries_ indexing is corrupt");
    }
    return entry;
}

void AtomTable::setOrigin(uint32_t id, AtomOrigin origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    entryForIdLocked(id, "AtomTable::setOrigin").origin = origin;
}

void AtomTable::setConfidence(uint32_t id, float confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    AtomEntry& entry = entryForIdLocked(id, "AtomTable::setConfidence");
    entry.confidence = std::clamp(confidence, 0.0f, 1.0f);
    if (entry.arg_number.has_value()) {
        entry.arg_number->confidence = entry.confidence;
    }
}

void AtomTable::setCategory(uint32_t id, AtomCategory category) {
    std::lock_guard<std::mutex> lock(mutex_);
    entryForIdLocked(id, "AtomTable::setCategory").category = category;
}

//--------------------------------------------------//
// GPU Packing (numeric side channels plus type-specific flags)
//--------------------------------------------------//

void AtomTable::packValue(AtomEntry& entry,
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
        else if constexpr (std::is_same_v<T, AtomBoolean>) {
            entry.flags = arg.value ? 1u : 0u;
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

    // Pack legacy packed-float / flags / types into host arrays for transfer.
    std::vector<float> h_numeric(num_atoms);
    std::vector<uint32_t> h_flags(num_atoms);
    std::vector<uint32_t> h_types(num_atoms);
    for (size_t i = 0; i < num_atoms; ++i) {
        const AtomEntry& entry = entries_[i];
        h_numeric[i] = entry.numeric_value;
        h_flags[i] = entry.flags;
        h_types[i] = static_cast<uint32_t>(entry.type);
    }

    // Transactional upload: allocate and copy into a fresh temporary, then
    // synchronize the stream before swapping it in (the host vectors above are
    // local transaction buffers, so the copies must complete before returning).
    GPUAtomData temp{};
    struct TransferSpec {
        void** device_ptr;
        const void* host_src;
        size_t bytes;
    };
    const TransferSpec transfers[] = {
        {reinterpret_cast<void**>(&temp.d_numeric_values), h_numeric.data(), num_atoms * sizeof(float)},
        {reinterpret_cast<void**>(&temp.d_numeric_float_values), numeric_float_values_.data(), num_atoms * sizeof(double)},
        {reinterpret_cast<void**>(&temp.d_numeric_int_values), numeric_int_values_.data(), num_atoms * sizeof(int64_t)},
        {reinterpret_cast<void**>(&temp.d_numeric_kind), numeric_kinds_.data(), num_atoms * sizeof(uint8_t)},
        {reinterpret_cast<void**>(&temp.d_flags), h_flags.data(), num_atoms * sizeof(uint32_t)},
        {reinterpret_cast<void**>(&temp.d_types), h_types.data(), num_atoms * sizeof(uint32_t)},
    };
    for (const TransferSpec& transfer : transfers) {
        if (cudaMalloc(transfer.device_ptr, transfer.bytes) != cudaSuccess ||
            cudaMemcpyAsync(*transfer.device_ptr, transfer.host_src, transfer.bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            freeGPUData(temp);
            return false;
        }
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
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

