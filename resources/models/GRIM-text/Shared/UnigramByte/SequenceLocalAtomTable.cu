//======================================================//
//  SequenceLocalAtomTable.cu
//======================================================//

#include "SequenceLocalAtomTable.hpp"

#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace GRIM {
namespace Tokenizer {
namespace {

constexpr char kStreamMagic[4] = {'S', 'L', 'A', 'T'};
constexpr uint32_t kMaxSerializedValuesPerType = 10000000u;

size_t typeOffset(AtomType type, const char* caller) {
    return static_cast<size_t>(atomTypeIndexOrThrow(type, caller));
}

void requireLabel(const char* label, const char* caller) {
    if (!label || label[0] == '\0') {
        throw std::runtime_error(std::string(caller) + ": empty stream label");
    }
}

void writeExact(std::ostream& stream,
                const void* data,
                size_t bytes,
                const char* sink,
                const char* field) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error(
            std::string("SequenceLocalAtomTable failed writing ") + field + " to " + sink);
    }
}

void readExact(std::istream& stream,
               void* data,
               size_t bytes,
               const char* source,
               const char* field) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error(
            std::string("SequenceLocalAtomTable failed reading ") + field + " from " + source);
    }
}

} // namespace

SequenceLocalAtomAddress SequenceLocalAtomTable::ticket(
    AtomType type,
    std::string_view raw_text) {
    const size_t offset = typeOffset(type, "SequenceLocalAtomTable::ticket");
    auto& lookup = lookup_by_type_[offset];
    const auto existing = lookup.find(std::string(raw_text));
    if (existing != lookup.end()) {
        return SequenceLocalAtomAddress{type, existing->second};
    }

    auto& values = values_by_type_[offset];
    if (values.size() >= static_cast<size_t>(kLocalAtomIndexNone)) {
        throw std::runtime_error(
            "SequenceLocalAtomTable::ticket exhausted the uint32 local index space for type " +
            std::string(atomTypeName(type)));
    }

    const uint32_t local_index = static_cast<uint32_t>(values.size());
    values.emplace_back(raw_text);
    lookup.emplace(values.back(), local_index);
    return SequenceLocalAtomAddress{type, local_index};
}

std::optional<uint32_t> SequenceLocalAtomTable::find(
    AtomType type,
    std::string_view raw_text) const {
    const size_t offset = typeOffset(type, "SequenceLocalAtomTable::find");
    const auto& lookup = lookup_by_type_[offset];
    const auto it = lookup.find(std::string(raw_text));
    if (it == lookup.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<std::string> SequenceLocalAtomTable::getRawText(
    AtomType type,
    uint32_t local_index) const {
    const size_t offset = typeOffset(type, "SequenceLocalAtomTable::getRawText");
    const auto& values = values_by_type_[offset];
    if (local_index == kLocalAtomIndexNone ||
        static_cast<size_t>(local_index) >= values.size()) {
        return std::nullopt;
    }
    return values[local_index];
}

bool SequenceLocalAtomTable::contains(AtomType type, uint32_t local_index) const {
    const size_t offset = typeOffset(type, "SequenceLocalAtomTable::contains");
    return local_index != kLocalAtomIndexNone &&
           static_cast<size_t>(local_index) < values_by_type_[offset].size();
}

size_t SequenceLocalAtomTable::size(AtomType type) const {
    return values_by_type_[typeOffset(type, "SequenceLocalAtomTable::size")].size();
}

size_t SequenceLocalAtomTable::totalSize() const {
    size_t result = 0;
    for (const auto& values : values_by_type_) {
        result += values.size();
    }
    return result;
}

void SequenceLocalAtomTable::clear() {
    for (auto& values : values_by_type_) {
        values.clear();
    }
    for (auto& lookup : lookup_by_type_) {
        lookup.clear();
    }
}

void SequenceLocalAtomTable::serializeToStreamOrThrow(
    std::ostream& stream,
    const char* sink) const {
    requireLabel(sink, "SequenceLocalAtomTable::serializeToStreamOrThrow");
    writeExact(stream, kStreamMagic, sizeof(kStreamMagic), sink, "magic");

    const uint32_t type_count = static_cast<uint32_t>(kAtomTypeCount);
    writeExact(stream, &type_count, sizeof(type_count), sink, "type_count");
    for (size_t type_offset = 0; type_offset < values_by_type_.size(); ++type_offset) {
        const auto& values = values_by_type_[type_offset];
        if (values.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error(
                std::string("SequenceLocalAtomTable has too many values for type offset ") +
                std::to_string(type_offset) + " while writing " + sink);
        }
        const uint32_t value_count = static_cast<uint32_t>(values.size());
        writeExact(stream, &value_count, sizeof(value_count), sink, "value_count");
        for (const std::string& value : values) {
            if (value.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::runtime_error(
                    std::string("SequenceLocalAtomTable value exceeds uint32 length while writing ") + sink);
            }
            const uint32_t length = static_cast<uint32_t>(value.size());
            writeExact(stream, &length, sizeof(length), sink, "value_length");
            if (length > 0) {
                writeExact(stream, value.data(), length, sink, "value_bytes");
            }
        }
    }
}

void SequenceLocalAtomTable::deserializeFromStreamOrThrow(
    std::istream& stream,
    const char* source) {
    requireLabel(source, "SequenceLocalAtomTable::deserializeFromStreamOrThrow");

    char magic[sizeof(kStreamMagic)]{};
    readExact(stream, magic, sizeof(magic), source, "magic");
    if (std::memcmp(magic, kStreamMagic, sizeof(magic)) != 0) {
        throw std::runtime_error(
            std::string("SequenceLocalAtomTable invalid magic in ") + source);
    }

    uint32_t type_count = 0;
    readExact(stream, &type_count, sizeof(type_count), source, "type_count");
    if (type_count != static_cast<uint32_t>(kAtomTypeCount)) {
        throw std::runtime_error(
            std::string("SequenceLocalAtomTable type_count=") +
            std::to_string(type_count) + " != active type count=" +
            std::to_string(kAtomTypeCount) + " in " + source);
    }

    SequenceLocalAtomTable decoded;
    for (uint32_t type_offset = 0; type_offset < type_count; ++type_offset) {
        uint32_t value_count = 0;
        readExact(stream, &value_count, sizeof(value_count), source, "value_count");
        if (value_count > kMaxSerializedValuesPerType) {
            throw std::runtime_error(
                std::string("SequenceLocalAtomTable implausible value_count=") +
                std::to_string(value_count) + " in " + source);
        }
        const AtomType type = static_cast<AtomType>(type_offset);
        for (uint32_t expected_index = 0; expected_index < value_count; ++expected_index) {
            uint32_t length = 0;
            readExact(stream, &length, sizeof(length), source, "value_length");
            std::string value(length, '\0');
            if (length > 0) {
                readExact(stream, value.data(), length, source, "value_bytes");
            }
            const SequenceLocalAtomAddress address = decoded.ticket(type, value);
            if (address.local_index != expected_index) {
                throw std::runtime_error(
                    std::string("SequenceLocalAtomTable duplicate serialized value for type ") +
                    atomTypeName(type) + " in " + source);
            }
        }
    }

    *this = std::move(decoded);
}

} // namespace Tokenizer
} // namespace GRIM
