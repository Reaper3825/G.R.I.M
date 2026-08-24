//======================================================//
//  ByteAtomAlignment.cu
//  Byte-position B/E supervision derived from tokenizer atoms
//======================================================//

#include "ByteAtomAlignment.hpp"

#include "AtomTable.hpp"
#include "TokenLayout.hpp"

#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

namespace {

std::string callerPrefix(const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error("ByteAtomAlignment requires a non-empty caller label");
    }
    return std::string(caller) + ": ByteAtomAlignment";
}

void requireEntryMatchesSpan(const StructuralSpan& span,
                             const AtomTable& atom_table,
                             const std::string& prefix) {
    if (span.atom_entry_id == kAtomEntryNone) {
        throw std::runtime_error(prefix + ": atom span has no AtomTable entry");
    }
    const auto entry = atom_table.getAtom(span.atom_entry_id);
    if (!entry.has_value()) {
        throw std::runtime_error(prefix + ": atom_entry_id=" +
                                 std::to_string(span.atom_entry_id) +
                                 " is absent from AtomTable");
    }
    if (entry->type != span.atom_type) {
        throw std::runtime_error(prefix + ": atom_entry_id=" +
                                 std::to_string(span.atom_entry_id) +
                                 " type does not match StructuralSpan");
    }
}

} // namespace

void ByteAtomAlignment::validate(const std::vector<StructuralSpan>& atoms,
                                 const AtomTable& atom_table,
                                 const char* caller) const {
    const std::string prefix = callerPrefix(caller);
    const std::size_t expected_boundaries = byte_ids.size() + 1;
    if (begin_atom_entry_ids.size() != expected_boundaries) {
        throw std::runtime_error(prefix + ": begin boundary count=" +
                                 std::to_string(begin_atom_entry_ids.size()) +
                                 " != byte count + 1=" +
                                 std::to_string(expected_boundaries));
    }
    if (end_atom_entry_ids.size() != expected_boundaries) {
        throw std::runtime_error(prefix + ": end boundary count=" +
                                 std::to_string(end_atom_entry_ids.size()) +
                                 " != byte count + 1=" +
                                 std::to_string(expected_boundaries));
    }

    for (std::size_t index = 0; index < byte_ids.size(); ++index) {
        if (!isByteTokenId(byte_ids[index])) {
            throw std::runtime_error(prefix + ": byte_ids[" +
                                     std::to_string(index) + "]=" +
                                     std::to_string(byte_ids[index]) +
                                     " is outside the existing byte-token range");
        }
    }

    std::size_t begin_index = 0;
    std::size_t end_index = 0;
    std::vector<std::size_t> begin_slots;
    std::vector<std::size_t> end_slots;
    begin_slots.reserve(atoms.size());
    end_slots.reserve(atoms.size());
    for (std::size_t slot = 0; slot < expected_boundaries; ++slot) {
        const std::uint32_t begin_id = begin_atom_entry_ids[slot];
        if (begin_id != kAtomEntryNone) {
            if (begin_index >= atoms.size() ||
                begin_id != atoms[begin_index].atom_entry_id) {
                throw std::runtime_error(prefix +
                                         ": begin boundaries do not match ordered atom spans at slot=" +
                                         std::to_string(slot));
            }
            requireEntryMatchesSpan(atoms[begin_index], atom_table, prefix);
            begin_slots.push_back(slot);
            ++begin_index;
        }

        const std::uint32_t end_id = end_atom_entry_ids[slot];
        if (end_id != kAtomEntryNone) {
            if (end_index >= atoms.size() ||
                end_id != atoms[end_index].atom_entry_id) {
                throw std::runtime_error(prefix +
                                         ": end boundaries do not match ordered atom spans at slot=" +
                                         std::to_string(slot));
            }
            requireEntryMatchesSpan(atoms[end_index], atom_table, prefix);
            end_slots.push_back(slot);
            ++end_index;
        }
    }

    if (begin_index != atoms.size() || end_index != atoms.size()) {
        throw std::runtime_error(prefix + ": B/E marker count does not match atom count=" +
                                 std::to_string(atoms.size()));
    }
    for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
        if (begin_slots[atom_index] > end_slots[atom_index]) {
            throw std::runtime_error(prefix + ": atom end precedes begin for occurrence=" +
                                     std::to_string(atom_index));
        }
        if (atom_index > 0 && end_slots[atom_index - 1] > begin_slots[atom_index]) {
            throw std::runtime_error(prefix + ": atom boundary slots overlap at occurrence=" +
                                     std::to_string(atom_index));
        }
    }
}

ByteAtomAlignment buildByteAtomAlignment(
    std::string_view annotated_source,
    const std::vector<StructuralSpan>& atoms,
    const AtomTable& atom_table,
    const char* caller) {
    const std::string prefix = callerPrefix(caller);
    ByteAtomAlignment alignment;
    alignment.byte_ids.reserve(annotated_source.size());
    alignment.begin_atom_entry_ids.push_back(kAtomEntryNone);
    alignment.end_atom_entry_ids.push_back(kAtomEntryNone);

    auto appendBytes = [&](std::size_t begin, std::size_t end) {
        if (begin > end || end > annotated_source.size()) {
            throw std::runtime_error(prefix + ": source byte range is out of bounds");
        }
        for (std::size_t index = begin; index < end; ++index) {
            const auto byte_value = static_cast<std::uint8_t>(
                static_cast<unsigned char>(annotated_source[index]));
            alignment.byte_ids.push_back(byteToTokenId(byte_value));
            alignment.begin_atom_entry_ids.push_back(kAtomEntryNone);
            alignment.end_atom_entry_ids.push_back(kAtomEntryNone);
        }
    };

    std::size_t source_cursor = 0;
    for (const StructuralSpan& span : atoms) {
        const std::size_t content_begin = static_cast<std::size_t>(span.content_offset);
        const std::size_t content_end = content_begin +
            static_cast<std::size_t>(span.content_length);
        if (span.start < source_cursor || span.start > span.end ||
            span.end > annotated_source.size() ||
            content_begin < span.start || content_end < content_begin ||
            content_end > span.end) {
            throw std::runtime_error(prefix +
                                     ": atom spans must be ordered, non-overlapping, and contained in source text");
        }
        requireEntryMatchesSpan(span, atom_table, prefix);

        appendBytes(source_cursor, span.start);
        const std::size_t begin_slot = alignment.byte_ids.size();
        if (alignment.begin_atom_entry_ids[begin_slot] != kAtomEntryNone) {
            throw std::runtime_error(prefix + ": multiple atom beginnings share boundary slot=" +
                                     std::to_string(begin_slot));
        }
        alignment.begin_atom_entry_ids[begin_slot] = span.atom_entry_id;

        // Keep only model-visible atom content. Authored opening/closing tags
        // and trimmed non-string padding inside those tags are supervision, not
        // bytes that the identifier should see at inference time.
        appendBytes(content_begin, content_end);
        const std::size_t end_slot = alignment.byte_ids.size();
        if (alignment.end_atom_entry_ids[end_slot] != kAtomEntryNone) {
            throw std::runtime_error(prefix + ": multiple atom endings share boundary slot=" +
                                     std::to_string(end_slot));
        }
        alignment.end_atom_entry_ids[end_slot] = span.atom_entry_id;
        source_cursor = span.end;
    }
    appendBytes(source_cursor, annotated_source.size());

    alignment.validate(atoms, atom_table, caller);
    return alignment;
}

} // namespace Tokenizer
} // namespace GRIM
