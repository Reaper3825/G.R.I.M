//======================================================//
//  ByteAtomAlignment.hpp
//  Byte-position B/E supervision derived from tokenizer atoms
//======================================================//

#pragma once

#include "Detectors/StructuralSpan.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace GRIM {
namespace Tokenizer {

class AtomTable;

// A de-annotated byte view for atom-boundary identification. Authored typed
// delimiters are omitted, while atom content and ordinary source bytes remain.
// Boundary arrays address the slots around bytes and therefore have N + 1
// entries for N byte IDs. This represents exclusive ends and empty strings
// without inventing another vocabulary or copying AtomTable payloads.
struct ByteAtomAlignment {
    std::vector<int> byte_ids;
    std::vector<std::uint32_t> begin_atom_entry_ids;
    std::vector<std::uint32_t> end_atom_entry_ids;

    std::size_t byteSize() const noexcept { return byte_ids.size(); }
    std::size_t boundarySize() const noexcept { return begin_atom_entry_ids.size(); }
    bool empty() const noexcept { return byte_ids.empty(); }

    void validate(const std::vector<StructuralSpan>& atoms,
                  const AtomTable& atom_table,
                  const char* caller) const;
};

ByteAtomAlignment buildByteAtomAlignment(
    std::string_view annotated_source,
    const std::vector<StructuralSpan>& atoms,
    const AtomTable& atom_table,
    const char* caller);

} // namespace Tokenizer
} // namespace GRIM
