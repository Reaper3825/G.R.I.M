//======================================================//
//  StructuralSpan.hpp
//  Base StructuralSpan declaration for tokenizer atom spans
//======================================================//

#pragma once

#include "../TokenLayout.hpp"

#include <cstddef>
#include <cstdint>

namespace GRIM {
namespace Tokenizer {

struct StructuralSpan {
    size_t start;           // Start position in text (may include leading whitespace)
    size_t end;             // End position (exclusive)
    AtomType atom_type = AtomType::ATOM_INT; // Meaningful only for placeholder-emitting structures
    uint32_t atom_entry_id = kAtomEntryNone; // Per-sequence AtomTable entry ID once registered

    const char* buffer_ptr; // Pointer to original text buffer
    uint32_t offset;        // Offset in buffer
    uint32_t length;        // Length of span (end - start)

    uint32_t content_offset; // Offset to content
    uint32_t content_length; // Length of content

    int placeholder_id = -1; // Token ID of placeholder when this structure emits an atom
};

} // namespace Tokenizer
} // namespace GRIM
