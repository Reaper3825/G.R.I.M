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

struct ArgData {
    
};

struct StructuralSpan {
    size_t start = 0;       // Start position in text (may include leading whitespace)
    size_t end = 0;         // End position (exclusive)
    AtomType atom_type = AtomType::ATOM_INT; // Meaningful only for placeholder-emitting structures
    uint32_t atom_entry_id = kAtomEntryNone; // Per-sequence AtomTable entry ID once registered
    uint32_t local_atom_index = kLocalAtomIndexNone; // Index within this type's sequence-local table

    const char* buffer_ptr = nullptr; // Pointer to original text buffer
    uint32_t offset = 0;              // Offset in buffer
    uint32_t length = 0;              // Length of span (end - start)

    uint32_t content_offset = 0; // Offset to content
    uint32_t content_length = 0; // Length of content

    int open_token_id = -1;  // Typed opening boundary token, e.g. <INT>
    int close_token_id = -1; // Matching typed closing boundary token, e.g. </INT>
};

} // namespace Tokenizer
} // namespace GRIM
