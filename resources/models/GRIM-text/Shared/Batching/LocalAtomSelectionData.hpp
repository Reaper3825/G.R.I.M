//======================================================//
//  LocalAtomSelectionData.hpp
//  Runtime-only local atom selector metadata authoring
//======================================================//

#pragma once

namespace GRIM {
namespace Batching {

struct BatchPayload;

// Derive compact selector metadata from the final padded token rows and their
// sequence-local typed atom addresses. This is the sole authoring boundary for
// local selector queries, first-occurrence candidate content, and causal
// NO_REFERENCE/reference targets.
void materializeLocalAtomSelectionMetadata(
    BatchPayload& payload,
    const char* caller);

} // namespace Batching
} // namespace GRIM

