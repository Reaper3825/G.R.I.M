//======================================================//
//  AtomInsertionDecisionLayout.hpp
//  Task-local atom span decision classes
//======================================================//

#pragma once

#include "../UnigramByte/TokenLayout.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM::AtomInsertion {

// Atom identification is a span-state task, not a ten-way tokenizer-token
// prediction task. An opening decision carries the atom type; EXIT is
// deliberately type-free because inference renders the matching close token
// from the type persisted by the active span state.
inline constexpr int kOpenDecisionClassCount = Tokenizer::kAtomTypeCount;
inline constexpr int kExitDecisionClassIndex = kOpenDecisionClassCount;
inline constexpr int kAtomDecisionClassCount = kOpenDecisionClassCount + 1;

// The existing full-vocabulary LM head remains the parameter owner. The atom
// task consumes a compact contiguous window beginning at ATOM_TOKEN_OFFSET.
// The final class is task-local EXIT; it must not be interpreted as a tokenizer
// close token even though it occupies the first close-token row in the shared head.
inline constexpr int kAtomDecisionVocabColumnOffset =
    Tokenizer::ATOM_TOKEN_OFFSET;

static_assert(kAtomDecisionClassCount <= Tokenizer::ATOM_VOCAB_SIZE,
              "atom decisions must fit inside the reserved atom-token rows");
static_assert(kAtomDecisionVocabColumnOffset + kExitDecisionClassIndex ==
                  Tokenizer::ATOM_CLOSE_TOKEN_OFFSET,
              "task-local EXIT must occupy the first reserved close-token row");

inline int openDecisionClassIndexOrThrow(Tokenizer::AtomType type,
                                         const char* caller) {
    const int type_index = Tokenizer::atomTypeIndexOrThrow(type, caller);
    if (type_index < 0 || type_index >= kOpenDecisionClassCount) {
        throw std::runtime_error(
            std::string(caller) + ": atom type index is outside open decisions");
    }
    return type_index;
}

inline int decisionVocabColumnOrThrow(int decision_class,
                                      const char* caller) {
    if (decision_class < 0 || decision_class >= kAtomDecisionClassCount) {
        throw std::runtime_error(
            std::string(caller) + ": decision class=" +
            std::to_string(decision_class) + " is outside [0," +
            std::to_string(kAtomDecisionClassCount) + ")");
    }
    return kAtomDecisionVocabColumnOffset + decision_class;
}

} // namespace GRIM::AtomInsertion
