//======================================================//
//  SequenceLocalAtomTable.hpp
//  Transient typed atom references scoped to one sequence
//======================================================//

#pragma once

#include "TokenLayout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct SequenceLocalAtomAddress {
    AtomType type = AtomType::ATOM_INT;
    uint32_t local_index = kLocalAtomIndexNone;
};

// A sequence-local value namespace. Each AtomType owns an independent dense
// index space, so the complete address of a local value is (type, local_index).
// Exact repeated values of the same type reuse their existing local index.
// This table is intentionally independent of AtomTable and atom_entry_id.
class SequenceLocalAtomTable {
public:
    SequenceLocalAtomAddress ticket(AtomType type, std::string_view raw_text);

    std::optional<uint32_t> find(AtomType type, std::string_view raw_text) const;
    std::optional<std::string> getRawText(AtomType type, uint32_t local_index) const;
    bool contains(AtomType type, uint32_t local_index) const;

    size_t size(AtomType type) const;
    size_t totalSize() const;
    bool empty() const { return totalSize() == 0; }
    void clear();

    void serializeToStreamOrThrow(std::ostream& stream, const char* sink) const;
    void deserializeFromStreamOrThrow(std::istream& stream, const char* source);

private:
    using ValuesByType = std::array<std::vector<std::string>, kAtomTypeCount>;
    using LookupByType =
        std::array<std::unordered_map<std::string, uint32_t>, kAtomTypeCount>;

    ValuesByType values_by_type_;
    LookupByType lookup_by_type_;
};

} // namespace Tokenizer
} // namespace GRIM
