//======================================================//
//  ConceptGenerator — deterministic ConceptBlock assembly.
//
//  Models read raw documents and fill a typed binding set.
//  This class owns only resource parsing, validation, and
//  deterministic rendering. Model transport is provided by
//  IConceptBindingAdapter.
//======================================================//

#pragma once

#include "concept_block.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace GRIM::DataCollection {

enum class GeneratorValueType : uint8_t {
    Text,
    Integer,
    Float,
    Boolean,
    String,
    Mixed
};

enum class GeneratorSlotType : uint8_t {
    Text,
    TextArray,
    EvidenceArray,
    TypedAtomArray
};

enum class GeneratorSegmentKind : uint8_t {
    Literal,
    Lexicon,
    Binding
};

enum class GeneratorTraversal : uint8_t {
    SeededRandom,
    RoundRobin
};

struct GeneratorLexiconEntry {
    std::string        text;
    GeneratorValueType type = GeneratorValueType::Text;
};

struct GeneratorLexiconGroup {
    std::string                        id;
    std::string                        label;
    GeneratorValueType                 type = GeneratorValueType::Text;
    bool                               required = false;
    std::vector<GeneratorLexiconEntry> entries;
};

struct GeneratorSlotDefinition {
    std::string       id;
    GeneratorSlotType type = GeneratorSlotType::Text;
    std::string       description;
    bool              required = false;
    bool              grounded = false;
    size_t            minItems = 0;
    size_t            maxItems = 0;
};

struct GeneratorFrameSegment {
    GeneratorSegmentKind kind = GeneratorSegmentKind::Literal;
    std::string          key;
    std::string          text;
    std::string          fallback;
    std::string          prefix;
    std::string          suffix;
    bool                 optional = false;
};

struct GeneratorFieldTemplate {
    std::string                        target;
    std::vector<GeneratorFrameSegment> segments;
};

struct GeneratorFrameDefinition {
    std::string                          id;
    std::string                          label;
    std::string                          formatType = "chain_of_thought";
    std::vector<GeneratorSlotDefinition> slots;
    std::vector<GeneratorFieldTemplate>  fields;
};

// Raw document/dataset row presented to an adapter. The generator never
// expects callers to pre-fill semantic slots inside this input contract.
struct GeneratorDocument {
    std::string id;
    std::string title;
    std::string text;
    std::string metadataJson;
};

struct GeneratorEvidenceBinding {
    std::string criterion;
    std::string quote;
};

struct GeneratorAtomBinding {
    std::string        id;
    GeneratorValueType type = GeneratorValueType::Text;
    std::string        value;
    std::string        unit;
    std::string        sourceQuote;
    bool               derived = false;
};

// Stable adapter output contract. Ollama fills this today; a native GRIM
// adapter can fill the same type later, including memory-grounded context.
struct GeneratorBindingSet {
    std::string                           documentId;
    std::string                           adapterId;
    std::string                           question;
    std::string                           targetState;
    std::vector<std::string>              reasoningSteps;
    std::string                           answer;
    std::vector<GeneratorEvidenceBinding> evidence;
    std::vector<GeneratorAtomBinding>     atoms;
    std::string                           rawResponse;
    bool                                  valid = false;
    std::string                           error;
};

struct GeneratorSelection {
    std::string        groupId;
    size_t             entryIndex = 0;
    GeneratorValueType type = GeneratorValueType::Text;
    std::string        text;
};

struct GeneratedConceptFrame {
    GRIM::ConceptBlock              conceptBlock;
    uint64_t                        seed = 0;
    uint64_t                        iteration = 0;
    bool                            sourceBound = false;
    std::vector<GeneratorSelection> selections;
};

class ConceptGenerator {
public:
    bool loadCatalog(const std::filesystem::path& path);
    bool loadDataSource(const std::filesystem::path& path);

    const std::vector<GeneratorLexiconGroup>& groups() const { return groups_; }
    const std::vector<GeneratorFrameDefinition>& frames() const { return frames_; }
    const std::vector<GeneratorDocument>& documents() const { return documents_; }
    const std::string& lastError() const { return lastError_; }
    uint32_t schemaVersion() const { return schemaVersion_; }

    bool validateBindings(const GeneratorDocument& document,
                          const GeneratorFrameDefinition& frame,
                          GeneratorBindingSet& bindings) const;

    GeneratedConceptFrame assemble(size_t frameIndex,
                                   const GeneratorBindingSet& bindings,
                                   uint64_t baseSeed,
                                   uint64_t iteration,
                                   GeneratorTraversal traversal) const;

    static const char* valueTypeName(GeneratorValueType type);
    static const char* slotTypeName(GeneratorSlotType type);
    static std::string renderAtom(const GeneratorAtomBinding& atom);

private:
    const GeneratorLexiconGroup* findGroup(const std::string& id) const;
    const GeneratorLexiconEntry* selectEntry(const GeneratorLexiconGroup& group,
                                             uint64_t baseSeed,
                                             uint64_t iteration,
                                             size_t groupOrdinal,
                                             GeneratorTraversal traversal,
                                             size_t& selectedIndex) const;
    std::string renderField(const GeneratorFrameDefinition& frame,
                            const std::string& target,
                            const GeneratorBindingSet& bindings,
                            uint64_t baseSeed,
                            uint64_t iteration,
                            GeneratorTraversal traversal,
                            std::vector<GeneratorSelection>& selections) const;

    uint32_t                              schemaVersion_ = 0;
    std::vector<GeneratorLexiconGroup>    groups_;
    std::vector<GeneratorFrameDefinition> frames_;
    std::vector<GeneratorDocument>        documents_;
    std::string                           lastError_;
};

} // namespace GRIM::DataCollection
