#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::TokenizerArtifacts { struct GrmtSequence; }

namespace GRIMText::Training {

enum class ConceptField : std::uint8_t {
    Prompt = 0,
    TargetState,
    SuccessCriteriaAndEvidence,
    Constraints,
    Knowns,
    Unknowns,
    Reasoning,
    Determine,
    Define,
    Execute,
    Update,
    Answer,
    Count,
};

const char* conceptFieldName(ConceptField field);

ConceptField parseConceptField(const std::string& name,
                               const std::string& source);

// Project the configured ConceptBlock field policy onto one complete GRMT
// sequence before boundary injection and window construction. Returns false
// when the row contains none of the configured supervised fields.
bool projectConceptSupervision(
    GRIM::TokenizerArtifacts::GrmtSequence& sequence,
    const std::vector<ConceptField>& supervised_fields,
    const std::vector<ConceptField>& unsupervised_fields,
    const std::string& source);

} // namespace GRIMText::Training
