#pragma once
#include "../../../Shared/ConceptBlock/NamedConceptSpans.hpp"
#include <string>
namespace GRIM::TokenizerArtifacts { struct GrmtSequence; }
namespace GRIMText::Training {
// Resolve inherited policy over the complete tree, selectively retain tokens,
// and author causal targets before window construction. No field-name cases.
bool projectConceptSupervision(
    GRIM::TokenizerArtifacts::GrmtSequence& sequence,
    const GRIM::NamedConceptSpanDefinitions& definitions,
    const std::string& source);
}
