//======================================================//
//  AtomInsertionDecode.hpp
//  Request-local OPEN-type + EXIT span state machine
//======================================================//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::AtomInsertion {

// Decodes compact [gap_count, kAtomDecisionClassCount] logits. Span state is
// request-local: an OPEN decision persists its type until the type-free EXIT
// decision fires, and the matching close token is rendered from that type.
std::string decodeAtomDecisionPredictions(
    const std::string& plain_text,
    const std::vector<std::uint8_t>& valid_gap_mask,
    const std::vector<float>& decision_logits,
    float decision_logit);

} // namespace GRIM::AtomInsertion
