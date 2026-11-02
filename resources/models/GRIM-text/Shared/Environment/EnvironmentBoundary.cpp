#include "EnvironmentBoundary.hpp"
#include "../UnigramByte/TokenLayout.hpp"
#include <stdexcept>

namespace GRIM::Environment {
EnvironmentBoundary inspectEnvironmentBoundary(const GeneratedSequence& sequence,
    std::size_t prompt_tokens, std::optional<std::size_t> previous_close) {
    if (sequence.return_reason != Inference::InferenceReturnReason::SpanBoundary ||
        sequence.finished || !sequence.concept_span_delimiters || sequence.span_events.empty() ||
        sequence.token_origins.size() != sequence.token_ids.size())
        throw std::invalid_argument("Environment generation did not deliver a valid span pause");
    const auto& event = sequence.span_events.back();
    const auto& definitions = sequence.concept_span_delimiters->nodes();
    if (event.kind != GeneratedSpanEventKind::Close || event.definition >= definitions.size() ||
        event.begin < prompt_tokens || event.begin >= event.token_position ||
        event.token_position >= sequence.token_ids.size() ||
        event.token_position + 1 != sequence.token_ids.size() ||
        (previous_close && event.begin <= *previous_close))
        throw std::invalid_argument("Environment span boundary is incomplete, stale, or out of range");
    for (const auto& item : sequence.span_events)
        if ((!previous_close || item.token_position > *previous_close) &&
            item.kind == GeneratedSpanEventKind::InvalidMarker)
            throw std::invalid_argument("Environment continuation contains an invalid structural marker");
    const auto& definition = definitions[event.definition];
    if (definition.parent != kNoNamedConceptSpanEntry ||
        (definition.name != "determine" && definition.name != "define" && definition.name != "execute" &&
         definition.name != "update" && definition.name != "answer") ||
        sequence.token_ids[event.begin] != definition.open_id ||
        sequence.token_ids[event.token_position] != definition.close_id ||
        sequence.token_origins[event.begin] != TokenOrigin::SampledAction ||
        sequence.token_origins[event.token_position] != TokenOrigin::SampledAction)
        throw std::invalid_argument("Environment boundary is not a generated DDEA span");
    for (std::size_t p = event.begin; p <= event.token_position; ++p)
        if (sequence.token_origins[p] != TokenOrigin::SampledAction &&
            sequence.token_origins[p] != TokenOrigin::ForcedToken)
            throw std::invalid_argument("Environment generated span contains injected context");

    std::optional<Tokenizer::AtomType> atom;
    for (std::size_t p = prompt_tokens; p < sequence.token_ids.size(); ++p) {
        const int id = sequence.token_ids[p];
        if (Tokenizer::isAtomOpenTokenId(id)) {
            if (atom) throw std::invalid_argument("Environment continuation contains a nested atom");
            atom = Tokenizer::tokenIdToAtomType(id);
        } else if (Tokenizer::isAtomCloseTokenId(id)) {
            if (!atom || id != Tokenizer::atomTypeToCloseTokenId(*atom))
                throw std::invalid_argument("Environment continuation contains an unmatched atom close");
            atom.reset();
        }
    }
    if (atom) throw std::invalid_argument("Environment span closed inside an unfinished atom");
    return {definition.name, event.begin, event.token_position + 1};
}
}
