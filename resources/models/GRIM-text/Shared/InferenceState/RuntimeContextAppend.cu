#include "RuntimeContextAppend.hpp"
#include "InferenceGenerationSession.hpp"
#include "../UnigramByte/UniByte.hpp"
#include "../UnigramByte/AtomTable.hpp"
#include "../UnigramByte/SequenceLocalAtomTable.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace GRIM::Inference {

std::size_t appendRuntimeContext(InferenceGenerationSession& session,
    const Tokenizer::UniByteResult& encoded, const HyperParameters::ModelHP& model_hp,
    int vocab_size) {
    constexpr const char* caller = "appendRuntimeContext";
    const auto& sequence = session.sequence;
    if (sequence.finished || sequence.return_reason != InferenceReturnReason::SpanBoundary)
        throw std::logic_error("appendRuntimeContext: session is not span-paused");
    encoded.validate(caller);
    const auto count = encoded.token_ids.size();
    const int remaining = remainingContextTokenCapacity(session, model_hp);
    if (sampledTokenCount(session) >= session.config.max_new_tokens || remaining <= 0 ||
        count >= static_cast<std::size_t>(remaining))
        throw std::length_error("appendRuntimeContext: insufficient continuation capacity");
    const auto size = sequence.token_ids.size();
    if (sequence.token_origins.size() != size || sequence.token_numeric_values.size() != size ||
        sequence.token_atom_mask.size() != size || sequence.atom_entry_ids.size() != size ||
        sequence.token_local_atom_indices.size() != size)
        throw std::logic_error("appendRuntimeContext: trajectory channels are misaligned");
    if (!count) return 0;
    if (!sequence.context_atom_table || !sequence.context_local_atom_table)
        throw std::logic_error("appendRuntimeContext: session atom tables are missing");

    // Stage every allocation and registry change before publishing. Shared
    // tables on prior returned snapshots must retain their original contents.
    auto next = sequence;
    auto atoms = sequence.context_atom_table->cloneHostForGeneration();
    auto local = std::make_shared<Tokenizer::SequenceLocalAtomTable>(
        *sequence.context_local_atom_table);
    bool inside_atom = false;
    Tokenizer::AtomType active_type{};
    for (std::size_t i = 0; i < count; ++i) {
        const int id = encoded.token_ids[i];
        if (id < 0 || id >= vocab_size)
            throw std::invalid_argument("appendRuntimeContext: token outside vocabulary");
        auto entry = Tokenizer::kAtomEntryNone;
        auto index = Tokenizer::kLocalAtomIndexNone;
        if (Tokenizer::isAtomOpenTokenId(id)) {
            if (inside_atom) throw std::invalid_argument("appendRuntimeContext: nested atom");
            inside_atom = true;
            active_type = Tokenizer::tokenIdToAtomType(id);
            const auto raw = encoded.local_atom_table->getRawText(
                active_type, encoded.token_local_atom_indices[i]);
            if (!raw || raw->empty() || raw->size() > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("appendRuntimeContext: invalid atom value");
            Tokenizer::StructuralSpan span;
            span.atom_type = active_type;
            span.buffer_ptr = raw->data();
            span.end = span.length = span.content_length = static_cast<uint32_t>(raw->size());
            entry = atoms->registerSpan(span);
            index = local->ticket(active_type, *raw).local_index;
        } else if (Tokenizer::isAtomCloseTokenId(id)) {
            if (!inside_atom || id != Tokenizer::atomTypeToCloseTokenId(active_type))
                throw std::invalid_argument("appendRuntimeContext: unmatched atom close");
            inside_atom = false;
        }
        next.token_ids.push_back(id);
        next.token_origins.push_back(TokenOrigin::RuntimeContext);
        next.token_numeric_values.push_back(encoded.token_numeric_values[i]);
        next.token_atom_mask.push_back(encoded.token_atom_mask[i]);
        next.atom_entry_ids.push_back(entry);
        next.token_local_atom_indices.push_back(index);
    }
    if (inside_atom) throw std::invalid_argument("appendRuntimeContext: incomplete atom");
    next.context_atom_table = std::move(atoms);
    next.context_local_atom_table = std::move(local);
    session.sequence = std::move(next);
    return count;
}

} // namespace GRIM::Inference
