#pragma once

#include "../UnigramByte/AtomTable.hpp"
#include "../UnigramByte/SequenceLocalAtomTable.hpp"
#include "../Execution/ExecutionMetadata.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GRIM { struct Goal; }
namespace GRIM { struct ConceptBlockSpans; }

namespace GRIM::TokenizerArtifacts {

struct GrmtSequence {
    // Opaque source identity, persisted in GRMT and inherited by every window.
    // Scheduling metadata only; never tokenized or used as a model target.
    std::string concept_block_id;
    std::vector<int> token_ids;
    std::vector<int> targets;
    // Numeric atom metadata is aligned to token_ids but populated only at a
    // typed atom opening boundary. Span content and closing boundaries carry
    // zero/none values in every atom side channel.
    std::vector<float> token_numeric_values;
    std::vector<std::uint8_t> token_atom_mask;
    std::vector<std::uint32_t> token_atom_flags;
    std::shared_ptr<GRIM::Tokenizer::AtomTable> atom_table;
    std::vector<std::uint32_t> atom_entry_ids;
    // Independent transient address plane for within-sequence references.
    // The opening token supplies AtomType; this channel supplies only the
    // corresponding type-local index. It is metadata, never a supervision target.
    std::shared_ptr<GRIM::Tokenizer::SequenceLocalAtomTable> local_atom_table;
    std::vector<std::uint32_t> token_local_atom_indices;
    // Runtime-only causal span mask authored after BOS/EOS insertion and
    // sliding-window construction. For <TYPE> value </TYPE>, positions from
    // the opening boundary through the final value token are 1; BatchPayload
    // suppresses LM supervision on those rows. The close-boundary position is
    // 0 because LM supervision resumes there. This derived channel is never
    // serialized in GRMT.
    std::vector<std::uint8_t> token_atom_aux_target_mask;
    // Compiled dense runtime indices only. Semantic identities are carried by
    // compiled_slot_bindings and never inferred from these tensor addresses.
    std::vector<std::int32_t> token_exec_slot_indices;

    bool execution_active = false;
    GRIM::Execution::ExecutionGateTarget execution_gate_target =
        GRIM::Execution::ExecutionGateTarget::UNSUPERVISED;
    // Functional prompt geometry. In SFT this spans every token before the
    // answer, including model-visible goal/context fields; it is not limited
    // to the canonical renderer's literal <prompt> byte span.
    std::int32_t prompt_end_pos = -1;
    std::int32_t prompt_length = 0;
    std::vector<GRIM::Execution::CompiledSlotBinding> compiled_slot_bindings;
    std::vector<GRIM::Execution::CompiledTransitionBinding> compiled_transition_bindings;
    std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings;
    std::vector<GRIM::Execution::TransitionInvocation> transition_targets;

    // Authored row-level goal metadata. Shared ownership lets sliding-window
    // rows retain one immutable Goal without copying its runtime Tensor state.
    std::shared_ptr<const GRIM::Goal> goal;

    // Authored top-level ConceptBlock known/unknown metadata. This remains
    // independent of Goal while sharing the same immutable row lifetime.
    std::shared_ptr<const GRIM::ConceptBlockSpans> concept_block_spans;

    bool hasAnyValidTarget() const;
    void validateForWrite(const std::string& source) const;
};

} // namespace GRIM::TokenizerArtifacts
