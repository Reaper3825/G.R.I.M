#pragma once
//======================================================//
//  Startup/Payload/PayloadBuildInputs.hpp
//
//  Phase1-authored snapshot of the static inputs to
//  GRIM::Batching::buildBatchPayload.
//
//  RATIONALE
//  =========
//  The fields that buildBatchPayload(...) needs from "the run" — model cache
//  geometry, execution-block sizes, vocab size, token layout, MTP head count —
//  are all run-invariant. Phase2 used to re-read them per batch from
//  the LanguageModel config accessor and a re-derived fixed-shape grouping, AND re-check the invariant
//  that the model and the capacity stem agreed on cache dimensions. Both the
//  reads and the contract check are properly Phase1's job:
//
//    - The reads are static: nothing about them changes between batches.
//    - The contract check is a Phase1 handoff invariant ("if the model and
//      capacity stem disagree, that is a startup contract bug, not something
//      Phase2 should hide" — verbatim from the old per-batch check).
//
//  This module snapshots those values once at startup and crashes loudly if
//  the contract is violated. Phase2's payload builder then becomes a thin
//  consumer of ctx.payload_build_inputs.
//======================================================//

#include <cstddef>

namespace GRIMText::Training {

struct TrainingContext;

// Snapshot of GRIM::Tokenizer::TokenLayout's 4 size fields. Held inline as a
// POD so this header does not need to include UniByte.hpp (which transitively
// pulls cuda_runtime.h). Phase2 reconstructs a real TokenLayout from these
// values at the buildBatchPayload call site.
struct TokenLayoutSnapshot {
    int num_special = 0;
    int num_bytes   = 0;
    int num_atoms   = 0;
    int num_unigram = 0;
};

struct PayloadBuildInputs {
    std::size_t configured_batch_size = 0;
    std::size_t max_cached_seq   = 0;
    int execution_block_num_slots = 0;
    int execution_block_num_ops   = 0;
    int execution_block_num_steps = 0;
    int vocab_size = 0;
    int train_mtp_k = 0;  // Effective MTP head count for training (0 if MTP disabled). Validation always uses 0.
    TokenLayoutSnapshot token_layout{};
};

// Snapshots run-invariant payload inputs and re-validates the model ↔
// grouped fixed-shape cache-dimension contract. Throws on any mismatch (Rule 20).
// Requires ctx.model, ctx.config, and ctx.tokenizer to be initialized — i.e.
// must run after ModelAllocated.
PayloadBuildInputs derivePayloadBuildInputsOrThrow(const TrainingContext& ctx);

void PayloadBuildInputsReady(TrainingContext& ctx);

} // namespace GRIMText::Training
