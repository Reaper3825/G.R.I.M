#pragma once
//======================================================//
//  Startup/InitFacts.hpp
//
//  Init-time structural-invariant verification + LogRecorder/telemetry emission.
//  Replaces the inline "[TieEmbeddingsVerify]" block that
//  previously lived in Phase1_Startup.cu.
//
//  One-shot init facts are emitted as a full key/value dump through the
//  LogRecorder text sink in the shared session log (`training_<session>.log`). The compact
//  bool/count mirror is also published through TELEMETRY lattice streams
//  INIT_TIE_CFG..INIT_OPT_GROUPS_LM (indices 48-54). Those are written to
//  ctx.telemetry.last_obs[48..54]; Phase2's per-step lattice update naturally
//  flushes them into telemetry_<session>.csv. They are constant for the run,
//  so mu == value and sigma == 0.
//
//  Assertion-side: tie/grad pointer mismatches and over-counted
//  parameter groups against the tied buffer become hard
//  std::runtime_error throws (Rule 20: fail loud, never log-and-continue).
//
//  Must be called AFTER Startup/Model parameter-group registration; reads
//  pointer-equality + group counts from the live model.
//======================================================//

#include "../Phase1_Startup.hpp"  // GRIMText::Training::TrainingContext

namespace GRIMText::Training {

// Verify init-time structural invariants of the constructed model, dump them
// to LogRecorder, and emit the compact telemetry mirror.
//
// Side effects:
//   - Writes a full `[INIT_FACTS]` key/value block to LogRecorder tape
//   - Writes ctx.telemetry.last_obs[48..54] (constant for run)
//   - Throws std::runtime_error on:
//       * config.tie_embeddings != (emb_weight_ptr == lm_weight_ptr)
//       * config.tie_embeddings && (emb_grad_ptr  != lm_grad_ptr)
//       * config.tie_embeddings && optimizer references emb buffer
//         separately from lm buffer (would double-step the tied weights)
//
// Pre-conditions:
//   - ctx.model is fully constructed and parameter-group registration has run
void verifyAndDumpInitFacts(TrainingContext& ctx);

} // namespace GRIMText::Training
