#pragma once
//======================================================//
//  Startup/InitFacts.hpp
//
//  Init-time structural-invariant verification + dual emission.
//  Replaces the inline "[TieEmbeddingsVerify]" block that
//  previously lived in Phase1_Startup.cu.
//
//  Two-channel emission for one-shot init facts:
//
//    1. TELEMETRY (lattice streams INIT_TIE_CFG..INIT_OPT_GROUPS_LM,
//       indices 48-54) — float-compatible bools and counts. Written
//       to ctx.telemetry.last_obs[48..54]; Phase2's per-step lattice
//       update naturally flushes them into telemetry_<session>.csv.
//       Constant for the run, so mu == value and sigma == 0.
//
//    2. INIT FACTS CSV (init_facts_<session>.csv next to the existing
//       training_<session>.log and telemetry_<session>.csv) — captures
//       the same bools/counts AND the raw pointer values, which can't
//       fit in a float stream. Two columns: key,value.
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

// Verify init-time structural invariants of the constructed model and
// emit them via both telemetry stream slots and a sibling CSV file.
//
// Side effects:
//   - Writes ctx.telemetry.last_obs[48..54] (constant for run)
//   - Creates <log_dir>/init_facts_<session_id>.csv with key,value rows
//   - Throws std::runtime_error on:
//       * config.tie_embeddings != (emb_weight_ptr == lm_weight_ptr)
//       * config.tie_embeddings && (emb_grad_ptr  != lm_grad_ptr)
//       * config.tie_embeddings && optimizer references emb buffer
//         separately from lm buffer (would double-step the tied weights)
//
// Pre-conditions:
//   - ctx.model is fully constructed and parameter-group registration has run
//   - ctx.logging.session_id is set
//   - ctx.config.paths.log_dir exists (created in earlier Phase1 step)
void verifyAndDumpInitFacts(TrainingContext& ctx);

} // namespace GRIMText::Training
