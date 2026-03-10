# Training Loop Sync Optimization — Verbose Logging Gate (Addendum)

## Goal

Gate **argmax/collapse logging** and **all other extensive, costly logs** behind a single **verbose logging** flag. Default = off so the hot path stays minimal; when enabled, full diagnostics run as today.

## Verbose gate

- **Option A (recommended):** Env var `GRIM_VERBOSE_LOGGING=1` (default 0). No config change required.
- **Option B:** New hyperparameter `verbose_logging` (or `expensive_diagnostics`) in training config, default false.

Helper: `isVerboseLogging(ctx)` — true only when the chosen verbose gate is on.

## What to gate behind verbose

When **verbose is false** (default), skip the following. When **verbose is true**, keep current behavior (still respect equation logger and intervals).

### 1. Argmax / collapse diagnostic block (Phase2 ~3281–3420+)

- **Current gate:** `kCollapseDiagEnabled` = equation logger AND (batch 0 or every 10).
- **New gate:** `kCollapseDiagEnabled && isVerboseLogging(ctx)`.
- **Contains:** flat_targets D2H, **128 serial logit row cudaMemcpy**, [COLLAPSE_DETECT] log, Token277, HiddenState277, FeedbackLoop, PC1CausalityTest, EQ_LOG for collapse/hidden/feedback.
- **Effect:** No argmax detection, no 128 copies, no collapse/Token277/HiddenState/Feedback/PC1 unless verbose is on.

### 2. Costly blocks inside `shouldSyncDiagnostics` (Phase2)

Only run these when **both** `shouldSyncDiagnostics(ctx, batch_idx)` **and** `isVerboseLogging(ctx)`:

- **BATCH_PRED_DIST** (~2092–2161): 100 × vocab_size blocking cudaMemcpy, argmax counts, LogitTrace, CollapseTokenDiag, PtPvDump (including per-position logit row fetches for positions &gt; sample).
- **LogitSignal block** (~2456–2530): 50 × vocab_size logit copy, argmax stats, [LogitSignal] log.
- **Logit scale / hidden / weight row diagnostics** (~2527–3080): logit scale equation, hidden state copies, weight row copies, cosine similarity, etc.
- **Embedding gradient equation** (Phase2 ~3830–3855): `kEmbGradDiagEnabled` → add `&& isVerboseLogging(ctx)` so `computeEmbGradEquation` (full vocab gradient D2H + token IDs D2H) runs only when verbose.

Optional (can be left as-is or also gated for consistency):

- GradStats flushAndLog, sampleWeightStats, sampleOptimizerMomentStats, computePerComponentUpdateTrace — these already run only when `sync_diag` (i.e. shouldSyncDiagnostics); if desired, also require verbose so that *no* D2H diagnostic runs without verbose.

## Implementation sketch

1. **Phase2_TrainingLoop.cu**
   - Add `isVerboseLogging(ctx)` (read `GRIM_VERBOSE_LOGGING` or `ctx.config.hyperparameters.verbose_logging` if added).
   - `kCollapseDiagEnabled` → `kCollapseDiagEnabled && isVerboseLogging(ctx)`.
   - `kEmbGradDiagEnabled` → `kEmbGradDiagEnabled && isVerboseLogging(ctx)`.
   - Inside the `if (shouldSyncDiagnostics(ctx, batch_idx))` block, wrap the expensive subsections (BATCH_PRED_DIST, LogitSignal, logit scale / hidden / weight row diag) in `if (isVerboseLogging(ctx)) { ... }`.
2. **Config (if Option B):** In `control/ai_config_paths.hpp` (TrainingHyperparameters) add `bool verbose_logging = false;`, load from config, and pass through to `TrainingContext` so `isVerboseLogging(ctx)` can use it.

## Result

- **Default (verbose off):** No argmax/collapse block, no 128 logit copies, no BATCH_PRED_DIST / LogitSignal / hidden/weight/emb-grad heavy diagnostics. Sync count and D2H volume drop sharply.
- **Verbose on:** Same as current when equation logging + interval conditions are met.

This addendum should be applied together with the main Training Loop Sync Optimization plan (Token277 removal, redundant sync removal, etc.).
