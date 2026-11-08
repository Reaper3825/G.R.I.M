# Model and Training Documentation

GRIM-text is separate from the host build. These guides are organized by
current contract, not by the date of a migration or the name of a bug.
Executable code remains authoritative; no runtime validation is implied by
moving a document here.

## Current contracts

| Guide | Scope |
|---|---|
| [Build and runtime](BUILD_RUNTIME.md) | Host/model/HTTP-worker ownership and historical generator context |
| [Autograd](AUTOGRAD.md) | Registry-owned persistent gradients, tape, and batch cleanup |
| [Loss](LOSS.md) | TextLossGradFn, saved log probabilities, fused logits backward, and objective assembly |
| [Parameters](PARAMETERS.md) | Actual tensor names/shapes/gates, tying, precision, LoRA, and checkpoint inventory |
| [RMSNorm and readout](RMSNORM_READOUT.md) | Row-local forward/backward math, gamma ownership, and inference last-row readout |
| [Batching](BATCHING.md) | Fixed context rows, token budget, ordering, and partial batches |
| [Training math](TRAINING_MATH.md) | Per-objective accumulation weights, global clipping, GQA, and optimizer semantics |
| [Telemetry](TELEMETRY.md) | Lattice equations/raw mirrors, update ordering, active measurements, and retained observer boundary |
| [Rank geometry](RANK_GEOMETRY.md) | Exact rank-one algebra, optimizer counterexample, and limits on collapse interpretations |
| [Readiness](READINESS.md) | Source-grounded release checks; not a production certificate |
| [Bridges-2 model watch](BRIDGES2_MODEL_WATCH.md) | Existing cluster tooling and operational caveats; remote state needs separate evidence |

The [model-side feature index](../../resources/models/GRIM-text/GRIM/Docs/README.md)
contains additional specialized kernel/runtime documentation. Neither index
replaces the source when implementation and older feature notes disagree.

## Designs and unfinished integration

Keep the intended direction without presenting it as an implemented runtime:

| Plan | Boundary |
|---|---|
| [Define stage](plans/DEFINE_STAGE.md) | Proposed variable transaction/staged-training semantics |
| [RL training loop](plans/RL_TRAINING_LOOP.md) | Scored-group policy/reference replay and optimizer design |
| [RL grammar](plans/RL_GRAMMAR.md) | Proposed TOOL/READ grammar and mechanism insertion |
| [Self-correction RL](plans/SELF_CORRECTION_RL.md) | Collection, retry, reward/advantage, and update integration |
| [LoRA personalization](plans/LORA_PERSONALIZATION.md) | Implemented model adapters versus pending MMO publication/queue/reload |

Environment interaction is implemented separately from the throwing RL update
scaffold. Model-side LoRA exists; that does not complete all personalization
designs. The shared [reasoning contract](../GRIM_REASONING_DATA_CONTRACT.md)
remains the cross-feature authoring/inference boundary.

## Historical investigations

- [RMSNorm training analysis](investigations/RMSNORM_TRAINING_ANALYSIS.md):
  original log-specific measurements and health assessments.
- [Vocabulary metrics](investigations/VOCAB_METRICS.md):
  the March corpus/artifact snapshot, not current vocabulary facts.
- [Equation trace](investigations/equation_trace.txt):
  preserved historical run algebra and measurements.
- [Plateau investigation](../PLATEAU_BUG_INVESTIGATION.md):
  preserved established root document with an explicit historical notice.
- [Project reports](../reports/): wider-project activity, not model contracts.

Run identity, log filename/timestamp, artifact/config identity, and the
historical/current distinction matter. Use [log attribution](../LOG_FILE_CONVENTION.md).
Do not restore removed centering, PC1, MTP, or neural execution mechanisms from
historical notes or [legacy documents](../legacy/).
