# Three-Phase Training Architecture

Entry point: `train_gpu.cu` → `executePhase1()` → `executePhase2()` → `executePhase3()`. Data flows via the `TrainingContext` struct (no globals).

| Phase | File | Responsibility |
|-------|------|----------------|
| 1 — Startup | `training/Phases/Phase1_Startup.cu` | Config, tokenizer, data loading, model init, optimizer setup |
| 2 — Training Loop | `training/Phases/Phase2_TrainingLoop.cu` | Batching, forward/backward, gradient clipping, epoch metrics, checkpointing |
| 3 — Cleanup | `training/Phases/Phase3_Cleanup.cu` | Final checkpoint, summary, resource release |

**Edit the phase file**, never `train_gpu.cu`, when modifying training logic.

Static hyperparameter groupings are Phase 1 handoff facts, not Phase 2 loop state. For loss, Phase 1 initializes `TrainingContext.loss_config` from `lossConfigHP()` after validating hyperparameters; Phase 2 passes that grouping directly to autograd, where `AutogradContext` borrows it by required reference. Do not add `initialized` sentinels, `TrainingLoopState` wrappers, or runtime assignment paths that rebuild/revalidate static hyperparameter groupings during training.

Tokenizer artifact preparation is Phase 1 startup work, not a pre-phase owned by `train_gpu.cu`. The order is `LoggingReady()` / `loadStartupConfig()` → `MemorySnapshotReady()` → `HyperparametersReady()` → tokenizer subprocess (`train_tokenizer`, vocab + GRMT preparation) → `CapacityStemReady()` → `DataInfoReady()`. This keeps the tokenizer run behind the same validated `StartupConfig` / `TrainingHyperparameters` path that model startup uses, while still running before `DataInfoReady()` consumes the generated vocab and training-data files. If `subprocess.tokenizer.only_mode=true`, `executePhase1()` returns `Phase1Outcome::tokenizer_only_complete` and the orchestrator exits cleanly without entering Phases 2-3.

Batch schedule ownership is single-pass: `PlannedBatchesReady()` is the only Phase 1 startup step that calls `buildEpochBatches()` for training. `EpochPlanReady()` must derive `total_batches` from the authored `train_payloads` count, not from a scheduler dry-run/preflight wrapper. This prevents duplicated batch-policy execution and keeps Phase 2 consuming only Phase1-authored payloads/order.

`GRMTDataLoader` owns only raw `.grmt` row deserialization into `TrainingSequence` objects. It must not expose stable `seq_id` sample accessors or long-lived row views, because Phase 1 still has to split train/val data, inject boundaries, apply sliding windows, and filter invalid rows. Training `seq_ids` are authored only after `applySlidingWindows()` when `DataInfo.cu` builds `train_views` / `val_views`; `Batching_GPU.cu` schedules those post-window view indices, and `BatchPayload.cu` consumes them.

`.grmt` header ownership is centralized in `Shared/GRMT/GrmtFormat.hpp`: the magic value, four-field header layout, current-version check, nonzero sequence/vocab guards, and header writes live there. Startup/data tools may read the header only through that boundary; after it returns a validated header, callers continue with their own stream-specific payload work.

Startup model tensor registration lives in `training/Phases/Startup/Model/ParameterGroupRegistration.{hpp,cu}`. `LanguageModel` still owns the durable `parameter_groups_` vector and parameter tensors; the startup module only discovers trainable tensors, records non-owning `ParameterGroup` metadata, and binds optimizer moment tensors owned by `OptimizerState`.

`buildParameterGroups()` is transaction-safe: registration rebuilds a local vector, performs all configured trainable-tensor validation there, and swaps it into `LanguageModel::parameter_groups_` only after success. A thrown registration check must never leave `LanguageModel` with a half-rebuilt parameter inventory.

Startup model allocation lives in `training/Phases/Startup/Model/ModelAllocationState.{hpp,cu}`. It owns the Phase 1 model startup order: CUDA device context, `LanguageModel` construction, stream controller, cuBLAS, PBM, GPU model assembly, parameter registration/verification, `TrainingState` allocation, and checkpoint load/resume.

Durable GPU layer assembly lives in `training/Phases/Startup/Model/ModelGpuAssembly.cu` as `LanguageModel::initGPU(weight_init_seed)`. This Startup/Model source creates the GPU encoder, embedding layer, LM head, ScratchBlock, and optional reasoning/execution/decode-time/MTP heads. It must not allocate `TrainingState` activation caches, optimizer state, parameter groups, checkpoints, forward/backward paths, or Phase 2 loop state.

`LanguageModel::getModelStats()` lives in `Common/ModelStats.cu`. It reads the model-owned `parameter_groups_` inventory after startup registration and must classify counts from explicit `ParamStatsBucket` metadata written by `ParameterGroupRegistration`. It must not estimate parameter counts from config formulas, compare raw tensor pointers, or re-slice hyperparameters.
