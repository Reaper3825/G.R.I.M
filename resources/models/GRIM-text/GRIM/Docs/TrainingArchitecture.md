# Three-Phase Training Architecture

Entry point: `train_gpu.cu` → `executePhase1()` → `executePhase2()` → `executePhase3()`. Data flows via the `TrainingContext` struct (no globals).

| Phase | File | Responsibility |
|-------|------|----------------|
| 1 — Startup | `training/Phases/Phase1_Startup.cu` | Config, tokenizer, data loading, model init, optimizer setup |
| 2 — Training Loop | `training/Phases/Phase2_TrainingLoop.cu` | Batching, forward/backward, gradient clipping, validation, checkpointing |
| 3 — Cleanup | `training/Phases/Phase3_Cleanup.cu` | Final checkpoint, summary, resource release |

**Edit the phase file**, never `train_gpu.cu`, when modifying training logic.

Startup model tensor registration lives in `training/Phases/Startup/Model/ParameterGroupRegistration.{hpp,cu}`. `LanguageModel` still owns the durable `parameter_groups_` vector and parameter tensors; the startup module only discovers trainable tensors, records non-owning `ParameterGroup` metadata, and binds optimizer moment tensors owned by `OptimizerState`.

`LanguageModel::getModelStats()` lives in `Common/ModelStats.cu`. It reads the model-owned `parameter_groups_` inventory after startup registration and must classify counts from explicit `ParamStatsBucket` metadata written by `ParameterGroupRegistration`. It must not estimate parameter counts from config formulas, compare raw tensor pointers, or re-slice hyperparameters.
