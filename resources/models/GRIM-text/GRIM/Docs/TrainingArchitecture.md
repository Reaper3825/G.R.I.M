# Three-Phase Training Architecture

Entry point: `train_gpu.cu` → `executePhase1()` → `executePhase2()` → `executePhase3()`. Data flows via the `TrainingContext` struct (no globals).

| Phase | File | Responsibility |
|-------|------|----------------|
| 1 — Startup | `training/Phases/Phase1_Startup.cu` | Config, tokenizer, data loading, model init, optimizer setup |
| 2 — Training Loop | `training/Phases/Phase2_TrainingLoop.cu` | Batching, forward/backward, gradient clipping, validation, checkpointing |
| 3 — Cleanup | `training/Phases/Phase3_Cleanup.cu` | Final checkpoint, summary, resource release |

**Edit the phase file**, never `train_gpu.cu`, when modifying training logic.
