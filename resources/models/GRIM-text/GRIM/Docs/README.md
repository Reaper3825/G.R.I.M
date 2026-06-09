# GRIM-text Feature Docs

Per-feature documentation. Load only the feature(s) relevant to the current task. Universal coding rules (Rule 20, Rule 21) live in [`.github/copilot-instructions.md`](../../../../../.github/copilot-instructions.md).

## Index

| Doc | Scope |
|-----|-------|
| [Build.md](Build.md) | Build commands, CMake cache traps |
| [TrainingArchitecture.md](TrainingArchitecture.md) | Three-phase training entry/orchestration |
| [Optimizer.md](Optimizer.md) | Optimizer Window boundary and dispatch ownership |
| [TrainingState.md](TrainingState.md) | Centralized GPU resource ownership |
| [GraphStateOwnership.md](GraphStateOwnership.md) | Single-graph phase ownership: upload, forward, loss, backward, optimizer |
| [ForwardChronology.md](ForwardChronology.md) | Chronological training/inference paths from batch entry to first forward broadcast |
| [InferenceBoundary.md](InferenceBoundary.md) | Inference/training forward split TODO |
| [ForwardReadOnlyPlan.md](ForwardReadOnlyPlan.md) | Plan to make shared forward read-only over durable parameter state |
| [Autograd.md](Autograd.md) | TensorContract, GradFn, intermediates lifetime |
| [Loss.md](Loss.md) | Unified loss, registered global gradient clipping |
| [GQA.md](GQA.md) | Grouped Query Attention shapes & backward scaling |
| [FlashAttention.md](FlashAttention.md) | FA2 kernel ordering, GQA backward buffers |
| [LMHead.md](LMHead.md) | Tied embeddings, γ_final, hidden-state centering |
| [Encoder.md](Encoder.md) | Encoder layer, bias autograd, FFN cache, LayerScale |
| [ScratchBlock.md](ScratchBlock.md) | ScratchBlock forward/backward and buffer sync |
| [ExecutionBlock.md](ExecutionBlock.md) | Row-final execution memory and causal readback contract |
| [PositionEncoding.md](PositionEncoding.md) | ALiBi & RoPE NTK |
| [Tokenizer.md](Tokenizer.md) | Unigram+byte fallback, AtomTable, sliding window |
| [AtomTableEntryLogAudit.md](AtomTableEntryLogAudit.md) | Audit of AtomTable entry failures from tokenizer log, including fail-loud requirements for sign and `arg_number` handling |
| [Initialization.md](Initialization.md) | Xavier/Philox, residual projection init gain, embedding scale |
| [Diagnostics.md](Diagnostics.md) | RMSNorm formula, CUDA events, LibTorch baselines |
| [Config.md](Config.md) | ai_config.json conventions, fail-loud defaults |
| [CppCudaFootguns.md](CppCudaFootguns.md) | General language/runtime traps |
| [DeletedCode.md](DeletedCode.md) | Removed subsystems and marked-for-removal class targets — do not recreate or deepen dependencies |
