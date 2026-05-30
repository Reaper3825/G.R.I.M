# ScratchBlock Reasoning Layer

Structured reasoning layer for atom-aware hidden-state injection. See [Tokenizer.md](Tokenizer.md) for atom token layout and `BatchPayload` atom metadata.

## Ownership

`ScratchBlockLayer` is durable model topology (Pattern B), owned by `TrainingContext::gpu_model.scratch_block_layer` (`Startup::GpuModelState`) and assembled in `training/Phases/Startup/Model/ModelGpuAssembly.cu` inside `GRIMText::Training::Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed)`.

Startup runtime allocation must not construct, configure, reset, or allocate `ScratchBlockLayer`. TrainingState owns only reusable runtime cache tensors. If `HyperParameters::scratchBlockConstructionHP(config).enabled=true`, startup model assembly creates the layer before parameter registration; `ParameterGroupRegistration` fails loud if the configured layer is missing. Runtime code must not toggle ScratchBlock or ask `LanguageModel` whether it is enabled; it reads the authored grouping and treats layer presence as a startup invariant.

Static construction values come from `HyperParameters::scratchBlockConstructionHP()` in `Shared/HyperParameters/HyperparameterGroupings.hpp`. Runtime startup resources are passed separately: the grouping does **not** own `cudaStream_t`; model assembly supplies an explicit init stream to the layer constructor.

## Forward/backward

The current entry point is `autograd::scratch_block_inject()`. It returns a new `Tensor` with `ScratchBlockGradFn` attached to the tape:

- Forward computes `output = input + atom_scale * project(atom_embedding)` for detected atom positions.
- Atom embeddings merge learned atom type vectors with numeric value, atom flags, and slot-binding metadata.
- `ScratchBlockGradFn` owns saved atom activations and backward scratch inside the autograd boundary.
- Backward propagates identity gradient to the input chain and accumulates gradients into `atom_projection` and `atom_type_embeddings`.

Do not reintroduce the deleted dropout-gradient tap path or external ScratchBlock pools. Batch upload copies `BatchPayload` host atom/numeric metadata into TrainingState device cache tensors; ScratchBlock reads those per-step device bindings during forward.
