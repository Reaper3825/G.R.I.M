# ScratchBlock Reasoning Layer

Structured reasoning layer for atom-aware hidden-state injection. See [Tokenizer.md](Tokenizer.md) for atom token layout and `BatchPayload` atom metadata.

## Ownership

`ScratchBlockLayer` is the durable ScratchBlock runtime shell (Pattern B), owned by `TrainingContext::gpu_model.scratch_block_layer` (`Startup::GpuModelState`) and assembled in `training/Phases/Startup/Model/ModelGpuAssembly.cu` inside `GRIMText::Training::Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed)`. Its reusable atom-position/embedding buffers live there; the trainable tensors do **not**.

Startup runtime allocation must not construct, configure, reset, or allocate `ScratchBlockLayer`. TrainingState owns only reusable runtime cache tensors. If `HyperParameters::scratchBlockConstructionHP(config).enabled=true`, startup model assembly creates the runtime shell and `ParameterGroupRegistration::initializeScratchBlockParameterTensors(...)` allocates the durable trainable tensors on `TrainingContext::parameter_registry.scratch_block_parameters`. ScratchBlock forward/autograd ops consume that registry-owned tensor bundle explicitly in their signatures; the runtime shell does not cache or bind a registry pointer. `ParameterGroupRegistration` fails loud if either the configured runtime shell or the registry owner is missing. Runtime code must not toggle ScratchBlock or ask `LanguageModel` whether it is enabled; it reads the authored grouping and treats layer presence as a startup invariant.

Static construction values come from `HyperParameters::scratchBlockConstructionHP()` in `Shared/HyperParameters/HyperparameterGroupings.hpp`. Runtime startup resources are passed separately: the grouping does **not** own `cudaStream_t`; model assembly supplies an explicit init stream to the layer constructor.

`ScratchBlockReasoning_GPU.{hpp,cu}` no longer owns a nested `ScratchBlockConfig` sidecar. The runtime shell keeps only reusable buffers / logging state, while ScratchBlock ops consume the grouped `ScratchBlockConstructionHP`, explicit `BatchPayload` / `BatchDeviceBindings`, and explicit registry-owned `ScratchBlockParameterTensors` in their signatures. Do not add config wrappers or layer-side parameter accessors back onto ScratchBlock.

## Forward/backward

The current entry points are `autograd::scratch_block_inject()` and `autograd::scratch_block_project_all_tokens()`. They return tensors with ScratchBlock autograd nodes attached when gradients are tracked, and they consume the active grouped ScratchBlock HP plus explicit batch payload/bindings:

- Forward computes `output = input + atom_scale * project(atom_embedding)` for detected atom positions.
- Atom embeddings merge learned atom type vectors with numeric value, atom flags, and slot-binding metadata.
- Execution-first placeholder mode (`scratch_block_execution_first_type_only=true`) is now strict neutral mode: ScratchBlock emits an exact-zero atom embedding/structured state for atom placeholders, and shared embedding lookup now neutralizes atom placeholder token IDs structurally so `<INT>` / `<FLOAT>` never add a separate token-identity signal to encoder input. Numeric truth stays exclusively on execution-memory / slot-map side channels.
- `ScratchBlockGradFn` owns saved atom activations and backward scratch inside the autograd boundary.
- Backward propagates identity gradient to the input chain and accumulates gradients into `atom_projection` and `atom_type_embeddings` only when the forward path depends on them. Strict neutral execution-first placeholders intentionally do not backprop into `atom_type_embeddings`.

Do not reintroduce the deleted dropout-gradient tap path or external ScratchBlock pools. Batch upload copies `BatchPayload` host atom/numeric metadata into TrainingState device cache tensors; ScratchBlock reads those per-step device bindings during forward.
