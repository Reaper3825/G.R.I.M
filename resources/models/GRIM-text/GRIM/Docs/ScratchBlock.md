# ScratchBlock Reasoning Layer

ScratchBlock is currently a dormant experimental boundary. Keep the code compiled, but shared forward does **not** invoke it.

## Ownership

Semantic atom data is authored before Phase2 graph execution on the prepared payload boundary:

- `BatchPayload` owns token IDs, numeric values, atom masks/flags, slot maps, and host-side atom entry facts.
- `BatchPayload.atom_positions` / `BatchPayload.atom_types` are the compact authored atom containers ScratchBlock consumes; they are materialized once during payload build, not reconstructed in forward.
- `BatchDeviceBindings` is the only device-side borrow of that authored data for a step.
- `BatchDeviceBindings.d_atom_positions` / `BatchDeviceBindings.d_atom_types` are the device-side compact authored atom views for ScratchBlock.
- `TrainingContext::parameter_registry.scratch_block_parameters` owns the durable trainable ScratchBlock tensors.

Do **not** describe those facts as ScratchBlock “workspace”. They are prepared data.

`ScratchBlockLayer` is only the startup-owned runtime shell on `TrainingContext::gpu_model.scratch_block_layer`, assembled in `training/Phases/Startup/Model/ModelGpuAssembly.cu`. It no longer owns atom position/count containers. The only surviving layer-local buffer is the transient `d_atom_embeddings_` scratch used to stage per-atom forward vectors after the authored payload data has already named which atoms exist.

The old row-local atom view in ExecutionBlock is already deleted. Do not recreate a second per-row atom container there.

## Entry points

- `autograd::scratch_block_inject()` is retained as the additive injection path.
- `autograd::scratch_block_project_all_tokens()` is a removal-target helper. Do not add new callers.
- Shared forward currently does **not** invoke ScratchBlock while experiments are running.

## Autograd

When `scratch_block_inject()` is used:

- `ScratchBlockGradFn` owns saved atom activations and backward scratch inside the tape.
- Backward propagates identity gradient to the input chain.
- Parameter gradients accumulate into `atom_projection` and `atom_type_embeddings` only when the forward path depends on them.

## Guardrails

- Forward/autograd must consume prepared `BatchPayload` / `BatchDeviceBindings` data only; they must not construct a second semantic atom container during forward.
- Do not move ScratchBlock semantic data ownership onto `ScratchBlockLayer`, `TrainingState`, or ExecutionBlock runtime structs.
- Batch upload copies authored payload data into the payload-attached `BatchDeviceStorage` owner, not `TrainingState` cache tensors.
