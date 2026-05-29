# Read-Only Shared Forward Plan

Scope: make the shared model forward boundary read-only over durable parameter state while preserving the training autograd graph and keeping inference/prefill on the shared forward primitive.

The phase-ownership doctrine behind this plan is documented in [GraphStateOwnership.md](GraphStateOwnership.md): forward reads state, execution/inference use state, backward measures gradient change, and optimizer/update is the durable state writer.

> Note: `Layers/ReasoningHead/reasoning_head_GPU.{hpp,cu}` has since been deleted. Any remaining references below are historical plan context only; live shared forward now routes structured reasoning through ScratchBlock + ExecutionBlock instead of a parallel reasoning-head subsystem.

## Implementation status

The following ownership fixes are already implemented and should be treated as current architecture, not future work:

- `Shared/Forward/ModelForwardRuntimePayload.hpp` now owns the mutable forward runtime payload boundary (`AutogradIntermediates`, trace vectors, read-gate workspace). `Shared/Forward/ModelForward_GPU.hpp/.cu` consumes that payload instead of a training-state god pointer or inline sink bag.
- `Shared/Forward/ModelForward_GPU.hpp` now takes `ModelForwardGraphPolicy` instead of a training/inference mode enum. Training/inference identity is orchestration-only; shared forward sees only graph connectivity/dropout/retention policy.
- Shared prefill now detaches read-only parameter views at the boundary for embedding lookup, encoder/attention/FFN, ScratchBlock structured projection, LM head, and reasoning head.
- `LMHeadLayer::centered_weights_` is deleted; the effective LM-head weight tensor is now per-call Category 1 state.

The remaining items below are the outstanding follow-up queue.

This doc is the implementation plan for the next step after `InferenceBoundary.md` Phase 2. The goal is **not** to duplicate kernels or fork the model into separate training and inference stacks. The goal is to make `Shared/Forward/ModelForward_GPU.cu` and any mode-neutral layer forward it calls behave like a true forward primitive:

- training adapters may read live trainable tensors so autograd can attach graph edges,
- read-only prefill/inference must consume detached or const parameter views,
- no forward path may mutate durable model or training state (`requires_grad`, shape, ownership, registration facts, persistent runtime owners, EMA/stat counters),
- forward-derived workspaces must live in Category 1 intermediates, not on long-lived layer objects.

## Target contract

### Durable-vs-transient rule

- Category 2 parameter tensors are authored at startup / load / registration time.
- Shared forward may **read** those tensors, but it must not change:
  - `requires_grad`
  - shape metadata
  - ownership flags
  - tied-weight / grad-sharing relationships
- If a mode needs different behavior, the caller selects the correct view before entering forward:
  - training graph → live trainable tensors read as autograd-connected inputs
  - inference prefill / decode → detached read-only views

### Read-only prefill rule

When the caller supplies a read-only `ModelForwardGraphPolicy`:

- the call must not mutate parameter tensors,
- the call must not rely on live parameter `requires_grad=true` state,
- parameter-bearing ops must receive detached or const views,
- any forward-derived effective weight tensor must be per-call state,
- training-only ops such as dropout remain outside the read-only path.

### Training graph rule

Training still owns:

- graph creation,
- loss assembly,
- backward,
- parameter grad accumulation,
- optimizer window / update.

Training forward may pass live parameter tensors to layer math so the resulting outputs keep the correct autograd connectivity, but that does **not** grant permission to mutate any durable state during forward.

During training, forward is still read-only over durable model/training state. Its job is limited to:

- reading parameters and explicit caller inputs,
- producing activations / logits / loss inputs,
- constructing Category 1 graph state needed by backward.

Durable change happens only in the later owners:

- backward accumulates into gradient buffers,
- gradient accumulation windows preserve those gradients across microbatches,
- optimizer/update code mutates parameter values and optimizer state.

If a forward call changes anything durable, that is an ownership bug.

Parameter lifecycle setup must already be complete before the forward call begins.

### Explicit runtime-owner rule

- Shared forward may write only explicit forward outputs and explicit per-call scratch/snapshot payloads passed in the request.
- Shared forward must not evolve durable training/session state as part of ordinary forward execution.
- Per-step scratch, diagnostics, and telemetry sinks must not be hidden inside durable layer members.
- Durable training statistics, grad buffers, optimizer slots, and EMA state belong to backward/update owners, not mode-neutral layer objects.
- Process-global counters / one-time logging statics must not be advanced from layer forward bodies.

## Current boundary violations

### Parameter metadata writes inside forward paths

The following are ownership violations because they mutate durable parameter state during forward execution:

- `Shared/Forward/ModelForward_GPU.cu`
  - `embedding_layer->tokenWeights().requires_grad = is_training`
- `Layers/LMHead/lm_head_GPU.cu`
  - `final_rms_gamma_frozen_or_trained_.requires_grad = true`
  - `weights_.requires_grad = true`
  - `bias_.requires_grad = true`
  - `weights_.shape = ...`
- `Layers/ReasoningHead/reasoning_head_GPU.cu`
  - `w_op_.requires_grad = true`
  - `w_arg1_.requires_grad = true`
  - `w_arg2_.requires_grad = true`

These are not forward-math responsibilities. They belong to construction, load validation, freeze policy, or parameter registration.

### Shared prefill uses live parameter tensors

The shared `InferencePrefill` path currently runs layer math against live trainable tensors in several places:

- embedding lookup on `EmbeddingLayer::tokenWeights()`
- encoder RMSNorm / attention / FFN on live encoder parameters
- LM head RMSNorm / projection on live LM-head parameters
- ScratchBlock projection reads `atomTypeEmbeddings()` / `atomProjection()` directly even when `track_grad=false`
- ReasoningHead forward uses live parameter tensors if it participates in shared forward

This was fixed by centralizing detached read-only parameter views in `Shared/Forward/ModelForward_GPU.cu`; the old `training/Inference_GPU.cu` decode path has been deleted.

### Forward-derived tensors stored on durable layer objects

`LMHeadLayer::centered_weights_` is a forward-derived effective-weight tensor kept as a layer member so backward can still see it. That is Category 1 graph-time state parked on a long-lived object next to Category 2 parameters.

That storage belongs in forward intermediates with graph-window lifetime.

### Shared forward hidden-runtime-owner violation (fixed)

This violation is now fixed by `Shared/Forward/ModelForwardRuntimePayload.hpp`. Kept here as the resolved architectural rationale for the payload split:

- `Shared/Forward/ModelForward_GPU.cu`
  - used to resize / clear / append `TrainingState::execution_runtime.execution_trace_by_row`
  - used to allocate / mutate `TrainingState::execution_runtime.trace_state_by_row`
  - used to pass `TrainingState::read_gate_accum_tensor.data` as a hidden mutable workspace into execution-block readback
- `Shared/Forward/ModelForward_GPU.hpp`
  - used to carry only `TrainingState* runtime_state`, so inference-prefill had no way to express generation/session-owned runtime sinks without going through a training owner

Current contract: `ModelForwardRequest` carries immutable forward inputs plus a caller-authored `ModelForwardGraphPolicy`, and `executeModelForward(...)` receives `ModelForwardRuntimePayload` directly for the mutable runtime sinks. A read-only graph policy reads model state and writes only caller-authored forward outputs / explicit session runtime state. It must not rediscover mutable runtime owners by reaching through `TrainingState`.

### ScratchBlock still keeps forward workspace and telemetry on the durable layer object

`ScratchBlockLayer` still mixes durable parameters with forward-time mutable workspace and telemetry:

- `Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp`
  - durable forward workspace members: `d_atom_positions_`, `d_num_atoms_`, `d_atom_embeddings_`
  - forward telemetry mutation: `recordForwardCall(int num_atoms)` increments durable layer stats from the forward path
- `Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu`
  - `scratch_block_project_all_tokens(..., track_grad=false, ...)` still reads live `atomTypeEmbeddings()` / `atomProjection()` tensors directly instead of detached / const parameter views

The workspace buffers are Category 3 at best, but they are still parked on a long-lived layer object instead of an explicit runtime/intermediate owner. The telemetry mutation is a separate durable-state write from the forward path.

### ExecutionBlock still keeps per-step scratch and training baseline on the durable layer object

`ExecutionBlockLayer` still owns mutable device buffers that are reset and rewritten from the step-execution path:

- `Layers/ExecutionBlock/execution_block_GPU.hpp`
  - `d_numeric_error_flag_`
  - `d_div_clamp_count_`
  - `d_div_invalid_flag_`
  - `d_exec_idx_`
  - `d_exec_record_i_`
  - `d_exec_record_f_`
  - `d_reinforce_baseline_`
- `Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
  - `cudaMemsetAsync(LayerAccess::numericErrorFlag(layer), 0, ...)`
  - arg-selection / writeback code writes `LayerAccess::execIndices(layer)` from forward-time argmax decisions
- `Layers/ExecutionBlock/execution_block_data_stream_GPU.cu`
  - execution diagnostics and record capture reuse `LayerAccess::execRecordI(layer)` / `execRecordF(layer)`
  - `kernelArgReinforceLossForward(...)` updates `baseline[0] = ...` through `LayerAccess::reinforceBaseline(layer)`

This is two separate ownership violations:

- the error / selection / record buffers are Category 3 per-step workspace parked on a durable layer object,
- the REINFORCE EMA baseline is Category 2 training state parked on that same shared layer object.

Both must move behind explicit owners. Shared forward should consume caller-owned step workspaces, and any persistent training baseline must live in training-owned state rather than a mode-neutral layer.

### Encoder / attention still mutate process-global forward state

There are still forward-time writes that do not touch parameters but do mutate hidden global/static state:

- `Layers/Encoding/Encoding_GPU.cu`
  - `std::atomic<uint64_t> g_encoder_forward_counter{0};`
  - `g_encoder_forward_counter.fetch_add(1, std::memory_order_relaxed)`
- `Layers/FlashAttention/EncoderSelfAttention_GPU.cu`
  - `static bool logged_rope_alibi_config = false;`
  - `logged_rope_alibi_config = true;`

These are process-global forward side effects. Even when they are “just for seeding” or “just for one-time logging,” they still make layer forward impure, hide determinism policy from the caller, and violate the goal that mode-neutral forward consumes explicit inputs and writes explicit outputs only.

## Violator ledger — patch order / agent handoff format

Use this section as the implementation queue. The order below is the architectural patch order, not merely the order files appear in the tree.

### Patch 1 — shared forward boundary (`forward-boundary` agent)

- **Primary files:**
  - `Shared/Forward/ModelForward_GPU.hpp`
  - `Shared/Forward/ModelForward_GPU.cu`
- **Violation class:** durable metadata mutation, hidden runtime-state mutation, incomplete prefill detach boundary
- **Concrete offenders:**
  - `embedding_layer->tokenWeights().requires_grad = is_training`
  - `InferencePrefill` detaches only `structuredGateWeight()` while the rest of the parameter-bearing path still consumes live tensors
  - writes `TrainingState::execution_runtime.execution_trace_by_row` / `trace_state_by_row`
  - uses `TrainingState::read_gate_accum_tensor` as implicit mutable workspace
- **Required patch:**
  - centralize mode-based parameter acquisition at the shared forward boundary
  - remove forward-time parameter metadata writes
  - replace hidden `TrainingState` reach-through with explicit caller-owned output/scratch payloads that do not evolve durable state as a side effect of forward
- **Exit signal:**
  - `ModelForwardGraphPolicy{false,false,false,false}` can be described as “reads model parameters through detached views, writes only forward outputs and explicit session/runtime payloads”

### Patch 2 — runtime-owner split (`boundary-owner` agent)

- **Primary files:**
  - `Shared/Forward/ModelForward_GPU.hpp`
  - `Shared/TrainingState/TrainingState_GPU.hpp`
  - `Shared/InferenceState/GenerationState_GPU.hpp`
  - `training/Phases/Phase2_InferenceLoop.cu`
- **Violation class:** training/session ownership conflation at the shared forward boundary
- **Concrete offenders:**
  - shared forward request exposes only `TrainingState* runtime_state`
  - inference/session execution trace ownership already lives in `GenerationState`, but shared prefill still has no typed way to target it
- **Required patch:**
  - make execution trace / trace-state / read-gate sinks explicit boundary payloads
  - keep training diagnostics in `TrainingState`
  - keep session/decode state in `GenerationState`
- **Exit signal:**
  - inference-prefill no longer depends on training-owned containers for runtime trace/session writes

### Patch 3 — ScratchBlock read-only + workspace cleanup (`scratchblock-boundary` agent)

- **Primary files:**
  - `Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp`
  - `Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu`
- **Violation class:** live-parameter prefill, forward workspace parked on durable layer, forward telemetry mutation
- **Concrete offenders:**
  - `track_grad=false` projection path reads live `atomTypeEmbeddings()` / `atomProjection()` tensors
  - layer-owned mutable workspace: `d_atom_positions_`, `d_num_atoms_`, `d_atom_embeddings_`
  - `recordForwardCall()` mutates durable layer stats from forward helpers
- **Required patch:**
  - route read-only projection through detached / const parameter views
  - move per-call atom detection / projection workspace to an explicit runtime or forward-intermediate owner
  - separate forward math from durable telemetry mutation, or make the telemetry sink caller-owned and explicit
- **Exit signal:**
  - ScratchBlock forward math becomes a pure parameter-read + output-write step with no hidden durable layer-state mutation

### Patch 4 — ExecutionBlock workspace / baseline ownership (`execution-runtime` agent)

- **Primary files:**
  - `Layers/ExecutionBlock/execution_block_GPU.hpp`
  - `Layers/ExecutionBlock/execution_block_GPU.cu`
  - `Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
  - `Layers/ExecutionBlock/execution_block_data_stream_GPU.cu`
- **Violation class:** per-step workspace parked on durable layer, training baseline parked on mode-neutral layer
- **Concrete offenders:**
  - layer-owned mutable device buffers: `d_numeric_error_flag_`, `d_div_clamp_count_`, `d_div_invalid_flag_`, `d_exec_idx_`, `d_exec_record_i_`, `d_exec_record_f_`
  - per-step resets / writes like `cudaMemsetAsync(LayerAccess::numericErrorFlag(layer), 0, ...)` and argmax outputs stored in `LayerAccess::execIndices(layer)`
  - persistent training EMA `d_reinforce_baseline_` updated by `baseline[0] = ...`
- **Required patch:**
  - move step scratch / error flags / execution-record buffers to explicit caller-owned runtime workspace
  - move REINFORCE baseline ownership to training-owned durable state
  - keep `ExecutionBlockLayer` as parameter owner + math entry point, not runtime scratch owner
- **Exit signal:**
  - execution-block forward math no longer mutates hidden raw device buffers stored on the durable layer object

### Patch 5 — LM head metadata + effective-weight lifetime (`lmhead-boundary` agent)

- **Primary files:**
  - `Layers/LMHead/lm_head_GPU.hpp`
  - `Layers/LMHead/lm_head_GPU.cu`
- **Violation class:** durable metadata mutation, Category 1 tensor stored on durable layer
- **Concrete offenders:**
  - `final_rms_gamma_frozen_or_trained_.requires_grad = true`
  - `weights_.requires_grad = true`
  - `bias_.requires_grad = true`
  - `weights_.shape = ...`
  - `centered_weights_` stored as a layer member
- **Required patch:**
  - delete forward-time metadata writes
  - validate parameter shapes before forward
  - move `W_eff` / centered-token-type-gated weights into Category 1 forward intermediates
- **Exit signal:**
  - LM head owns only durable parameters; all effective-weight tensors are per-call graph-time state

### Patch 6 — Reasoning head metadata + read-only consumption (`reasoning-boundary` agent)

- **Primary files:**
  - `Layers/ReasoningHead/reasoning_head_GPU.hpp`
  - `Layers/ReasoningHead/reasoning_head_GPU.cu`
- **Violation class:** durable metadata mutation, no explicit read-only parameter-consumption path
- **Concrete offenders:**
  - `w_op_.requires_grad = true`
  - `w_arg1_.requires_grad = true`
  - `w_arg2_.requires_grad = true`
- **Required patch:**
  - delete forward-time parameter flag writes
  - accept detached / const parameter views when the head participates in shared prefill
- **Exit signal:**
  - reasoning-head forward consumes caller-selected parameter views without mutating durable state

### Patch 7 — process-global forward state cleanup (`forward-purity` agent)

- **Primary files:**
  - `Layers/Encoding/Encoding_GPU.cu`
  - `Layers/FlashAttention/EncoderSelfAttention_GPU.cu`
- **Violation class:** hidden process-global forward side effects
- **Concrete offenders:**
  - `g_encoder_forward_counter.fetch_add(...)`
  - `static bool logged_rope_alibi_config`
- **Required patch:**
  - move nonce / determinism inputs to explicit caller-owned request payloads or delete the implicit counter path
  - move one-time logging to startup / validation / explicit logger ownership instead of forward-local statics
- **Exit signal:**
  - encoder / attention forward no longer mutates globals or local statics as part of ordinary execution

### Patch 8 — encoder / attention / FFN read-only entry path (`encoder-readonly` agent)

- **Primary files:**
  - `Layers/Encoding/Encoding_GPU.{hpp,cu}`
  - `Layers/FlashAttention/EncoderSelfAttention_GPU.{hpp,cu}`
  - `Layers/FeedForward/Feed_Forward_GPU.{hpp,cu}`
- **Violation class:** prefill still flows through live encoder parameter tensors; APIs still assume mutable ownership access
- **Concrete offenders:**
  - shared prefill currently feeds live RMS gammas, attention weights/biases, and FFN weights/biases into encoder math
  - encoder/attention/FFN entry points do not yet have an explicit read-only parameter-view contract
- **Required patch:**
  - add read-only parameter-view entry paths or helper payloads
  - keep kernels shared; only the parameter-consumption boundary changes
- **Exit signal:**
  - encoder stack can run in prefill from detached / const parameter views without rediscovering mutable state

### Patch 9 — accessor tightening (`api-boundary` agent)

- **Primary files:**
  - `Layers/Embedding/Embedding_GPU.hpp`
  - `Layers/Encoding/Encoding_GPU.hpp`
  - `Layers/LMHead/lm_head_GPU.hpp`
  - `Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp`
  - `Layers/ReasoningHead/reasoning_head_GPU.hpp`
  - `Layers/ExecutionBlock/execution_block_GPU.hpp`
- **Violation class:** boundary not enforced by the type system
- **Concrete offenders:**
  - unconditional mutable `Tensor&` parameter accessors for read-only consumers
- **Required patch:**
  - add / prefer const accessors for read-only forward callers
  - leave mutable access only at startup, registration, checkpoint I/O, optimizer, and other explicit mutation owners
- **Exit signal:**
  - inference-prefill call sites no longer need mutable parameter access just to read model state

## Refactor plan

### Phase 1 — Seal parameter metadata ownership

Goal: eliminate forward-time mutation of durable parameter metadata.

### Work

- Remove parameter `requires_grad` writes from:
  - `Shared/Forward/ModelForward_GPU.cu`
  - `Layers/LMHead/lm_head_GPU.cu`
  - `Layers/ReasoningHead/reasoning_head_GPU.cu`
- Remove forward-time shape restamping of LM-head parameters.
- Move any required validation to:
  - constructors,
  - startup model assembly,
  - checkpoint load validation,
  - parameter registration.

### Exit criteria

- No shared forward or layer-forward file mutates durable parameter metadata.
- Parameter shapes are validated before forward, not repaired during forward.
- Freeze/tie policy is established before the first batch.

### Phase 2 — Build the read-only parameter-view boundary at the caller

Goal: make the caller-authored read-only graph policy explicitly build detached/read-only parameter views before entering layer math.

### Work

- In `Shared/Forward/ModelForward_GPU.cu`, branch the parameter acquisition policy by mode:
  - `TrainingGraph` keeps live tensors,
  - `InferencePrefill` uses detached views.
- Use the read-only parameter-view pattern in `Shared/Forward/ModelForward_GPU.cu` for:
  - embedding weights,
  - encoder RMS gammas,
  - attention weights/biases,
  - FFN weights/biases,
  - LM-head weights/bias/gamma,
  - ScratchBlock gate weights.
- Keep the detach point centralized at the forward boundary so sublayers do not need to rediscover mode.

### Exit criteria

- Shared prefill no longer passes live parameter tensors into model math.
- The mode switch is visible in one place: the shared forward boundary.
- Layer code no longer needs to toggle parameter flags based on mode.

### Phase 3 — Add read-only layer entry paths without duplicating kernels

Goal: keep the math shared while separating read-only parameter consumption from training-graph parameter ownership.

### Work

- Prefer overloads or helper entry points that accept read-only tensors / detached views rather than adding duplicate kernels.
- Files expected to change:
  - `Layers/Encoding/Encoding_GPU.{hpp,cu}`
  - `Layers/FlashAttention/EncoderSelfAttention_GPU.{hpp,cu}`
  - `Layers/FeedForward/Feed_Forward_GPU.{hpp,cu}`
  - `Layers/LMHead/lm_head_GPU.{hpp,cu}`
  - `Layers/ScratchBlock/ScratchBlockReasoning_GPU.{hpp,cu}`
  - `Layers/ReasoningHead/reasoning_head_GPU.{hpp,cu}`
- Keep ownership simple:
  - read-only path consumes const/detached tensor arguments,
  - training path consumes live trainable tensors only as autograd-connected read inputs,
  - both reuse the same kernels and operator implementations.

Training and inference differ in graph connectivity, not in permission to mutate durable state from forward.

### Notes

Do **not** introduce a second authored config object or a sidecar policy wrapper. If a callee already needs many parameter tensors, use a small per-call view payload that directly slices existing tensors; do not mirror model configuration or own state in that payload.

### Exit criteria

- Inference-prefill-capable layer forwards can run without touching live parameter ownership.
- No new inference-only ownership leaks are introduced into training structs.
- Kernel implementations remain shared.

### Phase 4 — Move LM-head effective weights into Category 1 intermediates

Goal: stop storing forward-derived LM-head tensors on the durable layer object.

### Work

- Remove `LMHeadLayer::centered_weights_` as a durable layer member.
- Store effective LM-head weight tensors (`W_eff`) in a per-call forward workspace.
- Candidate owners:
  - a small LM-head forward-intermediate payload, or
  - `AutogradIntermediates` / shared forward intermediates when the tensor must survive until backward.
- Keep the lifetime explicit: the tensor exists only for the active forward/backward window.

### Exit criteria

- No forward-derived effective weights live on `LMHeadLayer`.
- The tensor lifetime is clearly Category 1 and torn down with the rest of the graph.

### Phase 5 — Tighten mutable APIs to match ownership

Goal: make the type system reflect the boundary instead of relying on call-site discipline alone.

### Work

- Audit parameter accessors that currently expose unconditional `Tensor&` and add/prefer const access for read-only consumers.
- Use mutable accessors only for:
  - startup assembly,
  - parameter registration,
  - checkpoint load/save write paths,
  - optimizer / training-only mutation owners.
- Candidate headers:
  - `Layers/Embedding/Embedding_GPU.hpp`
  - `Layers/Encoding/Encoding_GPU.hpp`
  - `Layers/LMHead/lm_head_GPU.hpp`
  - `Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp`
  - `Layers/ReasoningHead/reasoning_head_GPU.hpp`
  - `Layers/ExecutionBlock/execution_block_GPU.hpp` (training-only today, but should still expose const reads where possible)

### Exit criteria

- Read-only forward callers do not need mutable parameter access.
- Remaining mutable access sites are narrow and intentional.

### Phase 6 — Finish build-target separation

Goal: make the read-only forward boundary enforceable at build time, not just by convention.

### Work

- Continue `InferenceBoundary.md` Phase 4 after the read-only split is in place.
- Server/inference target links:
  - shared forward,
  - layers,
  - TensorContract,
  - generation/session state.
- Training target links:
  - shared forward,
  - autograd orchestration,
  - loss/backward,
  - optimizer window.

### Exit criteria

- Inference/server target does not compile or link training autograd orchestration.
- Any accidental dependency on training-only files fails at build time.

## File-by-file starting checklist

| File | Required change |
|------|------------------|
| `Shared/Forward/ModelForwardRuntimePayload.hpp` | Own the explicit mutable runtime payload that shared forward may write during one call |
| `Shared/Forward/ModelForward_GPU.hpp` | Replace training-only runtime reach-through with explicit runtime-owner payloads / sinks for shared forward |
| `Shared/Forward/ModelForward_GPU.cu` | Stop mutating embedding metadata; build detached parameter views for `InferencePrefill`; keep mode switch centralized |
| `Shared/TrainingState/TrainingState_GPU.hpp` | Keep training-only diagnostics/workspaces training-owned; stop being the only shared-forward runtime owner |
| `Shared/InferenceState/GenerationState_GPU.hpp` | Expose generation/session-owned trace/runtime sinks needed by shared prefill |
| `training/Autograd/AutogradTraining.cu` | Own training-only live-tensor prep if any remains after cleanup |
| `training/Phases/Phase2_InferenceLoop.cu` | Route inference scoring through the centralized shared-forward boundary only |
| `Layers/Embedding/Embedding_GPU.hpp` | Prefer const/read-only parameter access for prefill callers |
| `Layers/Encoding/Encoding_GPU.cu` | Accept/read parameter views without mutating ownership metadata or process-global forward state |
| `Layers/Encoding/Encoding_GPU.hpp` | Add const/read-only parameter accessors and/or view payload entry points |
| `Layers/FlashAttention/EncoderSelfAttention_GPU.cu` | Consume read-only attention-weight views in prefill path and remove forward-local static logging state |
| `Layers/FlashAttention/EncoderSelfAttention_GPU.hpp` | Carry read-only attention parameter-view contracts |
| `Layers/ExecutionBlock/execution_block_GPU.cu` | Stop owning step scratch / baseline on the durable layer object; route explicit runtime/training owners instead |
| `Layers/ExecutionBlock/execution_block_GPU.hpp` | Remove durable raw-pointer scratch ownership from the layer boundary; expose explicit runtime workspace payloads |
| `Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu` | Write per-step scratch only through caller-owned runtime workspace buffers |
| `Layers/ExecutionBlock/execution_block_data_stream_GPU.cu` | Move execution records / flags / REINFORCE baseline updates behind explicit runtime or training owners |
| `Layers/FeedForward/Feed_Forward_GPU.cu` | Reuse FFN math with read-only parameter views |
| `Layers/FeedForward/Feed_Forward_GPU.hpp` | Keep mutable access narrow; expose const/read-only FFN parameter reads |
| `Layers/LMHead/lm_head_GPU.cu` | Remove metadata mutation; move `W_eff` out of durable layer state |
| `Layers/LMHead/lm_head_GPU.hpp` | Remove durable `centered_weights_` ownership and tighten read-only access |
| `Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu` | Keep `track_grad=false` projection read-only at the boundary; avoid live-parameter ownership assumptions in prefill |
| `Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp` | Stop mixing durable params with forward workspace; add const/read-only accessors |
| `Layers/ReasoningHead/reasoning_head_GPU.cu` | Remove forward-time parameter flag writes; support read-only consumption if used in shared prefill |
| `Layers/ReasoningHead/reasoning_head_GPU.hpp` | Expose const/read-only parameter accessors for prefill callers |
| `GRIM/Docs/InferenceBoundary.md` | Track this plan as the next ownership-tightening step |

## Validation plan

### Static validation

- Search forward/layer-forward sources for parameter metadata writes:
  - `requires_grad =`
  - `requires_grad_(` on already-owned parameters inside forward bodies
  - forward-time parameter `shape =`
- Search forward/layer-forward sources for hidden process-global mutation:
  - `fetch_add(`
  - `static bool`
- Confirm shared prefill call sites use detached or const parameter views.
- Confirm no Category 1 effective-weight tensor remains on a layer member.

### Runtime validation

- Training forward/backward still produces gradients for all registered parameter groups.
- Training forward alone does not mutate durable parameter/runtime ownership state; any durable parameter change appears only after backward accumulation and optimizer/update ownership runs.
- Inference prefill produces identical logits/KV snapshots for the same prompt before and after the refactor.
- Inference prefill does not allocate or depend on parameter grad buffers.
- Existing decode path and shared prefill path agree on detached-parameter behavior.

### Ownership validation

- The read-only `ModelForwardGraphPolicy` can be described as “reads model state through detached views, writes only explicit forward outputs / caller-owned per-call snapshots”.
- Training forward can be described as “reads model state, builds Category 1 graph state, writes explicit outputs only”.
- Parameter registration, backward grad accumulation, optimizer stepping, and freeze policy remain startup/backward/update responsibilities only.

## Non-goals

- Do not duplicate CUDA kernels just to make read-only wrappers.
- Do not move optimizer or loss ownership into shared forward.
- Do not add sidecar config owners or parallel hyperparameter wrappers.
- Do not weaken fail-loud behavior to preserve temporary compatibility paths.

## Completion summary

The shared forward is done when the graph-policy primitive still serves both training and inference, but the ownership split is finally honest:

- training graph reads live trainable tensors only to build autograd connectivity,
- read-only prefill uses detached/const parameter views,
- no forward code mutates durable model or training state,
- forward-derived weight workspaces live in Category 1 intermediates only.
