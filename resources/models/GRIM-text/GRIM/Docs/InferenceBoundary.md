# Inference / Training Forward Boundary TODO

Scope: split inference execution from training/autograd orchestration while still sharing model layers and CUDA kernels.

Detailed ownership-tightening work for making the shared forward primitive read-only over durable parameter state lives in [ForwardReadOnlyPlan.md](ForwardReadOnlyPlan.md).

## Target ownership boundary

- Training owns `AutogradContext`, loss assembly, backward, optimizer state, and `AutogradStepScope`.
- Phase2 inference owns generation session state and the explicit shared-forward request/runtime payload it authors over caller-authored `BatchPayload` objects, including a `GenerationState::forward_outputs` sink that is separate from training-owned `TrainingState::forward_outputs`.
- Shared code owns only mode-neutral forward primitives that consume explicit device views and a caller-authored graph policy. It must not branch on training vs inference identity.
- No inference-only fields may live in `AutogradContext`.
- No forward path may rediscover the active step by reading `TrainingState.cached_*` as an implicit current batch; callers must pass explicit bindings.

## Phase 1 — Explicit inference prefill seam

Status: implemented.

- [x] Add a shared inference-prefill forward primitive outside `training/Autograd`.
- [x] Route the Phase2-owned inference loop through that primitive.
- [x] Build a per-call `BatchDeviceBindings` view from the caller-authored inference payload.
- [x] Build inference prompt ingestion through `Batching::buildInferenceBatchPayload()` and Phase2-owned explicit shared-forward calls over those payloads instead of server/vector-authored CUDA copies or model-owned generation-session wrappers.
- [x] Keep existing `TrainingState` cache tensors as temporary backing storage only.
- [x] Keep training on the shared forward primitive while Phase2 still owned only upload + loss/backward orchestration.

Exit criteria:

- Inference prefill no longer calls `initAutogradContext()`.
- Inference prefill no longer calls any training-only forward adapter; optional MTP logits come from the same shared-forward request boundary as primary logits.
- `AutogradContext` has no inference-only fields or inference initializer overload.

## Phase 2 — Shared full-forward primitive

Status: implemented.

- [x] Extract the training/eval full-forward math from `AutogradTraining.cu` into `Shared/Forward/ModelForward_GPU.cu`.
- [x] Route training and inference through the same shared forward primitive; Phase2 training now calls `Forward::executeModelForward(...)` explicitly and autograd owns only loss/backward.
- [x] Route inference prefill through the shared primitive with `ModelForwardGraphPolicy{false,false,false,false}`: read-only parameter graph, no backward retention, no dropout, and optional forward extras requested explicitly.
- [x] Delete per-layer K/V preservation for inference; Phase2 inference uses the shared full-context graph rather than a separate KV-cache decode graph.
- [x] Delete the temporary `InferenceForward_GPU.{hpp,cu}` primitive.
- [x] Keep loss/backward code in `training/Autograd`.

Exit criteria:

- `AutogradTraining.cu` contains orchestration only: context validation, forward adapter, loss, backward, training-step bridge.
- Shared training/eval forward code takes a graph-policy request, not `AutogradContext` and not a training/inference mode enum.
- Inference prefill calls `Shared/Forward/ModelForward_GPU.cu`, not a separate inference-prefill primitive.

## Phase 3 — Move generation runtime buffers

Status: in progress.

Move inference/session-owned state from `TrainingState` into `GenerationState` or typed inference owners:

- [ ] token id cache
- [ ] numeric side-channel cache
- [ ] atom mask / flags
- [ ] token-to-slot map
- [ ] inference encoder/logit snapshots if only generation consumes them
- [x] persistent inference execution memory
- [x] decode-time selector result
- [x] decode trace state if it is session state
- [x] single-token scratch tensors

Exit criteria:

- Inference can run without treating `TrainingState` as its session object.
- Training-time sampling explicitly allocates/borrows generation state instead of relying on training cache identity.

## Phase 3b — Shared bootstrap / inference loop seam

Status: implemented for the Phase2 inference entrypoint and trainer-owned inference worker routing.

- [x] `Forward::ModelForwardRequest` no longer exposes `ModelForwardMode::TrainingGraph` / `InferencePrefill`; orchestration authors graph policy before entry.
- [x] Read-only shared prefill detaches embedding, encoder, ScratchBlock, LM-head, execution-block, and selector parameter views at the boundary.
- [x] Add `Phase2_InferenceLoop.*` next to `Phase2_TrainingLoop.*` so `train_gpu --inference` can drive inference orchestration without embedding inference policy inside shared forward or the HTTP bridge.
- [x] Keep `Phase1_Startup` as the shared train/inference bootstrap path.
- [x] Move text prompt tokenization, inference `BatchPayload` construction, generation config slicing, and decode into the trainer process (`executePhase2TextInference(...)` plus the train_gpu worker), not `grim_text_server`.

Exit criteria:

- `grim_text_server` launches and proxies to `train_gpu --inference`; it does not include Phase1/Phase2 headers, load config, store/borrow `TrainingContext`, touch tokenizer artifacts, build request `BatchPayload`, derive `GenerationHP`, decode tokens, or hand-initialize CUDA/model topology/inference state/checkpoints.
- `train_gpu --inference` owns the Phase1-authored inference `TrainingContext`, exposes the internal worker endpoint, and routes request text/options into Phase2 inference over that state.
- Forward files can be described as read-only graph primitives: they read explicit model/input/runtime payloads and write only explicit per-call outputs/sinks.

## Phase 4 — Build graph separation

Status: not started.

- Remove training autograd/loss files from the inference server target.
- Server target links shared forward + layers + TensorContract only.
- Training target links shared forward + autograd loss/backward/training orchestration.

Exit criteria:

- `grim_text_server` does not compile or link `training/Autograd/AutogradTraining.cu`.
- `grim_text_server` does not compile or link Phase1/Phase2/training/model CUDA objects; it links only HTTP/JSON/process-bridge dependencies.
- Any accidental server dependency on training loss/backward fails at build time.
