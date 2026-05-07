# Inference / Training Forward Boundary TODO

Scope: split inference execution from training/autograd orchestration while still sharing model layers and CUDA kernels.

## Target ownership boundary

- Training owns `AutogradContext`, loss assembly, backward, optimizer state, and `AutogradStepScope`.
- Inference owns generation session state through `GenerationState` and inference-specific forward requests.
- Shared code owns only mode-neutral forward primitives that consume explicit device views.
- No inference-only fields may live in `AutogradContext`.
- No forward path may rediscover the active step by reading `TrainingState.cached_*` as an implicit current batch; callers must pass explicit bindings.

## Phase 1 — Explicit inference prefill seam

Status: implemented.

- [x] Add a shared inference-prefill forward primitive outside `training/Autograd`.
- [x] Route `LanguageModel::executeInferenceForward_()` through that primitive.
- [x] Build a per-call `BatchDeviceBindings` view in inference code from the currently staged generation buffers.
- [x] Keep existing `TrainingState` cache tensors as temporary backing storage only.
- [x] Keep training on `Autograd::executeAutogradForward()` for now.

Exit criteria:

- Inference prefill no longer calls `initAutogradContext()`.
- Inference prefill no longer calls `executeAutogradForward()`.
- `AutogradContext` has no inference-only fields or inference initializer overload.

## Phase 2 — Shared full-forward primitive

Status: not started.

- Extract the common forward math from `AutogradTraining.cu` into `Shared/Forward/ModelForward_GPU.cu`.
- Make training call it through a training adapter that supplies autograd workspace and graph mode.
- Make eval/inference call it through non-training request structs.
- Keep loss/backward code in `training/Autograd`.

Exit criteria:

- `AutogradTraining.cu` contains orchestration only: context validation, loss, backward, training-step bridge.
- Shared forward code takes a mode-explicit request, not `AutogradContext`.

## Phase 3 — Move generation runtime buffers

Status: not started.

Move inference/session-owned state from `TrainingState` into `GenerationState` or typed inference owners:

- token id cache
- numeric side-channel cache
- atom mask / flags / text features
- token-to-slot map
- inference encoder/logit snapshots if only generation consumes them
- persistent inference execution memory
- decode-time selector result
- decode trace state if it is session state

Exit criteria:

- Inference can run without treating `TrainingState` as its session object.
- Training-time sampling explicitly allocates/borrows generation state instead of relying on training cache identity.

## Phase 4 — Build graph separation

Status: not started.

- Remove training autograd/loss files from the inference server target.
- Server target links shared forward + layers + TensorContract only.
- Training target links shared forward + autograd loss/backward/training orchestration.

Exit criteria:

- `grim_text_server` does not compile or link `training/Autograd/AutogradTraining.cu`.
- Any accidental server dependency on training loss/backward fails at build time.
