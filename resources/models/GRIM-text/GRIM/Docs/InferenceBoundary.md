# Inference / Training Forward Boundary TODO

Scope: split inference execution from training/autograd orchestration while still sharing model layers and CUDA kernels.

Detailed ownership-tightening work for making the shared forward primitive read-only over durable parameter state lives in [ForwardReadOnlyPlan.md](ForwardReadOnlyPlan.md).

## Target ownership boundary

- Training owns `AutogradContext`, loss assembly, backward, optimizer state, and the Phase2 `processBatch()` step-state clear guard.
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
- Inference prefill no longer calls any training-only forward adapter.
- `AutogradContext` has no inference-only fields or inference initializer overload.

## Phase 2 — Shared full-forward primitive

Status: implemented.

- [x] Extract the training/eval full-forward math from `AutogradTraining.cu` into `Shared/Forward/ModelForward_GPU.cu`.
- [x] Route training and inference through the same shared forward primitive; Phase2 training now calls `Forward::executeModelForward(...)` explicitly and autograd owns only loss/backward.
- [x] Route inference prefill through the shared primitive with `ModelForwardGraphPolicy{false,false,false,false}`: read-only parameter graph, no backward retention, no dropout, and optional forward extras requested explicitly.
- [x] Use the shared full-forward primitive with a session-scoped KV cache for incremental decode.
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

## Phase 5 — Session KV cache

Status: implemented (pending GPU-box verification).

Replaces the O(n²) full-context recompute decode with a proper incremental
decoder, without changing the public entry/exit
(`executePhase2TextInference(...) -> Phase2TextInferenceResult`) or adding a
separate decode graph file.

- `GenerationState` owns a session-scoped `KvCacheState`
  ([Shared/InferenceState/KvCacheState_GPU.hpp](../../Shared/InferenceState/KvCacheState_GPU.hpp)):
  per-layer bf16 K/V capacity buffers `[1, max_cached_seq_len, n_kv_heads, head_dim]`,
  a device `cache_seqlens` + host mirror, and fused-rotary cos/sin tables built
  once from `pbm.rope_inv_freq` via `PBM::launchBuildRotaryCosSinTables`.
- `ModelForwardRequest` gains an optional `kv_cache` pointer. When set, the
  shared forward's read-only (no_grad) branch routes each layer's attention
  through `encoderSelfAttentionForwardCached` (fused-rotary
  `flash_attn_fwd_kvcache_rotary` + the SAME `attention_off_by_one` epilogue as
  training), and advances `cache_seqlens` by `q_len` once after the layer loop.
  Training/eval callers leave `kv_cache` null and are unaffected.
- Phase2 `generateOneSequence` prefills the prompt (`q_len = prompt_len`), then
  decodes one token at a time (`q_len = 1`).
Known limitations (tracked under "other missing pieces"): no HTTP token
streaming; no stop-sequences; single-sequence only (batch_size == 1); the cache
is rebuilt per request (no cross-request prefix reuse).

ExecutionBlock Phase 1 is supported on the cached path: inference prefill may
run the learned EXECUTE/NOOP gate and structured steps, apply causal readback at
the final prompt token in downstream layers, and move the resulting row-local
register file into `GenerationState::exec_memory` before temporary forward
outputs are cleared. ExecutionBlock Phase 2 lets later cached decode windows
borrow that session memory for downstream cross-attention readback. Decode does
not re-run the gate, bootstrap, or execution steps, and it does not modify
semantic register values/validity.

ExecutionBlock Phase 3 exposes one strict terminal result to generation. A
result is available only when the learned stop controller ends execution after
a completed step; that step's explicit write slot must be valid and finite.
Reaching the configured step limit does not select a result, and there is no
first/last-valid or recency fallback. While a terminal result is pending, only
its matching `<INT>` or `<FLOAT>` placeholder is sampleable. The placeholder is
bound to a model-generated entry in a session-owned AtomTable, so decode renders
the concrete value and the next cached token can consume full NumberEncoder
metadata. Once no execution result is pending, numeric placeholders are enabled
only when the numeric-meaning selector has a same-type candidate.

### Verification (run on an SM80+ GPU box)

Build the `train_gpu` target (e.g. `training/build/Release/train_gpu.exe`), then:

1. **Prefill parity** — compare last-token logits of the new prefill-with-cache
   against the previous full-forward over the same prompt; expect a match within
   bf16 tolerance (~1e-2 rtol). This gates RoPE + ALiBi + off-by-one parity
   between the fused-rotary kernel and the training SDPA path.
2. **Decode equivalence** — greedy KV-cached decode must produce the identical
   token sequence as the previous full-recompute decode for several prompts.
3. **Server smoke** — launch `grim_text_server`, `POST /api/generate`, confirm
   the response + stats and that decode latency scales ~linearly (not
   quadratically) with generated length.
4. **Watch** — cache capacity at `max_seq_len`, `cache_seqlens` rollback,
   GQA mapping (`n_kv_heads`), host/device `cache_seqlens` sync.
