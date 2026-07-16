# GRIM-text LoRA Support Plan

Status: design map for the first implementation slice. No rolling-context boundary, LoRA runtime, or LoRA trainer is implemented yet.

This document turns the MMO requirement that "LoRA is the personalization bridge" into a repository-grounded implementation plan for GRIM-text. It covers the permanent-session source boundary, adapter artifact, inference application, process/config handoff, publication/reload, training ownership, and verification needed before MMO can require a personalized router.

## Outcome

GRIM-text should run the router and synthesizer as:

```text
immutable base checkpoint
        +
validated router-only LoRA artifact
        |
        v
GRIM-text inference worker
```

The adapter remains separate for its entire lifetime. Inference must never merge the LoRA delta into the base tensor buffers or write either artifact. A separate trainer may read the base checkpoint and training inputs, but may publish only a LoRA artifact.

GRIM has one permanent conversation. `SessionContextManager` is Buffer 1: the active short-term context for that conversation. When token pressure advances beyond a safe boundary, it freezes and retires part of the oldest active span. That boundary feeds both the existing memory pipeline and the near-term LoRA personalization queue. Context retirement and LoRA training are separate transactions: Buffer 1 advances without waiting for training, and inference retains the current window and adapter until a candidate is accepted. Whether requests run concurrently with training is a resource-mode decision.

## What exists today

The body-side MMO model contract is partially ready:

- `MMO/Shared/MMD.hpp` has router-only `ModelInfo::lora_path` and `hard_copy_path` fields.
- `MMO/Core/ModelRegistry.cpp` rejects those fields on sub-models.
- `ai_config.json` contains an empty `mmo.router.lora_path`.
- `MMO/Core/CorrectionTuple.*` and `MMO/Core/ToolTrainingParser.*` already export two important classes of future LoRA training records.
- `memory/atomic_writer.hpp` provides the write-temp + rename primitive expected by the MMO plan.

The GRIM-text runtime is not ready yet:

- `resources/models/GRIM-text/GRIM/grim_text_server.cpp` is an HTTP bridge. The actual model owner is the `train_gpu --inference` child process.
- `train_gpu --inference` is explicitly the model owner and loads the canonical model/vocabulary fields from `ai_config.json` through `loadAiConfigSnapshot()`.
- The bridge accepts named `--public-port`, `--worker-port`, and optional `--train-gpu` process arguments. It does not accept or override model/vocabulary paths.
- `MMO/Core/ProcessManager.cpp` passes only those process-scoped ports, launches from the GRIM root so the worker can resolve `ai_config.json`, and no longer performs a competing model/vocabulary handoff.
- `/api/status` proxies the worker status, including `config_source`, `configured_model_path`, `loaded_checkpoint_path`, and `vocab_path`, so the body can verify what `train_gpu` actually loaded.
- Phase 1 loads only the base checkpoint into `ParameterRegistry::StartupParameterRegistry`.
- No LoRA artifact schema, tensor owner, loader, forward application, trainer mode, adapter identity endpoint, or reload protocol exists.
- `ModelRegistry` enforces "sub-models cannot have LoRA" but does not yet enforce "an enabled/enforced router must have a valid LoRA."

The startup prerequisite is now fixed: `ai_config.json` is the single model configuration authority and `train_gpu` is the consumer. LoRA support should follow the same rule by reading the router adapter policy/path from the worker's existing config snapshot rather than adding a competing process-manager override.

## V1 decisions

### Adapter targets

V1 targets the two attention projections in every transformer layer:

| Target | Current base shape | LoRA factors | Forward delta |
|---|---:|---:|---|
| `encoder.{layer}.attention.W_qkv` | `[qkv_dim, d_model]` | `A[rank, d_model]`, `B[qkv_dim, rank]` | `(x A^T) B^T * scale` |
| `encoder.{layer}.attention.W_o` | `[d_model, d_model]` | `A[rank, d_model]`, `B[d_model, rank]` | `(x A^T) B^T * scale` |

`scale = alpha / rank`.

The fused `W_qkv` target matches the model's real GQA layout and avoids inventing separate Q/K/V base weights that do not exist. `W_o` uses the same `[out, in]` convention. Both existing base projections call `autograd::matmul(..., transpose_b=true)`, so one LoRA projection helper can serve both.

V1 deliberately excludes embeddings, RMS gains, biases, the LM head, FFN, NumberEncoder, selector, and ExecutionBlock. This keeps the first artifact small, avoids vocabulary coupling, and makes the base-preservation proof narrow. New targets can be added by a later schema version after attention-only quality is measured.

With the current config (`d_model=768`, `num_heads=12`, `num_kv_heads=4`, `num_layers=12`), `qkv_dim=1280`. Rank 8 attention adapters contain 344,064 parameters, approximately 1.31 MiB in FP32 or 0.66 MiB in BF16, before metadata.

### Initialization and numeric policy

- Trainable `A` uses a seeded small random initialization.
- Trainable `B` starts at zero, making a new adapter exactly behavior-neutral before training.
- Accumulation and the scaled add use FP32 for V1, even if stored factors later use BF16.
- Adapter dropout is a trainer-only setting. It is disabled during inference and does not belong in the inference execution path.
- Multiple active adapters and per-request adapter selection are out of scope. One GRIM-text router process has exactly one active adapter.

### Failure policy

- When `mmo.enabled=true` and `mmo.mode="enforced"`, the router must have a non-empty, readable, compatible `lora_path`; startup fails closed otherwise.
- Shadow mode may explicitly set `personalization.required=false` to compare base-only behavior, but this must be visible in status and telemetry.
- Sub-model LoRA remains a startup error in every mode.
- Missing tensors, duplicate targets, unknown targets, shape/rank mismatch, non-finite values, unsupported schema versions, or base-fingerprint mismatch are fatal adapter-load errors. None may silently degrade to base-only inference when personalization is required.

## Artifact contract

Create a dedicated FlatBuffers schema rather than extending the base-checkpoint schema:

`resources/models/GRIM-text/training/schemas/grim_text_lora.fbs`

Suggested logical payload:

```text
LoraArtifact
  schema_version: uint32
  adapter_id: string
  adapter_revision: uint64
  created_at_utc: string
  base_checkpoint_sha256: string
  base_checkpoint_format_version: uint32
  architecture:
    d_model, num_layers, num_heads, num_kv_heads, qkv_dim
  training_manifest_sha256: string
  default_rank: uint32
  default_alpha: float
  storage_dtype: FP32 | BF16
  targets[]:
    canonical_name
    layer_index
    target_kind: ATTENTION_QKV | ATTENTION_OUTPUT
    in_features
    out_features
    rank
    alpha
    a_data
    b_data
```

The artifact must not contain base weights, optimizer moments, raw memory, prompts, or user data. Trainer-resume state belongs in a separate trainer checkpoint.

Canonical target names are part of the compatibility contract. Loading should resolve them through an allowlisted target-kind switch, not by arbitrary string-to-pointer reflection.

The base SHA-256 binds the adapter to the exact checkpoint bytes. Architecture fields provide useful diagnostics but are not an adequate substitute for the fingerprint: two checkpoints can share shapes and still be incompatible.

## Runtime ownership and code seams

### 1. Configuration and body-side validation

Modify:

- `MMO/Shared/MMD.hpp`
- `MMO/Core/ModelRegistry.cpp`
- `ai_config.json`

Add router personalization policy fields, preferably as a nested object while retaining `lora_path` for the artifact path:

```json
{
  "mmo": {
    "router": {
      "lora_path": "resources/models/GRIM-text/personalization/active.grimlora",
      "personalization": {
        "required": true,
        "expected_adapter_id": "grim-user-primary",
        "reload_policy": "restart"
      }
    }
  }
}
```

Validation responsibilities:

- resolve paths relative to the GRIM root once;
- require LoRA only on the router;
- require it in enforced MMO mode;
- reject directories and non-regular files;
- do not parse or trust artifact internals in the body--the model runtime is authoritative for tensor compatibility;
- log the resolved path without logging artifact contents or training data.

### 2. Process argument contract

Modify:

- `MMO/Core/ProcessManager.cpp`
- `ai/grim_text_server_manager.cpp` (legacy/direct path until retired)
- `resources/models/GRIM-text/GRIM/grim_text_server.cpp`
- `resources/models/GRIM-text/training/train_gpu.cu`
- `resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp`

The bridge accepts only process-scoped named arguments:

```text
grim_text_server
  --public-port <port>
  --worker-port <port>
  --train-gpu <path>
```

The bridge passes only `--inference` and `--inference-worker-port` to its worker. `train_gpu` loads model, vocabulary, and future LoRA configuration from the canonical `ai_config.json` snapshot before Phase 1. Unknown or incomplete bridge arguments fail with a useful error. There is intentionally no model, vocabulary, or LoRA CLI override on this path.

The public bridge should also proxy `/api/mmo/route`, `/api/mmo/synthesize`, and an adapter-aware status endpoint when those endpoints are implemented.

### 3. Separate adapter tensor ownership

Add:

- `resources/models/GRIM-text/Shared/LoRA/LoraTypes.hpp`
- `resources/models/GRIM-text/Shared/LoRA/LoraParameterRegistry.hpp`
- `resources/models/GRIM-text/Shared/LoRA/LoraArtifactIO.cu`
- `resources/models/GRIM-text/Shared/LoRA/LoraProjection.cu`
- `resources/models/GRIM-text/Shared/LoRA/LoraProjection.hpp`

`LoraParameterRegistry` owns adapter GPU tensors and metadata separately from `StartupParameterRegistry`, which continues to own base tensors. A useful shape is:

```text
LoraParameterRegistry
  enabled
  metadata / verified base fingerprint
  encoder_layers[num_layers]
    optional attention_qkv { A, B, rank, scale }
    optional attention_output { A, B, rank, scale }
```

The base registry must not gain mutable "effective weight" buffers. The forward request should borrow both registries explicitly.

### 4. Phase 1 loading

Modify:

- `resources/models/GRIM-text/training/Phases/Phase1_Startup.*`
- `resources/models/GRIM-text/training/Phases/Startup/CheckpointLoad.*`
- `resources/models/GRIM-text/training/Phases/Startup/Validation/Phase2Handoff.*`
- the Phase 1/Phase 2 context type that owns durable inference state

Required order:

1. Resolve and hash the base checkpoint.
2. Assemble the base model and load its checkpoint exactly as today.
3. Read and verify the LoRA FlatBuffer before allocating adapter GPU tensors.
4. Validate schema, base hash, architecture, target allowlist, tensor counts, shapes, ranks, and finite values.
5. Allocate and upload the separate adapter tensors.
6. Carry a read-only adapter registry into Phase 2 inference.
7. Emit one adapter identity event: adapter ID, revision, artifact hash, base hash, target count, rank summary, and required/enabled state.

If any required adapter step fails, destroy the partially allocated adapter registry and abort startup. The base model must not begin serving requests.

### 5. Forward application

Modify:

- `resources/models/GRIM-text/Shared/Forward/ModelForwardRuntimePayload.hpp`
- `resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.cu`
- `resources/models/GRIM-text/Layers/Encoding/Encoding_GPU.hpp`
- `resources/models/GRIM-text/Layers/Encoding/Encoding_GPU.cu`
- `resources/models/GRIM-text/Layers/FlashAttention/EncoderSelfAttention_GPU.hpp`
- `resources/models/GRIM-text/Layers/FlashAttention/EncoderSelfAttention_GPU.cu`

For a base projection `y = x W^T`, compute:

```text
low_rank = x A^T
delta    = low_rank B^T
y        = base_projection + (alpha / rank) * delta
```

Apply the QKV delta before Q/K/V slicing and KV-cache insertion. Apply the output delta before output bias, layer scaling, dropout, and the residual add, matching the semantic location of the corresponding base linear operation.

Both full-sequence and cached inference paths in `EncoderSelfAttention_GPU.cu` must call the same LoRA projection helper. A test must prove the cached and uncached paths do not diverge because one omitted the adapter.

The helper should support two execution policies:

- inference: no autograd graph, read-only A/B tensors;
- LoRA trainer: autograd tracks A/B while base projection tensors remain detached/frozen.

V1 may use two additional cuBLAS matmuls plus a scaled add per adapted projection. Kernel fusion is a measured optimization, not a prerequisite.

### 6. Status and observability

Add adapter identity to the internal and public status surfaces:

```json
{
  "model": "grim-text",
  "base_checkpoint_sha256": "...",
  "personalization": {
    "enabled": true,
    "required": true,
    "adapter_id": "grim-user-primary",
    "adapter_revision": 42,
    "artifact_sha256": "...",
    "targets": 24,
    "state": "ready"
  }
}
```

The body should not mark the router ready merely because the port answers. Readiness requires the expected adapter ID/revision and verified base fingerprint when personalization is required.

Log per-process identity at startup. Per-request logs should include adapter ID/revision, not file contents. Add aggregate adapter overhead timing around the low-rank projections before considering fusion.

## Publication and reload

Training is out-of-band. The publication sequence is:

1. Trainer writes `<lora_path>.tmp` in the same directory.
2. Trainer flushes and closes the file.
3. Trainer reopens it through the production artifact reader and validates it against the base checkpoint.
4. Evaluation gates compare the candidate against the current adapter and required regression/canary sets.
5. Publisher preserves the current artifact as a rollback candidate.
6. Publisher atomically renames the validated temporary artifact onto `lora_path` using the existing atomic-write approach.
7. The body detects the new artifact identity and restarts the router process.
8. Readiness succeeds only after status reports the new adapter identity.

V1 uses process restart, not in-process tensor mutation. This follows the MMO v1 one-process-per-model rule and guarantees that no request observes a half-replaced adapter registry. If the new process fails readiness, restore/restart the previous known-good artifact and surface a structured personalization-load error.

In-flight requests may finish on the old process before restart. New requests must not be admitted between drain and verified readiness.

## Permanent session and rolling-context source boundary

### Lifecycle invariant

`SessionContextManager` represents one GRIM conversation for the lifetime of the process. It is not a user-facing collection of chats.

- Create the canonical state during GRIM bootstrap.
- Keep it across idle time, topic changes, task completion, context compression, adapter updates, and model switches.
- Destroy it only during process shutdown or an isolated test.
- On restart, restore continuity from committed context capsules and the persistent memory system rather than interpreting the restart as a new chat.
- Keep one centrally defined session identity. The current `"default"` key reflects the intended single-session topology, although its duplicated literals should eventually be centralized.

`beginTurn()` and `recordOutcome()` are turn lifecycle operations. They do not create or close conversational sessions. `clearHistory()` must eventually mean "rebuild the active model prompt view," not "forget the conversation."

### Buffer ownership

The data flow is:

```text
one permanent GRIM conversation
  -> Buffer 1: SessionContextManager active short-term window
  -> rolling token-pressure boundary
       -> Buffer 2/3 memory consolidation and exact retrieval references
       -> eligible personalization sequences for LoRA
  -> bounded LoRA-only trainer (1-interval set/learned sequences)
  -> evaluated candidate adapter
  -> safe-boundary adapter activation
```

Buffer 1 owns what the model needs now: system/personality instructions, current dialogue state, recent verbatim messages, pending interactions, live referents, relevant memory retrievals, recent action outcomes, and room for the next response. The long-term memory system remains the exact persistence and retrieval authority. LoRA gradually makes repeated, rewarded user-specific behavior implicit, but it is not the exact transcript or memory database.

### Token-pressure boundary

The boundary is based on tokens produced by the real GRIM tokenizer, not message count. The current `ai_config.json` has `training.config.max_seq_len=1024`, while `conversation_history_size=10` trims by message count. Message-count trimming must be replaced by token accounting before this feature is enabled.

Let:

- `C` be the effective context capacity reported for the loaded base model;
- `R` be a safety reserve for the next user input, output tokens, tool schemas/results, retrieval injection, and counting uncertainty;
- `H = C - R` be the high-water mark;
- `L` be the low-water target after compaction;
- `T` be the protected recent verbatim tail;
- `P` be the currently serialized prompt token count.

When `P >= H`, freeze the oldest eligible contiguous span after the previous compression frontier. Exclude the protected tail and any pinned live state. Retire enough material that the reconstructed active prompt is at or below `L`. Do this before the hard sequence limit is reached; advancing beyond `C` is a failure, not the trigger.

Initial reserve and low-water ratios should be conservative and measured locally. They belong in canonical `ai_config.json`, but must not be hard-coded from the current 1024-token model or confused with the training-only `sliding_window_stride` field.

Pin at least:

- unresolved confirmations, corrections, questions, and follow-ups;
- active goals, constraints, and commitments;
- referents required by the protected tail;
- tool calls whose results are not yet acknowledged;
- current safety and policy state;
- a source span already owned by an in-flight compression transaction.

### Boundary artifact

Each retired span produces one immutable, versioned artifact before Buffer 1 advances its frontier:

```text
ContextBoundaryArtifact
  boundary_id
  canonical_session_id
  source_event_begin / source_event_end
  source_token_begin / source_token_end
  source_hash
  recent_tail_overlap_hash
  dialogue_state
    active_goals
    unresolved_questions
    pending_actions
    commitments
    live_referents
    user_constraints
  episodic_gist
  durable_memory_ids
  correction_tuple_ids
  tool_training_example_ids
  reward_event_ids
  personalization_candidates
  adapter_revision_observed
  retention_and_privacy_class
  validation_status
```

The artifact is both a compact continuation record and the handoff envelope. It preserves exact source/memory references so a gist does not become the only copy of a fact. The LoRA queue consumes only its eligible `personalization_candidates`; it does not blindly train on every retired message.

Good personalization candidates include confirmed corrections, repeated stable preferences, successful recurring intent-to-tool mappings, response-style preferences, and bounded RL/direct-feedback signals tied to inspectable source events. One-off facts, secrets, temporary paths, transient visual observations, unresolved claims, and raw capsule prose are memory/retrieval material unless separately promoted by policy.

### Non-blocking boundary transaction

Compression works against an immutable prefix while new events continue after its cutoff:

```text
reach high-water mark
  -> snapshot [current frontier, frozen cutoff]
  -> continue appending new events beyond cutoff
  -> build dialogue state + episodic gist + memory references
  -> derive eligible LoRA candidates
  -> validate coverage, pending-state preservation, and token savings
  -> atomically persist the boundary artifact
  -> compare-and-swap the compression frontier
  -> rebuild Buffer 1 from capsules/retrieval + untouched recent tail
  -> enqueue eligible LoRA candidates independently
```

Only one transaction may own a given frontier. If the frontier or source hash changed, discard or rebase the candidate. Capsule failure must not silently truncate input. LoRA failure must not prevent the already validated boundary artifact from advancing Buffer 1.

### Chosen V1 approach

Use a hybrid of:

- Dialogue State Tracking for goals, slots, constraints, pending actions, and referents;
- MemGPT-style virtual context management for paging exact material between the active prompt and GRIM's existing memory tiers;
- a short episodic gist for conversational and narrative continuity;
- LoRA for frequent, asynchronous consolidation of the behavioral signal emitted at the boundary.

Recurrent Memory Transformer remains a later experiment. It requires segment-level training so memory tokens learn read/write behavior, plus contracts for restart persistence and compatibility across adapter revisions. It may eventually augment Buffer 1, but it is not required for the first rolling-boundary-to-LoRA implementation.

Research anchors for these choices:

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) for explicit paging between a finite model context and external memory tiers.
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) for learned memory tokens and segment-level recurrence.
- [“Do you follow me?”: A Survey of Recent Approaches in Dialogue State Tracking](https://aclanthology.org/2022.sigdial-1.33/) for the role and limits of structured dialogue-state tracking.

## Trainer boundary

Prefer a separate `grim_lora_trainer` executable/target that reuses shared tokenizer, forward, autograd, checkpoint, and LoRA artifact code. A distinct target makes it easier to prove that the inference executable has no write path and that the full-model optimizer cannot accidentally update base parameters.

Training inputs are body-approved personalization candidates emitted at the rolling boundary and can be built from:

- confirmed `CorrectionTuple` records;
- `ToolTrainingExample` success/failure/gap records;
- retained memory records explicitly approved for personalization;
- routing and synthesis exemplars with explicit desired outputs;
- negative examples for rejected tools, unsafe actions, and unsupported assumptions.

Add a dataset builder that normalizes these sources into versioned examples with boundary ID, source event/memory IDs, provenance, reward weight, privacy/retention class, train/evaluation split, and deduplication key. Do not let the model runtime read raw memory to train itself. Crossing the compression frontier is not, by itself, permission to train on a record.

Trainer invariants:

- load the base checkpoint read-only and record its SHA-256;
- allocate only allowlisted LoRA targets;
- exclude every base `ParameterGroup` from the optimizer;
- optimize only A/B tensors;
- save optimizer state separately from the inference artifact;
- hash the base checkpoint before and after training and require equality;
- publish only after validation and evaluation;
- write no base-model checkpoint as a side effect of LoRA training.

Initial hyperparameters should be configuration, not constants. A reasonable starting experiment is rank 8, alpha 16, adapter dropout 0.05, attention QKV+output targets, and all 12 layers. Quality evaluation should decide whether to add FFN targets or change rank.

## Frequent micro-training operating model

The expected workload is not a conventional fine-tuning run. Rolling context boundaries may accumulate roughly 1-interval set/learned new eligible personalization sequences and trigger training as often as once per hour. The trainer must therefore be resumable, bounded, cheap to invoke, and safe on tiny datasets. It remains a near-term component of this design, not a deferred replacement for context management.

### Trigger and queue policy

The body owns a durable pending-personalization queue fed by committed `ContextBoundaryArtifact` records. Inference never starts training directly and the model never writes its own training records. Context compaction commits first; training consumes a frozen manifest independently. Inference keeps the current window and adapter revision; in exclusive-GPU mode new requests may queue briefly, while a future capacity-approved concurrent mode may continue serving them.

Recommended trigger rule:

- start when pending examples reach `max_new_sequences_per_run` (initially 10); or
- start when the oldest pending example reaches `max_wait_minutes` (initially 60); and
- require at least `min_new_sequences_per_run` (initially 1); and
- debounce after new input so a burst becomes one run rather than several one-example runs.

The scheduler takes an immutable manifest snapshot of at most 1-interval set/learned new sequence IDs, including their boundary and source-memory references. Records arriving after the snapshot remain pending for the next run. A single-writer lease prevents overlapping trainers or publishers.

Suggested canonical `ai_config.json` section:

```json
{
  "context_management": {
    "mode": "permanent_rolling",
    "capacity_source": "loaded_model",
    "tokenizer_source": "grim_text_active",
    "reserve_ratio": 0.28,
    "low_water_ratio": 0.60,
    "minimum_recent_tail_tokens": 256,
    "minimum_generation_reserve_tokens": 64,
    "compression_async": true,
    "preserve_source_event_refs": true
  },
  "lora_training": {
    "enabled": false,
    "trigger_source": "context_boundary_candidates",
    "min_new_sequences_per_run": 1,
    "max_new_sequences_per_run": 10,
    "max_wait_minutes": 60,
    "debounce_seconds": 120,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_optimizer_steps_per_run": 10,
    "passes_over_new_sequences": 1,
    "replay_ratio": 0.25,
    "max_replay_sequences": 8,
    "learning_rate": 0.0001,
    "resource_mode": "exclusive_gpu",
    "publish_only_if_evaluation_passes": true
  }
}
```

These are starting limits, not claimed optimal values. Training telemetry must record actual startup time, step time, peak VRAM, and evaluation time so the interval and step budget can be tuned locally.

### What "LoRA-only training" means mechanically

The full transformer still performs forward computation, and backward propagation must carry activation gradients through frozen base operations to reach adapters in earlier layers. That is different from training the full model:

- every base tensor has `requires_grad=false`;
- base tensors receive no gradient buffers and are absent from the optimizer inventory;
- LoRA A/B tensors have `requires_grad=true`;
- a dedicated `LoraParameterGroupRegistry` contains only A/B groups;
- the LoRA optimizer is constructed only from that registry;
- base matmuls still compute input gradients when needed, but never weight gradients;
- checkpoint save code available to this mode writes only LoRA artifacts and LoRA trainer state.

The existing TensorContract matmul creates a gradient node when either input requires gradients and records the two inputs' flags independently. This is compatible with frozen base weights plus trainable LoRA branches, but it needs an explicit test for input-gradient propagation with `base_weight.requires_grad=false` before relying on it.

Do not implement LoRA mode by building the normal full-model `parameter_groups` and then assigning a zero learning rate. Base groups must be structurally absent from the LoRA optimizer. This avoids full-model moment allocation, accidental weight decay, unnecessary gradient buffers, and a future configuration mistake re-enabling base updates.

### Resumable trainer state

Frequent tiny runs should not initialize a fresh optimizer every hour. Keep two separate products:

- inference artifact: `active.grimlora`, containing only A/B tensors and inference metadata;
- trainer state: `active.grimlora.trainstate`, containing LoRA-only optimizer moments, optimizer step, RNG state, base hash, adapter revision, target inventory hash, and consumed example IDs.

The existing `OptimizerCheckpoint` pattern can be reused conceptually, but its current implementation is bound to `TrainingContext::parameter_registry`. Add a LoRA-specific state reader/writer that validates exact A/B group names, ordering, sizes, base hash, and adapter revision.

If trainer state is missing or incompatible, training may deliberately restart optimizer state from the active adapter, but it must log that reset. It must never fall back to full-model optimizer state.

### Tiny-dataset protection

One to ten sequences are easy to overfit. Each run should:

- use token-level response masking so prompt/context tokens are not trained as desired output unless explicitly intended;
- mix a small, deterministic replay sample from previously accepted personalization examples;
- cap passes and optimizer steps rather than expanding a tiny batch into many epochs;
- clip adapter gradients and reject non-finite loss/gradients;
- monitor adapter norm and output-delta norm against configured limits;
- evaluate the new examples, replay examples, and stable router/synthesizer canaries;
- decline publication when the candidate has no meaningful benefit or exceeds a regression threshold.

With only one new sequence, replay and canaries are especially important. "No publish" is a successful training outcome when the evidence is too weak.

### Local resource modes

The adapter itself is small, but LoRA training still needs base-model forward/backward compute and activations. Its runtime cost is therefore not proportional only to the 1.31 MiB rank-8 adapter.

V1 defaults to `resource_mode="exclusive_gpu"`:

1. Body acquires a ResourceCoordinator training lease.
2. New router requests are paused and in-flight inference drains.
3. The inference worker is stopped/unloaded if VRAM cannot hold both processes.
4. `grim_lora_trainer` loads the base read-only, the active adapter, and LoRA trainer state.
5. It runs the bounded micro-training/evaluation job and exits.
6. If accepted, the candidate is atomically published.
7. The router restarts and must report the new adapter revision before requests resume.

A later `concurrent_if_capacity` mode may keep inference active while a separate trainer process holds another base copy, but only after ResourceCoordinator reserves the required VRAM. Sharing mutable adapter buffers between inference and training processes is out of scope.

### One micro-training transaction

```text
eligible boundary-derived examples (1-10)
  -> immutable run manifest + replay sample
  -> acquire single-writer/resource lease
  -> load base read-only
  -> load active LoRA + LoRA-only optimizer state
  -> bounded forward/backward steps (A/B only)
  -> candidate LoRA + candidate trainer state
  -> validation and canary evaluation
  -> atomic publish or reject
  -> mark manifest IDs consumed only after accepted/no-retry completion
  -> restart/verify inference when a new revision was published
```

On crash, the manifest remains recoverable. Temporary candidates are never treated as active. Example IDs make retries idempotent and prevent training the same event twice by accident.

An adapter update never changes the permanent session identity, token positions, boundary hashes, memory IDs, or committed capsules. Training observes one adapter revision; publication creates the next revision and activation occurs only between requests.

Manifest completion has three explicit outcomes:

- `accepted`: candidate published; example IDs move to the replay-eligible accepted set;
- `rejected`: evaluation completed but the candidate was not safe/useful; IDs are quarantined with the evaluation result and are not automatically retried unchanged;
- `retryable_failure`: infrastructure/resource/crash failure before a valid evaluation; IDs return to pending.

This prevents an hourly scheduler from repeatedly training and rejecting the same tiny batch forever.

## Implementation order

### Context source track - Buffer 1 and rolling boundary

- Make the one process-lifetime session invariant explicit and centralize its canonical ID.
- Reserve `destroySession()` for shutdown/tests and remove new-chat semantics from `clearConversationHistory()`/model switching.
- Add tokenizer-measured prompt accounting, monotonic event/token positions, pinned state, a recent-tail budget, and a compression frontier to `SessionContextManager`.
- Add immutable `ContextBoundaryArtifact` snapshots with dialogue state, episodic gist, memory references, correction/reward provenance, and personalization candidates.
- Commit boundary artifacts atomically and advance the frontier with compare-and-swap semantics while new turns append beyond the frozen cutoff.
- Feed eligible candidates from committed boundaries into the durable 1-interval set/learned sequence LoRA queue.

Exit criterion: one permanent conversation can pass multiple configured context limits without changing session identity, silently dropping live state, or waiting for LoRA training; every retired span has exact source/memory provenance and can feed the trainer.

### Slice 0 - Canonical startup contract (completed)

- Added named public/worker port arguments to the bridge while leaving model/vocabulary ownership with `train_gpu`.
- Removed the ignored positional model/vocabulary arguments from `ProcessManager`.
- Made MMO launches use the GRIM root as the working directory so `loadAiConfigSnapshot()` reliably finds `ai_config.json`.
- Removed hard-coded runtime ports from the bridge launch path.
- Added `/api/status`, which identifies the canonical config source and the model/vocabulary paths loaded by the worker.
- Removed a same-process health-check mutex re-entry in the Windows reuse path.

Exit criterion met in code: ports flow as process-scoped arguments, while worker status exposes the model/vocabulary selected from canonical `ai_config.json`. Focused build verification is required after each change to this chain.

### Slice 1 - Artifact and loader, no forward effect

- Add `grim_text_lora.fbs` and generated bindings.
- Add CPU validation, base hashing, GPU tensor ownership, and Phase 1 load/handoff.
- Create a neutral rank-8 fixture (`B=0`).
- Expose adapter identity in status.
- Tighten router-only/fail-closed config validation.

Exit criterion: GRIM-text starts with a compatible neutral artifact, rejects corrupt/incompatible artifacts, and reports exact adapter identity.

### Slice 2 - Inference application

- Add the shared low-rank projection helper.
- Wire both QKV and output projections in full and cached attention paths.
- Add numerical, determinism, cached/uncached, and base-immutability tests.
- Measure latency and VRAM.

Exit criterion: a deterministic non-zero fixture changes expected logits, a zero adapter is equivalent to base within the defined numeric tolerance, and base checkpoint/tensor hashes remain unchanged.

### Slice 3 - Trainer and candidate publication

- Add the separate on-demand LoRA micro-trainer target and A/B-only optimizer inventory.
- Add the durable 1-interval set/learned sequence queue, immutable run manifests, replay sampling, and idempotent example IDs.
- Add dataset normalization for boundary/correction/tool/approved-memory inputs with response-token loss masks.
- Add LoRA-only optimizer-state resume, bounded step budgets, candidate evaluation, atomic publication, and rollback.
- Integrate ResourceCoordinator leases and the default exclusive-GPU drain/train/restart lifecycle.
- Prove base checkpoint bytes are unchanged across a training run.

Exit criterion: jobs containing 1, 5, and 10 new sequences can resume LoRA-only optimizer state, train within configured step/resource bounds, validate, publish or reject, and roll back without allocating base optimizer state or writing the base checkpoint.

### Slice 4 - Body lifecycle integration

- Watch adapter identity or react to an explicit trainer-complete event.
- Drain/restart the GRIM-text router.
- Require adapter-aware health before routing resumes.
- Add structured errors and UI/telemetry state for load failure and rollback.

Exit criterion: an atomic adapter update becomes active on the next verified router process without partial reads or base-only fallback.

## Required tests

### Artifact and validation

- FlatBuffer round trip for FP32 and, when added, BF16 storage.
- Reject truncated files, unsupported versions, duplicates, unknown target kinds, bad layer indices, zero/excessive rank, bad element counts, NaN/Inf, and base-hash mismatch.
- Reject a valid router adapter when placed on a sub-model config.
- Reject enforced MMO startup with missing LoRA.

### Math and forward paths

- CPU reference for `(x A^T) B^T * alpha/rank` against GPU output.
- `B=0` produces base-equivalent logits within an explicit tolerance.
- Non-zero fixture produces a known logit delta.
- Full-sequence and KV-cache paths both apply the adapter.
- Batch/sequence edge cases, including one token and maximum configured sequence.
- Adapter tensors are read-only during inference.

### Base immutability

- Hash the base checkpoint before and after inference, training, publication, and reload.
- Snapshot/hash base GPU tensors around LoRA-only training steps in a test build.
- Assert the LoRA optimizer contains no pointer owned by `StartupParameterRegistry`.
- Ensure inference binaries contain no adapter-save or base-save call path.

### Frequent micro-training

- Trigger at 1-interval set/learned pending sequences or the configured maximum age, never with zero sequences.
- Snapshot at most 1-interval set/learned IDs; examples arriving during a run stay queued for the next run.
- Run successfully with 1, 5, and 10 new sequences plus deterministic replay.
- Verify normal base `parameter_groups` are absent from the trainer optimizer and have no moment tensors.
- Verify frozen base matmuls propagate required activation gradients without producing base-weight gradients.
- Save/reload LoRA-only optimizer state and reproduce the next update within numeric tolerance.
- Reject stale trainer state whose base hash, adapter revision, target inventory, name, order, or size differs.
- Crash before publication, after candidate write, and after publication; recover without activating a partial file or double-consuming examples.
- Reject over-norm, non-finite, no-benefit, and canary-regressing candidates without interrupting the current active adapter.

### Permanent session and rolling boundary

- Session identity remains constant across idle time, topic changes, task completion, compaction, model switching, and adapter publication.
- Only process shutdown/tests destroy the session; restart reconstructs continuity from capsules and persistent memory.
- Token accounting matches the actually serialized prompt within an explicit tolerance.
- Compression triggers before the hard limit and reaches the configured low-water target.
- Events appended after the frozen cutoff are never included in or removed by that transaction.
- Pending interactions, active constraints, commitments, and required referents survive compaction.
- Each boundary resolves to immutable source hashes and durable memory references.
- Boundary/capsule failure never silently truncates context; trainer failure never blocks a valid boundary commit.
- Context retirement does not automatically authorize LoRA training; only eligible candidate IDs enter manifests.
- Adapter revision changes do not invalidate session positions, boundary artifacts, or exact memory.

### Lifecycle

- Process arguments survive `ProcessManager -> grim_text_server -> train_gpu` unchanged.
- Adapter-aware readiness rejects wrong adapter ID/revision.
- Concurrent requests during publication see either the old verified process or the new verified process, never a partially loaded artifact.
- Candidate startup failure rolls back to the previous known-good adapter.

### Performance

- Record startup hash/load/upload time.
- Record adapter VRAM by rank and target count.
- Record prefill and decode latency with base-only shadow mode versus LoRA.
- Establish a regression budget before enabling enforced mode.

## Definition of done

LoRA support is ready for enforced MMO only when all of the following are true:

- the selected router config reaches the actual inference worker without hidden fallback;
- one separate adapter artifact is required and verified for the router;
- sub-models cannot load adapters;
- base and adapter tensors have separate owners;
- both attention execution paths apply the adapter;
- status proves which base and adapter are active;
- the trainer updates only adapter tensors and cannot save the base;
- hourly/on-demand 1-interval set/learned sequence jobs resume LoRA-only optimizer state and obey bounded step/resource policies;
- committed rolling boundaries feed those jobs without changing the single permanent session or placing training on the context-compaction critical path;
- publication is atomic and restart/rollback is coordinated;
- corrupt, missing, stale, or incompatible adapters fail closed;
- base immutability, numerical correctness, lifecycle, and performance tests pass.

## Deferred decisions

- FFN, LM-head, selector, or ExecutionBlock targets.
- Per-user multi-adapter hosting.
- Adapter composition or weighted stacking.
- Request-time enable/disable or scaling.
- In-process hot reload.
- Quantized adapter storage and fused CUDA kernels.
- Standard-format import/export (for example, PEFT/safetensors). The native runtime should first have a stable internal contract; converters can follow without making an external format the live ownership model.
