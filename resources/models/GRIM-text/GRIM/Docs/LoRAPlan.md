# Model-Side LoRA Extension Plan

Status: planning, not approved for implementation.

Last updated: 2026-09-04.

This is the living plan for adding low-rank adapters to GRIM-text model
matrices. Update the decision ledger and implementation phases as requirements
arrive; do not infer an adapter target merely because a tensor is trainable.

## Initial objective

Add LoRA to selected encoder attention and feed-forward matrix projections
without changing the semantic role or durable ownership of the base model
parameters. Initial work excludes embeddings, the main LM head, biases,
normalization parameters, residual controls, and domain-specific atom tensors.

For a selected base projection `W`, the intended mathematical boundary is:

`W_effective = W_base + delta_W`

`delta_W = (alpha / r) * B A`

`y = x W_effective`

where `W` remains the base matrix and `A` and `B` are separately owned LoRA
parameters. Rank ownership is per matrix class. Exact stored orientation,
scaling, and precision follow the per-class contracts below and must be
validated against each existing TensorContract matmul call during implementation.

The equations above use logical LoRA notation. The class-specific physical
`A`/`B` storage order is defined below and preserves
`delta_W = (alpha / r) * B A` mathematically.

## Frozen base-model training contract

Normal LoRA training freezes the entire pre-existing base model. This applies
to every model parameter whether its matrix class is enabled for LoRA, disabled,
or excluded from LoRA targeting. Enabled adapter factors are the only trainable
parameters in the model instance:

| Tensor role | Training state |
|---|---|
| Any pre-existing base-model parameter | Frozen |
| Enabled `lora_A` | Trainable |
| Enabled `lora_B` | Trainable |
| Disabled LoRA pair | Absent |

This includes embeddings, all attention and FFN weights and biases, RMSNorm
gammas, LayerScale tensors, attention residual gates, the LM head and its
residual MLP, selector/retrieval parameters, atom-domain parameters, and every
other parameter that existed before LoRA adapter allocation.

Frozen projection weights still participate in derivatives with respect to
their inputs, so gradients continue through base-model operations into earlier
activations and ultimately into upstream LoRA branches. The LoRA autograd path
must compute `grad_input`, `grad_A`, and `grad_B`; LoRA training must not
allocate, accumulate, register, clip, zero, checkpoint as optimizer state, or
update a gradient for any pre-existing base-model parameter.

Conceptually, each adapted parameter identity is:

```text
layerN_<projection>.base    FROZEN
layerN_<projection>.lora_A  TRAINABLE
layerN_<projection>.lora_B  TRAINABLE
```

The optimizer inventory contains only enabled `lora_A` and `lora_B` tensors. Any
pre-existing base-model parameter appearing in that inventory is a startup
validation error. Startup must set every pre-existing parameter non-trainable
before adapter registration and must validate that no base parameter owns an
allocated gradient buffer in LoRA training mode.

## V1 merge policy

V1 never merges an adapter into its base weight:

```text
base weight   immutable
adapter A/B   separately owned
inference     computes base output + scaled adapter output dynamically
```

The effective projection is computed without materializing or persisting a
modified base matrix:

`y = x W_base + (alpha / rank) * ((x A_logical) B_logical)`

The exact physical multiplication order follows the resolved GRIM tensor
orientation, but the ownership invariant does not change. The following pattern
is forbidden in training, inference, loading, checkpointing, and export:

```cpp
W_base += delta_W;
```

There is no in-place merge, temporary durable merged weight, export-time merge,
or runtime merge API in v1. Unloading the adapter leaves the original base
tensor byte-for-byte unchanged.

## Target decision ledger

| Registry name / tensor | Initial LoRA | Priority | Decision |
|---|---:|---|---|
| `layerN_qkv_weight` / `W_qkv` | Yes | Highest | First-tier attention target |
| `layerN_wo_weight` / `W_o` | Yes | High | Second-tier attention target |
| `layerN_ffn_w_gate` / `W_gate` | Yes | Highest | First-tier FFN target |
| `layerN_ffn_w1` / `W1` | Yes | Highest | First-tier FFN target |
| `layerN_ffn_w2` / `W2` | Yes | Highest | First-tier FFN target |
| Attention and FFN biases | No | Low value | Keep outside LoRA |
| RMSNorm gammas | No | Avoid | Keep outside LoRA |
| LayerScale tensors | No | Avoid | Keep outside LoRA |
| Attention residual gate | No, initially | Separate behavior | Do not model as a generic projection adapter |
| Embedding weights | No | Avoid | Keep outside initial adapter surface |
| Main LM-head weight | No, initially | Avoid | Revisit separately, especially with tied embeddings |
| LM-head residual MLP | Maybe later | Interesting | Three independently selectable future projections; deferred from v1 |
| Atom-insertion weights | No | Wrong semantic domain | Never auto-select from generic matrix shape |
| Local-atom-retrieval key | No | Wrong semantic domain | Never auto-select from generic matrix shape |

The initial target set is therefore exactly five matrix classes per encoder
layer. Selection must be explicit by parameter identity; shape-, bucket-, or
`requires_grad`-based discovery is not acceptable because it would capture
semantically excluded tensors.

### Deferred LM-head residual MLP targets

If LM-head residual MLP LoRA is added after v1, it has three independently
selectable projection targets:

- `LMHeadParameterTensors::mlp_W_gate`
- `LMHeadParameterTensors::mlp_W_up`
- `LMHeadParameterTensors::mlp_W_down`

Configuration may enable any subset of these three. Selecting one must not
implicitly allocate, register, load, or activate adapters for either of the
others. These targets remain outside the initial five-class encoder scope.

## Per-layer adapter topology

Every transformer layer `N` supports five distinct LoRA delta slots:

| Base projection | Effective projection |
|---|---|
| `layerN_qkv_weight` / `W_qkv` | `W_qkv + delta_qkv` |
| `layerN_wo_weight` / `W_o` | `W_o + delta_o` |
| `layerN_ffn_w_gate` / `W_gate` | `W_gate + delta_gate` |
| `layerN_ffn_w1` / `W1` | `W1 + delta_1` |
| `layerN_ffn_w2` / `W2` | `W2 + delta_2` |

For each enabled matrix class, every transformer layer owns one layer-local,
projection-local delta. A disabled class owns no adapter tensors in any layer.
There is no adapter sharing across transformer layers or across these five
matrix classes. `W_qkv` remains one fused base projection with one fused LoRA
delta; it is not split into independent Q, K, and V adapter targets.

## Physical orientation contract

The v1 adapter orientations follow the existing GRIM projection storage and
TensorContract matmul direction exactly.

| Matrix class | Base shape | `A` shape | `B` shape | Unscaled factored delta output |
|---|---|---|---|---|
| `W_qkv` | `[qkv_dim, d_model]` | `[r_qkv, d_model]` | `[qkv_dim, r_qkv]` | `(x A^T) B^T` |
| `W_o` | `[d_model, d_model]` | `[r_o, d_model]` | `[d_model, r_o]` | `(x A^T) B^T` |
| `W_gate` | `[d_model, d_ff]` | `[r_gate, d_ff]` | `[d_model, r_gate]` | `(x B) A` |
| `W1` | `[d_model, d_ff]` | `[r_1, d_ff]` | `[d_model, r_1]` | `(x B) A` |
| `W2` | `[d_ff, d_model]` | `[r_2, d_model]` | `[d_ff, r_2]` | `(x B) A` |

For attention projections, GRIM stores the base output-major and calls
TensorContract matmul with `transpose_b=true`:

```text
W_qkv:
      base [qkv_dim, d_model]
      A    [r_qkv, d_model]
      B    [qkv_dim, r_qkv]
      delta(x) = (x A^T) B^T

W_o:
      base [d_model, d_model]
      A    [r_o, d_model]
      B    [d_model, r_o]
      delta(x) = (x A^T) B^T
```

Their physical stored deltas are `B A`, matching each output-major base shape.

For FFN projections, GRIM stores the base in direct forward-matmul orientation:

```text
W_gate:
      base [d_model, d_ff]
      B    [d_model, r_gate]
      A    [r_gate, d_ff]
      delta(x) = (x B) A

W1:
      base [d_model, d_ff]
      B    [d_model, r_1]
      A    [r_1, d_ff]
      delta(x) = (x B) A

W2:
      base [d_ff, d_model]
      B    [d_ff, r_2]
      A    [r_2, d_model]
      delta(x) = (x B) A
```

Their physical stored deltas are also `B A`, matching each direct base shape.
The runtime computes the factored delta path and must not materialize that full
matrix. Startup and artifact loading must validate these exact class-specific
shapes and orientations; square `W_o` geometry must not be used to infer or
hide an incorrect transpose contract.

## Authored per-class configuration

`model_config.json.lora_model` is the immutable model-level LoRA semantic. It
is compiled into `model.grimcfg`, included in the required-capability digest,
and must match whether the training policy enables any LoRA matrix class.

All v1 LoRA settings are required flat leaves on `TrainingHyperparameters`.
Each matrix class has an explicit enable flag, rank, alpha, and precision:

| Matrix class | Enable field | Rank field | Alpha field | Precision field |
|---|---|---|---|---|
| `W_qkv` | `lora_qkv_enabled` | `lora_qkv_rank` | `lora_qkv_alpha` | `parameter_precision_lora_qkv` |
| `W_o` | `lora_o_enabled` | `lora_o_rank` | `lora_o_alpha` | `parameter_precision_lora_o` |
| `W_gate` | `lora_gate_enabled` | `lora_gate_rank` | `lora_gate_alpha` | `parameter_precision_lora_gate` |
| `W1` | `lora_w1_enabled` | `lora_w1_rank` | `lora_w1_alpha` | `parameter_precision_lora_w1` |
| `W2` | `lora_w2_enabled` | `lora_w2_rank` | `lora_w2_alpha` | `parameter_precision_lora_w2` |

`HyperparameterGroupings.hpp` exposes one typed LoRA read view with one
class-settings value per matrix class. Each class-settings value contains only:

```cpp
bool enabled;
uint32_t rank;
float alpha;
HyperParameters::ParameterGroupPrecision precision;
```

The grouped LoRA read view additionally carries one adapter-set-wide authored
field:

```cpp
float learning_rate_lora;
```

`learning_rate_lora` is not duplicated into the five class-settings values.

The grouping is a read view over `TrainingHyperparameters`, not a config owner.
Startup allocation, registration, artifact validation, and forward-view
construction consume this grouped payload rather than receiving raw config or
separate enable/rank/alpha/precision arguments.

When a class is enabled, startup allocates and registers one `LoRAParameterPair`
for that class in every transformer layer and validates its rank, alpha,
precision, shape, and orientation. When a class is disabled, startup allocates
and registers no pair for that class; a forward or artifact path that supplies
one is an error. Missing authored fields are configuration errors, not requests
for implicit defaults.

## V1 layer targeting policy

Each `lora_*_enabled` flag is model-wide across transformer layers. Enabling a
matrix class enables that projection in every layer `0..num_layers-1` with the
same class rank, alpha, scale, and precision. Per-layer targeting is unsupported
in v1.

Configuration and artifacts must not contain layer allowlists, denylists,
ranges, masks, sparse target maps, or per-layer enable/disable overrides. For an
enabled class, the registry and adapter artifact must contain exactly one target
record for every transformer layer. A missing layer, duplicate layer, extra
layer, or partially populated class is a hard startup/load failure. For a
disabled class, they must contain zero target records.

## Rank policy

LoRA rank is configured per matrix class:

- `W_qkv` rank
- `W_o` rank
- `W_gate` rank
- `W1` rank
- `W2` rank

Every transformer layer uses the configured rank for its corresponding matrix
class. Ranks may differ between matrix classes, but there are no per-layer rank
overrides and no single global rank that silently controls all five classes.
Startup validation must reject missing, zero, negative, or dimensionally
invalid rank values for every enabled class.

## Alpha and scaling policy

LoRA `alpha` is authored per matrix class, matching the five independent rank
classes:

| Matrix class | Rank | V1 authored alpha | Initial scale |
|---|---|---|---|
| `W_qkv` | `r_qkv` | `alpha_qkv = r_qkv` | `1.0` |
| `W_o` | `r_o` | `alpha_o = r_o` | `1.0` |
| `W_gate` | `r_gate` | `alpha_gate = r_gate` | `1.0` |
| `W1` | `r_1` | `alpha_1 = r_1` | `1.0` |
| `W2` | `r_2` | `alpha_2 = r_2` | `1.0` |

Every class computes scaling from its persisted values:

`scale_class = alpha_class / rank_class`

V1 authors `alpha_class = rank_class`, so:

`scale = alpha / rank = 1.0`

Alpha and rank remain explicit, separately persisted values even when equal.
Every transformer layer uses its matrix class's authored alpha and rank; there
are no global or per-layer alpha overrides. When an authored v1 configuration
changes a class rank, its authored default alpha changes with it, preserving
scale `1.0` instead of accidentally changing adapter strength. For example,
`r_qkv = 8, alpha_qkv = 8` and `r_1 = 16, alpha_1 = 16` both produce scale
`1.0`.

The config authoring boundary must materialize and persist both values. Runtime
startup must reject missing or non-positive alpha for an enabled class and must
not silently substitute rank for a missing alpha. Keeping alpha explicit allows
later experiments such as `alpha_qkv = 2 * r_qkv` while `alpha_1 = r_1` without
changing the scaling equation or introducing one global alpha.

## Precision policy

V1 uses FP32 adapter tensor storage, FP32 adapter compute, and FP32 optimizer
state. Precision remains authored per matrix class so a future implementation
can change one adapter class without introducing a model-wide mode:

- `parameter_precision_lora_qkv`
- `parameter_precision_lora_o`
- `parameter_precision_lora_gate`
- `parameter_precision_lora_w1`
- `parameter_precision_lora_w2`

These are proposed flat `training.config.parameter_precision_*` leaves authored
on `TrainingHyperparameters`. `HyperparameterGroupings.hpp` must expose the
typed grouped read view consumed by startup and parameter registration. Do not
copy LoRA fields into a legacy config handoff, create a LoRA precision sidecar,
or infer adapter precision from the base attention/FFN group. Parameter
registration must stamp each `A` and `B` tensor and its `ParameterGroup` from
the corresponding LoRA matrix-class field.

All five fields must be authored as `fp32` in v1. Any other value, including
`bf16_compute`, is a fail-loud startup error until that precision contract is
explicitly implemented for LoRA storage, compute, and optimizer state.

## V1 optimizer policy

LoRA training reuses the same configured optimizer family and optimizer
dispatch as normal model training. V1 does not introduce a LoRA-specific
optimizer implementation.

The required flat `TrainingHyperparameters` field `learning_rate_lora` supplies
one learning rate for the entire active adapter set. During model-wide-frozen LoRA
training, the Optimizer Window uses it as the current learning rate for the
adapter-only parameter inventory. V1 has no per-matrix-class LoRA learning
rates, no LR ratio derived from the normal model learning rate, and no second
optimizer invocation. A missing, non-finite, or non-positive
`learning_rate_lora` is a fail-loud configuration error.

Registration stamps every LoRA `A` and `B` parameter group with:

```text
weight_decay_multiplier = 0.0
lr_multiplier = 1.0
parameter_precision = FP32
```

Both factors therefore use `learning_rate_lora` directly and neither receives
weight decay. Optimizer-state allocation must create and validate FP32 first-
and second-moment tensors for every LoRA group. Optimizer family,
algorithm-specific coefficients, and step ownership remain on the existing
optimizer configuration and Optimizer Window boundaries; LoRA must not mirror
them in its config.

## Initialization and dropout policy

For every enabled adapter in v1:

- `lora_A` uses seeded Xavier uniform initialization with gain `1.0` over
      `A`'s own logical `fan_in` and `fan_out` dimensions:
      `A_ij ~ U[-sqrt(6 / (fan_in + fan_out)), +sqrt(6 / (fan_in + fan_out))]`.
- `lora_B` is exactly zero-initialized.
- The initial `delta_W = (alpha / r) * B A` is therefore exactly zero, making
      the adapter an exact no-op at creation.
- Adapter dropout is not used; its v1 value is `0.0`.

This preserves the base model's output when an untrained adapter is attached.
On the first backward pass, `B` can receive a gradient while `A` initially has
a zero gradient because `B` is zero; after `B` changes, both factors can train.
Startup must compute Xavier bounds from the adapter `A` tensor, not from the
full base matrix. The seed must come from an authoritative authored startup
value and be derived from stable adapter identity rather than allocation order,
wall-clock state, or an implicit global RNG. A missing required seed is a
fail-loud configuration error. V1 must not allocate a dropout mask or add a
dropout operation to the adapter path.

## Active adapter cardinality

A LoRA-enabled model instance has exactly one active adapter set. That set owns
the `A` and `B` factors for all enabled matrix classes and transformer layers in
the instance. V1 does not stack, blend, compose, or concurrently load multiple
named adapter sets. Startup/load validation must reject a second active set
instead of defining an implicit ordering or combination rule.

## Manifest and user-selection policy

The intended product choice is to attach LoRA to the router. That choice is
user-facing policy, not an invariant of `ModelRole`. The manifest schema must
therefore permit `lora_path` for the model instance explicitly selected by the
user, whether its role is `Router` or `SubModel`; runtime code must not infer
LoRA eligibility solely from model role.

The user-facing flow should present the router as the intended selection while
persisting the user's explicit target model. A LoRA-enabled instance then loads
its one active adapter set from its authored `lora_path`. A configured adapter
selection with a missing path is an error, and an unselected model must not load
an adapter merely because a path happens to be present.

The current `grim_model_manifest.fbs` comment combines `lora_path` and
`hard_copy_path` under one router-only restriction. The schema revision must
separate those policies. This LoRA decision does not change `hard_copy_path`
ownership.

## Adapter artifact compatibility contract

Every adapter artifact must carry enough immutable metadata to prove that its
single adapter set belongs to the exact base checkpoint and projection geometry
being loaded. Required artifact-level fields are:

- adapter artifact schema version,
- adapter ID,
- adapter revision,
- exact base checkpoint SHA-256,
- architecture metadata required for diagnostics.

### Adapter lineage and revision identity

`adapter_id` identifies the logical adapter lineage across continual adaptation.
It remains stable across every revision produced for the same purpose and
lineage. For example:

```text
adapter_id = user-writing-style
```

`adapter_revision` identifies one immutable trained tensor checkpoint within
that lineage. It is an explicit monotonically increasing integer, for example:

```text
adapter_revision = 17
```

The composite `(adapter_id, adapter_revision)` is the durable adapter artifact
identity. Revision numbers are scoped to their adapter ID; revision `17` of one
adapter is unrelated to revision `17` of another.

Published revisions are append-only and immutable. Saving or publishing uses
create-new semantics and hard-fails if the destination identity already exists.
It must never overwrite revision `17` with different tensors, metadata, or base
compatibility facts. Resuming or continually adapting revision `17` reads it as
an immutable source and writes a new revision with a greater revision number.
Checkpoint cleanup must not delete or rewrite published adapter revisions.

`lora_path` resolves to one explicit immutable adapter revision. A mutable
`latest` alias may be a user-interface convenience, but it must resolve to a
specific `(adapter_id, adapter_revision)` before loading and must not become the
persisted compatibility identity.

The base checkpoint digest covers the exact immutable base-weight artifact, not
only a model name, config hash, architecture family, or latest-checkpoint alias.
The loader must compare all 32 SHA-256 bytes before activating the adapter. A
wrong base hash is an unconditional hard load failure.

Every adapted projection record must contain:

- stable target identity,
- transformer layer index,
- matrix class,
- base matrix shape,
- physical orientation contract,
- rank,
- alpha,
- `A` shape,
- `B` shape,
- storage dtype.

Architecture metadata for diagnostics must include enough facts to explain and
validate target geometry, including at least `d_model`, `num_layers`,
`num_heads`, `num_kv_heads`, fused `qkv_dim`, and `d_ff`. It is diagnostic and
validation metadata, not an alternate model-config owner.

Artifact loading is all-or-nothing. Before any adapter tensor becomes active,
the loader must validate schema support, adapter ID/revision, exact base digest,
architecture facts, complete target inventory, unique target identities, layer
bounds, enabled matrix classes, base shapes, orientations, rank/alpha values,
`A`/`B` shapes, and storage dtypes. Missing, unknown, duplicate, inconsistent,
or unsupported metadata is a hard load failure; partial adapter activation is
forbidden.

## Checkpoint resume ownership

A deployable LoRA adapter artifact and a resumable LoRA training checkpoint are
different artifacts with different owners and contents.

### LoRA adapter artifact

The inference/deployment artifact owns only:

- immutable trained `A` and `B` tensors,
- adapter schema and target metadata,
- `adapter_id` and immutable `adapter_revision`,
- exact base-checkpoint SHA-256,
- rank, alpha, orientation, shapes, dtype, and diagnostic architecture metadata.

It does not own optimizer moments, gradients, optimizer/training steps, gradient
accumulation state, scheduler state, RNG state, data-order state, or other
training-loop state. `lora_path` points to this deployable artifact only. An
adapter artifact is insufficient for exact training resume and must not trigger
a fresh-optimizer fallback when resume was requested.

### LoRA training checkpoint

The training checkpoint owns the complete Category 2 state required to resume
the interrupted LoRA run exactly:

- the working adapter `A` and `B` tensors plus complete adapter metadata,
- `adapter_id`, parent/source adapter revision when applicable, and the intended
      unpublished output revision,
- exact base-checkpoint SHA-256 and base artifact identity,
- FP32 optimizer first- and second-moment tensors for every `A` and `B`,
- optimizer family/configuration and optimizer step,
- `learning_rate_lora` and complete LR scheduler/soft-restart state,
- gradient-accumulation cursor/current micro-step,
- accumulated `A` and `B` gradients when the cursor is inside an accumulation
      window,
- epoch/batch/data-order cursor required to resume the same sample sequence,
- deterministic RNG/seed state required by initialization, data order, and any
      stochastic training operation,
- training configuration/schema compatibility metadata.

The frozen base tensors are not copied into the LoRA training checkpoint. Resume
must load the exact external base checkpoint identified by the recorded SHA-256
and hard-fail on any mismatch.

Training checkpoints may be written only at a safe graph boundary after a
microbatch backward has completed and Category 1 graph state has been cleared.
They never serialize activations, forward intermediates, GradFns, or autograd
tape state. If the accumulation cursor is nonzero, the persisted adapter
gradients are mandatory because they are Category 2 accumulation-window state;
missing gradients are a hard resume failure.

Resume is all-or-nothing. A requested LoRA resume requires a valid LoRA training
checkpoint and restores tensors, moments, steps, scheduler, accumulation cursor
and gradients, data cursor, and RNG state before the next microbatch. It must not
silently accept an inference adapter artifact, zero optimizer state, reset the
scheduler, restart the accumulation window, or choose a different base.

A working training checkpoint is not a published immutable adapter revision.
When training completes or explicitly publishes, the publisher extracts the
deployable adapter state and creates a new immutable `adapter_revision`. It does
not overwrite the parent revision or expose optimizer/training state through
`lora_path`.

The v1 host-load I/O boundary is `GRIM::Checkpoint::loadLoRATrainingCheckpoint`
in `Common/LoRATrainingCheckpoint.hpp`. A bare `.grimlorackpt` filename resolves
under `<selected-model-directory>/lora_checkpoints`, where the selected model
directory is taken directly from the loaded `model.grimcfg` source path. Absolute
paths, parent components, alternate model directories, and non-LoRA checkpoint
extensions are rejected. The future startup caller must pass the identity and
SHA-256 of the base checkpoint it just loaded plus the authoritative LoRA
training-config SHA-256. The loader returns a fully validated host snapshot and
does not mutate `TrainingContext`, GPU tensors, optimizer state, or registry
ownership. Caller wiring and GPU-state restoration remain part of the later
durable-owner phase.

## Ownership constraints

- Base tensors remain Category 2 durable state owned by
  `ParameterRegistry::StartupParameterRegistry`.
- LoRA tensors are also Category 2 durable training state. They must have one
  owner, one registration path, and independent optimizer/checkpoint identity.
- Every pre-existing base-model parameter remains durable model state but is
      frozen during LoRA training and absent from the optimizer inventory, whether
      its class is enabled, disabled, or excluded from LoRA.
- Forward may read base and adapter tensors and create Category 1 intermediates,
  but it must not merge into or mutate the durable base tensor.
- Training, inference, serialization, and export must keep adapter tensors
      separate from the immutable base tensor; v1 has no merge operation.
- Read-only inference must receive detached base and adapter views through the
  existing shared-forward boundary.
- All new LoRA configuration is authored on `TrainingHyperparameters` and read
      through typed views in `HyperparameterGroupings.hpp`. Do not add LoRA fields
      to a legacy config mirror or create another config owner.
- Disabled LoRA must mean adapter tensors are absent. Do not allocate inert
  zero-rank tensors or silently fall back when an enabled adapter is missing.
- Per-class enable flags control allocation and registration exactly: enabled
      means one pair per transformer layer; disabled means no pair in any layer.
- V1 has no per-layer targeting or override surface. Enabled classes must cover
      the complete transformer layer range.
- A LoRA-enabled model instance owns exactly one active adapter set; adapter
      tensors from multiple sets must never coexist in its active registry state.
- Domain-specific matrices remain excluded even if they have compatible
  dimensions.

## LoRA type and ownership boundary

V1 uses two types. There is no `TensorLoRA` type.

### Durable parameter owner

```cpp
enum class LoRAMatrixClass : uint8_t {
      QKV,
      ATTENTION_OUTPUT,
      FFN_GATE,
      FFN_UP,
      FFN_DOWN
};

struct LoRAParameterPair {
      GRIM::Tensor A;
      GRIM::Tensor B;

      uint32_t rank;
      float alpha;
      float scale;

      LoRAMatrixClass matrix_class;
      GRIM::HyperParameters::ParameterGroupPrecision precision;
};
```

`LoRAParameterPair` is Category 2 durable state owned by
`ParameterRegistry::StartupParameterRegistry`, adjacent to the base parameter
bundle for the same layer and matrix class. It owns only adapter tensors and
immutable adapter facts. `rank`, `alpha`, `matrix_class`, and `precision` are
stamped from the authoritative grouped configuration/artifact contract at
startup. `scale` is derived once as `alpha / rank` and validated; it is not a
second independently authored scaling value.

The pair does not own, alias, or wrap the base tensor. It must not contain
temporary `(x A^T)`, `(x B)`, or other activation buffers, GradFns, saved
backward state, dropout state, optimizer moments, or per-call execution state.
Optimizer moments remain owned through the existing optimizer/`ParameterGroup`
boundary rather than inside the pair.

### Borrowed forward view

```cpp
struct LoRAProjectionView {
      GRIM::Tensor* A;
      GRIM::Tensor* B;

      uint32_t rank;
      float scale;
};
```

`LoRAProjectionView` is a non-owning, per-call forward payload. It borrows views
derived from one registry-owned `LoRAParameterPair`; those views and their
underlying tensors must outlive the forward call. An enabled projection requires
non-null, valid `A` and `B` views and matching rank, scale, shapes, matrix class,
and precision facts. Missing or mismatched views fail loudly.

The borrowed fields are non-const `Tensor*`, not `TensorView*` or
`const Tensor*`. The existing `TensorContract::TensorView` carries raw
data/shape/precision only and cannot identify an autograd leaf or receive
persistent `A`/`B` gradients. `Tensor::ensure_grad()` and mutable
`Tensor::grad_data()` also require non-const tensor identity. Borrowing mutable
`Tensor` identities preserves autograd connectivity without transferring
ownership or requiring `const_cast`.

The mutability is gradient-specific: forward treats `A.data` and `B.data` as
read-only, while backward accumulates only into their preallocated leaf-gradient
buffers. Startup must call `ensure_grad()` for every enabled `A` and `B` before
forward. No LoRA code may cast away constness, mutate adapter parameter data in
forward/backward, or lazily reinterpret a raw `TensorView` as a trainable leaf.

The view contains no base tensor because the base projection is supplied through
its existing base-tensor view boundary. It contains no `alpha` because forward
consumes the validated derived scale, and it contains no matrix class or
precision duplicate because those facts are validated while constructing the
class-specific views and are already carried by the borrowed `TensorView`
metadata where applicable.

Training constructs views connected to live adapter parameters; read-only
inference constructs detached adapter views. Neither mode permits the forward
view to become a durable owner or mutate the base or adapter tensors.

```text
StartupParameterRegistry
      |
      +-- base W
      |
      +-- LoRAParameterPair
                  +-- A
                  +-- B
                  +-- rank
                  +-- alpha
                  +-- scale
                  +-- matrix_class
                  +-- precision

forward
      +-- existing BaseTensorView
      +-- borrowed LoRAProjectionView
```

Category 1 LoRA activations and GradFn saved state belong to the autograd tape
created by the LoRA TensorContract primitive. They must never be stored on
`LoRAParameterPair`, `LoRAProjectionView`, or `StartupParameterRegistry`.

## Forward primitive API

The shared autograd primitive is:

```cpp
enum class MatmulOrientation : uint8_t {
      TRANSPOSED_WEIGHT,
      DIRECT_WEIGHT
};

GRIM::Tensor lora_linear(
      const GRIM::Tensor& x,
      const GRIM::Tensor& W_base,
      const LoRAProjectionView* lora,
      MatmulOrientation orientation,
      cudaStream_t stream);
```

The CUDA stream is explicit because all GRIM forward execution handles come
from the caller. `MatmulOrientation` is explicit and never inferred from shape:

- `TRANSPOSED_WEIGHT`: `W_qkv` and `W_o`.
- `DIRECT_WEIGHT`: `W_gate`, `W1`, and `W2`.

Dispatch semantics are:

```text
lora == nullptr
      ordinary base projection

lora != nullptr
      base projection + (alpha / rank) * factored adapter projection
```

A null pointer is an explicit ordinary-projection mode selected by a caller for
an authored-disabled matrix class. It is not a missing-adapter fallback. Before
calling `lora_linear`, the layer boundary must fail loudly if the class is
enabled but its view is null, or disabled but a view is supplied. In LoRA
training mode, the null path still receives a frozen `W_base` because startup
has frozen the entire base model, not because `lora_linear` infers trainability
from pointer presence.

The primitive performs:

```text
TRANSPOSED_WEIGHT:
      base_out  = matmul(x, W_base, transpose_b=true)
      rank_out  = matmul(x, A,      transpose_b=true)
      delta_out = matmul(rank_out, B, transpose_b=true)

DIRECT_WEIGHT:
      base_out  = matmul(x, W_base, transpose_b=false)
      rank_out  = matmul(x, B,      transpose_b=false)
      delta_out = matmul(rank_out, A, transpose_b=false)

output = base_out + scale * delta_out
```

It never materializes `delta_W`, mutates `W_base`, or changes any tensor's
trainability metadata. The primitive is mode-neutral: startup owns trainability.
In LoRA training, every base tensor passed to either the null or non-null path is
frozen while autograd remains connected to `x` and, when present, `A` and `B`.
Outside LoRA training, the null path may serve ordinary full-model training
under that mode's separately established parameter policy.

The primitive validates non-null tensor data, rank, finite positive scale,
orientation-specific base/`A`/`B` shapes, precision agreement, and output-shape
agreement before launching work. All Category 1 `rank_out`, `delta_out`, scaled
delta, and addition state remains on the autograd tape.

## Backward equations and gradient ownership

Let `G = dL/dY` and `s = alpha / rank`. These equations are normative for the
LoRA backward implementation.

### Direct FFN orientation

For `W_gate`, `W1`, and `W2`:

```text
Y = X W + s (X B) A
R = X B

dA = s R^T G
dB = s X^T (G A^T)

dX_adapter = s (G A^T) B^T
dX_base    = G W^T
dX         = dX_base + dX_adapter

dW_base = NONE
```

### Transposed attention orientation

For `W_qkv` and `W_o`:

```text
Y = X W^T + s (X A^T) B^T
R = X A^T

dB = s G^T R
dA = s (G B)^T X

dX_adapter = s (G B) A
dX_base    = G W
dX         = dX_base + dX_adapter

dW_base = NONE
```

`dW_base = NONE` applies throughout LoRA training because the entire base model
is frozen, including authored-disabled classes that take the null adapter path.
The base projection still contributes `dX_base`, so freezing parameters must not
detach the activation graph.

Every `dA` and `dB` write targets a Category 2 persistent leaf-gradient buffer
and must accumulate with `+=` or `atomicAdd`; it must never overwrite an existing
contribution. `dX_base` and `dX_adapter` must both reach the same input gradient
through autograd accumulation. Owned non-leaf scratch is zero-initialized before
use and remains Category 1. The implementation must not allocate or write a base
parameter gradient buffer in LoRA mode.

When `lora == nullptr` in LoRA mode, backward computes only `dX_base` and no
parameter gradient. When `lora != nullptr`, backward computes `dX_base`,
`dX_adapter`, `dA`, and `dB`, while still producing no `dW_base`.

## Planned implementation phases

### Phase 0: Resolve contracts

- [x] Freeze every pre-existing base-model parameter during LoRA training;
      train only enabled adapter `A` and `B` tensors.
- [x] Configure rank per matrix class, consistently across all transformer
      layers, with no global or per-layer rank override.
- [x] Initialize `A` with seeded gain-1 Xavier uniform over its own logical
      shape and initialize `B` to exact zeros for every adapter, yielding an
      exact no-op at creation.
- [x] Use no adapter dropout in v1; its value is `0.0` and there is no adapter
      dropout mask or operation.
- [x] Author alpha per matrix class and initialize `alpha = rank`, giving each
      v1 class `alpha / rank = 1.0` with no per-layer overrides.
- [x] Use FP32 storage, FP32 compute, and FP32 optimizer state in v1, configured
      through five LoRA matrix-class `parameter_precision_*` fields on the
      authoritative `TrainingHyperparameters` surface.
- [x] Add `lora_qkv_enabled`, `lora_o_enabled`, `lora_gate_enabled`,
      `lora_w1_enabled`, and `lora_w2_enabled`, with explicit rank, alpha, and
      precision fields for each class on `TrainingHyperparameters`.
- [x] Compile the required `model_config.json.lora_model` semantic into the
      model artifact and require it to match the active per-class policy.
- [x] Expose the five class settings through one typed grouped read view; do not
      pass raw config or parallel setting arguments to consumers.
- [x] Apply each enabled matrix class to every transformer layer; do not support
      layer masks, lists, ranges, or per-layer overrides in v1.
- [x] Use output-major `A`/`B` adapters with transpose matmuls for `W_qkv` and
      `W_o`, and direct-forward `B`/`A` adapters for `W_gate`, `W1`, and `W2`,
      with the exact shapes in the physical orientation contract.
- [x] Use one fused LoRA delta for each layer's fused `W_qkv` projection.
- [x] Never merge adapters into base weights in v1; compute base and adapter
      contributions dynamically and keep their durable ownership separate.
- [x] Require artifact schema version, adapter ID/revision, exact base-checkpoint
      SHA-256, per-target identity/geometry/scaling/dtype metadata, and
      architecture diagnostics metadata.
- [x] Define `adapter_id` as the stable logical lineage and `adapter_revision` as
      one immutable, monotonically increasing trained checkpoint in that lineage.
- [x] Treat the LoRA target as an explicit user-facing model selection, with
      the router as the intended selection rather than a `ModelRole` invariant.
- [x] Model future LM-head residual MLP support as three independently
      selectable projections: `mlp_W_gate`, `mlp_W_up`, and `mlp_W_down`.
- [x] Use registry-owned Category 2 `LoRAParameterPair` plus borrowed per-call
      `LoRAProjectionView`; do not create a combined `TensorLoRA` owner.

### Phase 1: Durable parameter ownership

- [ ] Add `LoRAMatrixClass` and `LoRAParameterPair` adjacent to the corresponding
      encoder and FFN base tensor owners in `ParameterRegistry.hpp`.
- [ ] Add `LoRAProjectionView` at the shared forward/TensorContract request
      boundary as a borrowed, non-owning payload.
- [ ] Allocate adapter pairs only for explicitly enabled target identities.
- [ ] Register only adapter `A` and `B` tensors as optimizer parameter groups;
      reject any pre-existing base-model parameter found in that inventory, and stamp
      LoRA precision from the corresponding authored matrix-class field.
- [ ] Before adapter registration, mark every pre-existing model parameter
      non-trainable and fail if any owns an allocated gradient buffer.
- [ ] Add fail-loud startup validation for target existence, dimensions, rank,
      duplicate identities, missing enabled adapters, and missing initialization
      seed state.
- [x] Require exact base-checkpoint SHA-256 compatibility and hard-fail every
      wrong-base load before adapter activation.

### Phase 2: Forward and autograd

- [ ] Add a TensorContract/autograd LoRA projection primitive with complete
      gradients for input, `A`, and `B`; in LoRA mode it must never allocate or
      accumulate a gradient for any base parameter.
- [x] Define exact direct-FFN and transposed-attention backward equations,
      including `dX_base + dX_adapter` and `dW_base = NONE`.
- [x] Use non-const borrowed `Tensor*` for `A` and `B` so backward can legally
      accumulate leaf gradients without `const_cast`; parameter data remains
      read-only outside optimizer update.
- [x] Define `lora_linear(x, W_base, lora, orientation, stream)` with explicit
      direct/transposed orientation and null meaning authored ordinary projection.
- [ ] Integrate the primitive at the five selected projection call sites; do
      not fork complete encoder or FFN forward implementations.
- [ ] Preserve the shared-forward read-only contract and Category 1 lifetime
      for adapter intermediates.
- [ ] Add Rule 21 equation diagnostics for the base projection, adapter delta,
      scaling, and combined output.
- [ ] Verify persistent leaf gradients accumulate with `+=` or `atomicAdd` in
      every new GradFn path.

### Phase 3: Training and optimizer policy

- [x] Define normal LoRA trainability as an entirely frozen pre-existing base
      model with only enabled adapter `A`/`B` tensors trainable.
- [x] Reuse the configured model optimizer family and existing optimizer
      dispatch for LoRA training.
- [x] Add one required `learning_rate_lora` for all adapter `A`/`B` groups; do
      not add per-class LoRA learning rates in v1.
- [x] Stamp `weight_decay_multiplier=0.0` and `lr_multiplier=1.0` on every LoRA
      `A` and `B` parameter group.
- [x] Require FP32 optimizer state for every v1 adapter parameter group.
- [ ] Ensure clipping, telemetry, finite checks, and accumulation windows include
      adapter gradients exactly once.
- [ ] Verify zeroing and optimizer-state ownership do not alias base tensors.
- [ ] Add LoRA training-checkpoint save/resume for working `A`/`B`, FP32 moments,
      optimizer/scheduler steps, accumulation cursor and partial gradients, data
      cursor, RNG state, and exact base identity.
- [x] Add strict host-side `.grimlorackpt` load I/O and schema validation rooted
      in the currently selected model-store directory; leave startup caller and
      GPU-state mutation unwired until LoRA durable owners exist.
- [ ] Restrict training-checkpoint writes to post-backward graph boundaries and
      exclude all Category 1 activations, GradFns, and tape state.

### Phase 4: Serialization and inference

- [x] Permit exactly one active adapter set per LoRA-enabled model instance;
      do not support stacking, blending, or concurrent named adapters in v1.
- [ ] Revise `grim_model_manifest.fbs` so `lora_path` is legal for the
      user-selected model role; split its policy from `hard_copy_path`.
- [ ] Persist and validate the explicit user-selected LoRA target instead of
      deriving adapter eligibility from `ModelRole`.
- [ ] Save adapters independently from base weights with target names, shapes,
      rank, scaling, and base-model compatibility metadata.
- [ ] Publish adapter revisions with create-new, append-only semantics; reject an
      existing `(adapter_id, adapter_revision)` instead of overwriting it.
- [ ] Keep deployable adapter artifacts free of optimizer, scheduler,
      accumulation, gradient, RNG, and training-loop state.
- [ ] Reject missing, duplicate, unknown, dimensionally incompatible, or
      integrity-invalid adapter entries, reject loading a second active set, and
      reject any non-exact base-checkpoint SHA-256 match.
- [ ] Validate exact layer coverage per class: enabled classes contain one target
      for every layer and disabled classes contain none.
- [ ] Load detached adapter views for inference without mutating base weights.
- [ ] Compute base and scaled adapter contributions dynamically during inference
      without materializing a durable merged weight.

### Phase 5: Validation

- [ ] Unit-test target selection so every excluded class remains excluded.
- [ ] Compare LoRA projection forward and backward against a CPU/reference
      implementation for every stored matrix orientation.
- [ ] Validate all five exact base/`A`/`B` shape equations and transpose flags,
      including square `W_o` where shape alone cannot detect wrong orientation.
- [ ] Run finite-difference checks for `A` and `B` on small tensors.
- [ ] Compare analytical `dA`, `dB`, `dX_base`, and `dX_adapter` against a
      reference for both physical orientations and verify `dW_base` is absent.
- [ ] Verify the same seed and adapter identity reproduce identical `A` values
      and that every newly created adapter has an exactly zero delta.
- [ ] Test microbatch accumulation and shared/tied-gradient behavior.
- [ ] Enumerate every pre-existing parameter after LoRA startup and verify it is
      non-trainable, has no gradient buffer, and is absent from optimizer groups;
      verify every enabled `A`/`B` has the opposite three properties.
- [ ] Test disabled, partially targeted, malformed, and rank-boundary configs.
- [ ] Reject per-layer targeting config and artifacts with missing, duplicate,
      extra, or partially populated class layer records.
- [ ] Test adapter checkpoint round trips and wrong-base rejection.
- [ ] Test immutable revision publication, duplicate identity rejection, and
      continual adaptation from revision `N` into a new revision greater than
      `N` without modifying the source artifact.
- [ ] Test exact mid-window resume restores adapter gradients, accumulation
      cursor, FP32 moments, optimizer/scheduler steps, data order, and RNG state.
- [ ] Reject resume from a deployable adapter artifact and reject incomplete
      training checkpoints without resetting or reconstructing missing state.
- [ ] Test missing/unsupported schema versions, adapter ID/revision failures,
      malformed target identities, wrong layers/classes, base-shape and
      orientation mismatches, rank/alpha mismatches, `A`/`B` shape mismatches,
      unsupported storage dtypes, and architecture metadata drift.
- [ ] Verify training, inference, save/load, and adapter unload leave every base
      tensor byte-for-byte unchanged.
- [ ] Measure memory, training throughput, and inference latency by target class.

## Open decisions

No unresolved design decisions are currently recorded. Add new questions here
as the implementation plan is refined.

## Change log

- 2026-09-04: Added the strict host-side resumable LoRA checkpoint schema and
      load I/O. Bare `.grimlorackpt` names resolve under the selected
      `model.grimcfg` directory's `lora_checkpoints` child, and loading validates
      exact base/config SHA-256 identities, architecture, complete per-class
      layer coverage, orientations, shapes, ranks, alpha, FP32 tensor checksums,
      optimizer/scheduler state, accumulation gradients, data order, and RNG
      state before returning a host snapshot. No startup caller or GPU mutation
      is wired yet.
- 2026-09-04: Defined normative backward equations for direct FFN and transposed
      attention orientations. Changed `LoRAProjectionView::A/B` to non-const borrowed
      `Tensor*` so leaf gradients can accumulate legally without `const_cast`; only
      gradient buffers are mutable during backward and `dW_base` is always absent.
- 2026-09-04: Strengthened LoRA training freeze policy from adapted matrices to
      the entire pre-existing base model. Enabled `A`/`B` tensors are the only
      trainable and optimizer-visible parameters; disabled and excluded base
      parameters remain frozen and own no gradient buffers.
- 2026-09-04: Split deployable LoRA adapter artifacts from resumable LoRA
      training checkpoints. Training checkpoints own complete durable optimizer,
      scheduler, accumulation, data-cursor, and RNG state; inference artifacts own
      only immutable adapter tensors and compatibility metadata.
- 2026-09-04: Defined `adapter_id` as a stable logical adapter lineage and
      `adapter_revision` as one immutable trained checkpoint. Published revisions
      are append-only; continual adaptation reads an old revision and emits a new,
      greater revision without overwriting the source.
- 2026-09-04: Defined the actual `lora_linear` API, explicit direct/transposed
      orientation, nullable authored-disabled dispatch, factored forward equations,
      and fail-loud validation. Corrected `LoRAProjectionView` to borrow `Tensor`
      identities so training autograd can accumulate `A`/`B` gradients.
- 2026-09-04: Finalized v1 optimizer policy: reuse the configured optimizer
      family, use one adapter-set-wide `learning_rate_lora`, apply zero weight decay
      to both `A` and `B`, and keep optimizer state FP32.
- 2026-09-04: Made v1 matrix-class enablement model-wide across all transformer
      layers. Per-layer lists, masks, ranges, overrides, and partial artifact target
      inventories are unsupported and fail loudly.
- 2026-09-04: Defined the complete per-class authored config surface: five
      `lora_*_enabled` flags plus rank, alpha, and precision for `W_qkv`, `W_o`,
      `W_gate`, `W1`, and `W2`. Enabled classes allocate one pair per layer;
      disabled classes allocate none.
- 2026-09-04: Resolved the type boundary as two types: registry-owned Category 2
      `LoRAParameterPair` and borrowed per-call `LoRAProjectionView`. Neither owns
      the base tensor or Category 1 activations/GradFns; no `TensorLoRA` type exists.
- 2026-09-04: Finalized all five physical adapter orientations. `W_qkv` and
      `W_o` use `(x A^T) B^T`; `W_gate`, `W1`, and `W2` use `(x B) A`, with
      class-specific shapes matching their existing GRIM base matrices.
- 2026-09-04: Finalized adapter artifact compatibility metadata: schema version,
  adapter ID/revision, exact base-checkpoint SHA-256, complete per-target
  identity/geometry/scaling/dtype facts, and architecture diagnostics metadata.
  Wrong base hashes and any incomplete or inconsistent metadata hard-fail before
  adapter activation.
- 2026-09-04: Set v1 to never merge adapters. Base weights remain immutable,
  adapter factors remain separately owned, and inference computes base plus
  scaled adapter contributions dynamically. `W_base += delta_W` is forbidden.
- 2026-09-04: Excluded the legacy model-config handoff from the LoRA design.
      New LoRA fields belong only on `TrainingHyperparameters` and flow to consumers
      through typed read views in `HyperparameterGroupings.hpp`.
- 2026-09-04: Set alpha per matrix class with initial `alpha = rank`, making
      v1 scaling `alpha / rank = 1.0`. Set v1 adapter storage, compute, and optimizer
      state to FP32 while retaining five authored LoRA matrix-class precision fields
      through the existing `ParameterGroupPrecision` config/registration path.
- 2026-09-04: Defined future LM-head residual MLP LoRA as three independently
      selectable projections: `mlp_W_gate`, `mlp_W_up`, and `mlp_W_down`. They
      remain deferred from v1.
- 2026-09-04: Made the LoRA target a user-facing model selection. The router is
      the intended choice, but `lora_path` must be legal for whichever model the
      user explicitly selects; the manifest must split LoRA policy from the
      unchanged `hard_copy_path` policy.
- 2026-09-04: Required seeded, deterministic Xavier-uniform initialization for
      `lora_A` and exact-zero initialization for `lora_B`, making every adapter an
      exact no-op at creation. Seed derivation must use stable adapter identity.
- 2026-09-04: Finalized factor initialization: `lora_A` uses gain-1 Xavier
      uniform based on its own logical shape, and `lora_B` uses exact zeros.
- 2026-09-04: Limited each LoRA-enabled model instance to exactly one active
      adapter set. V1 has no adapter stacking, blending, composition, or concurrent
      named-adapter loading.
- 2026-09-04: Set v1 initialization to random `A` and exact-zero `B`, producing
      an exact-zero initial adapter delta. Set adapter dropout to `0.0` with no v1
      dropout mask or operation.
- 2026-09-04: Set LoRA rank per matrix class (`W_qkv`, `W_o`, `W_gate`, `W1`,
  and `W2`), shared across layers of each class with no global or per-layer rank
  override.
- 2026-09-04: Set the normal training contract to a frozen pre-existing base
      model with `delta_W = (alpha / r) * B A`; only enabled adapter `A` and `B` are
      trainable and optimizer-visible, while activation gradients still propagate
      through frozen base operations.
- 2026-09-04: Defined five distinct LoRA deltas per transformer layer, forbade
      cross-layer and cross-projection sharing, and selected one fused delta for
      each fused `W_qkv` matrix.
- 2026-09-04: Created the plan. Recorded the initial five encoder matrix target
  classes, exclusions, priorities, ownership constraints, and unresolved
      adapter type boundary.