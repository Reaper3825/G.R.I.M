# NumberEncoder Atom Reconstruction Refactor Plan

> **Status:** Planned  
> **Implementation order:** Config plumbing -> ParameterGroupRegistration and ParameterRegistry -> Forward pass logic -> Loss logic -> Backward logic -> Serialization logic  
> **Compatibility:** Breaking model/checkpoint change. Do not silently load a checkpoint that lacks the enabled reconstruction parameters.  
> **Corpus compatibility:** No GRMT format change is required. Reconstruction targets are derived from each sequence's persisted `AtomTable` and uploaded as batch-owned tensors.

## Objective

Restore the NumberEncoder's two original jobs:

1. reduce numeric token/vocabulary pressure by representing a complete numeric atom with `<INT>` or `<FLOAT>` plus atom metadata;
2. teach the model numeric composition, value, and magnitude rather than treating an `atom_entry_id`, digit, or `pow10` bucket as an arbitrary label.

The refactor adds explicit atom reconstruction supervision. A compressed numeric atom representation must reconstruct:

- ordered mantissa digits;
- the base-10 value of every digit;
- the `pow10` attached to every real mantissa digit;
- real-slot count/mask;
- sign;
- numeric type;
- global mantissa/exponent or equivalent magnitude facts;
- surface facts needed to distinguish related values with different written forms.

The model-facing reconstruction result must remain independent of the final AtomTable materialization path so a later implementation can replace factorized reconstruction with another backend without changing the NumberEncoder, transformer, or loss ownership boundaries.

## Current-state diagnosis

The active NumberEncoder currently computes:

```text
slot_i =
    digit_emb[digit_i]
  + pow10_emb[pow10_i]
  + contribution_mlp(slot_features_i)

number_embedding =
    mean(real slot_i)
  + global_mlp(global_features)

x_t =
    embedding[<INT>/<FLOAT>]
  + number_embedding
```

This is a useful numeric feature injector, but it is not an autoencoder and is not required by any loss to preserve the supplied `AtomNumber`.

The current selector loss only asks a hidden state to select the correct candidate entry. That can teach entry discrimination without teaching digit reconstruction, place-value composition, or magnitude.

Three structural problems must be corrected:

1. **Mean pooling is not numeric composition.** Numeric value follows `sign * sum(digit_i * 10^pow10_i)`. Averaging makes the representation depend on digit count and lets zero-valued written digits change the meaning vector.
2. **Meaning and surface identity are mixed.** Current global features include value facts and written-form facts in one vector. `42`, `042`, `42.0`, and `4.2e1` should have related numeric meaning while retaining distinct surface identity.
3. **No reconstruction objective exists.** Downstream LM and selector gradients do not prove that `digit_emb` and `pow10_emb` can recover the atom metadata supplied to the input.

There is also a current syntax defect to remove before implementation:

```cpp
if (num_pool_atoms <= 0) {+
```

in `Shared/TensorContract/GradFns/NumberEncoderGradFn.cu`.

## Locked architectural decisions

### Keep the placeholder embedding

`<INT>` and `<FLOAT>` remain learned token embeddings. They represent structural atom type and, when embeddings are tied, remain part of the LM output classifier.

Do not add an extra mean-of-digit-embeddings vector to the placeholder rows.

Use a controlled fusion:

```text
x_t = placeholder_embedding_t
    + sigmoid(fusion_logit) * project(atom_latent_t)
```

The fusion gate is initialized so a newly initialized NumberEncoder cannot dominate the placeholder embedding at startup.

### Separate numeric meaning from surface identity

The encoder produces:

```cpp
struct NumberEncoderForwardResult {
    Tensor meaning_latents;       // [A, d_model]
    Tensor surface_latents;       // [A, d_model]
    Tensor atom_latents;          // [A, d_model], fused reconstruction source
    Tensor scattered_embeddings; // [total_tokens, d_model], zero at non-atoms
};
```

Where `A` is the number of authored numeric atoms.

The meaning branch owns:

- sign;
- nonzero digit/place contributions;
- canonical magnitude;
- normalized mantissa/exponent facts.

The surface branch owns:

- written mantissa order;
- leading zeros;
- explicit sign presence;
- decimal-point presence and position;
- explicit exponent presence;
- exponent spelling facts;
- `<INT>` versus `<FLOAT>` form distinctions not already carried by the placeholder.

### Make zero contribution exact

The meaning contribution of a zero digit must be exactly zero:

```text
meaning_contribution(0, pow10) = 0
```

Zero digits remain available to the surface branch and reconstruction head. They are not discarded from the atom.

### Replace mean composition in the meaning branch

The meaning branch uses a masked sum over nonzero digit contributions, followed by controlled projection/normalization and an explicit magnitude component:

```text
meaning =
    normalize(sum_i meaning_contribution(digit_i, pow10_i))
  + magnitude_projection(canonical_global_features)
```

It must not divide by written digit count. Leading zero formatting must not change numeric meaning.

### Reuse the digit and pow10 codebooks in the decoder

The first reconstruction backend is factorized. Decoder slot states are classified against the NumberEncoder's existing:

- `digit_emb [10, d_model]`;
- `pow10_emb [P, d_model]`.

Conceptually:

```text
digit_logits[n,s,d] = dot(decoded_slot[n,s], digit_emb[d])
pow10_logits[n,s,p] = dot(decoded_slot[n,s], pow10_emb[p])
```

This ties the encoding and reconstruction bases without duplicating digit or place classifiers.

Ordinary byte-token embeddings for ASCII `0` through `9` are not averaged into atom placeholders. If alignment with those base byte tokens is added later, it must use a learned projection because byte and atom token rows occupy different hard-gated embedding subspaces.

### Train two causal reconstruction paths

The shared factorized decoder is applied to two sources:

1. **Current-atom self reconstruction**

   ```text
   AtomNumber_t -> NumberEncoder -> atom_latent_t
   atom_latent_t -> reconstruct AtomNumber_t
   ```

   This directly teaches the NumberEncoder to preserve atom structure.

2. **Next-atom contextual reconstruction**

   ```text
   transformer_hidden_t -> reconstruct AtomNumber_(t+1)
   ```

   This teaches the language model to produce an atom entry after the LM head predicts `<INT>` or `<FLOAT>`.

Target metadata for `t+1` is supervision only. It must never enter the input or hidden-state computation at `t`.

### Keep reconstruction and AtomTable materialization separate

The model produces backend-neutral reconstruction facts:

```cpp
struct AtomReconstruction {
    Tokenizer::AtomType type;
    uint8_t base;
    std::vector<uint8_t> mantissa_digits;
    std::vector<int16_t> pow10;
    ReconstructedSurfaceForm surface;
    ReconstructedMagnitude magnitude;
    float confidence;
};
```

The inference boundary consumes those facts through:

```cpp
class IAtomReconstructionBackend {
public:
    virtual ~IAtomReconstructionBackend() = default;
    virtual AtomReconstruction reconstruct(
        const AtomReconstructionContext& context) = 0;
};
```

An `AtomTableMaterializer` validates and registers the result. The model-side decoder must not mutate an `AtomTable`.

Candidate selection, factorized generation, and terminal execution results can later be separate implementations of this interface.

## Shared tensor and target contract

Let:

- `A` = current authored numeric atom count;
- `N` = supervised next-atom target count;
- `S` = `number_encoder_max_digit_slots`;
- `P` = `2 * number_encoder_max_abs_pow10 + 1`;
- `M` = `d_model`;
- `H` = reconstruction hidden width;
- `G` = fixed global target width.

BatchPayload authors two target views from existing AtomTables:

```text
Current/self targets, aligned with atom_positions:
  recon_self_digit_values       [A, S]
  recon_self_pow10_index        [A, S]
  recon_self_slot_mask          [A, S]
  recon_self_global_targets     [A, G]
  recon_self_atom_types         [A]

Next/context targets, compacted to supervised source positions:
  recon_next_source_positions   [N]
  recon_next_digit_values       [N, S]
  recon_next_pow10_index        [N, S]
  recon_next_slot_mask          [N, S]
  recon_next_global_targets     [N, G]
  recon_next_atom_types         [N]
```

Targets are derived from the already persisted `AtomNumber` and raw-text spans. Padding slots use:

```text
slot_mask = 0
digit target ignored
pow10 target ignored
```

The loss must never interpret padding as digit zero or as a negative slot example.

The V1 global target contract contains:

```text
sign class
atom type class
digit count class
integer digit count class
has explicit sign
has decimal point
has explicit exponent
explicit exponent sign class
canonical exponent class/bucket
normalized mantissa regression target
```

`base` is authored and validated as `10` in V1. Do not spend model capacity predicting a constant. The backend-neutral result still carries the base so another backend or future tokenizer can extend the contract.

Exact exponent marker case and exponent leading-zero spelling may be added as surface fields. Their absence must not be confused with canonical numeric reconstruction.

---

# Phase 1 - Config plumbing

## Goal

Establish one authoritative configuration path before allocating parameters or changing runtime behavior.

The ownership chain remains:

```text
ai_config.json
  -> control/ai_config_paths.hpp snapshot
  -> Shared/HyperParameters/HyperParameters_GPU.hpp
  -> Shared/HyperParameters/HyperparameterGroupings.hpp
  -> Phase 1 startup assignment
```

No decoder, loss helper, inference backend, or serialization function may parse raw JSON directly.

## Config leaves

Add flat leaves under `training.config`:

```json
{
  "number_reconstruction_enabled": true,
  "number_reconstruction_backend": "factorized_v1",
  "number_reconstruction_d_hidden": 64,
  "number_reconstruction_bias_enabled": true,
  "number_reconstruction_self_alpha": 0.25,
  "number_reconstruction_next_alpha": 0.25,
  "number_reconstruction_digit_alpha": 1.0,
  "number_reconstruction_pow10_alpha": 1.0,
  "number_reconstruction_slot_alpha": 0.5,
  "number_reconstruction_surface_alpha": 0.25,
  "number_reconstruction_magnitude_alpha": 0.5,
  "number_reconstruction_warmup_steps": 0,
  "number_encoder_fusion_gate_init": -2.0,
  "parameter_precision_number_reconstruction": "fp32"
}
```

Do not add a duplicate configured `pow10_buckets`; derive it from `number_encoder_max_abs_pow10`.

## Typed views

Add:

```cpp
enum class NumberReconstructionBackend {
    FACTORIZED_V1
};

struct NumberReconstructionConstructionHP {
    bool enabled;
    NumberReconstructionBackend backend;
    int d_model;
    int d_hidden;
    int max_digit_slots;
    int max_abs_pow10;
    int pow10_buckets;
    bool bias_enabled;
    float fusion_gate_init;
};

struct NumberReconstructionLossHP {
    bool enabled;
    float self_alpha;
    float next_alpha;
    float digit_alpha;
    float pow10_alpha;
    float slot_alpha;
    float surface_alpha;
    float magnitude_alpha;
    int warmup_steps;
};
```

Expose these through:

```cpp
numberReconstructionConstructionHP(snapshot)
numberReconstructionLossHP(snapshot)
```

Construction and loss views are separate so forward allocation does not own loss policy.

## Validation

Fail startup when:

- reconstruction is enabled but `number_encoder_enabled=false`;
- reconstruction is enabled but `use_atom_data=false`;
- backend text is unknown;
- `d_hidden <= 0`;
- any loss coefficient is negative or non-finite;
- all reconstruction top-level coefficients are zero while reconstruction is enabled;
- `number_encoder_max_digit_slots <= 0`;
- `number_encoder_max_abs_pow10 <= 0`;
- the precision leaf is missing or invalid;
- factorized reconstruction is enabled with a non-base-10 AtomTable contract.

Permit:

- self reconstruction enabled while next reconstruction alpha is zero;
- next reconstruction enabled while self alpha is zero;
- surface alpha zero for a meaning-only ablation.

## Files

- `ai_config.json`
- `Shared/HyperParameters/HyperParameters_GPU.hpp`
- `Shared/HyperParameters/HyperparameterGroupings.hpp`
- `training/Phases/ConfigDump.cu`
- `GRIM/Docs/Config.md`

## Tests

- valid factorized configuration produces both grouped views;
- every missing required leaf fails at the HyperParameters boundary;
- unknown backend fails before model allocation;
- derived `pow10_buckets` equals `2 * max_abs_pow10 + 1`;
- disabled reconstruction allocates and executes nothing;
- config dump contains every authored reconstruction leaf.

## Exit criteria

- All new runtime code receives typed immutable views.
- Raw JSON is read only at the existing config boundary.
- Disabled configuration is a true no-op.

---

# Phase 2 - ParameterGroupRegistration and ParameterRegistry

## Goal

Create durable startup-owned parameter tensors and register them exactly once with optimizer, gradient, statistics, precision, and checkpoint ownership.

## Parameter group

Add:

```cpp
ParamGroupType::NUMBER_RECONSTRUCTION
```

and increment `ParamGroupType::COUNT`.

Use a distinct group instead of placing decoder tensors in `NUMBER_ENCODER` because:

- the reconstruction implementation is replaceable;
- decoder learning rates and diagnostics may differ from input encoding;
- disabling or replacing a decoder must not alter NumberEncoder parameter ownership;
- shared `digit_emb` and `pow10_emb` still receive gradients through their existing NumberEncoder group.

Map the new group through:

- parameter group names;
- precision lookup;
- optimizer trace labels;
- gradient norm aggregation;
- model statistics accounting;
- diagnostic switch statements.

## Registry owner

Add:

```cpp
struct NumberReconstructionParameterTensors {
    Tensor W_source;       // [d_model, d_hidden]
    Tensor b_source;       // [1, d_hidden], optional
    Tensor slot_queries;   // [S, d_hidden]
    Tensor W_slot;         // [d_hidden, d_model]
    Tensor b_slot;         // [1, d_model], optional
    Tensor W_slot_present; // [d_model, 1]
    Tensor b_slot_present; // [1, 1], optional
    Tensor W_global;       // [d_hidden, G]
    Tensor b_global;       // [1, G], optional
};
```

The decoder does not own digit or pow10 output matrices. It borrows:

```text
NumberEncoderParameterTensors::digit_emb
NumberEncoderParameterTensors::pow10_emb
```

as tied reconstruction codebooks.

Add:

```cpp
std::unique_ptr<NumberReconstructionParameterTensors>
    number_reconstruction_parameters;

getNumberReconstructionParameters()
requireNumberReconstructionParameters(caller)
```

## NumberEncoder parameter changes

Extend `NumberEncoderParameterTensors` for meaning/surface separation and controlled fusion:

```cpp
Tensor surface_position_emb; // [S, d_model]
Tensor W_s1;                 // [surface_feature_dim, d_hidden]
Tensor b_s1;                 // optional
Tensor W_s2;                 // [d_hidden, d_model]
Tensor fusion_logit;         // [1, d_model] or [1, 1], choose once and serialize exactly
```

Keep existing digit and pow10 codebooks. Repurpose the existing contribution MLP as the meaning contribution encoder and the existing global MLP as the canonical magnitude encoder.

The final choice between scalar and per-channel fusion gate must be made here and then treated as a checkpoint shape invariant. A per-channel gate is preferred because numeric information may need different strength across model dimensions.

## Initialization

Add:

```cpp
initializeNumberReconstructionParameterTensors(
    StartupParameterRegistry&,
    const NumberReconstructionConstructionHP&,
    uint64_t seed,
    cudaStream_t);
```

Initialization rules:

- Xavier initialize projection matrices;
- zero initialize optional biases;
- initialize `slot_queries` with Xavier or a small normal distribution;
- initialize `fusion_logit` from `number_encoder_fusion_gate_init`;
- allocate nothing when disabled;
- reject double initialization;
- synchronize and fail loudly on CUDA error before publishing the owner.

Use deterministic, non-overlapping seed offsets and document them.

## Registration inventory

Add a constexpr tensor inventory:

```cpp
kNumberReconstructionTensorParameters
```

All durable tensors must appear in exactly one inventory. Optional biases use `addConfigGatedTensor`.

Extend `kNumberEncoderTensorParameters` with the new surface and fusion tensors.

Registration must prove:

- enabled owner exists and every required tensor is allocated;
- disabled owner does not exist;
- no parameter pointer appears in two optimizer groups;
- the shared digit/pow10 codebooks remain registered only as NumberEncoder parameters;
- tied codebook use creates gradient accumulation, not duplicate optimizer ownership.

## Transitional serialization guard

Because serialization is deliberately Phase 6, add a temporary fail-loud guard:

```text
number_reconstruction_enabled=true
and checkpoint serialization support not compiled/available
-> reject save and resume
```

Do not allow partial checkpoints during Phases 2-5.

## Files

- `Shared/TensorContract/TensorContract_GPU.hpp`
- `training/Phases/Startup/Model/ParameterRegistry.hpp`
- `training/Phases/Startup/Model/ParameterGroupRegistration.hpp`
- `training/Phases/Startup/Model/ParameterGroupRegistration.cu`
- `training/Phases/Startup/Model/ModelGpuAssembly.cu`
- gradient/statistics/optimizer switch owners that enumerate `ParamGroupType`

## Tests

- enabled allocation creates all expected shapes;
- disabled allocation leaves a null owner;
- double initialization throws;
- missing optional bias is accepted only when its config gate is false;
- parameter inventory count equals registered count;
- digit/pow10 codebook pointers have one optimizer owner;
- reconstruction parameters obey configured precision;
- fusion gate initializes to the configured value.

## Exit criteria

- Startup is the only parameter allocation boundary.
- Every new parameter is registered once.
- No forward or loss code owns durable weights.

---

# Phase 3 - Forward pass logic

## Goal

Produce reconstructable atom latents, controlled placeholder fusion, and factorized reconstruction logits for both causal training paths.

## Batch target materialization

Extend `BatchPayload` and `BatchDeviceBindings` with the shared target contract defined above.

Target authoring occurs after:

- token IDs and atom masks are finalized;
- `atom_entry_ids` are available;
- per-row AtomTables are attached;
- current atom positions are materialized.

Build current/self targets directly from `atom_positions`.

Build next/context targets by scanning valid positions `t` where `t+1`:

- remains in the same batch row;
- is a real, non-padding token;
- is a numeric atom placeholder;
- has a valid AtomTable entry;
- has valid `AtomNumber` metadata.

Store `t` in `recon_next_source_positions`. Never store or gather `t+1` hidden states as the prediction source.

Upload targets through existing batch storage/upload ownership:

```text
BatchPayload
  -> BatchDeviceStorage
  -> uploadBatchToDevice
  -> BatchDeviceBindings
```

No forward kernel may dereference host AtomTables.

## Refactor NumberEncoder result ownership

Change `number_encode(...)` to return a structured result rather than only one scattered tensor.

Preferred graph:

```text
immutable batch channels
  -> dense meaning/surface/atom latents [A, M]
  -> differentiable scatter by atom_positions
  -> scattered number embeddings [T, M]
  -> gated residual fusion with placeholder embeddings
```

The dense atom latent is the single reconstruction source. The scattered tensor is derived from it, preventing two independent encodes of the same current atom.

## Meaning branch

For each real digit slot:

```text
raw contribution =
    digit_emb[digit]
  + pow10_emb[pow10]
  + contribution_mlp(meaning_slot_features)

meaning contribution =
    (digit != 0) * raw contribution
```

Then:

```text
meaning_latent =
    normalize(sum(real meaning contributions))
  + global_magnitude_mlp(canonical_global_features)
```

The zero gate is an exact authored mask or exact digit comparison, not a learned approximation.

## Surface branch

For each real written slot:

```text
surface slot =
    digit_emb[digit]
  + surface_position_emb[index_left]
  + surface_mlp(surface_slot_features)
```

Pool surface slots with a mask-aware method that preserves written positions through `surface_position_emb`. Add global surface flags after pooling.

The surface branch includes zero digits. Therefore `042` remains distinguishable from `42`.

## Atom latent and placeholder fusion

Fuse the two branches:

```text
atom_latent = project_or_normalize(meaning_latent + surface_latent)
number_delta = sigmoid(fusion_logit) * atom_latent
x_t = token_embedding_t + scatter(number_delta)
```

Do not mutate embedding table rows. Fusion applies to the per-token activation.

All non-atom rows in the scattered tensor remain exactly zero.

## Factorized decoder

Add a module such as:

```text
Layers/NumberReconstruction/NumberReconstruction_GPU.hpp
Layers/NumberReconstruction/NumberReconstruction_GPU.cu
```

Forward contract:

```cpp
struct NumberReconstructionLogits {
    Tensor digit_logits;       // [K, S, 10]
    Tensor pow10_logits;       // [K, S, P]
    Tensor slot_logits;        // [K, S]
    Tensor global_outputs;     // [K, G]
};

NumberReconstructionLogits reconstructNumbers(
    const Tensor& sources,     // [K, d_model]
    ...);
```

Decoder math:

```text
source_hidden = tanh(source * W_source + b_source)

decoded_slot[k,s] =
    project(tanh(source_hidden[k] + slot_queries[s]))

digit_logits[k,s,d] =
    dot(decoded_slot[k,s], digit_emb[d])

pow10_logits[k,s,p] =
    dot(decoded_slot[k,s], pow10_emb[p])

slot_logits[k,s] =
    decoded_slot[k,s] * W_slot_present + b_slot_present

global_outputs[k] =
    source_hidden[k] * W_global + b_global
```

Use standard autograd tensor operations where practical. Add custom fused kernels only when required by shape/layout or measured performance.

## Self reconstruction forward

When training and self alpha can become positive:

```text
self_logits = reconstructNumbers(number_encoder.atom_latents)
```

Store the logits on `ModelForwardOutputs`.

## Next reconstruction forward

Gather transformer/LM-input hidden rows at `recon_next_source_positions`:

```text
next_sources = gather(encoder_output_or_lm_head_input, source_positions)
next_logits = reconstructNumbers(next_sources)
```

Lock the source boundary before implementation. Preferred source is the final normalized representation immediately before LM vocabulary projection, because it is the state responsible for predicting the next placeholder. Do not use post-logit values.

Use the same decoder parameters for self and next reconstruction.

## Forward graph policy

Extend the forward request/graph policy with explicit booleans:

```text
emit_number_reconstruction_self
emit_number_reconstruction_next
```

Training enables them from typed configuration and active loss schedule. Inference enables next reconstruction only when the selected backend needs factorized logits.

## ModelForwardOutputs

Retain:

```cpp
NumberEncoderForwardResult number_encoder;
NumberReconstructionLogits number_reconstruction_self;
NumberReconstructionLogits number_reconstruction_next;
Tensor number_reconstruction_next_sources;
```

`ModelForwardOutputs::clear()` remains the sole forward-state teardown owner.

## Interchangeable inference adapter

Introduce the backend-neutral types and factory, but keep V1 backend selection static from Phase 1 config.

Refactor the numeric-placeholder commit boundary in `Phase2_InferenceLoop.cu`:

```text
sample <INT>/<FLOAT>
  -> reconstruction backend
  -> AtomReconstruction
  -> validate sampled placeholder type
  -> AtomTableMaterializer
  -> commit token + atom_entry_id
```

Existing-entry selection and terminal execution results must adapt to this boundary rather than bypassing it.

## Files

- `Shared/Batching/BatchPayload.hpp/.cu`
- `Shared/Batching/BatchDeviceStorage.hpp`
- `Shared/Batching/BatchDeviceBindings.hpp`
- `Shared/Batching/BatchDeviceUpload.cu`
- `Shared/TensorContract/GradFns/NumberEncoderGradFn.hpp/.cu`
- new NumberReconstruction layer files
- `Shared/Forward/ModelForward_GPU.hpp/.cu`
- `Shared/Forward/ModelForwardOutputs.hpp`
- `training/Phases/Phase2_InferenceLoop.cu`
- training CMake target lists

## Tests

- `42` produces digit targets `[4,2]` and pow10 `[1,0]`;
- `042` shares meaning behavior with `42` but differs in surface latent/targets;
- `42.0` and `4.2e1` have related value targets and different surface targets;
- zero digit has exact zero meaning contribution;
- non-atom rows receive exact zero NumberEncoder delta;
- self target order follows `index_left`;
- next source position is `t`, never `t+1`;
- batch row boundaries cannot create a next-atom target;
- padded slots produce no decoder source and no valid target;
- factorized decoder uses the same digit/pow10 parameter pointers as NumberEncoder;
- disabling reconstruction emits no reconstruction tensors;
- inference backend type must agree with sampled `<INT>`/`<FLOAT>`.

## Exit criteria

- Current atoms produce dense reconstructable latents.
- The transformer still consumes learned placeholder embeddings plus a gated numeric delta.
- Both reconstruction paths exist without target leakage.
- AtomTable mutation remains outside model forward.

---

# Phase 4 - Loss logic

## Goal

Make reconstruction, place value, magnitude, and surface identity explicit supervised objectives and add them to the single canonical loss tensor.

## Loss components

### Slot-presence loss

Binary cross entropy over all `S` slots:

```text
L_slot = BCE(slot_logits, slot_mask)
```

This is the only component where padded slots participate as negative examples.

### Digit reconstruction loss

Masked cross entropy:

```text
L_digit =
    sum(slot_mask * CE(digit_logits, digit_target))
    / max(1, real_slot_count)
```

Padding contributes zero.

### Pow10 reconstruction loss

Masked cross entropy:

```text
L_pow10 =
    sum(slot_mask * CE(pow10_logits, pow10_target))
    / max(1, real_slot_count)
```

Padding contributes zero.

### Surface loss

Use categorical/BCE terms for:

- explicit sign presence and sign class;
- atom type;
- digit count;
- integer digit count/decimal position;
- decimal-point presence;
- explicit exponent presence;
- exponent sign class;
- any enabled exact-spelling fields.

Normalize by reconstructed atom count, not total token count.

### Magnitude loss

V1 uses stable factored targets:

- canonical exponent classification;
- normalized mantissa Huber loss;
- sign classification where not already included.

Do not regress an unbounded raw numeric value.

Optional differentiable consistency:

```text
predicted digit/pow10 distributions
  -> expected canonical magnitude
  -> compare with global magnitude prediction
```

Do not add this until the base factorized losses are numerically stable.

## Path totals

For either path:

```text
L_recon =
    digit_alpha     * L_digit
  + pow10_alpha     * L_pow10
  + slot_alpha      * L_slot
  + surface_alpha   * L_surface
  + magnitude_alpha * L_magnitude
```

Then:

```text
L_total =
    L_lm
  + scheduled_self_alpha * L_recon_self
  + scheduled_next_alpha * L_recon_next
  + existing_selector_loss
  + existing_execution_losses
```

Warmup applies only to the top-level self/next coefficients. Component ratios remain stable during warmup.

## Single loss ownership

Add reconstruction losses inside `computeAutogradLoss(...)` and combine them using existing scalar autograd operations:

```cpp
loss_state.loss_tensor =
    autograd::add(loss_state.loss_tensor, scaled_reconstruction, stream);
```

Do not create a second backward call or a second scalar root.

`AutogradLossState::loss_tensor` remains the only backward root.

## Loss result and telemetry

Add separately reported host scalars:

```text
number_reconstruction_self_loss
number_reconstruction_next_loss
number_reconstruction_digit_loss
number_reconstruction_pow10_loss
number_reconstruction_slot_loss
number_reconstruction_surface_loss
number_reconstruction_magnitude_loss
```

Do not hide reconstruction loss inside `execution_loss` or selector residual accounting.

Log:

- valid self atom count;
- valid next atom count;
- real digit-slot count;
- exact digit accuracy;
- exact pow10 accuracy;
- exact full-atom structural reconstruction rate;
- magnitude exponent accuracy;
- mantissa error;
- skipped/no-target reason.

## Fail-loud rules

Throw when:

- an enabled path has valid targets but corresponding forward logits are absent;
- logit shapes disagree with `A`, `N`, `S`, `P`, or `G`;
- a real slot target digit is outside `[0,9]`;
- a real pow10 target is outside `[0,P)`;
- valid-count metadata disagrees with masks;
- any component or scaled loss is non-finite;
- next targets exist without source positions;
- a next source position equals or follows its target position.

An empty target set is a logged no-op, not an error.

## Files

- new `Shared/Loss/ComputeLoss/NumberReconstructionLoss.hpp/.cu`
- `training/Autograd/AutogradTraining.hpp/.cu`
- `training/Autograd/AutogradIntermediates.hpp`
- telemetry and training-log schemas/owners as appropriate
- training CMake target lists

## Tests

- perfect logits produce near-zero digit/pow10/surface loss;
- padded slots do not change digit or pow10 loss;
- changing a padded digit target has no effect;
- slot-presence loss still supervises padded positions as absent;
- swapping `4@10^1` and `4@10^0` increases loss;
- same mantissa with different canonical exponent increases magnitude loss;
- self and next losses normalize independently;
- no-atom batches leave total loss unchanged;
- warmup coefficient matches authored schedule;
- total loss decomposition matches logged channel sum;
- target leakage proof verifies next reconstruction uses only state at `t`.

## Exit criteria

- NumberEncoder latents cannot minimize training loss without reconstructing digits and places.
- The next-token hidden state is explicitly trained to reconstruct the next atom.
- All reconstruction loss flows through the canonical autograd root.

---

# Phase 5 - Backward logic

## Goal

Deliver gradients from both reconstruction paths to decoder parameters, transformer states, NumberEncoder parameters, shared digit/pow10 codebooks, and placeholder fusion without introducing a parallel backward system.

## Preferred autograd composition

Build the decoder from existing differentiable primitives:

- matmul;
- broadcast add;
- tanh or another existing activation;
- reshape/view with preserved ownership;
- gather rows;
- dot/matmul against digit and pow10 codebooks.

This gives automatic gradients for:

- `W_source`, `b_source`;
- `slot_queries`;
- `W_slot`, `b_slot`;
- slot/global heads;
- input sources;
- shared digit/pow10 codebooks.

Only introduce custom GradFns for operations that do not already exist or cannot express the required masking/layout safely.

## Required custom gradient boundaries

### Differentiable atom scatter/gather

Add or reuse explicit operations:

```text
scatter_atom_rows([A,M], atom_positions) -> [T,M]
gather_rows([T,M], source_positions) -> [N,M]
```

Backward:

```text
scatter backward: gather grad[T,M] into grad[A,M]
gather backward: atomic scatter-add grad[N,M] into grad[T,M]
```

Validate every position in forward and backward.

### NumberEncoder dense backward

Refactor `NumberEncoderGradFn` so its primary gradient input is the dense atom latent `[A,M]`.

Gradients from:

- placeholder fusion/LM path;
- self reconstruction;
- selector candidate use where applicable;

must accumulate into the same NumberEncoder leaf buffers with `atomicAdd`.

The meaning branch backward must match masked-sum semantics:

- zero digit meaning slots receive no meaning-branch gradient;
- surface branch still sends gradients to zero digit codebook entries;
- there is no `1 / real_slot_count` factor in the meaning sum;
- normalization and magnitude projection use exact chain-rule derivatives.

### Reconstruction loss GradFn

Implement a `NumberReconstructionLossGradFn` or a small set of focused GradFns that writes gradients only to logits/global outputs.

It must:

- apply `slot_mask` to digit and pow10 gradients;
- normalize by real-slot or atom counts exactly as forward does;
- handle zero valid counts as no-op;
- apply upstream scalar gradient;
- never write parameter values;
- never mutate target tensors or AtomTables.

Standard decoder GradFns then propagate from logits to parameters and sources.

## Shared codebook accumulation

Digit and pow10 codebooks receive gradient contributions from:

```text
LM input NumberEncoder path
+ selector candidate-key path
+ self reconstruction classification
+ next reconstruction classification
```

All writes accumulate into the single registry-owned gradient buffers.

The optimizer steps those buffers once.

## Transformer and causality gradients

Next reconstruction gradient path:

```text
L_next
  -> next reconstruction logits
  -> shared decoder
  -> gathered hidden state at t
  -> transformer
```

There must be no gradient/input edge from target metadata at `t+1` into hidden state `t`. Targets are constant batch data.

Self reconstruction gradient path:

```text
L_self
  -> self reconstruction logits
  -> shared decoder
  -> dense atom latent_t
  -> NumberEncoder parameters
```

This path does not need to traverse the transformer.

## Fusion backward

For:

```text
number_delta = sigmoid(fusion_logit) * atom_latent
```

backward must produce:

```text
grad_atom_latent += grad_delta * sigmoid(fusion_logit)
grad_fusion_logit +=
    grad_delta * atom_latent * sigmoid(g) * (1 - sigmoid(g))
```

Use existing sigmoid and elementwise multiplication GradFns where possible.

## Tape lifecycle

Do not change:

- `AutogradLossState` as scalar-root owner;
- `Tensor::backward()` as the only traversal entry point;
- registry leaf buffers as gradient destinations;
- optimizer as the only parameter mutation boundary;
- `ModelForwardOutputs::clear()` as forward retained-state teardown.

## Gradient verification

Add finite-difference or small deterministic CUDA tests for:

- decoder source projection;
- slot query;
- digit codebook;
- pow10 codebook;
- slot-presence head;
- global magnitude head;
- gather/scatter;
- fusion gate;
- zero-digit semantic mask;
- combined LM plus self plus next accumulation.

Add an optimizer-step test proving shared codebook tensors are stepped exactly once.

## Files

- `Shared/TensorContract/GradFns/NumberEncoderGradFn.hpp/.cu`
- new reconstruction loss GradFn files
- new gather/scatter GradFn files if not already available
- `Shared/TensorContract/TensorContract_GPU.hpp`
- gradient accounting/diagnostic owners
- training CMake target lists

## Exit criteria

- Both reconstruction paths deliver finite gradients.
- Shared digit/pow10 codebooks accumulate all intended sources.
- Padding and zero semantic contributions obey exact gradient masks.
- One backward root and one optimizer update remain.

---

# Phase 6 - Serialization logic

## Goal

Persist and restore the complete NumberEncoder plus factorized reconstruction backend with exact shape/config agreement.

## Checkpoint version

Bump:

```cpp
GRIM_MODEL_VERSION: 15 -> 16
```

Document:

```text
v16: Added NumberEncoder meaning/surface/fusion weights and
     factorized NumberReconstruction decoder weights.
```

Because this is a breaking architecture change, an enabled reconstruction model must reject earlier checkpoints. Do not silently retain freshly initialized reconstruction weights on resume.

GRMT format remains unchanged because reconstruction targets are derived from the existing per-sequence AtomTable.

## FlatBuffer schema

Append the new NumberEncoder fields to `NumberEncoderWeights`:

```text
surface_position_emb_data
w_s1_data
b_s1_data
w_s2_data
fusion_logit_data
```

Add:

```fbs
table NumberReconstructionWeights {
  w_source_data: [float];
  b_source_data: [float];
  slot_queries_data: [float];
  w_slot_data: [float];
  b_slot_data: [float];
  w_slot_present_data: [float];
  b_slot_present_data: [float];
  w_global_data: [float];
  b_global_data: [float];
  backend: NumberReconstructionBackend;
}
```

Append to `TransformerModel`:

```fbs
number_reconstruction: NumberReconstructionWeights;
```

Do not repurpose the stale `NumericHeadWeights` table. Leave it tombstoned or remove it only as a separate explicit schema cleanup. Reusing it would blur a legacy scalar head with the new structured reconstruction contract.

Regenerate:

```text
training/schemas/grim_transformer_model_generated.h
```

using the repository's canonical FlatBuffers generation path.

## Serialization views and requests

Add read/write views:

```cpp
SerializationNumberReconstructionReadView
SerializationNumberReconstructionWriteView
```

Extend:

- serialization source views;
- save requests;
- load requests;
- capability requirements;
- load report;
- common registry-to-view binding.

Add the new NumberEncoder tensors to its existing views.

## Save

When reconstruction is enabled:

- require every non-gated tensor;
- require enabled optional biases;
- reject disabled optional bias tensors that unexpectedly own storage;
- write backend enum;
- write all tensor vectors;
- include the new table in checksum/verification coverage.

When disabled:

- omit the reconstruction table;
- require no registry owner;
- continue saving the NumberEncoder only if independently enabled.

## Load

When runtime config requires reconstruction:

- require model version 16;
- require `number_reconstruction` table;
- require backend equality with finalized config;
- require every tensor length to match the registry allocation exactly;
- require the extended NumberEncoder tensor lengths;
- upload all tensors;
- mark reconstruction loaded only after every upload succeeds.

When runtime config disables reconstruction:

- reject a live reconstruction registry owner;
- the presence of checkpoint reconstruction weights may be logged and ignored only if repository-wide checkpoint policy already permits disabled components; otherwise reject consistently with current component rules.

Do not partially load a backend.

## Shape validation

Validate exact element counts:

```text
W_source       = M * H
b_source       = H or 0 when disabled
slot_queries   = S * H
W_slot         = H * M
b_slot         = M or 0
W_slot_present = M
b_slot_present = 1 or 0
W_global       = H * G
b_global       = G or 0
```

Also validate all new NumberEncoder shapes and fusion gate width.

## Files

- `training/schemas/grim_transformer_model.fbs`
- regenerated `training/schemas/grim_transformer_model_generated.h`
- `Common/grim_model_serialization_version.hpp`
- `Common/grim_model_serialization.cu`
- `Layers/Serialization/Serialization_views.hpp`
- `Layers/Serialization/Serialization_requests.hpp`
- `Layers/Serialization/Serialization_save.cu`
- `Layers/Serialization/Serialization_load.cu`
- `Layers/Serialization/Serialization_validate.cu`

## Tests

- enabled save/load round trip preserves every tensor bit-for-bit;
- disabled reconstruction omits its table;
- enabled runtime rejects a v15 checkpoint;
- enabled runtime rejects a missing reconstruction table;
- backend mismatch fails;
- each truncated or oversized vector fails validation;
- optional bias presence follows config;
- new NumberEncoder surface/fusion tensors round trip;
- checksum verification includes new data;
- loaded model reproduces reconstruction logits from a fixed test batch.

## Exit criteria

- Saving and resuming an enabled model is complete and deterministic.
- No reconstruction parameter can remain freshly initialized after a claimed successful resume.
- Transitional Phase 2 serialization guards are removed.

---

# End-to-end acceptance criteria

The refactor is complete only when all of the following hold:

1. `<INT>` and `<FLOAT>` remain the only LM vocabulary entries required for complete numeric atoms.
2. Placeholder embeddings remain learned and are not overwritten with digit means.
3. The NumberEncoder's meaning branch uses additive digit/place contributions, not written-digit mean pooling.
4. A zero digit has exact zero numeric-meaning contribution.
5. Surface identity retains zero digits and written positions.
6. Current atom latents reconstruct ordered mantissa digits and `pow10` values.
7. Decoder digit and pow10 classifiers reuse the NumberEncoder codebooks.
8. The model reconstructs the next numeric atom from hidden state `t` without consuming metadata from `t+1`.
9. Padded slots contribute zero digit and pow10 loss.
10. `42`, `042`, `42.0`, and `4.2e1` can share related meaning while remaining distinguishable in surface reconstruction.
11. Magnitude supervision uses bounded/factored targets rather than raw unbounded values.
12. Reconstruction loss is added to the one canonical `AutogradLossState::loss_tensor`.
13. Backward writes gradients only; optimizer owns parameter mutation.
14. Shared digit/pow10 codebooks have one optimizer owner and accumulate every intended gradient source.
15. Model forward and loss code never mutate AtomTables.
16. Numeric placeholder inference passes through an interchangeable reconstruction backend and a separate AtomTable materializer.
17. Enabled checkpoints persist and restore every NumberEncoder and reconstruction tensor.
18. Old or incomplete checkpoints fail loudly.

# Recommended implementation checkpoints

After each phase, stop and prove its exit criteria before starting the next phase:

```text
P1 Config:
  finalized typed views and validation only

P2 Registry:
  deterministic allocation and exact registration only

P3 Forward:
  correct target materialization, latents, logits, and inference boundary

P4 Loss:
  numerically stable loss values and causal target accounting

P5 Backward:
  gradient correctness and single-step ownership

P6 Serialization:
  strict checkpoint round trip and version enforcement
```

Do not combine Phase 4 and Phase 5 into a custom monolithic loss/backward shortcut. Forward values, scalar loss construction, gradient propagation, and parameter updates must remain separate owners.
