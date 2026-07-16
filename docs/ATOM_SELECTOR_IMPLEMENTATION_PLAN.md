# Atom Selector Implementation and Decoupling Plan

> **Status:** In progress — W0 (boundary lock), W1 (rip-out), and W2 (numeric meaning encoding input path) are DONE; W3 (supervision heads) is pending; W4 has the inference placeholder/result bridge but still needs selector-to-execution argument routing.
>
> **W1 done:** the execution-entangled decode-time slot selector is deleted end-to-end (`Layers/DecodeTimeSlotSelector/`, `Shared/Execution/DecodeTimeNumPolicy.{hpp,cu}`, `DecodeTimeResolveResult.hpp`, `AutogradSelectorSupervisionLoss.{hpp,cu}`, registry owner, `selector_*` config leaves, `slot_selection_targets` GRMT channel → GRMT v13, checkpoint `slot_selector` table, `GenerationState::decode_selector`). Phase2 inference now dynamically enables a numeric placeholder only when a model-confirmed terminal execution result or a same-type numeric-meaning selector candidate can bind it.
>
> **W2 done (input path):** `NumberEncoderParameterTensors` (digit_emb, pow10_emb, contribution MLP, global mantissa/exponent MLP) registered as `ParamGroupType::NUMBER_ENCODER` through Phase-1 startup; `number_encoder_*` config leaves through the HyperParameters boundary with `numberEncoderConstructionHP()`; `BatchPayload` digit-place channels (current-token arg_number only, mask-padded, fail-loud) uploaded via `BatchDeviceBindings::d_atom_digit_*`; `autograd::number_encode()` + `NumberEncoderGradFn` fused into shared forward via `residual_add`. Checkpoint save/load fails loud while NumberEncoder weights have no FlatBuffer table (transitional guard).
>
> **Scope:** Numeric meaning representation, atom selector refactor, and selector/execution decoupling  
> **Non-goal:** Redesigning execution behavior itself  
> **Compatibility target:** DO NOT PRESERVE BACKWARDS COMPATIBILITY. This is a breaking refactor that will change the selector contract and execution behavior. The goal is to establish a clean, future-proof architecture for numeric meaning representation and selection, not to maintain compatibility with the current selector implementation. EXECUTION BEHAVIOR WILL CHANGE. The input PRIMARILY WILL NOT. The current selector logic will be ripped out and replaced with a new architecture that learns numeric meaning. The execution plumbing will remain materially the same, but it will consume the new selector output instead of the old logic. This is a clean-slate refactor, not a compatibility-preserving rewrite.

---

## Objective

This plan establishes the basis for replacing the current atom selection logic with a dedicated selector architecture that learns **numeric meaning** rather than arbitrary numeric labels.

The central requirement is:

- rip out the current atom selection logic,
- decouple selector logic from execution logic,
- keep the execution plumbing and call path materially the same,
- make execution a **consumer** of selector output when `execution_active` is true,
- keep execution architecture out of scope except where a stable selector-consumer boundary must be enforced.

In short: **selector owns selection; execution only uses the selection result.**

---

## Config boundaries and ownership

This plan must respect the existing GRIM-text config ownership chain exactly.

### Static assignment boundary

Static assignment must remain inside the **Phase 1 startup boundary**.

That means:

- Phase 1 startup is the only place where static startup config assignment is authored and stamped into long-lived runtime/model state
- this selector refactor must not move static assignment into Phase 2, decode, execution, or ad-hoc subsystem setup code
- no selector-side or execution-side code may become a second config assignment boundary

If a config value is needed later, it must arrive through the already-owned finalized config or immutable grouped payloads prepared from that boundary.

### Raw config entry point: `ai_config_paths.hpp`

`control/ai_config_paths.hpp` owns the **single raw config entry point**.

Its ownership is:

- `AiConfigSnapshot` is the raw JSON snapshot type
- `AiConfigSnapshot::document` is the authored JSON snapshot document
- `loadAiConfigSnapshot()` is the single raw load entry point

This file is the raw snapshot layer only. It must not become a second typed config owner, policy layer, validation fan-out layer, or selector-specific config wrapper.

### Compute/derivation boundary: `HyperParameters_GPU.hpp`

`Shared/HyperParameters/HyperParameters_GPU.hpp` owns config computation for entries that must be computed relative to other config entries.

That includes:

- root document consumption
- derivation of formula-relative config values
- root validation
- finalized config computation

This plan must not move relative config computation into selector code, execution code, decode code, or ad-hoc helpers.

If selector behavior depends on config relationships, those relationships must still be computed at the HyperParameters boundary first.

### Immutable view boundary: `HyperparameterGroupings.hpp`

`Shared/HyperParameters/HyperparameterGroupings.hpp` owns immutable grouped views.

These are:

- immutable read views
- grouped slices of already-authored or already-computed config
- consumer-facing payloads for subsystem use

They are not mutable config owners, and they are not a second derivation boundary.

### Allowed branch-off rule

The only allowed branch-off from the raw snapshot layer in this refactor is under the immutable-grouping boundary.

That branch-off must be:

- a single explicit typed grouping/view definition
- declared in `HyperparameterGroupings.hpp`
- taking `AiConfigSnapshot` in its signature when a snapshot-rooted bridge is required
- kept obviously thin and immutable

In other words, if this refactor needs a snapshot-rooted selector view before broader cleanup is complete, that branch is allowed only as a thin immutable grouping typed in `HyperparameterGroupings.hpp`, not as a new wrapper, facade, sidecar policy object, or duplicated config subtree.

### Ownership rules for this selector plan

This plan must obey all of the following:

- do not create a selector-specific config owner beside `AiConfigSnapshot`, `LanguageModelConfig`, or grouped HP views
- do not mirror config into a second selector policy struct outside the HyperParameters / HyperparameterGroupings boundaries
- do not forward-declare or re-host grouping payload types elsewhere
- do not compute relative selector defaults outside `HyperParameters_GPU.hpp`
- do not let decode or execution become config-authoring sites
- do not let Phase 2 mutate what Phase 1 statically assigned

The selector may consume authoritative config. It may not become an owner of config computation.

---

## Parameter ownership and forward lifetime

This plan must also respect the existing parameter ownership chain exactly.

### Single parameter access point

`training/Phases/Startup/Model/ParameterRegistry.hpp` is the only durable parameter access point.

Concretely:

- `::ParameterRegistry::StartupParameterRegistry` is the startup-owned writable parameter registry
- migrated writable parameter bundles live there
- durable `ParameterGroup` inventory lives there
- consumers must not invent a second parameter owner, mirror, or sidecar registry

If a subsystem needs parameters, it receives the registry through an explicit boundary signature. It does not rediscover parameters by reaching through unrelated owners.

### Pre-allocation and registration boundary

Pre-allocation must happen **inside the Phase 1 startup boundary**.

The ownership chain is:

- pre-allocation happens inside Phase 1 startup
- `training/Phases/Startup/Model/ParameterGroupRegistration.hpp` owns parameter registration/allocation transaction functions
- `training/Phases/Startup/Model/ModelGpuAssembly.cu` sequences assembly and initialization across startup-owned model construction
- registered writable parameters are published onto `::ParameterRegistry::StartupParameterRegistry`

This selector refactor must not move parameter allocation, registration, or parameter-owner construction into Phase 2, shared forward internals, decode logic, or execution logic.

### Registration rule

The rule is:

```text
pre-allocate inside Phase1Startup
register into StartupParameterRegistry
pass the registry explicitly to the operation that needs it
```

No later phase may behave as if it owns a second registration pass.

### Forward-boundary parameter access rule

Inside the forward boundary, parameter access must come from `executeModelForward(...)`, which is the forward-pass entry point.

That means:

- `Shared/Forward/ModelForward_GPU.hpp` owns the forward entry request boundary
- `ModelForwardRequest.parameter_registry` is the explicit parameter handoff into shared forward
- shared forward consumes parameters from the request-bound registry only

Forward code must not reacquire parameters through `LanguageModel`, `TrainingContext`, hidden globals, or ad-hoc reach-through helpers once it is already inside the `executeModelForward(...)` boundary.

### Post-forward lifetime rule

Once forward runs, forward-local state should be cleared as if forward never ran, **except for the resulting `ModelForwardOutputs`**.

Operationally, that means:

- forward-local scratch and borrowed transient state are not durable owners
- the retained product of a forward call is `ModelForwardOutputs`
- anything not part of the returned forward-output boundary should behave as ephemeral

`ModelForwardOutputs` is the single retained forward product for the active window. It may survive long enough for loss assembly, backward, sampling, or other caller-owned post-forward work, but it is still Category 1 forward-owned state and must be cleared by orchestration once that window closes.

### Ownership rules for this selector plan

This plan must obey all of the following:

- do not create a selector-specific parameter owner beside `::ParameterRegistry::StartupParameterRegistry`
- do not allocate selector trainable tensors outside the Phase 1 startup boundary
- do not register parameters outside startup-owned registration ops
- do not let shared forward become a registration or allocation owner
- do not let decode/execution pull trainable parameters through side channels when `executeModelForward(...)` already owns the forward parameter handoff
- do not retain extra forward-local state as a hidden durable owner; only `ModelForwardOutputs` is the retained forward product

The selector may consume registered parameters. It may not become the owner of parameter allocation, registration, or a second durable registry.

---

## Data ownership and runtime device access

This plan must also respect the existing data ownership and runtime-address boundaries exactly.

### Single realized data container

`BatchPayload` is the single realized data container used by both:

- Phase 2 training
- Phase 2 inference

For inference, `BatchPayload` is the prompted-sequence / decode-step data carrier.
For training, `BatchPayload` is the authoritative data lifetime container for the active training batch.

It is the container that carries all of the data the model should ever need for that call boundary.

### Data ownership rule

Data ownership must be resolved before model execution consumes it.

Operationally, that means:

- upstream sequence/batch realization produces `BatchPayload`
- `Phase2_TrainingLoop` consumes `BatchPayload`
- `Phase2_InferenceLoop` consumes `BatchPayload`
- shared forward, loss, selector, decode, and execution consume the payload passed to them

They do **not** invent a second current-batch owner, prompted-sequence owner, or sidecar runtime data container.

### `BatchPayload` ownership contract

`Shared/Batching/BatchPayload.hpp` owns the host-side realized runtime payload boundary.

Its role is:

- one struct carries everything a batch or prompt needs
- host semantic data is materialized once behind the payload boundary
- downstream code consumes it as an explicit payload
- host semantic fields remain immutable after build

For this plan, that means selector and execution logic must consume model data from `BatchPayload`, not from recomputed side channels.

### `BatchDeviceBindings` ownership contract

Because this is a CUDA system, runtime memory addresses are needed for uploaded runtime data.

`Shared/Batching/BatchDeviceBindings.hpp` is the **single access point** for runtime device addresses of runtime data.

That means:

- device pointers for the active upload boundary live on `BatchDeviceBindings`
- forward/loss/selector/execution code borrow runtime addresses through `BatchDeviceBindings`
- no code should stash hidden mutable `d_*` pointers back onto `BatchPayload`
- no code should reacquire runtime data addresses through a second device-address owner

`BatchDeviceBindings` owns no memory. It is the borrowed device-address view for the active step only.

### Host/device split rule

The split is strict:

- `BatchPayload` = authoritative host semantic/runtime payload
- `BatchDeviceBindings` = authoritative runtime device-address access point

Do not blur the two roles.

`BatchPayload` must not become a hidden mutable device-pointer owner.
`BatchDeviceBindings` must not become a second semantic data owner.

### Lifetime rule

The data lifetime contract for this refactor is:

- `BatchPayload` is the caller-owned data container for the active batch/prompt boundary
- `BatchDeviceBindings` is valid only for the active upload/step boundary
- Phase 2 must not cache `BatchDeviceBindings` beyond the step that produced it

If a later call needs runtime data again, it must receive a new explicit `BatchDeviceBindings` view for that step rather than reaching back into stale addresses.

### Ownership rules for this selector plan

This plan must obey all of the following:

- do not create a selector-specific data container beside `BatchPayload`
- do not create a selector-specific runtime-address owner beside `BatchDeviceBindings`
- do not let shared forward, decode, or execution rediscover the current batch from `TrainingState` or another hidden owner
- do not let the selector require data that is not present on `BatchPayload` or explicitly uploaded into `BatchDeviceBindings`
- do not write mutable runtime address state back onto `BatchPayload`
- do not treat `BatchDeviceBindings` as a durable semantic owner

The selector may consume `BatchPayload` and `BatchDeviceBindings`. It may not create a second data lifetime boundary.

---

## Backward, autograd, and tape ownership

This plan must also respect the existing backward/autograd/tape boundaries exactly.

### Backward entry contract

Backward runs from the explicit Phase 2 autograd boundary.

The important rule is:

- shared forward returns `ModelForwardOutputs`
- retained forward tensors in that output carry attached `grad_fn` chains
- loss assembly produces the scalar root on the explicit loss owner
- `executeAutogradBackward(...)` propagates gradients through that attached tape

Backward is therefore a tape-propagation boundary, not a second forward owner and not an optimizer substitute.

### Explicit step-state owners

The active backward window is driven by explicit caller-owned step state:

- `Forward::ModelForwardOutputs` holds Category 1 retained forward products for the active call window
- `AutogradLossState::loss_tensor` is the scalar loss root for backward
- `AutogradContext` is a thin input boundary that borrows these owners; it does not become a tensor owner itself

This means backward consumes explicit step-state owners. It must not recreate those owners on `TrainingState`, `LanguageModel`, selector code, or execution code.

### Core autograd primitives

The primitive propagation chain is:

```text
ModelForwardOutputs retained tensors
  -> Tensor.grad_fn attachments
  -> AutogradLossState.loss_tensor
  -> Tensor::backward(..., payload, device_bindings)
  -> GradFn::apply(...)
  -> upstream tensor grad buffers
  -> optimizer-window step later mutates weights
```

The important primitive roles are:

- `ModelForwardOutputs` keeps forward tensors alive long enough for backward
- `Tensor::grad_fn` is the attached backward edge on each autograd-tracked tensor
- `GradFn::apply()` is the operator-specific propagation primitive
- `Tensor::backward()` is the tape traversal / root-seeding / cleanup boundary
- registered parameter grad buffers are the accumulation destination for leaf parameters

### Batch-aware backward rule

Backward propagation in this system is batch-aware.

`Tensor::backward()` receives:

- `BatchPayload`
- `BatchDeviceBindings`

and forwards them to batch-aware `GradFn::apply(...)` calls.

That means:

- backward-time geometry/supervision facts come from `BatchPayload`
- backward-time device addresses come from `BatchDeviceBindings`
- GradFns must not reconstruct batch semantics or rediscover runtime addresses from hidden owners

### Gradient-write boundary

Backward does **not** write directly to weights.

Backward writes gradients into gradient buffers.

That means:

- leaf parameter tensors accumulate into `Tensor.grad_` / `tensor.grad_data()`
- persistent parameter gradient lifecycle is owned through the registry-owned `ParameterGroup` inventory
- `executeAutogradBackward(...)` may zero registered gradients at the accumulation-window boundary when required
- backward propagation itself must not mutate parameter values as if it were the optimizer

In short:

- backward computes and accumulates gradients
- optimizer owns parameter updates

That optimizer boundary must remain intact.

### Optimizer ownership boundary

Parameter-value mutation belongs to the optimizer boundary only.

So this plan must preserve:

- backward produces gradient state
- clipping / normalization / optimizer-window policy stay outside backward
- optimizer update is the only boundary that writes new parameter values

Do not smuggle parameter updates into GradFns, `Tensor::backward()`, selector loss helpers, execution loss helpers, or any post-backward diagnostic path.

### Tape lifecycle rule

The tape lifecycle is explicit and single-owner:

- forward builds the live tape through attached `grad_fn` nodes
- backward traverses the tape from `loss_tensor`
- `Tensor::backward()` is responsible for applying GradFns, synchronizing the active stream, flushing deferred cleanup, and releasing saved forward state
- once the active backward/loss window ends, caller-owned step-state clear logic tears down `ModelForwardOutputs` and `AutogradLossState`

This must remain a single-owner lifecycle. No extra teardown owner should be introduced.

### Ownership rules for this selector plan

This plan must obey all of the following:

- do not let selector code become a second backward entry point
- do not let backward own parameter updates
- do not let GradFns write parameter values directly
- do not move retained forward tensors off `ModelForwardOutputs`
- do not move the scalar loss root off `AutogradLossState`
- do not introduce a second tape owner or extra cleanup authority beside the current backward/step-state lifecycle
- do not bypass the registry-owned parameter gradient lifecycle by enumerating trainable tensors ad hoc in backward

The selector may participate in the tape through normal forward tensors, retained selector forward results, and GradFns. It may not become the owner of optimizer stepping or a parallel backward system.

---

## Core design truth

The model should **not** learn that `pow10` is just a label.

It should learn that `pow10` expresses **base-10 place meaning**:

- `pow10 = 1` means the digit contributes in the tens place
- `pow10 = 0` means the digit contributes in the ones place
- `pow10 = -1` means the digit contributes in the tenths place

For a number with sign $s \in \{-1, +1\}$, digits $d_i$, and place exponents $p_i$, the intended semantic value is:

$$
value = s \sum_i d_i 10^{p_i}
$$

`pow10` is therefore not a class id; it is part of the number's semantic basis.

---

## The representation split: numeric meaning vs surface identity

The architecture must represent **two concepts, not one**.

### Numeric meaning

This is the meaning-side representation:

- digit sequence
- digit-place structure
- sign
- mantissa-like scale information
- exponent / magnitude behavior
- contribution semantics for each digit slot

This is the side that should capture relationships such as:

- `42` is numerically close to `43`
- `42` is numerically closer to `40` than `4000`
- `42`, `42.0`, and `4.2e1` are related in value

### Surface identity

This is the form-side representation:

- exact written atom form
- `atom_entry_id`
- formatting flags
- sign formatting
- decimal-point presence
- exponent formatting
- leading-zero structure

This side preserves distinctions such as:

- `42` != `042`
- `42` != `42.0`
- `42.0` != `4.2e1`

### Required rule

Numeric meaning and surface identity must be **separate but jointly available**.

Meaning-side may say:

- `42 == 42.0 == 4.2e1` in semantic value space

Surface-side must still preserve:

- `"42" != "42.0" != "4.2e1"`

That distinction matters for next-token prediction, reconstruction, and selector behavior.

---

## Target encoding contract

The strongest baseline for this refactor is:

```text
NumberEncoder(arg_number):
    for each digit:
        encode (digit, pow10) as a contribution slot
    pool contribution slots
    add global mantissa/exponent embedding
    keep surface identity separate
```

### Digit-place contribution slots

Each digit slot should encode:

- digit identity
- place identity
- digit contribution meaning at that place

Conceptually:

```text
slot_i = DigitPlaceEncoder(
    digit_i,
    pow10_i,
    sign,
    atom_type
)
```

A useful implementation shape is:

```text
slot_i =
    digit_emb[digit_i]
  + pow10_emb[pow10_i]
  + MLP([
        digit_i / 9,
        pow10_i / max_pow10,
        is_zero_digit,
        sign,
        atom_type
    ])
```

### Important prohibition

Do **not** reduce the design to:

```text
digit_embedding[digit] + pow10_embedding[pow10]
```

`pow10_emb[p]` alone only says **which place** is present.
The selector/encoder must also learn **what contribution the digit makes at that place**.

So:

- `4 @ 10^1` must differ from `4 @ 10^0`
- `4 @ 10^0` must differ from `4 @ 10^-1`

Those may share digit identity, but they must not collapse in contribution meaning.

---

## End-to-end model contract

The intended path is:

```text
token_ids + atom_entry_ids
  -> token embeddings + NumberEncoder(arg_number)
  -> transformer
  -> LM head + numeric heads
  -> losses
```

At the token level:

```text
x_t = token_embedding[token_id]
x_t = token_embedding[<INT>/<FLOAT>] + number_embedding(arg_number)
```

The normal LM head still predicts placeholder token identity:

- `<INT>`
- `<FLOAT>`

A separate numeric path predicts atom metadata and numeric structure.

### Training-time causal boundary

During training, current atom metadata may be used **only** for the current token input.

That means:

- position $t$ may consume metadata attached to token $t$
- position $t$ may **not** consume atom metadata belonging to target token $t+1$
- atom metadata for $t+1$ is supervision only

Formally, the training contract is:

$$
h_t = f(x_t, metadata_t)
$$

with supervision applied as:

$$
\mathcal{L}_{meta}(h_t, target\_metadata_{t+1})
$$

but **not**:

$$
h_t = f(x_t, metadata_t, target\_metadata_{t+1})
$$

This is a hard anti-leakage rule, not a preference.

The model may learn to predict next-atom structure from the current hidden state, but it must not be handed the next token's atom metadata as an input feature at the previous position.

---

## Selector and execution boundary

This refactor must establish a clean boundary between **selection** and **execution**.

### Selector responsibilities

Selector-owned behavior includes:

- consuming atom metadata / numeric representation inputs
- learning numeric meaning
- scoring or resolving candidate numeric structure
- producing the selector result used downstream
- owning the logic that decides what numeric binding is being provided

### Execution responsibilities

Execution-owned behavior includes:

- checking whether execution is active
- consuming the selector-provided result when execution is active
- using the existing execution plumbing and consumer path
- remaining agnostic to selector internals

### Required decoupling rule

Execution must **not** contain or recreate selector logic.

Selector must **not** depend on execution behavior in order to define numeric meaning.

Execution is a downstream consumer only.

### Plumbing invariant

The execution plumbing should remain the **exact same plumbing** unless a boundary-preserving adapter is strictly required.

That means this refactor should preserve, as much as possible:

- the current execution activation gate
- the current execution call chain
- the current consumer-side data flow
- the current execution ownership boundaries

What changes is **who owns atom selection logic**, not the overall execution orchestration.

### Consumption rule

- if `execution_active == false`, execution does not consume selector output
- if `execution_active == true`, execution uses what the selector provides

No hidden fallback path should allow execution to re-run old selection behavior.

### Temporary decode contract

This decode contract is **temporary until it is proven to work**.

For the temporary phase, decode will:

- use the provided metadata
- look up `AtomTable` entry ids from that metadata
- fill the atom placeholder from the resolved `atom_entry_id`

Conceptually:

```text
predicted/provided metadata
  -> AtomTable entry-id lookup
  -> resolved atom_entry_id
  -> fill atom placeholder during decode
```

This is an implementation bridge, not the final design.

During this temporary phase:

- decode may rely on metadata-to-`atom_entry_id` lookup
- placeholder filling is allowed to be id-backed
- execution remains a consumer of the resolved decode output when active

But this temporary path must remain clearly marked as transitional. It must not be treated as the final selector contract.

### Later decode contract

The later target design is:

- no `atom_entry_id` dependency in the decode contract
- generate values directly
- pass generated values into execution
- kick off an actual reasoning loop

Conceptually:

```text
predicted numeric structure / value
  -> decoded value
  -> execution consumes value if active
  -> reasoning loop operates on values rather than AtomTable ids
```

So the temporary decode contract is:

- **metadata -> AtomTable id -> placeholder fill**

while the later decode contract becomes:

- **metadata/value prediction -> value generation -> execution/reasoning**

The architecture should therefore keep the decode boundary narrow enough that the temporary id-lookup bridge can be deleted cleanly once the value-generation path is proven.

---

## Training heads and supervision

The architecture should use the normal LM loss plus numeric auxiliary losses that force place-value semantics.

### Loss 1: normal language-model loss

Standard next-token prediction remains unchanged:

$$
L_{lm} = \mathrm{cross\_entropy}(lm\_logits[t], target\_token\_id[t])
$$

Gradient path:

```text
L_lm
  -> LM head
  -> h_t
  -> transformer
  -> token embeddings
```

This teaches normal text prediction.

### Loss 2: next-atom metadata prediction

At position $t$, if the target at $t+1$ is an atom, the model predicts ArgNumber-style metadata for that next atom.

The target metadata for $t+1$ is used **only** to compute supervision. It is not an input to position $t$.

For target `42`, the metadata target may include:

```text
sign = positive
digit_count = 2
digit[0] = 4
pow10[0] = 1
digit[1] = 2
pow10[1] = 0
```

This teaches explicit number structure.

### Variable-length digit-slot masking

Numeric atoms have variable digit counts:

- `42`
- `0042`
- `4.2`
- `4.2e1`

Therefore any per-digit supervision path must distinguish between:

- real digit slots
- padded digit slots

The required contract is:

```text
digit_loss_mask[i] = true only for real digit slots
padding digit slots do not contribute loss
```

If a fixed maximum number of digit slots is used, then for a target with `digit_count = N`:

$$
digit\_loss\_mask[i] = (i < N)
$$

and all digit-level losses must be masked by that boolean before reduction.

Examples:

- `42` -> two real digit slots
- `0042` -> four real digit slots if leading zeros are preserved in surface structure
- `4.2` -> two real digit slots
- `4.2e1` -> mantissa digit slots come from the written mantissa digits; exponent structure is supervised separately, not as fake mantissa padding digits

This rule applies to any digit-level target, including:

- digit identity
- `pow10`
- digit-place contribution class
- per-slot value bucket
- any auxiliary digit reconstruction head

Padding slots must contribute exactly zero supervision weight. They are not negative examples, and they are not "missing digits" to be predicted.

### Loss 3: value reconstruction loss

From predicted digit/pow10 structure, reconstruct a magnitude-oriented target such as:

- sign
- mantissa bucket or mantissa approximation
- exponent
- approximate value bucket

Examples:

- `42` -> sign `+`, mantissa ≈ `4.2`, exponent `1`
- `420` -> sign `+`, mantissa ≈ `4.2`, exponent `2`
- `4.2` -> sign `+`, mantissa ≈ `4.2`, exponent `0`

This forces the model to learn that changing `pow10` changes magnitude.

### Loss 4: digit-place contribution loss

Each digit slot should predict its place contribution semantics.

For `42`:

- `4` at `pow10=1` contributes in the tens place
- `2` at `pow10=0` contributes in the ones place

Reasonable targets include:

- `contribution_exponent = pow10`
- `contribution_digit = digit`
- `contribution_nonzero = digit != 0`
- `slot_value_bucket = digit × 10^pow10` using a bucketed or log-scaled target

Do **not** use giant raw numeric targets directly.

If this loss is applied per digit slot, it must use `digit_loss_mask` so padded slots have zero contribution.

### Loss 5: contrastive numeric geometry

This is the loss that teaches numbers not to behave like arbitrary ids.

Desired behavior:

$$
distance(embed(42), embed(43)) < distance(embed(42), embed(420))
$$

and similarly:

- `42` closer to `40` than to `4000`
- `4.2` related to `42`, but not identical in scale

This teaches meaningful local numeric geometry in embedding space.

---

## Backward-pass and data ownership rules

The backward rule is simple and strict:

- gradients train the model to represent and predict numeric structure
- gradients do **not** mutate the `AtomTable`
- gradients do **not** mutate provided `ArgNumber` metadata
- gradients do **not** mutate `atom_entry_id` values supplied from text
- gradients do **not** justify feeding target $t+1$ atom metadata into position $t$ as an input shortcut

The `AtomTable` remains fixed during a training step.

This refactor is about improving learned numeric representation and selector behavior, not about making source atom metadata trainable.

---

## Refactor workstreams

### Workstream 0 - lock the boundary before code moves

Define the selector/execution seam explicitly before ripping out logic.

Required outcomes:

- document the selector output contract
- identify the current execution consumer boundary
- identify which plumbing must remain unchanged
- identify the exact legacy atom-selection logic to delete
- define fail-loud behavior for missing numeric metadata or invalid selector output
- define the temporary decode contract separately from the later no-id decode contract
- document the config ownership chain: `ai_config_paths.hpp` -> `HyperParameters_GPU.hpp` -> `HyperparameterGroupings.hpp` -> Phase1 static assignment
- explicitly keep static assignment inside Phase 1 startup only
- document the parameter ownership chain: Phase1 pre-allocation -> `ParameterGroupRegistration.hpp` -> `ParameterRegistry.hpp` -> `executeModelForward(...)`
- explicitly keep parameter allocation and registration inside Phase 1 startup only
- document the data ownership chain: realized sequence/batch data -> `BatchPayload` -> upload -> `BatchDeviceBindings` -> shared forward/loss/decode/execution
- explicitly keep runtime device-address access on `BatchDeviceBindings` only
- document the backward chain: `ModelForwardOutputs` -> attached `GradFn`s -> `AutogradLossState::loss_tensor` -> `Tensor::backward()` -> registry-owned gradient buffers -> optimizer update later
- explicitly keep parameter-value mutation out of backward and inside the optimizer boundary only

### Workstream 1 - rip out current atom selection ownership

Move atom selection logic out of execution-owned behavior and into a dedicated selector-owned path.

Required outcomes:

- current selector logic is removed from the wrong owner
- selector logic becomes its own module or ownership boundary
- execution no longer reconstructs or shadows selector decisions
- no silent fallback path preserves the old mixed design
- no selector-side config mirror or sidecar policy object is introduced
- no selector-side parameter mirror, sidecar registry, or late registration path is introduced
- no selector-side data container or runtime-address sidecar is introduced
- no selector-side parallel backward path or weight-update shortcut is introduced

### Workstream 2 - implement numeric meaning encoding

Add the number-side encoding path centered on digit/place contribution slots.

Required outcomes:

- `NumberEncoder(arg_number)` exists as the numeric representation path
- digit-place contribution encoding is explicit
- global mantissa/exponent features are included
- surface identity remains separate
- training inputs remain causal: current-token metadata only, next-token metadata supervision only
- variable-length digit supervision is masked so padding slots contribute zero loss
- selector config relationships are still computed at the HyperParameters boundary, not inside the encoder path

### Workstream 3 - add numeric supervision heads

Introduce the heads and losses needed to teach the intended semantics.

Required outcomes:

- next-atom metadata prediction head
- value reconstruction head
- digit-place contribution supervision
- contrastive numeric geometry objective or equivalent training signal
- explicit protection against target metadata leakage across the $t \rightarrow t+1$ boundary
- explicit `digit_loss_mask` handling for variable-length digit-slot targets
- selector losses propagate through normal tape primitives and do not perform direct parameter writes

### Workstream 4 - route selector output into unchanged execution plumbing

Execution integration is intentionally narrow.

Required outcomes:

- execution activation behavior remains intact
- execution consumes selector output only when active
- execution plumbing remains materially the same
- selector/execution adapter, if needed, is thin and structural only
- execution does not gain new selector heuristics
- temporary decode may fill placeholders through `AtomTable` entry-id lookup, but the boundary stays narrow enough for later deletion
- decode/execution consume authoritative config views only; they do not author or recompute static config
- forward-time selector parameter access comes only through `executeModelForward(...)` request boundaries
- forward retains only `ModelForwardOutputs` as its caller-visible post-run product
- training/inference runtime data access comes only through `BatchPayload` + `BatchDeviceBindings`
- backward consumes the caller-owned autograd step state and writes gradients only; optimizer owns weight mutation later

### Workstream 5 - validate parity and semantics

Prove that the new design learns the intended semantics and preserves the intended boundary.

Required outcomes:

- place-value distinctions are learnable and testable
- numeric meaning and surface identity remain separable
- selector logic is no longer execution-owned
- execution path still consumes selector output through the same plumbing shape
- AtomTable immutability is preserved during training

---

## Acceptance criteria

This plan is successful when all of the following are true:

1. `pow10` is learned as place-value semantics, not as a flat class label.
2. `4 @ 10^1`, `4 @ 10^0`, and `4 @ 10^-1` are distinguishable in internal contribution meaning.
3. `42`, `042`, `42.0`, `+42`, and `4.2e1` can share numeric meaning while preserving distinct surface identity.
4. The LM head still predicts placeholder token identity normally.
5. Numeric auxiliary heads teach structure, magnitude, and local numeric geometry.
6. Atom selection logic is no longer entangled with execution logic.
7. Execution uses selector-provided output only when execution is active.
8. The execution plumbing remains effectively the same from the consumer side.
9. No fallback path reintroduces the old mixed selector/execution design.
10. The AtomTable remains fixed during a training step.
11. During training, position $t$ never receives target token $t+1$ atom metadata as input.
12. Padding digit slots contribute zero loss for variable-length numeric targets.
13. The temporary decode path may use metadata to resolve `AtomTable` entry ids and fill placeholders, but that contract is explicitly transitional and removable.
14. Static config assignment remains inside the Phase 1 startup boundary.
15. `ai_config_paths.hpp` remains the single raw config entry point through `AiConfigSnapshot::document`.
16. Relative config computation remains owned by `HyperParameters_GPU.hpp`.
17. Any snapshot-rooted selector branch-off exists only as a thin immutable grouping in `HyperparameterGroupings.hpp` taking `AiConfigSnapshot` in its signature.
18. `training/Phases/Startup/Model/ParameterRegistry.hpp` remains the only durable parameter access point.
19. Parameter pre-allocation and registration remain inside the Phase 1 startup boundary.
20. Shared forward receives the parameter registry only through `executeModelForward(...)` request signatures.
21. The only retained product of a forward call is `ModelForwardOutputs`; other forward-local state behaves as ephemeral and is cleared by the owning boundary.
22. `BatchPayload` remains the single realized data container for both Phase 2 training and Phase 2 inference.
23. `BatchDeviceBindings` remains the single access point for runtime device addresses of runtime data.
24. No second current-batch owner, prompted-sequence owner, or runtime-address owner is introduced beside `BatchPayload` and `BatchDeviceBindings`.
25. Backward propagates through attached GradFns from caller-owned `ModelForwardOutputs` / `AutogradLossState`; it does not become a second forward owner.
26. Backward writes gradients, not parameter values.
27. The optimizer boundary remains the only owner of parameter-value mutation.

---

## Explicit non-goals

This plan does **not** attempt to:

- redesign the broader execution system
- redesign the config ownership chain
- redesign the parameter ownership chain
- redesign the data ownership chain built around `BatchPayload` / `BatchDeviceBindings`
- redesign the autograd/tape ownership chain
- change execution scheduling semantics
- make execution logic the owner of numeric meaning
- mutate AtomTable contents during backpropagation
- collapse surface identity into numeric meaning
- rely on raw unbounded numeric regression as the main supervision signal
- create a second selector-owned config document, wrapper, or sidecar policy tree
- create a second selector-owned parameter registry, late registration pass, or hidden forward parameter reach-through path
- create a second selector-owned data container, current-batch mirror, or runtime device-address sidecar
- create a selector-owned optimizer shortcut, direct weight-write path, or parallel backward/tape system

---

## Relationship to existing execution docs

This document should remain compatible with `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md`.

This document should also remain compatible with the GRIM-text config ownership rules captured in:

- `control/ai_config_paths.hpp`
- `resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp`
- `resources/models/GRIM-text/Shared/HyperParameters/HyperparameterGroupings.hpp`
- `resources/models/GRIM-text/GRIM/Docs/Config.md`

This document should also remain compatible with the parameter ownership and forward-entry boundaries captured in:

- `resources/models/GRIM-text/training/Phases/Startup/Model/ParameterRegistry.hpp`
- `resources/models/GRIM-text/training/Phases/Startup/Model/ParameterGroupRegistration.hpp`
- `resources/models/GRIM-text/training/Phases/Startup/Model/ModelGpuAssembly.cu`
- `resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.hpp`
- `resources/models/GRIM-text/Shared/Forward/ModelForwardOutputs.hpp`

This document should also remain compatible with the data/runtime-address boundaries captured in:

- `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`
- `resources/models/GRIM-text/Shared/Batching/BatchDeviceBindings.hpp`

This document should also remain compatible with the backward/autograd/tape boundaries captured in:

- `resources/models/GRIM-text/GRIM/Docs/Autograd.md`
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.hpp`
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- `resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp`
- `resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.cu`

Ownership split:

- this document owns the **selector and numeric-meaning refactor basis**
- the execution cutover document owns the **execution architecture and runtime execution contract**
- the config chain above owns raw snapshot load, derivation, immutable grouped views, and Phase1 static assignment boundaries
- the parameter/forward chain above owns startup pre-allocation, registration, explicit forward parameter handoff, and post-forward output lifetime
- the batching/device-binding chain above owns realized runtime data payloads and the sole borrowed device-address boundary
- the autograd/tape chain above owns retained forward tape state, GradFn propagation, gradient-buffer writes, and cleanup, while optimizer stepping owns parameter-value mutation

If a future code change affects both selector ownership and execution flow, both documents should be updated in the same change.

---

## Short implementation summary

The model should learn that a digit's value depends on **where it sits**, not just **what digit it is**.

The selector should own numeric selection logic.
Execution should stay downstream and use what the selector provides when execution is active.
The plumbing should stay the same; the logic ownership should not.
