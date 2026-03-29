# ExecutionBlock Structured Execution Cutover Plan

**Status:** Workstream 0 complete; Workstream 1 next  
**Scope:** `token_to_slot_map`, `teacher_steps`, row-local ExecutionBlock execution, GRMT cutover, inference alignment  
**Policy:** **No backwards compatibility**. Delete legacy behavior instead of preserving it.

> Living refactor companion: [`EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md`](EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md)
>
> Living flow diagram: [`EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md`](EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md)
>
> Any change that affects ownership boundaries, file splits, payload shape, validation rules, execution gating, selector behavior, runtime flow, or deleted legacy paths must update both living artifacts in the **same change**.

---

## Objective

Make structured execution in GRIM-text behave as a single, explicit system:

- `execution_block_GPU.cu` is reduced to live, layer-owned code before semantic refactor work continues,
- plain numeric atoms remain legal text tokens,
- execution-active rows carry explicit register metadata,
- training and validation enforce the exact same contract,
- ExecutionBlock only consumes row-local state,
- execution / selector learned parameters are real tensor state, optimizer-visible through `ParameterGroup`, and checkpointed through the main serialization layer,
- `BatchPayload` is the only batched transport for compiled execution metadata, while `TensorContract` and `TensorConversion` stay narrow tensor-contract/layout-conversion boundaries,
- old debug-only serialization paths are removed.

The current codebase mixes two incompatible assumptions:

1. `token_to_slot_map == -1` means **non-state-bearing** numeric token.
2. Every `<NUM>` token must have a slot when `execution_block_enabled=true`.

This plan removes that ambiguity completely.

---

## Mandatory living artifacts during the cutover

These artifacts are not optional project decoration. They are part of the cutover contract.

### Companion codoc contract

`EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md` is the **living refactor ledger** for this cutover.

Every qualifying change must update the codoc in the same change with, at minimum:

1. active workstream id and status delta
2. changed files
3. what changed structurally
4. what now owns the concern
5. allowed integration / hook points after the change
6. migrated callers or consumers
7. deleted or removed legacy paths
8. validation performed, skipped, or still blocked
9. remaining gaps / next gate
10. diagram delta summary

If code changes the architecture but the codoc does not change, the refactor step is incomplete.

### Flow diagram contract

`EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md` is the maintained flow artifact for this cutover.

Every qualifying change must update the flow artifact in the same change when it alters:

- workstream sequencing
- ownership boundaries
- data transport shape
- validation entry points
- runtime execution flow
- inference / selector flow
- deleted fallback or legacy paths

The flow artifact must:

- keep a Mermaid diagram that reflects the **current enforced flow**, not stale target-state wishful thinking
- remove deleted legacy nodes/edges immediately
- show where validation gates occur
- show where the codoc/plan expect same-change documentation maintenance

### Global completion gate for every workstream

No workstream is complete until all three artifacts are current in the same change:

- this implementation plan
- the living codoc companion
- the maintained flow diagram

---

## Truth hierarchy

The plan must treat **one thing** as truth and everything else as a projection of that truth.

### Actual semantic truth

The actual truth for execution-active rows is the **canonical structured execution record**:

- the initial state literals,
- their intended register identities,
- the ordered execution steps,
- the expected step outputs,
- the final result semantics.

That structured record exists **before tokenization** and is the only place where execution meaning originates.

### Derived projection 1 — runtime bootstrap truth

`token_exec_slots` / `token_to_slot_map` is the **runtime binding projection** of the structured record.

It answers only:

- which token position initializes which register slot.

It is not the semantic source of the execution program. It is the bootstrap substrate used by `bootstrapMemoryFromSlotMap` and inference prompt upload.

### Derived projection 2 — supervision truth

`teacher_steps` is the **execution supervision projection** of the same structured record.

It answers only:

- which slot should be read as arg1,
- which slot should be read as arg2,
- which op should fire,
- which slot should receive the write,
- what scalar result is expected.

It is not the source of token/register alignment. It is the supervisory view over the already-defined structured execution program.

### Consequence for the refactor

`token_exec_slots` and `teacher_steps` are **not parallel truths** and must never be authored independently.

They are a paired, compiled representation of one upstream structured record:

- `token_exec_slots` = runtime binding view
- `teacher_steps` = execution supervision view

The pipeline must derive both from the same builder pass and validate that they remain consistent.

### Compiled structured-execution payload authority

Runtime, batching, validation, and orchestration must consume one **compiled structured-execution payload** derived from the canonical `StructuredExecutionRecord`.

Suggested compiled form:

- `CompiledStructuredExecutionPayload { execution_active, token_exec_slots, compiled_bootstrap_bindings, teacher_steps, slot_selection_targets }`

If some storage or transport layer keeps these as adjacent fields instead of one nested C++ object, they still form one conceptual compiled payload and `execution_active` is the authoritative activation bit.

Rules:

- runtime/batching/validation decide whether a row is execution-active from compiled payload activation state, **not** from `teacher_steps.size()`
- `teacher_steps` being non-empty is a supervised-training payload validity rule in this cutover, not the activation source
- non-execution rows serialize/transport `execution_active = false`
- execution-active rows serialize/transport `execution_active = true`

---

## Formal derivation rules

These definitions remove the remaining ambiguity in the validator and inference contract.

### Builder-side bound literal definition

`StructuredExecutionRecord` must contain an explicit list of **bootstrap literal bindings**.

Each binding identifies:

- a semantic literal occurrence in the canonical structured record,
- the slot it must initialize,
- the rendered occurrence that is allowed to seed memory.

Suggested shape:

- `BootstrapLiteralBinding { literal_id, slot_id, occurrence_role, rendered_span_id }`

After tokenization, the builder must emit a compiled provenance view suitable for GRMT/runtime validation, e.g.:

- `CompiledBootstrapBinding { binding_id, token_pos, slot_id }`

For the cutover architecture, **only bootstrap literal bindings create state-bearing token positions**.

That means:

- not every numeric token in the rendered text is state-bearing,
- not every textual repetition of the same value is state-bearing,
- only literals explicitly named in `bootstrap_bindings[]` may initialize registers.

### Tokenization contract for a bound literal

For each `BootstrapLiteralBinding`, the canonical builder must render the bound literal into one marked span and then tokenize that rendered sequence.

That bound span must resolve to **exactly one** token position satisfying all of:

- `token_id == Tokenizer::atomTypeToTokenId(ATOM_NUM)`
- `token_atom_mask[pos] == 1`
- `atom_entry_ids[pos] != kAtomEntryNone`

If tokenization yields:

- zero numeric atom tokens for the bound span, or
- more than one numeric atom token for the bound span,

the builder must **throw** and refuse to emit the sequence.

There is no multi-piece binding rule. The mechanism is: **one bound literal span must compile to one ATOM_NUM token**.

### Formal definition of required state-bearing token positions

After builder compilation, the required state-bearing token positions for a row are exactly:

$$
R = \{\, pos \mid token\_exec\_slots[pos] \neq -1 \,\}
$$

and `R` must be exactly the set of token positions obtained from `bootstrap_bindings[]` by the one-span → one-ATOM_NUM-token compilation rule above.

Therefore the validator must **not** infer requiredness by scanning arbitrary `<NUM>` tokens in the text.

It validates the compiled result:

- every `pos in R` must carry a slot in `[S, V)`,
- every `pos not in R` must carry `-1`,
- `R` must correspond to the serialized compiled bootstrap provenance emitted by the builder.

### Bootstrap binding uniqueness rules

Within one execution-active row, bootstrap bindings are a **one-to-one initialization map**.

That means all of the following are mandatory:

- each compiled bootstrap binding has exactly one `(token_pos, slot_id)` pair,
- no two bootstrap bindings may target the same `token_pos`,
- no two bootstrap bindings may initialize the same `slot_id`,
- therefore one state-bearing token position may not appear more than once for the same slot,
- therefore duplicate slot initialization is illegal.

So the compiled bootstrap binding set for a row must be injective in both dimensions:

$$
\forall i \neq j:\; token\_pos_i \neq token\_pos_j \;\land\; slot\_id_i \neq slot\_id_j
$$

If the builder encounters either:

- duplicate `slot_id` initialization, or
- duplicate `token_pos` targeting,

it must **throw immediately** and refuse to emit the row.

If the runtime or loader later observes such a row anyway, the shared validator must also fail hard. Two bindings targeting the same `token_pos` are structurally impossible under the current contract and are treated as corruption, not as a recoverable case.

If a future design ever wants aliasing, repeated initialization, or many-to-one bootstrap behavior, that must be introduced as a **new explicit binding mode** with its own serialized representation and runtime rules. It is not allowed implicitly under `bootstrap_bindings[]`.

### Formal definition of the row-local slot domain

The row-local legal slot domain is an explicit, finite, immutable set for each execution-active row.

Define it as:

$$
D_{row} = \{\, s \mid s \text{ appears as a slot id anywhere in the row's canonical structured execution record} \,\}
$$

Concretely, `D_row` is the unique set of slot ids referenced by that row's structured execution program:

- every `bootstrap_bindings[].slot_id`
- every teacher-step read slot
- every teacher-step write slot

Required properties:

- `D_row` is fixed before tokenization and does not change during execution.
- `D_row` contains no duplicate slot ids.
- `D_row \subseteq [S, V)`.
- every bootstrap binding slot id belongs to `D_row`.
- every teacher-step arg/read/write slot id belongs to `D_row`.
- a slot may be first-written later only if it was already a member of `D_row` before execution began.
- no loader, validator, or runtime path may substitute configured slot ranges for `D_row`.

Configured slot ranges such as `[S, V)` are only outer bounds. They are **not** the row-local slot domain itself.

If `StructuredExecutionRecord` carries an explicit `slot_domain` field, that field must equal `D_row` exactly rather than acting as a loose superset or advisory hint.

### Runtime materialization of `D_row`

For this cutover, GRMT and `BatchPayload` do **not** serialize a separate runtime `D_row` field.

Whenever runtime or validation code needs the row-local slot domain, it must reconstruct:

$$
D_{row}^{runtime} = \{\, b.slot\_id \mid b \in compiled\_bootstrap\_bindings \,\} \cup \{\, step.arg1\_slot,\; step.arg2\_slot,\; step.write\_slot \mid step \in teacher\_steps \,\}
$$

Rules:

- runtime `D_row` comes only from the union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`
- selector supervision targets do **not** add slot-domain membership
- configured slot ranges `[S, V)` remain outer bounds only
- if a future schema later serializes `D_row` explicitly, loader/validator code must require exact equality with `D_row^{runtime}`; until then, the reconstructed union is the only legal runtime materialization

### Slot identity and teacher-step write rules

Slot identity is **stable for the entire lifetime of one row**.

That means:

- a slot id refers to the same semantic register identity across all bootstrap bindings and all teacher steps in that row,
- bootstrap initializes a slot's starting value but does not define a temporary meaning that can later be reassigned,
- a teacher-step write may update the scalar value stored in a slot,
- a teacher-step write may target a slot that was not bootstrap-initialized,
- but that target slot must already belong to `D_row` and must be in-range,
- no teacher step may invent a new slot id dynamically,
- no teacher step may rebind a slot to mean some different logical register midway through the row.

So the allowed distinction is:

- **first write to an already-legal slot** = allowed,
- **dynamic creation of a new slot identity** = forbidden,
- **rebinding an existing slot to a new meaning** = forbidden.

Under this contract, teacher-step write targets do **not** need to be bootstrap-initialized first, but they do need to belong to `D_row` already; configured slot ranges provide only outer bounds and never define row membership.

### Builder-time invalid execution rows

An execution-active row whose compiled structured-execution payload would be emitted with `execution_active = true`, but whose canonical structured record has:

- `bootstrap_bindings.empty()`

is malformed and must be rejected by the builder **before** tokenization/serialization is emitted.

This is a builder-time validity rule, not merely a validator-time warning. The shared validator should still reject such a row if corrupted data somehow reaches runtime, but the canonical builder is not allowed to produce it.

For supervised training rows in this cutover, the same active compiled payload must also produce non-empty `teacher_steps`; that remains a payload validity requirement, but it is not the source of execution-active status.

### Role of expected scalar results

`StructuredExecutionRecord` may carry expected scalar results for each execution step and/or final state.

Those expected values have a narrow purpose:

1. compile into `TeacherStep.expected_value` for execution supervision,
2. support runtime execution-loss/debug assertions,
3. support tests that compare executed values against the canonical structured program.

They are **not** used to infer slot-bearing token positions.

They are **not** part of the structural slot-map validator except for basic sanity checks such as:

- presence when a supervised step requires one,
- finite numeric value,
- representation range checks if the builder imposes them.

Static payload validation is about compiled bindings and slot consistency, not about replaying arithmetic semantics.

### Deterministic generation-time `<NUM>` policy

Generation must use one deterministic admission/selection/binding rule for `<NUM>`.

Define the legal candidate set from row-local execution memory:

$$
L = \{\, s \mid s \in [S, V) \;\land\; valid\_mask[s] = 1 \;\}
$$

Decode must then run an explicit slot-selection policy over `L` **before** `<NUM>` is admissible.

Call the result:

`selected_num_slot := resolveDecodeTimeNumSlotSelection(L, decode_context)`

The contract is:

1. Build `L` from **all valid live slots** in the row-local execution memory.
2. If the slot-selection policy resolves **exactly one** legal slot `s`, then `<NUM>` may remain admissible and, if sampled, binds to:
	- `new_token_slot_id = s`
	- `token_numeric_value = inference_exec_memory.values[s]`
3. If the slot-selection policy resolves **no** legal slot, the sampler masks `<NUM>` to `-inf` **before sampling**.
4. If multiple legal candidates exist but the slot-selection policy cannot choose one, generation masks `<NUM>` instead of guessing; validation/debug/deterministic inspection paths must fail hard.
5. If `<NUM>` is ever observed without a pre-resolved selected slot, throw immediately.

There is no fallback where `<NUM>` is emitted with `slot_id = -1` or `numeric_value = 0.0f`.

There is also no special rule that binds `<NUM>` to “the last write” merely because it was the most recent numeric result.

Cutover requirement: `resolveDecodeTimeNumSlotSelection(...)` must be driven by an explicit slot-selection mechanism.

Acceptable implementations:

- a dedicated slot-selection head
- a pointer-style selector over the valid live slots

Unacceptable implementations:

- recency heuristics (`last write`, `most recent live slot`)
- positional heuristics (`first valid`, lowest/highest slot index, nearest-by-position shortcuts)
- text-matching heuristics (`closest textual number`, nearest rendered numeric atom)
- post-sample slot inference
- any other implicit fallback selector

This means decode-time `<NUM>` is architecturally incomplete until an explicit slot-selection mechanism exists.

This is not optional prose; it is the decode-time mechanism that must be implemented before the rest of the cutover relies on it.

### Decode-time slot selector contract

`resolveDecodeTimeNumSlotSelection(...)` is a **row-local, explicit reference resolver** over live execution slots.

Its job is **not** to predict a number.

Its job is to choose which legal live slot a decode-time `<NUM>` would refer to.

#### Purpose

Given:

- current decode state,
- current row-local execution memory,
- current valid live-slot set,

the selector must return exactly one of:

- one legal selected slot,
- explicit null selection,
- explicit ambiguity/failure state.

It must never:

- infer a slot after sampling,
- use recency as a hidden selector,
- use text-matching as a hidden selector,
- emit an unbound `<NUM>`.

#### Inputs

1. **Decode query state**
	- A row-local decode query vector derived from the current decode hidden state.
	- Suggested form: `q_t = W_q * h_t`
	- `h_t` = current decode hidden state
	- `q_t` = selector query for this decode step

2. **Live slot candidate set**
	- The legal candidate set is:

	$$
	L = \{\, s \mid s \in [S, V) \;\land\; valid\_mask[s] = 1 \;\}
	$$

	- The selector may choose only from `L`.
	- It must not see invalid slots, dead slots, out-of-range slots, or slots from other rows.

3. **Slot-state representations**
	- For each `s in L`, first assemble a deterministic raw slot-feature record from row-local runtime state.
	- Suggested fixed form:
		- `slot_features[s] = { slot_id, numeric_value, valid_bit, recent_write_bit, usage_scalar, optional domain_id, optional provenance_id, optional state_flags }`
	- Ownership boundary:
		- `DecodeTimeNumPolicy.hpp/.cu` constructs `L` and assembles ordered `slot_features[s]` for `s in L` from row-local `ExecutionMemory` plus compiled provenance/runtime metadata only.
		- `DecodeTimeSlotSelectorLayer` consumes those fixed features, applies all learned embeddings / learned slot-state encoding, and produces `slot_repr[s]`.
	- Minimum fixed contents before learned encoding:
		- slot id
		- current numeric value
		- valid/live bit
		- last-write / usage features
		- provenance/domain identifiers where available
	- Suggested learned form: `slot_repr[s] = DecodeTimeSlotSelectorLayer::encodeSlotFeatures(slot_features[s])`
	- Slot/domain/provenance information enters fixed feature assembly as ids/scalars; embeddings for those ids, when used, are owned and applied only by `DecodeTimeSlotSelectorLayer`.
	- The selector chooses slot identity from these learned representations, not from raw numeric values alone.

4. **Optional selector policy inputs**
	- Allowed:
		- type masks if later multiple slot types exist
		- phase/mode flags for generation vs debug/validation
		- explicit null-selection priors or threshold configuration
	- Forbidden:
		- nearest rendered number shortcuts
		- text-span proximity shortcuts
		- `latest write` shortcuts
		- `first valid` shortcuts

#### Concrete selector owner module and tensor inventory

For this cutover, trainable decode-time selector state has **one** owner module only:

- `resources/models/GRIM-text/Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp/.cu` **(new)**
- implemented as `DecodeTimeSlotSelectorLayer`
- model-owned through `LanguageModel`

`DecodeTimeNumPolicy.hpp/.cu` does **not** own trainable selector tensors. It consumes scores/logits from `DecodeTimeSlotSelectorLayer` and applies the policy decision over $\{ NULL \} \cup L$.

`ExecutionBlockLayer`, `Common/grim_language_model_gpu.cu`, and `training/Inference_GPU.cu` may call the selector, but they may not own selector weights, null-option parameters, or learned ambiguity state.

The required minimal trainable selector tensor set for the cutover pointer-selector baseline is:

- `W_q_select` — query projection from decode hidden state to selector space
- `W_k_select` — key projection from slot-state vector to selector space
- `null_key_select` — learnable key/vector representing the explicit `NULL` choice
- `null_logit_bias` — learnable scalar/logit bias for the explicit `NULL` choice

Optional selector tensors are allowed only when explicitly adopted by the architecture, and if present they must live in the same owner module and checkpoint path:

- `E_slot_select` — slot-id embedding table
- `E_slot_domain_select` / `E_slot_provenance_select` — optional domain/provenance embedding tables
- `ambiguity_margin_param` — learned scalar ambiguity threshold; if ambiguity margin is fixed instead, it remains config-only and is **not** serialized as a tensor
- `W_score_*` / `b_score_*` — explicit richer pairwise scorer MLP weights if the architecture uses MLP scoring instead of the minimal pointer-selector baseline
- `W_slot_state_encode_*` / `b_slot_state_encode_*` — any learned slot-state encoder tensors beyond the fixed baseline slot feature assembly

No other learned selector tensors are allowed to appear “incidentally” in other modules. If a new selector-side learned quantity is added later, it must be declared here as part of the same owner-module contract.

The slot-state construction boundary is therefore explicit:

- `DecodeTimeNumPolicy.hpp/.cu` owns **fixed, non-trainable** assembly of ordered `slot_features[s]` for `s in L`
- `DecodeTimeSlotSelectorLayer` owns **all learned** slot-state encoding, embeddings, query/key projections, `NULL` representation, and score production

`slot_repr[s]` is not an independently-owned artifact elsewhere in the model; it exists only as the selector layer's learned encoding of fixed row-local slot features.

#### Selector / policy score interface

`DecodeTimeSlotSelectorLayer` returns a **single ordered score/logit vector over $\{ NULL \} \cup L$**.

Required output ordering for the cutover contract:

- index `0` = explicit `NULL` option
- indices `1..|L|` = candidates corresponding to the row-local ordered members of `L`

`DecodeTimeNumPolicy.hpp/.cu` consumes that complete score vector. It does **not** inject, append, or separately score the `NULL` option after the selector runs.

#### Outputs

The selector must return a structured result, not a vague score.

Suggested runtime contract:

```cpp
enum class SlotSelectionStatus : uint8_t {
    Selected,
    Null,
    Ambiguous
};

struct SlotSelectionResult {
    SlotSelectionStatus status;
    int32_t selected_slot;  // valid only when status == Selected
    float confidence;
};
```

The real choice set is:

$$
\{ NULL \} \cup L
$$

not just `L`.

#### Required invariants

- `selected_slot` must be in `L` when `status == Selected`.
- `Selected` means exactly one legal slot has been resolved.
- `Null` means no slot is resolved.
- `Ambiguous` means legal candidates exist but no unique slot was resolved.
- The selector must never output an out-of-range slot.
- The selector must never output a dead slot.
- The selector must never output more than one slot.
- The selector must never silently coerce ambiguity into a guessed slot.

#### Decode-time behavior

After selector output:

- **Case 1: `Selected`**
	- `<NUM>` remains admissible.
	- If `<NUM>` is sampled:
		- `token_slot_id = selected_slot`
		- `token_numeric_value = execution_memory.values[selected_slot]`
- **Case 2: `Null`**
	- `<NUM>` is masked before sampling.
- **Case 3: `Ambiguous`**
	- generation path: mask `<NUM>`
	- validation/debug/deterministic inspection path: fail hard

#### Selector scoring rule

The selector should be a masked pointer over live slots.

Clean form:

- `key[s] = W_k * slot_repr[s]`
- `score[s] = q_t · key[s]`

Allowed richer form:

- `score[s] = MLP([q_t ; key[s] ; q_t * key[s]])`

After scoring, apply masking so only `L` is legal, with `NULL` represented as a first-class selectable option.

#### Null-selection semantics

`Null` must be a first-class output, not an accident.

That means:

- `Null` is not “all scores were low and we guessed no”
- `Null` is not merely “no live slots exist”
- `Null` is an explicit legal selector outcome

Reasons:

- many timesteps do not refer to any slot
- live slots existing does not mean `<NUM>` should be legal now
- explicit `Null` keeps the system debuggable

#### Ambiguity semantics

The selector must use an explicit confidence or margin gate.

One acceptable operational rule is:

- `Selected` iff `argmax(score)` exists and `top1 - top2 >= selection_margin`
- otherwise `Ambiguous`

The plan does not require that exact formula, but it does require an explicit ambiguity rule. “Pick top-1 anyway” is forbidden.

#### Training contract

The selector requires direct supervision.

For each supervised decode timestep, the training target must be exactly one of:

- `NULL`
- one legal slot id in `L`

That means the data path must carry slot-reference supervision for bound numeric output positions.

Training loss must be:

- masked cross-entropy over `\{ NULL \} \cup L`

and must **not** be:

- scalar regression loss
- indirect learning through token loss alone

Reason: the selector is learning reference resolution, not numeric value prediction.

#### Selector supervision alignment contract

`slot_selection_targets` must define selector supervision by **dense decode-position alignment only**.

Each supervised decode position must carry **exactly one** of:

- one legal slot id
- `NULL`
- `IGNORE`

`IGNORE` excludes that position from selector loss and must **not** be conflated with `NULL`.

Semantics:

- `NULL` = the correct selector outcome for that decode position is explicit null selection
- `IGNORE` = this decode position does not participate in selector loss

Required encoding:

1. **Dense token-position-aligned encoding**
	- `slot_selection_targets.size()` matches the decode-position axis it supervises
	- each index corresponds to exactly one decode position
	- every supervised decode position stores one of: legal slot id / `NULL` / `IGNORE`
	- BOS/EOS insertion and padding must remap this metadata exactly like every other position-sensitive decode-aligned artifact
	- newly inserted BOS/EOS/pad positions must receive `IGNORE` unless explicitly supervised otherwise

Suggested dense storage shape:

```cpp
enum class SlotSelectionTargetKind : uint8_t {
    Slot,
    Null,
    Ignore
};

struct SlotSelectionTarget {
    SlotSelectionTargetKind kind;
    int32_t slot_id;                 // valid only when kind == Slot
};
```

Sparse selector-supervision storage is forbidden for this cutover.

If sparse alignment is ever reconsidered later, it must come with measured justification, an explicit replacement contract, and updated serialization/validation/remap rules. Until then, dense decode-position alignment is the only legal representation.

#### Failure conditions

Fail hard if:

- `status == Selected` but `selected_slot not in L`
- `<NUM>` is emitted without prior `Selected`
- decode path tries to bind `<NUM>` with `slot_id = -1`
- selector receives batch-global rather than row-local candidates
- selector output is missing in a context where `<NUM>` is being evaluated
- selector falls back to a heuristic policy forbidden by this contract
- selector supervision conflates `IGNORE` with `NULL`
- any sparse selector-supervision representation or API is encountered

---

## Non-negotiable architectural decisions

1. **Canonical structured execution record is the single source of truth.**  
	`token_exec_slots` and `teacher_steps` are derived projections, not peer authorities.

2. **ExecutionBlock is explicit, not implicit.**  
	A numeric atom is **not** automatically a register operand.

3. **Training rows are execution-active only when their compiled structured-execution payload marks them active.**  
	For supervised training rows in this cutover, non-empty `teacher_steps` remains a required payload invariant, but it is not the activation source.

4. **Plain corpus numbers stay plain.**  
	They keep `token_to_slot_map = -1` and must never be rejected for that.

5. **Execution-active rows must be fully specified.**  
	No missing slots, no inferred slots, no silent bootstrap skip, no “best effort” execution.

6. **There is one validation contract.**  
	Training and validation must call the same execution metadata validator.

7. **Execution is row-local.**  
	No per-row `ExecutionMemory` may inspect atoms from other batch rows.

8. **Concept-block debug redundancy is removed.**  
	The trailing `__SLOTS__` text hack is deleted after the structured sequence builder exists.

9. **Old GRMT data is invalid after cutover.**  
	Bump GRMT format version and require regeneration. Do not support legacy GRMT v10 semantics.

10. **Decode-time `<NUM>` is architecturally incomplete without explicit slot selection.**  
	`<NUM>` may be sampled only when an explicit slot-selection mechanism resolves exactly one legal live slot; otherwise `<NUM>` is masked. No heuristic stand-ins are allowed.

11. **Decode-time slot selection is reference resolution, not number prediction.**  
	The selector chooses one legal live slot, `Null`, or `Ambiguous`; it does not regress numeric values and it does not infer slot identity after token sampling.

12. **Execution / selector learned parameters introduced by this cutover are tensor-backed model state only.**  
	No raw host-side learnable floats, vectors, or config-owned trainable values are allowed.

13. **Trainable execution / selector tensors must participate in the same optimizer contract as the rest of the model.**  
	They are surfaced through `ParameterGroup`, not hidden behind side structures.

14. **`BatchPayload` is the only batch-time transport for compiled execution metadata.**  
	Learned weights, optimizer state, and checkpoint blobs never travel there.

15. **`TensorContract` owns tensor/autograd/layout/`ParameterGroup` semantics; `TensorConversion` owns only layout conversion between declared tensor layouts.**  
	Neither may become an alternate semantic metadata layer.

16. **Checkpoint persistence for learned execution / selector tensors goes through `grim_model_serialization.cu` + `Layers/Serialization/*` only.**  
	No sidecars, no loader reconstruction, no compatibility shims.

17. **Decode-time selector tensors have one owner module only: `DecodeTimeSlotSelectorLayer`.**  
	They do not live in `ExecutionBlockLayer`, `DecodeTimeNumPolicy`, inference-only orchestration, or loose `LanguageModel` core fields.

18. **If ambiguity thresholding is learned, that threshold is a selector tensor and lives in the selector owner module.**  
	If it is fixed, it remains config-only and is not checkpointed as a tensor.

---

## End-state contract

### Non-execution training row

- `CompiledStructuredExecutionPayload.execution_active == false` (or the equivalent serialized/transported activation bit)
- `teacher_steps.empty() == true`
- `compiled_bootstrap_bindings.empty() == true`
- every `token_exec_slots[pos] == -1`
- numeric atoms are allowed
- ExecutionBlock is not entered for that row

### Execution-active training row

- `CompiledStructuredExecutionPayload.execution_active == true` (or the equivalent serialized/transported activation bit)
- for supervised training rows in this cutover, `teacher_steps.empty() == false`
- `bootstrap_bindings.empty() == false` at builder time, therefore compiled rows with active execution payload must carry at least one compiled bootstrap binding
- runtime `D_row` is reconstructed as the union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`; configured slot ranges remain outer bounds only
- every required state-bearing token position `pos in R` has `token_exec_slots[pos] in [num_scratch_slots, num_slots)`
- every non-state token has `token_exec_slots[pos] == -1`
- `teacher_steps.size()` equals the row's compiled execution-step count; no implicit coupling to a global `execution_block_num_steps` is allowed unless fixed-step execution is made an explicit architecture rule and enforced as such
- all `TeacherStep` slot indices are in range
- every bootstrap/read/write slot referenced by the row belongs to reconstructed `D_row`
- teacher-step write targets may be first-written later, but they must already belong to reconstructed `D_row`; no step may invent or rebind slot identity
- row must fit in one sequence window; no sliding-window fragmentation
- ExecutionBlock is entered for that row only

Where `R` is the builder-compiled set of required state-bearing token positions defined above; it is **not** inferred by scanning all `<NUM>` tokens.

### Inference / generation contract

- Prompt-side `token_to_slot_map` stays explicit
- Decode-time `<NUM>` is admitted **only** when an explicit decode-time slot-selection mechanism resolves exactly one legal slot from the set of valid live slots in persistent execution memory
- If admitted, `<NUM>` binds to exactly that selected slot and its current value
- If no legal slot is resolved, `<NUM>` is masked from sampling before sampling occurs
- If multiple live slots exist and the policy cannot choose one, generation masks `<NUM>` instead of guessing
- No automatic slot inference from sampled text
- No post-sample fallback to `slot_id = -1`
- No last-write-only binding rule

---

## Immediate file-size reality check

`execution_block_GPU.cu` is currently carrying too much surface area for one layer implementation.

A direct audit of the current file shows:

- 60 `__global__` kernels in one compilation unit,
- at least two kernels with no launch site in the file: `kernelSliceColumns`, `kernelFourOpMixForward`,
- unused public layer surface: `encodeState()` and `lastDivClampCount()` currently appear as header/docs/definition surface, not as live runtime call paths,
- unused `executeStep(...)` inputs: `expected_read_v1`, `expected_read_v2`,
- `ExecutionBlockConfig` currently carries several fields that do not participate in `execution_block_GPU.cu` behavior at all (`execution_block_layer`, `memory_slot_bias`, `temp_start`, `temp_end`, `temp_schedule`, `entropy_weight`, `diag_logging`, `exec_gate_warmup_steps`, `causal_w1_transition`, `cublas_handle`; `deterministic` is not used for layer control flow here either).

That means the cutover must start by deleting stale layer-local code and evicting orchestration-only knobs from the layer boundary. Do this before adding new metadata or validator complexity.

Three existing compensating runtime behaviors are bugs, not acceptable bridges, and must be removed rather than preserved during the split:

- silent execution skip when no value slots were bootstrapped,
- old decode-time `<NUM>` emission fallback with invalid slot/value binding,
- batch-global atom context fed into per-row execution.

The plan assumes **zero fallback logic** for those paths.

---

## File separation of concerns

The split must create **layers with one-way ownership**, not just smaller files. Semantic definitions, compiled artifacts, transport, validation, runtime execution, and orchestration must stay separate.

| File / Module | Owns after refactor | Must not own |
|---|---|---|
| `resources/models/GRIM-text/Shared/Execution/ExecutionMetadata.hpp` **(new)** | Canonical semantic/compiled execution data types only: `StructuredExecutionRecord`, `CompiledStructuredExecutionPayload`, `BootstrapLiteralBinding`, `CompiledBootstrapBinding`, `TeacherStep`, `SlotSelectionTarget`, `D_row`-related types/invariants | JSON parsing, tokenization, padding/remap, validation logic, CUDA/autograd code, GRMT IO |
| `resources/models/GRIM-text/Shared/Execution/DecodeTimeNumPolicy.hpp/.cu` **(new)** | Decode-time `<NUM>` policy only: selector result types, legal candidate-set construction, fixed row-local `slot_features[s]` assembly, null/ambiguity rules, narrow policy interface for mask-or-bind decisions using scores from the selector layer | Sampler state ownership, sampler mask application, `ExecutionMemory` allocation/lifecycle, GRMT IO, training batching, trainable selector tensors, learned slot-state encoding |
| `resources/models/GRIM-text/Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp/.cu` **(new)** | Sole owner of trainable decode-time selector tensors, learned slot-state encoding, and selector forward pass returning ordered scores over $\{ NULL \} \cup L$; required baseline tensors are `W_q_select`, `W_k_select`, `null_key_select`, `null_logit_bias` | Sampler masking policy, candidate-set construction, `ExecutionMemory` ownership, GRMT row metadata, heuristic fallback policy, checkpoint schema decisions outside the serialization layer |
| `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.hpp/.cu` **(new)** | Canonical build from structured concept source into `TrainingSequence` plus compiled execution metadata | Batch padding/flattening, runtime validation, GPU execution, sampler policy |
| `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu` | Dataset/cache orchestration, tokenizer training, GRMT read/write orchestration, invokes concept builder | Ad-hoc execution semantics, duplicated validator rules, per-row execution runtime |
| `resources/models/GRIM-text/training/training_data_loader.hpp` | Serialized `TrainingSequence` schema and loader/storage helpers for compiled artifacts, including explicit compiled payload activation state | Semantic derivation rules, runtime validation policy, GPU execution |
| `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp/.cu` | Padded/remapped batch transport of already-compiled row metadata into batch form, including explicit execution-payload activation state | Deriving execution semantics, inferring `R`/`D_row`, duplicate validation logic, GPU execution |
| `resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp/.cu` | Cross-layer tensor contract only: `Tensor`, `TensorShape`, autograd behavior, layout validation, `ParameterGroup` semantics for learned state | Execution metadata semantics, batch transport, checkpoint schema selection, loader/builder policy |
| `resources/models/GRIM-text/Shared/TensorConversion/TensorConversion.hpp/.cu` | Pure layout conversion kernels between already-declared tensor layouts (`BHSD`, `BSHD`, `BSM`, fused QKV forms) | Learned-parameter ownership, semantic reinterpretation, checkpoint IO, hidden host/device shadow state |
| `resources/models/GRIM-text/Shared/Execution/ExecutionPayloadValidation.hpp/.cu` **(new)** | Host-side semantic validation of compiled/batched execution payloads | Padding/remap, H2D, file IO, source JSON reconstruction, GPU execution |
| `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu` | Validation-path orchestration: call shared validator, H2D, forward/loss path | Owning execution validation rules, selector policy internals, metadata derivation |
| `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` | Training-path orchestration: call shared validator, build row-local views, invoke ExecutionBlock | Owning execution validation rules, metadata derivation, private ExecutionBlock internals |
| `resources/models/GRIM-text/training/Inference_GPU.cu` | Inference-side row-local atom hookup and ExecutionBlock orchestration | Fixed slot-feature assembly rules, selector scoring/policy internals, trainable selector tensor ownership, null-atom fallback, duplicated validation rules |
| `resources/models/GRIM-text/GRIM/grim_language_model_cuda.hpp` | Public model declaration surface for tensor-owned modules, including model-owned `DecodeTimeSlotSelectorLayer`, plus `parameterGroups()`, `buildParameterGroups()`, `save()`, and `load()` entrypoints | Schema-specific checkpoint packing, batch metadata derivation, tensor-conversion internals, loose selector tensors outside a dedicated owner module |
| `resources/models/GRIM-text/Common/grim_language_model_gpu.cu` | Sampler integration: call `DecodeTimeNumPolicy`, apply `<NUM>` mask/binding result to generation flow | Selector scoring, candidate construction, fixed slot-feature assembly, `ExecutionMemory` ownership, heuristic fallback policy |
| `resources/models/GRIM-text/Common/grim_model_serialization.cu` | Single model-checkpoint bridge from model-owned tensors to `SerializationSaveRequest` / `SerializationLoadRequest` | GRMT row metadata, sidecar persistence for cutover-added tensors, heuristic reconstruction of missing weights |
| `resources/models/GRIM-text/Layers/Serialization/Serialization_*.{hpp,cu}` | Schema-specific checkpoint request/view/save/load/validate path for model tensors and capability gates | Runtime layer ownership, `ParameterGroup` construction, batch payload semantics, semantic execution metadata |
| `resources/models/GRIM-text/Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp/.cu` | Atom detection plus row-local atom view extraction support | Row activation policy, slot validation semantics, execution-program logic |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp` | Only the public ExecutionBlock API: public config, public memory/diagnostic structs, `ExecutionBlockLayer` declaration | Private kernels/helpers, `TeacherStep`/`BatchPayload` types, builder/validator contracts |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp` **(new, private)** | Private stream-local view types, stage IDs, macros, and internal helper declarations shared only inside the split ExecutionBlock implementation | Public API surface, training/data-loader types, validator or selector policy types |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu` | Public entrypoints and orchestration between memory-stream and data-stream internals | Monolithic kernel inventory, semantic validation, batch-global data assumptions |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu/.hpp` **(new)** | `ExecutionMemory` ownership/lifecycle, bootstrap, slot read/write, valid-mask/recent-write/usage maintenance, memory-side helpers | Token-context reduction, trace encoding, `TeacherStep` semantics, `BatchPayload`/GRMT knowledge |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu/.hpp` **(new)** | Token/data-stream compute: context reduction, trace encoding, arg/op/write logits, result decode, result injection into `H`, step-local token-side assembly | `ExecutionMemory` ownership/lifecycle, metadata parsing, semantic validation |
| `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu` | BOS/EOS insertion, padding/remap, sequence splitting policy, fragmentation rejection | Deriving new execution semantics |
| `resources/models/GRIM-text/Common/grim_model_serialization_version.hpp` | Checkpoint model version + GRMT format version constants and explicit incompatibility gates | Migration heuristics, compatibility shims |

### Hard boundary rules

1. `ExecutionMetadata.hpp` is the **only** cross-layer definition site for semantic execution types.
2. `ExecutionPayloadValidation.hpp/.cu` is the **only** semantic execution validator. Other files call it; they do not clone it.
3. `BatchPayload.hpp/.cu` is transport only. It may carry, pad, and remap compiled metadata, but it may not derive or reinterpret execution semantics.
4. `DecodeTimeNumPolicy.hpp/.cu` owns candidate construction and null/ambiguity/mask-or-bind policy over selector outputs. Callers consume the result; they do not re-implement the policy.
5. `execution_block_internal.hpp` is private to `Layers/ExecutionBlock/` and must not be included from outside that folder.
6. `execution_block_GPU.hpp` is the only public include path for non-ExecutionBlock code. The split stream files are not public extension points.
7. The split ExecutionBlock files must communicate through narrow internal view/POD types declared in `execution_block_internal.hpp`; they must not include batching, loader, or validator headers to reach across layers.
8. No file may both derive execution metadata and validate it.
9. No file may both validate execution metadata and execute the runtime program.
10. Configured slot ranges, GRMT fields, and batch transport are all downstream artifacts; none of them may redefine semantic truth owned by `ExecutionMetadata.hpp`.
11. `TensorContract_GPU.hpp/.cu` is the only cross-layer contract for learned tensor shape/layout/autograd/`ParameterGroup` semantics.
12. `TensorConversion.hpp/.cu` only converts between layouts already declared by `TensorContract`; it may not own trainable state or invent semantic meaning.
13. Every learned execution / selector parameter introduced by this cutover must exist as a `GRIM::Tensor` owned by model/layer code. No host-only source of truth is allowed.
14. Every trainable execution / selector tensor introduced by this cutover must be surfaced through `LanguageModel::buildParameterGroups()` / `parameterGroups()` so optimizer, grad diagnostics, and checkpoint save/load see the same object.
15. `BatchPayload` may carry per-row numeric values, slot ids, and supervision metadata only. It must never carry learned weights, Adam moments, or checkpoint blobs.
16. `grim_model_serialization.cu` + `Layers/Serialization/*` are the only checkpoint persistence path for learned execution / selector tensors; loaders fail if required tensors are missing or shape-mismatched.
17. `DecodeTimeSlotSelectorLayer` is the only legal owner of trainable decode-time selector tensors (`W_q_select`, `W_k_select`, `null_key_select`, `null_logit_bias`, plus any explicit optional selector tensors).
18. `DecodeTimeNumPolicy.hpp/.cu` owns no trainable selector state; it consumes selector outputs and applies policy only.
19. `ExecutionBlockLayer`, `Common/grim_language_model_gpu.cu`, and `training/Inference_GPU.cu` may not keep duplicate selector weights, shadow null parameters, or learned ambiguity tensors.
20. If a richer selector scorer or learned slot-state encoder is introduced later, its tensors must live in `DecodeTimeSlotSelectorLayer`; do not spread them across execution, inference, or policy files.
21. `DecodeTimeNumPolicy.hpp/.cu` is the only place that may assemble fixed ordered `slot_features[s]` for `s in L` from row-local runtime/provenance state; that assembly is deterministic and non-trainable.
22. `DecodeTimeSlotSelectorLayer` is the only place that may apply learned slot-state encoding/embeddings/projections, and it must return the complete ordered score vector over $\{ NULL \} \cup L$; policy must not inject `NULL` after the fact.
23. Compiled payload activation state is the only legal runtime/batch execution-active signal; code must not infer row activity solely from `teacher_steps.size()`.
24. For this cutover, runtime `D_row` is reconstructed exactly as the union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`; no other runtime source may invent the row-local slot domain.

## Chronological workstream order

Treat the workstreams below as **gated phases**, not a grab bag. The workstream numbers remain stable identifiers; the table below is the authoritative execution order. Do not begin the next workstream until the current workstream satisfies its completion criteria.

This table controls implementation order. The codoc and flow artifacts must track status transitions through this order as the refactor advances.

| Execution order | Workstream | Why it must finish first |
|---|---|---|
| 1 | Workstream 0 — execution_block file deflation before semantic cutover | Deletes dead layer surface and locks the ExecutionBlock split boundary before semantic refactor work starts. |
| 2 | Workstream 1 — canonical structured execution source-of-truth model | Establishes the semantic types and compiled payload contract that every downstream stage consumes. |
| 3 | Workstream 2 — canonical structured sequence builder replaces `__SLOTS__` | The builder must emit the canonical compiled payload before storage, remap, or validation can be trusted. |
| 4 | Workstream 3 — GRMT format cutover | Freezes the on-disk compiled payload only after the canonical builder and metadata contract exist. |
| 5 | Workstream 5 — Phase1 sequence handling rules | BOS/EOS/padding/remap semantics must be fixed before the shared validator can enforce post-Phase1 alignment invariants. |
| 6 | Workstream 4 — single shared execution payload validator | The validator becomes the common gate before runtime orchestration changes rely on fail-fast metadata checks. |
| 7 | Workstream 6 — row-local execution orchestration | Runtime execution can become row-local only after the payload contract and validator are stable. |
| 8 | Workstream 7 — delete silent execution skips | Fail-loud execution behavior belongs after row-local orchestration and validator-backed activation checks exist. |
| 9 | Workstream 8 — align validation path and training path | Training and validation should be unified only after both use the same validator and execution gating model. |
| 10 | Workstream 3A — tensor-backed learned parameter and checkpoint ownership | Selector/runtime learning state should be formalized only once the data/runtime contract is already stable. |
| 11 | Workstream 9 — inference and generation alignment | Decode-time bind-or-mask behavior depends on selector ownership, compiled metadata, and row-local runtime state. |
| 12 | Workstream 10 — tests that lock the contract in place | Tests close the cutover only after every behavior they assert is actually implemented. |

---

## Workstream 0 — execution_block file deflation before semantic cutover

**Progress note (2026-03-29):** Workstream 0 is complete. The public `ExecutionBlockLayer` surface has been trimmed, the implementation is split into a thin coordinator plus private memory/data stream files, `executeStep()` now enforces explicit row-local atom views, the null-atom decode path is gone, and the old decode-time `<NUM>` last-write fallback has been deleted in favor of pre-sampling masking until the explicit selector workstream lands. Layer docs/tests are now aligned with the surviving surface. Full CUDA build validation remains locally blocked by the missing toolchain/header environment, but the Workstream 0 architecture gate is closed.

### Files

- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp`
- **New:** `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp`
- **New:** `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.hpp`
- **New:** `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
- **New:** `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.hpp`
- **New:** `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu`
- Update: `resources/models/GRIM-text/training/TrainingOps.cu`
- Update: `resources/models/GRIM-text/Layers/InitInferenceState/InitinferenceState.cu`
- Update: `resources/models/GRIM-text/GRIM/grim_language_model_cuda.hpp`
- Update: `resources/models/GRIM-text/Tests/ExecutionBlockTest.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/DOCUMENTATION.md`

**Suggested tools:** `search_subagent`, `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Delete dead kernels with no launch sites.
	- `kernelSliceColumns`
	- `kernelFourOpMixForward`
2. Delete unused public layer APIs that have no live caller.
	- `ExecutionBlockLayer::encodeState()`
	- `ExecutionBlockLayer::lastDivClampCount()`
3. Delete unused `executeStep(...)` inputs.
	- remove `expected_read_v1`
	- remove `expected_read_v2`
	- remove any comments/docs/tests that still pretend operand-teacher inputs are wired
4. Treat the three compensating runtime behaviors as hard blockers during cleanup/split.
	- Do not preserve silent execution skip behind a renamed helper.
	- Do not preserve batch-global atom arrays plus internal row filtering as an interim adapter.
	- Do not preserve the old decode-time `<NUM>` invalid slot/value emission branch until later “cleanup”.
5. Shrink `ExecutionBlockConfig` so it contains only layer-owned behavior.
	- Delete globally dead config surface that no live layer code reads: `memory_slot_bias`, `diag_logging`, `deterministic`, `cublas_handle`
	- Move orchestration-owned knobs out of `ExecutionBlockConfig` and back to the training/inference orchestration boundary: `execution_block_layer`, `temp_start`, `temp_end`, `temp_schedule`, `entropy_weight`, `exec_gate_warmup_steps`, `causal_w1_transition`
	- Keep only fields that `execution_block_GPU.cu` actually consumes
6. Rewrite `ExecutionBlockTest.cu` so it validates only the surviving layer surface.
	- remove default-value assertions for deleted knobs
	- add assertions for the trimmed API/config boundary instead of preserving legacy field count
7. Rewrite `Layers/ExecutionBlock/DOCUMENTATION.md` to stop documenting deleted kernels, deleted APIs, deleted/moved config knobs, and any compensating runtime behavior as if it were intentional.
8. Split the surviving live implementation **inside the same `Layers/ExecutionBlock/` folder** instead of keeping one monolith.
	- `execution_block_GPU.hpp` stays as the minimal public façade.
	- `execution_block_GPU.cu` becomes a thin orchestration file only.
	- `execution_block_memory_stream_GPU.*` owns the register-memory stream.
	- `execution_block_data_stream_GPU.*` owns the token/data stream.
	- `execution_block_internal.hpp` holds private shared internals needed by both streams.
9. Define the split boundary explicitly.
	- **Memory stream** = `ExecutionMemory` allocate/clear, slot bootstrap, slot gather/read/write, `valid_mask`, `recent_write_mask`, `usage`, memory-side addressing helpers, cross-attention read over memory.
	- **Data stream** = context reduction over `H`, trace-state encoding, arg/op logits, op-result assembly, result embedding, result injection into `H`, step-record assembly.
	- `executeStep()` remains the top-level coordinator, but its body must mostly call stream-local helpers instead of embedding every kernel path inline.
10. Keep shared custom `GradFn` code in the stream file that owns the forward path it differentiates.
	- If a `GradFn` naturally belongs to token/data flow, keep it in `execution_block_data_stream_GPU.cu`.
	- If a `GradFn` naturally belongs to register-memory flow, keep it in `execution_block_memory_stream_GPU.cu`.
	- Do **not** create a third junk-drawer `.cu` just to move clutter sideways.
11. Add one mechanical rule for this cutover: every helper or kernel that remains in the ExecutionBlock implementation must have a direct live call path in the same architecture. If not, delete it.
12. Define narrow internal interface/view types in `execution_block_internal.hpp`.
	- Stream files must exchange row-local/token-local/memory-local views through those internal types instead of reaching into batching, loader, or validator types.
	- `execution_block_internal.hpp` is the only place shared private stream-facing structs/macros are allowed to live.
13. Enforce include boundaries mechanically.
	- `execution_block_internal.hpp` may be included only by `execution_block_GPU.cu`, `execution_block_memory_stream_GPU.*`, and `execution_block_data_stream_GPU.*`.
	- Non-ExecutionBlock code must include only `execution_block_GPU.hpp`.
	- The split stream files must not include `BatchPayload.hpp`, `training_data_loader.hpp`, `ExecutionPayloadValidation.hpp`, or builder headers.

### No-backwards-compatibility rule

- Do **not** keep dead kernels for “future use.”
- Do **not** keep docs-only or test-only public APIs.
- Do **not** keep config mirrors inside `ExecutionBlockConfig` when the layer never reads them.
- Do **not** preserve default-value tests for deleted knobs.
- Do **not** split the file first and carry dead code into multiple smaller files.
- Do **not** let memory-stream and data-stream responsibilities collapse back into one new private junk drawer.
- Do **not** move orchestration policy into the stream files.
- Do **not** let the split stream files include batching, loader, validator, or builder headers to reach across layers.
- Do **not** expose `execution_block_internal.hpp` outside `Layers/ExecutionBlock/`.
- Do **not** preserve silent execution skip as an interim safety valve.
- Do **not** preserve batch-global atom context feeding per-row execution behind internal filtering logic.
- Do **not** preserve the old decode-time `<NUM>` invalid slot/value emission branch while waiting for a later inference cleanup.

### Completion criteria

- `execution_block_GPU.cu` is reduced to a thin public coordinator and the surviving live implementation is split cleanly across the memory-stream and data-stream files.
- Deleted kernels, deleted APIs, deleted `executeStep(...)` inputs, and deleted/moved config knobs are absent from code, docs, and tests.
- `execution_block_internal.hpp` remains private to `Layers/ExecutionBlock/`, and the split stream files respect the include-boundary rules.
- No renamed or hidden version of silent execution skip, batch-global atom adaptation, or invalid-slot/value decode-time `<NUM>` fallback remains in the ExecutionBlock layer.

---

## Workstream 1 — canonical structured execution source-of-truth model

### Files

- **New:** `resources/models/GRIM-text/Shared/Execution/ExecutionMetadata.hpp`
- Update: `resources/models/GRIM-text/training/training_data_loader.hpp`
- Update: `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`

**Suggested tools:** `search_subagent`, `read_file`, `apply_patch`, `get_errors`, `get_changed_files`

### Operations

1. Define `StructuredExecutionRecord` in `ExecutionMetadata.hpp` as the single semantic source of truth for execution-active rows, and define `CompiledStructuredExecutionPayload` as the compiled runtime/supervision payload derived from it.
	- initial literals / slot identities
	- explicit row-local slot domain `D_row`
	- `bootstrap_bindings[]` that define the only literals allowed to seed registers
	- bootstrap uniqueness rules: one binding per `slot_id`, one binding per `token_pos`
	- stable row-local slot identity across all steps
	- legal teacher-step write domain for in-range first-writes vs forbidden dynamic slot creation/rebinding
	- compiled bootstrap provenance type (`CompiledBootstrapBinding`)
	- ordered execution steps
	- expected scalar results with explicit supervision/debug-only semantics
2. Move `TeacherStep` out of `BatchPayload.hpp` into the shared execution metadata header as a **derived supervision view**, not a source object.
3. Extend `TrainingSequence` to carry compiled training-time execution payload with explicit activation state:
	- `bool execution_active` (or equivalent explicit compiled-payload activation bit)
	- `std::vector<int32_t> token_exec_slots`
	- `std::vector<GRIM::Execution::TeacherStep> teacher_steps`
 	- `std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings`
	- `std::vector<GRIM::Execution::SlotSelectionTarget> slot_selection_targets` for supervised decode-time slot reference resolution (exactly one of legal slot id / `NULL` / `IGNORE` per supervised decode position)
	- runtime `D_row` is reconstructed from `compiled_bootstrap_bindings` and `teacher_steps`, not serialized separately for this cutover
4. Extend `TrainingSampleView` to expose the compiled execution metadata needed downstream:
	- `execution_active`
	- `teacher_steps`
	- `compiled_bootstrap_bindings`
	so batching/validation can consume provenance without re-reading or reconstructing it elsewhere.
5. Extend `BatchPayload` to remain the padded transport for:
	- explicit `execution_active` bit from the compiled structured-execution payload
	- `token_to_slot_map`
	- `teacher_steps`
 	- compiled bootstrap provenance per row
	- selector supervision targets for decode-time slot selection training
6. Make `TrainingSequence -> TrainingSampleView -> BatchPayload` the only batched transport path for compiled execution metadata.
	- No parallel side-channel vectors, maps, or ad-hoc structs may bypass `BatchPayload` once batch assembly begins.
7. Define the row activation rule in one place:
	- **training row is execution-active iff the compiled structured-execution payload marks it active**
	- non-empty `teacher_steps` is a supervised-training payload validity rule in this cutover, not the activation source
8. State explicitly in code comments and docs that runtime sees one compiled payload containing:
	- `execution_active` = authoritative activation bit
	- `token_exec_slots` = runtime binding projection
	- `teacher_steps` = supervision projection
	- all derived from `StructuredExecutionRecord`
9. State explicitly that `token_exec_slots` is compiled only from `bootstrap_bindings[]`, not from arbitrary numeric tokens in rendered text.
10. State explicitly that, for this cutover, runtime `D_row` is reconstructed as the union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`, and is not serialized separately in GRMT/`BatchPayload`.
11. State explicitly that execution-active rows with active compiled payload and zero `bootstrap_bindings[]` are malformed and must be rejected by the builder.
12. Define `SlotSelectionTarget` as reference-resolution supervision: target per supervised decode timestep is one of legal slot id / `NULL` / `IGNORE`, never a numeric scalar regression target.
13. Define selector supervision alignment explicitly.
	- Dense form only: target index is decode-position aligned.
	- Sparse form is forbidden unless a later measured-justification redesign explicitly replaces this policy.
	- `slot_selection_targets[i]` is the selector supervision target for decode position `i` in the row's post-Phase1 sequence representation.
	- The array length must equal that row's post-Phase1 decode-position length exactly.
	- `IGNORE` excludes selector loss and is distinct from `NULL`.

### No-backwards-compatibility rule

- Do **not** keep a second `TeacherStep` definition in `BatchPayload.hpp`.
- Do **not** author `teacher_steps` directly from ad-hoc row logic.
- Do **not** author `token_exec_slots` independently of the canonical structured record.
- Do **not** infer slot-bearing positions by scanning all numeric atoms after tokenization.
- Do **not** allow duplicate bootstrap initialization of the same `slot_id`.
- Do **not** allow two bootstrap bindings to target the same `token_pos`.
- Do **not** allow teacher steps to invent or rebind slot identity mid-row.
- Do **not** treat configured slot ranges as equivalent to `D_row`.
- Do **not** drop compiled bootstrap provenance from the serialized/training artifact if the validator depends on exact `R` membership.
- Do **not** infer execution-active status from `teacher_steps.size()` when the compiled payload activation bit is available.
- Do **not** allow supervised execution-active training rows without `teacher_steps`.
- Do **not** allow execution-active rows with active compiled payload and zero `bootstrap_bindings[]` to survive builder emission.

### Completion criteria

- `ExecutionMetadata.hpp` is the single cross-layer definition site for semantic execution metadata types, including `StructuredExecutionRecord`, `TeacherStep`, `CompiledBootstrapBinding`, and `SlotSelectionTarget`.
- `TrainingSequence`, `TrainingSampleView`, and `BatchPayload` all carry the same explicit compiled payload fields: `execution_active`, `token_exec_slots`, `teacher_steps`, `compiled_bootstrap_bindings`, and `slot_selection_targets`.
- Activation state is explicit and authoritative everywhere; no downstream contract still treats `teacher_steps.size()` as the execution-activation signal.
- Code comments and docs explicitly state that `token_exec_slots` and `teacher_steps` are paired projections of one canonical record, while runtime `D_row` is reconstructed rather than inferred from configured slot ranges.

---

## Workstream 2 — delete the `__SLOTS__` debug serialization path and replace it with canonical structured sequence building

### Files

- **New:** `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.hpp`
- **New:** `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.cu`
- Update: `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu`
- Update: `resources/models/GRIM-text/Shared/DataLoader/DataLoader.hpp`

**Suggested tools:** `search_subagent`, `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`

### Operations

1. Replace the current debug-only concept path with a canonical builder that:
	- parses concept JSON into `StructuredExecutionRecord`,
	- reads structured concept JSON,
	- emits canonical training text,
	- records structured execution metadata while building,
	- tokenizes once,
	- maps `bootstrap_bindings[]` to token positions deterministically,
	- requires each bound literal span to compile to exactly one `ATOM_NUM` token,
	- rejects duplicate bootstrap `slot_id` initialization,
	- rejects duplicate compiled `token_pos` targeting,
	- rejects execution-active rows with active compiled payload and zero `bootstrap_bindings[]`,
	- derives `token_exec_slots` and `teacher_steps` from the same `StructuredExecutionRecord`,
	- emits `compiled_bootstrap_bindings` from the same builder pass,
	- writes those compiled projections directly into `TrainingSequence`.
2. Remove the trailing `__SLOTS__` text block from concept serialization.
3. Remove slot assignment based on matching tail numeric atoms emitted as plain text.
4. Keep concept rows as structured curriculum rows, not duplicated UI/debug records embedded into training text.

### No-backwards-compatibility rule

- Delete `__SLOTS__`-driven slot recovery.
- Delete or fully replace `slotOrderFromConceptJson()` once the canonical builder lands.
- Delete or fully replace the old `teacherStepsFromConceptJson()` helper if its logic moves into the new builder.
- Do **not** allow one projection to be emitted without the other for execution-active rows.
- Do **not** support a many-token binding rule for one semantic bound literal.

### Rationale

The current expedient path is explicitly documented as temporary. This cutover removes the temporary architecture instead of preserving it.

### Completion criteria

- `ConceptExecutionSequenceBuilder` is the only builder that emits execution-active concept rows into `TrainingSequence`.
- The `__SLOTS__` tail block, tail-number slot recovery, and the superseded slot/teacher-step helper paths are deleted rather than retained beside the new builder.
- Builder-time failures exist for every structural violation the plan calls out: zero bootstrap bindings on an active row, duplicate `slot_id` initialization, duplicate compiled `token_pos` targeting, and bound literals that do not compile to exactly one `ATOM_NUM` token.
- `token_exec_slots`, `teacher_steps`, and `compiled_bootstrap_bindings` are emitted together from one builder pass and are never authored independently.

---

## Workstream 3 — GRMT format cutover

### Files

- Update: `resources/models/GRIM-text/Common/grim_model_serialization_version.hpp`
- Update: `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu`
- Update: `resources/models/GRIM-text/training/training_data_loader.hpp`

**Suggested tools:** `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Bump `GRMT_FORMAT_VERSION` from `10` to `11`.
2. Serialize explicit compiled payload activation state in GRMT.
3. Serialize compiled bootstrap provenance in GRMT beside `token_exec_slots`.
4. Serialize `teacher_steps` in GRMT beside `token_exec_slots`.
5. Serialize `slot_selection_targets` in GRMT beside `teacher_steps` for decode-time selector supervision.
6. Reject GRMT v10 unconditionally at load time.
7. Force rebuild of `.grmt` data from source JSON/cache.

### Suggested serialization order

After the existing per-sequence fields:

1. `uint8_t execution_payload_active`
2. `token_exec_slots[len]`
3. `uint32_t compiled_bootstrap_binding_count`
4. `CompiledBootstrapBinding[compiled_bootstrap_binding_count]`
5. `uint32_t teacher_step_count`
6. `TeacherStep[teacher_step_count]`
7. `uint32_t slot_selection_target_count`
8. `SlotSelectionTarget[slot_selection_target_count]`

### Storage policy

GRMT does **not** store the full semantic `StructuredExecutionRecord`.

GRMT stores the compiled runtime/supervision artifact plus enough compiled provenance to validate exact state-bearing token membership without reconstructing semantic intent:

- `token_exec_slots`
- `compiled_bootstrap_bindings`
- `teacher_steps`
- `slot_selection_targets`

GRMT stores compiled per-row metadata only. It does **not** store learned execution / selector weights, optimizer state, or tensor layout contract data; those belong to model checkpoints, not dataset payload.

For this cutover, GRMT does **not** serialize `D_row` separately. Runtime reconstructs `D_row^{runtime}` from `compiled_bootstrap_bindings` and `teacher_steps` exactly as defined above.

`slot_selection_targets` is serialized only as a dense decode-position-aligned array. Sparse selector-supervision encodings are forbidden for this cutover.

### No-backwards-compatibility rule

- No dual loader for v10/v11.
- No loader that infers execution-active status solely from `teacher_steps.size()`.
- No translation shim from v10 “slot-only” rows to v11 “slot + teacher” rows.
- No loader that fabricates selector supervision from heuristics after deserialization.
- No loader that conflates `IGNORE` with `NULL` in selector supervision.
- No validator that depends on rebuilding semantic intent from source JSON after GRMT load.
- No GRMT field or `BatchPayload` transport blob used to persist learned parameters or optimizer state.

### Completion criteria

- GRMT read/write uses the new compiled payload layout with explicit activation state, compiled bootstrap provenance, teacher steps, and dense selector supervision targets.
- GRMT v10 is rejected immediately with no translation shim, no compatibility mode, and no heuristic reconstruction path.
- Loader logic does not fabricate execution activation, bootstrap provenance, or selector supervision after deserialization.
- The required dataset/cache regeneration step is documented as part of the cutover and old GRMT artifacts are treated as invalid inputs.

---

## Workstream 3A — tensor-backed learned parameter and checkpoint ownership

**Chronology note:** Keep this workstream grouped near serialization concerns in the document, but execute it only after Workstream 8 is complete and immediately before Workstream 9 begins.

### Files

- Update: `resources/models/GRIM-text/GRIM/grim_language_model_cuda.hpp`
- **New:** `resources/models/GRIM-text/Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp`
- **New:** `resources/models/GRIM-text/Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.cu`
- Update: `resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp`
- Update: `resources/models/GRIM-text/Shared/TensorConversion/TensorConversion.hpp`
- Update: `resources/models/GRIM-text/Common/grim_model_serialization.cu`
- Update: `resources/models/GRIM-text/Common/grim_model_serialization_version.hpp`
- Update: `resources/models/GRIM-text/Layers/Serialization/Serialization_requests.hpp`
- Update: `resources/models/GRIM-text/Layers/Serialization/Serialization_validate.hpp`
- Update: `resources/models/GRIM-text/Layers/Serialization/Serialization_validate.cu`
- Update: `resources/models/GRIM-text/training/schemas/grim_transformer_model.fbs`

**Suggested tools:** `search_subagent`, `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Introduce `DecodeTimeSlotSelectorLayer` in `Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp/.cu` as the **sole owner module** for trainable decode-time selector tensors. `LanguageModel` owns exactly one instance when selector functionality is enabled and exposes stable accessors.
2. The required minimal trainable selector tensor set for the cutover pointer-selector baseline is:
	- `W_q_select`
	- `W_k_select`
	- `null_key_select`
	- `null_logit_bias`
3. Optional selector tensors are legal only when explicitly adopted by the architecture, and if present they must still live in `DecodeTimeSlotSelectorLayer` and checkpoint through the same path:
	- `E_slot_select`
	- `E_slot_domain_select` / `E_slot_provenance_select`
	- `ambiguity_margin_param` when ambiguity thresholding is learned
	- `W_score_*` / `b_score_*` for richer scorer MLPs
	- `W_slot_state_encode_*` / `b_slot_state_encode_*` for learned slot-state encoders beyond the baseline feature assembly
4. If ambiguity thresholding is fixed instead of learned, it remains a config scalar owned by policy/config code and is **not** serialized as a tensor.
5. Fixed raw `slot_features[s]` (slot id, scalar value, valid/live bits, usage/recent-write stats, optional provenance/domain ids) are assembled by `DecodeTimeNumPolicy.hpp/.cu` and are **not** trainable tensors.
6. Every selector tensor listed above must be represented as `GRIM::Tensor` with explicit `TensorContract::TensorShape`; no host-only shadow state may act as the semantic source of truth.
7. Every trainable tensor introduced by this cutover must be surfaced through `LanguageModel::buildParameterGroups()` / `parameterGroups()` so optimizer stepping, gradient diagnostics, and checkpoint save/load all reference the same object.
8. If optimizer state is needed for those tensors, it remains tensor-backed through `ParameterGroup::m_tensor` / `v_tensor`; do not create parallel raw-array ownership models.
9. `BatchPayload` remains batch metadata transport only. Training/validation/runtime consume compiled execution metadata through `BatchPayload`, but learned tensors never travel through batch payloads, GRMT rows, or config JSON.
10. `TensorContract` remains the single contract for layout, gradient presence, and parameter grouping of learned tensors. New code must not hand-roll raw pointer + shape conventions for trainable state.
11. `TensorConversion` is restricted to converting between declared tensor layouts required by kernels. It may not own trainable tensors, stash shadow copies, or reinterpret batch metadata as learned state.
12. Extend the checkpoint request/view types so every selector tensor introduced by this cutover has an explicit save/load field and, when the owning module is optional, an explicit capability requirement bit.
13. Extend `grim_model_serialization.cu` so save/load wires the exact tensors owned by `DecodeTimeSlotSelectorLayer` into `SerializationSaveRequest` / `SerializationLoadRequest`.
14. `DecodeTimeSlotSelectorLayer` returns the complete ordered score vector over $\{ NULL \} \cup L$; policy does not append or separately score `NULL` after layer execution.
15. Bump `GRIM_MODEL_VERSION` when the checkpoint tensor set changes. If the GRMT dataset payload changes too, bump `GRMT_FORMAT_VERSION` separately in the same version header; both loaders reject prior versions unconditionally.
16. Loader validation must fail if a required execution / selector tensor is missing, shape-mismatched, or absent from a checkpoint that claims the feature is enabled. No silent reinitialization, heuristic fill-in, or compatibility reconstruction is allowed.
17. This cutover may not introduce any new sidecar or ad-hoc checkpoint path for learned parameters. If the schema cannot represent a required trainable tensor, update the schema instead of adding an auxiliary file.
18. `DecodeTimeNumPolicy`, `ExecutionBlockLayer`, `Common/grim_language_model_gpu.cu`, and `training/Inference_GPU.cu` must not own duplicate selector tensors or alternate selector parameter stores.

### No-backwards-compatibility rule

- Do **not** keep learned execution / selector state as raw `float`, `std::vector<float>`, or config-only trainable values.
- Do **not** keep host mirrors as the semantic source of truth for learned tensors.
- Do **not** omit a trainable execution / selector tensor from `ParameterGroup` construction while still expecting it to learn.
- Do **not** serialize learned execution / selector tensors through sidecar files, debug dumps, or loader-time reconstruction.
- Do **not** store learned parameters or optimizer state in GRMT payloads or `BatchPayload`.
- Do **not** let `TensorConversion` become a second ownership layer for trainable state.
- Do **not** split selector tensor ownership across `LanguageModel` core fields, `ExecutionBlockLayer`, policy code, and inference-only code.
- Do **not** treat `DecodeTimeNumPolicy` as a convenient place to stash trainable selector weights just because it already owns policy logic.
- Do **not** move learned slot-state encoding into `DecodeTimeNumPolicy`; it assembles fixed features only.
- Do **not** inject a learned or separately-scored `NULL` option in policy after selector-layer scoring.

### Completion criteria

- `DecodeTimeSlotSelectorLayer` is the only owner of trainable decode-time selector tensors.
- Every required baseline selector tensor is represented as a `GRIM::Tensor` with explicit shape metadata and is surfaced through `LanguageModel::buildParameterGroups()` / `parameterGroups()`.
- Checkpoint request/load/validate paths include every required selector tensor and fail hard on missing, shape-mismatched, or capability-inconsistent state.
- No sidecar file, GRMT field, `BatchPayload` field, policy-owned shadow state, or inference-only mirror carries learned selector parameters or optimizer state.

---

## Workstream 4 — single shared execution payload validator

### Files

- **New:** `resources/models/GRIM-text/Shared/Execution/ExecutionPayloadValidation.hpp`
- **New:** `resources/models/GRIM-text/Shared/Execution/ExecutionPayloadValidation.cu`
- Update: `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`
- Update: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`

**Suggested tools:** `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

Implement one validator that enforces:

1. Non-execution row:
	- `execution_payload_active == false`
	- `teacher_steps.empty()`
	- `compiled_bootstrap_bindings.empty()`
	- all slots must be `-1`

2. Execution-active row:
	- `execution_payload_active == true`
	- for supervised training rows in this cutover, `teacher_steps.size()` equals the row's compiled execution-step count derived from the canonical structured execution record
	- `compiled_bootstrap_bindings` is non-empty
	- `D_row^{runtime}` is reconstructed as the union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`
	- every referenced slot is in range
	- compiled bootstrap bindings are injective in `token_pos`
	- compiled bootstrap bindings are injective in `slot_id`
	- every required state-bearing token position `pos in R` has a valid slot in `[S, V)`
	- at least one slot-bearing token exists for the row
	- teacher-step write targets may be non-bootstrap first-writes; validator enforces in-range slot ids here, while builder semantics enforce stable slot-domain membership / no rebinding
	- selector supervision targets, where present, are one of legal slot id / `NULL` / `IGNORE`
	- `IGNORE` is excluded from selector loss and is not treated as `NULL`

3. Batch-level consistency:
	- execution metadata dimensions match batch geometry
	- no row is “half execution-active”
	- rows with `execution_payload_active == false` must not carry active execution semantics by implication from non-empty `teacher_steps`
	- `token_exec_slots` and `teacher_steps` are mutually consistent projections of one execution row
	- `R = { pos | token_exec_slots[pos] != -1 }` matches serialized `compiled_bootstrap_bindings` exactly
	- `D_row^{runtime}` is reconstructed exactly from the row's bootstrap bindings and teacher steps; validator must not substitute `[S, V)` for `D_row`
	- for every serialized compiled binding `(token_pos, slot_id)`, `token_exec_slots[token_pos] == slot_id`
	- dense selector supervision length equals the row's post-Phase1 decode-position length exactly
	- `slot_selection_targets[i]` supervises post-Phase1 decode position `i`

### Required call sites

- `buildBatchPayload()` for structural validation
- `computeLossBatch()` before any GPU work
- `autogradTrainingStep()` before any GPU work

### No-backwards-compatibility rule

- Delete `computeLossBatch()`-only `<NUM>` slot validation logic after the shared validator is in place.
- Do **not** move semantic validator rules into `BatchPayload.hpp/.cu`, H2D helpers, or other orchestration call sites.
- Do **not** keep separate training/validation execution rules.

### Completion criteria

- One shared validator exists and is callable from `buildBatchPayload()`, `computeLossBatch()`, and `autogradTrainingStep()`.
- The validator reconstructs `R` from `compiled_bootstrap_bindings` and reconstructs `D_row` from compiled bootstrap bindings plus teacher steps rather than substituting configured slot ranges.
- Invalid execution metadata fails before GPU work begins in both training and validation paths.
- Duplicate semantic validation logic is removed from `BatchPayload`, `computeLossBatch()`, and training-only helpers.

---

## Workstream 5 — Phase1 sequence handling rules for execution-active rows

### Files

- Update: `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu`

**Suggested tools:** `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Preserve compiled structured-execution payload activation state, `teacher_steps`, and `token_exec_slots` through BOS/EOS insertion.
2. Preserve and remap `compiled_bootstrap_bindings` through BOS/EOS insertion by applying the exact token-position offset introduced by inserted special tokens.
	- If BOS shifts the sequence right by one token, every bound `token_pos` shifts right by one token.
	- If EOS is appended at the tail, existing bound `token_pos` values remain unchanged.
	- If any future special-token insertion changes prefix offsets, compiled provenance must receive the same remap or the row is invalid.
3. Preserve `teacher_steps` through padding.
4. Preserve `token_exec_slots` through padding.
	- Newly introduced pad positions must be filled with `-1`.
	- Existing pre-pad slot-bearing positions must keep their original slot ids.
5. Preserve `compiled_bootstrap_bindings` through padding.
	- Tail padding must not change existing bound `token_pos` values.
	- Padding tokens must never receive compiled bootstrap bindings.
	- Phase1 must treat compiled provenance as position-sensitive contract data, not optional debug metadata.
6. Preserve and remap `slot_selection_targets` through BOS/EOS insertion and padding.
	- Selector supervision is dense/token-position aligned and must be remapped exactly like other decode-position-sensitive metadata.
	- After Phase1 remap, `slot_selection_targets[i]` is the selector supervision target for decode position `i` in that row's post-Phase1 sequence representation.
	- After Phase1 remap, `slot_selection_targets.size()` must equal that row's post-Phase1 decode-position length exactly.
	- Newly inserted BOS/EOS/pad positions must receive `IGNORE` unless explicitly supervised otherwise.
	- Padding must not silently turn `IGNORE` into `NULL` or vice versa.
	- Phase1 must reject any sparse selector-supervision representation instead of remapping it.
7. **Reject sliding-window fragmentation for execution-active rows.**
	- If an execution-active row exceeds `max_seq_len`, throw immediately.
	- Do not split it into windows.
8. Keep non-execution rows on the current sliding-window path.

### No-backwards-compatibility rule

- No “best effort” window slicing for structured execution rows.
- No automatic remapping of `teacher_steps` after arbitrary window splitting.
- No silent dropping, partial remapping, or stale-position carryover for `compiled_bootstrap_bindings` after BOS/EOS insertion or padding.
- No selector-supervision remap that conflates `IGNORE` with `NULL`.
- No sparse selector supervision at all until there is measured justification and an explicit replacement contract.

### Completion criteria

- BOS/EOS insertion and padding preserve `token_exec_slots`, `compiled_bootstrap_bindings`, `teacher_steps`, and `slot_selection_targets` with exact position-sensitive remap semantics.
- Newly inserted BOS/EOS/pad positions receive only the legal defaults required by the contract (`-1` for slot maps and `IGNORE` for dense selector supervision unless explicitly supervised otherwise).
- Execution-active rows that do not fit in one sequence window fail immediately instead of being fragmented.
- Post-Phase1 decode-position length is the explicit alignment target for dense selector supervision and is treated as such in code comments and checks.

---

## Workstream 6 — row-local execution orchestration

### Files

- Update: `resources/models/GRIM-text/Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp`
- Update: `resources/models/GRIM-text/Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.hpp`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.hpp`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu`

**Suggested tools:** `search_subagent`, `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Add row-local atom extraction support after ScratchBlock forward.
	- Build row-local index lists from the batch-global atom position buffer as the required correctness mechanism.
	- If a future optimization later compacts row-local atoms into temporary buffers, that is a performance change only and must preserve the same row-local index semantics.
2. In `AutogradTraining.cu`, call `executeStep()` with:
	- row-local atom positions,
	- row-local `num_atoms`,
	- row-local slot-map base pointer,
	- row-local token offset and row token count.
3. Change the split ExecutionBlock contract so `executeStep()` receives only row-local atom views and row-local slot-map views.
	- Passing batch-global atom arrays into `executeStep()` and compensating with internal filtering/offset logic is forbidden.
4. In the split ExecutionBlock internals, remove any assumption that `atom_positions` spans the whole batch.
	- `execution_block_data_stream_GPU.cu` must treat atom positions as row-scoped token/data-stream inputs.
	- `execution_block_memory_stream_GPU.cu` must treat slot-validation inputs as row-scoped memory-stream checks.
5. Ensure slot validation checks only the atoms belonging to that row.
6. Keep `execution_block_GPU.cu` as a thin coordinator that wires row-local data-stream and memory-stream helpers together instead of re-embedding batch-global assumptions.

### No-backwards-compatibility rule

- Do **not** continue passing full-batch atom lists into per-row execution.
- Do **not** treat cross-row visibility as acceptable behavior.

### Completion criteria

- ScratchBlock and training orchestration produce row-local atom index lists/views and pass only row-local state into `executeStep()`.
- `ExecutionBlock` public and private interfaces accept only row-local atom views and row-local slot-map views; batch-global atom arrays are no longer part of the runtime contract.
- The split memory-stream and data-stream files enforce row-scoped validation and computation rather than compensating internally for full-batch inputs.
- Mixed-batch/runtime tests show that one row cannot inspect atoms from a neighboring row.

---

## Workstream 7 — delete silent execution skips

### Files

- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`

**Suggested tools:** `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Remove the silent `return` path in `executeStep()` that skips when `M.valid_mask` has no populated value slots.
	- During the split, move the empty-memory check into `execution_block_memory_stream_GPU.cu` and make it fail-loud there; do not preserve a silent skip behind the new file boundary.
2. Move the “should this row execute?” decision to the orchestrator (`AutogradTraining.cu`) and base it on compiled payload activation state, not `teacher_steps.size()`.
3. New behavior:
	- non-execution row: skip before calling `executeStep()`
	- execution-active row with empty memory / no valid slot bootstrap: **throw**

### No-backwards-compatibility rule

- No runtime healing.
- No “row had bad execution metadata, but we just left H untouched.”

### Completion criteria

- The orchestrator decides execution solely from compiled payload activation state.
- Non-execution rows skip before `executeStep()` is called.
- Execution-active rows with empty bootstrap memory or no valid slot initialization throw immediately from the runtime path.
- No silent early-return path remains in `executeStep()` or in any split helper that replaced it.

---

## Workstream 8 — align validation path and training path

### Files

- Update: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- Update: `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`

**Suggested tools:** `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Make `computeLossBatch()` and `autogradTrainingStep()` invoke the same shared validator.
2. Make both paths fail on the same invalid rows.
3. Keep validation-loop exception logging in `Phase2_TrainingLoop.cu`, but do not let it be the first place invalid metadata is detected.

### No-backwards-compatibility rule

- No “validation is stricter than training.”
- No “training path tolerates broken rows because bootstrap just skipped them.”

### Completion criteria

- `computeLossBatch()` and `autogradTrainingStep()` call the same shared validator and fail on the same invalid rows.
- Validation-loop logging in `Phase2_TrainingLoop.cu` remains diagnostic only and does not introduce a separate acceptance/rejection policy.
- No training-only or validation-only execution-metadata rule remains in the codebase.

---

## Workstream 9 — inference and generation alignment

### Files

- **New:** `resources/models/GRIM-text/Shared/Execution/DecodeTimeNumPolicy.hpp`
- **New:** `resources/models/GRIM-text/Shared/Execution/DecodeTimeNumPolicy.cu`
- **New:** `resources/models/GRIM-text/Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp`
- **New:** `resources/models/GRIM-text/Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.cu`
- Update: `resources/models/GRIM-text/Common/grim_language_model_gpu.cu`
- Update: `resources/models/GRIM-text/training/Inference_GPU.cu`
- Update: `resources/models/GRIM-text/GRIM/grim_language_model_cuda.hpp`

**Suggested tools:** `search_subagent`, `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Operations

1. Keep prompt slot maps explicit.
2. Delete the old decode-time `<NUM>` fallback branch atomically with the introduction of the new binding mechanism.
	- No emission with `slot_id = -1`.
	- No emission with `numeric_value = 0.0f` as a fallback payload.
3. Implement `resolveDecodeTimeNumSlotSelectionOrMask(...)` as the shared mechanism that:
	- checks `execution_block_enabled`
	- checks scratchblock generation activation
	- checks `has_inference_exec_memory`
	- builds the legal candidate set from **all valid live slots** in row-local execution memory
	- assembles deterministic ordered fixed `slot_features[s]` for `s in L` inside `DecodeTimeNumPolicy.hpp/.cu`
	- obtains selector scores/logits from model-owned `DecodeTimeSlotSelectorLayer`
	- requires `DecodeTimeSlotSelectorLayer` to apply all learned slot-state encoding from `slot_features[s] -> slot_repr[s]` and return the complete ordered score vector over $\{ NULL \} \cup L$
	- runs an explicit slot-selection policy over that candidate set **before** `<NUM>` is admissible
	- returns either **mask `<NUM>`** or **bind `<NUM>` to one exact selected slot/value**
4. Define the selection behavior explicitly:
	- if exactly one legal slot is resolved, `<NUM>` may be sampled and binds to that slot
	- if no legal slot is resolved, `<NUM>` is masked
	- if multiple legal slots exist but the policy cannot disambiguate, generation masks `<NUM>` and validation/debug contexts throw
5. Make the cutover requirement explicit: decode-time `<NUM>` selection must be driven by an explicit slot-selection mechanism.
	- Acceptable implementations: dedicated slot-selection head; pointer-style selector over the valid live slots.
	- Unacceptable implementations: recency heuristics (`last write`, `most recent live slot`), positional heuristics (`first valid`, nearest-by-position shortcuts), text-matching heuristics (`closest textual number`), post-sample slot inference, and any other implicit fallback selector.
	- Do **not** use `inference_exec_last_write_slot` as an implicit selector for `<NUM>`.
	- Do **not** infer the slot after sampling from emitted text.
6. Define selector outputs explicitly in code/API surface.
	- `Selected(slot_id)`
	- `Null`
	- `Ambiguous`
	- Do **not** collapse these into an unstructured score or a guessed top-1 slot.
7. Implement selector scoring as a masked pointer over `\{ NULL \} \cup L`.
	- `DecodeTimeNumPolicy.hpp/.cu` owns fixed feature assembly only; it does not learn or checkpoint `slot_features[s]`
	- `DecodeTimeSlotSelectorLayer` alone converts `slot_features[s]` into learned `slot_repr[s]`
	- `DecodeTimeSlotSelectorLayer` returns one ordered score vector with `index 0 = NULL` and `indices 1..|L| = ordered row-local candidates`
	- clean form: `key[s] = W_k * slot_repr[s]`, `score[s] = q_t · key[s]`
	- richer form allowed: `score[s] = MLP([q_t ; key[s] ; q_t * key[s]])`
	- an explicit `NULL` option must exist in the selector choice set
8. Define an explicit ambiguity rule.
	- acceptable operational rule: `Selected` iff `top1 - top2 >= selection_margin`, else `Ambiguous`
	- forbidden rule: “pick top-1 anyway and pretend certainty”
9. Add direct selector supervision to training.
	- target per supervised decode timestep is exactly one of legal slot id / `NULL` / `IGNORE`
	- `IGNORE` excludes selector loss and must not be treated as `NULL`
	- train with masked cross-entropy over `\{ NULL \} \cup L`
	- do **not** rely on scalar regression loss or token loss alone to teach slot reference resolution
10. Apply that mechanism before sampling by updating sampler bad-token masks for `<NUM>` when binding is not legal.
11. After sampling, if `<NUM>` is emitted, require that the mechanism already resolved a valid binding; otherwise throw.
12. Add shared slot-range validation for inference prompt maps.
13. Route decode-time ExecutionBlock through the existing atom decode path.
	- `Inference_GPU.cu` must not call `executeStep()` with `nullptr` atom pointers and `num_atoms = 0` when decode-time execution is active.
	- The existing decode-time atom path must provide the row-local atom view used by `executeStep()`.
	- If decode-time execution is active and that atom view is unavailable, throw immediately.
14. Ensure docs and public API comments say clearly:
	- empty map means all `-1`
	- `-1` means non-state-bearing
	- decode-time `<NUM>` binds only to an explicitly selected live slot
	- otherwise `<NUM>` is masked and cannot be generated
15. Keep decode-time file ownership explicit.
	- `DecodeTimeSlotSelectorLayer` owns all trainable selector tensors and produces selector logits/scores.
	- `DecodeTimeNumPolicy.hpp/.cu` owns candidate-set construction, deterministic fixed `slot_features[s]` assembly, and null/ambiguity/mask-or-bind policy over those selector outputs, but owns no trainable tensors.
	- `DecodeTimeSlotSelectorLayer` alone applies learned slot-state encoding from fixed `slot_features[s]` to `slot_repr[s]` and returns the complete ordered score vector over $\{ NULL \} \cup L$.
	- `Common/grim_language_model_gpu.cu` only consumes the returned decision to update sampler masks/bound token payloads.
	- `training/Inference_GPU.cu` only supplies row-local execution inputs/state needed to ask the policy question.

### No-backwards-compatibility rule

- No automatic slot assignment from generated text.
- No fallback that silently uses stale or inferred slot ids.
- No fallback that emits `<NUM>` with `slot_id = -1`.
- No decode-time policy that binds `<NUM>` to `inference_exec_last_write_slot` merely because it was the latest write.
- No decode-time heuristic stand-ins: no `last write`, `first valid`, nearest-by-position, text-matching, or other implicit selector.
- No selector implementation that predicts numeric value directly instead of resolving slot identity.
- No candidate-construction/null-ambiguity/mask-or-bind policy split across `DecodeTimeNumPolicy.hpp/.cu` and sampler/inference orchestration files.
- No selector tensor ownership split across `DecodeTimeSlotSelectorLayer`, `ExecutionBlockLayer`, policy files, and inference-only code.
- No learned slot-state encoding, embeddings, or selector-side `NULL` parameters owned by `DecodeTimeNumPolicy.hpp/.cu`.
- No policy-side append/rescore/injection of `NULL` after `DecodeTimeSlotSelectorLayer` returns ordered scores over $\{ NULL \} \cup L$.
- No decode-time `executeStep()` call with `nullptr` atom pointers / `num_atoms = 0` when ExecutionBlock decode is active.
- No alternate decode-time slot policy unless it replaces this mechanism wholesale.

### Completion criteria

- `DecodeTimeNumPolicy.hpp/.cu` owns candidate construction, deterministic fixed `slot_features[s]` assembly, and bind-or-mask policy over selector outputs, while `DecodeTimeSlotSelectorLayer` owns learned slot scoring.
- `<NUM>` admissibility is decided before sampling, and every emitted `<NUM>` carries a valid pre-resolved slot/value binding or the path fails hard.
- No heuristic selector fallback (`last write`, `first valid`, positional, text-matching, or post-sample inference) remains anywhere in decode-time generation.
- Decode-time ExecutionBlock uses the existing atom decode path rather than null atom pointers, and the public/docs surface explicitly describes `Selected`, `Null`, and `Ambiguous` outcomes.

---

## Workstream 10 — tests that lock the contract in place

### Files

- Update: `resources/models/GRIM-text/Tests/ExecutionBlockTest.cu`
- **New:** `resources/models/GRIM-text/Tests/BatchPayloadExecutionContractTest.cu`
- **New:** `resources/models/GRIM-text/Tests/GRMTExecutionSerializationTest.cu`
- **New:** `resources/models/GRIM-text/Tests/ExecutionTensorSerializationTest.cu`

**Suggested tools:** `read_file`, `apply_patch`, `get_errors`, `Build_CMakeTools`, `get_changed_files`

### Required test cases

1. Plain numeric text row with all `-1` slots passes validation.
2. Execution-active row with active compiled payload, valid slots, and valid `teacher_steps` passes.
3. Execution-active row missing a required slot fails before GPU execution.
4. Execution-active row with out-of-range slot fails.
5. Mixed batch (plain numeric row + execution row) passes.
6. Per-row execution does not inspect atoms from neighboring rows.
7. Execution-active row larger than `max_seq_len` fails in Phase1.
8. GRMT v10 is rejected after version bump.
9. BOS/EOS insertion preserves execution metadata alignment.
10. Padding preserves `teacher_steps`, compiled bootstrap provenance, and `-1` slot fill semantics.
11. Direct compiled-provenance corruption fails before GPU work.
	- `token_exec_slots` / slot map says one thing.
	- `compiled_bootstrap_bindings` says another.
	- shared validator rejects the row or batch before any GPU execution begins.
12. Duplicate bootstrap initialization of the same `slot_id` fails hard.
13. Two compiled bootstrap bindings targeting the same `token_pos` fail hard.
14. Execution-active row with active compiled payload and zero bootstrap bindings fails at builder time.
15. Teacher-step first write to an in-range non-bootstrap slot is allowed, but any step that invents an out-of-domain slot id fails.
16. Multiple live decode-time slots with a resolved explicit selection bind `<NUM>` to the selected slot/value, not merely the most recent write.
17. Multiple live decode-time slots with no resolved selection cause `<NUM>` to be masked in generation.
18. `<NUM>` emitted without a pre-resolved decode-time slot binding fails hard.
19. Multiple live decode-time slots with no explicit selector output do **not** fall back to `last write`, `first valid`, positional, or text-matching heuristics.
20. Selector output `Selected(slot)` where `slot not in L` fails hard.
21. Selector `Null` masks `<NUM>` even when live slots exist.
22. Selector `Ambiguous` masks `<NUM>` in generation and fails hard in validation/debug contexts.
23. Selector supervision targets are validated as exactly one of legal slot id / `NULL` / `IGNORE`, never numeric regression targets.
24. Dense selector supervision remaps through BOS/EOS insertion and padding with `IGNORE` semantics preserved.
25. `slot_selection_targets[i]` is the selector supervision target for decode position `i` in the row's post-Phase1 sequence representation, and the dense array length equals that post-Phase1 decode-position length exactly.
26. Any sparse selector-supervision payload/serialization path is rejected for this cutover.
27. `IGNORE` excludes selector loss and is not treated as `NULL`.
28. Every trainable execution / selector tensor introduced by this cutover appears in `ParameterGroup` construction and participates in checkpoint save/load through the main serialization layer.
29. Checkpoint save/load round-trip preserves execution / selector tensor values without sidecar files.
30. Checkpoint load fails when a required execution / selector tensor is missing, absent behind a claimed capability, or shape-mismatched.
31. GRMT rows and `BatchPayload` transport do not contain learned weights, Adam moments, or checkpoint blobs.
32. Selector-enabled checkpoints fail load if any required baseline selector tensor (`W_q_select`, `W_k_select`, `null_key_select`, `null_logit_bias`) is missing.
33. If ambiguity thresholding is fixed, no selector ambiguity tensor is checkpointed; if it is learned, `ambiguity_margin_param` must round-trip through checkpoint save/load.
34. Selector parameter ownership is single-site: no duplicate selector `ParameterGroup` entries originate from `ExecutionBlockLayer`, `DecodeTimeNumPolicy`, or inference-only code.
35. `DecodeTimeNumPolicy` assembles deterministic fixed `slot_features[s]` from row-local runtime/provenance state only; those features are not trainable tensors and do not appear as checkpoint fields.
36. `DecodeTimeSlotSelectorLayer` alone encodes `slot_features[s]` into learned `slot_repr[s]` and returns one ordered score vector with `index 0 = NULL` and `indices 1..|L| = ordered row-local members of L`.
37. Policy consumes that selector-layer score vector as-is for null/ambiguity/selection decisions; it does not append, rescore, or inject a separate `NULL` option afterward.
38. Row with `execution_payload_active = false` and non-empty `teacher_steps` fails validation; `teacher_steps` alone do not activate execution.
39. Runtime/validator reconstruct `D_row` from `compiled_bootstrap_bindings ∪ teacher_steps` without any separate serialized `D_row` field.

### Completion criteria

- The required builder, GRMT, Phase1, validator, row-local runtime, selector, and checkpoint-ownership tests exist and pass.
- Negative tests cover every forbidden compatibility/fallback behavior called out by the cutover plan.
- Post-cutover smoke tests include GRMT regeneration, mixed-batch execution, and a short training/validation run.
- The test suite proves both semantic correctness and ownership boundaries, not just happy-path runtime behavior.

---

## Documentation updates

### Files

- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/ADDITION_SEQUENCES_AND_ARG_LEARNING.md`
- Update: `docs/PLATEAU_BUG_INVESTIGATION.md` only if this work affects the active bug narrative
- Update: `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md`
- Update: `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md`
- Keep: `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md` as the implementation checklist

### Operations

1. Remove the “debug / expedient” concept-block path description once deleted.
2. Document the final execution-row contract.
3. Document that execution-active rows must fit inside one sequence.
4. Document the GRMT cutover and required cache regeneration.
5. Maintain the codoc companion in the same change as every qualifying refactor step.
6. Maintain the Mermaid flow companion in the same change as every qualifying sequencing, ownership, validation, or runtime-flow change.

---

## Chronological implementation sequence

1. Purge dead code and non-layer config/API baggage from `execution_block_GPU.*`.
2. Split surviving ExecutionBlock code inside `Layers/ExecutionBlock/` into public façade + memory-stream + data-stream files, with explicit public/private include boundaries and narrow internal view types.
3. Create shared execution metadata header.
4. Define `StructuredExecutionRecord` + `BootstrapLiteralBinding` + `CompiledBootstrapBinding` and clarify expected-value semantics.
5. Add compiled execution-payload activation state, projections/provenance, runtime `D_row` reconstruction contract, and selector supervision alignment to `TrainingSequence` / `TrainingSampleView` / `BatchPayload`.
6. Build canonical concept-row builder and remove `__SLOTS__` path.
7. Add GRMT v11 serialization and loader rejection for old data.
8. Update Phase1 to preserve/remap compiled execution metadata exactly and reject fragmented execution rows.
9. Add shared execution payload validator.
10. Refactor `AutogradTraining.cu` plus the split ExecutionBlock stream files to enforce row-local execution and delete batch-global atom compensation.
11. Refactor `execution_block_GPU.cu` and `execution_block_memory_stream_GPU.cu` to remove silent skip behavior with zero fallback logic and compiled-payload activation as the only runtime gate.
12. Update `computeLossBatch()` and `Phase2_TrainingLoop.cu` so validation and training fail on the same invalid rows for the same reason.
13. Introduce `DecodeTimeSlotSelectorLayer` plus tensor-backed learned parameter / checkpoint ownership rules for execution / selector modules, including explicit `ParameterGroup` participation and serialization request fields.
14. Implement explicit decode-time slot-selection mechanism, fixed `slot_features[s]` assembly in policy, selector-layer learned slot encoding / full $\{ NULL \} \cup L$ scoring, selector supervision, and `<NUM>` admission/binding.
15. Add/expand tests.
16. Regenerate `.grmt` data and run validation/training smoke tests.

---

## Explicit deletions

Delete these behaviors during cutover:

1. `kernelSliceColumns`
2. `kernelFourOpMixForward`
3. `ExecutionBlockLayer::encodeState()`
4. `ExecutionBlockLayer::lastDivClampCount()`
5. `executeStep(...)` parameters `expected_read_v1` / `expected_read_v2`
6. `ExecutionBlockConfig` fields that the layer does not read directly: delete dead ones, move orchestration-owned ones out of the layer boundary
7. `computeLossBatch()` rule: “every `<NUM>` requires a slot when execution is enabled”
8. The trailing `__SLOTS__` text serialization hack
9. Full-batch atom lists passed into per-row `executeStep()`, including any internal filtering/offset compensation that tries to make batch-global atom context behave row-local after the fact
10. Silent `executeStep()` return when no slots were bootstrapped
11. Any duplicate `TeacherStep` definition
12. Any loader path that accepts old GRMT v10 execution semantics
13. Any training-row path that permits slot-bearing metadata without `teacher_steps`
14. Any code path that treats `teacher_steps` and `token_exec_slots` as separately-authored sources of truth
15. The old decode-time `<NUM>` fallback branch that emits invalid slot/value payloads (`slot_id = -1`, `numeric_value = 0.0f`), any other decode-time path that emits `<NUM>` without a valid bound slot/value, and any decode-time heuristic or implicit selector (`last write`, `most recent live slot`, `first valid`, positional shortcut, text-matching, post-sample inference)
16. Any decode-time ExecutionBlock path that calls `executeStep()` with `nullptr` atom pointers / `num_atoms = 0` instead of using the existing decode-time atom path
17. Any validator that infers required slot-bearing positions by scanning raw `<NUM>` tokens instead of compiled bindings
18. Any duplicate bootstrap initialization of the same `slot_id`
19. Any two bootstrap bindings that target the same `token_pos`
20. Any builder path that emits an execution-active row with active compiled payload and zero bootstrap bindings
21. Any teacher-step path that invents or rebinds slot identity mid-row
22. Any selector path that predicts numeric value directly instead of selecting slot identity from `\{ NULL \} \cup L`
23. Any selector output that silently coerces ambiguity into a guessed slot
24. Any selector-supervision path that conflates `IGNORE` with `NULL`
25. Any sparse selector-supervision path or serialization form
26. Any validator that requires rebuilding semantic intent after GRMT load in order to know exact `R`
27. Any non-ExecutionBlock include of `execution_block_internal.hpp`
28. Any split ExecutionBlock stream include of batching, loader, validator, or builder headers
29. Any selector candidate-construction/null-ambiguity/mask-or-bind logic implemented outside `DecodeTimeNumPolicy.hpp/.cu`, or any trainable selector scoring implemented outside `DecodeTimeSlotSelectorLayer`
30. Any execution / selector learned parameter stored as raw `float`, `std::vector<float>`, or config-owned trainable state instead of `GRIM::Tensor`
31. Any execution / selector trainable tensor omitted from `ParameterGroup` construction while still participating in learning
32. Any checkpoint path for execution / selector learned tensors that uses sidecar files, debug dumps, or loader-time reconstruction instead of `grim_model_serialization.cu` + `Layers/Serialization/*`
33. Any GRMT or `BatchPayload` field used to carry learned weights, Adam moments, or checkpoint blobs
34. Any `TensorConversion` helper that owns, caches, or reinterprets trainable state instead of only converting declared tensor layouts
35. Any batched execution-metadata side channel that bypasses `BatchPayload`
36. Any selector tensor owned directly by `ExecutionBlockLayer`, `DecodeTimeNumPolicy`, `Common/grim_language_model_gpu.cu`, or `training/Inference_GPU.cu` instead of `DecodeTimeSlotSelectorLayer`
37. Any policy-side learned slot-state encoder, embedding table, or trainable fixed-feature representation for `slot_features[s]`
38. Any policy-side or orchestration-side append/rescore/injection of explicit `NULL` after `DecodeTimeSlotSelectorLayer` has produced scores over $\{ NULL \} \cup L$
39. Any code path that decides execution-active status solely from `teacher_steps.size() > 0` instead of compiled payload activation state
40. Any runtime `D_row` source other than the exact union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`

---

## Cutover / rebuild requirements

After implementation:

1. Delete existing `.grmt` files.
2. Regenerate training data with the canonical structured execution builder.
3. Re-run ExecutionBlock unit tests.
4. Run a mixed-batch smoke test:
	- plain text numeric rows
	- structured execution rows
5. Run one short training session and one validation pass.
6. Confirm no row-local execution failure occurs from neighboring batch rows.

---

## Success criteria

The cutover is complete only when all of the following are true:

1. Plain numeric text with `slot = -1` is legal in both training and validation.
2. Execution-active rows fail fast if any required slot metadata is missing.
3. Training and validation detect invalid execution rows identically.
4. ExecutionBlock sees only row-local atoms.
5. There is no `__SLOTS__` tail serialization anywhere in the data pipeline.
6. Old GRMT data is rejected and rebuilt.
7. No silent fallback or compatibility shim remains in the codebase.
8. `teacher_steps` and `token_exec_slots` are emitted only as paired projections of one canonical structured execution record.
9. Every builder-declared bound literal compiles to exactly one `ATOM_NUM` token or the row is rejected.
10. Bootstrap bindings are one-to-one in both dimensions: no duplicate `slot_id` initialization and no duplicate `token_pos` targeting.
11. Generation emits `<NUM>` only when an explicit decode-time slot-selection mechanism resolves exactly one legal live slot from row-local execution memory; otherwise `<NUM>` is masked pre-sampling.
12. Slot identity is stable across all steps in a row: teacher steps may first-write legal slots, but may not invent or rebind slot identity.
13. Execution-active rows with active compiled payload and zero bootstrap bindings are rejected by the builder and by the validator if corrupted data slips through.
14. GRMT contains enough compiled provenance to validate exact `R` membership without reconstructing semantic intent.
15. No compensating runtime behavior remains: no silent execution skip, no batch-global atom feed into per-row execution, no invalid-slot/value decode-time `<NUM>` emission, no heuristic or implicit decode-time `<NUM>` selector, and no decode-time ExecutionBlock call that bypasses the existing atom decode path by passing null atom context.
16. The cutover is not considered complete unless an explicit decode-time slot-selection mechanism exists; “we will add the selector later” is not an acceptable architecture state.
17. Decode-time slot selection returns only `Selected(slot_id)`, `Null`, or `Ambiguous`, with `Selected(slot_id)` valid only when `slot_id in L`.
18. Selector training uses direct slot-reference supervision over `\{ NULL \} \cup L`; it is not learned indirectly through numeric regression or token loss alone.
19. Selector supervision alignment is explicit: each supervised decode position carries exactly one of legal slot id / `NULL` / `IGNORE`, with `IGNORE` excluded from selector loss and never conflated with `NULL`.
20. Dense selector supervision is the only legal representation for this cutover and remaps through BOS/EOS insertion and padding exactly like other position-sensitive metadata; sparse selector supervision is forbidden until there is measured justification for a replacement design.
21. `slot_selection_targets[i]` is the selector supervision target for decode position `i` in the row's post-Phase1 sequence representation, and the dense array length equals that row's post-Phase1 decode-position length exactly.
22. The row-local slot domain is explicit and immutable: `D_row` is the exact set of legal slot ids referenced by that row's canonical structured execution record, and configured slot ranges are only outer bounds, not the domain itself.
23. Public/private include boundaries are enforced: non-ExecutionBlock code includes only `execution_block_GPU.hpp`, and `execution_block_internal.hpp` remains private to `Layers/ExecutionBlock/`.
24. ExecutionBlock stream files remain runtime-only: they do not include batching, loader, validator, or builder headers to reach across layers.
25. Decode-time selector policy ownership is centralized: `DecodeTimeNumPolicy.hpp/.cu` owns candidate construction and null/ambiguity/mask-or-bind rules over selector outputs, while sampler/inference orchestration only consumes the returned decision.
26. Every learned execution / selector parameter introduced by this cutover is a `GRIM::Tensor` with explicit `TensorShape`; no host-only learnable source of truth remains.
27. Every trainable execution / selector tensor introduced by this cutover is surfaced through `ParameterGroup`, so optimizer state, grad diagnostics, and checkpointing all operate on the same objects.
28. `BatchPayload` is the only batched transport for compiled execution metadata; no parallel side channel feeds runtime or loss code.
29. `TensorContract_GPU.hpp/.cu` remains the sole tensor/autograd/layout/`ParameterGroup` contract, and `TensorConversion.hpp/.cu` remains pure layout conversion with no trainable-state ownership.
30. Checkpoint persistence for execution / selector tensors is centralized in `grim_model_serialization.cu` + `Layers/Serialization/*`, with explicit request fields, capability validation, and version bump.
31. Required execution / selector tensors missing or shape-mismatched at load fail immediately; no sidecar compatibility path, silent reinitialization, or heuristic reconstruction remains.
32. The concrete baseline selector tensor inventory is explicit and singly owned: `W_q_select`, `W_k_select`, `null_key_select`, and `null_logit_bias` live in `DecodeTimeSlotSelectorLayer`, while optional selector tensors (slot/domain embeddings, learned ambiguity margin, richer scorer weights) live there too if present.
33. `DecodeTimeNumPolicy` owns policy but no trainable selector tensors, and `ExecutionBlock` / inference orchestration own no duplicate selector weights or shadow null-selection parameters.
34. `DecodeTimeNumPolicy` assembles deterministic fixed row-local `slot_features[s]` only; `DecodeTimeSlotSelectorLayer` alone turns those features into learned `slot_repr[s]` and returns the complete ordered score vector over $\{ NULL \} \cup L$.
35. Decode-time policy does not append, rescore, or inject `NULL` after selector-layer scoring; the selector output ordering is explicit with `index 0 = NULL` and `indices 1..|L|` matching the ordered row-local candidate set.
36. Execution-active status is carried by explicit compiled structured-execution payload activation state all the way through builder, GRMT, `TrainingSequence`, `TrainingSampleView`, and `BatchPayload`; runtime code does not infer row activity from `teacher_steps.size()`.
37. For this cutover, runtime `D_row` is not serialized separately; it is reconstructed exactly as the union of slot ids referenced by `compiled_bootstrap_bindings` and `teacher_steps`, with configured slot ranges remaining outer bounds only.
38. `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md` is maintained throughout the refactor and updated in the same change for every architecture-affecting workstream delta, with explicit ownership, migrated-call-site, deletion, validation, and next-gate notes.
39. `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md` is maintained throughout the refactor and its Mermaid diagrams reflect the current sequencing and flow contract, with deleted legacy paths removed instead of preserved as stale diagram branches.

---

## Final implementation stance

This refactor should be treated as a **hard architecture cutover**, not a compatibility patch.

If a dataset, cache file, or code path still depends on the old ambiguous semantics, delete or regenerate it.  
If a row is execution-active, require complete metadata and crash when it is wrong.  
If a row is not execution-active, do not pretend it is.

That is the simplest contract, the most debuggable contract, and the one that matches the project’s fail-loud rules.
