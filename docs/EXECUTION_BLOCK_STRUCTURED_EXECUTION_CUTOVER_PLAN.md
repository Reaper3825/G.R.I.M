# ExecutionBlock Structured Execution Cutover Plan

**Status:** Proposed implementation plan  
**Scope:** `token_to_slot_map`, `teacher_steps`, row-local ExecutionBlock execution, GRMT cutover, inference alignment  
**Policy:** **No backwards compatibility**. Delete legacy behavior instead of preserving it.

---

## Objective

Make structured execution in GRIM-text behave as a single, explicit system:

- `execution_block_GPU.cu` is reduced to live, layer-owned code before semantic refactor work continues,
- plain numeric atoms remain legal text tokens,
- execution-active rows carry explicit register metadata,
- training and validation enforce the exact same contract,
- ExecutionBlock only consumes row-local state,
- old debug-only serialization paths are removed.

The current codebase mixes two incompatible assumptions:

1. `token_to_slot_map == -1` means **non-state-bearing** numeric token.
2. Every `<NUM>` token must have a slot when `execution_block_enabled=true`.

This plan removes that ambiguity completely.

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

An execution-active row with:

- `teacher_steps.size() > 0`, and
- `bootstrap_bindings.empty()`

is malformed and must be rejected by the builder **before** tokenization/serialization is emitted.

This is a builder-time validity rule, not merely a validator-time warning. The shared validator should still reject such a row if corrupted data somehow reaches runtime, but the canonical builder is not allowed to produce it.

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
	- For each `s in L`, provide a slot-state vector.
	- Minimum required contents:
		- slot id embedding
		- current numeric value embedding
		- valid/live bit
		- last-write / usage features
		- provenance features where available
	- Suggested form: `slot_repr[s] = SlotEncoder(slot_id, value, usage, provenance, state_flags)`
	- The selector chooses slot identity from these representations, not from raw numeric values alone.

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

3. **Training rows are execution-active only when they carry structured execution metadata.**  
	For training, the canonical marker is `teacher_steps.size() > 0`.

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

---

## End-state contract

### Non-execution training row

- `teacher_steps.empty() == true`
- every `token_exec_slots[pos] == -1`
- numeric atoms are allowed
- ExecutionBlock is not entered for that row

### Execution-active training row

- `teacher_steps.empty() == false`
- `bootstrap_bindings.empty() == false` at builder time, therefore compiled rows with non-empty `teacher_steps` must carry at least one compiled bootstrap binding
- `D_row` is the exact immutable set of legal slot ids for the row; it is defined by the row's canonical structured execution record, not by the full configured slot range
- every required state-bearing token position `pos in R` has `token_exec_slots[pos] in [num_scratch_slots, num_slots)`
- every non-state token has `token_exec_slots[pos] == -1`
- `teacher_steps.size()` equals the row's compiled execution-step count; no implicit coupling to a global `execution_block_num_steps` is allowed unless fixed-step execution is made an explicit architecture rule and enforced as such
- all `TeacherStep` slot indices are in range
- every bootstrap/read/write slot referenced by the row belongs to `D_row`
- teacher-step write targets may be first-written later, but they must already belong to `D_row`; no step may invent or rebind slot identity
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

| File / Module | Owns after refactor | Must not own |
|---|---|---|
| `resources/models/GRIM-text/Shared/Execution/ExecutionMetadata.hpp` **(new)** | Canonical structured execution record (`StructuredExecutionRecord`, `BootstrapLiteralBinding`, `CompiledBootstrapBinding`, ordered steps) plus derived metadata types (`TeacherStep`, `SlotSelectionTarget`) | Data loading logic, batching, GPU execution |
| `resources/models/GRIM-text/Shared/Execution/DecodeTimeNumPolicy.hpp/.cu` **(new)** | Deterministic generation-time `<NUM>` admission/binding mechanism, selector contract types (`SlotSelectionStatus`, `SlotSelectionResult`), legal live-slot candidate set construction, explicit decode-time slot selection, null/ambiguity handling | Sampling implementation internals unrelated to decode-time register semantics |
| `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.hpp/.cu` **(new)** | Build structured concept rows into `TrainingSequence` + execution metadata | Batch flattening, runtime validation |
| `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu` | Cache ingestion, tokenizer training, GRMT writing, invokes concept builder | Ad-hoc execution contract rules spread across unrelated helpers |
| `resources/models/GRIM-text/training/training_data_loader.hpp` | Serialized training schema (`TrainingSequence`, loader) carrying compiled execution projections + compiled provenance | Runtime execution policy |
| `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp/.cu` | Padded batch flattening and batch-level execution metadata transport, including compiled bootstrap provenance | Tokenization, execution runtime, duplicate validation |
| `resources/models/GRIM-text/Shared/Execution/ExecutionPayloadValidation.hpp/.cu` **(new)** | Single source of truth for execution-row validation | GPU kernels, file IO |
| `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu` | Validation-time payload validation + H2D + forward/loss path | Unique execution contract logic not shared with training |
| `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` | Training-time payload validation + row-local execution orchestration | Private copy of validation rules |
| `resources/models/GRIM-text/Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp/.cu` | Atom detection and row-local atom index extraction support | Deciding whether a row is execution-active |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp` | Minimal public ExecutionBlock API: config, memory/diagnostic structs, `ExecutionBlockLayer` declaration | Bulk private kernel/helper declarations |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp` **(new, private)** | Shared private stage IDs, macros, internal helper declarations used by the split implementation files | Public API surface, training metadata contracts |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu` | Thin layer façade/orchestrator: constructor, destructor, config validation, top-level method wiring between internal modules | Monolithic kernel inventory, mixed memory/data helper implementations |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu/.hpp` **(new)** | Register-memory stream ops: `ExecutionMemory` lifecycle, bootstrap, slot gather/read/write, valid-mask/recent-write/usage maintenance, memory-side read helpers | Token-stream context/trace logic, semantic row activation, compiled execution metadata validation |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu/.hpp` **(new)** | Token/data stream ops: context reduction, trace encoding, arg/op selection, result decode, result injection into `H`, token-stream-side step assembly | `ExecutionMemory` ownership/lifecycle, bootstrap/write bookkeeping |
| `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu` | BOS/EOS insertion, padding, splitting policy | Inventing execution metadata |
| `resources/models/GRIM-text/Common/grim_model_serialization_version.hpp` | GRMT version bump and explicit incompatibility | Compatibility shims |

---

## Workstream 0 — execution_block file deflation before semantic cutover

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

### No-backwards-compatibility rule

- Do **not** keep dead kernels for “future use.”
- Do **not** keep docs-only or test-only public APIs.
- Do **not** keep config mirrors inside `ExecutionBlockConfig` when the layer never reads them.
- Do **not** preserve default-value tests for deleted knobs.
- Do **not** split the file first and carry dead code into multiple smaller files.
- Do **not** let memory-stream and data-stream responsibilities collapse back into one new private junk drawer.
- Do **not** move orchestration policy into the stream files.
- Do **not** preserve silent execution skip as an interim safety valve.
- Do **not** preserve batch-global atom context feeding per-row execution behind internal filtering logic.
- Do **not** preserve the old decode-time `<NUM>` invalid slot/value emission branch while waiting for a later inference cleanup.

---

## Workstream 1 — canonical structured execution source-of-truth model

### Files

- **New:** `resources/models/GRIM-text/Shared/Execution/ExecutionMetadata.hpp`
- Update: `resources/models/GRIM-text/training/training_data_loader.hpp`
- Update: `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`

### Operations

1. Define `StructuredExecutionRecord` in `ExecutionMetadata.hpp` as the single semantic source of truth for execution-active rows.
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
3. Extend `TrainingSequence` to carry compiled training-time execution projections:
	- `std::vector<int32_t> token_exec_slots`
	- `std::vector<GRIM::Execution::TeacherStep> teacher_steps`
 	- `std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings`
	- `std::vector<GRIM::Execution::SlotSelectionTarget> slot_selection_targets` for supervised decode-time slot reference resolution (exactly one of legal slot id / `NULL` / `IGNORE` per supervised decode position)
4. Extend `TrainingSampleView` to expose the compiled execution metadata needed downstream:
	- `teacher_steps`
	- `compiled_bootstrap_bindings`
	so batching/validation can consume provenance without re-reading or reconstructing it elsewhere.
5. Extend `BatchPayload` to remain the padded transport for:
	- `token_to_slot_map`
	- `teacher_steps`
 	- compiled bootstrap provenance per row
	- selector supervision targets for decode-time slot selection training
6. Define the row activation rule in one place:
	- **training row is execution-active iff `teacher_steps` is non-empty**
7. State explicitly in code comments and docs that runtime sees a compiled pair:
	- `token_exec_slots` = runtime binding projection
	- `teacher_steps` = supervision projection
	- both derived from `StructuredExecutionRecord`
8. State explicitly that `token_exec_slots` is compiled only from `bootstrap_bindings[]`, not from arbitrary numeric tokens in rendered text.
9. State explicitly that execution-active rows with non-empty `teacher_steps` and zero `bootstrap_bindings[]` are malformed and must be rejected by the builder.
10. Define `SlotSelectionTarget` as reference-resolution supervision: target per supervised decode timestep is one of legal slot id / `NULL` / `IGNORE`, never a numeric scalar regression target.
11. Define selector supervision alignment explicitly.
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
- Do **not** allow execution-active rows without `teacher_steps`.
- Do **not** allow execution-active rows with non-empty `teacher_steps` and zero `bootstrap_bindings[]` to survive builder emission.

---

## Workstream 2 — delete the `__SLOTS__` debug serialization path and replace it with canonical structured sequence building

### Files

- **New:** `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.hpp`
- **New:** `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.cu`
- Update: `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu`
- Update: `resources/models/GRIM-text/Shared/DataLoader/DataLoader.hpp`

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
	- rejects execution-active rows with non-empty `teacher_steps` and zero `bootstrap_bindings[]`,
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

---

## Workstream 3 — GRMT format cutover

### Files

- Update: `resources/models/GRIM-text/Common/grim_model_serialization_version.hpp`
- Update: `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu`
- Update: `resources/models/GRIM-text/training/training_data_loader.hpp`

### Operations

1. Bump `GRMT_FORMAT_VERSION` from `10` to `11`.
2. Serialize compiled bootstrap provenance in GRMT beside `token_exec_slots`.
3. Serialize `teacher_steps` in GRMT beside `token_exec_slots`.
4. Serialize `slot_selection_targets` in GRMT beside `teacher_steps` for decode-time selector supervision.
5. Reject GRMT v10 unconditionally at load time.
6. Force rebuild of `.grmt` data from source JSON/cache.

### Suggested serialization order

After the existing per-sequence fields:

1. `token_exec_slots[len]`
2. `uint32_t compiled_bootstrap_binding_count`
3. `CompiledBootstrapBinding[compiled_bootstrap_binding_count]`
4. `uint32_t teacher_step_count`
5. `TeacherStep[teacher_step_count]`
6. `uint32_t slot_selection_target_count`
7. `SlotSelectionTarget[slot_selection_target_count]`

### Storage policy

GRMT does **not** store the full semantic `StructuredExecutionRecord`.

GRMT stores the compiled runtime/supervision artifact plus enough compiled provenance to validate exact state-bearing token membership without reconstructing semantic intent:

- `token_exec_slots`
- `compiled_bootstrap_bindings`
- `teacher_steps`
- `slot_selection_targets`

`slot_selection_targets` is serialized only as a dense decode-position-aligned array. Sparse selector-supervision encodings are forbidden for this cutover.

### No-backwards-compatibility rule

- No dual loader for v10/v11.
- No translation shim from v10 “slot-only” rows to v11 “slot + teacher” rows.
- No loader that fabricates selector supervision from heuristics after deserialization.
- No loader that conflates `IGNORE` with `NULL` in selector supervision.
- No validator that depends on rebuilding semantic intent from source JSON after GRMT load.

---

## Workstream 4 — single shared execution payload validator

### Files

- **New:** `resources/models/GRIM-text/Shared/Execution/ExecutionPayloadValidation.hpp`
- **New:** `resources/models/GRIM-text/Shared/Execution/ExecutionPayloadValidation.cu`
- Update: `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`
- Update: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`

### Operations

Implement one validator that enforces:

1. Non-execution row:
	- `teacher_steps.empty()`
	- all slots must be `-1`

2. Execution-active row:
	- `teacher_steps.size()` equals the row's compiled execution-step count derived from the canonical structured execution record
	- `compiled_bootstrap_bindings` is non-empty
	- every slot referenced anywhere in the row belongs to `D_row`
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
	- `token_exec_slots` and `teacher_steps` are mutually consistent projections of one execution row
	- `R = { pos | token_exec_slots[pos] != -1 }` matches serialized `compiled_bootstrap_bindings` exactly
	- `D_row` matches the exact slot ids referenced by the row's bootstrap bindings and teacher steps; validator must not substitute `[S, V)` for `D_row`
	- for every serialized compiled binding `(token_pos, slot_id)`, `token_exec_slots[token_pos] == slot_id`
	- dense selector supervision length equals the row's post-Phase1 decode-position length exactly
	- `slot_selection_targets[i]` supervises post-Phase1 decode position `i`

### Required call sites

- `buildBatchPayload()` for structural validation
- `computeLossBatch()` before any GPU work
- `autogradTrainingStep()` before any GPU work

### No-backwards-compatibility rule

- Delete `computeLossBatch()`-only `<NUM>` slot validation logic after the shared validator is in place.
- Do **not** keep separate training/validation execution rules.

---

## Workstream 5 — Phase1 sequence handling rules for execution-active rows

### Files

- Update: `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu`

### Operations

1. Preserve `teacher_steps` and `token_exec_slots` through BOS/EOS insertion.
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

---

## Workstream 7 — delete silent execution skips

### Files

- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`

### Operations

1. Remove the silent `return` path in `executeStep()` that skips when `M.valid_mask` has no populated value slots.
	- During the split, move the empty-memory check into `execution_block_memory_stream_GPU.cu` and make it fail-loud there; do not preserve a silent skip behind the new file boundary.
2. Move the “should this row execute?” decision to the orchestrator (`AutogradTraining.cu`).
3. New behavior:
	- non-execution row: skip before calling `executeStep()`
	- execution-active row with empty memory / no valid slot bootstrap: **throw**

### No-backwards-compatibility rule

- No runtime healing.
- No “row had bad execution metadata, but we just left H untouched.”

---

## Workstream 8 — align validation path and training path

### Files

- Update: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`
- Update: `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- Update: `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`

### Operations

1. Make `computeLossBatch()` and `autogradTrainingStep()` invoke the same shared validator.
2. Make both paths fail on the same invalid rows.
3. Keep validation-loop exception logging in `Phase2_TrainingLoop.cu`, but do not let it be the first place invalid metadata is detected.

### No-backwards-compatibility rule

- No “validation is stricter than training.”
- No “training path tolerates broken rows because bootstrap just skipped them.”

---

## Workstream 9 — inference and generation alignment

### Files

- **New:** `resources/models/GRIM-text/Shared/Execution/DecodeTimeNumPolicy.hpp`
- **New:** `resources/models/GRIM-text/Shared/Execution/DecodeTimeNumPolicy.cu`
- Update: `resources/models/GRIM-text/Common/grim_language_model_gpu.cu`
- Update: `resources/models/GRIM-text/training/Inference_GPU.cu`
- Update: `resources/models/GRIM-text/GRIM/grim_language_model_cuda.hpp`

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

### No-backwards-compatibility rule

- No automatic slot assignment from generated text.
- No fallback that silently uses stale or inferred slot ids.
- No fallback that emits `<NUM>` with `slot_id = -1`.
- No decode-time policy that binds `<NUM>` to `inference_exec_last_write_slot` merely because it was the latest write.
- No decode-time heuristic stand-ins: no `last write`, `first valid`, nearest-by-position, text-matching, or other implicit selector.
- No selector implementation that predicts numeric value directly instead of resolving slot identity.
- No decode-time `executeStep()` call with `nullptr` atom pointers / `num_atoms = 0` when ExecutionBlock decode is active.
- No alternate decode-time slot policy unless it replaces this mechanism wholesale.

---

## Workstream 10 — tests that lock the contract in place

### Files

- Update: `resources/models/GRIM-text/Tests/ExecutionBlockTest.cu`
- **New:** `resources/models/GRIM-text/Tests/BatchPayloadExecutionContractTest.cu`
- **New:** `resources/models/GRIM-text/Tests/GRMTExecutionSerializationTest.cu`

### Required test cases

1. Plain numeric text row with all `-1` slots passes validation.
2. Execution-active row with valid slots and valid `teacher_steps` passes.
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
14. Execution-active row with non-empty `teacher_steps` and zero bootstrap bindings fails at builder time.
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

---

## Documentation updates

### Files

- Update: `resources/models/GRIM-text/Layers/ExecutionBlock/ADDITION_SEQUENCES_AND_ARG_LEARNING.md`
- Update: `docs/PLATEAU_BUG_INVESTIGATION.md` only if this work affects the active bug narrative
- Keep: `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md` as the implementation checklist

### Operations

1. Remove the “debug / expedient” concept-block path description once deleted.
2. Document the final execution-row contract.
3. Document that execution-active rows must fit inside one sequence.
4. Document the GRMT cutover and required cache regeneration.

---

## Ordered implementation sequence

1. Purge dead code and non-layer config/API baggage from `execution_block_GPU.*`.
2. Split surviving ExecutionBlock code inside `Layers/ExecutionBlock/` into public façade + memory-stream + data-stream files.
3. Create shared execution metadata header.
4. Define `StructuredExecutionRecord` + `BootstrapLiteralBinding` + `CompiledBootstrapBinding` and clarify expected-value semantics.
5. Add compiled execution projections/provenance and selector supervision alignment to `TrainingSequence` / `TrainingSampleView` / `BatchPayload`.
6. Implement explicit decode-time slot-selection mechanism, selector supervision, and `<NUM>` admission/binding.
7. Add GRMT v11 serialization and loader rejection for old data.
8. Build canonical concept-row builder and remove `__SLOTS__` path.
9. Add shared execution payload validator.
10. Update Phase1 to reject fragmented execution rows.
11. Refactor `AutogradTraining.cu` plus the split ExecutionBlock stream files to enforce row-local execution and delete batch-global atom compensation.
12. Refactor `execution_block_GPU.cu` and `execution_block_memory_stream_GPU.cu` to remove silent skip behavior with zero fallback logic.
13. Update `computeLossBatch()` to use the shared validator.
14. Align inference-side validators and API comments, route decode-time ExecutionBlock through the existing atom decode path, and replace the old decode-time invalid-slot/value / heuristic `<NUM>` selector policies with explicit slot-selection mechanism plus bind-or-mask.
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
20. Any builder path that emits an execution-active row with non-empty `teacher_steps` and zero bootstrap bindings
21. Any teacher-step path that invents or rebinds slot identity mid-row
22. Any selector path that predicts numeric value directly instead of selecting slot identity from `\{ NULL \} \cup L`
23. Any selector output that silently coerces ambiguity into a guessed slot
24. Any selector-supervision path that conflates `IGNORE` with `NULL`
25. Any sparse selector-supervision path or serialization form
26. Any validator that requires rebuilding semantic intent after GRMT load in order to know exact `R`

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
13. Execution-active rows with non-empty `teacher_steps` and zero bootstrap bindings are rejected by the builder and by the validator if corrupted data slips through.
14. GRMT contains enough compiled provenance to validate exact `R` membership without reconstructing semantic intent.
15. No compensating runtime behavior remains: no silent execution skip, no batch-global atom feed into per-row execution, no invalid-slot/value decode-time `<NUM>` emission, no heuristic or implicit decode-time `<NUM>` selector, and no decode-time ExecutionBlock call that bypasses the existing atom decode path by passing null atom context.
16. The cutover is not considered complete unless an explicit decode-time slot-selection mechanism exists; “we will add the selector later” is not an acceptable architecture state.
17. Decode-time slot selection returns only `Selected(slot_id)`, `Null`, or `Ambiguous`, with `Selected(slot_id)` valid only when `slot_id in L`.
18. Selector training uses direct slot-reference supervision over `\{ NULL \} \cup L`; it is not learned indirectly through numeric regression or token loss alone.
19. Selector supervision alignment is explicit: each supervised decode position carries exactly one of legal slot id / `NULL` / `IGNORE`, with `IGNORE` excluded from selector loss and never conflated with `NULL`.
20. Dense selector supervision is the only legal representation for this cutover and remaps through BOS/EOS insertion and padding exactly like other position-sensitive metadata; sparse selector supervision is forbidden until there is measured justification for a replacement design.
21. `slot_selection_targets[i]` is the selector supervision target for decode position `i` in the row's post-Phase1 sequence representation, and the dense array length equals that row's post-Phase1 decode-position length exactly.
22. The row-local slot domain is explicit and immutable: `D_row` is the exact set of legal slot ids referenced by that row's canonical structured execution record, and configured slot ranges are only outer bounds, not the domain itself.

---

## Final implementation stance

This refactor should be treated as a **hard architecture cutover**, not a compatibility patch.

If a dataset, cache file, or code path still depends on the old ambiguous semantics, delete or regenerate it.  
If a row is execution-active, require complete metadata and crash when it is wrong.  
If a row is not execution-active, do not pretend it is.

That is the simplest contract, the most debuggable contract, and the one that matches the project’s fail-loud rules.
