# GRIM Reasoning Inference Data Contract

> **Purpose:** define the exact structured text the reasoning network consumes
> at inference time and the meaning of every model-visible span.
>
> **Audience:** agents and humans creating reasoning examples, upstream systems
> supplying reasoning state, and engineers changing structured inference.
>
> **Living contract:** update this document whenever canonical rendering,
> reasoning-state transport, inference prefix construction, span projection, or
> reasoning supervision changes.

The normative implementation is split across:

- [`concept_block_canonical.hpp`](../DataCollection/concept_block_canonical.hpp):
  canonical model-visible text and logical byte spans.
- [`reasoning_state.hpp`](../DataCollection/reasoning_state.hpp) and
  [`reasoning_state_json.hpp`](../DataCollection/reasoning_state_json.hpp):
  inference-time state accepted from upstream producers.
- [`Phase2_InferenceLoop.hpp`](../resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.hpp):
  structured inference entrypoint.
- [`ConceptSupervision.cu`](../resources/models/GRIM-text/training/Phases/Startup/ConceptSupervision.cu):
  runtime projection of configured ConceptBlock fields onto causal targets and
  masked context before window construction.
- [`concept_block.fbs`](../DataCollection/concept_block.fbs): persisted training
  example fields. This is supporting storage, not the organizing model contract.

If this document and executable code disagree, treat the code as current and
update this document in the same change.

## 1. The inference contract in one view

GRIM structured inference is a prefix-to-continuation problem:

```text
UPSTREAM-SUPPLIED PREFIX                         MODEL CONTINUATION / OUTPUT

prompt
  -> target_state
  -> success criteria + available evidence
  -> constraints
  -> persisted knowns/unknowns (later passes only)
  || inference boundary ||
  -> determine
  -> define
  -> execute
  -> update
  -> answer
```

The left side describes the task, objective, verification envelope, and any
state persisted from earlier generation passes. The right side describes the
model-supervised reasoning transition and terminal response represented in a
complete training example.

The normal `ReasoningState` wire object accepts only:

- `goal.target_state`
- `goal.success_criteria`
- `goal.constraints`
- `knowns`
- `unknowns`

The user prompt travels separately and is combined with that state immediately
before inference. First-pass examples leave knowns and unknowns empty; those
fields are retained for state persisted into later generation passes.
`ReasoningState` deliberately excludes `determine`, `define`, `execute`,
`update`, `answer`, `raw`, and identity metadata. This prevents upstream callers
from smuggling an answer or output state into the supplied state.

`renderReasoningPrompt()` uses the same recursive, config-authored renderer as
training and applies the same `ignore` filtering. It does not remove a specially
named output field. `ReasoningState::withPrompt()` supplies only upstream state,
so normal prefixes end before the output fields because those values are absent.

## 2. Exact canonical model-visible order

Only non-empty fields render. Authored array order is preserved.

```text
<prompt>
PROMPT VALUE
</prompt>

<target_state>
TARGET STATE VALUE
</target_state>

<criteria>
<criterion>
CRITERION 1 VALUE
</criterion>

<evidence>
EVIDENCE 1 VALUE
</evidence>

<criterion>
CRITERION 2 VALUE
</criterion>

</criteria>

<constraints>
<constraint>
CONSTRAINT 1 VALUE
</constraint>

<constraint>
CONSTRAINT 2 VALUE
</constraint>

</constraints>

<knowns>
KNOWN 1 VALUE
</knowns>

<knowns>
KNOWN 2 VALUE
</knowns>

<unknowns>
UNKNOWN 1 VALUE
</unknowns>

OPTIONAL UNLABELED LEGACY REASONING LINE
<determine>
DETERMINE VALUE
</determine>

<define>
LOCAL DEFINITIONS
</define>

<execute>
EXECUTE VALUE
</execute>

<update>
UPDATE VALUE
</update>

<answer>
ANSWER VALUE
</answer>
```

Important rendering facts:

- The prompt and its `<prompt>...</prompt>` tags are model-visible. DataHub uses
  the same configured renderer; there is no separate inspection-only layout.
- Goal, state, phase, and answer tags shown above are literal model-visible
  tokens/bytes.
- Every known and unknown is rendered in its own repeated section. There is no
  outer knowns collection or outer unknowns collection.
- Each evidence span belongs to the immediately preceding criterion.
- Legacy `explanation`/`intermediates` text is unlabeled and appears immediately
  before `determine`.
- `determine`, `define`, `execute`, and `update` occur at most once each in the current
  canonical structure. A value may describe multiple internal substeps, but the
  schema does not represent repeated phase cycles as separate sections.
- A non-empty `raw` value bypasses this entire structure and is therefore not a
  valid structured reasoning-state input.

## 3. Inference prefix versus learned continuation

### 3.1 Supplied inference prefix

The normal structured inference prefix is:

```text
prompt
[target_state]
[criteria/evidence]
[constraints]
[knowns]
[unknowns]
```

Square brackets mean the section is optional and appears only when non-empty.
The prefix is context. It describes what the network is allowed to condition
on; it must not contain hidden future state.

### 3.2 Complete training sequence

A complete example may append:

```text
[legacy reasoning]
[determine]
[define]
[execute]
[update]
[answer]
```

These fields represent the continuation after the normal structured-state
prefix. For train/inference alignment, every token placed before the first
supervised output must be constructible at inference time. The desired
prefix-equivalence invariant is:

```text
normal_inference_prefix
    == complete_training_text[0:first_supervised_output]
```

If explicit reasoning phases are supervised as output, the first output begins
at `determine` (or whichever phase is selected first). If the answer alone is
supervised, all preceding content is training context and therefore must also
be available in the inference prefix for strict equivalence.

### 3.3 Current supervision policy

The authoritative policy is `training.config.concept_spans`, a recursive array
of objects. The former supervised/unsupervised name lists are removed. Each
object declares `name`, `supervision`, `source_path`, `repeat`,
`open_delimiter`, `close_delimiter`, and `children`.

- `source_path` is a JSON pointer relative to the parent's selected source value.
  An empty pointer selects that value itself.
- `repeat: true` emits one instance of the node per source-array item.
- A node with children traverses those definitions; a leaf renders text.
  An un-repeated text array renders newline-terminated lines in one span.
- Delimiters must be either both present or both empty. A delimiter-free parent
  is transparent: its children still belong to its span tree.
- `context` retains tokens with masked LM targets. `supervised` retains tokens
  and teaches their causal next-token targets, including delimiters/separators.
- `ignore` removes tokens from the loaded SFT row. `inherit` uses the nearest
  ancestor's policy; roots require an explicit policy. Explicit child policies
  override ancestor policy, including under an ignored ancestor.
- Every span follows these rules, including prompt and collection wrappers.
  Supervising the first retained token requires a preceding causal token (BOS).
- PT uses the complete compiled text with ordinary causal targets; SFT policy
  selection happens before window construction.

The active configuration supervises determine, define, execute, and answer.
Prompt, target state, criteria/evidence, constraints, knowns, unknowns, legacy
reasoning, and update are context. Criterion/evidence pair children and constraint
entries inherit their parents unless explicitly overridden.

For example the criteria root selects `/goal/success_criteria`; its transparent
`success_criterion` child selects `""` with `repeat: true`, and that child's
criterion/evidence leaves select `/criterion` and `/evidence`. Constraints use
the identical structure with a repeated scalar child. No field-name switches
are involved in rendering, token-span projection, or supervision.

## 4. Semantic contract for each model-visible span

The order is causal. Each later span may depend on earlier spans, but earlier
spans must not depend on information revealed only later.

### 4.1 Prompt — task stimulus

Canonical position: first.

Visible form: `<prompt>`, a newline, the prompt value, a newline, `</prompt>`,
and two trailing newlines, exactly like every other tagged scalar.

Inference ownership: supplied by the caller, separately from `ReasoningState`.

Definition: the observation, request, question, or situation that initiates the
reasoning process.

Rules:

- Must contain the user/source information needed to establish the problem.
- Must not leak a derived answer or future execution result.
- Should be understandable without relying on identity/provenance metadata.
- May contain unstructured language; later state spans normalize the parts that
  matter to reasoning.

### 4.2 Target state — intended terminal condition

Canonical position: after prompt, before every other structured state field.

Visible form:

```text
<target_state>
VALUE
</target_state>
```

Inference ownership: optionally supplied in `ReasoningState.goal`.

Definition: the state that should be true when reasoning is complete. It tells
the network where the transition is meant to end, not how to get there.

Rules:

- Describe the desired terminal condition, not the procedure.
- Preserve the prompt's scope and specificity.
- Do not include the unknown value unless it was already supplied by the input.
- Do not simply paraphrase the entire prompt; isolate the intended outcome.

Example:

```text
The consumed volume is known and reported in liters.
```

### 4.3 Success criterion — verifier condition

Canonical position: inside `<criteria>`, after target state.

Visible form:

```text
<criterion>
VALUE
</criterion>
```

Inference ownership: optionally supplied in `ReasoningState.goal`.

Definition: one observable condition by which the target state can be judged
successful.

Rules:

- State one independently checkable condition.
- Describe what must be true, not the reasoning steps used to make it true.
- Together, criteria should be sufficient to verify the target state.
- Do not add quality requirements that are absent from the task.

Example:

```text
The response reports the difference between the initial and remaining volumes.
```

### 4.4 Evidence — criterion support available to the model

Canonical position: immediately after its paired criterion.

Visible form:

```text
<evidence>
VALUE
</evidence>
```

Inference ownership: optionally supplied with its criterion. An empty evidence
value is allowed and produces no evidence section.

Definition: information that supports the paired success criterion. At
inference time, supplied evidence is context already available before the model
continuation; it must not pretend that a future answer has already been
generated.

Rules:

- Support only the adjacent criterion.
- Include only evidence known at the point where the prefix is constructed.
- Never place a future model answer or future execution result in the inference
  prefix as evidence.
- Leave evidence empty when it is intended to be produced or verified later.

This temporal distinction is critical: evidence text that quotes the authored
final answer may be valid as an offline verification annotation, but supplying
that same text in the inference prefix leaks the answer.

### 4.5 Constraint — invariant or boundary

Canonical position: after all criteria/evidence, before knowns.

Visible form:

```text
<constraints>
<constraint>
VALUE
</constraint>

</constraints>
```

Inference ownership: optionally supplied in `ReasoningState.goal`.

Definition: a boundary the reasoning transition and final response must
preserve.

Rules:

- Express one enforceable limit, prohibition, invariant, or requirement per
  entry.
- Do not restate the target state as a constraint.
- Do not invent restrictions absent from the task/environment.
- A later determine, define, execute, update, or answer span must not violate it.

### 4.6 Known — persisted prior-pass fact

Canonical position: after the complete goal contract, before unknowns.

Visible form, repeated once per entry:

```text
<knowns>
VALUE
</knowns>
```

Inference ownership: supplied in `ReasoningState.knowns` on later generation
passes. First-pass reasoning state leaves this collection empty.

Definition: a fact, value, observation, binding, or condition persisted from an
earlier generation pass and available before the next reasoning transition.

Rules:

- Must be supported by the prompt, environment, prior verified state, or
  explicitly supplied context.
- Must not contain a conclusion that still needs to be derived.
- Preserve entity identity, units, sign/polarity, and relevant uncertainty.
- Keep each entry independently usable; avoid packing unrelated facts together.
- Do not duplicate the same fact with cosmetic rewording.

Knowns form the factual read set for the next transition.

### 4.7 Unknown — persisted unresolved frontier

Canonical position: after all knowns and immediately before any reasoning
continuation.

Visible form, repeated once per entry:

```text
<unknowns>
VALUE
</unknowns>
```

Inference ownership: supplied in `ReasoningState.unknowns` on later generation
passes. First-pass reasoning state leaves this collection empty.

Definition: a value, proposition, decision, or state change that has not yet
been resolved and is needed to reach the target state.

Rules:

- Name the unresolved item without answering it.
- Include only unknowns relevant to the target or a necessary dependency.
- Separate unknowns that require distinct derivations or checks.
- Do not list an item as unknown if the same prefix already supplies it as a
  known or as evidence.

Unknowns form the work queue for the next transition.

### 4.8 Legacy reasoning — unlabeled compatibility context/output

Canonical position: after unknowns and before `determine`.

Visible form: newline-terminated plain text with no wrapper tag.

Source adapters normalize non-empty `explanation`, otherwise `intermediates`,
into `/explanation`. The configured delimiter-free `reasoning` leaf renders
that array as newline-terminated lines in one span. No reasoning tag is added.

Definition: older free-form reasoning content retained for compatibility.

Rules:

- New inference-oriented data should prefer the explicit phase spans below.
- Do not author conflicting `explanation` and `intermediates` arrays.
- Treat this field carefully because its lack of a visible label weakens the
  phase boundary presented to the network.

### 4.9 Determine — state what must be done

Canonical position: first explicit reasoning phase.

Visible form:

```text
<determine>
VALUE
</determine>
```

Definition: state the work needed to reach the answer. The wording should
naturally select the relevant context and relationship before execution without
describing the context-selection process itself.

Network role: maps the supplied task and state to a concise action objective.

Rules:

- State the action, method, comparison, calculation, lookup, or decision needed
  to achieve the answer.
- Name the relevant concepts and relationship naturally in the action statement.
- Use the semantic meaning of relevant knowns and unknowns without narrating
  that the model is collecting, identifying, or distinguishing context.
- Begin directly with the intended action. Do not start with “Determine,” “I
  need to,” or another phrase that repeats the function already expressed by
  the `<determine>` span tokens.
- Do not copy, enumerate, or mechanically restate the complete `knowns` or
  `unknowns` entries.
- Do not contain state-variable identifiers, symbolic bindings, placeholders,
  or schema-style names such as `volume_capacity`, `volume_remaining`, or
  `volume_used`.
- Do not include concrete input values, intermediate values, or the result.
- Refer to semantic roles in ordinary language, such as “starting volume,”
  “volume remaining,” and “volume used.”
- Leave operand binding and actual application of the method to `execute`.
- Respect all constraints.

Good boundary:

```text
Compute liters used by finding how many liters were subtracted from the tank.
```

Bad boundary:

```text
Subtract volume_remaining from volume_capacity to get 36 liters.
```

The good version directly states the required work and carries the relevant
context implicitly. The bad version copies structured state identifiers, binds
operands, and leaks the result.

### 4.10 Define — create local definitions

Canonical position: immediately after `determine` and before `execute`.

Visible form:

```text
<define>
VALUE
</define>
```

Definition: create the local definitions, bindings, and relationships needed to
perform the action stated by `determine`. This is model-supervised working
context for the current generation pass.

Network role: converts the natural-language action objective into explicit local
reasoning terms that `execute` can use.

Rules:

- Define only terms needed for the current reasoning action.
- Ground definitions in the prompt, goal, and any persisted prior-pass state.
- Make distinct concepts locally unambiguous.
- Place local bindings and relationships here rather than in `determine`.
- Do not perform the calculation or claim its result.
- Do not treat local definitions as persisted knowns or unknowns until a later
  state update explicitly persists them.
- In the arithmetic/calculator curriculum, assign a local value with
  `${variable} -> value;` and declare an undefined local with `${variable};`.
  Every entry ends with a semicolon. Numeric bindings contain only the number;
  units remain clear from the prompt and variable names. These are authored
  curriculum conventions for future assignment machinery, not a claim that
  assignment execution is already implemented.
- First-pass arithmetic examples leave `knowns` and `unknowns` empty. Both
  assigned and undefined local variables belong in `define`; the boundary
  between `define` and those collections is local versus persisted state.

Example:

```text
${tank_liters} -> 120;
${remaining_liters} -> 84;
${used_liters};
```

### 4.11 Execute — perform the intended action

Canonical position: after `define`.

Visible form:

```text
<execute>
VALUE
</execute>
```

Definition: apply the selected operation, transformation, tool call, or action
to its bound inputs and expose the resulting observation/value.

Network role: converts a proposed transition into a concrete result.

Rules:

- Must be consistent with `determine`.
- Make input/operand bindings and operation order unambiguous.
- Use only supplied knowns or verified prior results as inputs.
- Preserve units and types through the operation.
- Do not silently perform a different transition than the one selected.
- Tool syntax, when present, must match the grammar for that curriculum.

### 4.12 Update — commit the transition result to state

Canonical position: after `execute` and before `answer`.

Visible form:

```text
<update>
VALUE
</update>
```

Definition: integrate the execution result into the working state, identifying
what became known and what remains unresolved.

Network role: state transition/commit boundary.

Rules:

- Promote verified execution results from unknown to known.
- Preserve prior facts and constraints that remain valid.
- State remaining unknowns when the target has not yet been reached.
- Do not mutate unrelated state.
- Do not declare completion unless the target and criteria are satisfied.
- May be omitted for a terminal direct-answer example, but explicit-state
  curricula should use it consistently.

Example:

```text
consumed volume = 36 liters; no unknowns remain.
```

### 4.13 Answer — external terminal response

Canonical position: final.

Visible form:

```text
<answer>
VALUE
</answer>
```

Definition: the terminal user-facing response produced from the updated state.

Network role: exposes the resolved result outside the internal reasoning
contract.

Rules:

- Must satisfy the target state, criteria, and constraints.
- Must agree with the final known state and execution result.
- Directly answer the prompt at the requested level of detail.
- Preserve units, qualifiers, uncertainty, and safety boundaries when relevant.
- Do not introduce new unsupported reasoning or contradict the update.

## 5. State-transition invariants for training examples

Reasoning data should teach a coherent transition, not merely place plausible
sentences into labeled sections.

### 5.1 No future-state leakage

Information may flow forward:

```text
prompt/goal/state -> determine -> define -> execute -> update -> answer
```

It must not flow backward. In particular:

- knowns must not contain the derived result;
- unknowns must not name a value as though already resolved;
- determine must not contain the execution result;
- define must not perform the calculation or contain the execution result;
- inference-time evidence must not quote a future answer;
- execute must not rely on facts first introduced by update or answer.

### 5.2 Determine/define/execute alignment

The intended work, local definitions, and performed transition must match.
`determine` states the contextualized action, `define` creates the local terms
and relationships needed for it, and `execute` applies those definitions.

### 5.3 Execute/update alignment

Update may commit only what execute established. It may normalize wording or
record remaining work, but it must not manufacture an additional result.

### 5.4 Update/answer alignment

Answer is a presentation of terminal state, not a second calculation channel.
Any material result stated in the answer should already be supported by the
updated state or directly by the supplied context for a direct-answer example.

### 5.5 Goal consistency

Every reasoning phase must move toward the target state while respecting
criteria and constraints. Irrelevant but correct computations are still bad
training examples because they teach poor transition selection.

## 6. Canonical inference example

### 6.1 Upstream structured input

The prompt is supplied separately:

```text
A tank holds 120 liters. After using a few, 84 liters remain. How many liters were used?
```

The reasoning-state object is:

```json
{
  "goal": {
    "target_state": "The consumed volume is known and reported in liters.",
    "success_criteria": [
      {
        "criterion": "The response reports the difference between the initial and remaining volumes.",
        "evidence": ""
      }
    ],
    "constraints": [
      "Preserve liters as the unit."
    ]
  },
  "knowns": [],
  "unknowns": []
}
```

Evidence is intentionally empty: `36 liters` does not exist yet at the
inference boundary.

### 6.2 Exact model-visible inference prefix

```text
<prompt>
A tank holds 120 liters. After using a few, 84 liters remain. How many liters were used?
</prompt>

<target_state>
The consumed volume is known and reported in liters.
</target_state>

<criteria>
<criterion>
The response reports the difference between the initial and remaining volumes.
</criterion>

</criteria>

<constraints>
<constraint>
Preserve liters as the unit.
</constraint>

</constraints>

```

### 6.3 Semantically complete continuation

```text
<determine>
Compute liters used by finding how many liters were subtracted from the tank.
</determine>

<define>
${tank_liters} -> 120;
${remaining_liters} -> 84;
${used_liters};
</define>

<execute>
120 liters - 84 liters = 36 liters.
</execute>

<update>
consumed volume = 36 liters; no unknowns remain.
</update>

<answer>
36 liters were used.
</answer>
```

Whether the active model is trained to generate every continuation section or
only a configured subset depends on the active supervision/window policy. The
semantic ordering remains the same.

## 7. Span boundaries seen by the architecture

The renderer first records half-open UTF-8 byte ranges `[begin, end)`. Corpus
compilation projects them to half-open token ranges over the owning sequence.
Never assume one byte equals one token.

The renderer exposes a depth-first, parent-first ordered tree. Nodes store a
configured name, a half-open range, a parent index, and direct child indices.
Repeated instances share a name; names are unique only among sibling definitions,
not among rendered instances. Roots and siblings cannot overlap; children must
be contained by their parent. Every tagged node includes its opening/closing
delimiter, content, and following separator. There are no value-only exceptions.

Tagged nodes render an opening delimiter plus newline, children or
newline-terminated leaf content, then a closing delimiter plus two newlines.
Transparent parents add no bytes. Empty/missing values and parents with no
rendered children emit nothing. This uniform rule applies to the final answer
as well: it has the same trailing separator as every tagged scalar.

Each delimiter text resolves once to one exact vocabulary piece ID after the
tokenizer loads. Missing/exact-match failures are fatal, including UNK fallback.
Compilation tokenizes content with forced boundaries, inserts the cached IDs
with neutral atom side channels, and records ranges into the owning token row.
Delimiter strings are supplied by configuration, not duplicated in source data.
The cached IDs are derived state, never authored JSON.

GRMT v34 replaces fixed Goal, phase, collection, and Answer records with generic
named nodes. The wire representation stores each node's name, range, and parent;
child lists are reconstructed on read and validated. Token slices are not
duplicated in metadata. A corpus-level structural layout identity is stored once
and shared by all loaded rows. Changed names, paths, order, nesting, repetition,
or delimiters require corpus regeneration; changed supervision policies do not.
Adding a span does **not** require a GRMT binary format/version change.
Pre-v34 files must be regenerated; there is no legacy reader.

SFT projects policies before windowing. Ignore filtering remaps every aligned
token channel and the tree. Windowing clips intersecting ranges and rebuilds
parent/child indices for both pinned-prefix SFT and PT windows. Batch payloads
and forward outputs retain the same immutable generic tree per row. The separate
runtime prefix geometry is derived from the first supervised token; it is not
special prompt-span storage.

The source ConceptBlock FlatBuffer and its JSON adapter remain authoring
storage. Adding a genuinely new source property may still require changing that
source model/adapter, but not the GRMT span format.

## 8. Supporting storage and non-visible data

The FlatBuffer is the persistence envelope for training examples, but the
following fields do not become canonical model-visible text:

| Field | Type | Role |
|---|---|---|
| `id` | string | opaque source/example identity |
| `name` | string | human-readable curator label |
| `format_type` | string | authoring/display preset |
| `source_sequence_id` | string | provenance |
| `timestamp` | int64 | persisted timestamp |
| `intermediate_count` | int | derived compatibility count |
| `step_index` | vector of int | derived compatibility indices |

`raw:string` is the exception: when non-empty it replaces the entire structured
rendering. It has no structured goal/state/reasoning spans and is rejected as a
structured reasoning prompt.

## 9. Data-example review checklist

For every example, verify:

- The inference prefix contains only information available before generation.
- Fields appear in canonical order and use the exact visible labels.
- The target state is an outcome, criteria are checks, and constraints are
  boundaries.
- Each evidence value is temporally available at the point it is supplied.
- Knowns are supported facts; unknowns are genuinely unresolved.
- Determine directly states the contextualized work without narrating its own
  phase or mechanically copying state entries, identifiers, values, operand
  bindings, or results.
- Define creates the local terms and relationships needed by execute without
  performing the calculation.
- Execute performs exactly the intended action on valid inputs.
- Update commits only the execution result and accurately tracks remaining work.
- Answer reflects the terminal state without adding unsupported conclusions.
- The active supervision/window policy actually targets the phases the example
  is intended to teach.
- `raw` is empty for structured reasoning data.

## 10. Maintenance requirements

Review and update this contract when any of these change:

- canonical field order, tags, separators, or omission behavior;
- `ReasoningState` accepted fields or transport validation;
- the structured inference entrypoint or prefix construction;
- logical byte spans or token-span projection;
- supervision targets or section-window policy;
- the semantic role of determine, define, execute, update, or answer;
- persisted ConceptBlock fields that affect model-visible rendering.

New semantic rules should be added under the affected model-visible span, not
primarily under the storage schema. Clearly distinguish implemented behavior
from proposed curriculum guidance.
