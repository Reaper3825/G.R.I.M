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

- [`SpanTextRenderer.hpp`](../resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp)
  and [`SpanTextEncoder.hpp`](../resources/models/GRIM-text/Shared/ConceptBlock/SpanTextEncoder.hpp):
  shared configured text, logical byte spans, and token-span projection with atom
  side channels. These primitives have no training-entry or corpus dependency.
- [`concept_block_canonical.hpp`](../DataCollection/concept_block_canonical.hpp):
  training-entry adapter over those shared primitives.
- [`reasoning_state.hpp`](../DataCollection/reasoning_state.hpp) and
  [`reasoning_state_json.hpp`](../DataCollection/reasoning_state_json.hpp):
  inference-time state accepted from upstream producers.
- [`InferencePrefill.hpp`](../resources/models/GRIM-text/Shared/InferenceState/InferencePrefill.hpp):
  standalone prompt/state adapter that builds the realized inference-prefill payload.
- [`InferenceGenerationSession.hpp`](../resources/models/GRIM-text/Shared/InferenceState/InferenceGenerationSession.hpp):
  generated trajectory initialization, sampling policy, token bookkeeping, and completion.
- [`Phase2_InferenceLoop.hpp`](../resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.hpp):
  generation entrypoints.
- [`Phase2_InspectionLoop.hpp`](../resources/models/GRIM-text/training/Phases/Phase2_InspectionLoop.hpp):
  prefill lens capture and Jacobian inspection entrypoint.
- [`ConceptSupervision.cu`](../resources/models/GRIM-text/training/Phases/Startup/ConceptSupervision.cu):
  runtime projection of configured ConceptBlock fields onto causal targets and
  masked context before window construction.
- [`concept_block.fbs`](../DataCollection/concept_block.fbs): persisted training
  example fields. This is supporting storage, not the organizing model contract.

For descriptions of implemented behavior, executable code is authoritative.
The recurrent contract below is the selected direction; explicitly marked
pending integration and storage changes must not be mistaken for implemented
behavior. Update this document alongside those implementations.

## 1. The inference contract in one view

GRIM structured inference is a recurrent, feedback-conditioned process. DDEA
means Determine, Define, Execute, Answer; Define and Execute are optional and
need not occur together. Not every user input requires a tool call. A direct
response must not acquire artificial definitions, execution, or tool use merely
to fill the phase structure.

```text
UPSTREAM-SUPPLIED PREFIX                         MODEL CONTINUATION / OUTPUT

prompt
  -> target_state
  -> success criteria + available evidence
  -> constraints
  -> persisted knowns/unknowns (later passes only)
  || inference boundary ||
  -> determine
  -> [define]
  -> [execute]
  -> [update]
  -> answer -> completed answer-span boundary
                  |
                  +-> deliver response to the configured recipient
                  +-> receive permitted feedback when the task continues
                  +-> next reasoning cycle under the same active goal
```

The left side describes the task, objective, verification envelope, and any
state persisted from earlier generation passes. The right side describes the
model's working trajectory and externally submitted response for one cycle.
Square brackets mark optional phases. Pre-answer DDE spans are observed as
working context; they are separate from the answer payload delivered outside
that cycle. Answer completion does not by itself establish goal completion.

### 1.1 Recurrent boundary and recipient contract

- Each valid completed span can pause generation for controller observation.
  A DDE span close is not a TextWorld command-dispatch boundary.
- At a valid generated `</answer>`, the controller selects the contents of that
  answer instance only. It must not submit preceding DDE, optional update text,
  tags, earlier answers, or the cumulative decoded continuation.
- The recipient is task-specific. Ordinary conversation delivers a response to
  the user without an environment call. A TextWorld challenge validates the
  answer as one game command and submits that command to the environment.
- A command submission is an attempt, not proof that its requested transition
  occurred. Only the returned observation can establish the result.
- For a continuing environment task, append permitted feedback before generating
  the next cycle. Retain the active Goal until it is completed, failed, cancelled,
  or explicitly replaced. An intermediate answer does not end the challenge.
- Track the delivered span instance and command request so a resume or repeated
  cumulative result cannot submit the same action twice. Invalid or incomplete
  answers do not dispatch. Runtime feedback cannot impersonate generated markers.

The selected semantic loop is:

```text
Goal + permitted observation
  -> Determine -> [Define] -> [Execute] -> [Update] -> Answer
  -> submit Answer to the task recipient
  -> actual feedback / verified task status
  -> next cycle if needed
```

Span-boundary pause/resume is implemented (section 7). Automatic answer routing,
TextWorld model-command submission, and runtime observation insertion are pending.
The existing manual Play command field is not this model-driven controller.
Recurrent interaction does not imply recurrent-network architecture or a change
to the SFT loss policy.

### 1.2 Existing inference transport

The compatibility `reasoning_state` wire object (also serialized by
`ReasoningState` clients) accepts only:

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

The token-LM inference boundary is `executePhase2PayloadInference()` with a
caller-authored, compact single-row inference `BatchPayload`. Its input IDs,
atom registries/side channels, and `NamedConceptSpans` tree are realized before
forward; each tree entry uses a half-open `GoalTokenSpan`. `ModelForwardRequest`
borrows that payload, and lens capture reads its supplied span metadata.
Inference does not accept a training `ConceptBlock`.

`GRIM::Inference::buildInferencePrefill()` adapts the prompt and compatibility wire state
into that payload. It rejects model-generated fields, renders only supplied
input via the shared configured renderer (including `ignore` filtering), and
preserves supplied typed atom boundaries and their metadata. `ReasoningState`
and `withPrompt()` remain compatibility/data-authoring helpers; diagnostic and
worker inference do not use them to construct a training entry.

The tokenizer consumes already-authored or atom-model-produced typed spans; it
does not identify bare numeric text. The separate compiled atom-insertion model
continues to accept raw text through `executePhase2TextInference()` and returns
its annotated input. Token-LM inference consumes the supplied atom annotations
or an already-realized payload; it does not implicitly launch another model.

Token-LM results expose only `continuation_text` as decoded text; full-sequence
token IDs and metadata remain available. The continuation is decoded from the
realized prefill token boundary, with aligned atom side channels, after verifying that generated IDs preserve
that prefix. Diagnostics and worker responses use the continuation, so text
normalization cannot cause prefill state to be logged or returned. Atom-insertion
results return their annotated input in the same `continuation_text` field,
retaining their distinct annotated-input response contract. Prefill-only
inspection has no text output. This refactor preserves training encoding and does
not require corpus regeneration.

## 2. Existing single-block canonical model-visible order

Only non-empty fields render. Authored array order is preserved. This is the
current persisted ConceptBlock rendering for one cycle/example, not a limit of
one reasoning cycle per task. Recurrent trajectory serialization is not yet
represented by repeated phase fields in the source schema.

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
- Each phase and answer has at most one field in the current persisted
  ConceptBlock. The recurrent contract permits successive cycles, with optional
  Define/Execute in each. Do not pack an entire interactive trajectory into one
  scalar or imply that the existing schema already serializes repeated cycles.
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

A current single-cycle example may append:

```text
[legacy reasoning]
[determine]
[define]
[execute]
[update]
[answer]
```

These fields represent the continuation after the normal structured-state
prefix. Define and Execute are used only when needed; neither implies a mandatory
tool call. Recurrent episodes additionally have ordered feedback boundaries
between cycles, with feedback available only after the corresponding submission.
Their storage/projection must preserve these boundaries rather than treating
future observations as initial context. For train/inference alignment, every token placed before the first
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

This phase is optional. Omit it when no explicit local definitions are needed;
its presence does not require an Execute span or an external tool call.

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

Definition: carry out the computation, transformation, or working procedure
needed for the current response, using available inputs. This phase is optional
and does not necessarily involve an external tool. Omit it when the response
does not require such work.

Network role: develops the result or proposed action used to construct Answer.
For TextWorld, this span is observed, not submitted. An intended or predicted
environment transition remains unverified until the answer command has been
submitted and actual feedback received in a later boundary.

Rules:

- Must be consistent with `determine`.
- Make input/operand bindings and operation order unambiguous.
- Use only supplied knowns or verified prior results as inputs.
- Preserve units and types through the operation.
- Do not silently perform a different transition than the one selected.
- Tool syntax, when present, must match the grammar for that curriculum.

#### Tool-expression closure for self-correction RL

The following is a separately scoped, pending self-correction tool-path design.
It is not the recurrent TextWorld dispatch contract, does not make tool calls
mandatory, and must not cause DDE text to be sent to TextWorld. Its proposed
TOOL/READ dispatch needs explicit adapter-specific reconciliation before use
with the answer-routed interaction controller. Ordinary response cycles need
none of this grammar.

The proposed authoring contract for that tool path is:

```text
<TOOL>tool_expression -> ${destination}</TOOL>
```

`</TOOL>` closes the complete expression, including its output destination.
The destination must already be declared in `define`; it names where the runtime
will assign the actual result, not a value the model supplies. No trailing
semicolon or post-close destination is required or accepted as part of this
expression. Semicolons remain the terminators for declarations and input
bindings in `define`.

At the complete sampled `</TOOL>`, the planned runtime validates the enclosed
tool expression, arguments, and destination before dispatch. A partial or
invalid expression does not execute. This closes a tool expression, not the
execute phase or the attempt; `</answer>` remains the independent attempt/grading
boundary. Result assignment and configured automatic injection or explicit READ
access follow execution.

For this planned feedback path, `auto_inject_tool_results` controls successful
result values only. Its default is `false`: successful values stay assigned to
their variables until READ; `true` additionally injects them automatically.
Execution, call-validation, and READ-resolution errors are automatically injected
under either setting, with no READ required. Error feedback provides the failure
reason without leaking withheld result values or treating an error as a successful
assignment. It resumes the same attempt and does not replace `</answer>` as the
grading boundary.

This remains a pending, tool-path-specific design. Existing
arithmetic authoring currently emits `<TOOL>expression</TOOL> -> ${destination}`;
its generator, validators, and affected data must migrate before being used with
the new parser. The current TOOL atom recognition alone does not implement this
dispatch behavior. See the
[grammar and insertion map](GRIM_SELF_CORRECTION_RL_GRAMMAR_AND_INSERTION_MAP.md)
for implementation ownership.

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

Update remains optional and is not an additional mandatory DDEA phase. When
present, it may summarize already observed results, including feedback from a
previous cycle. It cannot commit the outcome of the current answer's environment
action before that answer has been submitted and feedback has arrived.

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

### 4.13 Answer — externally submitted response for the current cycle

Canonical position: final generated response span of the current cycle, not
necessarily the end of the task or recurrent trajectory.

Visible form:

```text
<answer>
VALUE
</answer>
```

Definition: the response delivered to the configured recipient. For ordinary
conversation this is user-facing text; for a TextWorld challenge it is the game
command submitted to the environment. DDE and optional update remain separate
working context and are not part of the submitted payload.

Network role: commits the current cycle's external response/action. A valid
generated close provides the controller boundary for validation and delivery.

Rules:

- Must respect the recipient's response format and the active goal/constraints.
  An intermediate action may advance the goal without satisfying it yet.
- Must agree with observed state and any actual computation; it must not present
  predicted environment outcomes as established facts.
- A final user response must satisfy the target and criteria at the requested
  level of detail. An intermediate TextWorld response contains one command,
  without surrounding explanations or a claim that the command succeeded.
- Preserve units, qualifiers, uncertainty, and safety boundaries when relevant.
- Do not introduce new unsupported reasoning or contradict the update.

## 5. State-transition invariants for training examples

Reasoning data should teach a coherent transition, not merely place plausible
sentences into labeled sections.

### 5.1 No future-state leakage

Information may flow forward:

```text
prompt/goal/state -> determine -> [define] -> [execute] -> [update] -> answer
    -> recipient feedback -> next cycle
```

It must not flow backward. In particular:

- knowns must not contain the derived result;
- unknowns must not name a value as though already resolved;
- determine must not contain the execution result;
- define must not perform the calculation or contain the execution result;
- inference-time evidence must not quote a future answer;
- execute must not rely on facts first introduced by update or answer.

### 5.2 Determine/define/execute alignment

When present, the intended work, local definitions, and performed work must match.
`determine` states the contextualized action, `define` creates the local terms
and relationships needed for it, and `execute` applies those definitions.

### 5.3 Execute/update alignment

Update may commit only what actual computation or already received feedback
established. It may normalize wording or record remaining work, but it must not
manufacture a result or anticipate feedback from the current answer submission.

### 5.4 Update/answer alignment

Answer is the externally submitted response for this cycle. Any result it
asserts must be supported by available context or completed work, even when
Define, Execute, or Update is omitted. A proposed environment command is not an
assertion of its success. Task completion is established separately by the
task's verifier, not by the existence of an answer close.

### 5.5 Goal consistency

Every reasoning phase must move toward the target state while respecting
criteria and constraints. Irrelevant but correct computations are still bad
training examples because they teach poor transition selection.

## 6. Single-cycle canonical inference example

This arithmetic example uses Define and Execute because they serve the task.
It does not require an external tool call and is not a template requiring all
phases for every user input. A recurrent environment episode can contain many
cycles before its goal is reached; the following example happens to finish in one.

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

Structured Phase 2 inference retains the canonical span tree through ignore
filtering and uses the same `encodeConceptBlockRender` delimiter insertion and
token-range projection as corpus compilation. The resulting immutable token
spans travel on `BatchPayload::named_concept_spans`. Ignore filtering remaps
parent/child indices and byte boundaries; explicit retained children under an
ignored ancestor remain represented by that ancestor's surviving range.

Lens prompt selection uses the configured prompt span's half-open token range,
including its delimiters and separators: its final token is `span.end - 1`.
The diagnostic resolves the prompt span name from the root definition with
`source_path: /prompt`. This is distinct from the final token of the entire
supplied prefix, which may end in constraints or persisted state. Prompt
membership is determined by span containment, not by prefill/decode mode or the
loss mask. Plain-text inference does not fabricate a structured tree; a lens
without span metadata requires an explicit token position, and prompt membership
remains unknown. Supervision policy and causal target projection are unchanged.

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

Structured inference now retains these resolved definitions in an immutable,
host-only `ConceptSpanDelimiterMap` attached to the prefill payload. The map is
built once per structured prefill from the complete configured definitions,
including continuation sections absent from the supplied input, and shared by
generation sessions and returned trajectories. It is not persisted in JSON,
GRMT, or checkpoints. A new prefill resolves against its current tokenizer and
configuration; there is no global cache that can retain stale vocabulary IDs.
Plain-text inference does not attach this map or interpret structural markers.

The reverse lookup maps a token ID to opening/closing marker candidates identified
by definition index and enclosing tagged definition, rather than name alone.
Delimiter-free groups introduce no runtime marker scope. Shared IDs under
different tagged parents are resolved by the active span stack; ambiguous IDs
within one scope and IDs used for both opening and closing are rejected.

`InferenceGenerationSession` tracks committed continuation tokens incrementally
and retains opening, closing, and invalid-marker events on `GeneratedSequence`.
Positions are absolute token offsets in the owning sequence. A valid close
records `[opening_position, closing_position + 1)`, including both markers but
excluding the following separator. This event range is distinct from canonical
rendered spans, which include trailing separators. Unexpected markers do not
close active spans. Each session starts with an empty stack at the completed
prefill boundary; incomplete spans at EOS or a token limit have no close event.
The inference return policy now yields on a valid close event at the newest
committed token. Opening markers, invalid closes, and older close events do not
trigger a yield. `InferenceReturnReason::SpanBoundary` is distinct from EOS;
the returned trajectory retains that reason and `finished == false`. Explicit
finish, token/context limits, and EOS remain terminal and take precedence over
a span event at the same position. Minimum-length EOS suppression does not
suppress span returns. The existing result decoder decodes the generated
continuation through the closing marker after the loop returns.

The begin entrypoint returns at the first completed configured span and retains
the live session in `GenerationState`, alongside its prefill, reusable device
storage, KV cache, and local-atom retrieval state. The newest committed marker
is still pending at this boundary. `resumePhase2PayloadInference()` acknowledges
the delivered boundary and continues the same sampler/trajectory without another
prefill. The KV cache's `host_seqlen` is the authoritative processed-token cursor:
every queued token is consumed once, in order, before another action is sampled.
An acknowledged close cannot immediately yield again. Returned results are
snapshots; IDs and decoded continuation remain cumulative from the original
prefill boundary. Resume requires a matching tokenizer and a span-paused session.
Terminal EOS/limits, inspection completion, and advance failures release the
live session; explicitly beginning another inference resets the previous one.

Generation length and minimum-length EOS suppression now count only
`TokenOrigin::SampledAction` tokens, including sampled structural markers.
Supplied/runtime context and forced retrieval tokens consume context capacity
but not the sampled-token budget. The context limit is the smaller of the model's
sequence limit and KV-cache capacity. Retrieval replay selection uses remaining
context capacity, rather than charging replay tokens to the sampled budget.
The runtime-context append/atom-remapping API and tool/READ dispatch remain
pending; the scheduling supports queued tokens but does not itself authorize or
construct runtime observations.

For recurrent TextWorld interaction, the controller must additionally distinguish
observed DDE boundaries from the Answer submission boundary, extract only the
newly closed answer instance, and insert the actual permitted environment result
before resuming. A span yield is a scheduling event; only the configured recipient
policy makes an answer actionable. No other span close automatically dispatches
a TextWorld command. The current cumulative continuation result is not itself a
command payload. A completed answer may finish one cycle while the generation
session and challenge remain active; terminal budgets/EOS still take precedence
under the existing return policy.

GRMT v34 replaces fixed Goal, phase, collection, and Answer records with generic
named nodes. The wire representation stores each node's name, range, and parent;
child lists are reconstructed on read and validated. Token slices are not
duplicated in metadata. A corpus-level structural layout identity is stored once
and shared by all loaded rows. Changed names, paths, order, nesting, repetition,
or delimiters require corpus regeneration; changed supervision policies do not.
Adding a span does **not** require a GRMT binary format/version change.
GRMT v35 removes retired execution gate, slot, bootstrap, and transition
metadata from every row. Prefix geometry and named spans remain; typed atom
channels and tables retain their encoding. Pre-v35 GRMT files must be
regenerated; there is no legacy reader.

SFT projects policies before windowing. Ignore filtering remaps every aligned
token channel and the tree. Windowing clips intersecting ranges and rebuilds
parent/child indices for both pinned-prefix SFT and PT windows. Batch payloads
and forward outputs retain the same immutable generic tree per row. The separate
runtime prefix geometry is derived from the first supervised token; it is not
special prompt-span storage.

The source ConceptBlock FlatBuffer and its JSON adapter remain authoring
storage. Adding a genuinely new source property may still require changing that
source model/adapter, but not the GRMT span format.

No source-schema, renderer configuration, supervision, or corpus migration is
performed by this documentation transition. Recurrent episode storage and
feedback projection must be implemented explicitly; a challenge, an interaction
cycle, and a persisted SFT ConceptBlock are not assumed to have a 1:1 mapping.

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

The retired register-machine payload is absent from ConceptBlock authoring,
JSON adapters, corpus compilation, and DataHub. Its former FlatBuffer field slot
is deprecated and reserved to preserve offsets in existing datasets; readers
ignore it and writers omit it. The model-visible `execute` text span and typed
atom annotations remain part of the current contract.

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
- Define, when needed, creates local terms without claiming an execution result.
- Execute, when needed, performs the intended work on available inputs; its
  presence or absence must not imply a mandatory external tool call.
- Optional Update commits only verified work/observations and tracks remaining work.
- Answer contains only the recipient-facing response for its cycle. For TextWorld,
  only its validated command is submitted; DDE is observed separately.
- Each feedback item follows the submission that produced it; no cycle sees
  hidden world facts, future results, or saved solutions as supplied evidence.
- Goal completion is checked independently from answer-span completion.
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
