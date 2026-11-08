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
- [`GoalContract.hpp`](../resources/models/GRIM-text/Shared/Goal/GoalContract.hpp):
  shared Goal/evidence types and strict JSON validation, independent of training
  entries; [`goal_flatbuffer.hpp`](../DataCollection/io/goal_flatbuffer.hpp)
  supplies the shared database/corpus Goal decoder.

For descriptions of implemented behavior, executable code is authoritative.
The recurrent contract below is the selected direction; explicitly marked
pending integration and storage changes must not be mistaken for implemented
behavior. Update this document alongside those implementations.

**Implemented Goal storage and rendering:** each success criterion pairs
model-facing `criterion` text with optional system-facing `evidence`; each
constraint pairs model-facing `constraint` text with optional `evidence` defining
a failure condition. Evidence is a structured condition (`predicate`, `subject`,
and optional `target`), not model-visible text or an observation/result. Source
schema version 3, C++ Goal types, JSON/database adapters, inference transport,
and active renderer configuration use this representation. Sections 4.4 and 4.5
define its semantics. Existing datasets have not been rewritten automatically;
explicit migration tooling is described below. TextWorld binding, evaluation,
the manual episode lifecycle, and RL Phase1's initial environment handoff are
implemented. The single-Goal model interaction controller dispatches completed Execute commands,
inserts permitted environment feedback, and grades the attempt at Answer close.
RL updates remain separate.

**Scheduled Goal loading:** `loadScheduledGoals()` in
`Shared/DataLoader/ConceptBlockCorpusReader` reuses curriculum membership and
`orderedConceptBlocks()` to return owned Goal assignments containing only the
course ID, course `environment_id`, source ConceptBlock ID, and Goal. Shared
blocks belong to their first authored course, matching the existing schedule.
Missing blocks, missing Goals, duplicate selected source IDs, or missing course
environment IDs reject the operation; there is no root-name fallback. The
FlatBuffer reader verifies the file buffer but materializes only selected IDs
and Goals, not authored prompt/DDEA/answer/knowns/unknowns. Legacy JSONL support
parses one complete line temporarily and retains only its ID and Goal. Evidence
remains system-facing. This operation does not start an environment, construct
an inference prefix, or change the current training corpus loading operation.
The Phase2 outer loop consumes this schedule across examples. Focused host checks:
`powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_scheduled_goals.ps1`.

**RL Phase1 environment handoff:** training startup with `training_stage: "rl"`
loads the existing tokenizer artifact and restores the model checkpoint without
running tokenizer generation, loading SFT sequences, or preparing optimizer/epoch
state. `EnvironmentReady()` loads the selected RL curriculum's Goal schedule using
the configured block ordering and initialized data seed. It resolves the first
assignment's `environment_id` through the config-relative
`resources/models/games/environment_registry.json`, then uses the existing worker
compile operation to start and evaluate that frozen Goal episode. Only a valid
initial response publishes `TrainingContext::environment_session`, which owns the
client, resolved world/path, immutable assignments, position zero, and full host
response. `environment_tokenizer` retains the generation tokenizer. Initial
success/failure/stopped states are preserved without forcing a command; the model
can still submit its final Answer for grading. Preparation
failure destroys the unpublished candidate and its worker. These responses are
retained as host data; Phase2 projects only the permitted visible fields described
below. Evidence and authoritative world facts remain private. Phase1 returns
`ready_for_environment`; the executable routes it to `executePhase2EnvironmentLoop()`
for all selected examples rather than entering SFT or claiming an RL update. Ordinary inference
and PT/SFT preparation retain their existing paths.
Host lifecycle checks: `tests/run_environment_preparation.ps1`; startup/handoff
checks: `python resources/models/GRIM-text/Tests/test_environment_startup_host.py`.

**Phase2 initial environment interaction:** `beginPhase2EnvironmentInteraction()`
consumes the prepared episode/tokenizer and returns at the first generation span
pause, EOS, or limit. `buildEnvironmentInferenceInput()` projects only string
fields `feedback`, `description`, `inventory`, `location`, and `location_path`
into labelled `environment_context` text, separate from the authored prompt. Host observation providers already place their readable
catalog/details/UI result in `feedback`; their structured envelopes are not
serialized again. Facts, engine state, requirement evaluations, admissible command
lists, internal IDs, and saved solutions are excluded. The frozen Goal's target,
criterion text, and constraint text form the structured `reasoning_state`; evidence
is removed before rendering regardless of configured span policies. Initial knowns
and unknowns are empty. The existing `buildInferencePrefill()` provides configured
span encoding and the full continuation delimiter map. Episode status remains host
control data, not part of the model prompt. Terminal world state forbids further commands but still permits generation of
the final Answer submission. A live generation session cannot be replaced by another begin.
The result preserves inference return reasons separately from Goal outcomes; a
span pause leaves the session in `TrainingContext` for caller-driven continuation.
The begin entrypoint remains available for inspection of that first return. The
executable now invokes `executePhase2EnvironmentInteraction()` to continue the
current Goal through reasoning pauses, Execute commands, feedback, and final
Answer submission.
`Phase2_EnvironmentLoop.cu/.hpp` owns both the Phase2 environment entrypoint and
its begin helper, following the native `Phase2_TrainingLoop` organization; there
is no separate Phase2 environment dispatch layer.
Host begin tests: `Tests/test_environment_interaction_host.py`.

**Recurrent environment controller:** every new span pause is validated against its
configured root definition, token range, sampled delimiters, and previous delivered
boundary. Invalid structural markers, stale boundaries, injected context inside a
generated span, and a phase closing inside an unfinished typed atom stop the run
without dispatch. Only Determine/Define/Execute/Update/Answer root spans are
accepted. Determine/Define/Update pauses resume without an environment call. Execute
decoding selects only the content tokens between that instance's delimiters, with aligned
atom metadata and strict UTF-8. Surrounding whitespace is trimmed; empty, multiline,
NUL-containing, and over-512-byte commands are rejected. Native commands and the
existing explicitly supported host UI commands use `TextWorldClient`'s normal
step route; the host UI adapter retains its Model-tool eligibility checks.

The episode owner retains its initial response plus ordered action receipts keyed
by Execute closing position, command, submission status, actual response, and error.
Repeating a completed submission returns the receipt without executing again;
conflicting/older boundaries and actions after terminal status are rejected.
Transport failures, malformed/mismatched responses, and unknown host completion
retain an unknown receipt and block further actions pending inspection. They are
never automatically resubmitted. A completed receipt establishes delivery, not
that a requested world transition or Goal success occurred.

After each continuing response, the same visible-field projection supplies runtime
context before resuming the existing generation session. The frozen Goal and
global generation/context budgets remain unchanged throughout the attempt. Native episode
success/failure/stopped status stops commands, but its permitted observation is
inserted before generation continues to the final Answer. Answer close freezes
and grades the Goal attempt against the latest authoritative world response; it
never dispatches an action. EOS/limits before submission freeze an incomplete
failed attempt, even if the world succeeded. Error/attempt termination releases
the live model session. Persistent training export is not implemented.
No rewards, optimizer work, or RL loss
is performed. Host controller tests: `tests/run_environment_dispatch.ps1`.


**Full curriculum example loop:** `executePhase2EnvironmentLoop()` wraps the
single-example `executePhase2EnvironmentInteraction()` and consumes every selected
course example in the existing loader's configured order. Course membership and
environment identity remain on each assignment; a Goal belongs to its example's
sequence/session, not a separate curriculum level. Each example starts at its
course environment's authored player spawn with a fresh generation context,
initial observation, Goal prefix, and generation budget. This is on-demand
inference tokenization, not the SFT/PT token-packing path.

`EnvironmentSession::advanceExample()` uses `start_episode` for another example
in the same environment, retaining the compiled artifact while resetting native
world state and host UI references. When the environment ID changes, it resolves
the course's registry entry and compiles/starts a candidate worker before replacing
the current one. No extra start/reset follows compile. Completed examples are
retained in `previousEpisodes()`; the current/final example remains in
`taskEpisode()`. These are in-memory records, not persistent SFT export.

Passing, failed, EOS/limit, and committed invalid-output attempts advance to the
next example without automatic retries. Infrastructure failures abort the run
with prior results retained. An uncertain reset transition cannot be retried
implicitly. The outer result reports completed/passed/failed example counts and
commands; reaching the end marks the schedule complete and prevents rerunning the
last example. This does not implement a shared-state multi-Quest course session:
each example starts from spawn, rather than inheriting the prior example's world
or model history. Focused host coverage is `tests/run_environment_dispatch.ps1`.

## 1. The inference contract in one view

GRIM structured inference pursues one Goal through an attempt. DDEA names
Determine, Define, Execute, Answer. Training data uses the canonical phase order,
with Determine first among explicit reasoning spans; phases should serve the task.
This authoring convention is not a runtime requirement to emit Determine or to
complete a DDEA sequence.

At runtime, Determine, Define, Execute, and Update are optional. The controller
validates emitted span syntax, configured boundaries, and action payloads without
requiring a phase's presence, a fixed phase order, or a repeated cycle. Answer
close is the normal completed-attempt submission boundary, not a guarantee that
every trajectory produces an Answer. EOS, sampled-token exhaustion, context
overflow, invalid output, and infrastructure errors retain their distinct
termination/failure handling; none requires manufacturing a missing Answer.

An attempt is the ordered trajectory of generated spans and disclosed environment
responses toward one frozen Goal. Intermediate session state is that accumulated
trajectory and its currently available context at any boundary. A task episode
holds the frozen Goal, action receipts, and attempt records. The word "cycle" is
only informal shorthand for interaction; it imposes no schema, grouping, or
required phase sequence. A direct response does not acquire artificial tool calls.

The following is a canonical authoring example, not a runtime phase grammar:

```text
supplied prompt + Goal text + available state
  -> Determine -> [Define]
  -> [Execute(command) -> actual permitted feedback -> further reasoning/actions]
  -> [Update]
  -> Answer -> submit frozen Goal attempt -> evaluate outcome
```

### 1.1 Action and attempt boundaries

- No non-return reasoning span is required at runtime; canonical training order
  does not impose a mandatory Determine-first or repeated DDEA sequence.
- A valid completed span can pause generation for controller observation.
- In the environment controller, a completed Execute contains exactly one
  environment command. Only that instance's content is decoded and dispatched;
  surrounding DDEA text and cumulative continuation are excluded.
- Determine, Define, and Update close without dispatch. Ordinary non-environment
  Execute retains its computation/working-text meaning; this command rule belongs
  to the environment recipient adapter.
- Insert actual permitted feedback after each command and before subsequent
  generation. A requested transition is not proof that it occurred.
- A completed Answer submits the entire Goal attempt for grading. Its text is
  never a TextWorld command, and Answer close does not itself prove success.
- A terminal world blocks further commands; the model may still submit its Answer.
  EOS/limits before that submission produce an incomplete failed attempt.
- Completed action receipts prevent duplicate side effects. Unknown completion
  blocks further actions and grading; it is infrastructure-invalid, not a retry.
- Runtime feedback cannot impersonate generated structural markers.

`Shared/Episode/TaskEpisode.hpp` defines the one shared record in `GRIM::Episodes`:
a frozen `ConceptBlockGoal`, final token trajectory, private environment responses,
command receipts, immutable attempts, and separate optional RL score slots.
`Shared/Environment/EnvironmentSession.hpp` owns the live client/world and this
record, rather than defining another episode type. The generation session owns
live tokens; the final snapshot is retained when the attempt ends. Grading freezes
the actual host response and never replays commands. A failed first attempt is
`AttemptSubmitted`; automatic retries and their world-reset policy remain pending.

The manual Play lifecycle remains separate. This change does not implement SFT
episode serialization or RL optimizer updates.

World Lab's manual Start, Reset, and Record bind the currently selected frozen
ConceptBlock Goal while resetting the environment. Reset/Record use the worker's
`start_episode` operation so a newly browsed Goal cannot inherit the previous
episode's binding. After a successful start, the panel rebuilds its initial model
input through `buildEnvironmentInferenceInput()`: the selected block's authored
prompt (when present), Goal target/criterion/constraint text, fresh permitted
observation, and empty knowns/unknowns. The initial-input overload preserves the
exact authored prompt in `/prompt`; cleaned observations are supplied separately
as `/environment_context`. There is no added `Task:` label inside the prompt.
Later feedback renders only the environment-context span and never repeats the task.
The authored prompt cannot substitute for a missing active-world observation.
DataHub's exact lookup and Goal search both retain the prompt in the frozen
manual selection and saved solution metadata. The worker receives only Goal and
source identity; authored reasoning and answer fields are not projected.
The Phase2 scheduled-Goal loader remains Goal-only. Initial Play history and the
Goal panel display prompt and visible Goal text; evidence remains host-only. Selecting a
different block or stopping/hiding the session invalidates this initial-input
preview. This is manual authoring/inspection plumbing, not an inference launch
or persisted recurrent SFT serialization.

World Lab renders its initial Play entry with `SpanText::renderInput()` and the
current `training.config.concept_spans` definitions, using only the projected
reasoning state plus the authored task prompt. This displays the
same configured text delimiters and ignore policies as structured prefill.
The initial entry adds no transcript command prefix, duplicate inventory, or
admissible-command author aid to that rendered text. No authored continuation
fields are supplied. Actual delimiter token IDs are still resolved by structured
prefill with its matching tokenizer. World Lab and the Tokenizer tab use the
same encode endpoint. It recognizes exact configured marker spellings in pasted
or rendered text and applies the structured delimiter encoder to those ranges.

World Lab manual history retains one accumulated preview with immutable byte
boundaries: initial supplied input (snapshot zero), each submitted command while
awaiting its result, then each disclosed environment response. Typed commands
render as configured Execute spans; responses use the shared permitted projection
and environment-context renderer. Earlier snapshots are retained text prefixes,
not reconstructed from the latest world state. No Determine, Define, Update,
Answer, or knowns/unknowns updates are invented. This is a manual text preview,
not a sampled or tokenized model trajectory.

Each boundary retains its disclosed observation, private host receipt, and
continuation status separately. Host details exposes the receipt outside the
model-visible text. Pending submissions contain no future observation; uncertain
transport completion terminates live play and retains the error without retrying.
World success/failure stops commands but is labelled awaiting Answer, not as model
submission. Manual Play does not generate or submit Answer. Selecting history only
changes inspection and canvas highlights; commands use the latest live state.
Successful Start/Reset/Record replace history; choosing another Goal clears it.
Stop/hide retains historical inspection with a stopped live status. Independent
validation and witness replay do not append live trajectory boundaries. Actual
model generation, Answer submission, and EOS/overflow outcomes remain owned by
the separate inference controller; the UI does not fabricate them.

The context-only `environment_context` root follows `prompt` in the active span
configuration and uses `<environment_context>`/`</environment_context>` markers.
It contains only permitted observation prose. The shared projection removes the
recognized TextWorld startup logo, room/status headings, standalone interpreter
prompts, and the duplicate `LOCATION:` line when a structured path is available.
Identical feedback and description/inventory are emitted once. Raw host responses
stay unchanged; action errors, tool feedback, and ordinary ASCII diagrams remain
available. Authored task and Goal text are never cleaned or rewritten.
Environment-only inference may omit an empty authored prompt when nonempty
environment context is supplied. The compatibility `withPrompt()` ConceptBlock
adapter rejects nonempty environment context because single-block storage cannot
represent it. New context delimiters require regenerated tokenizer/config/corpus
artifacts before running the model; existing vocabulary IDs are never reassigned.

### 1.2 Existing inference transport

The compatibility `reasoning_state` wire object (also serialized by
`ReasoningState` clients) accepts only:

- `goal.target_state`
- `goal.success_criteria`
- `goal.constraints`
- `knowns`
- `unknowns`
- `environment_context` (optional observation text; context-only)

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

## 2. Selected single-block canonical model-visible order

Only non-empty fields render. Authored array order is preserved. This is the
selected ConceptBlock rendering for one authored example, not a runtime phase
sequence or interaction limit. Recurrent trajectory serialization is not yet
represented by repeated phase fields in the source schema.
The active renderer configuration excludes evidence leaves entirely and renders
only the text from criterion and constraint objects.
The optional environment-context node is runtime input; the single-block
ConceptBlock source schema does not persist that observation field.

```text
<prompt>
PROMPT VALUE
</prompt>

<environment_context>
PERMITTED ENVIRONMENT OBSERVATION (WHEN SUPPLIED)
</environment_context>

<target_state>
TARGET STATE VALUE
</target_state>

<criteria>
<criterion>
CRITERION 1 VALUE
</criterion>

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
- Evidence stays paired with its criterion in system storage and is excluded
  from the model-visible text and span tree under the selected contract.
- Legacy `explanation`/`intermediates` text is unlabeled and appears immediately
  before `determine`.
- Each phase and answer has at most one field in the current persisted
  ConceptBlock. A runtime attempt may contain whichever valid generated spans
  are needed, interleaved with actual feedback, without a required cycle or phase
  sequence. Do not pack an entire interactive trajectory into one scalar or imply
  that the existing schema already serializes recurrent trajectories.
- A non-empty `raw` value bypasses this entire structure and is therefore not a
  valid structured reasoning-state input.

## 3. Inference prefix versus learned continuation

### 3.1 Supplied inference prefix

The normal structured inference prefix is:

```text
prompt
[environment_context]
[target_state]
[criteria text]
[constraints]
[knowns]
[unknowns]
```

Square brackets mean the section is optional and appears only when non-empty.
The prefix is context. It describes what the network is allowed to condition
on; it must not contain hidden future state.

### 3.2 Complete training sequence

A current single-block training example may append, in canonical authoring order:

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
after actions, with feedback available only after the corresponding Execute
submission. These boundaries do not require a new Determine or full DDEA sequence.
Their storage/projection must preserve action/feedback boundaries within an
attempt rather than treating future observations as initial context. Answer is
the attempt submission, not an intermediate command. For train/inference alignment, every token placed before the first
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
Evidence has no model-visible span or supervision policy in the active
configuration. Legacy evidence leaves have been removed.
Prompt, target state, criteria text, constraints, knowns, unknowns, legacy
reasoning, and update are context. Criterion/evidence pair children and constraint
entries inherit their parents unless explicitly overridden.

The criteria root selects `/goal/success_criteria`; its transparent
`success_criterion` child selects `""` with `repeat: true`, and its sole text
leaf selects `/criterion`. No evidence leaf is configured. Constraints repeat
objects through a transparent `constraint_entry` child under `/goal/constraints`
and render only each object's `/constraint` text, excluding its `/evidence`.
No field-name switches are involved in rendering, token-span projection, or
supervision. Custom configurations must preserve this system/model boundary;
legacy layouts must be updated before compiling new corpora.

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
For environment episodes, this is only the authored task text. The separate
`environment_context` span carries the current observation and actual feedback.

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

### 4.3 Success criterion — model-facing requirement

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

### 4.4 Evidence — system-facing executable condition

Implemented storage, transport, rendering, and manual TextWorld binding/evaluation
contract. The recurrent controller uses the same episode evaluator for model commands.
This field has no model-visible form. It is paired with one criterion or
constraint in system storage; only the paired requirement text is supplied to
the model. This section describes success evidence; section 4.5 applies the
same condition structure to failure evidence.

Definition: the structured condition the system checks to verify the paired
criterion. Evidence describes the check, not its current truth value, an
execution result, or a model-generated justification.

```json
{
  "criterion": "The player is carrying the key.",
  "evidence": {
    "predicate": "carrying",
    "subject": "key_1",
    "target": ""
  }
}
```

- `predicate` identifies the supported check.
- `subject` identifies the world entity being checked, using its stable world ID.
- `target` identifies a second entity when the predicate requires one. It may
  be omitted or empty for unary checks; for example, `in` needs a container
  target, whereas `carrying` does not.
- Evidence remains nested with its criterion, never in a parallel array or
  associated by matching criterion text.
- A criterion used for automatic TextWorld verification requires valid evidence.
  A criterion without an applicable verifier may omit evidence; the system must
  not infer that it passed or manufacture a TextWorld condition from its text.
- Structured evidence is excluded from model rendering, prefill, and token-span
  projection. Actual observations may enter permitted feedback or knowns under
  their own availability rules; they do not replace this condition definition.

World Lab binds authored predicates to typed TextWorld propositions:
`player_at` becomes `at(P, room)`, `carrying` becomes `in(object, I)`, and
`eaten`, `open`, `closed`, `locked`, `at`, `in`, and `on` map to corresponding
facts. The shared worker binder validates support, stable entity references,
argument types, and arity. Unary checks reject non-empty targets, binary checks
require a correctly typed target, and passage-state checks require a named door.
Binding errors include source ConceptBlock ID, requirement collection/index, and
entity reference when available. A non-sandbox Goal requires a source ID.

Attaching evidence does not insert facts or change entity state: carrying a key
is checked, never granted. The environment owns entities, initial state, and
transitions. The episode owns a frozen Goal snapshot, bound checks, results, and
terminal status. The pinned TextWorld 1.7 adapter reads tracked `_facts` with
typed runtime IDs; public facts use human-readable names and are not verifier
identities. Full verifier data remains separate from permitted model observations.

Manual Play and independent solution replay evaluate initial and post-command
states through the same binder and TextWorld-native `Quest`/`Event`/
`QuestProgression` adapter. All success evidence is combined into one win Event
(native win Events are alternatives); each executable constraint supplies a
separate fail Event. Each episode owns fresh progression using the compiled
game's knowledge base, updated from authoritative interpreter facts. Requirement
display uses native `Event.is_triggering` for current-state checks. An incomplete
verification envelope creates no win Event, while executable failure checks remain
active. No native Quest is created for an empty/sandbox Goal. Condition-only Events
carry no action-policy witness and do not establish general reachability.
Any true constraint-violation check
fails the episode before success is considered. Automatic whole-Goal success
requires at least one success criterion, all success checks true simultaneously,
and valid executable evidence for every criterion and constraint. Missing
evidence is unverified, never a pass. Checks do not encode alternative event
groups or temporal milestones. No Goal/empty success lists never win vacuously;
constraint-only Goals can fail. Engine termination and command limits stop an
otherwise active episode without establishing success or inventing a violation.

Environment compilation omits Goal-specific native Quest endings. Selecting a
different Goal starts fresh interpreter state on the same compiled artifact;
Reset retains the frozen Goal. Database edits do not silently change that snapshot.
The controller enforces terminal state and rejects later commands until reset or
new episode. `state.done/won/lost/score` describe the controller result, while
`state.engine` retains raw engine diagnostics. Goal completion is independent of
compiled-engine Quest status and generated answer-span completion. Native Python
Quest progression supplies success/failure; the controller applies coverage guards,
failure precedence when both native results are true, and non-Goal stop limits.
Quest replacement needs no compilation because compiled endings remain empty.
The `grim-textworld-native-quest-v2` semantics identity invalidates earlier custom
evaluator witnesses. Protocol 2 rejects stale
workers/clients; existing compiled endings require recompilation and are never
loaded as reusable artifacts. See [episode binding](WORLD_LAB.md#episode-evidence-binding)
and [compilation/termination](WORLD_LAB.md#compilation-and-challenge-termination).

Binding checks alone do not establish achievability. Current manual playtesting
reports unknown until independent successful replay produces a witness; verified
initial success has a zero-command witness. Saved witnesses include environment
content, initial state, source ID, Goal identity, and action/evaluator semantics.
Stale witnesses are rejected. General contradiction/reachability proof and
verified-challenge admission remain pending. Failed replay or compilation is not
proof of impossibility or solvability. See [achievability](WORLD_LAB.md#reusable-environments-and-goal-achievability).

The version-3 source schema appends structured evidence and constraint fields;
original string-evidence and scalar-constraint slots remain read-only migration
inputs. Native readers accept legacy empty evidence as absent and wrap legacy
constraint strings as text-only objects, but reject non-empty prose evidence
with an explicit migration error. JSON authoring and inference accept only the
new object shape; omit evidence when no executable verifier is available. Null,
string, malformed, or unknown evidence fields are rejected. Predicate and subject
must be non-empty strings; target may be omitted or an empty string. This is
structural validation, not a claim of predicate support or achievability.

`scripts/migrate_goal_evidence.py` accepts old JSONL or GRCB FlatBuffers and
exports new JSONL plus a separate archive of non-empty legacy prose with block
identity and source path. It leaves the source untouched and refuses existing
output/archive paths. Empty evidence becomes absent; scalar constraints become
text-only constraint objects. It never derives a predicate from prose. Authors
must supply valid conditions before automatic verification can cover these
requirements. `scripts/build_concept_blocks_flatbuffer.py` validates Goals and
writes version-3 datasets; legacy JSON imports fail rather than silently saving
a partially skipped dataset. Source-document quotations in generator bindings
remain source quotations and do not populate executable Goal evidence.

Example migration (writes new artifacts only):

```powershell
python scripts/migrate_goal_evidence.py --input path/to/concept_blocks.fb --output path/to/migrated.jsonl --archive path/to/legacy-evidence.jsonl
python scripts/build_concept_blocks_flatbuffer.py --jsonl path/to/migrated.jsonl --output path/to/migrated.fb
```

The DataHub Goal editor accepts optional condition JSON for success and violation
evidence and validates it before saving. Environment references, supported predicates, arity, and argument types are
validated by the worker binder when an episode starts. Focused verification uses
`tests/run_goal_evidence.ps1` and `python -m unittest discover -s scripts -p test_goal_evidence.py`.
Existing source datasets are not rewritten by this implementation. The changed
span layout requires regenerating affected GRMT corpora before use; regeneration
and trainer/runtime execution are not part of the storage milestone.

### 4.5 Constraint — invariant or boundary

Canonical position: after all criterion text, before knowns.

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

Implemented storage: each entry pairs `constraint` text
with `evidence` using the same condition structure as success criteria:

```json
{
  "constraint": "Do not eat the apple.",
  "evidence": {
    "predicate": "eaten",
    "subject": "apple_1",
    "target": ""
  }
}
```

Only `constraint` text renders inside `<constraint>`. Its evidence is system
metadata and must not enter model text, prefill, or token-span projection.

Constraint evidence defines the condition whose truth establishes a violation
and triggers failure. It does not define a condition that must remain true:
in this example, `eaten(apple_1)` being true means failure. Success-criterion
evidence being true means that criterion is satisfied. Both use `predicate`,
`subject`, and optional `target`, with the same entity/type validation.

The episode evaluator fails the challenge when any constraint's evidence is
true. A constraint used for automatic TextWorld enforcement requires valid evidence.
Without an applicable verifier, evidence may be omitted; the system must not
claim the constraint was checked or translate its prose into an invented check.
Failure evaluation uses the actual environment state and the quest lifecycle;
the model does not generate the evidence definition or decide whether it fired.

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
  known or as permitted feedback.

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

Canonical training-data position: first explicit reasoning phase. Runtime does
not require Determine to appear or to precede other valid generated spans.

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

Canonical training-data position: immediately after `determine` and before
`execute`. This is an authoring convention, not a runtime presence/order rule.

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

Canonical training-data position: after `define`, when Define is needed. Runtime
does not require Define, Execute, or this phase sequence.

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
For the environment adapter, this span contains exactly one command. Its
completed close dispatches that command, and actual feedback is inserted before
continuing the same Goal attempt. An intended or predicted transition remains
unverified until its feedback arrives. Outside this adapter, Execute remains
optional working text and need not invoke a tool.

Rules:

- Must be consistent with `determine`.
- Make input/operand bindings and operation order unambiguous.
- Use only supplied knowns or verified prior results as inputs.
- Preserve units and types through the operation.
- Do not silently perform a different transition than the one selected.
- Tool syntax, when present, must match the grammar for that curriculum.

#### Tool-expression closure for self-correction RL

The following is a separately scoped, pending self-correction tool-path design.
It is not the current environment command contract and does not make tool
calls mandatory. The environment adapter dispatches a single command at Execute
close; this pending TOOL/READ path must not create a second dispatch for that
same action. Ordinary responses need none of this grammar.

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
[grammar and insertion map](model/plans/RL_GRAMMAR.md)
for implementation ownership.

### 4.12 Update — commit the transition result to state

Canonical training-data position: after `execute` and before `answer`. Runtime
does not require Update or this phase sequence; claims still require prior evidence.

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
earlier command. It cannot commit the outcome of an Execute action before
that command has been submitted and feedback has arrived.

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

### 4.13 Answer — submit the Goal attempt

Canonical position: final generated response span of a normally submitted attempt.
EOS, token/context limits, invalid output, or infrastructure failure may end the
trajectory without a completed Answer; preserve that termination outcome.

Visible form:

```text
<answer>
VALUE
</answer>
```

Definition: the final response submitting the work toward the active Goal.
For an environment task, all command submissions and their actual feedback occur
before this submission.
Answer text is not executed as a command. Closing Answer freezes the attempt;
the authoritative verifier determines whether it passed.

Rules:

- Respect the recipient's response format and the active Goal/constraints.
- Agree with observed state and completed computation; do not present predicted
  environment outcomes as established facts.
- Preserve units, qualifiers, uncertainty, and safety boundaries where relevant.
- Do not introduce unsupported reasoning or contradict Update.
- Multiple commands may belong to one attempt; an intermediate command belongs
  in Execute, not Answer. No action is dispatched at Answer close.

## 5. State-transition invariants for training examples

Reasoning data should teach a coherent transition, not merely place plausible
sentences into labeled sections.

### 5.1 No future-state leakage

Information may flow forward:

```text
prompt/goal/state -> determine -> [define]
    -> [execute -> actual feedback -> further reasoning/actions]
    -> [update] -> answer -> attempt grading
```

It must not flow backward. In particular:

- knowns must not contain the derived result;
- unknowns must not name a value as though already resolved;
- determine must not contain the execution result;
- define must not perform the calculation or contain the execution result;
- system-facing evidence conditions must not be rendered into the model prefix;
- execute must not rely on facts first introduced by update or answer.

### 5.2 Determine/define/execute alignment

When present, the intended work, local definitions, and performed work must match.
`determine` states the contextualized action, `define` creates the local terms
and relationships needed for it, and `execute` applies those definitions.

### 5.3 Execute/update alignment

Update may commit only what actual computation or already received feedback
established. It may normalize wording or record remaining work, but it must not
manufacture a result or anticipate feedback from an uncompleted Execute action.

### 5.4 Update/answer alignment

Answer submits the Goal attempt. Any result it asserts must be supported by
available context or completed work, even when Define, Execute, or Update is
omitted. Execute commands and their actual feedback precede this submission. Task completion is established separately by the
task's verifier, not by the existence of an answer close.

### 5.5 Goal consistency

Every reasoning phase must move toward the target state while respecting
criteria and constraints. Irrelevant but correct computations are still bad
training examples because they teach poor transition selection.

## 6. Single-block canonical inference example

This arithmetic example uses Define and Execute because they serve the task.
It does not require an external tool call and is not a template requiring all
phases for every user input. A recurrent environment attempt can contain many
actions and responses before final submission, without a repeated phase sequence;
this example finishes in one pass.

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
        "criterion": "The response reports the difference between the initial and remaining volumes."
      }
    ],
    "constraints": [
      {"constraint": "Preserve liters as the unit."}
    ]
  },
  "knowns": [],
  "unknowns": []
}
```

This example uses the implemented constraint-object transport shape.
Evidence is omitted from both requirement types because this arithmetic example
has no TextWorld verifier.
The system-facing TextWorld evidence example in section 4.4 illustrates the
selected structured representation. Neither verifier metadata nor a future
answer is included in the model-visible prefix.

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
canonical training-data ordering remains the same; runtime does not enforce
that order or require omitted reasoning spans.

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
newline-terminated leaf content, then a closing delimiter plus one newline.
Transparent parents add no bytes. Empty/missing values and parents with no
rendered children emit nothing. This uniform rule applies to the final answer
as well: it has the same trailing separator as every tagged scalar. Regenerate existing
corpora to apply the separator change.

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
`appendPhase2RuntimeContext(ctx, tokenizer, context)` now tokenizes caller-projected
feedback with the existing tokenizer and appends it to the matching span-paused
session through `appendRuntimeContext()`. The operation validates aligned channels,
vocabulary IDs, complete typed atoms, and remaining capacity (including one slot
for continuation). It stages changes before publication and remaps atom-entry IDs
and typed local addresses into cloned session tables, preserving prior snapshots.
Inserted tokens have `RuntimeContext` provenance, no score entries, and produce no
generated span events or sampling callbacks. The operation neither acknowledges
the boundary nor changes the KV cursor. The existing resume operation consumes
the pending closing marker and queued context once before sampling again.
Runtime typed atoms contribute their supplied contents to retrieval candidates;
they never select/replay a different value, and their assigned raw-value address
is preserved independently of decoded text normalization. Appending is rejected
while atom generation/replay is unfinished. Empty context is a no-op on a valid
paused session. Tool/READ dispatch remains pending; the append API does not authorize
or construct observations. The environment controller supplies its permitted
projection before resume. Host coverage is in
`Tests/test_inference_trajectory_host.py`.

For recurrent TextWorld interaction, the controller distinguishes
Execute action boundaries from the final Answer submission. It extracts only the
newly closed Execute command and inserts its actual permitted result before
resuming. Answer freezes and grades the attempt without a worker call. A span yield is a scheduling event; only the configured recipient
policy makes Execute actionable. Answer close submits the attempt for grading;
no other root span dispatches a TextWorld command. The current cumulative continuation result is not itself a
command payload. A completed Answer ends the attempt. Terminal budgets/EOS still take precedence
under the existing return policy and cannot fabricate a completed submission.

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

The structured evidence milestone updates source storage, transport, and renderer
configuration; existing source datasets and GRMT corpora still require explicit
migration/regeneration. Persistent recurrent episode serialization and training projection remain
pending; runtime feedback projection is implemented. A task episode, an attempt,
an individual action/response, and a persisted SFT ConceptBlock are distinct;
there is no required cycle grouping or assumed 1:1 mapping.

## 8. Supporting storage and non-visible data

### Standalone recurrent authoring examples

`scripts/textworld/generate_internal_demonstrations.py` replays the verified GRIM
internal witnesses into `DataCollection/curated/grim_internal_state_v1/demonstrations`.
These standalone `grim-recurrent-authoring-example` version-1 JSONL records have
masked `input` plus ordered assistant and environment `events`. Each event names
its logical `span`, `content`, `origin`, and authoring `supervised` flag. Assistant
Determine/Execute/Answer events are targets; environment-context events are masked.
They preserve individual action/feedback boundaries, not scalar ConceptBlock
continuations. Span flags are authoring metadata; native trainer ingestion,
configured delimiter encoding, and token-level projection are not implemented.

Only permitted visible fields enter observations. Structured evidence, authoritative
facts, outcomes, and witness commands are retained separately in
`verification.private.jsonl`; that file must never be projected as model input.
Catalog feedback uses the explicitly stored empty-model-tool-registry fixture,
matching the existing host observation provider's empty-registry response; it
does not claim to capture a running host registry. Native worker responses remain
unchanged in audit receipts. Unknown provider requests reject generation.
Determine and Answer are authored demonstrations, not sampled model tokens;
Answer is appended after independent evidence success without claiming a native
controller submission/grading run. These files do not implement persistent
runtime `TaskEpisode` export or change existing corpus loading/supervision.

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
| `goal.success_criteria[].evidence` | optional structured condition | system-facing criterion verification; excluded from model text |
| `goal.constraints[].evidence` | optional structured condition | system-facing failure check for the paired constraint; excluded from model text |

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
- Each automatically verified TextWorld criterion has structured evidence with
  valid predicate/entity bindings; evidence is excluded from model input.
- Each automatically enforced TextWorld constraint pairs its model-facing text
  with structured evidence defining a violation, not a condition to preserve.
  Any such violation triggers failure under the current event arrangement.
- Knowns are supported facts; unknowns are genuinely unresolved.
- Determine directly states the contextualized work without narrating its own
  phase or mechanically copying state entries, identifiers, values, operand
  bindings, or results.
- Define, when needed, creates local terms without claiming an execution result.
- Execute, when needed, performs the intended work on available inputs; its
  presence or absence must not imply a mandatory external tool call.
- Optional Update commits only verified work/observations and tracks remaining work.
- Environment Execute contains one validated command and receives actual feedback.
- Answer submits the Goal attempt and is never dispatched as a command.
- Canonical data puts Determine first; runtime requires no non-return phase or
  fixed phase sequence. Answer close submits normally; failure/overflow may end
  the attempt without Answer.
- Each feedback item follows the Execute command that produced it; no span sees
  hidden world facts, future results, or saved solutions as supplied context.
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
