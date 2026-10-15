# GRIM self-correction RL grammar and mechanism insertion map

Companion to [the implementation TODO](GRIM_SELF_CORRECTION_RL_TODO.md).
The TODO owns the selected training schedule and objective equations; this
document explains the language, event boundaries, and code ownership needed to
implement them. Read the [reasoning data contract](GRIM_REASONING_DATA_CONTRACT.md)
before changing rendering or authoring examples.

Status: implementation design, not a description of an implemented RL loop.
Existing symbols below were inspected in the repository. Grammar labeled
**proposed** makes the pending work concrete; it does not silently change the
current tokenizer, corpus, configuration, or reasoning-state wire contract.

## 1. Grammar: what each construct means

### 1.1 Supplied context and generated phases

The current canonical order is prompt, target state, criteria/evidence,
constraints, persisted knowns/unknowns, then determine, define, execute, update,
and answer. Empty fields can be omitted according to the current contract.
The supplied prefix is context; the continuation is generated behavior.

Goal ownership for the planned RL path is undecided. The goal rows below describe
the existing supplied-context contract. A model-generated goal is also under
consideration, but its generation grammar, commitment boundary, and action-mask
treatment have not been selected. Keep task success requirements independently
anchored to the task verifier in either case.

| Construct | Owner and meaning | Runtime consequence |
| --- | --- | --- |
| `<prompt>` | Caller supplies the requested task | Establishes the task, not an executable command |
| `<target_state>` | Caller supplies the desired outcome | Input to grading after the answer boundary |
| `<criterion>` and paired `<evidence>` | Caller supplies checks and evidence already available | Future results must not appear here before disclosure |
| `<constraint>` | Caller supplies a required invariant | Validate relevant actions and grade compliance |
| `<knowns>` / `<unknowns>` | Supplied persisted prior-pass state | Context only; no implicit variable assignment or execution |
| `<determine>` | Model states the work needed | No tool dispatch; no result assignment |
| `<define>` | Model declares local variables and binds supplied inputs | Proposed declaration parser prepares the episode variable store |
| `<execute>` | Model emits actions and explicit reads | Proposed tool/read parser produces runtime events |
| `<update>` | Model describes what the observed results established | Does not itself mutate the authoritative variable store |
| `<answer>` | Model starts its terminal response | Collect the response for grading |
| `</answer>` | Model closes the answer | Immediately freezes the attempt record, independently of success |

The current SFT configuration treats update as context. RL action provenance is
a separate concern: if the policy samples update tokens, those tokens are
actions under the RL objective. Do not copy SFT span masks blindly into RL.

### 1.2 Variables, calls, and reads

Existing arithmetic authoring uses `${name};` for an unresolved local,
`${name} -> value;` for an input binding, and
`<TOOL>expression</TOOL> -> ${destination}` for a tool expression with an output
destination. This is visible in
[the curriculum generator](../scripts/generate_single_step_arithmetic_tool_curriculum.py).
Authoring syntax does not establish that a runtime assignment engine exists.

The following executable grammar implements the selected data-contract boundary:
the destination is inside the TOOL span and `</TOOL>` closes the whole expression.
Runtime parsing remains to be implemented.

```ebnf
identifier   = letter_or_underscore, { letter_or_digit_or_underscore } ;
variable     = "${", identifier, "}" ;
declaration  = variable, ";" ;
input_binding = variable, "->", supplied_literal, ";" ;
tool_call    = "<TOOL>", tool_payload, "->", variable, "</TOOL>" ;
read_request = "<READ>", variable, "</READ>" ;
```

Whitespace is allowed between grammar elements, not inside a variable name.
Names are case-sensitive and episode-local. `supplied_literal` must be a typed
value grounded in available input; it cannot stand in for an unobserved tool
result. `tool_payload` is parsed by the selected tool adapter into an available
tool ID and validated arguments. Its exact argument syntax must be specified
per adapter; this grammar does not make arbitrary text shell-executable.

`</TOOL>` is the expression terminator; no trailing semicolon is used. Dispatch
is eligible at that boundary only after the enclosed payload and destination
validate. Migrate the current curriculum's post-close destination into the TOOL
span, including its generator and validators, before using it with this parser.
Closing `</execute>` is not required to run each complete call. The outer `->`
binding must be parsed outside any quoted/nested adapter payload so an argument
containing those characters cannot be mistaken for the destination separator.

`<READ>${destination}</READ>` requests the stored value of that variable. It
does not supply a new value, perform a tool call, or begin another attempt.
The complete `</READ>` commits the read request; returned data is runtime
context, never a model-authored result. A valid read remains available in either
injection mode.

Proposed parser scope: declarations/input bindings are actionable only within
define; tool calls and reads are actionable only within execute. Text quoted in
the prompt, tool output, feedback, or answer must never dispatch an action.
Only sampled action tokens enter the executable parser. Returned text that
contains `</answer>` must not close an attempt. A generated answer close is the
independent boundary even if other grammar is malformed; record that malformed
structure for grading rather than extending generation to repair it.

For literal delimiter text inside payload strings, define escaping in each
adapter and preserve it through encoding. Never recognize boundaries by
searching a concatenated display string that has lost token provenance.

### 1.3 Assignment and disclosure are separate operations

A declaration creates an unresolved destination. A validated call reserves its
destination, executes once, and assigns the actual returned value. Assignment
does not require that the model see the value. Maintain at least variable name,
state, producing call ID, assignment version, typed value, and disclosure events.
Suggested states are unresolved, pending, assigned, failed, and invalidated.
These are proposed runtime records, not new ConceptBlock fields.

For the initial implementation, serialize calls within an attempt. On validated
reassignment, advance the destination version and invalidate its previous value
before execution; failure leaves that version failed, never apparently backed
by an older successful result. Invalid syntax does not dispatch or mutate a
binding. Redeclaring an existing name should report an error; a retry can reuse
the existing declared destination or declare a fresh name. Historical disclosed
values remain in the transcript with their original versions.

| Control | Value | What it controls |
| --- | --- | --- |
| `self_correction_rl.enabled` | `false` (default) | Existing training path; no RL rollout, reference allocation, RL loss, or retry-EMA update |
| `self_correction_rl.enabled` | `true` | Complete RL collection/scoring/update branch; all policy parameters participate |
| `auto_inject_tool_results` | `false` (default) | On success, assign privately; append value-free status; disclose the value only after a valid READ |
| `auto_inject_tool_results` | `true` | On success, assign and append the actual value automatically, then resume generation |
| Execution/validation/READ error | Either boolean value | Automatically inject error feedback; never require READ to discover the failure |
| Full sampled `</answer>` | Emitted | Close and grade the attempt regardless of runtime success |
| Runtime success verdict | Pass / fail | Reward and whether the initial attempt receives a retry; never generation termination |
| Generated-token budget | 256 per attempt | Counts all sampled segments, including calls/reads/markers, across pauses |
| Attempt budget | Initial plus one retry | Bounds correction; tool calls and reads do not consume this budget |

Both booleans are planned configuration, not installed keys. The RL gate belongs
to training configuration; result disclosure is shared by inference and rollout.
Its final configuration nesting still needs wiring. Snapshot both resolved
settings with each rollout group. Changing disclosure changes the policy's
observations, so never replay an old episode under a newly selected mode.

Explicit mode must not leak unread values through command messages, knowns,
evidence, grading summaries, or retry feedback. Verifiers may inspect private
results; the model may only receive the permitted observation stream. A read
of an unresolved/failed/invalidated variable returns a value-free error.

Error feedback is always automatically inserted in both modes and both attempts.
The boolean only selects disclosure of successful values. Include error identity
and actionable cause, while excluding stale/partial successful-result values.
An error does not become a successful variable assignment or an attempt-ending
event. Resume the same attempt after inserting the error context.

```text
if validation_or_execution_failed:
    record_failure_without_successful_assignment()
    append_error_context()                 # independent of the boolean
else:
    assign_actual_result_to_destination()
    if auto_inject_tool_results:
        append_successful_value_context()
    else:
        append_value_free_assignment_status()  # value awaits READ
```

### 1.4 Runtime insertion grammar

Status, disclosed values, and retry feedback need configured context spans.
Proposed span roles are `tool_status`, `tool_result`, and `attempt_feedback`;
their exact visible delimiters are not yet selected. These names are logical
roles, not newly accepted `ReasoningState` properties. Use the existing span
renderer/encoder and realized payload boundary when adding their wire mapping.

Status identifies the call, destination/version, and result availability or
error without a withheld value. Result context identifies the destination/version
and contains only the value actually disclosed. Retry feedback identifies the
closed attempt and unmet criteria. All three are runtime-authored, masked from
direct policy loss, and appended before the next sampled token. They must not
overwrite or retrospectively edit the prefix used for earlier actions.

## 2. Episode vocabulary and commit boundaries

An **episode** is one independent task instance under one policy version,
starting from an isolated fixture state and containing an initial attempt plus
at most one correction. It owns its variable store, ordered event record, goal,
tool snapshot, disclosure mode, and actual environment transitions. Four
episodes share a task for first-attempt advantage normalization, but do not
share mutable fixture state or variable stores.

An **attempt** is the generated DDEA continuation, including any tool/read
pauses and inserted observations, ending at `</answer>` or an incomplete-budget
termination. A **segment** is uninterrupted sampling between runtime events.
A segment is neither an attempt nor an episode. The retry retains first-attempt
history and valid state; it receives a fresh 256 sampled-token budget.

A **rollout group** contains 16 tasks times four episodes: 64 initial attempts
and zero to 64 actual retries. Collection uses one unchanged policy version.
A **microbatch** is a memory-sized scoring/backward partition of that group,
not a new normalization group or independent optimizer update.

| Boundary | Trigger | What becomes committed | What does not happen |
| --- | --- | --- | --- |
| Token append | One sampled or injected token is recorded | Token ID, provenance, aligned atom/span metadata | No automatic semantic execution |
| Declaration | Complete valid declaration/binding in define | Episode-local declared name or supplied input | No unknown tool value is fabricated |
| Call acceptance | Sampled `</TOOL>` closes a valid expression including its destination | Unique call ID, arguments, reserved destination/version | No dispatch on a partial or invalid TOOL span |
| Result assignment | That call completes | Actual result or failure and environment effects | No successful-value disclosure in explicit mode; errors always auto-inject |
| Result disclosure | Auto-injection or valid READ | Exact inserted observation tokens and source version | No tool re-execution |
| Answer / grading commit | Sampled `</answer>` | Immutable attempt endpoint and evidence snapshot | No waiting for runtime goal success |
| Episode close | First grade passes, or retry is graded | Final episode record | No third attempt and no immediate optimizer step |
| Optimizer commit | All 64 episode contributions accumulated and update succeeds | New policy version and consumed group | No reuse of the old group |

The generated `<update>` span is a statement about state, not the authority for
result assignment or optimizer commit. `commitToken` in current inference is
also only token bookkeeping. Keep these meanings distinct in code names and logs.

At the answer boundary, grade the frozen record. A required result absent at
that point is unmet evidence. No later completion may retroactively improve the
grade. Serialized call handling should normally prevent outstanding calls at
that boundary. EOS or exhausted budgets before `</answer>` produce an incomplete
failed attempt and zero reward; an initial failure can still receive its retry.
Infrastructure-invalid episodes remain separately excluded/replaced as in the TODO.

Fixture rollback is explicit: record it, invalidate affected bindings, and expose
the changed state before continuing. Closing or failing an answer does not undo
tool side effects by itself. No training replay may execute those effects again.

## 3. Before Phase 1: configuration and grammar artifacts

Paths in the insertion tables are relative to `resources/models/GRIM-text/`
unless they explicitly start with `DataCollection/`, `MMO/`, or `commands/`.

### Configuration and grammar insertion points

| Existing insertion point | Mechanism to add | Boundary invariant |
| --- | --- | --- |
| `training/ConfigCompiler/compile_model_config.cpp`, `training/schemas/grim_compiled_hyperparameters.fbs` | Compile the RL gate and result-disclosure control through the existing config flow | Do not read ad hoc JSON inside the token loop |
| `Shared/ModelConfig/CompiledModelConfig.hpp` and `.cpp`, `Shared/HyperParameters/HyperParameters_GPU.hpp` | Carry validated settings into existing runtime snapshots/accessors | Defaults and enabled-only validation agree between compilation and runtime |
| `Shared/UnigramByte/Detectors/AtomDelimiterDetector.cu`, `Shared/UnigramByte/TokenLayout.hpp` | Decide and implement distinct READ recognition/representation | Existing TOOL is an atom type, not a full call grammar; READ must not alias it |
| `Shared/ConceptBlock/SpanTextRenderer.hpp`, `SpanTextEncoder.hpp` | Define observation/feedback span rendering and token projection | Typed atom channels and exact delimiter boundaries survive insertion |

Tokenizer representation is an implementation decision: do not casually append
an atom enum or reserve numeric token IDs without checking vocabulary, checkpoint,
and artifact compatibility. Existing concept delimiters resolve to exact pieces;
READ handling must be reconciled with that machinery. Update the reasoning data
contract and regenerate affected artifacts when an implemented change requires
it. Retired execution metadata is not an implementation shortcut.

## 4. Phase 1: startup and ownership

| Existing insertion point | Mechanism to add | Produced handoff |
| --- | --- | --- |
| `training/Phases/Phase1_Startup.cu` / `.hpp` | Branch-aware initialization and ownership on `TrainingContext` | RL state only when enabled; normal startup when disabled |
| `training/Phases/Startup/Validation/StartupValidation.cu` and `Phase2Handoff.cu` | Validate grammar/tokenizer support, tool fixtures, bounds, reference identity, and freeze conflicts | Fail before collection if the requested mode cannot run |
| `training/Phases/Startup/CheckpointLoad.cu` | Restore policy and RL resume metadata; initialize separate frozen reference | Policy version, reference identity, retry baseline |
| `training/Phases/Startup/Model/ParameterGroupRegistration.cu`, `ParameterRegistry.hpp` | Include all policy parameters in existing registration | No RL subset; no reference parameters in optimizer groups |
| `training/Phases/Startup/OptimizerReady.cu` | Reuse optimizer initialization with RL settings | Existing RAdamW/accumulation ownership |

The live policy and frozen SFT reference need distinct state/ownership. The
reference is a scoring copy with gradients disabled, not a selectively frozen
part of the policy. Fixture control, variable stores, parser state, and tool
results are runtime data, not trainable parameters.

## 5. Phase 2 inference: grammar to runtime events

### 5.1 Prefill and determine/define

`buildPhase2InferencePrefill()` in
`training/Phases/Phase2_InferenceLoop.cu` currently renders supplied input,
resolves configured delimiters, encodes metadata, and builds a compact payload.
Keep that boundary. Supply available action/tool context without private fixture
answers; the replacement action approach and its input handoff remain open. Include
the goal here when supplied; the alternative generated-goal phase remains open.
`DataCollection/reasoning_state_json.hpp`
continues to validate the current wire shape until explicitly extended.

Initialize a fresh episode parser and variable store beside the generation
state. Determine remains text. As define declarations complete, validate local
names and supplied input bindings. Runtime results cannot be authored through
define. This parser is a new mechanism; canonical span rendering alone does
not provide executable semantics.

### 5.2 Execute: call, assignment, and READ

| Existing insertion point | Mechanism to insert | Grammar/event |
| --- | --- | --- |
| `generateOneSequence()` token loop | Incremental parser after recording sampled tokens and before sampling the next action | Complete declaration, bound call, READ, or answer close |
| `commitToken` / `commitForcedToken` lambdas | Explicit token provenance and sampled-token counters | Sampled action versus runtime context versus forced retrieval |
| `MMO/Core/ToolRegistry.hpp` | Resolve the episode's tool/version and validate metadata | `tool_payload` selects an available tool |
| `commands/commands_core.hpp` / `.cpp` | Controlled adapter to `dispatchCommand` / `CommandResult` where applicable | Full call accepted; never arbitrary generated text dispatch |
| New episode-owned assignment handler beside inference orchestration | Store call outcome; inject errors automatically; on success assign destination/version and select disclosure | Boolean gates successful values only |
| New read handler beside the same orchestration | Resolve variable/version and insert actual stored value | Complete `<READ>…</READ>`; no execution |

The command result envelope exposes status/message/error metadata, not a generic
typed result schema. Each selected tool adapter must identify its actual value
and separate it from status text before either disclosure path is reliable.
Do not scrape a message into the transcript in explicit mode and accidentally
reveal the withheld value.

Pause/resume support belongs at the generation boundary, not in an optimizer
kernel. After a runtime insertion, feed every inserted token through the cache
before the next sampled token, or rebuild from the exact recorded prefix if
cache continuation is not supported. Preserve atom registries and span offsets.
Current sequence-length-minus-prefill counting includes appended tokens; replace
its use as the RL action budget with provenance-based counting. Observation
tokens still occupy model context capacity even though they cost no action budget.

Current local retrieval can emit forced tokens with zero log probability.
Zero is not a provenance flag. Prevent injected or replayed context from becoming
executable grammar events, and resolve the TODO's retrieval-policy decision
before claiming all generated actions are sampled from the scored distribution.

### 5.3 Update/answer and correction

Add independent answer-close detection to the same generated-token event path.
Record all sampled closing-marker tokens and close at the full marker even if
goal checks would fail. In a token-boundary edge case, consume no additional
sampled token after the marker. Stop markers inside runtime observations do not
participate in this detector.

The episode controller then freezes the record and invokes the task verifier.
It maps the verdict to the TODO's reward, never to a revised endpoint. On a
failed initial attempt, append permitted feedback and start another generated
DDEA attempt with the original goal/history available. Retain valid variable
bindings; do not reconstruct private values as model-visible knowns. An attempt
boundary is explicit metadata, not inferred from counting repeated text tags.

`executePayload()` currently converts `GeneratedSequence` to
`Phase2TextInferenceResult` and drops token scores and much atom metadata.
Extend this handoff so the episode collector retains the complete record before
any display-only conversion. Do not recreate training trajectories from decoded
text. `GeneratedSequence::finished` alone cannot distinguish answer completion,
budget failure, or an intermediate runtime pause; add explicit event/end reasons.

## 6. Phase 2 training: committed episodes to gradients

### 6.1 Collection and payload construction

At `executePhase2()` / `runEpoch()` in
`training/Phases/Phase2_TrainingLoop.cu`, select the RL collection path using the
single branch gate. The existing SFT batch source must not remain an accidental
prerequisite for collecting online episodes. Reuse the update machinery after
forming the complete group, rather than replacing the training pipeline.

Episode records need goal origin, task/group/attempt IDs, policy/config versions, exact token
IDs and scores, provenance, atom channels, call/assignment/read events, disclosure
positions, termination reason, criteria outcomes, and rewards. Private stored
values may be retained for audit/grading but must not enter scoring prefixes
until their recorded disclosure position.

Adapt these records into `Shared/Batching/BatchPayload.hpp` and its existing
upload path. Carry RL action masks and detached attempt advantages aligned to
causal prediction rows: the row before an action predicts that action, including
the first action after a runtime insertion. A sampled call or READ is scored;
its inserted result is context. During retry scoring the first attempt is
context, not a recipient of retry advantage. Replayed rows never run the parser,
dispatcher, or variable-store mutations.

### 6.2 Forward, objective, and optimizer commit

| Existing insertion point | Mechanism to add | Required invariant |
| --- | --- | --- |
| `Shared/Forward/ModelForward_GPU.hpp` | Parameter-connected recomputation for policy; frozen-reference scoring on identical recorded prefixes | Cached collection stays non-differentiable; dropout disabled for both scoring paths |
| `training/Autograd/AutogradTraining.cu` | Explicit RL loss selection and assembly | Existing SFT objective remains available under the disabled gate |
| `Shared/Loss/ComputeLoss/AutogradLoss.cu` | Signed-advantage action loss and exact categorical forward KL | Same legal support; FP32 KL; reference has no gradients |
| Phase 2 accumulation helpers and `runOptimizerWindowFromEpoch()` | Group-aware accumulation, existing clipping/optimizer dispatch | One update for the complete group; denominator stays `256 * 64` |
| Existing Phase 2 logging/checkpoint handoff | Report RL outcomes and publish consumed-group/policy version | No CE/perplexity interpretation of policy loss |

Normalize initial advantages within each task's four episodes. Snapshot the
shared retry baseline once for all actual retries, calculate pending EMA values
after batch scoring, and apply them once for subsequent groups. Publish the
baseline and policy-version metadata consistently with the completed optimizer
update; a failed/incomplete update must not leave a checkpoint claiming the
group was consumed. Exact equations and one/no-retry edge cases remain in the TODO.

All policy parameters remain included, even when a particular batch produces
zero gradient for some of them. Auxiliary objectives are a separate pending
decision. Existing global RMS clipping semantics must be resolved against the
requested threshold before implementation; do not substitute an L2 rule here.

## 7. Phase 3: checkpoint and cleanup boundary

`training/Phases/Phase3_Cleanup.cu` owns final saves, status, and cleanup. Extend
the existing checkpoint lifecycle (including
`training/OptimizerCheckpoint.hpp` and `Common/ParameterCheckpoint.hpp` as
appropriate) with RL policy/reference identity and retry-EMA state. Restore it
through Phase 1. Select the precise metadata storage owner during implementation;
do not invent a second checkpoint format in the episode collector.

Discard incomplete rollout groups on exit/resume. Do not restart their calls or
restore partially applied gradients as completed episodes. Release episode
fixtures, variable stores, and reference resources through their owning lifecycle.
Final reporting should distinguish answered/graded episodes, incomplete attempts,
and infrastructure-invalid episodes.

## 8. Implementation checks and remaining grammar decisions

Implement focused parser/fixture tests before connecting GPU training: partial
TOOL spans cannot dispatch; destinations must exist; READ never re-executes;
unread values remain private through retries; injected marker text cannot trigger
actions or close an answer; sampled `</answer>` closes both passing and failing
attempts; and action masks preserve the causal row across insertions.

Still to finalize: tool-adapter payload syntax, exact runtime context delimiters
and transport mapping, READ's
tokenizer representation, call/read/time/context limits, fixture rollback rules,
and auxiliary/local-retrieval objectives. These are bounded design choices at
the insertion points above. They do not reopen the selected episode definition,
answer grading boundary, all-parameter policy training, or the two boolean controls.

Required migration: move existing post-`</TOOL>` output bindings inside the
expression in authoring generators, validators, and affected datasets. The
closing delimiter is selected; migration implementation remains outstanding.
