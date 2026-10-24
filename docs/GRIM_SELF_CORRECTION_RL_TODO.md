# GRIM self-correction RL implementation TODO

Scope: add feedback-conditioned, two-attempt policy training to the existing
GRIM-text pipeline. This is an implementation handoff, not a training-run request.

Companion: [grammar, commit boundaries, and phase-by-phase mechanism insertion map](GRIM_SELF_CORRECTION_RL_GRAMMAR_AND_INSERTION_MAP.md).
It expands the integration points below and marks proposed grammar separately
from current executable behavior.

Scope note: the [reasoning data contract](GRIM_REASONING_DATA_CONTRACT.md)
now specifies recurrent DDEA with optional Define/Execute and Answer as the
per-cycle external payload. TextWorld receives only that answer command, then
returns feedback for the next cycle; ordinary user responses need no tool call.
This TODO's two-attempt tool-training schedule and Execute-based TOOL/READ design
remain a separately scoped proposal, not the TextWorld interaction contract.
Do not equate each environment-command answer with a terminal challenge answer
or apply the two-attempt budget to environment steps without an explicit design.

The objective is goal-directed tool use: choose an appropriate available tool,
bind valid arguments, interpret its actual result, and satisfy the supplied
goal and constraints. Self-correction means using execution and goal-check
feedback to repair a failed attempt. Arithmetic is one task family for exercising
this behavior; numeric answer matching is not the general reward contract.

## Reuse existing infrastructure

Use existing tokenizer/configuration loading, checkpoint loading/saving,
structured goal rendering, atom handling, generation/sampling, BatchPayload,
shared forward, autograd, parameter registration, gradient accumulation,
clipping, optimizer dispatch, and logging. Extend their existing boundaries
only where this feature needs additional data or behavior. Do not create
parallel replacements or add generic infrastructure work to this task.

Existing integration points:

| Component | Existing source |
| --- | --- |
| Supplied goal and persisted state | `DataCollection/reasoning_state.hpp` and `reasoning_state_json.hpp` |
| Configured text and span rendering | `resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp` and `SpanTextEncoder.hpp` |
| Runtime delimiter lookup and span events | `resources/models/GRIM-text/Shared/ConceptBlock/ConceptSpanDelimiterMap.hpp` |
| Generation session and token accounting | `resources/models/GRIM-text/Shared/InferenceState/InferenceGenerationSession.hpp` and `.cu` (data-only state and free functions) |
| Return/termination decisions | `resources/models/GRIM-text/Shared/InferenceState/InferenceReturnPolicy.hpp` (free functions) |
| Live session, KV cache, and retrieval ownership | `resources/models/GRIM-text/Shared/InferenceState/GenerationState_GPU.hpp` |
| Tool identities, descriptions, parameters, and availability | `MMO/Core/ToolRegistry.hpp` (`ToolDescriptor`, `ToolRegistry`) |
| Command execution and result envelope | `commands/commands_core.hpp` and `.cpp` (`dispatchCommand`, `CommandResult`) |
| Generated sequence data | `resources/models/GRIM-text/Shared/Forward/GeneratedSequence.hpp` |
| Generation orchestration and public result | `resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.cu` and `.hpp` |
| Sampling distribution | `resources/models/GRIM-text/Shared/Sampling/Sampling.cu` |
| Training payload | `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp` |
| Shared forward and graph policy | `resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.hpp` |
| Loss assembly | `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` |
| Autograd loss primitives | `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu` |
| Training/update orchestration | `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu` |

These are integration boundaries, not an already-connected RL tool loop.
Structured prefill now retains an in-memory `ConceptSpanDelimiterMap` shared
with generation. Sessions identify configured marker IDs and retain generated
span events/ranges without text decoding. The loop now returns at a valid
closing marker with `SpanBoundary`, distinct from terminal EOS/limits, and the
returned sequence remains unfinished. The map is rebuilt per structured
prefill, never persisted. `GenerationState` now retains the live session and
cache across span returns; `resumePhase2PayloadInference()` continues without
re-prefill. Sampled-action counts carry across segments; runtime/forced tokens
consume context capacity only. Tool/READ representation, runtime-context
append/atom remapping, and dispatch remain pending.
`GRIM::Inference::buildInferencePrefill()` renders supplied prompt/state into a payload;
`executePhase2PayloadInference()` generates a continuation. The current public
`Phase2TextInferenceResult` retains the complete `GeneratedSequence` in its
optional `generated_sequence` field, including scores, score positions, token
provenance, and atom channels. Add the episode adapter there.
`ToolRegistry` supplies tool metadata; `dispatchCommand()` accepts a command and
argument string and returns `CommandResult` (`success`, `message`, `errorCode`,
`category`). Neither a registry entry nor `CommandResult.success` proves that
the user's goal was met. The RL bridge must validate generated calls, retain
execution observations, and apply task-specific goal checks. Existing Goal
criteria/spans describe the goal; they are not executable verifiers.

## Tool-use and goal-check contract

Each task supplies a prompt, available action/tool context, and a
runtime verifier. Goal ownership is undecided: the goal (`target_state`,
`success_criteria`, `constraints`) may be supplied as context alongside the
prompt or generated by the model. Current structured inference supports the
supplied-context path; model-generated goal semantics remain design work.
The replacement action approach remains to be specified.
The verifier consumes the recorded calls, actual tool
results, final response, and observable final state. Its implementation can
vary by task family while returning the same episode outcome structure.

| Graded behavior | Evidence for the outcome check |
| --- | --- |
| Tool selection | The chosen tool is available and appropriate for the requested operation; required tool use actually occurred |
| Argument binding | Arguments satisfy the tool contract and refer to the requested entities, inputs, and scope |
| Constraint following | Calls and resulting state preserve the supplied task constraints |
| Result use | The response and subsequent actions agree with actual returned data/errors; claimed results were not invented |
| Goal completion | Required criteria hold in the observed state and final response, including requested output form |
| Correction | A retry responds to the reported failure or unmet criterion and is checked against the same goal |

For the pilot, success is binary: all applicable required checks must pass.
Record individual check results and failure reasons for feedback/diagnostics;
do not add independent tool-call or formatting bonuses. A valid call that solves
the wrong problem fails. A plausible final answer without required execution
evidence fails. Allow equivalent valid tools/action sequences unless the goal
requires a particular one; do not grade against a single reference trace.
Tool use is required only where the task calls for it, not merely to earn reward.

Use a mixed set of bounded tasks: lookup/retrieval, structured extraction or
transformation, controlled state changes, and arithmetic/calculator use.
These are planned task families, not claims that their RL fixtures/verifiers
already exist. Select concrete registered tools and implement their fixtures
before rollout. Include wrong-tool, wrong-argument, tool-error, result-misread,
and unmet-goal cases so correction is learned from distinct feedback causes.
Arithmetic uses the same call/result/goal checks, with numeric comparison only
inside that family's verifier.

Keep model-visible observations separate from private grading data. Expose
tool values through the configured result-access path below; return execution
status and actionable unmet-criterion feedback without bypassing that path.
Keep hidden reference answers and expected action traces private. A value
genuinely returned by the selected tool is valid feedback once access is due,
even when it answers the task. Preserve the goal across attempts and introduce
observations only after execution.
Follow `docs/GRIM_REASONING_DATA_CONTRACT.md`: current wire state accepts only
goal, knowns, and unknowns; new feedback delimiters or repeated-attempt rendering
remain implementation work, not existing `ReasoningState` fields.

### Conditional tool-result access

Add a boolean `auto_inject_tool_results` to the shared tool-feedback
configuration used by rollout and inference. Proposed default: `false`, so the
initial implementation exercises explicit reads. Both values must remain
supported so automatic injection can be selected if explicit access becomes
too costly to maintain. This is a planned configuration entry, not an existing
key in `ai_config.json`; wire it through the existing configuration schema,
compiler, and runtime loader when implementing the feature.

In both modes, the model first defines an unresolved destination variable for
what it needs to obtain. The tool call binds its output to that variable; the
runtime assigns the actual result after execution. The model supplies the
destination, not the result value. Validate declaration, scope, and output
binding before dispatch; reject missing/invalid bindings without inventing a
variable or silently exposing the result. The complete expression is
`<TOOL>tool_expression -> ${destination}</TOOL>`: `</TOOL>` closes it and makes
it eligible for validated dispatch, with no trailing semicolon. Tool-adapter
argument syntax and runtime parsing remain to be implemented. Migrate existing
post-close output bindings as specified in the reasoning data contract.

| Outcome | `auto_inject_tool_results` | Behavior after execution |
| --- | --- | --- |
| Success | `true` | Assign the result to the model-defined variable, then automatically append its value as runtime observation context and resume generation |
| Success | `false` (default) | Assign the result privately; resume with value-free assignment/status feedback and expose the value only after a valid sampled `<READ>...</READ>` request |
| Error | Either value | Automatically append error feedback and resume; no READ is required to learn that the operation failed |

The boolean controls automatic injection of **successful result values only**.
Tool execution errors, call-validation errors, and READ-resolution errors are
automatically injected in both modes, including on the retry. Include the error
code/reason and relevant call or variable identity so the model can correct its
action. Do not assign an error as a successful destination value or expose stale,
partial, or otherwise withheld successful-result data inside the error message.
An injected error stays within the same attempt; it does not itself trigger a
retry or close grading before `</answer>`.

The `<READ>` and `</READ>` delimiters bound a variable-read request, analogous
to the existing `<TOOL>` and `</TOOL>` delimiters for tool references. Its body
identifies the model-defined variable to resolve; it does not contain an
asserted result value. At a complete valid read, pause generation, fetch the
assigned value, append it as runtime observation context, then resume the same
attempt. Reading does not execute the tool again or consume the correction
attempt. In automatic mode, valid explicit reads may return the same stored
value through the same handler; the boolean controls automatic successful-value disclosure.

Never accept a caller-supplied variable store or prepopulated local bindings.
The model produces all declarations and input bindings in `define`; the runtime
tracks those validated declarations and assigns actual tool results. Supplied
task context is not a supplied variable store.

Keep the variable store episode-local, with call identity and assignment
version/provenance. Preserve it into the retry consistently with fixture state;
clear it between episodes. Reads of undeclared, unresolved, failed, or invalidated
bindings return a value-free error. A failed call must not leave an older value
looking like its successful output. Specify rebinding and rollback behavior
before enabling state-changing fixtures.

With automatic injection disabled, result values must also stay out of echoed
command messages, knowns/evidence, summaries, and retry/verifier feedback until
read. Runtime grading may inspect the private result store, but model-visible
feedback must obey the same disclosure rule on both attempts. Execution errors
must be injected immediately without leaking the withheld value.

Implementation note: `Shared/UnigramByte/Detectors/AtomDelimiterDetector.cu`
currently recognizes `<TOOL>` as `AtomType::ATOM_TOOL`, described in
`Shared/UnigramByte/TokenLayout.hpp` as a tool identifier. That is not a complete
call dispatcher. `<READ>` needs its own recognized span and runtime semantics;
do not alias variable reads to tool identifiers. Resolve its representation,
tokenizer vocabulary/layout compatibility, and span projection explicitly.
Update `docs/GRIM_REASONING_DATA_CONTRACT.md` when these semantics are implemented.

## Selected pilot behavior

| Item | Decision |
| --- | --- |
| Self-correction RL branch | Existing `training.config.training_stage: "rl"` / `TrainingStage::RL`; stage parsing and preparation guard implemented, episode preparation and RL training dispatch pending |
| Trainable parameter selection | All policy-model parameters through existing registration; no RL-specific subset or layer freezing |
| Starting checkpoint | Best verified DDEA SFT checkpoint; exact ID pending |
| Tasks | Bounded tool-use and goal-following tasks across the families above; arithmetic is one fixture family |
| Attempts | Initial response plus at most one correction |
| Generated-token limit per attempt | 256 |
| Success grading | After `</answer>` is emitted, check goal completion, constraint compliance, and grounded use of required tool results; the verdict does not control the answer boundary |
| Initial failure | Append feedback permitted by the configured result-access mode and unmet criteria with the original goal available; generate one retry |
| Automatic tool-result injection | Successful values require READ by default (`false`); `true` auto-injects them. Errors auto-inject in both modes |
| Retry failure | End episode; no third attempt |
| Sampling method | Categorical |
| Temperature | 1.0 |
| Top-k | Disabled |
| Top-p | 1.0 (no truncation) |
| Min-p / typical sampling | Disabled |
| Repetition penalty | 1.0 (none) |
| Presence / frequency penalties | 0 |
| External logit biases | 0 |
| Beam search | Disabled |
| Forced structural tokens | Disabled |
| Dropout during generation and policy scoring | Disabled |
| Completion / grading boundary | Full generated `</answer>` marker, independently of runtime goal verification |
| Tool boundary | Pause on a complete validated call, execute and assign its output, apply configured disclosure, then resume the same attempt; exact call grammar/handoff remains to be wired |
| Read boundary | A complete valid `<READ>...</READ>` exposes the assigned variable value as context and resumes the same attempt |
| Other termination | EOS, token limit, or execution-budget exhaustion before `</answer>` is an incomplete attempt, not a successful answer completion |
| Episodes per task | 4 |
| Tasks per update | 16 |
| Completed episodes per update | 64 |
| Initial RAdamW learning rate | `3e-6` |
| Requested clipping threshold | 1.0; clipping-metric interpretation remains open below |
| Rollout reuse | None |

A first attempt closes at `</answer>` and is then graded. A passing grade
completes the episode; a failing grade triggers the one allowed retry. Runtime
goal checks must neither stop generation early nor extend it past `</answer>`.
An update contains 64 initial
attempts and 0–64 retries. Do not fabricate missing retries. This pilot uses
attempt-local rewards, not the full SCoRe two-stage/progress-bonus schedule or
PPO optimization epochs.

## Training-stage selection and trainable parameters

Use the existing `training.config.training_stage` selector: `"rl"` resolves to
`TrainingStage::RL`, alongside PT and SFT. This replaces the previously proposed
`self_correction_rl.enabled` boolean; do not add a second training-mode switch.
Other stages retain their configured paths without self-correction collection,
reference scoring, RL loss, or retry-baseline updates. Result disclosure remains
an independent planned `auto_inject_tool_results` boolean.

Implemented preparation boundary:

- `HyperparameterEnums.hpp` parses/serializes RL; existing enum values are preserved.
- Curriculum persistence, registry validation, and the Data Hub stage selector accept `rl`.
- Checkpoint filenames recognize `checkpoint_RL_*.grimckpt`; predecessor-stage selection uses SFT for RL initialization.
- `Startup/SlidingWindow.cu::applySlidingWindows()` selects PT/SFT/RL before mutating corpus rows. RL currently throws an explicit error requiring recorded episode batches because episode-to-payload preparation is not implemented. It does not perform PT windowing or SFT supervision projection.
- `Tests/test_rl_batch_preparation_host.py` passes focused host checks for stage round trips, checkpoint identity/predecessor selection, and preparation routing before corpus mutation. No training/CUDA target was built or run.

This is stage-selection scaffolding, not an executable RL training path.
Mode-aware startup planning/validation and the Phase 2 handoff remain pending.
The handoff should carry the resolved stage to the separate, proposed
`Phase2_RLTrainingLoop.cu/.hpp` forward/backward/update loop. Episode scheduling,
generation, retry selection, grading, and reward/advantage preparation belong
to a separate sequence that supplies a prepared, scored episode group.
The RL training loop consumes its microbatches, recomputes policy/reference
forwards, assembles RL loss, accumulates backward contributions, and commits one
optimizer update. Reuse existing execution/update plumbing without copying
the SFT epoch scheduler or its supervised-row normalization.

Include all policy-model parameters in the existing parameter registration and
optimizer path. Do not introduce a trainable-parameter allowlist, adapter-only
mode, selected-layer mode, or per-module RL freeze switches. Resolve any existing
freeze settings that would exclude policy parameters explicitly when enabling
this mode; reject conflicting configuration rather than silently training a
subset. Inclusion does not imply a nonzero gradient for every parameter on
every batch: gradients follow the actual connected objective.

Resolve policy and frozen-reference checkpoints by model identity through the
existing configured model store, including each model's `model.grimcfg` and
checkpoint artifacts. Do not add a separate RL checkpoint-path config control.
Pin the resolved reference identity at startup and persist/validate it on resume;
do not reselect a moving latest reference for each group. The separate reference
loading lifecycle remains implementation work.

The frozen SFT reference is a separate scoring copy, not part of the trainable
policy; it retains disabled gradients and is excluded from optimizer updates.
Tool execution, variable storage, reward checks, and recorded observations remain
outside autograd. Auxiliary-loss participation remains a separate objective
decision and must not become a parameter-freezing mechanism.

## 1. Add self-correction episode orchestration

### Implemented generation foundation

- [x] Resolve the complete configured span delimiter vocabulary against the loaded tokenizer during structured prefill, including continuation spans absent from the supplied input. Retain an immutable reverse lookup shared by payload/session/result; never persist resolved IDs in configuration, GRMT, or checkpoints.
- [x] Track generated opening/closing markers by definition and enclosing scope, including repeated spans. Retain absolute token ranges and invalid-marker events; reject ambiguous delimiter mappings. Opening markers, unmatched/mismatched closes, and stale close events do not stop generation.
- [x] Keep `InferenceGenerationSession` as data-only state operated on by free functions. Read vocabulary size and prefill geometry from the inference payload and sequence/cache limits from model configuration; do not cache those scalar values on the session.
- [x] Centralize return decisions in the classless `InferenceReturnPolicy.hpp`. Return at a valid configured close with `SpanBoundary` and `finished == false`; keep sampled EOS, token/context limits, and explicit finish distinct and terminal. Retain the return reason on `GeneratedSequence` and decode after the loop returns.
- [x] Preserve the live session, sampler/RNG, prefill, atom registries, retrieval state, and reusable device storage in `GenerationState` across span returns. Expose `resumePhase2PayloadInference()` to continue the same trajectory without re-prefill; acknowledge the delivered span event so it does not immediately yield again.
- [x] Read the existing `kv_cache.host_seqlen` directly in the inference loop as the processed-token cursor. On resume, process the pending closing marker and any queued tokens exactly once, in order, before sampling again. Keep this scheduling in `Phase2_InferenceLoop.cu`; no separate advance helper or duplicate processed-token counter.
- [x] Count only `TokenOrigin::SampledAction` toward the generation budget and minimum-length EOS suppression, carrying the count across segments. Runtime/supplied context and forced retrieval tokens consume context capacity only; retrieval replay selection uses remaining context capacity. Bound context by both the model sequence limit and KV-cache capacity.
- [x] Release the active session on terminal return, inspection completion, or advance failure; reject resume without a matching tokenizer and span-paused session. Beginning a new inference explicitly resets the previous live trajectory. Return cumulative trajectory snapshots rather than moving the live session out at a span boundary.
- [x] Pass focused host tests for delimiter scope/ambiguity, valid and malformed span boundaries, EOS versus span/limit reasons, two successive pause/resume segments, pending-marker processing, queued runtime tokens before the next sample, sampled/context accounting, and terminal cleanup. The trajectory test compiles the production scheduling loop with GPU forwarding substituted; the cached-forward host test also passes. No GPU/model/training binary was built or run for this work.

These are generation primitives, not a connected RL episode or tool loop.
The returned continuation is still cumulative from the original prefill boundary.
Runtime-context appending/atom remapping and the episode-specific answer-close,
execution, grading, and retry handlers remain below.

### Pending episode integration

- [ ] Add a validated `appendRuntimeContext()` free function beside the session operations: check channel lengths, vocabulary IDs, context capacity, and observation boundaries before mutation; merge/remap atom registries and aligned metadata; append with `RuntimeContext` provenance and no policy-score entries. Observation markers must not become generated grammar/dispatch events. Connect the adapter to the retained session and existing resume path.
- [ ] Wrap the existing generation entrypoint with the initial-attempt / outcome / optional-retry flow.
- [ ] Add the task/verifier adapter described above: return overall success, per-criterion outcomes, constraint violations, execution status, and model-visible feedback separately from private grading data.
- [ ] Bind available tool descriptions/parameters to the task prefix using the existing rendering path; record registry/tool versions and reject calls outside that task's available set.
- [ ] Connect generated call parsing and argument validation to controlled tool execution, adapting the existing command/result boundary where applicable. Do not treat free-form execute text as proof of execution.
- [ ] Add `auto_inject_tool_results` through the existing config schema/compiler/loading path and use the same resolved mode in rollout and inference; snapshot it for each episode group.
- [ ] Route execution/validation/READ errors to automatic feedback before applying the successful-result injection boolean. Test error injection under both values and on both attempts; no READ is required for errors.
- [ ] Implement declaration validation, tool-output assignment, and the episode-local variable store for both result-access modes.
- [ ] Add distinct `<READ>` / `</READ>` recognition and variable resolution, wiring request handling to the existing span pause/resume primitives; validate complete boundaries before servicing requests.
- [ ] Add a call/result handoff with conditional disclosure and an explicit read/result handoff. Tool and read pauses are not correction attempts; wire the existing sampled-token accounting to one shared 256-token budget per attempt.
- [ ] On initial failure, append permitted execution observations and unmet-criterion feedback with the original goal available. Retain prior generated history, variable bindings, and call/result associations without disclosing unread values.
- [ ] Define isolated fixture state and reset it between independent episodes. Within an episode, retry from the recorded post-attempt state unless the fixture explicitly rolls back and exposes that rollback; never silently replay side effects.
- [ ] Bound calls, observation/context size, and execution time per attempt. Treat tool/validation errors as observations the model can respond to; distinguish infrastructure failures from scored policy failures and replace invalid episodes before the update.
- [ ] Define the appended feedback/goal delimiters and the beginning of the next DDEA attempt using the existing rendering path.
- [ ] Connect retry feedback insertion to the runtime-context append adapter and existing cache-preserving resume path; feed all appended context before sampling the retry and initialize the retry's separate sampled-token budget explicitly.
- [ ] Close each attempt independently when the full `</answer>` marker is emitted, then grade its recorded trajectory. End the episode after a passing first attempt or after grading the retry; never stop generation because a runtime check reports success. Set the existing per-attempt generation limit to 256 tokens for this pilot.
- [ ] Configure the existing sampling pipeline for categorical sampling with the settings above; disable no-repeat n-gram blocking and minimum-length EOS suppression as additional distribution transforms.
- [ ] End normal generation after the full generated `</answer>` marker, configured EOS if present, or the token limit; also handle configured call/time-budget exhaustion explicitly. Handle the closing marker across token boundaries and retain all sampled marker tokens in the recorded action sequence and length count.
- [ ] Distinguish completion-marker, EOS, token-limit, and execution-budget termination in the episode result. At `</answer>`, freeze the grading record and evaluate the evidence available at that boundary; missing required results are unmet criteria, not a reason to continue generation or wait for later evidence to establish success.
- [ ] Treat EOS or budget exhaustion before `</answer>` as an incomplete failed attempt with zero reward, eligible for the usual retry when it is the initial attempt; retain the separate infrastructure-invalid episode handling.
- [ ] Add focused checks that early runtime success does not stop generation, `</answer>` closes both passing and failing attempts immediately, and missing-result or incomplete-answer cases cannot receive success credit.
- [ ] Disable forced structural-token insertion for this pilot; structural markers must be sampled from the policy.
- [ ] Group four episodes per task and 16 tasks per update under one unchanged policy version.
- [ ] Add focused tests for both boolean paths, no value leakage before READ in explicit mode (including retries/errors), correct variable resolution, failed/stale bindings, and reads that never re-execute tools.

Runtime outcome checks determine reward and retry selection after the answer
boundary; they do not determine when the model finishes its answer. An execution error
does not necessarily fail an attempt if the model recovers within its budget;
a successful command does not necessarily complete the goal. Final attempt
success comes from the common goal-check contract, not the tool's status alone.

## 2. Carry episode data into training

- [ ] Connect the retained `Phase2TextInferenceResult::generated_sequence` scores and atom side channels to the rollout-to-training adapter; the text-result handoff now preserves them.
- [ ] Add attempt boundaries, task/group identity, policy version, actual feedback context, and per-attempt outcome/reward to the episode data.
- [ ] Record result-access mode, variable declarations/assignments, read requests, assignment versions, and disclosure positions separately from private stored results; replay only values actually visible at each recorded prefix.
- [ ] Retain task family/verifier version, goal and tool snapshot, ordered calls with arguments, matched results/errors, environment-state evidence, and criterion outcomes. Store exact model-visible observation tokens in their causal positions for policy/reference replay; do not re-execute tools during loss computation.
- [x] Add explicit sampled-action versus forced-token provenance to generation output. `GeneratedSequence` owns aligned token origins and score positions; local retrieval's zero-score forced tokens are explicitly marked, so zero alone is never used as an action mask.
- [ ] Retain both failed and successful episodes until their group's single update completes.
- [ ] Adapt recorded episode tokens and existing atom metadata into the existing training payload path; add only the RL-specific masks/advantages needed by the objective.
- [ ] Mask task, tool descriptions, tool/read results, feedback, padding, and forced tokens from the direct policy objective. Sampled declarations, tool names/arguments, READ delimiters/variable references, and other generated continuation tokens remain actions and count toward length. During retry scoring, the initial attempt is context, not a recipient of retry advantage.

## 3. Implement attempt-local rewards and advantages

- [ ] Feed verified goal success from the common outcome contract into the rewards below for every task family. Keep per-check diagnostics separate from reward; do not substitute numeric answer matching or command success for goal success.
- [ ] Implement success-only length penalties with `lambda_L = 0.02 / 256 = 0.000078125` and `penalty(N) = min(0.02, lambda_L * N)`.
- [ ] Set initial-attempt reward to `1.0 - penalty(N1)` on success and exactly `0.0` on failure.
- [ ] Set retry reward to `0.75 - penalty(N2)` on success and exactly `0.0` on failure; count only the retry's own tokens.
- [ ] Count all sampled DDEA tokens, sampled structural markers, and the sampled completion terminator through the generated stop marker, including markers hidden from user display.
- [ ] Exclude prompt, tool descriptions/results, feedback, and runtime-injected/forced tokens from the length count; derive counts from token provenance, not displayed text.
- [ ] Normalize first-attempt rewards within each task's four episodes using unbiased sample variance, minimum standard deviation `0.10` (variance floor `0.01`), and advantage bounds `[-2, 2]`.
- [ ] Normalize every actual retry against one shared running baseline across tasks and optimizer updates; initialize `mu2 = 0.375` and `v2 = 0.140625`.
- [ ] Snapshot the retry baseline for the entire rollout batch and use that same snapshot for every retry advantage, including across microbatches.
- [ ] After scoring the batch, update retry statistics with EMA coefficient `0.10`; new statistics apply only to subsequent batches.
- [ ] Keep advantages detached and attempt-local. Retry success must not supply positive outcome credit to the failed initial attempt.
- [ ] Omit absent retries from statistics and loss terms; use the existing running baseline even when a task has only one retry.

The penalty is linear through 256 counted tokens and capped thereafter. The
pilot also uses a 256-token per-attempt generation limit and fixed loss scaling.

$$
p(N)=\min(0.02,\,0.000078125\,N),\qquad
R_{i,a}=\begin{cases}
b_a-p(N_{i,a}) & \text{if attempt }a\text{ succeeds},\\
0 & \text{otherwise},
\end{cases}
\quad b_1=1.0,\ b_2=0.75.
$$

Successful first attempts receive rewards in `[0.98, 1.0]`; successful retries
receive `[0.73, 0.75]`. Failure never receives a negative length-only reward.
These are raw reward ranges; separate attempt normalization does not guarantee
the same ordering of advantages or gradient magnitudes.

### First-attempt advantages

For each task, use its four first-attempt rewards (`n = 4`):

$$
\bar R_1=\frac{1}{n}\sum_i R_{i,1},\qquad
s_1^2=\frac{1}{n-1}\sum_i(R_{i,1}-\bar R_1)^2,
$$

$$
A_{i,1}=\operatorname{stopgrad}\left[
\operatorname{clip}\left(
\frac{R_{i,1}-\bar R_1}{\max(\sqrt{s_1^2},0.10)},-2,2
\right)\right].
$$

If all four first attempts fail, their rewards and advantages are zero.

### Retry advantages and running baseline

Initialize the baseline once for a new RL run:

$$
\mu_{2,0}=0.375,\qquad v_{2,0}=0.140625.
$$

For each actual retry, use the baseline snapshot from before this batch:

$$
A_{i,2}=\operatorname{stopgrad}\left[
\operatorname{clip}\left(
\frac{R_{i,2}-\mu_2}{\max(\sqrt{v_2},0.10)},-2,2
\right)\right].
$$

After all advantages have been scored, gather the actual retry rewards across
the entire 64-episode batch, not separately per task or microbatch. For `m`
actual retries, update the mean when `m >= 1`:

$$
\mu_2'=0.9\mu_2+0.1\bar R_2.
$$

Update the variance only when `m >= 2`, using unbiased sample variance:

$$
s_2^2=\frac{1}{m-1}\sum_{j=1}^{m}(R_{j,2}-\bar R_2)^2,
\qquad v_2'=0.9v_2+0.1s_2^2.
$$

With one retry in the entire batch, update only the mean and retain the previous
variance. With no retries, leave both statistics unchanged. Apply the standard
deviation floor when calculating advantages; the EMA variance update remains
the equation above. Update the baseline once per rollout batch, not once per
task, attempt, or microbatch.

| Actual retries for a task | Treatment |
| --- | --- |
| 0 | No retry loss or fabricated second attempt |
| 1 | Score against the existing shared running baseline |
| 2–4 | Score each against the same shared running baseline |

Under the initial baseline, a single successful retry has positive advantage
and a single failed retry has negative advantage. Retry outcomes never alter
first-attempt advantages.

## 4. Add policy scoring and loss

- [ ] Recompute recorded sampled-action probabilities through the existing parameter-connected shared forward; disable dropout specifically for this policy-scoring path.
- [ ] Score the same categorical model distribution used for rollout: temperature 1.0, no truncation, penalties, external biases, or forced structural tokens. Apply any fixed hard legal-token mask identically in rollout, current-policy scoring, and frozen-reference scoring.
- [ ] Obtain current-policy and frozen SFT-reference logits on the same recorded rollout prefixes through existing forward facilities; disable reference gradients.
- [ ] Implement exact forward categorical KL over all legal output IDs at each sampled-action position in both actual attempts, using FP32 log-softmax and KL accumulation.
- [ ] Apply the same hard legal-token mask to both distributions and normalize over that support; exclude illegal IDs from the KL sum.
- [x] Add authored runtime `training.config.rl_kl_coefficient: 0.02`, typed loading, finite/nonnegative validation, finalized-config export, and `rlTrainingHP()` accessors. Keep it outside the compiled model-geometry schema.
- [ ] Consume the resolved KL coefficient in the executable RL loss; pilot value is `0.02`, with adaptive KL disabled. The source pseudocode already reads the grouped config value.
- [ ] Add signed-advantage policy loss and reference KL to the existing autograd loss assembly with an explicit RL objective selection; retain the existing SFT objective.
- [ ] Sum action log probabilities and categorical KL over actual attempts and reduce both with the fixed denominator `256 * 64`; do not divide by realized completion length or actual attempt count.
- [ ] Omit absent attempts and empty action sets from sums without changing the fixed denominator.
- [ ] Apply the selected treatment of local retrieval and other existing auxiliary objectives in RL mode once resolved below.

$$
D_{\mathrm{KL},eat}
=\sum_{v\in V_{\mathrm{legal}}}P_\theta(v\mid h_{eat})
\log\frac{P_\theta(v\mid h_{eat})}{P_{\mathrm{SFT}}(v\mid h_{eat})}.
$$

With `B = 64` episodes, use:

$$
\mathcal L_{\mathrm{PG}}
=-\frac{1}{256B}\sum_{e,a} A_{ea}
\sum_{t\in M_{ea}}\log P_\theta(y_t\mid h_t),
$$

$$
\mathcal L_{\mathrm{KL}}
=\frac{0.02}{256B}\sum_{e,a}\sum_{t\in M_{ea}}D_{\mathrm{KL},eat},
\qquad \mathcal L=\mathcal L_{\mathrm{PG}}+\mathcal L_{\mathrm{KL}}.
$$

`e,a` ranges only over actual attempts. `M` contains sampled generated-action
positions, including sampled structural/stop tokens and excluding prompt,
tool descriptions/results, feedback, padding, and runtime-forced tokens.
Advantages are detached. The
current distribution remains differentiable throughout the categorical KL;
the reference distribution has no gradients. No gradient passes through the
discrete sampling decision or runtime outcome checker.

KL is exact over the legal vocabulary at each visited prefix; the prefixes
themselves are Monte Carlo rollout samples. Do not substitute sampled-token KL
or clipped log-probability ratios. Exact KL requires both models' vocabulary
logits at the scored positions.

For monitoring only, report `sum(KL) / actual scored generated-token count` in
nats per token (report unavailable for zero tokens). Keep this diagnostic
separate from the fixed-budget loss reduction and its `0.02` coefficient.

The existing cached forward requires `connect_parameter_graph == false`.
Collect with existing cached generation and recompute for gradients; this task
does not require a differentiable KV-cache implementation.

## 5. Integrate one update per episode group

Proposed forward/backward/update sequence:
[RL training loop pseudocode](GRIM_RL_TRAINING_LOOP_PSEUDOCODE.md).
Episode scheduling and retry collection remain separate from that loop.
The source scaffold now lives in
[`Phase2_RLTrainingLoop.hpp`](../resources/models/GRIM-text/training/Phases/Phase2_RLTrainingLoop.hpp)
and [`.cu`](../resources/models/GRIM-text/training/Phases/Phase2_RLTrainingLoop.cu).
Its entrypoint takes a preparation-owned group by const reference and currently
throws; the body documents pseudocode, not executable RL operations. Its current
success-result placeholder is `bool`; the update receipt remains a design
in the companion pseudocode document, not an implemented handoff type.

- [x] Add `TrainingStage::RL` / `training_stage: "rl"` to existing stage parsing, curriculum selection, checkpoint identity, and the batch-preparation switch; reject unsupported offline RL preparation before corpus mutation.
- [ ] Implement episode-to-payload preparation behind the RL branch; replace the explicit unimplemented guard only when recorded episode data can be prepared correctly.
- [ ] Make Phase 1 planning/validation and the Phase 2 handoff stage-aware so RL does not require prebuilt SFT payloads or an SFT epoch plan.
- [ ] Wire the existing `TrainingStage` enum through configuration, startup, the Phase 2 handoff, and training dispatch: `TrainingStage::RL` selects the complete RL branch, including the separate `Phase2_RLTrainingLoop.cu/.hpp` forward/backward/update loop consuming prepared, scored groups; other stages retain their existing training paths. Keep episode scheduling and retries in their separate sequence; do not introduce an RL enable boolean.
- [ ] Include all policy-model parameters through existing registration/optimizer dispatch; validate conflicting freeze settings and keep the frozen reference outside the optimizer.
- [x] Reject `TrainingStage::RL` with `lora_model=true` in `HyperParameters_GPU.hpp::validateRootConfigDocument()` for training execution. Config finalization rejects this before tokenizer work/model allocation; no separate Phase 1 guard is needed. LoRA-only assembly freezes base policy parameters. Inference and other training stages retain existing behavior. RMS-gamma and embedding-freeze checks remain pending.
- [ ] Add focused checks that non-RL stages perform no self-correction collection/reference work or EMA updates, and RL includes the complete policy parameter set without reference parameters.
- [ ] Pass the group's RL loss through existing backward, accumulation, clipping, and RAdamW dispatch with the pilot settings.
- [ ] Perform exactly one optimizer update after all 64 episode contributions; preserve episode weighting across microbatches.
- [ ] Preserve the single `1 / (256 * 64)` reduction across existing gradient accumulation; do not apply an additional token/attempt/microbatch mean.
- [ ] Mark the group consumed, advance the policy version, and reject unfinished/cached episodes from older versions.
- [ ] Add policy loss, KL, attempt rewards, and advantage values to existing logging without treating policy loss as CE/perplexity.
- [ ] Log first-attempt and retry goal success by task family, tool/argument failures, result-grounding failures, constraint violations, unmet criteria, and invalid infrastructure episodes so aggregate reward cannot hide arithmetic-only progress.
- [ ] Extend existing configuration only with new RL-specific settings.
- [ ] Persist retry EMA mean/variance alongside RL-specific reference identity/policy version through the existing checkpoint mechanism. Restore the EMA on resume rather than reinitializing it; discard incomplete groups rather than introducing a new episode-resume subsystem.
- [ ] Update the reasoning data contract for implemented feedback rendering/span changes and document the new objective at existing documentation locations.

## Remaining self-correction decisions

- [ ] Specify the replacement action approach and its task-input/runtime handoff.

- [ ] Decide supplied-context versus model-generated goals, including generated-goal grammar, commitment boundary, verification, and policy-loss participation.
- [ ] Supply the starting checkpoint ID.
- [ ] Select concrete registered tools, controlled fixtures, task-family mix, and executable goal checks for the first mixed rollout batch.
- [ ] Specify generated-call grammar, dispatch/result handoff, and feedback/next-attempt rendering through the existing payload/span path; update the reasoning data contract when implemented.
- [ ] Specify declaration/output-binding and READ-body syntax, rebinding/invalidation rules, read-request budgets, and tokenizer/span compatibility for the new delimiters. Keep both configured result-access paths supported.
- [ ] Set per-attempt call/time/context bounds and task-specific state inspection/reset semantics; define which infrastructure failures invalidate an episode.
- [ ] Decide whether local retrieval is fixed/supervised or included as a sampled RL action; specify existing auxiliary-loss participation.
- [ ] Resolve requested clipping threshold `1.0` against existing global RMS semantics; do not silently interpret it as global L2 or replace the clipping infrastructure.

Implementation order: task/tool fixtures and goal checks → call/result and
episode orchestration → episode-to-training data → rewards and advantages →
policy/reference scoring and loss → existing update-loop integration.
