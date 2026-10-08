# GRIM Define Stage — Design Decisions and Current Status

Status: design only. No variable runtime or staged-training implementation has
been introduced by this discussion.

The existing authoring contract is
[GRIM_REASONING_DATA_CONTRACT.md](GRIM_REASONING_DATA_CONTRACT.md).
This document records proposed runtime semantics and distinguishes agreed
decisions from questions that remain open.

## Scope

Turn declarations and literal assignments authored inside `<define>` into actual
local variable state, with an explicit lifetime and atomic commit boundary.

Included:

- definition syntax and literal values;
- staging, validation, and commit;
- ownership across model forward calls;
- definition failure semantics;
- staged multi-forward chronology and its training implications.

Excluded:

- atoms and atom integration;
- tool expressions and execution;
- tool-result binding;
- answer-reference resolution;
- a learned interface for reading variable values.

Atoms are not part of this variable system at this stage.

## 1. Existing Behavior

`define` is currently a text field rendered as a configured, supervised span.
The renderer and tokenizer do not create executable variable bindings.

Current arithmetic examples already use declarations and numeric assignments,
but those are authored text conventions, not runtime operations.

Relevant existing surfaces:

- [Configured span rendering](../resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp)
- [ConceptBlock source fields](../DataCollection/concept_block.hpp)
- [Training forward and loss](../resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)
- [Generation and cached continuation](../resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.cu)

## 2. Agreed Definition Syntax

```text
<define>
${starting_volume} -> 120;
${description} -> "water remaining";
${unresolved_quantity};
</define>
```

| Form | Meaning |
|---|---|
| `${name};` | Declare a local variable without an initialized value |
| `${name} -> literal;` | Declare and initialize a local variable |
| `"text"` | Text literal with JSON-style escaping |

Each definition ends with a semicolon.

Initializers are general literal values, including numbers and text. References
to other variables and evaluated expressions are not included in the agreed
initializer scope. The complete supported literal type set and numeric
representation remain undecided.

Text literals use double quotes and JSON-style escaping. Parsing must respect
quoted-string boundaries:

```text
${description} -> "a semicolon; and literal </define> text";
```

The embedded semicolon and closing-marker text are string content, not
structural boundaries.

The runtime's handling of generated section-marker token IDs versus marker text
inside quoted strings still needs an exact parser/decoder contract.

## 3. Atomic Definition Transaction

Definitions are staged until the complete `<define>` section closes and
validates.

```text
Open <define>
    -> collect and parse staged definitions
    -> validate the complete transaction
Close </define>
    -> commit all definitions together
```

No definition becomes committed merely because its semicolon has been emitted.
If one definition fails, the entire transaction fails.

Required invariants:

- No partial commit.
- Staged definitions are not available as committed local state.
- Rejection leaves previously committed state unchanged.
- An unfinished section commits nothing.
- A valid closing boundary commits the complete transaction before continuation
  is allowed to cross that boundary.
- Separate generated sequences have isolated environments.

The declaration state must distinguish an uninitialized variable from an
initialized variable. An uninitialized declaration is valid; it is not a
definition failure.

## 4. Ownership and File Separation

The committed environment belongs to the generated sequence/session. It
survives individual forward calls without retaining graph-owned tensors.

The responsibilities must remain separate:

| Component | Responsibility |
|---|---|
| Definition parser | Recognize generated syntax and produce staged declarations and typed literals |
| Definition transaction | Hold uncommitted definitions and validate the section as a whole |
| Local variable environment | Own committed variable identities, values, and initialized/uninitialized state |
| Sequence coordinator | Own stage chronology and authorize the commit boundary |

These are responsibility boundaries, not finalized filenames or class names.
The implementation should use cleanly separated files rather than embedding
the parser and environment inside the inference loop.

The variable environment must not be owned by the tokenizer, atom detector,
AtomTable, transient forward outputs, or autograd tape. It must not be a shared
global dictionary.

The KV cache and variable environment are distinct:

- KV cache preserves preceding model token context.
- Variable environment preserves committed declaration meaning.

## 5. Failure Handling

Model-authored definition rejection is an explicit unsuccessful sequence
outcome, not a process crash or a successful response containing an error
sentence.

Parser progress must distinguish:

| Outcome | Meaning |
|---|---|
| Pending | More generated text could complete valid syntax |
| Rejected | A definite violation has been detected |
| Committed | The closing boundary and complete transaction were accepted |

A fragment such as `${description} -> "water` is pending, not immediately
invalid. If generation terminates while the section remains pending, report an
incomplete definition and commit nothing.

For a rejected generated transaction:

1. Discard all staged definitions.
2. Preserve any previously committed environment.
3. Mark that sequence attempt failed.
4. Stop its generation.
5. Preserve the partial trace and rejection reason for diagnostics.

Do not skip invalid declarations, manufacture values, silently repair syntax,
automatically retry, or continue by using old state as a substitute for the
rejected transaction.

Failures should identify the stage, reason, relevant statement/variable when
available, and location in generated text. Exact error types and location
conventions remain to be designed.

Internal parser, storage, or commit invariant violations are system faults and
must surface through the established exception/error boundary. They must not
be mislabeled as bad model output.

## 6. Training Semantics

### Current teacher-forced training

The model predicts tokens against an authored sequence. A wrong definition
prediction receives loss, but the next input remains the authored token.

Consequently, a wrong prediction does not stop the existing teacher-forced
sequence or remove downstream supervision.

If a staged teacher-forced runtime is introduced, its definition transaction
must explicitly consume authored definitions. Invalid authored definitions are
dataset errors, not model prediction failures.

### Generated-definition rollouts

A rollout consumes the model's actual generated definitions. A rejected
transaction terminates that rollout at the definition boundary.

Do not silently replace rejected generated definitions with authored ones and
claim the same rollout succeeded.

Multiple forward calls alone do not turn teacher-forced training into rollout
training. Parsing and committing discrete generated text does not by itself
provide a gradient through validation or rejection.

Also, syntactic validity does not prove task correctness. A valid literal
assignment can still contain the wrong value. The objective that teaches
correct definitions remains a separate training-design question.

## 7. Staged Multi-Forward Chronology

Prompt prefill and KV-cached continuation are the agreed foundation for
generating the definition stage:

```text
Prompt prefill
    -> establish KV context

Cached generation through preceding continuation and <define>
    -> collect the pending definition transaction

Accept structural </define>
    -> validate the entire transaction
    -> atomically commit, or terminate the failed sequence

Process the closing token into the KV cache
    -> next model decision has completed textual context
    -> committed local state is available at the boundary
```

There is no inherent need to discard the cache, replay the prefix, or perform
another prompt prefill when a definition commits. The commit is a host/runtime
state transition between model decisions, not itself a neural forward pass.

The precise ordering requirement is:

> Before sampling the next decision after the definition boundary, validation
> and commit must have succeeded, and the closing token must have been processed
> into the causal model context.

Making committed variables available to the coordinator does not yet establish
how a future forward consumes them. That consumer interface is outside this
definition-only scope and has not been designed.

### Existing cache limitation for training

The existing
[shared-forward KV-cache interface](../resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.hpp)
requires parameter-graph connection to be disabled.

It provides the chronology for generated rollouts, but it is not currently a
graph-connected staged-training cache. We have not decided how to train through
the staged computation.

Options discussed, not selected:

- Generate a cached rollout and apply a separately specified learning objective.
- Recompute causal context in a supervised training pass with explicit
  definition visibility and nonduplicated targets.
- Design a differentiable context handoff between training stages.

No cache detachment, gradient policy, loss normalization, or optimizer timing
change has been approved.

## 8. Current Decision Status

### Settled

- Design only; no implementation yet.
- Define-only scope, independent of atoms.
- Declaration and literal-initialization syntax.
- Numbers and text are supported literal categories in the proposed design.
- Double-quoted text with JSON-style escaping.
- Whole-section atomic commit at `</define>`.
- One invalid definition rejects the entire transaction.
- Explicit failure without silent repair or partial state.
- Sequence/session ownership across forward calls.
- Separate parsing, transaction, environment, and orchestration responsibilities.
- Prefill once and retain KV context across the generated definition boundary.
- Teacher-forced prediction errors and generated-rollout rejection are different
  events.

### Still Open

- Variable-name grammar and case sensitivity.
- Duplicate declaration and reassignment policy.
- Complete literal type set and numeric precision/range.
- Limits on names, literals, declarations, and staged text.
- Whether later definition sections can extend or modify committed state.
- Exact incremental decoding and structural-marker recognition.
- Exact failure payload and source-location conventions.
- Environment lifetime across separate generation passes.
- Selection of teacher-forced staging versus generated-rollout training.
- Learning objective and gradient handling for multi-forward training.

The next design step is to settle the definition grammar and environment
invariants, then select the training semantics before writing implementation
code.
