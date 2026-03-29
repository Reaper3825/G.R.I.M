You are updating the GRIM Execution Block plan and implementation notes.

Your job is **not** to redesign the system from scratch.
Your job is to make the existing execution block plan **correct, internally consistent, future-facing, and aligned with the actual execution model already present in code**.

Treat the current CUDA file as the source of truth for what already exists and what must be fixed surgically, not replaced wholesale. The current implementation is in `execution_block_GPU.cu`. :contentReference[oaicite:0]{index=0}

# Mission

Refactor the plan so the Execution Block is treated as what it actually is:

- a **deterministic register-machine execution core**
- with **authoritative slot state**
- with **hard fail-fast invariants**
- and only **gradient plumbing where it genuinely supports learning**

Do **not** describe it like a generic neural layer.
Do **not** soften hard guarantees into warnings.
Do **not** propose backwards-compatibility scaffolding.
Do **not** add filler architecture prose.

The output should be a **clear corrective plan** for the existing system.

---

# Non-negotiable framing

The updated plan must explicitly recognize these truths:

1. **ExecutionMemory is authoritative state**
   - `M.values` is the scalar truth.
   - `M.state_embeds` is the embedding view of state.
   - `valid_mask` defines initialized slots.
   - The model is not the source of truth for numeric state. Memory is.

2. **This is a register machine**
   Each execution step is:
   - choose arg1
   - choose arg2
   - read slots
   - choose op
   - compute result
   - choose exactly one write slot
   - mutate exactly one authoritative slot
   - update embedding/key state for that slot

3. **Fail-hard behavior is intentional**
   The system must continue to treat the following as hard failures, not soft warnings:
   - invalid slot
   - missing slot mapping
   - slot read before initialization
   - invalid softmax
   - entropy collapse if configured as fatal
   - write collapse if configured as fatal
   - multi-slot mutation
   - transition error over threshold

4. **No fake “the model reasoned it internally” language**
   If the state transition is not represented in authoritative execution state, it did not happen.

5. **Future-facing direction is slot-space architecture**
   This is not “number vs non-number.”
   This is “slot type vs slot type.”
   `<NUM>` is only the first atom class.
   The plan must preserve future expansion to other authoritative slot spaces later.

---

# What is wrong in the current conceptual plan

You must correct the plan around these concrete issues:

## 1. Transition validation is conceptually broken
The current logic effectively computes a “hard transition error” using `v_out` against itself in one path, which cannot validate anything meaningful.

The corrected plan must require:

- explicit recomputation of the expected result from:
  - selected arg1 value
  - selected arg2 value
  - selected op id
- comparison of:
  - executed result
  - expected deterministic result
- hard failure if absolute error exceeds threshold after warmup

State clearly:
**transition validity must compare executed state against recomputed machine truth, not against itself.**

---

## 2. The system still mixes deterministic execution with probabilistic training language in the wrong places
The current setup uses soft distributions for learning but hard argmax for execution.

That is acceptable **only** if the plan clearly separates:

- **execution path** = authoritative, discrete, single-path
- **training path** = auxiliary gradient approximation

The updated plan must explicitly say:

- forward execution is discrete
- backward learning may use STE or soft surrogate paths
- surrogate gradients must never be mistaken for authoritative state updates
- any plan text that implies soft distributions are “the executed state” is wrong and must be removed

---

## 3. Write-slot learning is over-described as a distribution problem when execution is actually single-slot mutation
The plan must stop treating write behavior like a generic dense attention problem.

The corrected plan must state:

- only one value slot may be mutated per step
- write-slot selection is a routing problem, not a blended write problem
- write entropy regularization is secondary, not the core correctness mechanism
- correctness comes from:
  - selecting the correct slot
  - mutating only that slot
  - leaving all other authoritative slots unchanged

---

## 4. State integrity is being checked after mutation, but the plan must elevate the invariant
The plan must explicitly define this invariant:

> Per execution step, exactly one authoritative value slot may change, and all non-target value slots must remain numerically unchanged within tolerance.

And it must say that:
- post-write delta checks are required
- multi-slot mutation is a hard failure
- non-write-slot drift is a correctness violation, not just a training penalty

---

## 5. The current plan is missing a true reasoning-state feedback path
Right now the implementation has trace accumulation and injection, but the plan must acknowledge that this is not yet a full internal reasoning loop.

The corrected plan must identify the missing capability precisely:

- execution results need to feed back into subsequent step decisions in a stronger and more explicit way
- a dedicated **reasoning state** or **execution state summary** should exist as an evolving per-step latent
- this state must influence:
  - arg selection
  - op selection
  - write-slot selection
- simple one-token injection alone is not enough as the long-term design

Do **not** turn this into vague “add chain-of-thought” language.
Keep it mechanical and architectural.

---

# What the updated plan must add

Produce the revised plan so it includes these sections.

## Section A — Correct execution model
Define the execution block as:

- deterministic execution core
- authoritative slot memory
- one-step register-machine transition
- discrete forward path
- surrogate backward path only for learning

## Section B — Hard invariants
List the exact invariants that must always hold:
- only valid value slots are readable
- only value slots `[S, V)` are writable
- exactly one authoritative slot mutation per step
- no uninitialized slot reads
- result must match deterministic recomputation
- non-target slots must remain unchanged
- invalid numeric states crash immediately

## Section C — Immediate surgical fixes
Require the plan to specify concrete near-term fixes:

1. replace bogus transition self-comparison with recomputed-op validation
2. make executed op correctness explicit and hard-checked
3. keep STE clearly labeled as gradient approximation only
4. tighten write-slot correctness language around single-slot mutation
5. define state-integrity checking as a first-class invariant, not a side metric

## Section D — Missing medium-term capability
Require the plan to identify the next real architectural step:

- add a persistent per-step reasoning/execution summary state
- feed that state back into future execution decisions
- strengthen state reinjection beyond a single result-slot injection
- preserve hard authoritative slot ownership

## Section E — Future scaling rule
Require the plan to state that this design is the first instance of a broader pattern:

- slot-authoritative execution spaces
- `<NUM>` is only one atom family
- later spaces may include object/tool/other slot types
- ownership boundaries must remain explicit
- semantic hidden state must not become authority over numeric truth

---

# Style requirements for the rewritten plan

- Be blunt.
- Be technical.
- No motivational filler.
- No generic “consider” language.
- No hedging.
- No backwards-compatibility padding.
- No pretending the current system is more complete than it is.
- No vague “reasoning” language without defining state, mutation, and feedback.

When identifying what is missing, use direct language like:
- “missing”
- “incorrect”
- “not authoritative”
- “must be replaced”
- “must remain hard-fail”

---

# Required output format

Return the result as a structured engineering correction document with these exact headings:

1. `## What the Execution Block Actually Is`
2. `## Hard Invariants`
3. `## Immediate Corrections Required`
4. `## What Is Still Missing`
5. `## Forward-Compatible Design Rule`

Under each heading, write the corrected plan content directly.

Do not output code.
Do not output commentary about the prompt.
Do not summarize.
Just produce the corrected engineering plan.