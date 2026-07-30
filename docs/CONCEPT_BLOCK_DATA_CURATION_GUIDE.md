# Concept Block Data Curation Guide

> **Audience:** Dataset authors preparing `concept_blocks.jsonl` for GRIM-text training.
> This is a **data guide**, not a code guide. It tells you what to write, what the
> system expects, and what will crash if you get it wrong.

> **Storage transition:** `concept_blocks.fb` is now the authoritative DataHub
> and training dataset. The JSONL shape documented below remains the legacy
> authoring/import format. DataHub imports it when the FlatBuffer is missing or
> older, then writes the versioned `GRCB` FlatBuffer. Training refuses to use a
> stale FlatBuffer when a newer JSONL is present.

---

## Table of Contents

1. [What Is a Concept Block](#1-what-is-a-concept-block)
2. [How the System Uses Concept Blocks During Training](#2-how-the-system-uses-concept-blocks-during-training)
3. [JSON Schema Reference](#3-json-schema-reference)
4. [Slot Assignment Rules](#4-slot-assignment-rules)
5. [Execution Steps](#5-execution-steps)
6. [Selector Supervision (Advanced)](#6-selector-supervision-advanced)
7. [Worked Examples](#7-worked-examples)
8. [Canonical Text Rendering](#8-canonical-text-rendering)
9. [Validation Rules That Will Crash on Bad Data](#9-validation-rules-that-will-crash-on-bad-data)
10. [Configuration That Affects Your Data](#10-configuration-that-affects-your-data)
11. [Common Mistakes](#11-common-mistakes)
12. [Checklist Before Submitting Data](#12-checklist-before-submitting-data)

---

## 1. What Is a Concept Block

A concept block is a **structured training example** that teaches the model to
perform explicit numeric reasoning using an internal register machine called the
**Execution Block**.

Each concept block encodes:

| What | Purpose |
|------|---------|
| A natural-language **question** | The prompt the model reads |
| **Initial numeric values** (`state_0.atoms`) | Literal numbers that appear in the question/context and are loaded into named register slots |
| **Execution steps** | A ground-truth program: "read slot A and slot B, apply operation, write result to slot C" |
| An **answer** | The expected natural-language output containing the computed result |
| Optional **explanation** lines | Chain-of-thought reasoning steps |

The model does **not** see the execution program at inference time. During
training, the program provides **teacher supervision** so the model learns
which registers to read, which operation to apply, and where to store results.
At inference time, the model must learn to select those operations on its own.

---

## 2. How the System Uses Concept Blocks During Training

### 2.1 Data Loading Pipeline

```
concept_blocks.jsonl
        │
        ▼
┌─────────────────────────────┐
│  Parse JSON                 │  buildStructuredExecutionRecord()
│  Resolve arg slots          │  Value-based or explicit arg_slots
│  Validate result            │  Throws on mismatch
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Render canonical text      │  renderWithSpans()
│  Track byte offsets of each │  Each bootstrap literal gets a
│  bootstrap literal          │  RenderedLiteralSpan with exact
│                             │  byte range [start, end)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Tokenize rendered text     │  UniByte tokenizer
│  Detect ATOM_NUM tokens     │  Numbers → <NUM> placeholder with
│  Record StructuralSpans     │  numeric_value side channel
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Compile execution payload  │  Match each bootstrap literal's
│  via byte-offset            │  byte range against the tokenizer's
│  intersection               │  StructuralSpan content_offset.
│                             │  Assigns compiled SlotIndex to token position.
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Write GRMT binary          │  Per-sequence: token_exec_slots,
│  training data              │  transition_targets, compiled slot
│                             │  and transition bindings
└─────────────────────────────┘
```

### 2.2 What Happens at Train Time

1. **Bootstrap**: For each row in the batch where `execution_active = true`,
   the system copies the literal numeric values from the tokens marked in
   `token_exec_slots` into the register file (`ExecutionMemory.values[slot]`).

2. **Execute Steps**: The Execution Block runs `K` steps (configured by
   `execution_block.num_steps`). Each step:
   - Selects two operand slots via learned attention over register state embeddings
   - Selects an operation (+, −, ×, ÷) via a learned distribution
   - Computes the result and writes it to a learned destination slot
   - `TransitionInvocation` targets provide the executable transition and its
     ordered argument/result slot identities

3. **Cross-Attention Read**: After execution, subsequent encoder layers can
   read from the register file via cross-attention.

4. **Loss Computation**: The execution block contributes several loss terms:
   - **Structured CE**: Cross-entropy on argument, transition, and result-slot
     selection after opaque identities are lowered through per-row bindings
   - **Entropy regularization**: Prevents distribution collapse
   - **Causal consistency**: Validates multi-step sequential reasoning
   - **Selector supervision** (when enabled): Teaches the decode-time slot
     selector which slot to bind to `<NUM>` tokens

5. **Language Modeling**: The rest of the model predicts the answer tokens
   via standard next-token prediction. Good register use improves answer
   quality, providing implicit end-to-end gradient pressure.

---

## 3. JSON Schema Reference

Each line in `concept_blocks.jsonl` is a JSON object. Here is the complete
schema with required and optional fields:

```jsonc
{
    // ─── Optional metadata ───
    "name": "addition_basic_01",       // Optional. Human-readable block name.

    // ─── Required: the question ───
    "question": "What is 42 plus 17?", // Required. The problem statement.

    // ─── Required for execution: initial state ───
    "state_0": {
        "type": "arithmetic",          // Optional. Semantic type tag.
        "atoms": [42, 17]              // Required. Bootstrap literal values.
                                       // These MUST appear in canonical text
                                       // so the tokenizer can detect them.
                                       // Order determines slot assignment:
                                       //   atoms[0] → slot base_slot + 0
                                       //   atoms[1] → slot base_slot + 1
                                       //   ...
    },

    // ─── Required for execution: computation steps ───
    "execution": [
        {
            "op": "add",               // Required. Operation: "add"|"sub"|"mul"|"div"
            "args": [42, 17],          // Required. Operand values (min 2).
            "arg_slots": [0, 1],       // Optional. Explicit slot indices for args.
                                       // If omitted, resolved by value matching.
            "result": 59               // Required. Expected scalar output.
        }
    ],

    // ─── Optional: final state ───
    "state_1": {
        "result": 59                   // Optional. Final computed value,
                                       // rendered as "STATE1 result=59"
    },

    // ─── Optional: chain-of-thought ───
    "explanation": [                   // Optional. Array of reasoning strings.
        "We need to add 42 and 17.",   // Rendered as "EXP: ..." lines.
        "42 + 17 = 59."
    ],

    // ─── Required: the answer ───
    "answer": "The answer is 59."      // Required. Expected model output.
}
```

### Field aliases

- `"explanation"` and `"intermediates"` are interchangeable; the builder
  checks for both and uses whichever is present.

---

## 4. Slot Assignment Rules

### 4.1 How Slots Work

The execution block has a configurable number of **register slots** (`num_slots`,
default 4). Slots are divided into two ranges:

| Range | Name | Purpose |
|-------|------|---------|
| `[0, S)` | Scratch slots | Internal working memory (not user-addressable). `S` = `num_scratch_slots`, default 0. |
| `[S, V)` | Value slots | User-addressable registers that hold numeric values. `V` = `num_slots`. |

With the default configuration (`num_scratch_slots = 0`, `num_slots = 4`),
all 4 slots (0, 1, 2, 3) are value slots.

### 4.2 Bootstrap Slot Assignment

**Your `state_0.atoms` array determines which slots are initialized:**

```
atoms[0] → slot base_slot + 0
atoms[1] → slot base_slot + 1
atoms[2] → slot base_slot + 2
...
```

`base_slot` defaults to 0 (configurable via `GRIM_CONCEPT_EXEC_BASE_SLOT` env
var, but you should leave it at 0 unless you have a specific reason).

**Rules:**

- Every atom in `state_0.atoms` **must be a number** (integer or float). Non-numeric entries crash the builder.
- Every atom gets its own slot. **No two atoms can share a slot.**
- The number of bootstrap atoms plus the number of execution steps must not
  exceed `num_slots`. Each execution step writes its result to a new slot
  (the next available slot after the bootstrap slots).
- Bootstrap values must appear **literally** in the rendered canonical text
  so the tokenizer's atom detector can find them. The builder verifies this
  at the byte level and crashes on any mismatch.

### 4.3 Slot Numbering for Execution Steps

Slot numbers in source examples are authoring-local ordinals only. The durable
contract stores opaque `SlotId` identities and a per-row
`CompiledSlotBinding`; neither the teacher nor runtime may infer meaning from
the ordinal, a `STATE` label, or a dense `SlotIndex`.

Each execution step's result is automatically written to the next slot after
all bootstrap slots:

```
If state_0.atoms has 2 values (slots 0 and 1):
  Step 0 writes to slot 2
  Step 1 writes to slot 3
  ...
```

You do **not** specify a result slot in your JSON — its opaque `SlotId` is
derived automatically and placed in `TransitionInvocation.results`.
If you need more steps than available slots, increase `num_slots` in config.

### 4.4 Non-State-Bearing Tokens

Every token position that is **not** a bootstrap literal gets a slot assignment
of `-1`, meaning "this token does not initialize any register." This is
assigned automatically. You do not control it in your JSON.

---

## 5. Execution Steps

### 5.1 Supported Operations

| `op` string | Current executor transition | Dense `TransitionIndex` |
|-------------|-----------------------------|-------------------------|
| `"add"` or `"+"` | Addition | 0 |
| `"sub"` or `"-"` | Subtraction | 1 |
| `"mul"` or `"*"` | Multiplication | 2 |
| `"div"` or `"/"` | Division (safe — crashes on zero divisor) | 3 |

There are exactly 4 operations. No other op strings are accepted.

These numbers are positions in the current arithmetic dispatcher, not semantic
operation identities. The compiled payload assigns an opaque `TransitionId`
and lowers it through `CompiledTransitionBinding`. Future tools, model routes,
and physical actions use the same `TransitionInvocation` representation with
their own signatures; slot payload composition remains an `AtomTable` concern.

### 5.2 Arg Resolution

The builder needs to know **which slots** to read as arg1 and arg2 for each
execution step. There are two mechanisms:

#### Method A: Explicit `arg_slots` (preferred for clarity)

```json
{
    "op": "add",
    "args": [42, 17],
    "arg_slots": [0, 1],
    "result": 59
}
```

`arg_slots[0]` and `arg_slots[1]` are indices into the slot value array
(0-indexed relative to the bootstrap set). Even with explicit slots, the
builder **validates the result**: it computes `op(slot_values[0], slot_values[1])`
and verifies it matches `result`. If it doesn't, the builder crashes.

#### Method B: Value-based resolution (convenient for simple cases)

```json
{
    "op": "add",
    "args": [42, 17],
    "result": 59
}
```

Without `arg_slots`, the builder:

1. Finds all slots whose current value equals `args[0]` → candidate set C₁
2. Finds all slots whose current value equals `args[1]` → candidate set C₂
3. Tries every `(c1, c2)` pair from C₁ × C₂
4. Accepts the **unique** pair where `op(slot_values[c1], slot_values[c2]) == result`
5. **Crashes if zero pairs match** (bad data)
6. **Crashes if multiple pairs match** (ambiguous — use `arg_slots` instead)

### 5.3 When to Use Explicit `arg_slots`

Use `arg_slots` when:
- Two bootstrap values are identical (e.g., `atoms: [5, 5]`) — otherwise the
  builder can't distinguish slot 0 from slot 1
- An intermediate result happens to equal a bootstrap value
- You want deterministic, unambiguous data regardless of numeric coincidence

### 5.4 Multi-Step Execution

Each step reads from the slot state that exists **at that point in the program**:

```json
{
    "state_0": { "atoms": [10, 3, 2] },
    "execution": [
        { "op": "mul", "args": [10, 3], "result": 30 },
        { "op": "add", "args": [30, 2], "result": 32 }
    ]
}
```

Step 0: reads slots 0 (=10) and 1 (=3), writes 30 to slot 3.
Step 1: reads slot 3 (=30) and slot 2 (=2), writes 32 to slot 4.

**Important:** `num_steps` in config must be ≥ the number of steps in your
longest concept block. `num_slots` must be ≥ (number of atoms + number of steps).

For the example above: need `num_slots >= 5` and `num_steps >= 2`.

---

## 6. Result Emission

The old decode-time slot selector and its `slot_selection_targets` channel are
deleted. A completed execution result is exposed only from the explicit result
slot of the final `TransitionInvocation` after the learned stop controller
chooses `STOP`. Do not recreate an execution-state selector to infer this slot.

---

## 7. Worked Examples

### 7.1 Simple Addition

```json
{
    "name": "add_basic",
    "question": "What is 42 plus 17?",
    "state_0": {
        "type": "arithmetic",
        "atoms": [42, 17]
    },
    "execution": [
        { "op": "add", "args": [42, 17], "result": 59 }
    ],
    "state_1": { "result": 59 },
    "answer": "The answer is 59."
}
```

**What happens:**
- Slot 0 ← 42, Slot 1 ← 17
- Step 0: read slot 0 + slot 1 = 59, write to slot 2
- Model predicts answer "The answer is 59."

### 7.2 Multi-Step: Multiplication Then Addition

```json
{
    "name": "multi_step",
    "question": "A box has 6 rows of 8 apples, plus 5 loose apples. How many total?",
    "state_0": {
        "atoms": [6, 8, 5]
    },
    "execution": [
        { "op": "mul", "args": [6, 8], "result": 48 },
        { "op": "add", "args": [48, 5], "result": 53 }
    ],
    "state_1": { "result": 53 },
    "explanation": [
        "First multiply: 6 rows times 8 apples = 48 apples.",
        "Then add the 5 loose apples: 48 + 5 = 53."
    ],
    "answer": "There are 53 apples total."
}
```

**Config requirement:** `num_slots >= 5` (3 bootstrap + 2 steps), `num_steps >= 2`.

### 7.3 Division With Explicit arg_slots

```json
{
    "name": "divide_explicit",
    "question": "Split 100 cookies among 4 children equally. How many each?",
    "state_0": {
        "atoms": [100, 4]
    },
    "execution": [
        { "op": "div", "args": [100, 4], "arg_slots": [0, 1], "result": 25 }
    ],
    "answer": "Each child gets 25 cookies."
}
```

### 7.4 Subtraction With Identical Values

```json
{
    "name": "identical_values",
    "question": "You had 10 apples and ate 10. How many left?",
    "state_0": {
        "atoms": [10, 10]
    },
    "execution": [
        { "op": "sub", "args": [10, 10], "arg_slots": [0, 1], "result": 0 }
    ],
    "answer": "You have 0 apples left."
}
```

**Why `arg_slots` is mandatory here:** Both slots contain 10. Without explicit
indices, the builder cannot distinguish which is the minuend and which is the
subtrahend. The builder would crash with an "ambiguous" error.

### 7.5 Non-Execution Row (Plain Text)

```json
{
    "question": "What color is the sky?",
    "answer": "The sky is blue."
}
```

No `state_0` or `execution` → `execution_active = false`. This row trains
language modeling only. You can mix execution and non-execution rows freely
in the same `concept_blocks.jsonl`.

---

## 8. Canonical Text Rendering

Your JSON is **rendered** into a canonical text format before tokenization.
Understanding this format helps you write data that tokenizes correctly.

**Rendering template:**

```
[[name]]
Q: question
STATE0 type=type_tag atom0 atom1 atom2 ...
EXEC op arg0 arg1 => result
EXEC op arg0 arg1 => result
STATE1 result=final_result
EXP: explanation line 1
EXP: explanation line 2
A: answer
```

**Example for the simple addition block:**

```
[[add_basic]]
Q: What is 42 plus 17?
STATE0 type=arithmetic 42 17
EXEC add 42 17 => 59
STATE1 result=59
A: The answer is 59.
```

**Critical for atom detection:** The bootstrap literal values (`42`, `17`)
must appear literally in the `STATE0` line. The atom detector scans this
text, finds the numbers, and creates `<NUM>` tokens with the corresponding
numeric values in the side channel. The compilation step then matches those
`<NUM>` token positions back to bootstrap binding slots via byte offsets.

**What can go wrong:** If your `question` text contains the same number that
appears in `state_0.atoms`, the atom detector will find multiple `<NUM>` tokens
for that value. This is fine — the compilation uses **byte-offset intersection**
with the `STATE0` line specifically, not document-order matching. Only the
`<NUM>` tokens whose byte ranges fall within the rendered `STATE0` bootstrap
spans are assigned to slots.

---

## 9. Validation Rules That Will Crash on Bad Data

The builder and batch pipeline enforce these rules at data load time.
Violations produce an exception with a descriptive error message.

### Builder-Level Crashes

| Rule | Error |
|------|-------|
| `execution` present but no `state_0.atoms` | "execution steps present but no state_0.atoms" |
| `state_0.atoms[i]` is not a number | "state_0.atoms[i] is not a number" |
| Duplicate slot_id in bootstrap bindings | "duplicate slot_id N in bootstrap bindings" |
| Execution step has fewer than 2 args | "execution step N needs >= 2 args" |
| Unknown op string | "unknown op 'xyz'" |
| `arg_slots` index out of range | "arg_slots[0]=N out of range" |
| Explicit arg_slots produce wrong result | "arg_slots [A,B] produce X but expected Y" |
| Value-based resolution finds zero matches | "no (slot1, slot2) pair produces expected result" |
| Value-based resolution finds multiple matches | "ambiguous: N (slot1, slot2) pairs" |
| Division by zero in execution step | "division by zero" |
| Execution-active row with zero bootstrap bindings | "execution-active row has zero bootstrap bindings" |

### Compilation-Level Crashes

| Rule | Error |
|------|-------|
| Bootstrap literal doesn't match any `<NUM>` token | "bootstrap binding N matched zero ATOM_NUM tokens" |
| Bootstrap literal matches multiple `<NUM>` tokens | "bootstrap binding N matched M ATOM_NUM tokens (must be exactly 1)" |
| Two bootstrap bindings claim the same token position | "token_pos N claimed by multiple bootstrap bindings" |
| Atom content bytes don't match rendered literal bytes | "coordinate mismatch at binding N" |
| Rendered literal spans overlap or are non-monotonic | "renderer produced overlapping or non-monotonic spans" |

### Batch-Level Crashes

| Rule | Error |
|------|-------|
| `token_to_slot_map` slot_id outside `[0, num_slots)` and not -1 | "slot_id=N at position P out of range" |
| `transition_targets` count doesn't match config `num_steps` | Validation failure |
| Slot or transition identity has no compiled row binding | Validation failure |

---

## 10. Configuration That Affects Your Data

These settings in `ai_config.json` under `"execution_block"` constrain what
your data can contain:

| Config Field | Default | Constraint on Data |
|-------------|---------|-------------------|
| `num_slots` | 4 | Total bootstrap atoms + execution steps ≤ `num_slots` |
| `num_steps` | 2 | Must be ≥ longest `execution` array in your dataset |
| `num_ops` | 4 | Fixed at 4: add/sub/mul/div. Cannot be changed. |
| `enabled` | true | Must be true for execution training |
| `selector.enabled` | true | Must be true for decode-time `<NUM>` binding |
| `selector.supervision_weight` | 0.0 | Set > 0 only if providing explicit selector targets |

**Slot capacity formula:**

```
max_atoms + max_execution_steps ≤ num_slots
```

If your hardest problem has 3 input numbers and 3 computation steps, you need
`num_slots >= 6`.

---

## 11. Common Mistakes

### Mistake 1: Numbers not appearing literally in rendered text

```json
{
    "question": "What is two plus three?",
    "state_0": { "atoms": [2, 3] }
}
```

**Problem:** The question says "two" and "three" (words), but `state_0.atoms`
has `2` and `3` (digits). The rendered `STATE0` line will contain `2` and `3`,
so bootstrapping works — but if you want the question itself to contain the
numbers for natural phrasing, write them as digits:

```json
{ "question": "What is 2 plus 3?" }
```

### Mistake 2: Ambiguous arg resolution without arg_slots

```json
{
    "state_0": { "atoms": [5, 5] },
    "execution": [{ "op": "add", "args": [5, 5], "result": 10 }]
}
```

**Problem:** Both slots 0 and 1 contain 5. The value-based resolver finds
4 valid (slot1, slot2) pairs: (0,0), (0,1), (1,0), (1,1) — all produce 10.
**Crash: "ambiguous: 4 (slot1, slot2) pairs."**

**Fix:** Add `"arg_slots": [0, 1]`.

### Mistake 3: Result doesn't match computation

```json
{
    "execution": [{ "op": "add", "args": [10, 7], "result": 18 }]
}
```

**Problem:** 10 + 7 = 17, not 18. **Crash: "no (slot1, slot2) pair produces
expected result 18."**

### Mistake 4: Too many slots for config

```json
{
    "state_0": { "atoms": [1, 2, 3, 4, 5] },
    "execution": [{ "op": "add", "args": [1, 2], "result": 3 }]
}
```

With default `num_slots = 4`, this needs 5 bootstrap slots + 1 step slot = 6.
**Crash at batch validation.**

### Mistake 5: Forgetting that execution steps consume slots

```json
{
    "state_0": { "atoms": [10, 20] },
    "execution": [
        { "op": "add", "args": [10, 20], "result": 30 },
        { "op": "mul", "args": [30, 10], "result": 300 },
        { "op": "sub", "args": [300, 20], "result": 280 }
    ]
}
```

Slots used: 0 (=10), 1 (=20), 2 (=30), 3 (=300), 4 (=280).
Need `num_slots >= 5` and `num_steps >= 3`.

---

## 12. Checklist Before Submitting Data

Before feeding your `concept_blocks.jsonl` into the training pipeline:

- [ ] **Every `state_0.atoms` entry is a JSON number** (not a string, not null)
- [ ] **Every `execution[].op` is one of:** `"add"`, `"sub"`, `"mul"`, `"div"`,
      `"+"`, `"-"`, `"*"`, `"/"`
- [ ] **Every `execution[].args` has at least 2 numeric entries**
- [ ] **Every `execution[].result` is correct:** manually verify `op(arg1, arg2) == result`
      for every step, carrying intermediate results forward
- [ ] **Used `arg_slots` where values are ambiguous** (duplicate atom values,
      intermediate results equal to bootstrap values)
- [ ] **Slot count fits config:** count bootstrap atoms + execution steps ≤ `num_slots`
- [ ] **Step count fits config:** longest `execution` array length ≤ `num_steps`
- [ ] **No division by zero:** no step where `args[1]` (the divisor) is 0
- [ ] **Answer text contains the final result** so language modeling loss
      provides end-to-end gradient pressure
- [ ] **Consistent slot assignment rules** across examples: always use the same
      pattern (e.g., "first number → slot 0, second → slot 1") so the model
      can learn a stable mapping
- [ ] **Mixed non-execution rows for balance:** include plain text rows
      (no `state_0` or `execution`) so the model doesn't overfit to the
      execution format

---

## Appendix A: JSON Minimal Valid Examples

**Minimal execution row:**

```json
{"question":"What is 3 plus 4?","state_0":{"atoms":[3,4]},"execution":[{"op":"add","args":[3,4],"result":7}],"answer":"7"}
```

**Minimal non-execution row:**

```json
{"question":"What is the capital of France?","answer":"Paris."}
```

## Appendix B: Slot State Evolution Diagram

For `state_0.atoms = [10, 3, 2]` with two execution steps:

```
Before execution:
  Slot 0: 10  (bootstrap from atoms[0])
  Slot 1:  3  (bootstrap from atoms[1])
  Slot 2:  2  (bootstrap from atoms[2])
  Slot 3: empty
  Slot 4: empty

After Step 0 (mul slots 0,1 → slot 3):
  Slot 0: 10
  Slot 1:  3
  Slot 2:  2
  Slot 3: 30  ← result of 10 × 3
  Slot 4: empty

After Step 1 (add slots 3,2 → slot 4):
  Slot 0: 10
  Slot 1:  3
  Slot 2:  2
  Slot 3: 30
  Slot 4: 32  ← result of 30 + 2
```
