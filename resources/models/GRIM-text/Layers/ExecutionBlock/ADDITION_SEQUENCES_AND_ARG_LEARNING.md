# Addition-style sequences and teaching argument selection

This document explains how to **structure training (and inference) sequences** so the ExecutionBlock register machine can solve **addition-style** problems, and how the model **learns which registers to use as operands**—given the current GRIM-text implementation (slot-only candidates, hard read/write, softmax + argmax + straight-through estimators).

It is **not** a substitute for `DOCUMENTATION.md` kernel-level detail; it focuses on **data design**, **supervision**, and **expectations**.

---

## What you provide vs what the model learns

| Responsibility | You (data / pipeline) | Model (trained weights) |
|----------------|-------------------------|-------------------------|
| Which literal initializes which runtime register | `token_to_slot_index_map` per token (`-1` = non-state) + numeric side channel; `bootstrapMemoryFromSlotMap` copies into `M.values[index]` | — |
| Which semantic slot is arg1 / arg2 | `teacher_steps` supplies opaque `SlotId` targets; `compiled_slot_bindings` lowers them to per-row `SlotIndex` values | `w_arg1_select` scores arg1 over **value-slot** rows; `W_arg1_to_arg2` projects its soft candidate summary into the `w_arg2_select` query before argmax and hard reads from `M.values` |
| Which operation (+ − * /) | — | `W_op_select_` from pooled context + soft arg hiddens; softmax → argmax → `kernelHardPickOpForward` |
| Where to write the result | — | Write head softmax over `V` (scratch masked) → argmax → `kernelHardWriteScalarDev` |
| What token to predict next | Targets in the batch / LM loss | LM head, injection path, rest of encoder |

The compiled `teacher_steps` payload supplies gold arg1/arg2 slots. When
structured CE is enabled, those targets directly supervise both selector
logits; arg2 CE also flows through the conditional summary into arg1 once the
zero-initialized conditional projection begins to move.

---

## Structuring an addition-style sequence

### 1. Tokens and side channels

- Represent numbers with your tokenizer’s **numeric atom** path (e.g. `<NUM>`-style tokens) so **ScratchBlock** and **numeric values** align with training.
- Fill **`numeric_values`** at each token position with the **literal** (e.g. `3.0`, `5.0`).
- Set **`atom_mask`** (and related flags) consistently with how `buildBatchPayload` / the dataloader already builds batches.

### 2. Compiled slot-index map (mandatory for register semantics)

- For each **state-bearing numeric token**, set **`token_to_slot_index_map[pos]`** to a dense runtime index in **`[num_scratch_slots, num_slots)`** (value registers only).
- Use **two distinct slots** for two addends (e.g. first literal → slot `S`, second → `S+1`, or fixed indices like `0` and `1` when `num_scratch_slots == 0`).
- All **non-state** positions should be **`-1`**.
- Carry semantic identities separately as opaque `SlotId` values and provide one explicit `compiled_slot_bindings` bijection for the row. Never infer identity from the dense index.

Bootstrap runs once per forward at the execution layer (before `executeStep` loops): it copies literals into **`M.values[slot]`** and sets **`valid_mask`**. It also uses `bootstrap_slot_to_pool_index` to fuse the authored token's NumberEncoder-derived selector key into **`M.state_embeds[slot]`**. Generated writes replace that state embedding, so authored provenance cannot remain stale after an overwrite.

Operand candidate rows add the learned absolute address **`E_slot[slot]`** to the current runtime content. Consequently, equal values or repeated selector candidates can still occupy distinguishable registers while `w_arg1_select_` and `w_arg2_select_` retain ownership of operand-role selection.

### 3. Prompt shape (recommended for curriculum)

Start with a **stable surface form**, then diversify:

- Early training: fixed template, e.g. a single phrasing for “What is A plus B?” so layout and slot policy are predictable.
- Later: paraphrases, but keep a **consistent rule** for “first number in the prompt → slot A, second → slot B” (or another rule you can maintain in the dataloader).

### 4. Targets / supervision

- **Language modeling:** the answer tokens (e.g. digits or a single numeric atom for the sum) should be **reachable** only if the network uses useful representations; that provides **implicit** pressure on execution heads when the block is on the backward path.
- **Numeric head / other heads:** if you supervise a scalar at a position, ensure it is **consistent** with the story you want (register truth remains **`M.values`** for execution; readout heads are a separate contract—see project docs).

### 5. Execution depth

- **`execution_block_num_steps` (`K`)**: one step can do one binary op and one write. Multi-hop arithmetic may need **multiple steps** or **clear intermediate writes** to slots the next step can read.

---

## How the model learns to “identify args”

### Default (no new code)

- Arg logits are built from **slot-only** candidate hiddens (`M.state_embeds` masked by validity).
- Forward: **argmax** chooses operand slots; **values** come only from **`M.values`** at those indices.
- Backward: **straight-through** style routing sends gradients through **softmax** on arg distributions (see `SlotValueSTGradFn` and related autograd nodes in `execution_block_GPU.cu`).

So the model learns **which slots to read** only if **incorrect reads hurt** the scalar or representation paths that your **actual loss** differentiates through.

### If arg selection is too weak

Add **explicit supervision** (requires training changes), for example:

- Auxiliary cross-entropy on **`p_arg1` / `p_arg2`** against gold opaque `SlotId` targets lowered through `compiled_slot_bindings`, or
- Intermediate **copy/move** tasks (“value in slot i should appear in slot j”) before harder arithmetic.

---

## Inference and generation

- **Single forward** (`forward`, `forwardInit`, explicit Phase2 inference shared-forward calls, etc.): upload **`token_to_slot_index_map`** together with tokens and numerics so bootstrap and `executeStep` see the same contract as training.
- **Autoregressive decode:** generated tokens default to runtime index `-1`. Decode-time values require an explicit policy that allocates a semantic `SlotId`, lowers it to a row-lifetime `SlotIndex`, and updates both projections together.

---

## Quick checklist (addition-style)

1. **`execution_block_enabled`** and scratch path enabled as required by `AutogradTraining.cu` gating.
2. **Two literals** in the side channel with **two distinct value slots** in `token_to_slot_index_map`.
3. **Consistent mapping rule** across examples (or explicit gold for future aux loss).
4. **Answer tokens / numeric targets** aligned with the task so wrong operands hurt the objective.
5. **`K` large enough** if the expression needs more than one ALU step.
6. **Inference:** H2D copy of **`token_to_slot_index_map`**; **generation:** define slot policy for new `<NUM>` if you need register execution while decoding.

---

## Code path reference (where ops run)

End-to-end, ops are invoked from:

1. **`Autograd::materializeTrainingGraphActivations`** (`training/Autograd/AutogradTraining.cu`): encoder loop.
2. At **`exec_layer`**: **`executionBlockBootstrapMemoryFromSlotMap`**, then **`executionBlockStep`** × **`K`**.
3. Inside **`executionBlockStep`** (`Layers/ExecutionBlock/execution_block_GPU.cu` → data/memory stream impls): softmax heads → argmax → **`kernelReadSlotValueByRelIdx`** → **`kernelFourOps`** → **`kernelHardPickOpForward`** → hard write + injection.

Later encoder layers may run **`crossAttentionRead`** from memory starting at **`exec_layer`**.

---

## Related files

- `execution_block_GPU.hpp` / `execution_block_GPU.cu` — register machine and kernels.
- `AutogradTraining.cu` — when the block runs; bootstrap and `executeStep` invocation.
- `Shared/Batching/BatchPayload.*` — where `token_to_slot_index_map` is assembled for training.
- `Shared/Batching/BatchDeviceUpload.cu` — H2D of `token_to_slot_index_map` into `BatchDeviceBindings::d_token_to_slot_index_map`.
- `Phase2_InferenceLoop.cu` / `Shared/Forward/ModelForward_GPU.cu` — Phase2 authors inference payloads, uploads slot maps through `BatchDeviceBindings`, and drives shared-forward model scoring.

---

## Concept blocks and training data

### Current behavior (debug / expedient)

**Concept blocks are not only a curriculum ID registry here** — `PrepareTrainingDataFromCache` also **reads `concept_blocks.jsonl`** (same directory as `merged_verified_cache.jsonl`) and **appends each line as its own GRMT sequence** before cache rows. That lets you debug ExecutionBlock + slot maps without wiring mass-dataset resolution yet.

Structured fields in JSON may include:

- `state_0`: `{ "atoms": [...], "type": "..." }`
- `execution`: `[{ "op", "args", "result" }, ...]`
- `state_1`: `{ "result" }`
- `explanation` (or legacy `intermediates`) and `answer`

Encoding path: canonical text (`Q:`, `STATE0`, `EXEC`, …) plus a trailing **`__SLOTS__`** block (one numeric per line; order = `state_0.atoms` then each execution step’s `args` + `result`). The **last K numeric atoms** in the tokenized sequence get `token_exec_slots` = `base + 0..K-1` (base from env **`GRIM_CONCEPT_EXEC_BASE_SLOT`**, default `0`).

**GRMT v10** appended `token_exec_slots[len]` after per-token atom strings. Current GRMT uses opaque `SlotId` teacher targets plus explicit compiled slot bindings.

### Target architecture (what it should look like later)

Concept blocks stay **thin curriculum records**: primarily **`id`** (+ optional display fields) and a **stable pointer** to the real payload, e.g. **`source_sequence_id`** referencing a row in **`mass_dataset.jsonl`** / merged cache (or a dedicated structured-sequence store).

**Training prep** should:

1. Resolve `cb_id` → canonical structured sequence (not re-embed the full block from `concept_blocks.jsonl`).
2. Run **one** encoding pipeline on that resolved text/structure (same as other corpus rows).
3. Derive **`token_exec_slot_indices` / numeric side channels** from the **structured record** (or a shared schema), not from a duplicate `__SLOTS__` serialization of the block file.

Until that exists, the debug path above is intentionally redundant (block JSON is both UI curriculum and training source).

---

*Last updated to match the slot-only, hard-read/hard-write ExecutionBlock design and inference slot-map wiring.*
