# Atom Placeholder + Selector Inference Cutover Plan

> **Status:** Proposed (agent plan).
> **Scope:** Make the `<INT>` / `<FLOAT>` atom-placeholder + selector mechanism work
> end-to-end for **both** training and inference now that `BatchPayload` is the single
> shared ingestion boundary for both modes.
> **Build constraint:** GRIM-text is a SEPARATE build. No `../../../../core/` headers.
> **Discipline:** Rule 20 (fail loud, no fallbacks, delete legacy paths) and the
> ownership taxonomy (Cat1 graph-transient / Cat2 durable / Cat3 workspace) apply to
> every phase below.

---

## 0. Design Intent (the contract we are restoring)

The atom mechanism is a **placeholder + selector** design, not a literal-digit design:

1. **Placeholders are vocabulary tokens.** There is exactly **one placeholder token per
   numeric atom type** — today `<INT>` (`ATOM_INT`) and `<FLOAT>` (`ATOM_FLOAT`). The model
   does **not** emit digits; it emits a placeholder token meaning "a number of this type
   goes here."
2. **The numeric side channel is the candidate-entry pool.** `numeric_values` /
   the `AtomTable` exact payloads are **the set of values that could fill a placeholder
   slot** — they are *entries*, not the model's output stream.
3. **The selector fills the slot.** Cross-entropy supervises the placeholder **token**
   (which atom type). A separate **selector** resolves *which entry* of that type is bound
   into the slot. (`SlotSelectionResult` / `DecodeTimeResolveResult` in
   [Shared/Execution/DecodeTimeResolveResult.hpp](../resources/models/GRIM-text/Shared/Execution/DecodeTimeResolveResult.hpp).)
4. **Inference must draw entries from the user's prompt.** When the model emits a
   placeholder during generation, the selector must bind it to a real number **taken from
   the prompt's atoms** (and from numbers the model has already produced) — that is the
   "reasoning over numbers" goal. **Row-local atom identity** exists precisely so a prompt's
   atoms are distinguishable from any other row's atoms.

### What actually happens today (the break)

| Intended | Actual (code-grounded) |
|---|---|
| Placeholder slot is filled from row-local AtomTable entries | Selector value comes from `decode_selector.selected_value` (a **float** slot read), not the AtomTable |
| Generated atoms are registered so later steps can reuse them | Decode loop pushes `kAtomEntryNone` for **every** generated token (`Phase2_InferenceLoop.cu`) — generated atoms are never registered |
| Row-local AtomTable reaches the device selector | `seq_atom_tables` is **host-only, NOT transferred to GPU** (`BatchPayload.hpp:151,155`); only lossy `numeric_values` + `atom_mask` reach the device |
| Exact int64/double entries feed the slot | `executionBlockBootstrapMemoryFromSlotMap(...)` consumes `const float* device_numeric_values` (`execution_block_memory_stream_GPU.cu:396`) — exactness is lost before the selector ever runs |

**Root cause:** the candidate-entry pool that the selector chooses from is the **lossy,
pre-gathered float lane**, and the **row-local AtomTable** (the actual entry identity +
exact value store) never crosses the host→device boundary. Inference additionally never
writes generated atoms back, so the model cannot reuse numbers it just produced.

---

## Phase Map

```
Phase 1  Entry-pool boundary    : upload row-local atom entries (exact) + per-token entry index to GPU
Phase 2  Exact bootstrap        : selector/bootstrap consumes exact entries via gather, not float
Phase 3  Inference registration : decode loop registers generated atoms into a growing row table
Phase 4  Row-local resolution   : selector candidate pool = this row's atom entries (prompt + generated)
Phase 5  Float-lane retirement  : delete numeric_values once gather path is proven equal-or-better
Phase 6  Symmetry + validation  : training and inference share one population/upload/consume path
```

Each phase is independently shippable and leaves the build green. Phases 1–2 are the
foundation; Phase 3–4 deliver the inference behavior the user is asking for; Phase 5 is
the Rule-20 cleanup; Phase 6 locks the shared contract.

---

## Phase 1 — Entry-Pool Boundary (upload row-local atom entries to GPU)

**Goal:** Get the row-local atom **entries** (exact values + types + a per-token entry
index) onto the device, so the selector can choose from real entries instead of the float
lane. No consumer behavior changes yet — this phase only makes the data *available*.

**Current state**
- `BatchPayload` already carries the host side: `atom_entry_ids[total_tokens]` and
  `seq_atom_tables[batch_size]` (`BatchPayload.hpp:155`), plus compact
  `atom_positions` / `atom_types` (`BatchPayload.hpp:103-104`).
- `AtomTable` already has exact per-atom GPU arrays in `GPUAtomData`:
  `d_numeric_int_values`, `d_numeric_float_values`, `d_numeric_kind`, `d_types`, `d_flags`
  (`AtomTable.hpp:288`), filled by `uploadToGPU()` (`AtomTable.cu:1154`).
- **Gap:** `BatchDeviceUpload.cu` uploads only `numeric_values` + `atom_mask` +
  compact positions/types. `BatchDeviceBindings` has **no** atom-entry fields.

**Target state**
- Merge the per-row `seq_atom_tables` into **one batch-level entry pool** at payload-build
  time (Option A from the design review — single flat upload, batch-global entry ids).
- Rewrite `atom_entry_ids` to be **batch-global** (index into the merged pool) instead of
  row-local-into-`seq_atom_tables`.
- Upload: `d_atom_entry_ids[total_tokens]`, plus merged pool arrays
  `d_atom_int_values`, `d_atom_float_values`, `d_atom_kind`, `d_atom_type` `[num_pool_atoms]`,
  and `d_row_atom_offset[batch_size]` (so row-local identity is recoverable on device).

**Concrete edits**
1. [Shared/Batching/BatchPayload.hpp](../resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp)
   - Add a materialized merged entry pool (Cat2 semantic data, immutable after build):
     `std::vector<int64_t> pool_int_values;`
     `std::vector<double> pool_float_values;`
     `std::vector<uint8_t> pool_kind;`
     `std::vector<uint32_t> pool_type;`
     `std::vector<int> row_atom_offset;  // [batch_size+1] prefix offsets`
   - Document `atom_entry_ids` as **batch-global pool index** (not `seq_atom_tables` index).
2. [Shared/Batching/BatchPayload.cu](../resources/models/GRIM-text/Shared/Batching/BatchPayload.cu)
   - In `buildBatchPayload(...)`: after atom side channels are assembled, walk all rows,
     dedup/merge atoms into the pool, fill `row_atom_offset`, and rewrite each token's
     `atom_entry_ids[p]` to `row_atom_offset[row] + local_entry_id`.
   - In `buildInferenceBatchPayload(...)`: same merge from the single provided `atom_table`
     (batch_size==1 → `row_atom_offset = {0, atom_table->size()}`).
   - **Fail loud:** assert every non-`kAtomEntryNone` `atom_entry_ids[p]` resolves to a
     valid pool index; throw with file:line on mismatch. No silent clamp.
3. [Shared/Batching/BatchDeviceBindings.hpp](../resources/models/GRIM-text/Shared/Batching/BatchDeviceBindings.hpp)
   - Add device-view pointers: `const uint32_t* d_atom_entry_ids;`,
     `const int64_t* d_atom_int_values;`, `const double* d_atom_float_values;`,
     `const uint8_t* d_atom_kind;`, `const uint32_t* d_atom_type;`,
     `const int* d_row_atom_offset;`, plus `int num_pool_atoms;`.
4. [Shared/Batching/BatchDeviceUpload.cu](../resources/models/GRIM-text/Shared/Batching/BatchDeviceUpload.cu)
   - Mirror the existing `d_atom_mask` upload for `d_atom_entry_ids` (same `[total_tokens]`
     shape) and upload the pool arrays + `d_row_atom_offset` into newly-allocated storage in
     `createBatchDeviceStorage` / `uploadBatchToDevice`; set the new bindings.

**Ownership / Rule 20**
- The merged pool + offsets are **Cat2** (durable semantic data, immutable after build) —
  they live on `BatchPayload`, owned through `device_storage`, never on
  `AutogradIntermediates`.
- Do **not** reuse `AtomTable::getGPUBuffer()` device pointers as the batch binding source:
  that buffer's lifetime is owned by the (host-only) `AtomTable` instance, not by
  `device_storage`. Copy into batch-owned storage so the boundary owner is single and clear.

**Exit criteria**
- Build green. New device fields populated and non-null whenever `num_pool_atoms > 0`.
- A debug assert (Rule 21 equation log optional) confirms: for every atom token `p`,
  `float(pool_value[d_atom_entry_ids[p]]) == numeric_values[p]` within tolerance. This proves
  the new exact pool agrees with the legacy float lane **before** any consumer switches.

---

## Phase 2 — Exact Bootstrap (consume entries via gather, not the float lane)

**Goal:** Make the execution-memory bootstrap (the selector's candidate source) read the
**exact** entry through `atom_entry_ids`, not the lossy `device_numeric_values` float.

**Current state**
- `ModelForward_GPU.cu:642-652` feeds `bindings->d_numeric_values + tok_off` and
  `d_token_to_slot_map + tok_off` into
  `executionBlockBootstrapMemoryFromSlotMap(...)`.
- That function signature is `const float* device_numeric_values`
  (`execution_block_memory_stream_GPU.cu:396`); kernels `kernelBootstrapSlotMarkValid` /
  `kernelBootstrapSlotWriteValues` write `M.values` from the float lane.

**Target state**
- Bootstrap resolves each token's value by **gather**:
  `e = d_atom_entry_ids[p]; value = (kind==INT) ? double(d_atom_int_values[e]) : d_atom_float_values[e];`
- `ExecutionMemory.values` is populated from exact entries; the float lane is no longer the
  source of truth for slot values.

**Concrete edits**
1. [Layers/ExecutionBlock/execution_block_GPU.hpp](../resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp)
   and [execution_block_memory_stream_GPU.cu](../resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu)
   - Add an exact-entry overload/replacement of
     `executionBlockBootstrapMemoryFromSlotMap(...)` taking
     `const uint32_t* device_atom_entry_ids`, the pool arrays, and `row_atom_base`
     (the row's offset) instead of `const float* device_numeric_values`.
   - Update `kernelBootstrapSlotWriteValues` (and the mark-valid pass if needed) to gather
     exact values by entry id. Keep the two-pass last-writer-wins slot resolution intact.
2. [Shared/Forward/ModelForward_GPU.cu](../resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.cu)
   - At the bootstrap call site (~:648), pass `bindings->d_atom_entry_ids + tok_off`, the
     pool pointers, and `row_atom_offset[row]` instead of `d_numeric_values + tok_off`.

**Ownership / Rule 20**
- `ExecutionMemory.values` is existing Cat2 execution state — unchanged ownership, only the
  data source changes.
- **Delete**, do not branch: once gather is wired, remove the `float* device_numeric_values`
  parameter from the bootstrap entry point. No "if exact else float" fallback (Rule 20).
  The Phase-1 agreement assert is the safety net during the switch, then it too is removed.

**Exit criteria**
- Training step numerically equivalent (within float tolerance) to pre-change on an
  arithmetic batch — proven by the Phase-1 agreement assert held across a full forward.
- Bootstrap no longer references `d_numeric_values`.

---

## Phase 3 — Inference Registration (grow a row-local table during decode)

**Goal:** Make generated atoms **first-class entries**. When the model emits a placeholder
during generation, register the selected number into a growing inference `AtomTable` and
carry a **real** entry id forward — so the next step can reuse it.

**Current state (the concrete hole)**
- In [training/Phases/Phase2_InferenceLoop.cu](../resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.cu),
  every generated token does `next_atom_entry_ids.push_back(kAtomEntryNone);` and
  `sequence.atom_entry_ids.push_back(kAtomEntryNone);`.
- The value is `token_numeric_value = generation_state.decode_selector.selected_value;`
  (a float), with no table entry created.

**Target state**
- Maintain a single **append-only inference `AtomTable`** per generated sequence, seeded
  from `prompt_atom_table`.
- On each generated **numeric** placeholder:
  1. Obtain the selected slot's **exact** payload (see Phase 4 — selector should return the
     entry, not just a float).
  2. `registerAtom(type, exact_value, raw_text)` into the inference table → real
     `entry_id`.
  3. `next_atom_entry_ids.push_back(entry_id);` (not `kAtomEntryNone`).
  4. `atom_table->uploadToGPU(stream)` before the next `buildInferenceBatchPayload(...)`
     (incremental: `uploadToGPU` only flushes `pending_gpu_upload_`, so appends are cheap —
     `AtomTable.cu:1154`).
- Pass the growing table forward as the `prompt_atom_table` argument for the next step's
  `buildInferenceBatchPayload(...)`.

**Concrete edits**
1. `generateOneSequence(...)` in `Phase2_InferenceLoop.cu`:
   - Replace the seed `prompt_atom_table` (currently treated as fixed) with a mutable
     `std::shared_ptr<AtomTable>` that is **cloned/owned** for the generation (it must be
     append-only and writable; the prompt table itself stays immutable upstream).
   - Replace the two `kAtomEntryNone` pushes for **numeric** atoms with the
     register→entry_id flow above. Non-numeric / EOS keep `kAtomEntryNone`.
   - Carry the table into `sequence.context_atom_table` so `decode()` round-trips faithfully
     (decode already prefers `atom_entry_ids + atom_table` over float — verified in
     `executePhase2TextInference`).

**Ownership / Rule 20**
- The growing inference table is **Cat2** for the lifetime of one generation (durable across
  decode steps), owned by the generation scope — **not** by any autograd intermediate.
- **Fail loud:** if a numeric placeholder is sampled but the selector did not resolve a slot,
  the existing throw stays (`Phase2_InferenceLoop.cu` already throws on
  `status != Selected`). Do not push a fabricated entry.

**Exit criteria**
- Generating a prompt like "3 plus 4 is" yields a placeholder whose decoded value is exact
  and whose entry is reusable: a follow-on placeholder can bind the just-produced number.
- No `kAtomEntryNone` is pushed for a successfully-resolved numeric placeholder.

---

## Phase 4 — Row-Local Resolution (selector chooses among *this row's* entries)

**Goal:** Make the selector's candidate pool be **the current row's atom entries**
(prompt + already-generated), keyed by row-local identity, instead of an undistinguished
global float read. This is the "use numbers from the user's prompt" behavior.

**Current state**
- `DecodeTimeResolveResult.selected_value` is a `float` slot read; `selected_slot` indexes
  execution memory `L`, with no link back to a specific row-local atom entry.
- Row-local identity exists only host-side (`seq_atom_tables`) and never reaches the device
  selector.

**Target state**
- After Phase 1, the device knows each row's entry window via `d_row_atom_offset[row]` →
  `d_row_atom_offset[row+1]`. The selector resolves a placeholder by choosing **among that
  window's entries** (this row's atoms only), guaranteeing prompt-local provenance.
- `DecodeTimeResolveResult` is extended to carry the **resolved entry id** (pool index) and
  the **exact payload** (int64/double + kind), not just a float — so Phase 3 can register the
  exact value.

**Concrete edits**
1. [Shared/Execution/DecodeTimeResolveResult.hpp](../resources/models/GRIM-text/Shared/Execution/DecodeTimeResolveResult.hpp)
   - Add `uint32_t selected_entry_id = kAtomEntryNone;`, `uint8_t selected_kind = 0;`,
     `int64_t selected_int = 0;`, `double selected_float = 0.0;` to
     `DecodeTimeResolveResult`. Keep `selected_value` only until Phase 5 (then delete).
2. Selector resolve path (decode-time `<NUM>` resolve — same subsystem that fills
   `generation_state.decode_selector`):
   - Constrain candidate slots to the row's entry window using `d_row_atom_offset`.
   - Populate the new exact fields from the pool arrays.
3. `Phase2_InferenceLoop.cu`:
   - Phase 3's `registerAtom(...)` uses `selected_int` / `selected_float` / `selected_kind`
     (exact), not `selected_value`.

**Ownership / Rule 20**
- `DecodeTimeResolveResult` lives on `GenerationState` (Cat2) — unchanged.
- **No fallback:** if the row window is empty but a numeric placeholder is sampled, throw
  (the model asked for a number with no candidate entries — a real error, surface it).

**Exit criteria**
- A two-number prompt where the answer must reuse a **specific** prompt number resolves to
  that number's entry (verifiable: `selected_entry_id` falls within the prompt's
  `[row_atom_offset[0], row_atom_offset[1])` window).

---

## Phase 5 — Float-Lane Retirement (Rule 20 cleanup)

**Goal:** Delete the lossy `numeric_values` float lane now that exact entries flow end to
end. No parallel paths (Rule 20: delete legacy, no "just in case").

**Concrete edits**
- Remove `numeric_values` from `BatchPayload` (host) and `d_numeric_values` from
  `BatchDeviceBindings` / `BatchDeviceUpload.cu`.
- Remove `selected_value` from `DecodeTimeResolveResult`.
- Remove the legacy `float numeric_value` reliance in `AtomEntry` packing where it was only
  feeding embeddings via the float lane (keep only if embeddings still require a float view;
  if so, derive it at the embedding boundary, not as the entry source of truth).
- Delete the Phase-1 agreement assert and any temporary dual-write.
- Audit `training_data_loader.hpp` (the known sanitize-log mismatch at ~:133-146): with the
  float lane gone, remove the float zeroing path and fix/delete the misleading
  "(mask cleared)" log.

**Exit criteria**
- `grep` for `numeric_values` / `d_numeric_values` / `selected_value` returns no live
  references in the train/inference forward path.
- Build green; training + inference numerically stable.

---

## Phase 6 — Symmetry + Validation (one shared path)

**Goal:** Prove training and inference use the **same** population → upload → consume path,
matching the "BatchPayload used roughly the same way for both" invariant.

**Concrete edits / checks**
- Confirm `buildBatchPayload` (training) and `buildInferenceBatchPayload` (inference) both
  produce the identical merged-pool + `atom_entry_ids` + `row_atom_offset` layout (factor the
  merge into one shared helper called by both — no duplicated merge logic).
- Confirm `BatchDeviceUpload.cu` has a single upload routine for the pool fields used by both
  modes (no mode-specific branch that diverges the data).
- Add a focused test (host-side, no GPU required for the merge logic): given per-row atom
  tables, the merge produces stable batch-global ids, correct `row_atom_offset`, and
  round-trips every token's entry to the right row window.
- Update feature docs: add/extend an entry under
  [GRIM/Docs](../resources/models/GRIM-text/GRIM/Docs/README.md) (e.g. `Tokenizer.md` or a new
  `AtomEntries.md`) describing the placeholder+selector+entry-pool contract. Do **not** add it
  to `copilot-instructions.md` (cross-cutting rules only).

**Exit criteria**
- Training and inference diff only by `mode` and geometry, never by atom-entry data flow.
- Merge-logic test green.

---

## Cross-Phase Risk Register

| Risk | Mitigation |
|---|---|
| Batch-global id rewrite corrupts row provenance | `row_atom_offset` keeps row windows explicit; Phase 6 test asserts every token maps to its own row's window |
| Exact→float embedding need still exists | Derive a float view at the embedding boundary only; keep entries (int64/double) as source of truth |
| Selector picks cross-row entry | Phase 4 constrains candidates to `[row_atom_offset[row], row_atom_offset[row+1])` |
| `uploadToGPU` per decode step too costly | It is incremental (`pending_gpu_upload_` only); appends flush a handful of atoms |
| Hidden third consumer of `d_numeric_values` | Phase 5 gated on a clean `grep`; do not delete until zero live references |

## Dependency Order

```
Phase 1 ─► Phase 2 ─► Phase 5
   └─────► Phase 3 ─► Phase 4 ─► Phase 5 ─► Phase 6
```

Phase 1 unblocks everything. Phases 2 and 3 can proceed in parallel after Phase 1. Phase 5
(deletion) must come after 2 **and** 4. Phase 6 closes out.
