---
name: Serialization layer refactor
overview: Split Serialization into headers/CUDA TUs; implement serializiationplan—VALIDATE then LOAD; all compatibility in validate_checkpoint_capabilities; expected tensor sizes MUST come from the GRIM TensorContract ([TensorContract_GPU.hpp](resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp))—TensorShape/Layout/total_elements and the same GQA/layout rules as training—not duplicated formulas in Serialization; DeviceWriteView.count only cross-checks contract-derived numel; load() deterministic; Pattern B in grim_model_serialization.cu.
todos:
  - id: split-headers
    content: Serialization_views.hpp, Serialization_requests.hpp, Serialization_validate.hpp; umbrella Serialization_GPU.hpp; CheckpointCapabilityRequirements + SerializationLoadReport on SerializationLoadRequest
    status: completed
  - id: validate-impl
    content: validate_checkpoint_capabilities—presence + ignore-not-required; expected sizes from TensorContract (shared helpers or TensorShape factories / same head_dim·GQA·FFN rules as Encoding/LMHead/ExecutionBlock); compare FlatBuffer vector sizes to contract numel; assert DeviceWriteView.count matches contract where destinations exist (drift = bug); EmitModuleError; no cudaMemcpy
    status: completed
  - id: split-cu
    content: Serialization_validate.cu, Serialization_load.cu, Serialization_save.cu (+ optional Serialization_layer.cu); retire monolithic Serialization_GPU.cu
    status: completed
  - id: refactor-load
    content: load() pipeline steps 1–6 from notes; after validate, load branches ONLY on requires_* (no if(fb_*) for requirement); no duplicate presence logic; pointers only destination; set LoadReport; Step 7 safety verify; remove silent/legacy paths
    status: completed
  - id: wire-capabilities
    content: grim_model_serialization.cu sets request.capabilities.* only; SerializationLayer never computes/overrides
    status: completed
isProject: false
---

# SerializationLayer refactor (aligned with serializiationplan.md)

## Product bar

Strict **compiler-style** loading: the system should be **hard to misuse**, not only correct when used carefully.

Constraints: **no unrelated refactors**, **do not modify CUDA copy behavior** (`cudaMemcpy` / `upload_device_vector` semantics unchanged—only ordering/guards), **no architecture changes**.

## Core rule

Checkpoint loading is a **pure validation → load** pipeline:

**VALIDATE → (success only) LOAD**

**No GPU memory writes** before validation completes successfully.

## Source of truth

[.cursor/plans/serializiationplan.md](.cursor/plans/serializiationplan.md) — implement exactly that.

---

## Tensor contract (non-negotiable)

Serialization validation is a **checkpoint ↔ tensor contract** check, not a free-standing math exercise.

**Authority:** [TensorContract_GPU.hpp](resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp) — `TensorContract::TensorShape`, `Layout` (e.g. `BSM`, `QKV_FUSED`, `LOGITS`), `total_elements()`, factories such as `make_BSM` / `make_QKV_FUSED` / `make_LOGITS`, and the documented **GQA** rule (`total_qkv_dim` / d_\text{model} + 2 \cdot n_\text{kv} \cdot \text{headdim}). The same **contiguous, layout-aware** assumptions and **ContractViolation** / `validate_conversion` philosophy apply: checkpoint payloads must match the **element counts implied by that contract** for the given `SerializationModelConfigView`.

**What to do in code:**

1. **Derive every expected FlatBuffer float vector length from the tensor contract** (or from thin helpers shared with the layers—e.g. encoder attention QKV weight matrix as a flattened 2D contract shape, LM head as `LOGITS`-style `vocab_size × d_model`, etc.). Do **not** maintain parallel, serializer-only size formulas that can drift from TensorContract or layer allocations.
2. **ParamGroup alignment:** treat checkpoint modules as matching [ParamGroupType](resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp) buckets (`ATTENTION`, `FFN`, `EXECUTION_BLOCK`, `SCRATCHBLOCK`, …) so naming and ownership stay consistent with the rest of the stack.
3. **ExecutionBlock / fixed-arch tensors:** where shapes are fixed by the ExecutionBlock layer (e.g. decode MLP 24×16), expected sizes must match **the same `Tensor::numel()` / shape the layer uses**, not re-guessed literals in Serialization.
4. `**DeviceWriteView.count`:** use as a **secondary** check that the **model-allocated** destination matches the **contract-derived** expected `numel`. If contract says `N` but the view says `M`, that is a **model vs contract wiring bug**—fail with `EmitModuleError` that names both numbers and the tensor slot.

If validation uses ad-hoc sizes that are not traceable to TensorContract (or shared layer helpers), that is a **bug** against this plan.

---

### Step 1 — Authoritative capability flags

`CheckpointCapabilityRequirements` (on `SerializationLoadRequest` as `capabilities` or equivalent) is the **only** source of truth for what the model requires.

`SerializationLayer` **must not** infer requirements from:

- pointers  
- buffer presence  
- optional FlatBuffer tables

Struct fields (match prior plan): `requires_execution_block`, `requires_numeric_head`, `requires_reasoning_head`, `requires_scratch_block`, `requires_final_rms_gamma`.

`SerializationLoadReport` (four module flags + use as needed for Step 7 safety) attached to the load request; populated during load.

---

### Step 2 — Single source of validation truth

**All** compatibility enforcement lives **only** in:

`validate_checkpoint_capabilities(...)`

**Not** in `load()` (no duplicated presence / mismatch logic there).

**Required checks:**

1. **Presence** — For each `requires_*`: if true, table MUST exist in FlatBuffer; if false, that component MUST be **ignored** (even if present in the file).
2. **Dimensions (tensor contract, not buffer-first)** — From **model config** plus **TensorContract**-consistent derived shapes (same rules as [Encoding](resources/models/GRIM-text/Layers/Encoding/Encoding_GPU.hpp), LM head, GQA QKV, FFN, etc.). Express expectations as **contract `total_elements()`** (or equivalent shared helpers), not serializer-local duplicates.
  - `**DeviceWriteView.count**`: cross-check that destinations were allocated with the **same** contract `numel` (secondary; catches allocator drift).
3. **Tensor sizes** — For each required module, checkpoint vector lengths MUST equal expected sizes **exactly** (no resize, padding, or truncation).
4. **final_rms_gamma** — If `requires_final_rms_gamma`: MUST exist and MUST match expected size; **no** fallback elsewhere.

On failure: `EmitModuleError` with **required capability**, **actual checkpoint state**, and **mismatch detail**; return `false`. Validator TU does **not** perform GPU copies.

---

### Step 3 — Structural guarantee: no pre-validation load

`SerializationLayer::load` ordering:

1. Read file
2. Verify FlatBuffer
3. Version check
4. Call `validate_checkpoint_capabilities()`
5. If false → return immediately
6. **Only then** call any `upload_device_vector` / GPU writes

It must be **unreachable** to hit `upload_device_vector` before validation succeeds.

---

### Step 4 — Strict module loading (no logic duplication)

`load()` **must not** re-check presence (e.g. no `if (fb_numeric_head)` as a *requirement* gate).

It **assumes** the validator already proved correctness.

Pattern:

- `if (req.requires_numeric_head) { load weights; report.numeric_head_loaded = true; }`  
- `else { skip completely }`

Same idea for reasoning, scratch, execution block, final RMS, and any other gated block.

**ExecutionBlock**: keep the existing strict per-field copy path, but eligibility is `**requires_execution_block` only**, not “pointer means required.”

---

### Step 5 — Remove all silent paths

Remove: silent ignore of missing pieces, conditional load based on “if table exists,” legacy compatibility comments/branches. Forbidden.

---

### Step 6 — Pointers are not requirements

`if (request.xyz.ptr)` means **destination exists** only. It must **never** mean the module is required. Requirements = `**req.capabilities.requires_`* only**.

---

### Step 7 — Final load verification

After loading: **safety** check (not primary validation)—if a required module was not marked loaded, **fail** and log via `EmitModuleError`.

---

### Step 8 — Logging

Failures use `**EmitModuleError`** and include:

- required capability  
- actual checkpoint state  
- mismatch detail

---

### Step 9 — Call site (Pattern B)

[grim_model_serialization.cu](resources/models/GRIM-text/Common/grim_model_serialization.cu) sets `**request.capabilities.`*** only.

`SerializationLayer` **must not** compute or override these flags.

---

## Final rule (from notes)

There is **exactly one** place where compatibility is decided: `**validate_checkpoint_capabilities()`**.

After that:

- load is **deterministic**  
- **no branching on checkpoint content** for compatibility decisions (only `requires_`*-gated copy from already-validated layout; no “is this optional table OK?” logic in `load()`)

If any compatibility decision exists outside the validator → **bug**.

---

## File layout (unchanged intent)


| File                                                                                           | Responsibility                                                                                                           |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Serialization_GPU.hpp](resources/models/GRIM-text/Layers/Serialization/Serialization_GPU.hpp) | Umbrella includes + `SerializationLayer` declaration                                                                     |
| `Serialization_views.hpp`                                                                      | Views, `SerializationModelConfigView`                                                                                    |
| `Serialization_requests.hpp`                                                                   | `SerializationConfig`, requests, capabilities + load report                                                              |
| `Serialization_validate.hpp` / `.cu`                                                           | `validate_checkpoint_capabilities` only; includes/reuses **TensorContract** (or shared size helpers) for expected numels |
| `Serialization_load.cu`                                                                        | `load()` pipeline above                                                                                                  |
| `Serialization_save.cu`                                                                        | `save()`                                                                                                                 |


CMake: [GRIM/CMakeLists.txt](resources/models/GRIM-text/GRIM/CMakeLists.txt) globs `Layers/**/*.cu`.

---

## Out of scope

- Changing CUDA **copy** behavior (only when copies run).  
- MTP sidecar unless you extend policy later.

## Verification

- Build GRIM text server.  
- Prove **no** code path calls `upload_device_vector` before a successful `validate_checkpoint_capabilities`.  
- Good checkpoint loads; bad checkpoint fails with `EmitModuleError` and no partial GPU writes from load.

