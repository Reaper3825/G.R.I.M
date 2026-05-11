# GRIM-text Development Guide

> **Current Focus:** GRIM-text model training and inference. The broader G.R.I.M assistant project exists but is not the active development target.

This file holds **universal coding rules** that apply to every session. Project-/feature-specific knowledge is split into per-feature docs under [`resources/models/GRIM-text/GRIM/Docs/`](../resources/models/GRIM-text/GRIM/Docs/README.md). Load only the docs the current task needs — do not pull all of them into context.

---

## 🔴 CRITICAL: FORBIDDEN CODE PATTERNS (Rule 20 - Fail Loud)

**NEVER generate these patterns. If you see them, DELETE THEM:**

❌ `x ? x : fallback` - NO fallbacks. NO Stubs. Require x, crash if null  
❌ `if (ptr) { use(ptr); }` - Require ptr, crash if null with clear error  
❌ `if (args.stream) { ... } else { use_config_stream(); }` - NO silent fallbacks  
❌ `try { } catch { /* ignore */ }` - NEVER swallow errors silently  
❌ `if (version == old) { legacy_path(); } else { new_path(); }` - DELETE legacy paths  
❌ `// TODO: remove after migration` - Remove NOW or never commit  
❌ `__attribute__((deprecated))` - DELETE deprecated code entirely  
❌ Any comment containing "backwards compatibility" - Forbidden concept  
❌ `if (!initialized) { std::cerr << "WARNING: ..."; return; }` - Throw exception instead

**REQUIRED patterns:**

✅ `if (!ptr) throw std::runtime_error("X is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));`  
✅ `if (!args.stream) throw std::runtime_error("stream is NULL - caller MUST provide valid stream");`  
✅ `assert(condition && "description");` for invariants  
✅ Crash with detailed error message on ANY unexpected state  
✅ Delete code instead of deprecating it

**Pre-commit checklist - search your code for:**

- `? :` (ternary with fallback) → rewrite to throw
- `if (args.` (optional parameter check) → require parameter
- `else {` (compatibility branch) → delete the else clause
- `catch` (exception handler) → ensure it re-throws or exits, never swallows

**Why:** Silent failures waste weeks debugging. If code is wrong, crash immediately with clear error. Backwards compatibility hides bugs and creates maintenance debt.

**Severity Hierarchy:** Architectural problems (wrong data flow, dead code paths, disconnected subsystems, silent fallbacks at system boundaries) are the **MOST SEVERE** class of bug. A single architectural misconnect (e.g., loss not connected to gradients, embeddings not scaled, weights tied but gradients canceling) can waste weeks of GPU compute producing meaningless training runs. Always prioritize architectural correctness over local code quality. Rule 26 (YAGNI/delete dead code) is the architectural arm of Rule 20 — dead code creates false confidence that a subsystem is functional when it isn't.

---

### 🔴 Rule 20 — GradFn Persistent Gradient Discipline

When editing `resources/models/GRIM-text/Shared/TensorContract/GradFns/*`, persistent leaf gradient buffers are Category 2 training state. They MUST be accumulated into, never overwritten.

**Why this is critical:** `Tensor::ensure_grad()` only allocates and zeroes a grad buffer the first time. It does **not** clear existing gradients during a backward pass or microbatch accumulation window. Any GradFn that writes `tensor.grad_data()` with `=` silently erases prior contributions from other branches, loss terms, tied/shared params, or microbatches.

**FORBIDDEN in GradFns:**

❌ `grad_input[idx] = ...` when `grad_input` may point at `x.grad_data()`  
❌ `dx[i] = ...` when `dx` may point at a leaf grad buffer  
❌ `a_grad[idx] = ...`, `b_grad[idx] = ...`, `input_grad[idx] = ...` for any pointer sourced from `.grad_data()`  
❌ Allocating private `input_grad` for a leaf and only calling `input_grad_fn->apply(...)` — leaf tensors usually have no `input_grad_fn`, so the leaf receives no grad  
❌ Reusing a forward-style kernel that assigns into output without separately accumulating the temporary into the leaf grad buffer

**REQUIRED in GradFns:**

✅ If a grad pointer comes from `.grad_data()`, write with `+=` or `atomicAdd`.  
✅ Owned non-leaf scratch grad buffers must be zeroed before use; using `+=` is still preferred so kernels are leaf-safe.  
✅ If a kernel must assign into an owned temporary (e.g. centering kernels), then explicitly accumulate that temporary into the leaf grad buffer before propagating.  
✅ For leaf-vs-non-leaf capture: leaf → `x.ensure_grad(); ptr = x.grad_data();`; non-leaf → allocate owned zeroed scratch.  
✅ Before finishing any GradFn edit, grep for assignment-style grad writes: `grad_.*\] =`, `.*_grad.*\] =`, `dx\[.*\] =`, `input_grad\[.*\] =`.

See `resources/models/GRIM-text/GRIM/Docs/Autograd.md` → “GradFn accumulation contract” for the implementation-level checklist.

---

### 🔴 Rule 20 — OWNERSHIP TAXONOMY (Boundary Discipline)

Every piece of state in the training loop MUST belong to **exactly one** of three categories. Mixing categories in a single struct, or letting one category leak across a boundary into another, is a Rule 20 violation of equal severity to silent fallbacks.

**Category 1 — Graph-owned (TRANSIENT)**
Lives only inside the autograd tape window (forward → loss assembly → backward).
Examples: `loss_tensor`, `logits_tensor`, `encoder_layer_outputs`, all `GradFn` saved tensors, dropout grad taps, aux-loss `autograd::add` results.
- ❌ MUST NOT be read by any code that runs after `Tensor::backward()` returns.
- ❌ MUST NOT be stored in a struct that also holds Category 2 or Category 3 state.
- ✅ MUST be cleared by a single RAII owner at the boundary (e.g. `AutogradStepScope`).
- ✅ If a "late consumer" needs information from a Category 1 object, snapshot a **reduced/scalar** form into a Category 2 telemetry struct **before** `clear()` runs. Never extend the lifetime of the tensor.

**Category 2 — Persistent training state (DURABLE)**
Lives across many steps; the contents are meaningful between steps.
Examples: `ParameterGroup.tensor` data, parameter `grad_` buffers (across the accumulation window), `optimizer_m_states` / `optimizer_v_states`, LR scheduler state, `optimizer_state.step`, `current_micro_step`, `global_step`, `best_val_loss`, telemetry lattice, EMA, checkpoint metadata, `BatchDiagnostics` snapshots, `BatchLogTape`.
- ✅ Owned by `TrainingState` or by long-lived loop structs (`TrainingContext.optimizer`, etc.).
- ❌ MUST NOT alias or wrap a Category 1 tensor.
- Parameter grads are Category 2 only because the accumulation window deliberately persists them; intermediate (activation) grads are Category 1.

**Category 3 — Workspace / cache (PERSISTENT BUFFER, STALE CONTENTS)**
A buffer that persists for performance reasons but whose contents are meaningless across the boundary.
Examples: `cached_token_ids_tensor`, `cached_targets_tensor`, `cached_token_numeric_values`, `cached_token_to_slot_map`, `d_read_gate_accum`, deferred-cleanup queue object, `cublas_handle`, `stream_ctrl`.
- ✅ Owned by `TrainingState` directly (NOT by `AutogradIntermediates`).
- ✅ Re-zeroed / re-filled at the start of the next pass; never read across the boundary as if the previous pass's contents were valid.
- ❌ A Category 3 buffer in a Category 1 struct is a violation, even if `clear()` "skips freeing it" — that is an admission the field is in the wrong struct.

**Boundary rules:**

1. **Single boundary owner.** `autograd_intermediates.clear()` MUST be called from exactly one site (an RAII scope around the autograd step). Multiple call sites = ownership smell.
2. **Single teardown owner.** `flushDeferredCleanup()` is owned by `Tensor::backward()`. Additional calls outside it are forbidden unless code between them enqueues new deferred work (it does not, today).
3. **Tape sealing is explicit.** Once `loss_tensor` is read out as a host scalar, no further `autograd::add` / tape mutation is permitted on it. Functions that finish loss assembly and functions that read the loss scalar must be distinct.
4. **No "exception field" in `clear()`.** If a struct's `clear()` method has to skip a field, that field belongs to a different category — move it.

**Pre-commit checklist for ownership:**

- New field added to `AutogradIntermediates`? → It MUST be Category 1. If it's a buffer that persists, move it to `TrainingState`.
- New `autograd_intermediates.clear()` call site? → Delete it; route through the RAII scope instead.
- New `flushDeferredCleanup()` call outside `Tensor::backward()`? → Delete it.
- New "let me peek at `intermediates.X` after backward" code? → Snapshot it into a Category 2 diagnostics struct **before** the boundary instead.

**Why:** Category mixing is how `logits_tensor` (Category 1, ~tokens × vocab floats of GPU memory) ended up surviving the autograd boundary just because a histogram wanted it, and how `d_read_gate_accum` (Category 3 workspace) ended up with a special-case exemption inside a Category 1 `clear()`. Both are silent contracts that the next refactor will break. Enforcing the taxonomy at the type / file-organization level removes the entire class of bug.

---

## 🟡 Equation-Based Diagnostic Logging (Rule 21)

When adding diagnostic logging for ML math, use the `[*_EQUATION]` format:

```
[OPERATION_EQUATION] name: formula
  INPUT (desc): shape=[...] min=X max=Y rms=Z
  ACTUAL result: shape=[...] min=X max=Y rms=Z
  [ANOMALY] description     // only if wrong
```

Required: equation, input shape + (min/max/rms), actual result. Optional: PARAMETERS, EXPECTED (when predictable), ANOMALY tag. Use for any new forward/backward kernel, GEMM, or when debugging grad explosion / vanishing / loss anomalies.

---

## 📚 Feature Documentation

Project-specific knowledge lives in [`resources/models/GRIM-text/GRIM/Docs/`](../resources/models/GRIM-text/GRIM/Docs/README.md). Load only what the current task requires — do not pull every doc into context.

| Working on… | Read |
|---|---|
| Build / running the trainer or server | [Docs/Build.md](../resources/models/GRIM-text/GRIM/Docs/Build.md) |
| Three-phase training orchestration | [Docs/TrainingArchitecture.md](../resources/models/GRIM-text/GRIM/Docs/TrainingArchitecture.md) |
| GPU resources, streams, cuBLAS, optimizer state | [Docs/TrainingState.md](../resources/models/GRIM-text/GRIM/Docs/TrainingState.md) |
| TensorContract, GradFn, intermediates lifetime | [Docs/Autograd.md](../resources/models/GRIM-text/GRIM/Docs/Autograd.md) |
| Loss kernel or per-component gradient clipping | [Docs/Loss.md](../resources/models/GRIM-text/GRIM/Docs/Loss.md) |
| Grouped Query Attention shapes / backward scaling | [Docs/GQA.md](../resources/models/GRIM-text/GRIM/Docs/GQA.md) |
| FlashAttention v2 kernels (Dao-AILab integration) | [Docs/FlashAttention.md](../resources/models/GRIM-text/GRIM/Docs/FlashAttention.md) |
| LM head, tied embeddings, γ_final, hidden centering | [Docs/LMHead.md](../resources/models/GRIM-text/GRIM/Docs/LMHead.md) |
| Encoder layer (attention + FFN), bias autograd, LayerScale | [Docs/Encoder.md](../resources/models/GRIM-text/GRIM/Docs/Encoder.md) |
| ScratchBlock reasoning layer | [Docs/ScratchBlock.md](../resources/models/GRIM-text/GRIM/Docs/ScratchBlock.md) |
| ALiBi / RoPE position encoding | [Docs/PositionEncoding.md](../resources/models/GRIM-text/GRIM/Docs/PositionEncoding.md) |
| Tokenizer, AtomTable, sliding window | [Docs/Tokenizer.md](../resources/models/GRIM-text/GRIM/Docs/Tokenizer.md) |
| Weight initialization / Xavier | [Docs/Initialization.md](../resources/models/GRIM-text/GRIM/Docs/Initialization.md) |
| Diagnostics, telemetry, kernel timing, baselines | [Docs/Diagnostics.md](../resources/models/GRIM-text/GRIM/Docs/Diagnostics.md) |
| `ai_config.json` fields and conventions | [Docs/Config.md](../resources/models/GRIM-text/GRIM/Docs/Config.md) |
| C++/CUDA language traps (vector refs, atomics, sync) | [Docs/CppCudaFootguns.md](../resources/models/GRIM-text/GRIM/Docs/CppCudaFootguns.md) |
| Removed subsystems — do not recreate | [Docs/DeletedCode.md](../resources/models/GRIM-text/GRIM/Docs/DeletedCode.md) |

External references:
- [docs/LOG_FILE_CONVENTION.md](../docs/LOG_FILE_CONVENTION.md) — verify which log file before making claims
- [docs/PLATEAU_BUG_INVESTIGATION.md](../docs/PLATEAU_BUG_INVESTIGATION.md) — active training investigation notes

When you discover new feature-level knowledge, add it to the appropriate `Docs/*.md` file (or create a new one and add an index row), **not** to this file. This file is for cross-cutting rules only.

---

## Project Snapshot

GRIM-text is a custom transformer (Flash Attention v2, cuBLAS, custom fused CUDA kernels) with GQA (12 heads / 4 KV heads), Unigram + byte-fallback tokenizer, ScratchBlock reasoning layer, unified focal+smoothed-CE+entropy loss, and hierarchical TelemetryLattice. Config is `ai_config.json`. Server: `grim_text_server.exe` (port 11435). Trainer: `train_gpu.exe` (three-phase).

**GRIM-text is a SEPARATE build.** It MUST NOT include headers from `../../../../core/` or any G.R.I.M main-program libraries.
