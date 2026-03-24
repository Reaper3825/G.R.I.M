# ExecutionBlock refactor — stakeholder survey

Related plan: [executionblockiterations.md](./executionblockiterations.md) (slot-only numeric truth, strict).

**Survey status:** All items **A1–G3** resolved **2026-03-23**. Two **open blockers** recorded at the end (implementation/policy still required).

---

## A. Data contract — `token_to_slot_map`

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **A1** | Where is `token_to_slot_map` produced? | Host prep / batch construction (first implementation). **Not** DataLoader. Built with numeric side channel; cached/uploaded via `BatchPayload` / `TrainingState`. **RESOLVED: 2026-03-23** |
| **A2** | Map indexing? | By **token position**: `token_to_slot_map[pos] -> slot_id`. `atom_positions[i]` gathers positions into that map (not atom-indexed map). **RESOLVED: 2026-03-23** |
| **A3** | Non-state-bearing positions? | Store **`-1`**. Sentinel valid **only** for non-state-bearing. `<NUM>` must **never** be `-1` → hard failure. **RESOLVED: 2026-03-23** |
| **A4** | Lifetime / ordering? | Stable full forward/execute sequence; stable for every `executeStep`. No mid-step mutation. DMA/upload: copy complete before ScratchBlock/ExecutionBlock read; read-only in step loop like other forward tensors. **RESOLVED: 2026-03-23** |

**Full text (A1–A4):** Not yet wired. First implementation in host prep / batch construction; DataLoader stays dataset-facing. `token_to_slot_map` is runtime reasoning substrate metadata, built alongside the existing numeric side channel, then cached/uploaded into `BatchPayload` / `TrainingState`. Indexed by token position; `atom_positions` is a gather list into that map. Non-state-bearing positions use `-1`; `<NUM>` with `-1` is a hard failure. Map is stable for the full forward/execute sequence; treat like other forward inputs (upload once, read-only for the step loop); guarantee device copy before any block read.

---

## B. Atom set vs slots

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **B1** | Mixed non-slot atoms in this path? | **v1:** every atom in this numeric path is state-bearing `<NUM>`. Do not support mixed non-slot atoms here; **filter before call**. Non-slot atoms must not enter numeric gather. **RESOLVED: 2026-03-23** |
| **B2** | Validation scope? | Validate **every** atom index passed into this `executeStep` — by contract only numeric/state-bearing; mixed atoms = caller violation, not gather masking. **RESOLVED: 2026-03-23** |
| **B3** | `num_atoms`, `V`, `C`? | Matches current `execution_block_GPU.cu` layout: prefix = atom candidates, tail = memory-slot candidates; **no padding** in this refactor unless forced later. **RESOLVED: 2026-03-23** |

---

## C. Semantics — validity and “initialized”

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **C1** | `M.valid_mask[slot] == 0`? | Not initialized / **not valid for read as authoritative state**. Reason (never written vs cleared vs new) **does not matter** to the read path. **RESOLVED: 2026-03-23** |
| **C2** | One message vs many? | **Distinct** messages at minimum: (1) invalid slot index, (2) missing slot mapping for required state-bearing token, (3) slot read before initialization (`valid_mask == 0`). **RESOLVED: 2026-03-23** |
| **C3** | `valid_mask == 0` in prod? | **Always fatal** on this execution-required numeric path. **No** debug bypass; tooling adapts to the contract. **RESOLVED: 2026-03-23** |

---

## D. Training and gradients

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **D1** | Signal path after MLP removal? | **Yes:** all numeric signal in ExecutionBlock via slot-backed reads/writes only; no learnable decode-from-embedding in this block. **RESOLVED: 2026-03-23** |
| **D2** | Old checkpoints? | **Break old runs.** No compat flag, no migration shim; retrain or invalidate experiments/checkpoints that depended on MLP numeric path. **RESOLVED: 2026-03-23** |
| **D3** | External `atom_decoded` assumptions? | May exist in debugging/logging/tests — **remove or update**. No external layer should keep `atom_decoded` as a live contract. **RESOLVED: 2026-03-23** |

---

## E. Scope and coordination

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **E1** | Other plans? | **[slot-referential_execution_7d963a3c.plan.md](./slot-referential_execution_7d963a3c.plan.md)** is authoritative; supersedes/absorbs older numeric-truth execution notes. Same implementation track; **no** split-brain PR stack keeping old decode semantics. **RESOLVED: 2026-03-23** |
| **E2** | Must-ship-first slice? | Plumbing/signature → slot-only gather → fail-hard validation → minimal tests. ScratchBlock masking / NumericHead scoping **immediately after**, but ExecutionBlock must **not** ship half-old/half-new. **RESOLVED: 2026-03-23** |
| **E3** | GPU scope? | **Single-GPU** first cut. Multi-GPU / complex stream topology out of scope; keep stream correctness, don’t design around distributed execution. **RESOLVED: 2026-03-23** |

---

## F. Verification

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **F1** | CI proof of “no hidden fallback”? | **Automated** test path: numeric execution case with required `<NUM>` atoms; removing/bypassing slot gather **must** hard-fail. Encode invariant in CI, not a manual recipe. **RESOLVED: 2026-03-23** |
| **F2** | Goldens? | **Yes for v1:** hard-failure tests **and** at least one tiny **golden** execution case for correct slot-backed numeric flow. **RESOLVED: 2026-03-23** |
| **F3** | Performance? | **Correctness-only** for this refactor; do not keep wrong architecture for MLP cost. **RESOLVED: 2026-03-23** |

---

## G. Open design edges

| ID | Question | Your answer (summary) |
|----|----------|------------------------|
| **G1** | `num_atoms == 0` and Phase 5 invariant? | **Pass by default** — no atoms ⇒ no required atom slot reads. Not an error unless a future explicit mode requires atoms (not v1). **RESOLVED: 2026-03-23** |
| **G2** | `kernelCheckFinite` scope? | **All `C` entries**, including slot tail — whole candidate numeric substrate must stay valid. **RESOLVED: 2026-03-23** |
| **G3** | Serialization byte-identical? | No requirement for this **runtime-only** map. Unrelated on-disk formats unchanged. New learnable params later ⇒ serialize **explicitly**, don’t hack around. **RESOLVED: 2026-03-23** |

---

## Open blockers (stakeholder-flagged)

These are **not** closed by the survey; they need explicit design or implementation follow-up.

1. **BLOCKER:** Generated `<NUM>` **slot assignment** still needs an explicit **runtime policy** if inference-side generation is expected to create new state-bearing numeric tokens during the same decoding run.

2. **BLOCKER:** **Caller contract** must guarantee `atom_positions` reaching this numeric path are **already filtered** to state-bearing numeric atoms only. If that cannot be guaranteed, the implementation needs an explicit **atom-kind** input plus a **validation layer** at the boundary.

---

## How to use

1. Implementation details live in [executionblockiterations.md](./executionblockiterations.md) (updated to mirror this survey).
2. Close blockers by editing this section or linking to a follow-on plan/PR.
