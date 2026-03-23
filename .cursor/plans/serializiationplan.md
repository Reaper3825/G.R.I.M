You are implementing the SerializationLayer refactor.

This is a strict compiler-style system.

The system must be impossible to misuse, not just correctly implemented.

DO NOT introduce unrelated refactors.
DO NOT modify CUDA copy behavior.
DO NOT change architecture.

---

# CORE RULE

Checkpoint loading must be a pure validation → load pipeline:

VALIDATE → (if success) LOAD

NO GPU memory writes may occur before validation completes successfully.

---

# STEP 1 — AUTHORITATIVE CAPABILITY FLAGS

CheckpointCapabilityRequirements is the ONLY source of truth.

SerializationLayer MUST NOT infer requirements from:

* pointers
* buffer presence
* optional tables

---

# STEP 2 — SINGLE SOURCE OF VALIDATION TRUTH

ALL of the following MUST be enforced ONLY inside:

validate_checkpoint_capabilities(...)

NOT in load()

---

## REQUIRED CHECKS

### 1. Presence

For each requires_* flag:

* Required → MUST exist in FlatBuffer
* Not required → MUST be ignored (even if present)

---

### 2. Dimensions (MODEL CONTRACT, NOT BUFFER CONTRACT)

Validation MUST use:

* model config (d_model, num_heads, etc.)
* mathematically derived expected sizes

DO NOT rely solely on DeviceWriteView.count

Example:

* W_qkv size = (num_heads + 2 * num_kv_heads) * head_dim * d_model
* FFN sizes derived from d_ff, d_model

DeviceWriteView.count may be used ONLY as a secondary consistency check.

---

### 3. Tensor sizes

For each required module:

* checkpoint tensor size MUST equal expected size EXACTLY
* no resizing
* no padding
* no truncation

---

### 4. final_rms_gamma (STRICT)

If requires_final_rms_gamma:

* MUST exist in checkpoint
* MUST match expected size

NO fallback allowed anywhere else in the system.

---

# STEP 3 — STRUCTURAL GUARANTEE: NO PRE-VALIDATION LOAD

SerializationLayer::load must be structured as:

1. Read file
2. Verify FlatBuffer
3. Version check
4. CALL validate_checkpoint_capabilities()
5. IF FALSE → return immediately
6. ONLY THEN begin ANY upload_device_vector

It must be impossible to reach upload_device_vector before validation.

---

# STEP 4 — STRICT MODULE LOADING (NO LOGIC DUPLICATION)

load() must NOT re-check presence logic.

It must assume validation already guarantees correctness.

Therefore:

* NO "if (fb_numeric_head)" checks for requirement
* ONLY load based on requires_* flags

Example:

if (req.requires_numeric_head) {
load weights
report.numeric_head_loaded = true
}

Else:
skip completely

---

# STEP 5 — REMOVE ALL SILENT PATHS

Delete all logic that:

* silently ignores missing components
* loads conditionally based on existence
* uses legacy compatibility comments

These are forbidden.

---

# STEP 6 — POINTERS ARE NOT REQUIREMENTS

This is critical:

if (request.xyz.ptr)

ONLY means:

* destination exists

It must NEVER mean:

* module is required

---

# STEP 7 — FINAL LOAD VERIFICATION

After loading:

if required module not loaded → FAIL

This is a safety check, not primary validation.

---

# STEP 8 — LOGGING

All failures must:

* use EmitModuleError
* include:

  * required capability
  * actual checkpoint state
  * mismatch detail

---

# STEP 9 — CALL SITE (Pattern B)

grim_model_serialization.cu sets:

request.capabilities.*

SerializationLayer MUST NOT compute or override these.

---

# FINAL RULE

There must be exactly ONE place where compatibility is decided:

validate_checkpoint_capabilities()

After that point:

* load is deterministic
* no branching based on checkpoint content

---

If any compatibility decision exists outside validator → this is a bug.

Implement exactly this.
