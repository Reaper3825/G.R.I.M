# 🔧 ExecutionBlock FINAL HARD-FAIL VALIDATION ENFORCEMENT

You are upgrading the ExecutionBlock to a **fail-fast, zero-tolerance system**.

Do NOT change architecture.  
Do NOT add soft warnings.  
Do NOT allow degraded execution.

ALL invalid states MUST terminate execution immediately.

---

## Implementation status (codebase) ✅

1. **Warning-only paths removed** — No `d_warning_flag_`, no `fprintf` warn path.
2. **Collapse = fatal** — `kernelCheckEntropyCollapse` / `kernelCheckWriteCollapse` use `d_numeric_error_flag_` with `atomicMax`.
3. **Unified error system** — All failures: NaN/Inf, magnitude, softmax validity, entropy collapse (p_arg1, p_arg2, p_op), write collapse (`max(p_write)`).
4. **Early checks** — Collapse kernels run **immediately after** each softmax **before** matmul / blended write uses that distribution.
5. **End of step** — Single `h_error` read + `cudaStreamSynchronize`; if `h_error > 0`, `throw std::runtime_error` with step, `stageIdToName`, and id. `debug_mode` may `fprintf` the same message before throw only.
6. **debug_mode** — Does not downgrade failures; only optional stderr + `ExecStepMetrics` when `diag_out` is set.

---

# 1. REMOVE ALL WARNING-ONLY PATHS ✅

# 2. COLLAPSE = FATAL ERROR ✅

## 2.1 Argument Collapse (p_arg1, p_arg2) ✅  
## 2.2 Operation Collapse (p_op) ✅  
## 2.3 Write Collapse (p_write, max only — no entropy gate on p_write) ✅

# 3. UNIFY ERROR SYSTEM ✅

# 4. HARD FAIL ON FIRST INVALID STATE ✅

# 5. NO DEGRADED EXECUTION ALLOWED ✅

# 6. EXPAND FAILURE CONDITIONS ✅

# 7. DEBUG MODE DOES NOT CHANGE BEHAVIOR ✅

# 8. ERROR MESSAGE QUALITY ✅

Message shape:  
`ExecutionBlock FATAL: invalid state at step N — <stage description> (stage id=M)`

---

# FINAL REQUIREMENT

ExecutionBlock must:

- never silently degrade ✅
- never continue in invalid state ✅
- terminate immediately on collapse ✅
- enforce correctness over completion ✅

This is a STRICT fail-fast system.
