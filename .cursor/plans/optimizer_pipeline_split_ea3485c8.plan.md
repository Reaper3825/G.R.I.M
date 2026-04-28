---
name: Optimizer pipeline split — hardened
overview: |
  Replace the fused AdamW / RAdamW kernel with a host-driven, single-owner pipeline
  that judges update magnitude in ONE global coordinate system. Hardened to mirror
  Phase1_Startup.cu: a flat sequence of XxxReady(step) calls, one folder per concern,
  one .hpp/.cu pair per concern, one __device__ helper per concern, one test file
  per concern. Math (τ/system_scale, ρ_pred damping, ε_upd, λ_g K2≡K3, activation
  hook on the final-microbatch layer_output pre center_columns) is preserved verbatim
  and pinned to the file that owns it.
todos:
  - id: p0-scaffold
    content: P0 — folders + headers + stub Ready functions; build wired (Shared/**/*.cu glob); driver compiles and runs no-ops
    status: pending
  - id: p1-helpers
    content: P1 — single-source helpers (UpdateDirection.cuh updElement, LambdaGTable, Eligibility, TensorDebugTag); λ_g K2≡K3 contract enforced by API shape (kernels only accept direction through updElement)
    status: pending
  - id: p2-direction
    content: P2 — UpdateDirection K1Ready (Adam moment update writes final u to per-group u_device; RAdamW rectification site fixed inside this stage with configurable optimizer_radamw_rectify_threshold HP, default 5.0, validated >= 4.0; K1Ready is the SOLE writer of u before MeasureK2 reads it)
    status: pending
  - id: p3-measure
    content: P3 — MeasureK2 (per-group ∑(u+λθ)² + ∑θ² + N over u authored by K1Ready), AggregateGlobal (host weighted mean + ε_upd, G_inc fallback)
    status: pending
  - id: p4-controller
    content: P4 — EmaState, ActivationProbe (final-µbatch hook with runtime TensorDebugTag validation), ActivationFactor, SystemScale (lerp + τ + ε_ss), RhoController (ρ_now + damped ρ_pred + max-ahead + c_pred), SoftCap, PersistState
    status: pending
  - id: p5-apply
    content: P5 — ApplyK3 (θ -= η·α_g·scale_u·(u+λ_g θ) using same UpdateDirection.cuh); rewire launchAdamWStep / launchRAdamWStep; Phase2 hands EMAs + activation_rms_inst through; OptimizerCheckpoint v2 persists 4 clamp scalars (bit-exact roundtrip)
    status: pending
  - id: p6-tests-sim
    content: P6 — one *_test.cu per pipeline module + optimizer_pipeline_invariants_test.cu (sentinel-NaN ordering test) + optimizer_pipeline_sim_test.cu (12-layer, K=4 accum, regime shift @ step 25, ρ spike @ step 30, checkpoint @ step 25, replay = numerical parity within tolerance on a deterministic reduction path)
    status: pending
  - id: p7-cleanup
    content: P7 — delete fused-kernel weight-decay path; remove any per-group ρ scaffolding; lock JSON schema; doc τ semantics + frozen-group exclusion + RAdamW rectification site (and threshold HP semantics) in HyperParameters_GPU.hpp comments
    status: pending
isProject: false
---

# Optimizer step split + global ρ + global scale (AdamW + RAdamW) — hardened

## North star

Replace fused kernels with a **host-driven pipeline** where update magnitude is judged in **one global coordinate system**, not per-tensor or per-group silos. **Single source of truth per concern, per file.** The host driver looks like `Phase1_Startup::executePhase1` — flat sequence, one `XxxReady(step)` per concern, no branching, no loops at the driver level. **The model is far from a toy** (multi-layer encoder, gradient accumulation, embedding freeze, tied heads, RAdamW path, checkpoint resume), so every stage must own its preconditions and fail loud (Rule 20).

## Single-source-of-truth pattern (anchor)

`[Phase1_Startup.cu](resources/models/GRIM-text/training/Phases/Phase1_Startup.cu)` is the discipline:

```20:38:resources/models/GRIM-text/training/Phases/Phase1_Startup.cu
std::unique_ptr<TrainingContext> executePhase1(int argc, char** argv) {
    auto ctx = std::make_unique<TrainingContext>();
    LoggingReady(*ctx, argc, argv);
    MemorySnapshotReady(*ctx);
    HyperparametersReady(*ctx);
    CapacityStemReady(*ctx);
    DataInfoReady(*ctx);
    ModelAllocated(*ctx);
    ResumeStateReady(*ctx);
    GuessCacheReady(*ctx);
    TelemetryReady(*ctx);
    SchedulerPreflightReady(*ctx);
    EpochPlanReady(*ctx);
    PayloadBuildInputsReady(*ctx);
    PlannedBatchesReady(*ctx);
    StartupValidated(*ctx);
    Phase2HandoffReady(*ctx);
    return ctx;
}
```

**Rules (carried over to the optimizer pipeline):**

1. **One owned output bundle per `XxxReady`.** Each function authors a **named, semantically coupled bundle** of fields on the per-step bag — not necessarily a single scalar. `rhoControllerReady` owns `{rho_now, rho_pred, rho_for_clamp}`. `systemScaleReady` owns `{theta_eff, activation_term, blend, system_scale}`. The bundle is the unit of ownership; intra-bundle fields are coupled by the math and computed together. **No two stages may write to the same bundle.**
2. **One file per concern.** `Startup/<concern>/Xxx.hpp` (forward decl + signature), `Startup/<concern>/Xxx.cu` (impl). For optimizer: `Shared/Optimizers/Pipeline/<concern>/Xxx.{hpp,cu}` (+ `Xxx.cuh` if a `__device__` helper is needed).
3. **Throw on precondition failure** (Rule 20: no silent fallback). Pattern:

```
   if (!ctx.model) throw std::runtime_error("FATAL: GuessCacheReady requires ModelAllocated to have authored ctx.model");
   

```

1. **Driver is flat.** No conditionals, no loops. If a stage is conditional (e.g. clamp disabled when `ρ_max == 0`), the *stage* owns the conditional internally and writes a deterministic bundle (`scale_u_global = 1.f`) so downstream stays branchless.
2. **One test file per `XxxReady`.** `Tests/<concern>_test.cu` — same naming as existing `flash_attention_test.cu` / `causality_proof_tests.cu`.
3. **No stage reads fields not authored by an earlier stage.** Each Ready function takes a typed input view (e.g. `const RhoControllerInputs&`) that lists exactly the upstream bundles it consumes. Reading a downstream-authored field is a structural error, enforced two ways:
  - **Documentation:** every Ready's `.hpp` lists its `Reads:` and `Authors:` bundles; review-time check.
  - **Runtime invariant test:** `Tests/optimizer_pipeline_invariants_test.cu` initializes every yet-to-be-authored field with a sentinel `NaN` (or `-1` for ints), runs each Ready in driver order, and asserts no NaN propagates out. A stage that reads a field it doesn't own gets caught immediately because the sentinel poisons its math.

## Driver — the optimizer-step equivalent of `executePhase1`

```cpp
// Shared/Optimizers/Pipeline/OptimizerStepPipeline.cu
OptimizerStepResult runOptimizerStep(OptimizerStepInputs& step) {
    // Eligibility flags + LambdaGTable are built ONCE per launch — before this driver.
    // ActivationProbe runs PER MICROBATCH from autograd — before this driver.
    // This driver runs ONCE per optimizer step (after K microbatches accumulated).

    updateDirectionReady(step);        // owns: per-group u_device[]  ← K1 LIVES HERE (Adam moments → final direction)
                                       //       AdamW: u = m̂ · rsqrt(v̂ + ε)
                                       //       RAdamW: rectification computed AND APPLIED in this stage
                                       //       Post-condition: u_device[g][i] is the final direction MeasureK2 will read.
    measureK2Ready(step);              // owns: {per_group_sums[g] = (∑(u+λθ)², ∑θ², N_g)}  using u from updateDirectionReady
    aggregateGlobalReady(step);        // owns: {theta_inst, rms_update, G_inc, sum_w}
    activationFactorReady(step);       // owns: {f_static, activation_scale_factor, regime_bypassed}
    systemScaleReady(step);            // owns: {theta_eff, activation_term, blend, system_scale}
    rhoControllerReady(step);          // owns: {rho_now, rho_pred, rho_for_clamp}     ← coupled bundle
    softCapReady(step);                // owns: {scale_u_global}     (=1 if G_inc==0 OR ρ_max==0)
    applyK3Ready(step);                // owns: device weight delta (θ -= η·α_g·scale_u·(u+λ_g θ)) using SAME u from updateDirectionReady
    persistStateReady(step);           // owns: {θ_ema, activation_rms_ema, factor_prev, rho_prev} advance
    return step.result;
}
```

**This is the entire driver.** Nine calls, one bundle per concern, in fixed order. No early returns, no conditionals. Edge cases (G_inc==0, clamp off, cold-start EMA, K=1 vs K>1, embedding freeze) live **inside** the owning stage.

**The K1 contract is hard:** `updateDirectionReady` is the **sole writer** of `step.u_device[g]`. By the time `measureK2Ready` reads it, `u` is the **final** direction the apply step will use. K2 and K3 see the **same bytes**. RAdamW rectification (and the SGD-of-`m̂` fallback when `ρ_t` is below the rectification threshold) happens inside `updateDirectionReady` — not later, not in K3.

### Sibling drivers (not part of `runOptimizerStep`)

- `**markRhoEligibilityReady(ctx)`** — runs **once** in Phase1, after `ModelAllocated`, **before** any optimizer step. Authors per-`ParameterGroup` `is_bias`, `is_norm_affine`, `exclude_from_global_rho` flags from `ParamGroupType` + `N_g`.
- `**markActivationRmsInstReady(ts, layer_output, is_final_microbatch)`** — runs **per microbatch** from `AutogradTraining.cu` at the canonical hook (last encoder layer, `layer_output`, **before** `center_columns`). Writes `ts.activation_rms_inst` only when `is_final_microbatch`. `K=1` is the degenerate case.
- `**buildLambdaGTableReady(ctx)`** — runs **once per launch record** (i.e. once per call to `launchAdamWStep` / `launchRAdamWStep`), authoring the device `lambdas[g]` array used by **both** K2 and K3 from one `OptimizerStepGroupDevice` builder.

## File taxonomy — single owner per file

Every row is **one owner of one bundle** + **one test file**. Mirrors `Startup/<concern>/Xxx.{hpp,cu}` exactly. Folders are new under `Shared/Optimizers/Pipeline/`.


| Stage (canonical order)             | Owner header / impl                                                                               | `__device__` helper                                                                                      | Owns (bundle written to `OptimizerStepInputs.step`)                                                                                                                        | Reads (must be authored upstream)                                                                                                                                                                            | Test file                                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Eligibility (one-shot @ Phase1)     | `Pipeline/Eligibility/Eligibility.{hpp,cu}`                                                       | —                                                                                                        | `ParameterGroup::{is_bias, is_norm_affine, exclude_from_global_rho}`                                                                                                       | `ParamGroupType`, `N_g`, `optimizer_global_rho_min_group_n`                                                                                                                                                  | `Tests/optimizer_eligibility_test.cu`                                                                                    |
| λ_g table (per launch)              | `Pipeline/UpdateDirection/LambdaGTable.{hpp,cu}`                                                  | —                                                                                                        | `step.lambda_g_device[]` (one device array per launch)                                                                                                                     | per-group `weight_decay_multiplier`, global `weight_decay`, `upsilon`                                                                                                                                        | `Tests/optimizer_lambda_g_test.cu`                                                                                       |
| `upd` helper (single defn)          | `Pipeline/UpdateDirection/UpdateDirection.cuh`                                                    | `__device__ float updElement(float u, float lambda_g, float theta) { return fmaf(lambda_g, theta, u); }` | — (header-only; **#include**'d by K2 and K3)                                                                                                                               | —                                                                                                                                                                                                            | covered transitively by K2 / K3 / golden tests                                                                           |
| **K1 — UpdateDirection (per step)** | `Pipeline/UpdateDirection/UpdateDirection.{hpp,cu}`                                               | —                                                                                                        | `step.u_device[g]` (per-group **final direction** buffer) — AdamW: `m̂·rsqrt(v̂+ε)`; RAdamW: rectified `m̂·rsqrt(v̂+ε)·r_t` when `ρ_t > radamw_rectify_threshold`, SGD-of-`m̂` fallback otherwise | groups (Adam state `m, v`), `optimizer_step` (bias correction), `optimizer_kind`, `optimizer_beta1/2`, `optimizer_epsilon`, `optimizer_radamw_rectify_threshold`                                                                                   | `Tests/optimizer_update_direction_test.cu`                                                                               |
| K2 measure                          | `Pipeline/MeasureK2/MeasureK2.{hpp,cu}`                                                           | uses `UpdateDirection.cuh`                                                                               | `step.per_group_sums[g] = {sum_upd_sq, sum_theta_sq, N_g}`                                                                                                                 | `step.u_device[g]` (from K1), `lambda_g_device`, `θ`, `eligible`, `frozen`                                                                                                                                   | `Tests/optimizer_measure_k2_test.cu`                                                                                     |
| Aggregate global                    | `Pipeline/AggregateGlobal/AggregateGlobal.{hpp,cu}`                                               | —                                                                                                        | `step.{theta_inst, rms_update, G_inc, sum_w}`                                                                                                                              | `per_group_sums`, `eligible`, `optimizer_global_rms_group_weight`, `optimizer_rho_rms_upd_eps`                                                                                                               | `Tests/optimizer_aggregate_global_test.cu`                                                                               |
| Activation probe (per µbatch)       | `Pipeline/ActivationProbe/ActivationProbe.{hpp,cu}`                                               | —                                                                                                        | `TrainingState::activation_rms_inst`                                                                                                                                       | last encoder `layer_output` pre `center_columns` **with `TensorDebugTag` matching `{layer_id == num_layers-1, stage_name == "encoder_layer_output_post_inject", centered == false}`**, `is_final_microbatch` | `Tests/optimizer_activation_probe_test.cu`                                                                               |
| Activation factor                   | `Pipeline/ActivationFactor/ActivationFactor.{hpp,cu}`                                             | —                                                                                                        | `step.{f_static, activation_scale_factor, regime_bypassed}`                                                                                                                | `θ_ema`, `activation_rms_ema`, `factor_prev`, all `optimizer_rho_activation_factor_*` HPs                                                                                                                    | `Tests/optimizer_activation_factor_test.cu`                                                                              |
| System scale                        | `Pipeline/SystemScale/SystemScale.{hpp,cu}`                                                       | —                                                                                                        | `step.{theta_eff, activation_term, blend, system_scale}`                                                                                                                   | `θ_inst`, `θ_ema`, `activation_rms_ema`, `activation_scale_factor`, `α`, `τ`, `γ`, `θ_floor`, `ε_ss`, `ε_act`, `s_act`                                                                                       | `Tests/optimizer_system_scale_test.cu`                                                                                   |
| ρ controller                        | `Pipeline/RhoController/RhoController.{hpp,cu}`                                                   | —                                                                                                        | `step.{rho_now, rho_pred, rho_for_clamp}`                                                                                                                                  | `η_base`, `rms_update`, `system_scale`, `rho_prev`, `optimizer_rho_slope_k`, `optimizer_rho_pred_damp`, `optimizer_rho_pred_max_ahead_ratio`, `optimizer_rho_pred_cap_multiplier`                            | `Tests/optimizer_rho_controller_test.cu`                                                                                 |
| Soft cap                            | `Pipeline/SoftCap/SoftCap.{hpp,cu}`                                                               | —                                                                                                        | `step.scale_u_global`                                                                                                                                                      | `rho_for_clamp`, `optimizer_update_rho_max`, `optimizer_update_soft_cap_start_ratio`, `optimizer_update_soft_cap_power`, `eps_clamp`, `G_inc`                                                                | `Tests/optimizer_soft_cap_test.cu`                                                                                       |
| Apply K3                            | `Pipeline/ApplyK3/ApplyK3.{hpp,cu}`                                                               | uses `UpdateDirection.cuh`                                                                               | device weight delta `θ -= η·α_g·scale_u·(u+λ_g θ)`                                                                                                                         | groups, `step.u_device[g]` (**same buffer** K2 read), `lambda_g_device`, `η_base`, `scale_u_global`, embedding-freeze policy, frozen flags                                                                   | `Tests/optimizer_apply_k3_test.cu`                                                                                       |
| Persist state                       | `Pipeline/PersistState/PersistState.{hpp,cu}`                                                     | —                                                                                                        | `θ_ema`, `activation_rms_ema`, `activation_scale_factor_prev`, `rho_prev` (in `OptimizerClampState`)                                                                       | end-of-step `θ_inst`, `activation_rms_inst`, applied factor, `rho_now`, β from `τ_steps`                                                                                                                     | `Tests/optimizer_persist_state_test.cu`                                                                                  |
| Driver                              | `Pipeline/OptimizerStepPipeline.{hpp,cu}`                                                         | —                                                                                                        | calls all 9 above in order                                                                                                                                                 | —                                                                                                                                                                                                            | `Tests/optimizer_pipeline_sim_test.cu` (E2E sim) + `Tests/optimizer_pipeline_invariants_test.cu` (sentinel-NaN ordering) |
| Checkpoint extension                | `[OptimizerCheckpoint.cu](resources/models/GRIM-text/training/OptimizerCheckpoint.cu)` (existing) | —                                                                                                        | persists/restores `OptimizerClampState` (4 scalars) — bumps sidecar `version` to **2**                                                                                     | `OptimizerClampState`                                                                                                                                                                                        | `Tests/optimizer_checkpoint_v2_test.cu`                                                                                  |
| Per-step bag (struct)               | `Pipeline/OptimizerStepInputs.hpp`                                                                | —                                                                                                        | declares `OptimizerStepInputs`, `OptimizerStepResult`, `OptimizerClampState`, `PerGroupSums`, `OptimizerHpView`, `TensorDebugTag`                                          | —                                                                                                                                                                                                            | (compile-only contract; covered by sim + invariants test)                                                                |


**Forbidden:** any optimizer-related `.cu` outside `Pipeline/` that re-implements `lambda * theta`, `lerp`, `system_scale`, `rho_pred`, or `scale_u` math. CI grep gate enforced (see `P6 — Cleanup`).

## Per-step bag

```cpp
// Shared/Optimizers/Pipeline/OptimizerStepInputs.hpp
struct OptimizerClampState {                   // checkpointed
    float theta_ema                       = 0.0f;     // cold-start sentinel
    float activation_rms_ema              = 0.0f;     // cold-start sentinel
    float activation_scale_factor_prev    = 0.0f;     // 0 ⇒ "invalid / no prev" (cold start)
    float rho_prev                        = 0.0f;     // 0 ⇒ first step
};

struct PerGroupSums {                          // produced by MeasureK2
    double sum_upd_sq = 0.0;
    double sum_theta_sq = 0.0;
    std::size_t N_g = 0;
    bool eligible = false;                     // mirrors ParameterGroup::exclude_from_global_rho == false
    bool frozen = false;                       // true ⇒ skipped by K1, K2, K3 (e.g. embedding past freeze step)
};

// Runtime tensor provenance. Attached to any tensor that flows into ActivationProbe.
// Replaces the (unenforceable) "compile-time tensor identity" idea — see § ActivationProbe.
struct TensorDebugTag {
    int   layer_id          = -1;              // 0..num_layers-1; -1 ⇒ unset
    const char* stage_name  = nullptr;         // e.g. "encoder_layer_output_post_inject"
    bool  centered          = true;            // true ⇒ post center_columns (forbidden source for ActivationProbe)
    bool  is_microbatch_final = false;         // set by AutogradTraining when this is the last µbatch's tensor
};

struct OptimizerStepInputs {
    // INPUTS (filled by caller / Phase2)
    std::vector<GRIM::ParameterGroup>* groups = nullptr;
    const float* lambda_g_device = nullptr;    // device pointer; one entry per group (LambdaGTable)
    float eta_base   = 0.0f;                   // already through the LR schedule
    float activation_rms_inst = 0.0f;          // FINAL microbatch, written by ActivationProbe
    OptimizerClampState clamp_state{};         // copy in; PersistState writes back
    OptimizerHpView hp{};                      // POD view of all `optimizer_`* HPs
    cudaStream_t stream = nullptr;
    int embedding_freeze_after_step = -1;
    int optimizer_step = 0;

    // PIPELINE-AUTHORED (each Ready writes one bundle; bundles are coupled fields)
    std::vector<float*> u_device;             // updateDirectionReady (K1) — per-group final direction (Adam or RAdam-rectified)
    std::vector<PerGroupSums> per_group_sums; // measureK2Ready
    // — AggregateGlobal bundle —
    int    G_inc      = 0;
    double sum_w      = 0.0;
    float  theta_inst = 0.0f;
    float  rms_update = 0.0f;
    // — ActivationFactor bundle —
    float  f_static               = 0.0f;
    float  activation_scale_factor= 0.0f;
    bool   regime_bypassed        = false;
    // — SystemScale bundle —
    float  theta_eff       = 0.0f;
    float  activation_term = 0.0f;
    float  blend           = 0.0f;
    float  system_scale    = 0.0f;
    // — RhoController bundle —
    float  rho_now       = 0.0f;
    float  rho_pred      = 0.0f;
    float  rho_for_clamp = 0.0f;
    // — SoftCap bundle —
    float  scale_u_global = 1.0f;             // safe pass-through default

    OptimizerStepResult result{};             // ApplyK3 fills (telemetry); PersistState advances state
};
```

`OptimizerHpView` is a POD copy of the optimizer HPs (built once at Phase1, refreshed when JSON reloads), so no Ready function reads `ctx.config` directly — it gets a typed view. This keeps each Ready unit-testable without a full `TrainingContext`.

**Bundle ownership invariant:** the comments in the struct above are normative — exactly one Ready writes to each bundle, in driver order. `Tests/optimizer_pipeline_invariants_test.cu` poisons every yet-unauthored field with `NaN` (or `-1` for ints / `nullptr` for pointers) before each stage runs and asserts no poison leaks into outputs (rule 6).

## Canonical host order (math reference — owned by SystemScale + RhoController + SoftCap)

Math is unchanged from the prior plan; restated here as the **only** reference. Every Ready function in the pipeline implements **exactly** the lines under its own header — no extra rules elsewhere.

```text
// updateDirectionReady (K1) — owns u_device[g], per group g (NOT frozen):
//   AdamW:   m'   = β1·m + (1-β1)·grad
//            v'   = β2·v + (1-β2)·grad²
//            m̂   = m' / (1 - β1^(step+1))
//            v̂   = v' / (1 - β2^(step+1))
//            u   = m̂ * rsqrt(v̂ + ε_adam)               ← FINAL DIRECTION (no decoupled WD here)
//   RAdamW:  ρ_t  = ρ∞ - 2t·β2^t / (1 - β2^t)
//            if ρ_t > optimizer_radamw_rectify_threshold:        // default 5.0; configurable; validated >= 4.0
//              r_t = sqrt(((ρ_t-4)(ρ_t-2)·ρ∞) / ((ρ∞-4)(ρ∞-2)·ρ_t))   // 4 is mathematical, NOT the threshold HP
//              u   = m̂ * rsqrt(v̂ + ε_adam) * r_t       ← rectified
//            else:
//              u   = m̂                                  ← SGD-of-m̂ fallback (per Liu et al. 2019)
//   Frozen groups: u_device[g] = nullptr (or unwritten); MeasureK2 / ApplyK3 skip them.
//   Post-condition: u_device[g][i] is the BYTES MeasureK2 will read AND ApplyK3 will apply.

// MeasureK2 (device) + AggregateGlobal (host)
//   per group g (eligible AND not frozen):
//     sum_upd_sq[g]   = Σ_i (u_i + λ_g θ_i)^2     ← uses UpdateDirection.cuh::updElement on u from K1
//     sum_theta_sq[g] = Σ_i θ_i^2
//     N_g             = group element count
//   host weighted mean (uniform w_g=1 default; log_n optional):
//     ms_θ_g    = sum_theta_sq[g] / N_g
//     ms_upd_g  = sum_upd_sq[g]   / N_g
//     ms̄_θ      = (Σ_g w_g · ms_θ_g)   / Σ_g w_g     // over eligible & not-frozen groups
//     ms̄_upd    = (Σ_g w_g · ms_upd_g) / Σ_g w_g
//     theta_inst = sqrt(ms̄_θ)
//     rms_update = sqrt(ms̄_upd + eps_upd)             // eps_upd default 1e-16f
//     G_inc      = count of eligible & not-frozen groups with N_g > 0
//   if G_inc == 0: short-circuit later — SoftCap will set scale_u_global = 1.

// ActivationFactor (host)
//   raw_factor = theta_ema / (activation_rms_ema + eps_act)
//   f_static   = clamp(raw_factor, f_min, f_max)
//   if !factor_prev_valid:                       // factor_prev == 0 ⇒ invalid
//       activation_scale_factor = f_static
//   else:
//       span = max(f_static / factor_prev, factor_prev / f_static)
//       if regime_bypass && span > regime_span_max:
//           activation_scale_factor = f_static          // regime_bypassed = true
//       else:
//           activation_scale_factor = clamp(f_static, factor_prev * r_min, factor_prev * r_max)

// SystemScale (host)
//   theta_eff       = max(theta_ema, gamma * theta_inst, theta_floor)        // three-way max
//   activation_scaled = activation_rms_ema * activation_scale_factor
//   activation_term   = activation_scaled * s_act                            // s_act default 1.f
//   blend           = lerp(theta_eff, activation_term, alpha_blend)          // α ∈ [0,1]
//   tau_clamped     = clamp(tau, 1e-6f, 1.f)                                  // τ in (0,1]
//   system_scale    = max(blend, tau_clamped * theta_eff, eps_ss)            // τ semantics: § Design anchor

// RhoController (host)
//   rho_now   = eta_base * rms_update / system_scale                          // system_scale already floored
//   delta_rho = k_slope * (rho_now - rho_prev)                                // first step ⇒ rho_prev=0 ⇒ delta=0
//   rho_pred  = rho_now + rho_pred_damp * delta_rho                           // d default 0.5f
//   rho_pred  = min(rho_pred, rho_now * (1.f + rho_pred_max_ahead_ratio))     // r_ahead default 0.35f
//   rho_pred  = min(rho_pred, rho_now * rho_pred_cap_multiplier)              // c_pred default 2.0f (wide backstop)
//   rho_for_clamp = max(rho_now, rho_pred)

// SoftCap (host)
//   if G_inc == 0 OR rho_max <= 0:
//       scale_u_global = 1.f
//   else:
//       rho_soft_start = rho_max * soft_start_ratio                           // default 0.9f
//       if rho_for_clamp <= rho_soft_start:
//           scale_u_global = 1.f
//       else:
//           scale_u_global = powf(rho_soft_start / (rho_for_clamp + eps_clamp), p)
//       // never scale_u_global > 1 by construction

// ApplyK3 (device, per group; eligibility is for ρ-MEASUREMENT, not for APPLY):
//   foreach group g (NOT frozen):
//       theta_i -= eta_base * alpha_g * scale_u_global * updElement(u_i, lambda_g, theta_i)
//       // u_i      comes from the SAME step.u_device[g] MeasureK2 read — bit-identical bytes.
//       // λ_g      comes from the SAME lambda_g_device MeasureK2 read — not recomputed.
//       // Excluded-from-ρ groups (biases, norm γ/β, tiny-N) STILL apply with the same scale_u_global.
//       // Frozen groups (e.g. embedding past freeze step) skip apply AND were already skipped by K1/K2.

// PersistState (host, end of step):
//   theta_ema                    = beta_w * theta_ema    + (1 - beta_w) * theta_inst
//   activation_rms_ema           = beta_a * activation_rms_ema + (1 - beta_a) * activation_rms_inst
//   activation_scale_factor_prev = activation_scale_factor   // applied (post-rate-clamp / post-bypass)
//   rho_prev                     = rho_now
//   // Cold-start: if theta_ema was 0 BEFORE this step, replace-instead-of-blend (one-shot).
```

### Design anchor — τ (preserved verbatim)

`τ` (`optimizer_system_scale_theta_rel_min`) is **not** a numeric floor. `system_scale = max(blend, τ·θ_eff, ε_ss)` with `τ ∈ (0,1]`:

- `ε_ss` exists for **numerical** sanity (denormal denominators).
- `τ·θ_eff` bounds **how far below `θ_eff`** `system_scale` may sit when `activation_term ≪ θ_eff` would otherwise drive `blend` lower. `**τ=0.88` ⇒ `system_scale ≥ 0.88·θ_eff` ⇒ at most ~12 % below `θ_eff` from the τ limb.**
- `τ=1` ⇒ uplift-only: forward activations **never** lower `system_scale` below `θ_eff` (activations only matter upward).
- `τ ≪ 1` ⇒ `τ·θ_eff` permits much larger downward pull before `blend` dominates.

**Tune `τ` by semantics**, not by chasing safety. Activation **down-voice** is **co-governed** by `α` (mix weight) and `τ` (down-influence ceiling).

### Effective denominator (preserved)

`θ_eff = max(θ_ema, γ·θ_inst, θ_floor)` — three-way max kills EMA-lag spikes when weights actually grew (γ default `1.0f` — `γ·θ_inst = θ_inst` whenever it exceeds `θ_ema`). `θ_floor` default `1e-3f` for cold start.

### Activation hook (canonical source — owned by ActivationProbe)

**One** reproducible tensor and **one** formula. Not a "global activation RMS":

- **Tensor:** the **last encoder layer** hidden state `layer_output` in `[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)` **after** `EncodingLayer::forward` and any `crossAttentionRead` injection (Issue #155 path), and **before** `if (cfg->center_encoder_residuals) { layer_output = autograd::center_columns(...); }`.
- **Scalar:** elementwise `RMS = sqrt((1/N) Σ x_i²)` over `[total_tokens, d_model]`.
- **Forbidden:** LM head input (post final RMSNorm / centering / projection), logits, embeddings-only norms, mid-stack layers, post-`center_columns` tensor.

#### Tensor identity validation (runtime, not compile-time)

The original spec asserted "compile-time tensor identity check." That is **fantasy** — C++ has no runtime tensor-graph metadata it can lift to types, and `Tensor`* carries no provenance at the type level. The check is **runtime, fail-loud** via a small debug tag attached to the tensor at the canonical hook site:

```cpp
// Set in AutogradTraining.cu IMMEDIATELY at the canonical hook point:
TensorDebugTag tag {
    .layer_id           = num_layers - 1,                       // last encoder layer ONLY
    .stage_name         = "encoder_layer_output_post_inject",   // string-equality matched
    .centered           = false,                                // pre center_columns
    .is_microbatch_final= (microbatch_idx == K - 1),            // gradient-accumulation gate
};

// Inside markActivationRmsInstReady (Pipeline/ActivationProbe/ActivationProbe.cu):
void markActivationRmsInstReady(TrainingState& ts, const Tensor& layer_output,
                                const TensorDebugTag& tag) {
    if (tag.layer_id != ts.num_encoder_layers - 1)  throw std::runtime_error("FATAL: ActivationProbe got layer_id=" + ... + ", expected last encoder layer");
    if (std::strcmp(tag.stage_name, "encoder_layer_output_post_inject") != 0)
                                                    throw std::runtime_error("FATAL: ActivationProbe got stage_name='" + ... + "', expected 'encoder_layer_output_post_inject'");
    if (tag.centered)                               throw std::runtime_error("FATAL: ActivationProbe got centered=true tensor — must be PRE center_columns");
    if (!tag.is_microbatch_final) return;           // earlier µbatches: no-op (writes nothing)
    ts.activation_rms_inst = computeRmsOnDevice(layer_output, ts.stream);
}
```

Three guarantees this provides:

1. Wrong layer ⇒ throw (e.g. accidentally hooked at layer `N/2`).
2. Wrong stage ⇒ throw (e.g. accidentally hooked on the LM head input or post-center).
3. Wrong gate ⇒ silent no-op for non-final microbatches; only the final one writes. Earlier values can never leak.

`TensorDebugTag` lives in `Pipeline/OptimizerStepInputs.hpp` (cheap POD; no allocation). Production builds keep the validation (Rule 20: fail loud). The tag is **set at exactly one site** in `AutogradTraining.cu`; any other caller of `markActivationRmsInstReady` would have to forge the tag — caught in code review (one new file to grep for).

**Gradient accumulation rule:** with `K ≥ 1` microbatches per optimizer step, `activation_rms_inst = last_microbatch_value` — the RMS scalar from the **final** forward in the accumulation window. The `is_microbatch_final` gate enforces this. Earlier microbatches do not write (no scratch leakage). `K=1` is the degenerate case (the only microbatch is the final one).

### λ_g and `u` consistency (K1 → K2 ≡ K3) — enforced by API shape

The real guarantee is **the kernel APIs cannot express the wrong thing.** K2 and K3 take the **same `u_device[g]` pointer** (written by K1) and the **same `lambda_g_device` pointer** (written by `LambdaGTable`), and the **only** way to combine them into the apply-direction is `updElement(u, λ_g, θ)`. There is no alternate path inside the kernels — they don't know about `weight_decay`, `weight_decay_multiplier`, or `upsilon`. Those collapse into `lambda_g_device[g]` upstream, in **one** function.

**Architectural invariants:**

1. **Single source of truth per group, per step** — one `lambda_g` written **once** by `buildLambdaGTableReady(...)` from the group's `weight_decay_multiplier × upsilon × global weight_decay` rule. **Only** that builder computes `weight_decay × multiplier`.
2. **Single `u_device[g]`** — written **once** per step by `updateDirectionReady` (K1). K2 and K3 read the **same bytes** (same device pointer, same stream-ordered launch). RAdamW rectification is applied **inside** K1 — K3 is unaware of optimizer kind.
3. **Single `__device__` helper** — `UpdateDirection.cuh::updElement(u, λ_g, θ) = fmaf(λ_g, θ, u)`. **Both** K2 and K3 `#include` it.
4. **Same launch contract** — `runOptimizerStep` passes one `step` bag containing both pointers; K2 and K3 read from the same `step.u_device[g]` and `step.lambda_g_device[g]`. They take these as `const float* __restrict__` kernel parameters — no global state.

**CI grep gate (guardrail, not guarantee).** `Tests/optimizer_lambda_g_test.cu` does a `ripgrep` scan over `Shared/Optimizers/**/*.cu` for the patterns:

- `\b(lambda|λ)_g?\s*\*\s*(theta|θ)`
- `weight_decay\s*\*\s*(param|theta)`
- `fmaf?\s*\(\s*weight_decay`

…outside `UpdateDirection.cuh` and the `LambdaGTable` builder. **Caveats explicitly documented:** the regex will miss `fmaf(wd, p, u)` if `wd` is renamed; it can match a comment that mentions the formula. **The grep is a smoke test for new contributors, not a soundness proof.** The actual enforcement is the API shape: kernels take typed pointers; recomputing `wd × param` inside a kernel requires either (a) introducing a new kernel parameter (caught in PR review) or (b) hardcoding a constant (caught by test 4 below — `λ_g` regression test fails if the kernel doesn't actually consume `lambda_g_device`).

### ρ-eligibility (preserved — owned by Eligibility) + frozen policy

**Two orthogonal exclusion axes:**

```text
// Axis 1 — STATIC eligibility (set once at Phase1, immutable for run)
if (group.is_bias || group.is_norm_affine || group.N < min_global_group_N)
    exclude_from_global_rho = true;

// Axis 2 — DYNAMIC freeze (set per step by Phase2 from embedding_freeze_after_step)
group.frozen = embedding_freeze_active && (group.type == EMBEDDING ||
               (group.type == LM_HEAD && group.name == "embedding_lm_head_tied"));
```

**Locked policy (decided — was previously ambiguous):**


| Axis                                                        | K1 (UpdateDirection)                                    | K2 (Measure)                               | Aggregate (`G_inc`) | K3 (Apply)                            |
| ----------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------ | ------------------- | ------------------------------------- |
| `exclude_from_global_rho == true` (bias / norm γβ / tiny-N) | **runs** (group still steps; needs Adam direction)      | **skipped**                                | not counted         | **runs** (uses same `scale_u_global`) |
| `frozen == true` (embedding past freeze step)               | **skipped** (no direction needed; weights don't change) | **skipped** (no `u`, no contribution to ρ) | not counted         | **skipped**                           |


**Rationale for excluding frozen groups from K2/aggregation:**

- A frozen group's `θ` does not change ⇒ its contribution to `θ_inst` is **stale** in a different way than a stepped group's. Including it pulls `θ_inst` toward a frozen reference that no longer reflects active training.
- More concretely, a frozen group has **no `u`** (K1 skipped). Including a `u=0` term in `ms̄_upd` would artificially deflate `rms_update` and pull `ρ_now` down — a silent free pass to other groups when the embedding stops absorbing signal.
- The opposite choice (keep frozen groups in K2 with `u=0`) is testable but rejected as default: see **embedding-freeze regression** in `optimizer_apply_k3_test.cu` and the `frozen-group ρ contamination` row in the sim-test assertions.

`**min_global_group_N` default `4096`**. Statically-excluded groups still receive the same `scale_u_global` at apply time — they are only omitted from **measuring** ρ. **Edge: `G_inc == 0` ⇒ `scale_u_global = 1`, no ρ damping that step.** Set on `ParameterGroup` once in Phase1 from `ParamGroupType` + `N_g` so K2 / host filtering is branchless. `frozen` is set per step by Phase2 (or by `runOptimizerStep`'s caller) from the existing `embedding_freeze_after_step` policy.

## Phases — dependency-clean ordering

Each phase is mergeable on its own. Each leaves the build green and (from P3 onward) the in-tree sim test green. The pipeline now has **9 host stages** (driver order): `updateDirectionReady → measureK2Ready → aggregateGlobalReady → activationFactorReady → systemScaleReady → rhoControllerReady → softCapReady → applyK3Ready → persistStateReady`.

### P0 — Scaffolding (one PR)

- Create folders `Shared/Optimizers/Pipeline/{Eligibility, UpdateDirection, MeasureK2, AggregateGlobal, ActivationProbe, ActivationFactor, SystemScale, RhoController, SoftCap, ApplyK3, PersistState}` plus `Pipeline/OptimizerStepPipeline.{hpp,cu}` and `Pipeline/OptimizerStepInputs.hpp`.
- Stub every `XxxReady(step)` to set its owned bundle to a documented sentinel and otherwise no-op. `**updateDirectionReady` stub allocates `step.u_device[g] = nullptr` per group; downstream stages must tolerate the stub.**
- Wire build (`Shared/**/*.cu` glob already includes — confirm). Add `Tests/optimizer_*_test.cu` to the test target if not already globbed.
- Driver compiles and `runOptimizerStep` returns a deterministic `result` with `scale_u_global = 1.f` for any input. **Existing `launchAdamWStep` / `launchRAdamWStep` untouched** (parallel scaffolding only).
- Add `OptimizerHpView` builder in `HyperParameters_GPU.hpp`; populate from `StartupConfig.hyperparameters` (read JSON keys listed in **Hyperparameters** below; defaults applied here).
- Add `TensorDebugTag` POD to `Pipeline/OptimizerStepInputs.hpp`.

**Exit criteria:** `runOptimizerStep` linkable from a unit test; sentinel result; no behavior change in training.

### P1 — Single-source helpers (one PR)

- `UpdateDirection.cuh` — single `__device__` definition of `updElement`. Header-only.
- `LambdaGTable.{hpp,cu}` — `buildLambdaGTableReady(groups, hp) → DeviceVector<float>`. One device array per launch. Builder is the only place `weight_decay × weight_decay_multiplier × upsilon` is computed.
- `Eligibility.{hpp,cu}` — `markRhoEligibilityReady(ctx)` runs in Phase1 right after `ModelAllocated`. Adds `is_bias`, `is_norm_affine`, `exclude_from_global_rho` fields to `ParameterGroup` (`TensorContract_GPU.hpp`).
- `Tests/optimizer_lambda_g_test.cu` — golden helper parity + λ-actually-consumed regression + grep guardrail.
- `Tests/optimizer_eligibility_test.cu` — biases excluded; `N < 4096` excluded; `LM_HEAD` weights included.

**Exit criteria:** `LambdaGTable` is the sole caller of `weight_decay × multiplier`; `updElement` is the sole helper for `λθ + u` math.

### P2 — UpdateDirection (K1) (one PR — NEW STAGE; replaces "K1 stays where it is")

K1 is a real, owned stage. Without it, MeasureK2 has no contract about what `u` looks like.

- `UpdateDirection.{hpp,cu}` — `updateDirectionReady(step)`:
  - **AdamW path** — Adam moment update + bias correction + `u = m̂ · rsqrt(v̂ + ε_adam)`. **No** decoupled weight decay here (that lives in K3 via `λ_g θ`).
  - **RAdamW path** — same moment update; compute `ρ_t`; if `ρ_t > step.hp.radamw_rectify_threshold` (HP, **default `5.0f`**, **validated `>= 4.0f` at HpView load**) apply rectification factor `r_t`; else fall back to **`u = m̂`** (SGD-of-`m̂`). **Rectification site is fixed:** inside this stage, before writing to `u_device[g]`. K3 is unaware of optimizer kind. The threshold is read from `step.hp` — **no literal `5` may appear in the kernel** (covered by the test matrix below).
  - **Frozen group** — leave `step.u_device[g] = nullptr` (or a documented "skip" sentinel); MeasureK2 and ApplyK3 check `frozen` and skip.
  - **Output buffer** — `step.u_device[g]` is a per-group device buffer of size `N_g`. Implementation choice: reuse the group's existing `m_state` buffer **after** the moment update (since `m` is no longer needed for that step's `u`) OR allocate scratch. Document the choice; the test asserts K3 reads the **same** bytes K2 read.
- Selector: `step.hp.optimizer_kind ∈ {"adamw", "radamw"}` chooses path. Validated at `OptimizerHpView` build (Rule 20: throw on unknown kind).
- `Tests/optimizer_update_direction_test.cu` — full table from "Per-stage edge case test matrix" (AdamW direction, RAdamW rectified, RAdamW SGD fallback, frozen skip, bias correction at step=0, K1↔K2 byte parity, K1↔K3 byte parity).

**Exit criteria:** for a synthetic `(m, v, grad, step)`, `step.u_device[g]` matches CPU `double` reference within `1e-6`; RAdamW rectified path matches reference; K1↔K2 byte parity proven on at least one production-shaped tensor.

### P3 — Measurement (one PR)

- `MeasureK2.{hpp,cu}` — per-group reduction kernel using `UpdateDirection.cuh`. **Reads `step.u_device[g]` produced by P2's `updateDirectionReady`.** Skips groups where `frozen == true` OR `eligible == false` (records `N_g = 0` so aggregator skips). `double` partial sums per group. **Deterministic reduction path** for the sim's replay parity (single-block tree reduction when `N_g ≤ 65536`; two-pass with deterministic partition by `blockIdx` for larger).
- `AggregateGlobal.{hpp,cu}` — host weighted mean (uniform `w_g=1` default; `log_n` optional). `double` accumulators. `ε_upd` floor on `ms̄_upd` **before** `sqrt`. Returns `{theta_inst, rms_update, G_inc, sum_w}`.
- `Tests/optimizer_measure_k2_test.cu`, `Tests/optimizer_aggregate_global_test.cu`.

**Exit criteria:** K2 + AggregateGlobal callable end-to-end; mocked `lambda_g_device` and `u_device` produce `rms_update` matching CPU `double` reference within `1e-5` for a 12-layer-shaped synthetic param set.

### P4 — Host controller (one PR per concern, or one stacked PR)

- `ActivationProbe.{hpp,cu}` — hook installed in `AutogradTraining.cu` at the **canonical** point (between `crossAttentionRead` merge and `center_columns`) with `TensorDebugTag` populated at the call site. **Runtime tag validation in `markActivationRmsInstReady` throws on mismatch** (rule 6 + tensor identity).
- `ActivationFactor.{hpp,cu}` — `f_static` + rate clamp + regime bypass. `factor_prev == 0.f` ⇒ "invalid / no prev" cold start ⇒ `activation_scale_factor = f_static`.
- `SystemScale.{hpp,cu}` — `theta_eff` three-way max → `lerp` → `max(blend, τ·θ_eff, ε_ss)`. **Only place** these three lines exist.
- `RhoController.{hpp,cu}` — damped `ρ_pred` + max-ahead + optional `c_pred` backstop + `ρ_for_clamp = max(ρ_now, ρ_pred)`. **Only place** ρ math exists.
- `SoftCap.{hpp,cu}` — `G_inc == 0 || ρ_max == 0 ⇒ scale_u_global = 1.f`. Else `powf` cap. Never `> 1`.
- `PersistState.{hpp,cu}` — end-of-step EMA advance + `factor_prev = applied` + `rho_prev = rho_now`. Cold-start replace-instead-of-blend on first step (one-shot).
- One `Tests/optimizer_<concern>_test.cu` per Ready (matrix below).
- **Add `Tests/optimizer_pipeline_invariants_test.cu`** — sentinel-NaN ordering test (rule 6 enforcement). Lands here so it can poison K1's output and prove no later stage reads it before K1 has run.

**Exit criteria:** `runOptimizerStep` produces a consistent `scale_u_global` end-to-end on synthetic inputs; per-stage tests green; invariants test green; `Tests/optimizer_pipeline_sim_test.cu` runs without ApplyK3 (K3 still no-op stub) and verifies all bundles land at expected values for a 50-step synthetic trace.

### P5 — Apply + call sites (one PR)

- `ApplyK3.{hpp,cu}` — kernel using `UpdateDirection.cuh` and reading `step.u_device[g]` (the **same** bytes K2 read). Embedding-freeze gating: skip groups where `frozen == true`. Same `lambda_g_device` pointer as K2.
- `launchAdamWStep` / `launchRAdamWStep` (existing) become **thin wrappers** over `runOptimizerStep`:

```cpp
  void launchAdamWStep(std::vector<ParameterGroup>& groups, float lr, float wd, int step,
                       cudaStream_t stream, int embedding_freeze,
                       OptimizerClampState& clamp_state, float activation_rms_inst,
                       const OptimizerHpView& hp) {
      // 1. Mark frozen flags from embedding_freeze policy on each group for this step.
      // 2. buildLambdaGTableReady(...)
      // 3. OptimizerStepInputs step{...};
      //    runOptimizerStep(step);            // K1 → K2 → ... → K3 → PersistState
      // 4. clamp_state = step.clamp_state;    // PersistState wrote advances
  }
  

```

  Note: K1 is now **inside** `runOptimizerStep` — `launchAdamWStep` does not call any moment-update code itself. The old fused `AdamWKernel` is deleted in P7.

- `[Phase2_TrainingLoop.cu](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)` hands `ctx.optimizer.clamp_state`, `ctx.model->getTrainingState().activation_rms_inst`, and `OptimizerHpView` through the existing `launchAdamWStep` call site (line ~669). **No other Phase2 changes.**
- `[OptimizerCheckpoint.cu](resources/models/GRIM-text/training/OptimizerCheckpoint.cu)`: bump sidecar `version: uint32 = 2`; add 4 floats (`theta_ema`, `activation_rms_ema`, `activation_scale_factor_prev`, `rho_prev`) to the header. Loader supports v1 (cold-start the new fields) and v2 (restore exact). `Tests/optimizer_checkpoint_v2_test.cu` covers round-trip + v1→v2 upgrade.
- `OptimizerContext` (in `Phase1_Startup.hpp`) gains `OptimizerClampState clamp_state{};` field — single owner.

**Exit criteria:**

- **Scalar state** (clamp scalars + step counter + `m`/`v`): **bit-exact** save/load round-trip via `memcmp`.
- **Numerical parity vs old fused kernel** (NOT bit-exact): real training end-to-end with `optimizer_update_rho_max = 0` (clamp off) gives **numerical parity within tolerance** (relative `5e-4`, absolute `1e-7` per element). Differences are real and expected from: (a) split FMA path (K1 produces `u`, K3 fuses `α·scale·(u+λθ)` differently from the old single-pass kernel), (b) `double` reductions in K2 vs single-pass `float` math in fused kernel, (c) `fmaf` ordering. **Bit-exactness is not achievable, not promised, and not a regression.**
- With `optimizer_update_rho_max = 1.0f`, `scale_u_global ≤ 1` always.
- Sim_test passes the full assertion matrix.

### P6 — Tests + sim (one PR — can land alongside P5)

- All per-stage `Tests/optimizer_<concern>_test.cu` files (taxonomy table).
- `Tests/optimizer_pipeline_sim_test.cu` — full end-to-end sim (specified in **End-to-end simulation spec** below).
- CI runs all of them on every `Pipeline/` change.

### P7 — Cleanup (one PR)

- Delete the old fused `AdamWKernel` weight-decay path (now redundant with K1 + K3 split). Same for the RAdamW fused kernel.
- Delete any draft / commented per-group ρ scaffolding.
- Run grep guardrail (warning, not failure) over `Shared/Optimizers/**/*.cu` to surface any remaining `lambda * theta` arithmetic outside `UpdateDirection.cuh` and `LambdaGTable.cu`.
- Confirm `optimizer_update_clamp_eps`, `optimizer_update_soft_cap_start_ratio` are sourced **only** through `OptimizerHpView`.
- Update doc comment in `[HyperParameters_GPU.hpp](resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp)`: describe τ semantics with `@see Pipeline/SystemScale/SystemScale.hpp`; describe frozen-group exclusion with `@see Pipeline/Eligibility/Eligibility.hpp`; describe RAdamW rectification site **and the `radamw_rectify_threshold` HP semantics** (default `5.0f`, mathematical floor `4.0f`, warning band `[4.0, 5.0)`) with `@see Pipeline/UpdateDirection/UpdateDirection.cu`.
- Lock JSON schema (default block in **Hyperparameters** below).

## Hyperparameters (single config block — JSON shape + defaults)

All keys live under `optimizer.`* in `ai_config.json`. `OptimizerHpView` is a typed POD copy. Defaults are the spec's recommended starting points for **non-toy** training.

```json
{
  "optimizer": {
    "kind": "adamw",                                       // existing
    "weight_decay": 0.01,                                  // existing — drives λ_g

    "update_rho_max":                       1.0,           // 0 disables clamp ⇒ scale_u_global=1
    "update_soft_cap_start_ratio":          0.9,
    "update_soft_cap_power":                1.0,
    "update_clamp_eps":                     1.0e-12,

    "rho_rms_upd_eps":                      1.0e-16,       // ε_upd inside sqrt(ms̄_upd + ε_upd)
    "rho_slope_k":                          1.0,           // k_slope on Δ_ρ
    "rho_pred_damp":                        0.5,           // d ∈ (0,1]
    "rho_pred_max_ahead_ratio":             0.35,          // r_ahead — primary brake
    "rho_pred_cap_multiplier":              2.0,           // c_pred — wide backstop

    "system_scale_activation_blend":        1.0,           // alpha_blend ∈ [0,1]
    "system_scale_theta_rel_min":           0.88,          // τ ∈ (0,1] — DOWNWARD ceiling vs θ_eff
    "system_scale_eps":                     1.0e-8,        // ε_ss

    "weight_rms_ema_tau_steps":             300,           // τ_steps ⇒ β = exp(-1/τ_steps)
    "weight_rms_inst_blend_gamma":          1.0,           // γ
    "weight_rms_theta_floor":               1.0e-3,        // θ_floor

    "activation_rms_ema_tau_steps":         150,           // optional; default ≤ weight τ
    "activation_term_smoothing":            1.0,           // s_act — keep 1.0 by default
    "activation_factor_regime_bypass":      true,
    "activation_factor_regime_span_max":    3.0,
    "rho_activation_factor_min":            0.05,          // f_min
    "rho_activation_factor_max":            0.5,           // f_max
    "rho_activation_factor_step_min_ratio": 0.5,           // r_min
    "rho_activation_factor_step_max_ratio": 2.0,           // r_max
    "rho_activation_denom_eps":             1.0e-8,        // ε_act

    "global_rho_min_group_n":               4096,          // min_global_group_N
    "global_rms_group_weight":              "uniform",     // "uniform" | "log_n"

    "radamw_rectify_threshold":             5.0            // ρ_t threshold above which RAdamW rectification engages;
                                                           // below ⇒ SGD-of-m̂ fallback (Liu et al. 2019).
                                                           // VALIDATED >= 4.0f at HpView load: the rectification formula
                                                           //   r_t = sqrt((ρ_t-4)(ρ_t-2)·ρ∞ / ((ρ∞-4)(ρ∞-2)·ρ_t))
                                                           // requires ρ_t > 4 for the radicand to be non-negative.
                                                           // Default 5.0 = canonical paper value (1.0 safety margin above 4).
                                                           // Only used when "kind" == "radamw"; ignored for "adamw".
  }
}
```

**Forbidden:** raw `optimizer.weight_rms_ema_decay`. β is **derived** from `τ_steps` only.

**Validation at `OptimizerHpView` build (Rule 20: throw on violation):**

- `system_scale_theta_rel_min` ∈ `(0, 1]`
- `rho_pred_damp` ∈ `(0, 1]`
- `weight_rms_ema_tau_steps > 0`
- `global_rms_group_weight ∈ {"uniform", "log_n"}` — `"sqrt_n"` rejected
- `kind ∈ {"adamw", "radamw"}`
- `radamw_rectify_threshold >= 4.0f` — **mathematical lower bound** of the rectification formula; values in `[4.0, 5.0)` are accepted but logged as a warning (no safety margin above the singularity)

## Per-stage edge case test matrix (one *_test.cu per stage)

Each row is a separate `Tests/optimizer_<concern>_test.cu`. Tests are **unit** scope (no full model) so they can run in CI under a few seconds each.

### `optimizer_eligibility_test.cu`


| Case                 | Setup                           | Assert                                       |
| -------------------- | ------------------------------- | -------------------------------------------- |
| Bias excluded        | `ParamGroupType` = bias variant | `exclude_from_global_rho == true`            |
| Norm γ/β excluded    | `is_norm_affine == true`        | excluded                                     |
| Tiny weight excluded | weight tensor with `N_g = 4095` | excluded                                     |
| Boundary             | `N_g = 4096` exactly            | **included**                                 |
| Big weight included  | `N_g = 1<<20`                   | included                                     |
| Toy-only-biases      | every group is bias             | `G_inc == 0` (precondition for SoftCap test) |


### `optimizer_lambda_g_test.cu`


| Case                          | Setup                                                                                  | Assert                                                                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Helper parity                 | random `u, θ, λ`                                                                       | `updElement(u,λ,θ) == fma(λ,θ,u)` bit-exact; matches `double` reference within `1e-6`                                            |
| Zero-decay                    | `λ_g = 0`                                                                              | `updElement == u` bit-exact                                                                                                      |
| Table builder                 | groups with mixed `weight_decay_multiplier`                                            | each `lambdas[g] = global_wd × multiplier × upsilon` exactly                                                                     |
| `λ_g` is actually consumed    | swap `lambda_g_device[1]` from `0.01` to `100.0` between K2 calls                      | `rms_update` jumps; **catches a kernel that ignores `lambda_g_device` and hardcodes a constant**                                 |
| K2/K3 receive same `u_device` | spy on kernel args; sample one `(g, i)`                                                | K2's `(u + λθ)²` summand at `(g,i)` equals K3's effective applied increment squared at `(g,i)` (deterministic single-block path) |
| Grep guardrail                | scan `Shared/Optimizers/**/*.cu` excluding `UpdateDirection.cuh` and `LambdaGTable.cu` | warns (not fails) if patterns hit; emit findings as test-output annotation                                                       |


### `optimizer_update_direction_test.cu` (K1)


| Case | Setup | Assert |
|---|---|---|
| AdamW direction | known `m, v, step` | `u_device[i] == m̂_i · rsqrt(v̂_i + ε)` matches CPU `double` reference within `1e-6` |
| RAdamW rectified (default threshold) | `kind="radamw"`, `radamw_rectify_threshold=5.0f` (default), step large enough that `ρ_t > 5` | `u_device[i] == m̂_i · rsqrt(v̂_i + ε) · r_t`; `r_t` derived from spec formula |
| RAdamW SGD-of-`m̂` fallback (default threshold) | `kind="radamw"`, default threshold, `step = 1` (`ρ_t < 5`) | `u_device[i] == m̂_i` exactly (no `v` term, no `r_t`) |
| **RAdamW threshold actually consumed (regression)** | run two configs at the **same step** with the same `(m, v)`: `radamw_rectify_threshold=5.0f` vs `radamw_rectify_threshold=ρ_t + 0.1f` (just above current `ρ_t`) | first config rectifies; second config falls back to SGD-of-`m̂` ⇒ `u_device` differs measurably. **Catches a kernel that hardcodes `5.0f` and ignores the HP.** |
| **RAdamW threshold validation (below math bound)** | `radamw_rectify_threshold = 3.5f` | `OptimizerHpView` builder throws at config load (`>= 4.0f` required); kernel never reached |
| **RAdamW threshold warning band** | `radamw_rectify_threshold = 4.5f` (in `[4.0, 5.0)`) | builder accepts, emits warning to log, training proceeds; assert log line present and `r_t` computed correctly when `4 < ρ_t < 4.5` would now rectify (close-to-singularity but mathematically valid) |
| Frozen group | `frozen == true` | K1 does **not** write `u_device[g]`; downstream MeasureK2 must see the frozen flag and skip |
| Bias correction | `step = 0` (first iteration) | `m̂ = m / (1 - β1)`, `v̂ = v / (1 - β2)`; no div-by-zero |
| K1 ↔ K3 byte parity | dump `u_device[g]` after K1; spy on K3's read of that buffer | bit-exact match (K1 is the **sole writer**) |
| K1 ↔ K2 byte parity | same buffer | bit-exact match (no intermediate mutation) |
| **No literal threshold in kernel** (guardrail) | grep `Pipeline/UpdateDirection/UpdateDirection.cu` for `>\s*5(\.0f?)?\b` outside comments | warns if found; the threshold-actually-consumed regression is the real bite |


### `optimizer_measure_k2_test.cu`


| Case             | Setup                          | Assert                                                               |
| ---------------- | ------------------------------ | -------------------------------------------------------------------- |
| Single tensor    | `N=1024` random                | per-group `sum_upd_sq` matches CPU double reference within `1e-5`    |
| Multi group      | 4 groups, varying sizes        | no cross-talk; per-group sums independent                            |
| All-zero `upd`   | `u=0`, `λ=0`                   | `sum_upd_sq == 0` exactly                                            |
| Excluded group   | `eligible == false`            | host aggregator reads it as if `N_g = 0`                             |
| Embedding-freeze | embedding group at frozen step | K2 still computes (clamp must see them via `eligible`); K3 will skip |


### `optimizer_aggregate_global_test.cu`


| Case                 | Setup                                         | Assert                                                           |
| -------------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| Uniform `w_g=1`      | 2 included groups, `ms_1=4`, `ms_2=16`        | `ms̄ = 10` ⇒ `theta_inst = sqrt(10)`                             |
| Log weight           | same with `log_n`                             | `ms̄ = (4·log(1+N_1) + 16·log(1+N_2)) / (log(1+N_1)+log(1+N_2))` |
| **Default not `√N`** | request `sqrt_n`                              | rejected at JSON load (whitelist)                                |
| `G_inc == 0`         | all excluded                                  | `theta_inst = 0`, `rms_update = 0`, **no div-by-zero**           |
| `sum_w == 0`         | log mode w/ all `N_g = 0`                     | same as `G_inc == 0`                                             |
| `ε_upd` floor        | all-zero upd over `G_inc > 0`                 | `rms_update == sqrt(ε_upd) > 0`; downstream `ρ_now` finite       |
| λ regression         | fix `u, θ`; raise `λ_g` on one included group | `rms_update` strictly increases; with `λ=0` recovers `RMS(u)`    |


### `optimizer_activation_probe_test.cu`


| Case                           | Setup                                                     | Assert                                                   |
| ------------------------------ | --------------------------------------------------------- | -------------------------------------------------------- |
| K=1 degenerate                 | one microbatch                                            | `activation_rms_inst == RMS(layer_output_pre_center)`    |
| K=2, last wins                 | µbatch1 RMS=10, µbatch2 RMS=2 (final)                     | `activation_rms_inst == 2` (not 6, not 10)               |
| K=4                            | µbatches {5, 5, 5, 1} (final=1)                           | `== 1`                                                   |
| Center-on guard                | `center_encoder_residuals=true` with non-zero column mean | hook-time RMS **>** post-center RMS                      |
| Forbidden source               | request RMS of LM head input or post-center               | function refuses (compile-time tensor identity check)    |
| `is_final_microbatch == false` | call N times                                              | `activation_rms_inst` **unchanged** until the final call |


### `optimizer_activation_factor_test.cu`


| Case              | Setup                                                       | Assert                                   |
| ----------------- | ----------------------------------------------------------- | ---------------------------------------- |
| Static low        | ratio < `f_min`                                             | `factor == f_min`                        |
| Static high       | ratio > `f_max`                                             | `factor == f_max`                        |
| First step        | `factor_prev == 0`                                          | `factor == f_static`                     |
| Rate clamp up     | `f_prev=0.1`, `f_static=0.4`, bypass off                    | applied = `0.2` (×2 cap)                 |
| Rate clamp down   | `f_prev=0.2`, `f_static=0.05` (at `f_min`)                  | applied = `0.1` (×0.5 floor)             |
| Regime bypass on  | `f_prev=0.1`, `f_static=0.35`, span=3.5, max=3, bypass=true | applied = `0.35`, `regime_bypassed=true` |
| Regime bypass off | same with bypass=false                                      | applied = `0.2`                          |
| Span ≤ max        | `span=2.5`, max=3                                           | rate clamp wins                          |


### `optimizer_system_scale_test.cu`


| Case                | Setup                                  | Assert                                                          |
| ------------------- | -------------------------------------- | --------------------------------------------------------------- |
| τ=1 uplift-only     | `θ_eff=1, act_term=0.5, α=1, τ=1`      | `system_scale = 1`                                              |
| τ=0.88              | same with `τ=0.88`                     | `system_scale = max(0.5, 0.88, ε) = 0.88`                       |
| τ=0.9               | `θ_eff=1, act_term=0.5, α=1, τ=0.9`    | `blend=0.5`, `system_scale=0.9`                                 |
| α=0                 | any `act_term`                         | `blend=θ_eff`, `system_scale=max(θ_eff, τ·θ_eff, ε_ss) = θ_eff` |
| activation > θ_eff  | `act_term=2, θ_eff=1, α=1`             | `blend=2`, `system_scale=2` (uplift)                            |
| `ε_ss` floor        | force `blend=ε_ss/2`, `τ·θ_eff=ε_ss/2` | `system_scale=ε_ss`                                             |
| Three-way max θ_eff | `θ_inst=2·θ_ema`, `γ=1`                | `theta_eff = θ_inst`                                            |


### `optimizer_rho_controller_test.cu`


| Case                | Setup                                                     | Assert                                                                       |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| First step          | `rho_prev=0`                                              | `Δ=0`, `rho_pred=rho_now`, `rho_for_clamp=rho_now`                           |
| Damped trend        | `rho_now=2, rho_prev=1, k=1, d=0.5`                       | `rho_pred = 2 + 0.5·1 = 2.5`, then capped by `(1+r_ahead)·rho_now=2.7` ⇒ 2.5 |
| Spike capped        | `rho_now=10, rho_prev=1, d=1, r_ahead=0.35`               | undamped would be 19; capped to `13.5`                                       |
| Falling             | `rho_now=1, rho_prev=2`                                   | `rho_pred = 0.5`, `rho_for_clamp = 1` (max wins)                             |
| `rho_now=0`         | any `rho_prev`                                            | `rho_pred=0`, `rho_for_clamp=0`                                              |
| Wide `c_pred` binds | `r_ahead=10`, `c_pred=2`, `rho_now=1`, large damped trend | `rho_pred = min(damped, 2)`                                                  |
| LR doubles          | `η×2` at fixed `rms`, `system_scale`                      | `rho_now` doubles                                                            |


### `optimizer_soft_cap_test.cu`


| Case             | Setup                                                     | Assert                                                         |
| ---------------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| Clamp off        | `ρ_max=0`                                                 | `scale_u_global = 1`                                           |
| `G_inc==0`       | any ρ                                                     | `scale_u_global = 1` (early bypass)                            |
| Below soft start | `rho_for_clamp = 0.8·ρ_max`                               | `scale_u_global = 1`                                           |
| At boundary      | `rho_for_clamp = ρ_soft_start`                            | `scale_u_global = 1` (continuous)                              |
| Above            | `rho_for_clamp = 2·ρ_max`, `p=1`                          | `scale_u_global = 0.45/(2·ρ_max+ε_clamp)` ≈ `0.45·1/(2·ρ_max)` |
| Never > 1        | sweep `rho_for_clamp ∈ [0, 100·ρ_max]`, `p ∈ {0.5, 1, 2}` | `scale_u_global ≤ 1` always                                    |
| Continuity       | sweep around `ρ_soft_start ± ε`                           | `                                                              |


### `optimizer_apply_k3_test.cu`


| Case                                      | Setup                                                                    | Assert                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Same `scale_u` to all stepped groups      | mock kernel arg spy (or post-step norm)                                  | every NOT-frozen group's apply observes the same `scale_u_global`                                                                                                                                                                                                                                                                                                                                               |
| `λ=0` group                               | apply                                                                    | `θ -= η·α·scale_u·u` exactly                                                                                                                                                                                                                                                                                                                                                                                    |
| Embedding freeze (apply side)             | `step >= freeze_after`                                                   | `EMBEDDING` group skipped; `LM_HEAD` named `embedding_lm_head_tied` skipped                                                                                                                                                                                                                                                                                                                                     |
| Embedding freeze (measure side)           | same                                                                     | the frozen group is **also** absent from `per_group_sums` (or its `frozen=true` ⇒ aggregator skips); `G_inc` excludes it                                                                                                                                                                                                                                                                                        |
| Frozen-group ρ contamination regression   | freeze embedding at step 5; spike fake `u` on frozen embedding at step 6 | `ρ_now` **does not move** (frozen group has no `u`, no contribution)                                                                                                                                                                                                                                                                                                                                            |
| Numerical parity vs fused (NOT bit-exact) | `ρ_max=0`, `scale_u_global=1`, identical `λ=wd`                          | weight delta matches old `launchAdamWKernel` within **relative tolerance `5e-4`** AND **absolute floor `1e-7`** per element. Differences expected from: (a) split FMA path (K1 produces `u`, K3 fuses `α·scale·(u+λθ)` differently), (b) `double` reductions in K2 vs single-pass `float` math in fused kernel, (c) `fmaf` ordering. **Bit-exact parity is not a goal** — see "Replay parity" note in sim spec. |
| Same K2 helper                            | sample one index `i` (deterministic single-block kernel)                 | K2 partial contribution at `i` equals `updElement²` from K3 path bit-exact (single-thread reduction over one element ⇒ no order ambiguity)                                                                                                                                                                                                                                                                      |


### `optimizer_pipeline_invariants_test.cu` (rule 6 enforcement)


| Case                    | Setup                                                                                                                             | Assert                                                                                                                                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Sentinel-NaN ordering   | poison every yet-unauthored `step.*` field with `NaN` (floats), `-1` (ints), `nullptr` (pointers); run each Ready in driver order | after each Ready, **only** that Ready's owned bundle is non-poisoned; any field a Ready accidentally reads from a downstream stage produces `NaN` ⇒ caught immediately. Final `step.scale_u_global` is finite and `≤ 1`. |
| Reverse order rejection | call `softCapReady(step)` before `rhoControllerReady(step)`                                                                       | `step.rho_for_clamp == NaN` ⇒ `step.scale_u_global == NaN` ⇒ test fails (this is what we want — proves the invariant has bite)                                                                                           |
| Bundle write singleness | run driver; instrument every bundle field with a write counter                                                                    | every authored field is written **exactly once** per driver call                                                                                                                                                         |


### `optimizer_persist_state_test.cu`


| Case                | Setup                              | Assert                                                             |
| ------------------- | ---------------------------------- | ------------------------------------------------------------------ |
| Cold start θ_ema    | `theta_ema=0` going in             | after step, `theta_ema = theta_inst` (replace-not-blend, one-shot) |
| Cold start act_ema  | same for `activation_rms_ema`      | replace                                                            |
| Warm advance        | `θ_ema=1, θ_inst=2, β=0.99`        | `θ_ema = 0.99 + 0.02 = 1.01`                                       |
| `factor_prev` write | applied factor `0.18`              | `clamp_state.activation_scale_factor_prev = 0.18`                  |
| `rho_prev` write    | `rho_now=0.7`                      | `clamp_state.rho_prev = 0.7`                                       |
| Skipped step        | accumulation continuing (no apply) | `PersistState` not called ⇒ no advance (caller invariant)          |
| β derived           | `τ_steps=200`                      | `β = exp(-1/200)`; raw decay JSON key rejected                     |


### `optimizer_checkpoint_v2_test.cu`


| Case            | Setup                | Assert                                           |
| --------------- | -------------------- | ------------------------------------------------ |
| v2 round-trip   | save → load          | all 4 clamp scalars bit-exact, plus existing m/v |
| v1 → v2 upgrade | load v1 sidecar      | clamp scalars cold-start (zeros), no exception   |
| Mismatch        | `num_groups` differs | throws (Rule 20)                                 |


## End-to-end simulation spec — `Tests/optimizer_pipeline_sim_test.cu`

This is the integration check that prevents regressions before the optimizer runs on the real model. **The model is far from a toy**, so the sim shape mirrors realistic transformer training.

### Fixture

- **Architecture:** 12 encoder layers, `d_model = 512`, `num_heads = 8`, `d_ff = 2048`, vocab = 32k. (Or pull `architecture` defaults from `HyperParameters_GPU.hpp` at runtime — `DEFAULT_D_MODEL` / `DEFAULT_NUM_LAYERS`.)
- **Parameter groups (built via `buildParameterGroups()` on a real `LanguageModel`):** ~70 groups including biases (excluded), norm γ/β (excluded), embedding (`N=32k·512`, included), LM head (tied or untied; included), per-layer attn QKV / out / FFN W1/W2 (included), per-layer norm γ/β (excluded). `G_inc ≈ 50`.
- **Gradient accumulation:** `accum_steps = 4`.
- **Steps simulated:** **50**.
- **Stream:** real `cudaStream_t` from a fresh `StreamController`.
- **No real loss** — the sim feeds **synthetic** gradients (next bullet) so we can inject specific dynamics deterministically.

### Synthetic gradient schedule (deterministic; `seed = 0xA1B2C3D4`)

For each step `s ∈ [0, 50)`:

- Per included group: `grad ~ N(0, σ_s)` with `σ_s = 0.01` baseline.
- Per excluded group: `grad ~ N(0, 100·σ_s)` — **must not** affect `ρ_now` (eligibility regression).
- `m`/`v` advanced via real `AdamWKernel` (so `u` is realistic, not raw grad).
- `activation_rms_inst` written by a **mock** `markActivationRmsInstReady` that sweeps a deterministic curve:
  - Steps `0..24`: `0.8 + 0.05·sin(s/3)` (mild, EMA-tracking range).
  - **Step 25 (regime shift):** instantaneous jump to `3.0` and held for 5 steps (`activation_rms_inst ∈ [2.5, 3.5]` for `s ∈ [25, 30)`).
  - Steps `30..49`: decays back to `1.2`.

### Injection schedule

- **Step 25 — regime shift:** `activation_rms_inst` jumps ~4×. Triggers `span > regime_span_max` ⇒ `regime_bypassed == true` for at least one of steps `[25, 26]`. Assert `activation_scale_factor` moves in **one step** to `f_static` (not trapped at `f_prev × r_max`).
- **Step 30 — ρ spike:** boost a single included group's `u` by 10× for one step (e.g. multiply `m` post-K1). Assert `ρ_now` spikes; `ρ_pred` is **damped** (within `(1+r_ahead) · ρ_now` of the step's `ρ_now`); `scale_u_global < 1` engages and `< 0.5`; `scale_u_global` returns to `1.0` within 3 steps.
- **Step 25 — checkpoint:** call `saveOptimizerState` immediately after `PersistState` of step 25. At step 50 end, restore + replay steps 26..50 from the checkpoint and verify replay parity. **Two distinct parity classes (do not conflate):**
  - **Scalar state load/save (BIT-EXACT, required):** the 4 `OptimizerClampState` floats + `optimizer_step` int + `m`/`v` buffers must round-trip through the sidecar **bit-exact** (`memcmp`, no tolerance). Anything less is a checkpoint bug, not a numerical artifact.
  - **Replay of dynamic quantities (TOLERANCE, expected):** per-step `(theta_inst, rms_update, system_scale, rho_now, scale_u_global)` after replay match the original within **relative tolerance `1e-3`** AND **absolute floor `1e-6`**. Drift is real and expected from: GPU reduction order (block scheduling), atomic ordering, FMA contraction differences across runs even on the same device. **To minimize drift, the sim configures K2 with a deterministic reduction path** (single-block tree reduction, fixed `block_size=256`, fixed `grid=1` per group when `N_g ≤ 65536`; for larger groups, two-pass with deterministic partition by `blockIdx`). This keeps replay drift small, but does not promise bit-exactness.

### Assertions (every step, in order they appear in the driver)

1. **MeasureK2 ↔ K3 parity at one sampled index** — at one random `(group, i)` per step, `updElement(u_i, λ_g, θ_i)` recomputed on host equals K3's effective `(θ_old - θ_new) / (η·α_g·scale_u_global)` within `1e-5`.
2. **AggregateGlobal ignores excluded groups** — boost `u` 100× on excluded biases ⇒ `Δ rms_update == 0`.
3. `**G_inc == 0` micro-test** (separate fixture, run alongside) — single-step model with biases-only ⇒ `scale_u_global == 1`, no NaN, EMAs unchanged for that step.
4. `**activation_rms_inst` is final-µbatch only** — assert `µbatch[0..K-2]` writes to a scratch field; `TrainingState::activation_rms_inst` only updated on µbatch `K-1`.
5. **EMA advances once per optimizer step** — `θ_ema` / `activation_rms_ema` change at most 50 times across 200 microbatches.
6. **Soft cap continuity** — sweep `ρ_for_clamp` across the soft-start boundary at step boundaries (already covered in unit; sim checks no live discontinuity).
7. `**scale_u_global ≤ 1` always.**
8. `**ρ_pred` damping at the spike** — at step 30, undamped `(ρ_now + Δ)` would exceed `1.5·ρ_now`; assert `ρ_pred ≤ 1.35·ρ_now` (default `r_ahead`).
9. **Regime bypass fired** — `regime_bypassed[25] == true`.
10. **Checkpoint replay parity** —
  - Scalar state (clamp scalars, optimizer_step, `m`/`v`): **bit-exact** via `memcmp` after restore.
    - Replayed dynamic quantities: relative `1e-3` / absolute `1e-6` (deterministic reduction path active).
11. **No NaN / Inf** in any `step.*` field across all 50 steps (also enforced by `optimizer_pipeline_invariants_test.cu`).
12. **Total weight RMS doesn't collapse** — `theta_inst[49] > 0.5 · theta_inst[0]` (sanity that the controller didn't starve the network into zero).
13. **Embedding freeze (locked policy):** at `embedding_freeze_after_step = 40`, assert (a) `EMBEDDING` group `θ` is **unchanged** from step 40 onward; (b) frozen group is **absent from `per_group_sums`** at every step ≥ 40; (c) `G_inc[s≥40] = G_inc[s=39] - 1`; (d) injecting a fake spike into the frozen group's hypothetical `u` (e.g. by writing it into a sidechannel) **does not move `ρ_now`**.

### How the sim runs

```cpp
// Tests/optimizer_pipeline_sim_test.cu
TEST(OptimizerPipelineSim, NonToyModel_FiftyStep_Deterministic) {
    auto fixture = buildSimFixture(/*num_layers=*/12, /*d_model=*/512, /*K=*/4, /*seed=*/0xA1B2C3D4);
    SimRecorder recorder;

    for (int s = 0; s < 50; ++s) {
        for (int micro = 0; micro < fixture.K; ++micro) {
            applySyntheticBackward(fixture, s, micro);
            markActivationRmsInstReady(fixture.training_state,
                                       syntheticActivationRMS(s, micro),
                                       /*is_final_microbatch=*/(micro == fixture.K - 1));
        }

        auto step = makeStepInputs(fixture, s);                  // pulls clamp_state, eta from sched
        runOptimizerStep(step);                                  // THE driver
        recorder.capture(s, step);

        if (s == 25) saveOptimizerState(fixture.ctx, "/tmp/sim_step25.opt");
    }

    recorder.assertAll();                                        // assertions 1..13

    // Replay parity: restore @ step 25, replay 26..49, compare
    auto fixture2 = buildSimFixture(...);
    loadOptimizerState(fixture2.ctx, "/tmp/sim_step25.opt");
    SimRecorder recorder2;
    for (int s = 26; s < 50; ++s) { /* same as above */ }
    recorder2.assertEqualTo(recorder, /*from_step=*/26, /*tol=*/1e-5f);
}
```

`SimRecorder::assertAll()` is the single place the 13 assertions are codified — adding a 14th is one diff.

## Files to create / modify (concrete list)

**New (P0–P4):**

```
resources/models/GRIM-text/Shared/Optimizers/Pipeline/
├── OptimizerStepInputs.hpp           (incl. TensorDebugTag, OptimizerClampState, OptimizerHpView)
├── OptimizerStepPipeline.hpp
├── OptimizerStepPipeline.cu          (the 9-call driver)
├── Eligibility/Eligibility.hpp
├── Eligibility/Eligibility.cu
├── UpdateDirection/UpdateDirection.cuh   (single __device__ updElement helper)
├── UpdateDirection/UpdateDirection.hpp   ← NEW (K1 host-side launcher)
├── UpdateDirection/UpdateDirection.cu    ← NEW (Adam moment update + RAdamW rectification site; writes step.u_device[g])
├── UpdateDirection/LambdaGTable.hpp
├── UpdateDirection/LambdaGTable.cu
├── MeasureK2/MeasureK2.hpp
├── MeasureK2/MeasureK2.cu
├── AggregateGlobal/AggregateGlobal.hpp
├── AggregateGlobal/AggregateGlobal.cu
├── ActivationProbe/ActivationProbe.hpp
├── ActivationProbe/ActivationProbe.cu    (runtime TensorDebugTag validation; throws on mismatch)
├── ActivationFactor/ActivationFactor.hpp
├── ActivationFactor/ActivationFactor.cu
├── SystemScale/SystemScale.hpp
├── SystemScale/SystemScale.cu
├── RhoController/RhoController.hpp
├── RhoController/RhoController.cu
├── SoftCap/SoftCap.hpp
├── SoftCap/SoftCap.cu
├── ApplyK3/ApplyK3.hpp
├── ApplyK3/ApplyK3.cu                    (reads SAME step.u_device[g] bytes K2 read)
├── PersistState/PersistState.hpp
└── PersistState/PersistState.cu
```

**New (P6 — tests):**

```
resources/models/GRIM-text/Tests/
├── optimizer_eligibility_test.cu
├── optimizer_lambda_g_test.cu
├── optimizer_update_direction_test.cu      ← NEW (K1: AdamW + RAdamW + frozen + byte parity)
├── optimizer_measure_k2_test.cu
├── optimizer_aggregate_global_test.cu
├── optimizer_activation_probe_test.cu
├── optimizer_activation_factor_test.cu
├── optimizer_system_scale_test.cu
├── optimizer_rho_controller_test.cu
├── optimizer_soft_cap_test.cu
├── optimizer_apply_k3_test.cu
├── optimizer_persist_state_test.cu
├── optimizer_checkpoint_v2_test.cu
├── optimizer_pipeline_invariants_test.cu   ← NEW (sentinel-NaN ordering; rule 6)
└── optimizer_pipeline_sim_test.cu
```

**Modified (P1, P4, P5):**


| File                                                                                                                              | Change                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[Shared/TensorContract/TensorContract_GPU.hpp](resources/models/GRIM-text/Shared/TensorContract/TensorContract_GPU.hpp)`         | Add `bool is_bias=false; bool is_norm_affine=false; bool exclude_from_global_rho=false; bool frozen=false;` to `ParameterGroup` (`frozen` is set per-step, not at construction)                                                                                                                  |
| `[Shared/TrainingState/TrainingState_GPU.hpp](resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp)`             | Add `float activation_rms_inst = 0.0f;` and `int num_encoder_layers = 0;` (used by ActivationProbe tag check)                                                                                                                                                                                    |
| `[Shared/Optimizers/OptimizerState.hpp](resources/models/GRIM-text/Shared/Optimizers/OptimizerState.hpp)`                         | (no change — step counter remains here)                                                                                                                                                                                                                                                          |
| `[Shared/Optimizers/AdamW/AdamW_Kernal_GPU.{hpp,cu}](resources/models/GRIM-text/Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu)`     | `launchAdamWStep` becomes thin wrapper over `runOptimizerStep` (P5). Old fused `AdamWKernel` deleted in P7.                                                                                                                                                                                      |
| `[Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.{hpp,cu}](resources/models/GRIM-text/Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.cu)` | Same — wrapper. **Rectification logic moves into `Pipeline/UpdateDirection/UpdateDirection.cu` (the K1 stage)** — not in K3, not in this kernel.                                                                                                                                                 |
| `[Shared/HyperParameters/HyperParameters_GPU.hpp](resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp)`     | Add `OptimizerHpView` POD + builder; document τ semantics, frozen-group exclusion policy, RAdamW rectification site                                                                                                                                                                              |
| `[training/Phases/Phase1_Startup.hpp](resources/models/GRIM-text/training/Phases/Phase1_Startup.hpp)`                             | `OptimizerContext` gains `OptimizerClampState clamp_state{};` + `OptimizerHpView hp_view{};`                                                                                                                                                                                                     |
| `[training/Phases/Startup/Optimizer/](resources/models/GRIM-text/training/Phases/Startup)`                                        | **New** `OptimizerEligibility.{hpp,cu}` + `OptimizerHpViewReady.{hpp,cu}`; called from `executePhase1` after `ModelAllocated`                                                                                                                                                                    |
| `[training/Phases/Phase2_TrainingLoop.cu](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)`                     | Pass `clamp_state`, `activation_rms_inst`, `hp_view` through the existing `launchAdamWStep` / `launchRAdamWStep` call site; mark `frozen` per group per step from existing `embedding_freeze_after_step` policy                                                                                  |
| `[training/Autograd/AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)`                       | Insert `markActivationRmsInstReady(ts, layer_output, tag)` between `crossAttentionRead` merge and `center_columns` on the **last** encoder layer; tag fields populated from local context (`layer_id`, `is_microbatch_final`, `centered=false`, `stage_name="encoder_layer_output_post_inject"`) |
| `[training/OptimizerCheckpoint.{hpp,cu}](resources/models/GRIM-text/training/OptimizerCheckpoint.cu)`                             | Bump sidecar version 1 → 2; serialize `OptimizerClampState`; v1 loader cold-starts the new fields. **Bit-exact round-trip** required (memcmp test).                                                                                                                                              |


## Out of scope (non-goals)

- Per-class scales (per-bias / per-norm) — default spec applies one `scale_u_global` to all stepped groups (excluded groups still apply, only excluded from **measuring** ρ).
- Per-group ρ as a primary signal — **global is the spec**; per-group remains optional telemetry only.
- Conservative ρ with `η_base × max_g α_g` — documented edge for future, not P0–P6.
- Replacing K1 (Adam moment update) — K1 stays where it is; this plan only splits the **measure / decide / apply** halves.

## Risk register


| Risk                                                                                 | Mitigation                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K1 / K2 / K3 disagree on what `u` is                                                 | K1 is the **sole writer** of `step.u_device[g]`; K2 and K3 read the same buffer; `optimizer_update_direction_test.cu` proves K1↔K2 and K1↔K3 byte parity on production-shaped tensors. RAdamW rectification is **inside K1**, not duplicated in K3.                                                     |
| RAdamW rectification threshold becomes hidden magic                                  | `optimizer_radamw_rectify_threshold` HP exposed in `OptimizerHpView` (default `5.0f`, validated `>= 4.0f` at HpView load with warning band `[4.0, 5.0)`). "Threshold actually consumed" regression test catches a kernel that ignores the HP. "No literal threshold" grep guardrail surfaces accidental hardcoding.                |
| K2 / K3 drift via duplicated `λ θ` math                                              | Single `UpdateDirection.cuh::updElement`; **kernel API shape** (typed `lambda_g_device` parameter; no `weight_decay` constant in K2/K3) is the real guarantee. CI grep is a guardrail, not a guarantee. The "λ_g actually consumed" regression test catches a kernel that hardcodes a constant.         |
| Activation hook samples wrong tensor (post-center, mid-stack, LM-head input)         | **Runtime** `TensorDebugTag` validation in `markActivationRmsInstReady` throws on `layer_id`, `stage_name`, `centered` mismatch. Tag is set at **one** site in `AutogradTraining.cu`. Compile-time identity is **not** claimed.                                                                         |
| Frozen group contaminates global ρ (free pass when embedding stops absorbing signal) | Locked policy: frozen groups are skipped by K1, K2, **and** Aggregate. Sim assertion 13 + `frozen-group ρ contamination regression` test verify. The opposite policy (keep frozen with `u=0`) is rejected and tested-against.                                                                           |
| EMA cold start NaN on first step                                                     | Replace-not-blend on first step; `ε_act` in `raw_factor` denominator; `θ_floor` in `theta_eff`; sentinel-NaN invariants test catches accidental reads of cold-start fields.                                                                                                                             |
| `ρ_pred` over-damps the response, `scale_u_global` lags reality                      | Sim step 30 spike is the canonical regression; `r_ahead` and `c_pred` defaults sized to engage in 1–2 steps                                                                                                                                                                                             |
| Checkpoint compatibility break                                                       | Sidecar version `2` with v1 cold-start path + `optimizer_checkpoint_v2_test.cu`. **Scalar state is bit-exact** via `memcmp`; **replayed dynamic quantities** are tolerance-checked on a deterministic reduction path (relative `1e-3`, absolute `1e-6`).                                                |
| Numerical parity with old fused kernel sets unrealistic expectations                 | P5 exit criterion explicitly says **"numerical parity within tolerance, not bit-exact"** with documented reasons (split FMA, `double` reductions, ordering). A drift outside tolerance is the regression; bit-exact match would be a **suspicious** result (suggests K2 collapsed into the fused path). |
| Stage reads downstream-authored field (out-of-order data flow)                       | Rule 6 + `optimizer_pipeline_invariants_test.cu` (sentinel-NaN test). Each Ready's `.hpp` lists `Reads:` and `Authors:` bundles for review-time check.                                                                                                                                                  |
| Hyperparameter sprawl                                                               | Single JSON block above; `OptimizerHpView` is the **only** read path; no Ready function reads `ctx.config`                                                                                                                                                                                              |


## Notes on running the sim

- The sim is a **CUDA test** (`Tests/optimizer_pipeline_sim_test.cu`) — runs on real device, not host-mock. Targets a single GPU; no multi-stream branching.
- Builds via the existing `Shared/**/*.cu` glob; `Tests/*.cu` glob if present (otherwise add `Tests/optimizer_*_test.cu` to the test target).
- Sim must complete in **< 30 s** on a single A100 / H100; if longer, reduce `d_model` to 256 (still non-toy: 12 layers × multi-head attention × 4-step accum is the **shape** that matters, not the absolute parameter count).
- Sim is gated on every PR that touches `Pipeline/`. The 13 assertions are the contract.

## Cleanup (P7)

Remove any draft text suggesting per-group ρ as the primary design. **Global is the spec.** Sweep `Shared/Optimizers/**/*.cu` for any remaining `lambda * theta` arithmetic outside `UpdateDirection.cuh` and `LambdaGTable.cu` (the grep guardrail surfaces candidates as test annotations; PR review removes them). The actual guarantee that K2/K3 cannot diverge is the kernel API shape, not the grep — see § **λ_g and `u` consistency**.