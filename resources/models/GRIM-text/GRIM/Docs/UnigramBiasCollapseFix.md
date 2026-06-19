# Unigram LM-Head Bias — Representation-Collapse (ρ buildup) Fix

Status: **implemented & validated** (telemetry `1781895428825949890`).
Supersedes the analysis in `ModeCollapseAblationInvestigation.md` (which was wrong).

---

## 1. Symptom

During training the residual stream collapses directionally: the cross-token
cosine similarity ρ of the final hidden state climbs from ~0.05 toward ~0.9–1.0,
i.e. all token representations rotate to point the same way (anisotropy / mode
collapse). The telemetry streams that track this are `rho_final`, `rho_atom_only`,
`rho_nonatom_only` (see `*_rho_hidden.png`) and the decomposition in
`*_rho_raw.png`.

This happened with **no** collapse-intervention active — no residual centering,
no LayerScale, no `lm_head_centering`, no `project_out_pc1`.

## 2. Ablation evidence (what we proved)

Using the sublayer ablation toggles in `Layers/Encoding/AblationFlags.hpp`
(`kZeroFfnResidual`, `kZeroAttnV`) we ran the three combinations:

| Run | FFN contribution | Attn value (V) | Outcome |
|-----|------------------|----------------|---------|
| `FFN0ATTNV0` | zeroed | zeroed | **No collapse** — body is an exact identity (`output == embeddings`); every body parameter gets zero gradient. ρ is just the static embedding geometry. |
| `FFN1ATTNV0` | active | zeroed | **Collapses.** `rho_atom_only` pins at ~0.98. |
| `FFN0ATTNV1` | zeroed | active | **Collapses.** ρ peaks ~0.9 then partially recovers. |

Key conclusion: collapse needs a **token-conditioned writer** on the residual
stream. With both writers off the body is inert, so there is nothing to inject a
shared direction. **Either** writer alone is sufficient to collapse — so the
cause is NOT specific to attention's causal prefix-averaging (the old
hypothesis); the FFN-only stack collapses with attention contributing nothing.

### The `rho_raw` decomposition — two distinct engines

`ρ = avg|dot(h_i,h_j)| / avg(‖h_i‖‖h_j‖·d)`. Reading the alignment **numerator**
and the dot/norm ratio at **step 0** (before meaningful training):

| Run | numerator @0 | ρ @0 | interpretation |
|-----|-------------:|-----:|----------------|
| `FFN0ATTNV0` (inert) | ~40 | ~0.06 | embedding geometry baseline |
| `FFN1ATTNV0` (FFN) | ~25 | ~0.03 | starts orthogonal → alignment is **learned** |
| `FFN0ATTNV1` (AttnV) | ~330 | ~0.43 | starts **half-collapsed** before any gradient step |

This is the smoking gun for two separate mechanisms reaching the same endpoint:

- **Mechanism A — FFN (learned).** A single shared MLP under shared (unigram)
  loss pressure learns an effectively rank-1 output: it emits the
  marginal-logit direction `u` for every token. RMSNorm never subtracts a
  cross-token direction, so `u` compounds across depth and is reinforced by
  LM-head feedback. Numerator climbs 25 → ~580 and **persists** (the direction
  is baked into `W2`).
- **Mechanism B — attention (structural).** The softmax convex combination
  (`Σ_j a_ij = 1`) contracts toward the value mean `V̄`. At init, near-uniform
  attention makes `attn_out_i ≈ V̄` for every query → a shared vector injected
  into every residual. This is present at t=0 (ρ≈0.43). It then **heals** as Q/K
  learn content routing (numerator overshoots down to ~180, below its own init).

## 3. Root cause

Both engines serve the **same unmet need**: the model must represent the unigram
marginal `p(v)` — predict frequent tokens by default, in every context. With
untied weights and **no LM-head output bias**, the logit is `W_lm[v]·h`, so for
the marginal to appear regardless of context, `h` must carry a component that is
the **same for every token** — a shared common-mode direction. That is exactly
the unigram-frequency direction tracked by the Issue-#150 detector
(`unigram_dir_cos_*`, telemetry streams 45–46).

A previously-tried `use_bias=true` did **not** fix this for two reasons:

1. The LM-head bias was allocated as **zeros** (`Tensor::zeros(...)` in
   `initializeLmHeadParameterTensors`), so at init it held no marginal. Adam had
   to *learn* it up to `log p(v)`, and by then the writers had already built the
   common mode.
2. `use_bias` is global — it also turns on `b_o` (attention) and `b2` (FFN),
   which are added identically to every token and are therefore themselves
   shared-direction injectors that **feed** the collapse.

## 4. The fix

A **dedicated LM-head output bias, initialized to `log p(v)`**, decoupled from
the global `use_bias` flag. This houses the unigram marginal in a parameter that
exists at step 0, so neither attention nor the FFN is rewarded for injecting it
into the residual. Mechanism A loses its driver; mechanism B loses the reason to
persist.

### Config

`ai_config.json` → `training.config`:

```jsonc
"use_bias": false,            // keeps b_o / b2 OFF (they are common-mode injectors)
"lm_head_unigram_bias": true  // dedicated log p(v) LM-head bias, ON
```

### Marginal source

The empirical unigram marginal is computed from the **training targets** (what
the LM head predicts) over `ctx.data.train_seqs[].targets` at startup, with
Laplace add-one smoothing so unseen tokens get a finite floor:

```
bias[v] = log( (count_v + 1) / (total_targets + V) )
```

> Note: we deliberately do **not** use the dormant
> `grim_embedding_weights.fbs` `VocabMetadata.token_frequencies` field — it is
> never written or read anywhere in the pipeline. The training-target marginal
> is the true, actually-used distribution.

### Lifecycle

- **Allocation (fresh weights live in init):**
  `initializeLmHeadParameterTensors` (`ParameterGroupRegistration.cu`) allocates
  the trainable `lm_head.bias` when `use_bias || unigram_bias` (zeros).
- **Population (access via ParameterRegistry):** `populateUnigramOutputBiases` in
  `Phase1_Startup.cu`, called right after `CheckpointLoaded`, fills the bias with
  `log p(v)` via `parameter_registry.requireLmHeadParameters(...)` (and the MTP
  heads via `parameter_registry.mtpHeadParameterTensors()` — see §4a). It runs
  **only on fresh init** (`loaded_checkpoint_path` empty); on resume the trained
  bias is kept.
- **Optimizer:** registered in the `LM_HEAD` group, gated on
  `use_bias || lm_head_unigram_bias`, so the bias trains.
- **Forward:** `forwardLmHead` (`lm_head_GPU.cu`) applies the bias whenever the
  bias tensor exists (gated on `lm_bias.data`, not `use_bias`).
- **Serialization:** save/load of the bias is **presence-driven**
  (`Serialization_save.cu`, `Serialization_load.cu`, `grim_model_serialization.cu`
  via `expect_bias = bias.data != nullptr`), so a trained `log p(v)` bias is
  never silently dropped on checkpoint round-trips.

## 4a. MTP auxiliary heads share the same prior

The Multi-Token-Prediction (MTP) heads (`request.mtp_heads[k]`) are independent
output projections (`weight [V, d_model]`, `bias [V]`) that read the **same
shared trunk hidden states** and predict the token at future shift `k+1`. In the
forward pass each head computes `logits_k = matmul(h, Wᵀ) + bias_k`
(`broadcast_add`, `ModelForward_GPU.cu`) and the MTP loss backprops into the
trunk scaled by `mtp_alpha` (`AutogradMtpAuxiliaryLoss.cu`).

Crucially, those biases were allocated as **zeros** (`initializeMtpHeads` in
`ModelGpuAssembly.cu`). So an MTP head faces the **identical** incentive that
drove the main collapse: to predict the marginal `p(v)` by default it would push
a shared common-mode direction into the trunk — re-creating mechanism A through
the auxiliary path. Because the shifted-target marginal equals the overall
unigram `p(v)` (up to negligible sequence-edge truncation), the fix is to seed
**every** MTP head's bias with the same `log p(v)` vector.

This is done in the same `populateUnigramOutputBiases` pass (gated on the same
`lm_head_unigram_bias` flag, fresh-init only). MTP head biases are always
allocated and always serialized, so no serialization change is needed; on resume
the trained values are kept. The startup log line reports `mtp_heads=<N>`.

### Files changed

| File | Change |
|------|--------|
| `Shared/HyperParameters/HyperParameters_GPU.hpp` | `lm_head_unigram_bias` field + load/write macros |
| `Shared/HyperParameters/HyperparameterGroupings.hpp` | field in `ModelHP` & `LMHeadLayerConstructionHP` + mappings |
| `ai_config.json` | `lm_head_unigram_bias: true` |
| `training/Phases/Startup/Model/ParameterGroupRegistration.cu` | allocate + register bias on `use_bias \|\| unigram_bias` |
| `Layers/LMHead/lm_head_GPU.cu` | apply bias on `lm_bias.data` presence |
| `training/Phases/Phase1_Startup.cu` | `populateUnigramOutputBiases` (`log p(v)` into LM head **and all MTP heads**, fresh-init only) |
| `Common/grim_model_serialization.cu`, `Layers/Serialization/Serialization_{save,load}.cu` | presence-driven bias save/load |

## 5. Results

Full-model run with `lm_head_unigram_bias=true`, `use_bias=false`
(telemetry `1781895428825949890`):

- Peak `rho_atom_only` ≈ **0.50** (step ~100), `rho_nonatom_only` ≈ 0.49,
  `rho_final` ≈ 0.44 — then all decline to ~0.20–0.25 by step ~390.
- Compare to the collapsing ablations before the fix: `FFN1ATTNV0`
  `rho_atom_only` **pinned ~0.98**; `FFN0ATTNV1` ρ peaked **~0.9**.
- Peak ρ roughly **halved** and the stream recovers cleanly instead of pinning.
- `h_rms_growth` now decreases (141.4 → ~139.6) as per-token norms differentiate
  — the healthy degree of freedom returns.

## 6. How to verify on future runs

1. `lm_head_unigram_bias: true`, `use_bias: false` in `ai_config.json`.
2. Watch `unigram_dir_cos_abs_mean` / `unigram_dir_cos_signed_mean` (streams
   45–46) — should stay low/flat instead of spiking.
3. Watch the `rho_raw` numerator — should start low and stay low for both the
   FFN-only and AttnV-only ablations.
4. Startup log line: `[Phase1] lm_head_unigram_bias: initialized bias = log p(v)
   | vocab=… seen_tokens=… total_targets=… mtp_heads=…` (`mtp_heads` = number of
   MTP heads also seeded; 0 if MTP is disabled).

## 7. Residual notes / further levers

- The fix removes the **incentive** (mechanism A) and the load-bearing part of
  mechanism B, but B's pure softmax-averaging at init is structural. If further
  reduction is wanted, enable `qk_norm_enabled: true` (sharpens routing so
  attention leaves the uniform-averaging regime faster) and/or an attention-sink
  / register token.
- The bias is trainable from step 0. If a run still shows an early transient,
  consider freezing the bias for the first N steps so the marginal is pinned out
  of the residual before the writers can commit to a common mode.
