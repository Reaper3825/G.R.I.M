# LM Head, Tied Embeddings, γ_final

## Tied embeddings (`tie_embeddings=true`)
LM head backward and embedding backward write to the **same buffer** via PyTorch-style direct accumulation (GPT-2 / LLaMA approach):

- `g_final = g_lm + g_emb` accumulated into one shared buffer
- LM head: dense GEMM (`grad_W = centered^T @ grad_logits`)
- Embedding: sparse scatter-add (`grad_W[tok] += grad_encoder[t]`)

`embedding_grads` and `lm_head_weight_grads` are the **same pointer**. Never zero both separately, register both in param groups, or free both.

`LMHeadLayer` consumes `HyperParameters::LMHeadLayerConstructionHP` directly. It stores that grouped construction-HP snapshot as `hp_` only because startup grouping temporaries go out of scope before forward/backward; it is **not** a second authored config owner. Runtime tying ownership must match the grouping: `tie_embeddings=true` requires a non-null embedding weight pointer, and `tie_embeddings=false` requires `nullptr`.

- `LMHeadLayer::forward` is read-only over durable parameter state. Training may pass the live trainable tensors so autograd can attach graph edges, but inference-prefill must pass detached views. Forward must not restamp `weights_.shape`, toggle `requires_grad`, or cache forward-derived `W_eff` on the durable layer object.
- The effective LM-head weight tensor (`W_eff` after token-type gating and/or row centering) is Category 1 graph-time state. It is built per call inside `LMHeadLayer::forward` and consumed immediately by `autograd::matmul`; it is not a persistent `LMHeadLayer` member.

## Hidden-state centering
- `LMHeadLayer::forward` treats fixed-shape training geometry as config-authored via its grouped construction HP snapshot (`training_batch_size`, `training_rows_per_sequence`) and only uses `BatchPayload` to mirror/validate that rectangle plus supply `seq_lengths`. Inference prefill/decode still source runtime geometry from the payload modes.
- Column-center `h` per sequence over real tokens only (`Σ_{t < seq_lengths[b]} h[b,t,d] = 0`) before LM head via `autograd::center_columns_by_sequence_lengths`. Never center over the full `[batch_size * seq_len, d_model]` matrix; that leaks batch-row information. Never center over the padded row stride without lengths; PAD activations are real vectors and would steer the mean.
- The length-aware centering op zeros padded rows in forward and zeros their gradients in backward. `BatchDeviceBindings.d_seq_lengths` is the required device-side source of truth for this mask.
- `project_out_pc1=true` composes after hidden-state centering when both flags are enabled. The LM-head matmul input is therefore `project_out_pc1(center_columns_by_sequence_lengths(RMSNorm(h)))`, not an implicit `center_hidden_states`-wins branch. This keeps the config truth aligned with the executed graph.
- `project_out_pc1` requires `pc1_power_iters >= 1`. `pc1_power_iters=0` only normalizes and projects the mean seed direction, which is not PC1; `lmHeadLayerConstructionHP(...)` and `autograd::project_out_pc1(...)` both fail loudly on zero/negative values.
- `project_out_pc1` backward must differentiate through the sequence-owned PC1 estimate, not treat the saved direction as a constant. The GradFn saves the forward input, all normalized power-iteration directions, and normalization scales, then reverses the final projection, every `xᵀxg` power step, and the initial column-mean seed.
- Row-center the LM head **effective weight** matrix (`Σ_d W_eff[v,d] = 0`) inside `LMHeadLayer::forward` rather than destructively row-centering `h`. With the default local experiment toggle still enabled, this is `autograd::center_rows_by_token_type_gate(weights_)`, so centering happens only inside the active token-type subspace and inactive dims stay exactly zero. If the local LM-head type-gate experiment is disabled, the centered branch falls back to plain `autograd::center_rows(weights_)`.
- Live Issue #132 order in `LMHeadLayer::forward`: `RMSNorm(h)` → optional `center_columns_by_sequence_lengths(...)` → optional `project_out_pc1(...)` → derive `W_eff` from `W_lm` (default: `center_rows_by_token_type_gate(W_lm)` when hidden centering is enabled, `type_gate_rows_by_token_type(W_lm)` otherwise; local experiment-off path: `center_rows(W_lm)` or raw `W_lm`) → `autograd::matmul(lm_input, W_eff, transpose_b=true)`. The row-centering stage is therefore after hidden column-centering, but it is applied to the LM projection weights rather than destructively row-centering `h`.
- `lm_head_GPU.cu` also exposes a file-local experiment toggle, `kEnableLmHeadTokenTypeGateExperiment`, for temporarily bypassing the LM-head hard token-type gate without plumbing a JSON/config field. Default `true` preserves the current gated path. Setting it `false` makes the centered branch use plain `center_rows(W_lm)` and the uncentered branch use raw `W_lm`; this is intentionally LM-head-local and can break embedding/LM symmetry when `tie_embeddings=true`.
- Single-token decode cannot apply hidden-state column centering because `Σ_t` has one row and would erase the hidden signal. `LMHeadLayer::forward` throws if `center_hidden_states=true` with `rows_per_sequence <= 1`.
- Generation must use the full-context prefill path, not KV single-token decode, whenever sequence-coupled geometry is enabled (`center_encoder_residuals`, `lm_head_center_hidden_states`, or `project_out_pc1`). KV decode is valid only for sequence-local configs; otherwise the sampler cannot compute sequence means/PC1 and will either throw or produce erased hidden states.
- Telemetry stream 38 (`rho_raw_rms_spread`): healthy 1.0–1.5×, >2× warn, >4× anomaly.

## Learned RMSNorm gammas
Registered as `ParamGroupType::RMSNORM` with `wd_mult=1.0` AND `lr_mult=0.1`. Without both, γ_final inflates as a logit temperature → mode collapse.

Empirically the inflation gradient still wins. Set `ai_config.json → training.config.lm_head_centering.freeze_learned_rms_gammas=true` to hold **all 25 learned RMSNorm gamma vectors** at 1.0 (24 encoder gammas + `γ_final`); the LM head W and the rest of the model absorb scale instead. Frozen mode skips `requires_grad_()`, `ensure_grad()`, checkpoint overwrite on load, and param-group registration.

When `freeze_learned_rms_gammas=false`, per-layer gammas (γ₁, γ₂) keep `lr_mult=1.0` and no decay — encoder nonlinearities give them mixed gradient signals that constrain growth naturally.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. ALiBi/RoPE inject position **inside** attention; the AIAYN scaling has no purpose here and creates a 27.7× gradient asymmetry with tied weights.
