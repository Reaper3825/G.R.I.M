# LM Head, Tied Embeddings, γ_final

## Tied embeddings (`tie_embeddings=true`)
LM head backward and embedding backward write to the **same buffer** via PyTorch-style direct accumulation (GPT-2 / LLaMA approach):

- `g_final = g_lm + g_emb` accumulated into one shared buffer
- LM head: dense GEMM (`grad_W = centered^T @ grad_logits`)
- Embedding: sparse scatter-add (`grad_W[tok] += grad_encoder[t]`)

`embedding_grads` and `lm_head_weight_grads` are the **same pointer**. Never zero both separately, register both in param groups, or free both.

`LMHeadLayer` consumes `HyperParameters::LMHeadLayerConstructionHP` directly. It stores that grouped construction-HP snapshot as `hp_` only because startup grouping temporaries go out of scope before forward/backward; it is **not** a second authored config owner. Runtime tying ownership must match the grouping: `tie_embeddings=true` requires a non-null embedding weight pointer, and `tie_embeddings=false` requires `nullptr`.

## Hidden-state centering
- Column-center `h` per sequence over real tokens only (`Σ_{t < seq_lengths[b]} h[b,t,d] = 0`) before LM head via `autograd::center_columns_by_sequence_lengths`. Never center over the full `[batch_size * seq_len, d_model]` matrix; that leaks batch-row information. Never center over the padded row stride without lengths; PAD activations are real vectors and would steer the mean.
- The length-aware centering op zeros padded rows in forward and zeros their gradients in backward. `BatchDeviceBindings.d_seq_lengths` is the required device-side source of truth for this mask.
- `project_out_pc1=true` composes after hidden-state centering when both flags are enabled. The LM-head matmul input is therefore `project_out_pc1(center_columns_by_sequence_lengths(RMSNorm(h)))`, not an implicit `center_hidden_states`-wins branch. This keeps the config truth aligned with the executed graph.
- `project_out_pc1` requires `pc1_power_iters >= 1`. `pc1_power_iters=0` only normalizes and projects the mean seed direction, which is not PC1; `lmHeadLayerConstructionHP(...)` and `autograd::project_out_pc1(...)` both fail loudly on zero/negative values.
- `project_out_pc1` backward must differentiate through the sequence-owned PC1 estimate, not treat the saved direction as a constant. The GradFn saves the forward input, all normalized power-iteration directions, and normalization scales, then reverses the final projection, every `xᵀxg` power step, and the initial column-mean seed.
- Row-center the LM head **weight** matrix (`Σ_d W[v,d] = 0`) via `autograd::center_rows(weights_)` inside `LMHeadLayer::forward` — equivalent invariance to row-centering `h`, but preserves per-position energy and enforces `Σ_d grad_h[t,d] = 0` in the backward pass.
- Single-token decode cannot apply hidden-state column centering because `Σ_t` has one row and would erase the hidden signal. `LMHeadLayer::forward` throws if `center_hidden_states=true` with `rows_per_sequence <= 1`.
- Generation must use the full-context prefill path, not KV single-token decode, whenever sequence-coupled geometry is enabled (`center_encoder_residuals`, `lm_head_center_hidden_states`, or `project_out_pc1`). KV decode is valid only for sequence-local configs; otherwise the sampler cannot compute sequence means/PC1 and will either throw or produce erased hidden states.
- Telemetry stream 38 (`rho_raw_rms_spread`): healthy 1.0–1.5×, >2× warn, >4× anomaly.

## γ_final (final RMSNorm gamma)
Registered as `ParamGroupType::RMSNORM` with `wd_mult=1.0` AND `lr_mult=0.1`. Without both, γ_final inflates as a logit temperature → mode collapse.

Empirically the inflation gradient still wins. Set `ai_config.json → training.config.lm_head_centering.freeze_final_rms_gamma=true` to hold γ_final at 1.0; the LM head W absorbs scale (GPT-2-style). Frozen mode skips `requires_grad_()`, `ensure_grad()`, and param-group registration.

Per-layer gammas (γ₁, γ₂) keep `lr_mult=1.0` and no decay — encoder nonlinearities give them mixed gradient signals that constrain growth naturally.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. ALiBi/RoPE inject position **inside** attention; the AIAYN scaling has no purpose here and creates a 27.7× gradient asymmetry with tied weights.
