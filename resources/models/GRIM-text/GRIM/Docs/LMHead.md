# LM Head, Tied Embeddings, γ_final

## Tied embeddings (`tie_embeddings=true`)
LM head backward and embedding backward write to the **same buffer** via PyTorch-style direct accumulation (GPT-2 / LLaMA approach):

- `g_final = g_lm + g_emb` accumulated into one shared buffer
- LM head: dense GEMM (`grad_W = centered^T @ grad_logits`)
- Embedding: sparse scatter-add (`grad_W[tok] += grad_encoder[t]`)

`embedding_grads` and `lm_head_weight_grads` are the **same pointer**. Never zero both separately, register both in param groups, or free both.

## Hidden-state centering
- Column-center `h` (`Σ_t h[t,d] = 0`) before LM head.
- Row-center the LM head **weight** matrix (`Σ_d W[v,d] = 0`) via `autograd::center_rows(weights_)` inside `LMHeadLayer::forward` — equivalent invariance to row-centering `h`, but preserves per-position energy and enforces `Σ_d grad_h[t,d] = 0` in the backward pass.
- Telemetry stream 38 (`rho_raw_rms_spread`): healthy 1.0–1.5×, >2× warn, >4× anomaly.

## γ_final (final RMSNorm gamma)
Registered as `ParamGroupType::RMSNORM` with `wd_mult=1.0` AND `lr_mult=0.1`. Without both, γ_final inflates as a logit temperature → mode collapse.

Empirically the inflation gradient still wins. Set `ai_config.json → training.config.lm_head_centering.freeze_final_rms_gamma=true` to hold γ_final at 1.0; the LM head W absorbs scale (GPT-2-style). Frozen mode skips `requires_grad_()`, `ensure_grad()`, and param-group registration.

Per-layer gammas (γ₁, γ₂) keep `lr_mult=1.0` and no decay — encoder nonlinearities give them mixed gradient signals that constrain growth naturally.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. ALiBi/RoPE inject position **inside** attention; the AIAYN scaling has no purpose here and creates a 27.7× gradient asymmetry with tied weights.
