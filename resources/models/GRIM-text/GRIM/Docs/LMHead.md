# LM Head, Tied Embeddings, γ_final

## Tied embeddings (`tie_embeddings=true`)
LM head backward and embedding backward write to the **same buffer** via PyTorch-style direct accumulation (GPT-2 / LLaMA approach):

- `g_final = g_lm + g_emb` accumulated into one shared buffer
- LM head: dense GEMM (`grad_W = lm_input^T @ grad_logits`)
- Embedding: sparse scatter-add (`grad_W[tok] += grad_encoder[t]`)

`embedding_grads` and `lm_head_weight_grads` are the **same pointer**. Never zero both separately, register both in param groups, or free both.

## Durable ownership

- Durable LM-head tensors now live on `TrainingContext::parameter_registry.lm_head_parameters` as the single startup-owned bundle for:
	- `weights`
	- `bias`
	- `final_rms_gamma`
- `GpuModelState::lm_head_layer` remains the forward/topology object, but it **borrows** that registry-owned bundle and must not be treated as the durable tensor owner.
- Direct runtime diagnostics/startup verification code should read LM-head tensors through the registry owner, not through `LanguageModel::getLmHeadLayer()` wrappers.
- Phase2 LM-head diagnostics/consumers must take `StartupParameterRegistry&` directly when they need LM-head durable tensors. The helper boundary should read `parameter_registry.requireLmHeadParameters(...)` from that explicit owner input, not from `TrainingContext`, `LanguageModel`, or `LMHeadLayer` reach-throughs. Current runtime examples include logit-scale, special-token, tie-verification, post-optimizer traces, and the post-clip parameter-gradient equation diagnostic.
- Tied embeddings remain valid under this ownership split because `lm_head_parameters.weights` aliases `EmbeddingLayer::tokenWeights()` and shares the same grad buffer.

`LMHeadLayer` consumes `HyperParameters::LMHeadLayerConstructionHP` directly. It stores that grouped construction-HP snapshot as `hp_` only because startup grouping temporaries go out of scope before forward/backward; it is **not** a second authored config owner. Runtime tying ownership must match the grouping: `tie_embeddings=true` requires a non-null embedding weight pointer, and `tie_embeddings=false` requires `nullptr`.

- `LMHeadLayer::forward` is read-only over durable parameter state. Training may pass the live trainable tensors so autograd can attach graph edges, but inference-prefill must pass detached views. Forward must not restamp `weights_.shape`, toggle `requires_grad`, or cache forward-derived `W_eff` on the durable layer object.
- The effective LM-head weight tensor (`W_eff` after optional token-type gating) is Category 1 graph-time state. It is built per call inside `LMHeadLayer::forward` and consumed immediately by `autograd::matmul`; it is not a persistent `LMHeadLayer` member.
- `LMHeadLayer::forward` now writes directly into the active `Forward::ModelForwardOutputs` sink. If the head materializes a transformed LM input (`RMSNorm(h)` or the residual SwiGLU adapter output), it stores that tape-local tensor in `ModelForwardOutputs::lm_head_input_tensor` and stores logits in `ModelForwardOutputs::logits_tensor` for same-boundary diagnostics and loss. Do not reintroduce a second wrapper/adopter type for these two fields.
- LM-head forward must not restamp `forward_outputs.logits_tensor.shape` after generic autograd ops. The live logits tensor is still a Category 1 per-call output; consumers validate the explicit `[tokens, vocab_size]` rectangle from the tensor's 2D shape and payload/config geometry instead of relying on a per-layer `Layout::LOGITS` relabel.

## Row-local readout
The head applies final RMSNorm, the optional residual SwiGLU adapter, the optional token-type weight gate, the LM projection, and bias. Each hidden row is read independently. Encoder residual centering, hidden-state/weight centering, logit centering, and PC1 projection were removed after the collapse investigation was resolved; they are not config options or lens constraints.

The token-type gate uses `type_gate_rows_by_token_type(W_lm)` when `kEnableLmHeadTokenTypeGateExperiment` is enabled and raw `W_lm` otherwise. Gating does not subtract means.

Training uses the configured rectangle (`training_batch_size`, `training_rows_per_sequence`); inference uses payload geometry. These geometry checks remain independent of the row-local math.

## Learned RMSNorm gammas
All learnable RMSNorm gammas register through the same startup parameter-registration path as the rest of the model. The current registration contract stamps the default optimizer multipliers (`wd_mult=1.0`, `lr_mult=1.0`) uniformly because these knobs are not actively authored yet.

Set `ai_config.json → training.config.freeze_learned_rms_gammas=true` to hold **all 25 learned RMSNorm gamma vectors** at 1.0 (24 encoder gammas + `γ_final`). Frozen mode skips `requires_grad_()`, `ensure_grad()`, checkpoint overwrite on load, and param-group registration.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. ALiBi/RoPE inject position **inside** attention; the AIAYN scaling has no purpose here and creates a 27.7× gradient asymmetry with tied weights.
