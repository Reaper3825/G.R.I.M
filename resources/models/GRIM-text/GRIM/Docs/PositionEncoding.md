# Position Encoding (ALiBi / RoPE)

Position info is injected **inside attention**, never in the residual stream. No position embeddings are added to token embeddings. Config fields are flat PBM leaves under `training.config`, loaded directly into `LanguageModelConfig` by the single HyperParameters root registry, computed there, and validated during HyperParameters model-architecture derivation.

Flat `training.config` PBM leaves are authored config, not a compile-time fallback. They must provide:

- `use_rope`
- `use_alibi`
- `rope_base_seq_len`
- `alibi_min_locality_distance`
- `alibi_slope_exponent`
- `alibi_max_bias`
- `rope_theta`
- `rope_scaling`

Construction reads use `HyperParameters::pbmConstructionHP()` / `PBMConstructionHP`; HyperParameters computes the derived ALiBi slopes and RoPE inverse-frequency host tables on the root `LanguageModelConfig`, and `HyperparameterGroupings.hpp` exposes immutable views of those tables to startup/runtime consumers. Startup PBM initialization consumes that grouped snapshot plus runtime-only options, not ad-hoc PBM config defaults or local re-derivation. RoPE launch wrappers take `BatchPayload` for per-call batch/sequence geometry and grouped attention HP for head geometry; callers must not unpack `batch_size`, `max_seq_len`, `head_dim`, or `rotary_dim` into sidecar PBM-owned geometry.

Durable PBM device state is owned by `TrainingContext::pbm_owner` (type `PBM::PBMStateOwner`, implemented in `Shared/PBM/PBMStateOwner.hpp/.cu`). The owner releases ALiBi slopes, RoPE inverse frequencies, and the upload event through RAII; it does **not** own static geometry, construction-policy mirrors, or the derived host tables after startup. Static geometry such as `rotary_dim`, `num_heads`, and `num_kv_heads`, plus the immutable host PBM tables, stay on config-owned grouped attention HP. `StreamController` supplies the initialization stream but does not own PBM buffers; `EmbeddingLayer` owns durable token weights; `BatchPayload` supplies per-call sequence geometry but does not own PBM state; autograd intermediates own transient Q/K/V/tape tensors only.

Phase1 seals PBM readiness before Phase2 shared forward begins. The shared forward boundary (`Forward::ModelForwardRequest`) carries the borrowed Phase1-owned `PBMState`; `forwardEncodingLayer(...)` and `encoderSelfAttentionForward(...)` take that borrowed PBM explicitly as part of the forward call chain. Encoder layers do not cache PBM as durable layer state, and attention does not perform startup-time PBM upload synchronization.

## ALiBi
- Slopes are capped via the authored `training.config.alibi_max_bias` value. A negative value such as `-10.0` ensures `exp(-10) ≈ 4.5e-5` (computable) instead of `exp(-256) ≈ 0` (underflow → gradient explosion). A value of `0.0` disables the cap and is still explicitly authored.
- **Sign convention (Dao FA2 kernel contract): slopes are POSITIVE magnitudes.** The vendored `alibi.h` applies `score += slope * col_idx` for causal attention, which equals the standard ALiBi penalty `-slope * (row - col)` up to a per-row constant that softmax cancels — but only when `slope > 0`. Passing negative slopes **inverts** the bias: the earliest keys (col 0) get the highest relative score in every head, so early tokens are systematically over-attended in both forward and backward (the bwd kernel recomputes the same scores for gradients). `computeDerivedPBMAlibiSlopes` emits positive magnitudes; `requirePBMComputedTables` fails loud on non-positive slopes (catches stale `pbm_alibi_slopes` snapshots/checkpoints from before the sign fix). The "penalty is negative" framing lives only in `alibi_max_bias ≤ 0`.
- Always match `max_seq_len` to actual context length — mismatched slopes cause weak attention at distance.

## RoPE NTK scaling
When `max_seq_len > training.config.rope_base_seq_len`:
$$\theta_{\text{eff}} = \theta \cdot \left(\frac{\text{max\_seq\_len}}{\text{rope\_base\_seq\_len}}\right)^{\frac{\text{rotary\_dim}}{\text{rotary\_dim} - 2}}$$
