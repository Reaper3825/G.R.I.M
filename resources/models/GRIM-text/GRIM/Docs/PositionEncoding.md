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

Construction reads use `HyperParameters::pbmConstructionHP()` / `PBMConstructionHP`; startup PBM initialization consumes that grouped snapshot and runtime-only options, not ad-hoc PBM config defaults. RoPE launch wrappers take `BatchPayload` for per-call batch/sequence geometry and grouped attention HP for head geometry; callers must not unpack `batch_size`, `max_seq_len`, `head_dim`, or `rotary_dim` into sidecar PBM-owned geometry.

Durable PBM device state is owned at model level by `LanguageModel::pbm_owner_` (`PBM::PBMStateOwner`, implemented in `Shared/PBM/PBMStateOwner.hpp/.cu`). The owner releases ALiBi slopes, RoPE inverse frequencies, host mirrors, and the upload event through RAII; it does **not** own static geometry or construction-policy mirrors after startup. `PBMSpec` is only a non-owning attention view into that owner (borrowed device buffers + upload event). Static geometry such as `rotary_dim`, `num_heads`, and `num_kv_heads` stays on config-owned grouped attention HP. `StreamController` supplies the initialization stream but does not own PBM buffers; `EmbeddingLayer` owns durable token weights; `BatchPayload` supplies per-call sequence geometry but does not own PBM state; autograd intermediates own transient Q/K/V/tape tensors only.

## ALiBi
- Slopes are capped via the authored `training.config.alibi_max_bias` value. A negative value such as `-10.0` ensures `exp(-10) ≈ 4.5e-5` (computable) instead of `exp(-256) ≈ 0` (underflow → gradient explosion). A value of `0.0` disables the cap and is still explicitly authored.
- FlashAttention expects **negative** slopes (library uses `+= slope * col_idx`).
- Always match `max_seq_len` to actual context length — mismatched slopes cause weak attention at distance.

## RoPE NTK scaling
When `max_seq_len > training.config.rope_base_seq_len`:
$$\theta_{\text{eff}} = \theta \cdot \left(\frac{\text{max\_seq\_len}}{\text{rope\_base\_seq\_len}}\right)^{\frac{\text{rotary\_dim}}{\text{rotary\_dim} - 2}}$$
