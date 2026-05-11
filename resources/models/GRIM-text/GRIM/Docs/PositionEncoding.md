# Position Encoding (ALiBi / RoPE)

Position info is injected **inside attention**, never in the residual stream. No position embeddings are added to token embeddings. Config field: `positional_encoding` (parsed in `loadConfiguration()`).

Construction reads use `HyperParameters::pbmConstructionHP()` / `PBMConstructionHP`; PBM kernels consume that grouped snapshot and runtime-only options, not ad-hoc PBM config defaults. RoPE launch wrappers take `BatchPayload` for per-call batch/sequence geometry and grouped attention HP for head geometry; callers must not unpack `batch_size`, `max_seq_len`, or `head_dim` into scalar PBM calls.

## ALiBi
- Slopes capped via `ALIBI_MAX_BIAS = -10.0f` in `HyperParameters_GPU.hpp`. Ensures `exp(-10) ≈ 4.5e-5` (computable) instead of `exp(-256) ≈ 0` (underflow → gradient explosion).
- FlashAttention expects **negative** slopes (library uses `+= slope * col_idx`).
- Always match `max_seq_len` to actual context length — mismatched slopes cause weak attention at distance.

## RoPE NTK scaling
When `max_seq_len > 2048`:
$$\theta_{\text{eff}} = \theta \cdot \left(\frac{\text{max\_seq\_len}}{2048}\right)^{\frac{\text{rotary\_dim}}{\text{rotary\_dim} - 2}}$$
