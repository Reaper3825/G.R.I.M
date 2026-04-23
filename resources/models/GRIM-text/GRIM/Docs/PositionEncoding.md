# Position Encoding (ALiBi / RoPE)

Position info is injected **inside attention**, never in the residual stream. No position embeddings are added to token embeddings. Config field: `positional_encoding` (parsed in `loadConfiguration()`).

## ALiBi
- Slopes capped via `ALIBI_MAX_BIAS = -10.0f` in `HyperParameters_GPU.hpp`. Ensures `exp(-10) ≈ 4.5e-5` (computable) instead of `exp(-256) ≈ 0` (underflow → gradient explosion).
- FlashAttention expects **negative** slopes (library uses `+= slope * col_idx`).
- Always match `max_seq_len` to actual context length — mismatched slopes cause weak attention at distance.

## RoPE NTK scaling
When `max_seq_len > 2048`:
$$\theta_{\text{eff}} = \theta \cdot \left(\frac{\text{max\_seq\_len}}{2048}\right)^{\frac{\text{rotary\_dim}}{\text{rotary\_dim} - 2}}$$
