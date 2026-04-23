# Unified Loss

Implementation: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu`.

`autograd::unified_loss()` is the **only** loss path. `cross_entropy_loss()` is a wrapper that calls it with plain-CE config.

## Formula
$$L = \alpha (1 - p_t)^\gamma \cdot \mathrm{CE}_{\text{smooth}} + \lambda \cdot H(p)$$

(focal + label smoothing + entropy regularization in a single autograd kernel)

## Config
`ai_config.json → training.config.loss`

## Per-component gradient clipping
Three **independent** clips:

1. **emb** — LM_HEAD (+ EMBEDDING if untied)
2. **enc** — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK
3. **num** — NUMERIC_HEAD

Never clip them jointly when one component dominates the L2 norm — it crushes the smaller ones.

## Footgun: double mean reduction
Loss backward already applies `1/N`. Do **not** add another `1/tokens` scaling in parameter gradient kernels (RMSNorm γ, etc.).
