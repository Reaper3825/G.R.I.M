---
name: Multi-Token Prediction Signal
overview: Auxiliary MTP heads on the shared encoder trunk produce an additive trajectory-aware loss that teaches the encoder to represent future token context, without destabilizing the primary next-token objective.
todos: []
isProject: false
---

# Multi-Token Prediction Training Signal

## Current Architecture

The training pipeline is:

1. `autogradTrainingStep()` in [`AutogradTraining.cu`](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu) runs a single forward-loss-backward cycle per batch
2. Forward: embedding → 12 encoder layers → LM head → logits `[total_tokens, vocab_size]`
3. Loss: `unified_loss()` in [`AutogradLoss.cu`](resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu) computes `log_softmax(logits)` then NLL loss (per-token CE with focal/smoothing/entropy options)
4. Backward: `loss.backward()` propagates gradients through the autograd graph
5. Phase2 calls `autogradTrainingStep()` from [`Phase2_TrainingLoop.cu`](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu) line ~1913

Model dimensions: `d_model=768, vocab_size=10000, num_layers=12, tie_embeddings=true`

---

## Why the Original Dual-Pass Design Fails

The previous design proposed running two forward-backward passes and selecting whichever gradient was "better" per token position. This is fundamentally broken:

1. **Non-differentiable selection destroys optimization.** `argmin` over discrete pass indices is not differentiable. The optimizer receives gradients from an objective that changes discontinuously between batches. This prevents convergence — the model oscillates instead of descending.

2. **Same logit row, different targets is incoherent.** Computing `CE(logits[t], target[t+2])` asks "does this hidden state predict the token 2 steps ahead?" but the LM head was trained to project for `target[t+1]`. A single linear head cannot learn K different projection tasks simultaneously. Each future position $k$ requires its own learned projection $W_k$ because the mapping $h_t \to \text{target}[t+k]$ is a different function for each $k$.

3. **"Best-of" hides the trajectory signal.** If the model is good at next-token but bad at $t+2$, selection always picks the next-token gradient and the model never learns trajectory. The whole point is the model must satisfy ALL prediction horizons simultaneously.

---

## Proposed Design: Auxiliary MTP Heads (Single Pass, Additive Loss)

Based on Meta's multi-token prediction (Gloeckle et al. 2024) and DeepSeek-V3's MTP training. The core idea: **add K auxiliary linear heads that predict future tokens from the same encoder output, and sum their losses into the main objective.**

### Architecture

```
                              encoder output h[t]  ∈ ℝ^{d_model}
                                       |
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
              W_lm (existing)    W_mtp_1 (new)  ...  W_mtp_K (new)
                    │                  │                   │
              logits_0[t]        logits_1[t]         logits_K[t]
                    │                  │                   │
           CE(·, target[t+1])  CE(·, target[t+2]) ... CE(·, target[t+K+1])
                    │                  │                   │
                    └────────── L_total = L_0 + α·mean(L_1..L_K) ──┘
```

Each MTP head is an independent linear projection `W_mtp_k ∈ ℝ^{d_model × vocab_size}` with its own bias `b_mtp_k ∈ ℝ^{vocab_size}`. They share the encoder trunk but have separate weights.

### Loss Function

$$L_{\text{total}} = L_{\text{CE}}(\text{logits}_0, \text{target}[t+1]) + \alpha \cdot \frac{1}{K} \sum_{k=1}^{K} L_{\text{CE}}(\text{logits}_k, \text{target}[t+k+1])$$

Where:
- $L_{\text{CE}}$ = the existing `unified_loss()` (focal + smoothing + entropy reg — same config for all heads)
- $\alpha$ = MTP loss coefficient (controls how strongly the auxiliary signal influences the encoder)
- $K$ = number of auxiliary future-prediction heads (2–4)
- $\text{logits}_k[t] = h_t \cdot W_{\text{mtp}_k}^T + b_{\text{mtp}_k}$

This is a **single forward pass** — the encoder runs once, and K+1 matmuls produce K+1 logit tensors. All K+1 losses are summed into a single scalar, and one `backward()` call propagates gradients through everything.

### Why This Teaches Trajectory

The encoder's hidden state $h_t$ must simultaneously minimize loss for predicting $\text{target}[t+1]$ (standard) AND $\text{target}[t+2], ..., \text{target}[t+K+1]$ (auxiliary). The only way it can do this is by encoding trajectory information — patterns about where the sequence is going, not just what the next token is.

The gradient from each auxiliary head flows back through the encoder via:

$$\frac{\partial L_{\text{total}}}{\partial h_t} = \frac{\partial L_0}{\partial h_t} + \alpha \cdot \frac{1}{K} \sum_{k=1}^{K} \frac{\partial L_k}{\partial h_t}$$

This is a clean additive gradient — no selection, no discontinuity. The encoder sees consistent pressure from all K+1 objectives every batch.

### Boundary Handling

For the last $K$ positions in a sequence, some auxiliary heads have no valid target. Two options:

- **Mask**: Set `target = -1` for out-of-bounds positions (existing `unified_loss` already skips `target == -1` tokens). The per-head loss denominator only counts valid tokens. **This is the correct approach** — it matches the existing mask contract.
- ~~Degrade to fewer heads~~ — adds complexity for no gain.

### Interaction with Tied Embeddings

The LM head (`W_lm`) is tied to the embedding matrix. The MTP heads (`W_mtp_k`) are **NOT tied** — they are independent parameters. This is correct because:
- The embedding matrix encodes token → hidden-state, the LM head inverts it. This duality only holds for next-token prediction.
- Predicting $t+k$ from $h_t$ is a different mapping. Tying would force the MTP head to use a projection optimized for next-token, which defeats the purpose.

### Interaction with Per-Component Gradient Clipping

MTP heads are a new component group. They should be clipped independently (same pattern as the existing per-component clipping in Phase2):

- **MTP clip** — `W_mtp_k`, `b_mtp_k` for all k

This prevents the MTP heads from dominating the gradient norm of other components. The encoder receives MTP gradient contributions through the autograd graph (as part of the encoder's own backward), so the encoder clip naturally covers them.

### α Schedule

Start with $\alpha = 0$ and linearly warm up to the target value over the first N steps. This lets the encoder stabilize on the primary objective before the auxiliary signal kicks in.

$$\alpha(t) = \alpha_{\text{target}} \cdot \min\left(1, \frac{t}{t_{\text{warmup}}}\right)$$

Recommended: $\alpha_{\text{target}} = 0.2$, $t_{\text{warmup}} = 500$ steps. These are starting points — tune based on the ratio of MTP loss to primary loss (they should be within 2–5× of each other after warmup).

---

## Trajectory Quality Diagnostic

The MTP heads provide a built-in diagnostic for whether the model is learning trajectory. Measure **accuracy per head** at each $k$:

$$\text{acc}_k = \frac{1}{T} \sum_{t} \mathbf{1}[\arg\max(\text{logits}_k[t]) = \text{target}[t+k+1]]$$

Plot $\text{acc}_k$ vs $k$. A model that understands trajectory will show:
- High $\text{acc}_1$ (next-token, same as standard)
- Slower decay for $k=2,3,4$ than a model trained without MTP

Also log the per-head loss ratio: $L_k / L_0$. If this ratio is stuck near $\ln(V) = \ln(10000) \approx 9.2$ (random guessing CE), the auxiliary heads aren't learning, which means α is too low or K is too high.

---

## VRAM Analysis

Model: d_model=768, vocab_size=10000, K=3 auxiliary heads.

**Per MTP head:**
- `W_mtp_k`: $768 \times 10000 \times 4$ bytes = **29.3 MiB** (weights)
- `b_mtp_k`: $10000 \times 4$ bytes = **39 KiB** (bias)
- `grad_W_mtp_k`: 29.3 MiB (gradient buffer)
- `optimizer_m/v`: 29.3 MiB each (AdamW states)
- Per-head total: **~117 MiB**

**K=3 total: ~351 MiB** for all auxiliary head parameters + optimizer states.

**Logit buffers** (computed per head, can be reused sequentially):
- `logits_k`: $\text{total\_tokens} \times 10000 \times 4$ bytes. At 8192 tokens max: **~312 MiB**
- Reuse the same buffer for each head (compute head k logits → loss → accumulate grad → next head)
- Total: **one 312 MiB logit buffer** (shared across heads, NOT K copies)

**Total additional VRAM: ~663 MiB** — trivial on a cluster.

If computing all heads concurrently: $K \times 312$ MiB = ~936 MiB for logit tensors + 351 MiB params = ~1.3 GiB. Still small.

---

## Implementation Plan

### 1. Config: `ai_config.json`

```json
"multi_token_prediction": {
    "enabled": false,
    "k": 3,
    "alpha": 0.2,
    "alpha_warmup_steps": 500
}
```

Parse into `LanguageModelConfig` (near `tie_embeddings`). Rule 20: `k` and `alpha` default to 0; throw if enabled && (k == 0 || alpha == 0.0).

### 2. MTP Head Weights: Ownership in `GPUGrimEncoder` or `GrimLanguageModel`

Add K weight+bias pairs alongside the existing LM head:

```cpp
// In the model class (wherever W_lm lives)
struct MTPHead {
    Tensor weight;  // [d_model, vocab_size]
    Tensor bias;    // [vocab_size]
};
std::vector<MTPHead> mtp_heads_;  // size = config.mtp_k
```

Initialize with Xavier uniform (same as LM head). Register in `buildParameterGroups()` as a new `MTP` component group.

### 3. Forward Pass Extension

After encoder output `h[t]` is computed but before the existing LM head matmul:

```
// Existing:
logits_0 = h @ W_lm^T + b_lm          // standard LM head

// New (for each k=1..K):
logits_k = h @ W_mtp_k^T + b_mtp_k    // auxiliary head k
loss_k   = unified_loss(logits_k, targets_shifted_by_k, ...)
```

The target shift is handled by passing `targets + k` pointer offset (with boundary masking). Each head's loss is computed identically to the main loss (same focal/smoothing/entropy config).

**Sequential head computation** (recommended for simplicity):
- Compute head k logits into a shared buffer
- Compute loss_k, accumulate into loss_total
- The autograd graph records each head's matmul + loss as part of the same graph
- One `backward()` at the end propagates through all heads + encoder

### 4. Backward: No Changes Needed

Because all K+1 losses are summed into a single scalar `loss_total`, the existing `loss_total.backward()` automatically propagates through the autograd graph to all heads and the encoder. No manual gradient management.

The encoder receives: `grad_h = grad_from_lm_head + α/K * Σ grad_from_mtp_head_k`

### 5. Optimizer: Register MTP Heads

Add MTP head parameters to `buildParameterGroups()` as a new component group with its own gradient clip threshold. Use the same LR and weight decay as the LM head.

### 6. Diagnostic Logging

Per phase2 diagnostic step, log:
```
[MTP_EQUATION] head_k=1: loss=X.XX  acc=Y.Y%  loss_ratio=L1/L0
[MTP_EQUATION] head_k=2: loss=X.XX  acc=Y.Y%  loss_ratio=L2/L0
[MTP_EQUATION] head_k=3: loss=X.XX  acc=Y.Y%  loss_ratio=L3/L0
[MTP_EQUATION] alpha_effective=0.XX  L_total = L0 + α*mean(L1..LK) = X.XX
```

### 7. Checkpoint Compatibility

MTP heads are optional. When loading a checkpoint:
- If checkpoint has MTP heads and config has MTP enabled: load them
- If checkpoint lacks MTP heads and config has MTP enabled: initialize fresh (Xavier)
- If checkpoint has MTP heads and config has MTP disabled: skip them (don't load dead weights)
- Serialize MTP head count in checkpoint header for validation

### 8. Inference

MTP heads are not used during inference. The model's `forward()` skips them when `is_training=false` (or when MTP is disabled). Zero runtime cost at inference.

Optionally, MTP heads can be repurposed for **speculative decoding** — head_k's argmax is a draft prediction for position $t+k+1$ that can be verified in a single forward pass. This is a future optimization, not a training concern.