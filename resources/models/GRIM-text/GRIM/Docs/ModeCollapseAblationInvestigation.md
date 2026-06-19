# Mode Collapse Ablation Investigation (rho buildup)

## Status

ACTIVE. The driver of residual-stream mode collapse (rho buildup) is being
isolated with the compile-time sublayer ablations in
`Layers/Encoding/AblationFlags.hpp`. This document records the experiment, the
result that must be explained, the surviving hypothesis, and the instrumentation
added to test it. It is intentionally written so the next run either confirms or
kills the hypothesis with a logged number.

## Experimental setup

All other interventions are OFF for this investigation so the residual stream is
made of exactly two contributions (attention, FFN) on top of the embeddings:

- `center_encoder_residuals` (residual centering): OFF
- LayerScale (`use_layer_scale`): OFF
- bias (`use_bias`): OFF  (so there is no `b_o`/`b_qkv` escape term)
- regularization: OFF
- causal mask: DISABLED by default (`GRIM_ABLATE_DISABLE_CAUSAL_MASK` defaults
  ON -> attention is bidirectional)

Relevant ablation flags (`AblationFlags.hpp`):

- `kZeroAttnResidual` — zero the WHOLE attention contribution at the residual add
  (`Encoding_GPU.cu`). Forward becomes `residual1 = input`. The header calls this
  the "embedding + FFN-only stack".
- `kZeroAttnV` — zero the attention VALUE vectors BEFORE SDPA
  (`EncoderSelfAttention_GPU.cu`). Softmax/QK routing still computes, but
  `attn_out = softmax(.) · 0 = 0`, so the forward residual is ALSO `residual1 = input`.

## The result that must be explained (ground truth)

Measured empirically (see `training/logs/ATTNCOLAPPSE_BASLINE/`):

- `kZeroAttnResidual = true`  -> NO collapse.
- `kZeroAttnV = true`         -> collapse.

This is the observed result, not a hypothesis. The investigation explains it; it
does not relitigate it.

## Why this is a real puzzle (not a config mistake)

With bias/LayerScale/centering OFF, both ablations produce a provably IDENTICAL
forward residual stream:

```
kZeroAttnV:        V = 0 -> attn_out = softmax(QKᵀ/√d + ALiBi) · 0 = 0
                   proj_out = matmul(attn_out=0, W_o) = 0   (no bias)
                   residual1 = input + 0 = input
kZeroAttnResidual: proj_out (real) zeroed at residual add
                   residual1 = input + 0 = input
```

`attn_out == 0` under `kZeroAttnV` is PROVEN by the logs, not assumed:

- FlashAttention output diagnostic: `[FA-FWD-OUT] output(bf16→fp32): zero=200/200 first=0.000000`
- Added forward guard: `[ABLATION][kZeroAttnV] layer=N OK: attn_out is exact-zero for zeroed V`
  (IEEE-754: finite · 0 == 0, checked bit-exact, no tolerance).

Two identical forward residual streams cannot collapse differently. Therefore the
collapse difference CANNOT come from attention's forward contribution. It must be
in the BACKWARD pass — the only place the two configs differ.

## Surviving hypothesis: FlashAttention backward gradient leak

The single structural difference between the two configs is what the
FlashAttention backward receives:

- `kZeroAttnResidual`: gradient is killed at the residual add, so the SDPA
  backward sees `dO = 0` and produces clean zeros everywhere.
- `kZeroAttnV`: the SDPA forward AND backward are fully live. The backward gets a
  real nonzero `dO = dproj_out · W_oᵀ`, with V == 0.

Pure autograd math says `V = 0` forces `dQ = dK = 0` (every term carries a factor
of V), so `W_qkv`/`W_o` must receive EXACTLY zero gradient — identical to
`kZeroAttnResidual`. BUT the score regime is extreme:

```
[ATTN_SCORE_EQUATION] ACTUAL score: min=-255.7500 max=1.6161 rms=104.4635
alibi_slope[head0] = 0.250000 (max_distance=1023 -> max_bias=255.7500)
```

Scores reaching -255 with rms≈104 is a brutal regime for a softmax-gradient
recompute. If the FlashAttention backward LEAKS a nonzero `dQ`/`dK` under this
ALiBi extreme (a `0·exp(huge)` cancellation that does not land on exact zero),
then in `kZeroAttnV` an attention-score-derived gradient flows back through
`W_qkv` into `ln1_out`, into the embeddings and `rms1_gamma`. That trains the
trunk toward the attention routing's shared direction and drives collapse — and
it happens ONLY in `kZeroAttnV`, because `kZeroAttnResidual` feeds the SDPA
backward a clean zero. The forward residual being identical is irrelevant; the
collapse is injected through the GRADIENT, not the activation.

This is falsifiable: under `kZeroAttnV` the math says `W_qkv.grad` and `W_o.grad`
RMS must be exactly 0. Any nonzero value is the leak.

## Instrumentation added

1. Forward exact-zero guard — `Layers/FlashAttention/EncoderSelfAttention_GPU.cu`
   (compiled only under `kZeroAttnV`). Spot-checks `attn_out` right after SDPA via
   a synchronous `cudaMemcpy` and bit-exact `== 0.0f`:
   - log tag `[ABLATION][kZeroAttnV] layer=N OK: attn_out is exact-zero for zeroed V`
   - or `[ABLATION][kZeroAttnV] layer=N FAIL: ... attn_out is NON-ZERO ...`
     (would implicate the kernel's output buffer; currently always OK).

2. Backward gradient-leak probe — `training/Autograd/AutogradTraining.cu`, in
   `verifyGradientsAreConnectedImpl` (runs AFTER backward, where parameter grads
   exist). Compiled only under `kZeroAttnV || kZeroAttnResidual`. Reports per-layer
   `W_qkv`/`W_o` gradient RMS via `probeGradientSignal()`:
   - log tag `[ABLATION-GRADLEAK][<config>] layer=N attnWqkv.grad rms=.. nonzero=.. finite=.. checked=.. -> LEAK | clean-zero`
   - `<config>` is `kZeroAttnV` or `kZeroAttnResidual` so the two runs are directly comparable.

## How to read the next run

Grep the run for `ABLATION-GRADLEAK`:

- `nonzero=no`, `rms=0` (both configs) -> attention backward delivers exactly
  zero gradient. The two configs are provably identical in BOTH passes; the
  collapse difference is coming from outside this path and the hunt redirects.
- `nonzero=yes`, `rms>0` under `kZeroAttnV` but `clean-zero` under
  `kZeroAttnResidual` -> CONFIRMED FlashAttention backward leak. That leak is the
  collapse driver and fully explains the observed `kZeroAttnV`-only collapse.

Note: a `dQ`/`dK` leak still surfaces as nonzero `W_qkv.grad`, because
`dW_qkv = ln1_outᵀ · dqkv`. So `W_qkv.grad` RMS is the single number to watch.

## Procedure to capture the comparison

1. Build + run with `kZeroAttnV = true` (current default). Collect the
   `[ABLATION-GRADLEAK][kZeroAttnV]` lines and the rho curve (collapse expected).
2. Set `kZeroAttnV = false`, `kZeroAttnResidual = true`. Build + run. Collect the
   `[ABLATION-GRADLEAK][kZeroAttnResidual]` lines and the rho curve (no collapse
   expected).
3. Compare `attnWqkv.grad` / `attnWo.grad` RMS between the two. The hypothesis
   predicts leak under `kZeroAttnV`, clean-zero under `kZeroAttnResidual`.

## Related files

- `Layers/Encoding/AblationFlags.hpp` — flag definitions and per-flag semantics.
- `Layers/Encoding/Encoding_GPU.cu` — `kZeroAttnResidual` / `kZeroFfnResidual` residual zeroing.
- `Layers/FlashAttention/EncoderSelfAttention_GPU.cu` — `kZeroAttnV` (and forward guard), `kZeroAttnQKScores`, `kZeroAttnOProj`, `kZeroRope`.
- `training/Autograd/AutogradTraining.cu` — gradient verifier + `[ABLATION-GRADLEAK]` probe.
