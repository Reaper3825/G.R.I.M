# GRIM-text Readiness Checklist

This is a source-grounded checklist, not a claim that the trainer, server,
every model configuration, or every GPU/platform is production-certified.
The old December 2025 list of manual TrainingOps/BackwardPhase fixes is no
longer the current implementation map.

## Implemented boundaries to preserve

| Area | Current implementation and invariant |
|---|---|
| Configuration | [train_gpu.cu](../../resources/models/GRIM-text/training/train_gpu.cu) loads/finalizes the authoritative config snapshot before Phase 1; grouped consumers do not own parallel config |
| GQA geometry | [HyperparameterGroupings.hpp](../../resources/models/GRIM-text/Shared/HyperParameters/HyperparameterGroupings.hpp) derives validated head/QKV dimensions; `G = D + 2 * Hkv * (D/H)` |
| Parameters | [ParameterGroupRegistration.cu](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterGroupRegistration.cu) validates and atomically publishes registered tensors; tying is registered once |
| Forward | [ModelForward_GPU.cu](../../resources/models/GRIM-text/Shared/Forward/ModelForward_GPU.cu) consumes registry/config/payload views; no old manual forward/backward phases |
| Attention | [AutogradAttention.cu](../../resources/models/GRIM-text/Shared/TensorContract/AutogradAttention.cu) wires pinned FA2 forward/backward, saved buffers, GQA, dropout, and padding bounds |
| Loss | [AutogradLoss.cu](../../resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu) owns saved log probabilities and directly contributes fused text-loss logits gradients |
| Accumulation | [Phase2_TrainingLoop.cu](../../resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu) authors objective-specific window weights and includes the final short window |
| Clipping/update | [GradientCC_GPU.cu](../../resources/models/GRIM-text/Shared/Gradients/GradientCC_GPU.cu) applies normalization once and measures registered global RMS; [OptimizerUpdate_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/OptimizerUpdate_GPU.cu) dispatches the configured optimizer |
| Checkpoints | [ParameterCheckpoint.cu](../../resources/models/GRIM-text/Common/ParameterCheckpoint.cu) validates exact registry manifests/checksums and verifies a written artifact |
| Logging | [Logging.cu](../../resources/models/GRIM-text/training/Phases/Startup/Logging.cu) derives the session log location from configured paths |
| HTTP/model ownership | [grim_text_server.cpp](../../resources/models/GRIM-text/GRIM/grim_text_server.cpp) is the bridge; the trainer-owned inference worker owns model startup and generation |

The old unchecked "hardcoded default log path" item must not be treated as an
unresolved defect in the current startup logging path. Likewise, reserved schema
enum values and archived architecture files are not active model features.

## Checks requiring actual execution before release

- [ ] Build the explicitly selected target with its pinned toolchain/dependencies.
- [ ] Exercise supported GQA/head dimensions, causal/padding settings, and dropout.
- [ ] Verify loss derivatives against the configured objective, including masking,
  smoothing/focal/entropy options, and class weights.
- [ ] Verify additive tied/branched/microbatch gradients and final-window weighting.
- [ ] Verify finite optimizer updates and resume chronology.
- [ ] Save, load, corrupt, and reject checkpoint fixtures for the actual inventory.
- [ ] Verify inference last-row readout and bridge/worker readiness/failure handling.
- [ ] Test the supported platform/GPU/model combinations explicitly.

These checkboxes are release evidence to collect, not assertions of known
failures. Documentation maintenance did not execute trainer/model/GPU targets.
RL environment interaction exists, but the policy-gradient update scaffold is
not a completed RL training path. See
[GRIM_RL_TRAINING_LOOP_PSEUDOCODE.md](plans/RL_TRAINING_LOOP.md).

For current detailed contracts, use [LEARNED_PARAMETERS.md](PARAMETERS.md),
[AUTOGRAD_MIGRATION.md](AUTOGRAD.md), and
[TRAINING_MATH_AUDIT.md](TRAINING_MATH.md).
