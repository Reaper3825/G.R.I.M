# GRIM RL forward/backward training loop pseudocode

Design pseudocode for the proposed
`training/Phases/Phase2_RLTrainingLoop.cu/.hpp`. Function/type names below are
illustrative, not implemented APIs. The existing `TrainingStage::RL` selector
is implemented; the handoff, RL preparation, and training loop remain pending.
The `.hpp` / `.cu` pair now exists as a source scaffold with an opaque prepared
group input, a `bool` success placeholder, commented loop pseudocode, and an
explicit unimplemented exception. The richer update receipt shown here remains
proposed. The source scaffold is not wired into training dispatch or the build.

The episode scheduler is a separate sequence. It supplies one complete,
prepared, scored group: 16 tasks, four independent episodes per task, and only
the retries that actually occurred. Generation, tool execution, grading,
rewards, advantage normalization, and pending retry-baseline calculation happen
before this loop. The loop returns an update receipt to that sequence.

```cpp
// Proposed stage dispatch after mode-aware startup/preparation.
switch (handoff.training_stage) {
    case TrainingStage::PT:
    case TrainingStage::SFT:
        return executePhase2TrainingLoop(handoff);  // existing epoch path

    case TrainingStage::RL:
        return executePhase2RLTrainingLoop(handoff, prepared_group);

    default:
        return dispatchExistingStageOrReject(handoff);
}

UpdateReceipt executePhase2RLTrainingLoop(handoff, const PreparedGroup& group) {
    require(handoff.training_stage == TrainingStage::RL);
    require(group.complete && !group.consumed);
    require(group.task_count == 16 && group.episode_count == 64);
    require(each_task_has_four_complete_valid_episodes(group));
    require(group.policy_version == handoff.policy.version);
    require(group.reference_identity == handoff.frozen_reference.identity);
    require(no_optimizer_update_or_other_policy_mutation_during_this_call());
    require(all_policy_parameters_registered_and_trainable());
    require(reference_parameters_frozen_and_outside_optimizer());
    require(auxiliary_objective_policy_and_clipping_semantics_are_resolved());

    // Validate the whole plan before backward, not one microbatch at a time.
    require(exactly_once_coverage_of_all_recorded_sampled_actions(group));
    require(prefixes_match_recorded_model_visible_history(group));
    require(no_missing_or_fabricated_retries(group));
    require(detached_attempt_local_advantages_from_one_group_snapshot(group));
    require(causal_action_masks_and_atom_channels_are_aligned(group));

    constexpr float denominator = 256.0f * 64.0f;
    const float kl_coefficient = rlTrainingHP(handoff.ctx.config).kl_coefficient;
    const auto policy_version_before = handoff.policy.version;
    const auto optimizer_step_before = handoff.optimizer.step;
    Metrics totals{};

    // Use existing gradient ownership/reset facilities. Start a fresh group
    // window; accumulate sums without an implicit division by microbatch count.
    beginGradientWindowWithZeroedPolicyGradients(handoff);

    try {
        for (const PreparedMicrobatch& batch : group.microbatches) {
            MicrobatchResourceScope resources;  // existing device/graph owners
            const auto& payload = batch.payload;
            uploadBatchPayload(payload, resources);

            if (batch.sampled_action_count == 0) {
                continue;  // empty contribution; denominator stays unchanged
            }

            // Replay recorded tokens only. No generation, parser, dispatcher,
            // variable-store mutation, or tool re-execution occurs here.
            auto reference_logits = sharedForward(
                handoff.frozen_reference, payload,
                connect_parameter_graph = false,
                dropout = false,
                use_generation_kv_cache = false);
            reference_logits = detach(reference_logits);

            auto policy_logits = sharedForward(
                handoff.policy, payload,
                connect_parameter_graph = true,
                dropout = false,
                use_generation_kv_cache = false);

            // Same categorical distribution and fixed legal support as rollout:
            // temperature 1, no truncation, penalties, or external biases.
            auto log_p = fp32LogSoftmax(policy_logits, batch.legal_token_mask);
            auto log_q = fp32LogSoftmax(reference_logits, batch.legal_token_mask);

            Tensor pg_sum = connectedZero(policy_logits);
            Tensor kl_sum = connectedZero(policy_logits);

            for (const ActionRow& action : batch.sampled_action_rows) {
                // prediction_row precedes the sampled target token. The first
                // action after an injected observation is included correctly.
                const int row = action.prediction_row;
                const int target = action.sampled_token_id;
                const float advantage = action.detached_attempt_advantage;

                pg_sum += -advantage * log_p[row, target];
                for (int v : batch.legal_output_ids(row)) {
                    kl_sum += exp(log_p[row, v]) *
                              (log_p[row, v] - log_q[row, v]);
                }
            }

            // KL is still scored when advantage is zero. Both log_p and exp
            // remain differentiable; only the reference and advantages detach.
            auto loss = (pg_sum + kl_coefficient * kl_sum) / denominator;
            loss += explicitlySelectedAuxiliaryLossIfAny(batch, handoff);
            requireFinite(loss);

            backward(loss, accumulate_into_policy_gradients = true);
            totals.addDetached(pg_sum, kl_sum, batch.sampled_action_count);
            // Release this microbatch graph/device owners before the next pass.
        }

        require(all_group_microbatches_finished());
        requirePolicyGradientsFinite(handoff);
    } catch (...) {
        discardPartialGradientWindow(handoff);
        throw;  // no optimizer commit or successful update receipt
    }

    // One clipping operation and one optimizer update for the entire group.
    // Reuse existing implementations; the adapter must preserve the reduction
    // above, including any averaging performed by existing accumulation code.
    const float lr = scheduledLearningRateForOptimizerStep(
        handoff, optimizer_step_before);
    clipPolicyGradientsUsingResolvedExistingMetric(handoff);
    launchExistingRAdamWUpdate(handoff, lr, optimizer_step_before);
    synchronizeAndCheckUpdatedPolicyFinite(handoff);
    completeOptimizerWindowBookkeeping(handoff);  // advance step exactly once

    totals.policy_loss = totals.pg_sum / denominator;
    totals.weighted_kl_loss = kl_coefficient * totals.kl_sum / denominator;
    totals.mean_kl_per_action = totals.action_count > 0
        ? optional(totals.kl_sum / totals.action_count)
        : unavailable;
    logRLMetrics(totals);  // no CE/perplexity interpretation

    return UpdateReceipt{
        group.id, policy_version_before, optimizer_step_before,
        handoff.optimizer.step, totals
    };
}
```

The receipt lets the owning sequence publish the consumed group, next policy
version, and already-prepared retry EMA together with the successful update
through existing checkpoint facilities. The loop does not calculate or update
the EMA. An optimizer failure must not publish success or permit reuse of the
group against potentially modified weights; stop and recover from a consistent
checkpoint. There is no implied rollback of an applied optimizer operation.

Action rows include sampled declarations, calls, READ requests, structural
markers, and the answer terminator. Prefixes, runtime observations, feedback,
padding, and forced tokens contribute context only. A retry can attend to the
initial attempt, but its advantage applies only to its own sampled actions.
The preparation adapter owns context-window handling and must preserve the
recorded prefixes and exactly-once action coverage across microbatches.

The fixed RL reduction replaces SFT supervised-row weighting. Microbatch count,
actual attempt count, and realized action length never introduce another mean.
The reference forward needs no graph; the policy forward must retain its graph
through exact categorical KL and backward. Existing auxiliary losses and the
requested clipping threshold still need the decisions recorded in the
[RL TODO](GRIM_SELF_CORRECTION_RL_TODO.md).

`training.config.rl_kl_coefficient` is an authored runtime setting, currently
`0.02` in `ai_config.json`. It is loaded, validated as finite and nonnegative,
preserved in the finalized config, and exposed by `rlTrainingHP()` for typed
config and snapshot callers. It is not model geometry and does not require a
new compiled-model schema field. The pseudocode reads the resolved value;
adaptive KL remains outside the pilot.

Resolve policy and frozen-reference artifacts through the existing model store
by model identity. Do not add an RL checkpoint-path configuration control.
Startup must pin the resolved reference checkpoint identity for the run and
validate it on resume, rather than reselecting a moving latest checkpoint for
each group. The separate frozen-reference loading lifecycle remains pending.
