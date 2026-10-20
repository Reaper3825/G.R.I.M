#include "Phase2_RLTrainingLoop.hpp"

#include <stdexcept>

namespace GRIMText::Training {

bool executePhase2RLTrainingLoop(
    TrainingContext& ctx,
    const PreparedRLEpisodeGroup& group)
{
    // PSEUDOCODE ONLY. Handoff types and RL forward/loss/update adapters must
    // be implemented before replacing the explicit failure below. Names inside
    // this block describe operations, not existing callable APIs.
    /*
    require(ctx.training_stage == TrainingStage::RL);
    require(group.complete && !group.consumed);
    require(group.task_count == 16 && group.episode_count == 64);
    require(each_task_has_four_complete_valid_episodes(group));
    require(group.policy_version == ctx.policy_version);
    require(group.reference_identity == ctx.frozen_reference.identity);
    require(all_policy_parameters_registered_and_trainable(ctx));
    require(reference_is_frozen_and_excluded_from_optimizer(ctx));
    require(auxiliary_objective_and_clipping_decisions_are_resolved(ctx));

    // Validate the complete prepared plan before accumulating any gradients.
    require(exactly_once_sampled_action_coverage(group));
    require(recorded_prefixes_and_atom_channels_are_preserved(group));
    require(advantages_are_detached_and_attempt_local(group));
    require(no_fabricated_retries_or_infrastructure_invalid_episodes(group));
    lock_policy_against_other_updates_for_this_group(ctx);

    const denominator = 256 * 64;
    const kl_coefficient = rlTrainingHP(ctx.config).kl_coefficient;
    const optimizer_step_before = ctx.optimizer.step;
    totals = empty_rl_metrics();
    begin_gradient_window_with_zeroed_policy_gradients(ctx);

    try {
        for (const microbatch : group.microbatches) {
            scoped_microbatch_resources resources;
            upload_existing_batch_payload(microbatch.payload, resources);

            if (microbatch.sampled_action_rows.empty()) {
                continue;  // Zero contribution; keep the fixed denominator.
            }

            // Replay recorded history only. No generation, tool calls,
            // grading, retry decisions, or variable-store mutations here.
            reference_logits = shared_forward(
                ctx.frozen_reference, microbatch.payload,
                connect_parameter_graph = false,
                dropout = false, use_generation_kv_cache = false);
            reference_logits = detach(reference_logits);

            policy_logits = shared_forward(
                ctx.policy, microbatch.payload,
                connect_parameter_graph = true,
                dropout = false, use_generation_kv_cache = false);

            // Both distributions match rollout: categorical, temperature 1,
            // identical fixed legal support, no truncation/penalties/biases.
            log_p = fp32_log_softmax(policy_logits, microbatch.legal_mask);
            log_q = fp32_log_softmax(reference_logits, microbatch.legal_mask);
            pg_sum = connected_zero(policy_logits);
            kl_sum = connected_zero(policy_logits);

            for (const action : microbatch.sampled_action_rows) {
                row = action.prediction_row;  // Row preceding sampled token.
                target = action.sampled_token_id;
                advantage = action.detached_attempt_advantage;

                pg_sum += -advantage * log_p[row, target];
                for (const v : microbatch.legal_output_ids(row)) {
                    kl_sum += exp(log_p[row, v]) *
                              (log_p[row, v] - log_q[row, v]);
                }
            }

            // KL remains differentiable through the current distribution,
            // including when advantage is zero. Never detach exp(log_p).
            loss = (pg_sum + kl_coefficient * kl_sum) / denominator;
            loss += explicitly_selected_auxiliary_loss_if_any(ctx, microbatch);
            require_finite(loss);
            backward(loss, accumulate_policy_gradients = true);
            totals.add_detached(pg_sum, kl_sum, microbatch.action_count);

            // Destroy this microbatch's graph/device owners before advancing.
        }
        require(all_group_microbatches_finished());
        require_policy_gradients_finite(ctx);
    } catch (...) {
        discard_partial_gradient_window(ctx);
        throw;  // No optimizer step or successful group publication.
    }

    // Group completion owns this boundary, not the SFT accumulation-step count.
    // Reuse existing clipping/RAdamW implementations without introducing their
    // SFT token weighting or an additional microbatch/attempt/token mean.
    lr = existing_lr_schedule(ctx, optimizer_step_before);
    clip_once_using_resolved_existing_clipping_metric(ctx);
    launch_existing_radamw_update_once(ctx, lr, optimizer_step_before);
    synchronize_and_check_updated_policy_finite(ctx);
    complete_existing_optimizer_window_bookkeeping_once(ctx);

    totals.policy_loss = totals.pg_sum / denominator;
    totals.weighted_kl_loss = kl_coefficient * totals.kl_sum / denominator;
    totals.mean_kl_per_action = totals.action_count > 0
        ? optional(totals.kl_sum / totals.action_count) : unavailable;
    log_rl_metrics_without_ce_or_perplexity(totals);
    return true;
    */

    // Caller publishes consumed-group / next-policy-version / prepared EMA
    // state only after success. A failure after optimizer dispatch is fatal:
    // recover from a consistent checkpoint; do not assume rollback or reuse
    // the group against possibly modified weights.
    (void)ctx;
    (void)group;
    throw std::runtime_error(
        "Phase2_RLTrainingLoop is a pseudocode scaffold; "
        "prepared episode-group handoff and RL forward/backward are not implemented");
}

} // namespace GRIMText::Training
