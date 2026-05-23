//======================================================//
//  ConfigDump.cu
//  Implementation of dumpAllHyperparameters.
//
//  All fields are pushed into a single vector<{name,value}>
//  via the local DUMP(field) macro, then iterated in one
//  loop. Adding a new hyperparameter = one new DUMP() line.
//======================================================//

#include "ConfigDump.hpp"

// HyperParameters_GPU.hpp is the single entry point; it defines
// GRIM_HP_GPU_DEFINED_TRAINING_STRUCTS and transitively includes
// control/ai_config_paths.hpp in the correct order.
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIMText { namespace Training {

namespace {

// Stringify each scalar type with one place per kind (no per-field branching).
std::string fmt(bool v)              { return v ? "true" : "false"; }
std::string fmt(int v)               { return std::to_string(v); }
std::string fmt(int64_t v)           { return std::to_string(v); }
std::string fmt(unsigned long v)     { return std::to_string(v); }
#if defined(_WIN64)
std::string fmt(std::size_t v)       { return std::to_string(v); }
#endif
std::string fmt(float v) {
    std::ostringstream oss;
    oss << std::setprecision(8) << v;
    return oss.str();
}
std::string fmt(const std::string& v) { return v; }
std::string fmt(::GRIM::HyperParameters::PositionalEncodingType v) {
    using PET = ::GRIM::HyperParameters::PositionalEncodingType;
    switch (v) {
        case PET::UNSPECIFIED:
            throw std::runtime_error(
                "fmt(PositionalEncodingType): UNSPECIFIED is invalid - config must author positional encoding");
        case PET::NONE:       return "NONE";
        case PET::ALIBI:      return "ALIBI";
        case PET::ROPE:       return "ROPE";
        case PET::ALIBI_ROPE: return "ALIBI_ROPE";
    }
    throw std::runtime_error("fmt(PositionalEncodingType): unknown enum value");
}
std::string fmt(::GRIM::HyperParameters::ModelExecutionMode v) {
    using MEM = ::GRIM::HyperParameters::ModelExecutionMode;
    switch (v) {
        case MEM::TRAINING:  return "TRAINING";
        case MEM::INFERENCE: return "INFERENCE";
    }
    throw std::runtime_error("fmt(ModelExecutionMode): unknown enum value");
}
std::string fmt(::GRIM::HyperParameters::ParameterGroupPrecision v) {
    return ::GRIM::HyperParameters::parameterGroupPrecisionToString(v);
}
std::string fmt(::GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern v) {
    using HCP = ::GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern;
    switch (v) {
        case HCP::DISABLED:          return "DISABLED";
        case HCP::RANDOM_CENTERED:   return "RANDOM_CENTERED";
        case HCP::ORTHOGONAL_W277:   return "ORTHOGONAL_W277";
        case HCP::ALIGNED_W277:      return "ALIGNED_W277";
        case HCP::CONSTANT_UNIFORM:  return "CONSTANT_UNIFORM";
        case HCP::ZERO_MEAN_SINE:    return "ZERO_MEAN_SINE";
    }
    throw std::runtime_error("fmt(HardcodedPattern): unknown enum value");
}

std::string fmtStringMap(const std::map<std::string, std::string>& values) {
    if (values.empty()) {
        return "{}";
    }

    std::ostringstream oss;
    bool first = true;
    for (const auto& kv : values) {
        if (!first) {
            oss << ", ";
        }
        oss << kv.first << "=" << kv.second;
        first = false;
    }
    return oss.str();
}

} // namespace

void dumpAllHyperparameters(
    const GRIM::Config::TrainingHyperparameters& hp,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpOptions& opts,
    const ConfigDumpLogFn& log_fn)
{
    if (!log_fn) return;

    std::vector<std::pair<std::string, std::string>> rows;
    rows.reserve(320);

#define DUMP(field) rows.emplace_back(#field, fmt(hp.field))
#define DUMP_ARCH(field) rows.emplace_back("architecture." #field, fmt(hp.architecture.field))
#define DUMP_LOG_RECORDER(field) rows.emplace_back("log_recorder." #field, fmt(hp.log_recorder.field))
#define DUMP_LOG_RECORDER_LAYER(field) rows.emplace_back("log_recorder.layers." #field, fmt(hp.log_recorder.layers.field))
#define DUMP_TAPE_LOGGING(field) rows.emplace_back("tape_logging." #field, fmt(hp.tape_logging.field))
#define DUMP_PARAM_PRECISION(json_field, config_field) rows.emplace_back("precision.parameter_groups." json_field, fmt(hp.architecture.config_field))
#define SECTION(label) rows.emplace_back(std::string("---"), std::string(label))

    SECTION("Run selectors");
    DUMP(current_model_training);
    DUMP(current_curriculum);

    SECTION("Logging");
    DUMP_LOG_RECORDER(enabled);
    DUMP_LOG_RECORDER(default_level);
    rows.emplace_back("log_recorder.modules", fmtStringMap(hp.log_recorder.modules));
    DUMP_LOG_RECORDER_LAYER(embedding);
    DUMP_LOG_RECORDER_LAYER(rms_norm);
    DUMP_LOG_RECORDER_LAYER(attention);
    DUMP_LOG_RECORDER_LAYER(feed_forward);
    DUMP_LOG_RECORDER_LAYER(residual);
    DUMP_LOG_RECORDER_LAYER(encoding);
    DUMP_LOG_RECORDER_LAYER(serialization);
    DUMP_LOG_RECORDER_LAYER(execution_block);
    DUMP_TAPE_LOGGING(default_level);
    DUMP_TAPE_LOGGING(equation_csv_enabled);
    DUMP_TAPE_LOGGING(stderr_enabled);
    rows.emplace_back("tape_logging.initial_capacity", std::to_string(hp.tape_logging.initial_capacity));
    rows.emplace_back("tape_logging.group_overrides", fmtStringMap(hp.tape_logging.group_overrides));

    SECTION("Model architecture");
    DUMP_ARCH(d_model);
    DUMP_ARCH(num_layers);
    DUMP_ARCH(num_heads);
    DUMP_ARCH(num_kv_heads);
    DUMP_ARCH(head_dim);
    DUMP_ARCH(d_ff);
    DUMP_ARCH(max_seq_len);
    DUMP_ARCH(dropout_rate);
    DUMP_ARCH(attention_dropout);
    DUMP_ARCH(tie_embeddings);
    DUMP_ARCH(positional_encoding);
    DUMP_ARCH(use_gpu);
    DUMP_ARCH(use_flash_attention);
    DUMP_ARCH(min_seq_len_for_flash);
    DUMP_ARCH(rms_epsilon);
    DUMP_ARCH(causal_mask);
    DUMP_ARCH(use_pre_norm);
    DUMP_ARCH(fuse_qkv);
    DUMP_ARCH(use_simd);
    DUMP_ARCH(num_threads);
    DUMP_ARCH(use_bias);
    DUMP_ARCH(execution_mode);

    SECTION("Parameter precision");
    DUMP_PARAM_PRECISION("embedding", parameter_precision_embedding);
    DUMP_PARAM_PRECISION("lm_head", parameter_precision_lm_head);
    DUMP_PARAM_PRECISION("attention", parameter_precision_attention);
    DUMP_PARAM_PRECISION("ffn", parameter_precision_ffn);
    DUMP_PARAM_PRECISION("rmsnorm", parameter_precision_rmsnorm);
    DUMP_PARAM_PRECISION("scratchblock", parameter_precision_scratchblock);
    DUMP_PARAM_PRECISION("mtp", parameter_precision_mtp);
    DUMP_PARAM_PRECISION("reasoning_head", parameter_precision_reasoning_head);
    DUMP_PARAM_PRECISION("execution_block", parameter_precision_execution_block);
    DUMP_PARAM_PRECISION("slot_selector", parameter_precision_slot_selector);

    SECTION("Core training");
    DUMP(epochs);
    DUMP(seed);
    DUMP(batch_size);
    DUMP(gradient_accumulation_steps);
    DUMP(single_batch_overfit_enabled);
    DUMP(single_batch_overfit_max_steps);
    DUMP(batch_strategy);
    DUMP(learning_rate);
    DUMP(weight_decay);
    DUMP(grad_clip_norm);
    DUMP(per_token_grad_scale);
    DUMP(warmup_fraction);
    DUMP(warmup_steps);
    DUMP(cosine_decay_enabled);
    DUMP(cosine_warm_restarts);
    DUMP(cosine_decay_min_lr);
    DUMP(min_seq_valid_tokens);
    DUMP(log_interval);
    DUMP(atom_stats_interval);
    DUMP(atom_stats_max_seqs);
    DUMP(validation_interval);
    DUMP(checkpoint_interval);

    SECTION("Soft restart");
    DUMP(soft_restart_enabled);
    DUMP(soft_restart_loss_increase_threshold);
    DUMP(soft_restart_max_step_window);
    DUMP(soft_restart_cooldown_steps);

    SECTION("Auto stop");
    DUMP(auto_stop_enabled);
    DUMP(auto_stop_plateau_patience);
    DUMP(auto_stop_plateau_min_delta);
    DUMP(auto_stop_high_loss_threshold);
    DUMP(auto_stop_high_loss_patience);

    SECTION("Shuffle");
    DUMP(shuffle_train_enabled);
    DUMP(shuffle_train_epochs);

    SECTION("Telemetry control");
    DUMP(telemetry_control_enabled);
    DUMP(telemetry_spike_mild_threshold);
    DUMP(telemetry_spike_moderate_threshold);
    DUMP(telemetry_spike_severe_threshold);
    DUMP(telemetry_moderate_grad_scale);
    DUMP(telemetry_moderate_cooldown_extension);
    DUMP(telemetry_min_grad_for_nonzero_loss);
    DUMP(telemetry_loss_threshold_for_grad_check);
    DUMP(telemetry_max_consecutive_zero_grad_steps);
    DUMP(telemetry_seq_len_regime_change_threshold);
    DUMP(telemetry_regime_change_suppression_steps);
    DUMP(telemetry_volatility_damping_threshold);
    DUMP(telemetry_max_volatility_damping);
    DUMP(telemetry_gradient_decay_threshold);
    DUMP(telemetry_max_decay_boost);
    DUMP(telemetry_progress_boost_threshold);
    DUMP(telemetry_max_progress_boost);
    DUMP(telemetry_outlier_frequency_trigger);
    DUMP(telemetry_outlier_persistence_trigger);
    DUMP(telemetry_anchor_drift_sigma_multiplier);
    DUMP(telemetry_soft_restart_cooldown_steps);
    DUMP(telemetry_warmup_steps);
    DUMP(telemetry_baseline_stabilization_steps);
    DUMP(telemetry_verbose_logging);
    DUMP(telemetry_fail_loud_on_accumulation_bug);
    DUMP(telemetry_plateau_noise_enabled);
    DUMP(telemetry_plateau_noise_patience);
    DUMP(telemetry_plateau_noise_variance_threshold);
    DUMP(telemetry_plateau_noise_std);
    DUMP(telemetry_plateau_noise_proportional);
    DUMP(telemetry_plateau_noise_cooldown);
    DUMP(telemetry_plateau_noise_max_per_epoch);

    SECTION("Telemetry lattice");
    DUMP(telemetry_lattice_num_levels);
    DUMP(telemetry_lattice_num_streams);
    DUMP(telemetry_lattice_beta_mu);
    DUMP(telemetry_lattice_beta_a);
    DUMP(telemetry_lattice_beta_delta);
    DUMP(telemetry_lattice_beta_r);
    DUMP(telemetry_lattice_beta_run);
    DUMP(telemetry_lattice_beta_v);
    DUMP(telemetry_lattice_k_out0);
    DUMP(telemetry_lattice_alpha_v);
    DUMP(telemetry_lattice_epsilon);
    DUMP(telemetry_lattice_strict_mode);

    SECTION("Loss options");
    DUMP(loss_label_smoothing_enabled);
    DUMP(loss_label_smoothing_epsilon);
    DUMP(loss_focal_enabled);
    DUMP(loss_focal_gamma);
    DUMP(loss_focal_alpha);
    DUMP(loss_preference_enabled);
    DUMP(loss_preference_beta);
    DUMP(loss_distillation_enabled);
    DUMP(loss_distillation_temperature);
    DUMP(loss_distillation_lambda);
    DUMP(loss_masking_enabled);
    DUMP(loss_masking_tag);
    DUMP(loss_entropy_reg_enabled);
    DUMP(loss_entropy_reg_lambda);
    DUMP(loss_class_balanced_enabled);
    DUMP(loss_class_balanced_beta);

    SECTION("LM head centering");
    DUMP(lm_head_centering_enabled);
    DUMP_ARCH(lm_head_center_hidden_states);
    DUMP_ARCH(freeze_learned_rms_gammas);
    DUMP_ARCH(center_logits);
    DUMP_ARCH(center_encoder_residuals);
    DUMP_ARCH(project_out_pc1);
    DUMP_ARCH(pc1_power_iters);

    SECTION("LayerScale / QK-norm");
    DUMP_ARCH(use_layer_scale);
    DUMP_ARCH(layer_scale_init);
    DUMP_ARCH(qk_norm_enabled);

    SECTION("Hardcoded hidden states diag");
    DUMP_ARCH(hardcoded_hidden_pattern);
    DUMP_ARCH(hardcoded_log_every_n_batches);

    SECTION("Embedding freeze");
    DUMP(embedding_freeze_enabled);
    DUMP(embedding_freeze_after_step);

    SECTION("Optimizer");
    DUMP(optimizer_kind);
    DUMP(optimizer_beta1);
    DUMP(optimizer_beta2);
    DUMP(optimizer_epsilon);

    SECTION("Stability overrides");
    DUMP(stability_overrides_enabled);
    DUMP(stability_override_batch_size);
    DUMP(stability_override_max_seq_len);
    DUMP(stability_override_clip_per_token);

    SECTION("Scratch blocks");
    DUMP(scratch_blocks_enabled);
    DUMP(scratch_max_tokens_per_block);
    DUMP(scratch_num_blocks);
    DUMP(scratch_write_combined);

    SECTION("ScratchBlock reasoning");
    DUMP_ARCH(use_scratch_block);
    DUMP_ARCH(scratch_block_atom_embedding_dim);
    DUMP_ARCH(scratch_block_max_atoms);
    DUMP_ARCH(scratch_block_atom_scale);

    SECTION("ReasoningHead");
    DUMP_ARCH(reasoning_head_enabled);
    DUMP_ARCH(reasoning_num_ops);

    SECTION("ExecutionBlock");
    DUMP_ARCH(execution_block_enabled);
    DUMP_ARCH(scratch_block_execution_first_type_only);
    DUMP_ARCH(execution_block_layer);
    DUMP_ARCH(execution_block_num_ops);
    DUMP_ARCH(execution_block_num_slots);
    DUMP_ARCH(execution_block_num_steps);
    DUMP_ARCH(execution_block_d_key);
    DUMP_ARCH(execution_block_d_type);
    DUMP_ARCH(execution_block_cross_attn_head_dim);
    DUMP_ARCH(execution_block_cross_attn_topk);
    DUMP_ARCH(execution_block_usage_decay);
    DUMP_ARCH(execution_block_diversity_kappa);
    DUMP_ARCH(execution_block_temp_start);
    DUMP_ARCH(execution_block_temp_end);
    DUMP_ARCH(execution_block_temp_schedule);
    DUMP_ARCH(execution_block_entropy_weight);
    DUMP_ARCH(step_x_multiplier);
    DUMP_ARCH(step_y_multiplier);
    DUMP_ARCH(step_y_overrides_x);
    DUMP_ARCH(entropy_aux_weight);
    DUMP_ARCH(value_match_epsilon);
    DUMP_ARCH(final_slot_consistency_weight);
    DUMP_ARCH(execution_block_transition_hard_threshold);
    DUMP_ARCH(execution_block_gate_warmup_steps);
    DUMP_ARCH(execution_block_causal_w1_transition);
    DUMP_ARCH(div_invalid_penalty_weight);
    DUMP_ARCH(div_magnitude_penalty_weight);
    DUMP_ARCH(arg_reinforce_weight);
    DUMP_ARCH(arg_reinforce_baseline_decay);
    DUMP_ARCH(structured_ce_enabled);
    DUMP_ARCH(structured_ce_weight);
    DUMP_ARCH(selector_enabled);
    DUMP_ARCH(selector_d_selector);
    DUMP_ARCH(selector_selection_margin);
    DUMP_ARCH(selector_supervision_weight);

    SECTION("CUDA execution mode");
    DUMP(single_stream_mode);
    DUMP(disable_async_frees);
    DUMP(synchronize_after_kernels);

    SECTION("Multi-token prediction");
    DUMP_ARCH(mtp_enabled);
    DUMP_ARCH(mtp_k);
    DUMP_ARCH(mtp_alpha);
    DUMP_ARCH(mtp_alpha_warmup_steps);
    DUMP(mtp_log_ratio_monitor);

    SECTION("Prediction comparison");
    DUMP(prediction_comparison_enabled);
    DUMP(prediction_comparison_interval);
    DUMP(prediction_comparison_top_k);
    DUMP(prediction_comparison_max_positions);
    DUMP(prediction_comparison_log_path);

    SECTION("Logit update trace");
    DUMP(logit_update_trace_enabled);
    DUMP(logit_update_trace_interval);

    SECTION("Attention diagnostics");
    DUMP(attention_diag_enabled);
    DUMP(attention_diag_layer);
    DUMP(attention_diag_head);

    SECTION("Tokenizer flags");
    DUMP(tokenizer_enable_scratch_block_reasoning);
    DUMP(tokenizer_detect_numbers);

    // Post-harmonize derived schedule (computed in HyperParameters_GPU.hpp from
    // batch_size / epochs / sequence_count ratios). These are the EFFECTIVE
    // values used by the training loop, not the raw JSON inputs.
    if (derived) {
        rows.emplace_back(std::string("---"), std::string("Derived schedule (post-harmonize)"));
        rows.emplace_back("derived.batches_per_epoch",   fmt(derived->batches_per_epoch));
        rows.emplace_back("derived.total_training_steps", fmt(derived->total_training_steps));
        rows.emplace_back("derived.safe_last_step",       fmt(derived->safe_last_step));
    }

#undef DUMP
#undef DUMP_ARCH
#undef DUMP_LOG_RECORDER
#undef DUMP_LOG_RECORDER_LAYER
#undef DUMP_TAPE_LOGGING
#undef DUMP_PARAM_PRECISION
#undef SECTION

    //==================================================//
    // Visual emission pass.
    //   - Banner rule with entry count + subtitle
    //   - Section divider lines (─── label ─────)
    //   - Field lines with auto-aligned name column
    //   - Footer rule
    // Adding a new field above never touches this block.
    //==================================================//

    // Count "real" fields (rows tagged "---" are section markers, not fields).
    std::size_t field_count = 0;
    std::size_t longest_name = 0;
    for (const auto& kv : rows) {
        if (kv.first == "---") continue;
        ++field_count;
        if (kv.first.size() > longest_name) longest_name = kv.first.size();
    }

    const std::size_t name_w =
        (opts.name_col_width > 0) ? opts.name_col_width : longest_name;

    auto rule = [](char ch, std::size_t n) {
        return std::string(n, ch);
    };

    // Banner: ===== Hyperparameters (post-policy, N entries) =====
    {
        std::ostringstream title;
        title << " " << opts.banner_label
              << " (" << opts.banner_subtitle
              << ", " << field_count << " entries) ";
        const std::string t = title.str();
        std::size_t pad = (opts.banner_width > t.size())
                              ? (opts.banner_width - t.size()) / 2
                              : 0;
        std::string banner = rule('=', pad) + t
                             + rule('=', opts.banner_width - pad - t.size());
        log_fn(opts.prefix + rule('=', opts.banner_width));
        log_fn(opts.prefix + banner);
        log_fn(opts.prefix + rule('=', opts.banner_width));
    }

    // Optional data-stats block — same prefix/width as the hyperparameters
    // table so it visually nests under the banner.
    if (data_stats) {
        DataStatsDumpOptions ds_opts;
        ds_opts.prefix         = opts.prefix;
        ds_opts.banner_width   = opts.section_width;
        ds_opts.name_col_width = opts.name_col_width;
        ds_opts.show_footer    = false;
        dumpDataStats(*data_stats, ds_opts, log_fn);
    }

    for (const auto& kv : rows) {
        if (kv.first == "---") {
            if (!opts.show_sections) continue;
            std::ostringstream line;
            line << "--- " << kv.second << " ";
            std::string s = line.str();
            std::size_t pad = (opts.section_width > s.size())
                                  ? (opts.section_width - s.size())
                                  : 0;
            log_fn(opts.prefix + s + rule('-', pad));
        } else {
            std::ostringstream line;
            line << "  " << std::left << std::setw(static_cast<int>(name_w))
                 << kv.first << " = " << kv.second;
            log_fn(opts.prefix + line.str());
        }
    }

    if (opts.show_footer) {
        log_fn(opts.prefix + rule('=', opts.banner_width));
    }
}

} } // namespace GRIMText::Training
