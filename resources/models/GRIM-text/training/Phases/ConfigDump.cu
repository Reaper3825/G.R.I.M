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

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#else
#  include <sys/resource.h>
#  include <unistd.h>
#endif

namespace GRIMText { namespace Training {

namespace {

// Stringify each scalar type with one place per kind (no per-field branching).
std::string fmt(bool v)              { return v ? "true" : "false"; }
std::string fmt(int v)               { return std::to_string(v); }
std::string fmt(int64_t v)           { return std::to_string(v); }
std::string fmt(unsigned long v)     { return std::to_string(v); }
std::string fmt(unsigned long long v){ return std::to_string(v); }
std::string fmt(float v) {
    std::ostringstream oss;
    oss << std::setprecision(8) << v;
    return oss.str();
}
std::string fmt(double v) {
    std::ostringstream oss;
    oss << std::setprecision(10) << v;
    return oss.str();
}
std::string fmt(const std::string& v) { return v; }
std::string fmt(const char* v)        { return v ? v : "(null)"; }

// Format byte counts as "<bytes> B (<MiB> MiB / <GiB> GiB)" for readability.
std::string fmtBytes(std::size_t bytes) {
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    const double gib = mib / 1024.0;
    std::ostringstream oss;
    oss << bytes << " B (" << std::fixed << std::setprecision(2)
        << mib << " MiB, " << std::setprecision(3) << gib << " GiB)";
    return oss.str();
}

std::string fmtPercent(double frac) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (frac * 100.0) << "%";
    return oss.str();
}

// Resident set size of the current process (host RAM actually held).
// Returns 0 if the platform query fails.
std::size_t queryHostResidentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<std::size_t>(pmc.WorkingSetSize);
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<std::size_t>(info.resident_size);
    }
    return 0;
#else
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        // ru_maxrss is KB on Linux, bytes on macOS (handled above).
        return static_cast<std::size_t>(ru.ru_maxrss) * 1024;
    }
    return 0;
#endif
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
    rows.reserve(256);

#define DUMP(field) rows.emplace_back(#field, fmt(hp.field))
#define SECTION(label) rows.emplace_back(std::string("---"), std::string(label))

    SECTION("Run selectors");
    DUMP(current_model_training);
    DUMP(current_curriculum);

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
    DUMP(max_seq_len);
    DUMP(min_seq_valid_tokens);
    DUMP(log_interval);
    DUMP(atom_stats_interval);
    DUMP(atom_stats_max_seqs);
    DUMP(validation_interval);
    DUMP(checkpoint_interval);
    DUMP(use_gpu);
    DUMP(use_flash_attention);
    DUMP(min_seq_len_for_flash);

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

    SECTION("Guess aux");
    DUMP(guess_aux_enabled);
    DUMP(guess_aux_lambda);
    DUMP(guess_aux_min_confidence);

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
    DUMP(lm_head_center_hidden_states);
    DUMP(lm_head_freeze_final_rms_gamma);
    DUMP(center_logits);
    DUMP(center_encoder_residuals);
    DUMP(project_out_pc1);
    DUMP(pc1_power_iters);

    SECTION("LayerScale / QK-norm");
    DUMP(use_layer_scale);
    DUMP(layer_scale_init);
    DUMP(qk_norm_enabled);

    SECTION("Hardcoded hidden states diag");
    DUMP(hardcoded_hidden_pattern);
    DUMP(hardcoded_log_every_n_batches);

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
    DUMP(stability_override_lr_min);

    SECTION("Scratch blocks");
    DUMP(scratch_blocks_enabled);
    DUMP(scratch_max_tokens_per_block);
    DUMP(scratch_num_blocks);
    DUMP(scratch_write_combined);

    SECTION("ScratchBlock reasoning");
    DUMP(scratch_block_reasoning_enabled);
    DUMP(scratch_block_reasoning_atom_embedding_dim);
    DUMP(scratch_block_reasoning_max_atoms);
    DUMP(scratch_block_reasoning_atom_scale);

    SECTION("ExecutionBlock");
    DUMP(execution_block_enabled);
    DUMP(scratch_block_execution_first_type_only);
    DUMP(execution_block_layer);
    DUMP(execution_block_num_ops);
    DUMP(execution_block_num_slots);
    DUMP(execution_block_num_steps);
    DUMP(execution_block_d_key);
    DUMP(execution_block_d_type);
    DUMP(execution_block_cross_attn_head_dim);
    DUMP(execution_block_cross_attn_topk);
    DUMP(execution_block_usage_decay);
    DUMP(execution_block_diversity_kappa);
    DUMP(execution_block_temp_start);
    DUMP(execution_block_temp_end);
    DUMP(execution_block_temp_schedule);
    DUMP(execution_block_entropy_weight);
    DUMP(execution_step_x_multiplier);
    DUMP(execution_step_y_multiplier);
    DUMP(execution_step_y_overrides_x);
    DUMP(execution_entropy_aux_weight);
    DUMP(execution_value_match_epsilon);
    DUMP(execution_final_slot_consistency_weight);
    DUMP(execution_block_transition_hard_threshold);
    DUMP(execution_block_gate_warmup_steps);
    DUMP(execution_block_causal_w1_transition);
    DUMP(execution_div_invalid_penalty_weight);
    DUMP(execution_div_magnitude_penalty_weight);
    DUMP(execution_arg_reinforce_weight);
    DUMP(execution_arg_reinforce_baseline_decay);
    DUMP(structured_ce_enabled);
    DUMP(structured_ce_weight);
    DUMP(selector_enabled);
    DUMP(selector_d_selector);
    DUMP(selector_selection_margin);
    DUMP(selector_supervision_weight);

    SECTION("Activation quantization");
    DUMP(activation_quantization_enabled);
    DUMP(activation_quantization_apply_to_embeddings);
    DUMP(activation_quantization_apply_to_encoder_outputs);
    DUMP(activation_quantization_apply_to_layer_caches);
    DUMP(activation_quantization_apply_to_qkv_cache);
    DUMP(activation_quantization_apply_to_logits);
    DUMP(activation_quantization_scale);
    DUMP(activation_quantization_clip_min);
    DUMP(activation_quantization_clip_max);
    DUMP(activation_quantization_zero_point);
    DUMP(activation_quantization_symmetric);

    SECTION("CUDA execution mode");
    DUMP(single_stream_mode);
    DUMP(disable_async_frees);
    DUMP(synchronize_after_kernels);

    SECTION("Multi-token prediction");
    DUMP(mtp_enabled);
    DUMP(mtp_k);
    DUMP(mtp_alpha);
    DUMP(mtp_alpha_warmup_steps);
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

    // Memory snapshot at config-dump time. Useful to know the headroom
    // before model allocation, and to see how much was already consumed by
    // tokenizer/data loading. cudaMemGetInfo failure is logged but does not
    // abort the dump (Rule 20: this is a diagnostic, not a load-bearing read).
    {
        rows.emplace_back(std::string("---"), std::string("Memory"));

        std::size_t gpu_free = 0, gpu_total = 0;
        cudaError_t mem_err = cudaMemGetInfo(&gpu_free, &gpu_total);
        if (mem_err == cudaSuccess) {
            const std::size_t gpu_used = (gpu_total >= gpu_free) ? (gpu_total - gpu_free) : 0;
            const double used_frac = (gpu_total > 0)
                ? static_cast<double>(gpu_used) / static_cast<double>(gpu_total) : 0.0;
            const double free_frac = (gpu_total > 0)
                ? static_cast<double>(gpu_free) / static_cast<double>(gpu_total) : 0.0;

            int device_id = -1;
            cudaGetDevice(&device_id);
            cudaDeviceProp prop{};
            const bool prop_ok = (device_id >= 0)
                && (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess);

            rows.emplace_back("memory.gpu.device_id",   fmt(device_id));
            rows.emplace_back("memory.gpu.device_name", fmt(prop_ok ? prop.name : "(unknown)"));
            rows.emplace_back("memory.gpu.total",       fmtBytes(gpu_total));
            rows.emplace_back("memory.gpu.used",        fmtBytes(gpu_used));
            rows.emplace_back("memory.gpu.free",        fmtBytes(gpu_free));
            rows.emplace_back("memory.gpu.used_pct",    fmtPercent(used_frac));
            rows.emplace_back("memory.gpu.free_pct",    fmtPercent(free_frac));

            // Diagnostic flag — surface a warning row when headroom is tight
            // BEFORE model allocation. Anything <20% free at this stage means
            // model + optimizer states + activation workspace will OOM.
            const char* status = "ok";
            if (used_frac >= 0.95) status = "CRITICAL (<5% free)";
            else if (used_frac >= 0.80) status = "WARNING (<20% free)";
            else if (used_frac >= 0.50) status = "elevated (<50% free)";
            rows.emplace_back("memory.gpu.status", fmt(status));
        } else {
            rows.emplace_back("memory.gpu.error",
                fmt(std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(mem_err)));
        }

        const std::size_t host_rss = queryHostResidentBytes();
        if (host_rss > 0) {
            rows.emplace_back("memory.host.rss", fmtBytes(host_rss));
        } else {
            rows.emplace_back("memory.host.rss", fmt("(unavailable)"));
        }
    }

#undef DUMP
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
