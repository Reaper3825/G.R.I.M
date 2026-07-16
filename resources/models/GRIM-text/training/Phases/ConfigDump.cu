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
#include <initializer_list>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIMText { namespace Training {

namespace {

using DumpRow = std::pair<std::string, std::string>;
using DumpSection = std::pair<std::string, std::vector<DumpRow>>;

// Stringify each scalar type with one place per kind (no per-field branching).
std::string fmt(bool v)          { return v ? "true" : "false"; }
std::string fmt(int v)           { return std::to_string(v); }
std::string fmt(int64_t v)       { return std::to_string(v); }
std::string fmt(unsigned long v) { return std::to_string(v); }
#if defined(_WIN64)
std::string fmt(std::size_t v)   { return std::to_string(v); }
#endif
std::string fmt(double v) {
    std::ostringstream oss;
    oss << std::setprecision(8) << v;
    return oss.str();
}
std::string fmt(float v)               { return fmt(static_cast<double>(v)); }
std::string fmt(const std::string& v) { return v; }

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

bool startsWith(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool isOneOf(const std::string& key, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (key == name) {
            return true;
        }
    }
    return false;
}

std::string formatJsonValue(const nlohmann::json& value);

std::string formatJsonObject(const nlohmann::json& value) {
    if (!value.is_object()) {
        throw std::runtime_error("formatJsonObject: value is not an object");
    }
    if (value.empty()) {
        return "{}";
    }

    bool all_string_values = true;
    for (const auto& item : value.items()) {
        if (!item.value().is_string()) {
            all_string_values = false;
            break;
        }
    }

    if (all_string_values) {
        std::map<std::string, std::string> string_map;
        for (const auto& item : value.items()) {
            string_map.emplace(item.key(), item.value().get<std::string>());
        }
        return fmtStringMap(string_map);
    }

    std::ostringstream oss;
    bool first = true;
    for (const auto& item : value.items()) {
        if (!first) {
            oss << ", ";
        }
        oss << item.key() << "=" << formatJsonValue(item.value());
        first = false;
    }
    return oss.str();
}

std::string formatJsonValue(const nlohmann::json& value) {
    if (value.is_null()) {
        return "null";
    }
    if (value.is_boolean()) {
        return fmt(value.get<bool>());
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<int64_t>());
    }
    if (value.is_number_unsigned()) {
        return std::to_string(value.get<uint64_t>());
    }
    if (value.is_number_float()) {
        return fmt(value.get<double>());
    }
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_object()) {
        return formatJsonObject(value);
    }
    return value.dump();
}

const char* classifyConfigSection(const std::string& key) {
    if (startsWith(key, "current_") || key == "training_curriculum") {
        return "Run selectors";
    }
    if (startsWith(key, "grim_text_")) {
        return "Paths";
    }
    if (startsWith(key, "log_recorder_")) {
        return "Log recorder";
    }
    if (startsWith(key, "logging_")) {
        return "Tape logging";
    }
    if (startsWith(key, "parameter_precision_")) {
        return "Parameter precision";
    }
    if (startsWith(key, "soft_restart_")) {
        return "Soft restart";
    }
    if (startsWith(key, "auto_stop_")) {
        return "Auto stop";
    }
    if (startsWith(key, "shuffle_train_")) {
        return "Shuffle";
    }
    if (startsWith(key, "telemetry_lattice_")) {
        return "Telemetry lattice";
    }
    if (startsWith(key, "telemetry_")) {
        return "Telemetry control";
    }
    if (startsWith(key, "loss_")) {
        return "Loss options";
    }
    if (startsWith(key, "generation_")) {
        return "Generation";
    }
    if (startsWith(key, "prediction_comparison_")) {
        return "Prediction comparison";
    }
    if (startsWith(key, "logit_update_trace_")) {
        return "Logit update trace";
    }
    if (startsWith(key, "attention_diag_")) {
        return "Attention diagnostics";
    }
    if (startsWith(key, "tokenizer_") || key == "force_rebuild_vocab" ||
        key == "clear_merged_cache_on_merge" || key == "subprocess_tokenizer_only_mode") {
        return "Tokenizer";
    }
    if (startsWith(key, "optimizer_") || key == "use_depth_aware_upsilon") {
        return "Optimizer";
    }
    if (startsWith(key, "embedding_freeze_")) {
        return "Embedding freeze";
    }
    if (startsWith(key, "stability_")) {
        return "Stability overrides";
    }
    if (startsWith(key, "scratch_block_")) {
        return "ScratchBlock reasoning";
    }
    if (startsWith(key, "scratch_")) {
        return "Scratch blocks";
    }
    if (startsWith(key, "execution_block_") ||
        isOneOf(key, {
            "step_x_multiplier",
            "step_y_multiplier",
            "step_y_overrides_x",
            "entropy_aux_weight",
            "value_match_epsilon",
            "final_slot_consistency_weight",
            "div_invalid_penalty_weight",
            "div_magnitude_penalty_weight",
            "arg_reinforce_weight",
            "arg_reinforce_baseline_decay",
            "structured_ce_enabled",
            "structured_ce_weight"
        })) {
        return "ExecutionBlock";
    }
    if (isOneOf(key, {"single_stream_mode", "disable_async_frees", "synchronize_after_kernels"})) {
        return "CUDA execution mode";
    }
    if (startsWith(key, "lm_head_") ||
        isOneOf(key, {
            "freeze_learned_rms_gammas",
            "center_logits",
            "center_encoder_residuals",
            "project_out_pc1",
            "pc1_power_iters"
        })) {
        return "LM head centering";
    }
    if (isOneOf(key, {"use_layer_scale", "layer_scale_init", "qk_norm_enabled", "attention_off_by_one"})) {
        return "LayerScale / QK-norm / attn off-by-one";
    }
    if (startsWith(key, "hardcoded_hidden_") || startsWith(key, "hardcoded_log_")) {
        return "Hardcoded hidden states diag";
    }
    if (startsWith(key, "pbm_") || startsWith(key, "rope_") || startsWith(key, "alibi_") ||
        isOneOf(key, {
            "d_model",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "heads_per_kv_group",
            "kv_dim",
            "qkv_dim",
            "d_ff",
            "max_seq_len",
            "max_cached_seq_len",
            "max_tokens_per_batch",
            "dropout_rate",
            "embedding_scale",
            "attention_dropout",
            "tie_embeddings",
            "positional_encoding",
            "use_rope",
            "use_alibi",
            "use_gpu",
            "use_flash_attention",
            "min_seq_len_for_flash",
            "rms_epsilon",
            "causal_mask",
            "use_pre_norm",
            "fuse_qkv",
            "use_simd",
            "num_threads",
            "use_bias",
            "execution_mode",
            "rotary_dim",
            "is_gqa",
            "residual_projection_init_gain",
            "vocab_size"
        })) {
        return "Model config";
    }
    if (isOneOf(key, {
            "epochs",
            "seed",
            "batch_size",
            "gradient_accumulation_steps",
            "single_batch_overfit_enabled",
            "single_batch_overfit_max_steps",
            "batch_strategy",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "grad_clip_norm",
            "grad_clip_enabled",
            "effective_per_token_grad_limit",
            "per_token_grad_scale",
            "warmup_fraction",
            "warmup_steps",
            "cosine_decay_enabled",
            "cosine_warm_restarts",
            "cosine_decay_min_lr",
            "min_seq_valid_tokens",
            "log_interval",
            "atom_stats_interval",
            "atom_stats_max_seqs",
            "validation_interval",
            "checkpoint_interval"
        })) {
        return "Core training";
    }
    return "Additional config";
}

std::vector<DumpSection> collectSnapshotSections(
    const GRIM::Config::AiConfigSnapshot& snapshot)
{
    static const std::vector<std::string> section_order = {
        "Run selectors",
        "Paths",
        "Log recorder",
        "Tape logging",
        "Model config",
        "Parameter precision",
        "Core training",
        "Soft restart",
        "Auto stop",
        "Shuffle",
        "Telemetry control",
        "Telemetry lattice",
        "Loss options",
        "Generation",
        "LM head centering",
        "LayerScale / QK-norm",
        "Hardcoded hidden states diag",
        "Embedding freeze",
        "Optimizer",
        "Stability overrides",
        "Scratch blocks",
        "ScratchBlock reasoning",
        "ExecutionBlock",
        "Decode-time selector",
        "CUDA execution mode",
        "Multi-token prediction",
        "Prediction comparison",
        "Logit update trace",
        "Attention diagnostics",
        "Tokenizer",
        "Additional config"
    };

    std::map<std::string, std::vector<DumpRow>> grouped_rows;
    const auto& config = GRIM::HyperParameters::snapshotTrainingConfig(snapshot);
    for (const auto& entry : config.items()) {
        grouped_rows[classifyConfigSection(entry.key())].emplace_back(
            entry.key(), formatJsonValue(entry.value()));
    }

    std::vector<DumpSection> sections;
    sections.reserve(section_order.size());
    for (const auto& section_name : section_order) {
        auto it = grouped_rows.find(section_name);
        if (it == grouped_rows.end() || it->second.empty()) {
            continue;
        }
        sections.emplace_back(it->first, std::move(it->second));
    }
    return sections;
}

void emitSnapshotDump(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpOptions& opts,
    const ConfigDumpLogFn& log_fn)
{
    std::vector<std::pair<std::string, std::string>> rows;
    const auto sections = collectSnapshotSections(snapshot);

    std::size_t reserved_rows = 0;
    for (const auto& section : sections) {
        reserved_rows += section.second.size();
        if (opts.show_sections) {
            ++reserved_rows;
        }
    }
    if (derived) {
        reserved_rows += opts.show_sections ? 4 : 3;
    }
    rows.reserve(reserved_rows);

    for (const auto& section : sections) {
        if (opts.show_sections) {
            rows.emplace_back(std::string("---"), section.first);
        }
        rows.insert(rows.end(), section.second.begin(), section.second.end());
    }

    if (derived) {
        if (opts.show_sections) {
            rows.emplace_back(std::string("---"), std::string("Derived schedule (post-harmonize)"));
        }
        rows.emplace_back("derived.batches_per_epoch", fmt(derived->batches_per_epoch));
        rows.emplace_back("derived.total_training_steps", fmt(derived->total_training_steps));
        rows.emplace_back("derived.safe_last_step", fmt(derived->safe_last_step));
    }

    //==================================================//
    // Visual emission pass.
    //   - Banner rule with entry count + subtitle
    //   - Section divider lines (─── label ─────)
    //   - Field lines with auto-aligned name column
    //   - Footer rule
    // Snapshot iteration above is now the single dump registry source.
    //==================================================//

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

} // namespace

void dumpAllHyperparameters(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpOptions& opts,
    const ConfigDumpLogFn& log_fn)
{
    if (!log_fn) return;
    emitSnapshotDump(snapshot, derived, data_stats, opts, log_fn);
}

} } // namespace GRIMText::Training
