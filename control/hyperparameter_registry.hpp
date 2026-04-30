#pragma once
//======================================================//
// HyperparameterRegistry — enumerates TrainingHyperparameters
// fields with display metadata for UI browsing.
//
// Each entry carries: JSON key, human label, category tag,
// type discriminator, and a typed pointer into the live
// TrainingHyperparameters struct so the UI can read (and
// optionally write) values directly.
//
// Categories map to logical feature groups for filtering:
//   Core, Optimizer, SoftRestart, AutoStop,
//   GuessAux, Shuffle, Telemetry, Loss, LMHead, Attention,
//   LayerScale, ScratchBlock, ExecutionBlock, ActivationQuant,
//   CUDA, MTP, Diagnostics
//======================================================//

#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include "../resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM {
namespace Config {

// ─────────────────────────────────────────────────────────
//  HyperparamType — value discriminator
// ─────────────────────────────────────────────────────────
enum class HyperparamType : uint8_t {
    Bool   = 0,
    Int    = 1,
    Int64  = 2,
    Float  = 3,
    String = 4,
    SizeT  = 5
};

// ─────────────────────────────────────────────────────────
//  HyperparamEntry — one registered parameter
// ─────────────────────────────────────────────────────────
struct HyperparamEntry {
    std::string key;           // JSON key (e.g. "learning_rate")
    std::string display_name;  // Human label (e.g. "Learning Rate")
    std::string category;      // Filter category (e.g. "Core")

    HyperparamType type = HyperparamType::Int;

    // Typed pointer into the live struct — exactly one is valid per type.
    bool*        ptr_bool   = nullptr;
    int*         ptr_int    = nullptr;
    int64_t*     ptr_int64  = nullptr;
    float*       ptr_float  = nullptr;
    std::string* ptr_string = nullptr;
    size_t*      ptr_sizet  = nullptr;

    // Display helpers
    std::string valueAsString() const {
        switch (type) {
            case HyperparamType::Bool:   return ptr_bool   ? (*ptr_bool ? "true" : "false") : "?";
            case HyperparamType::Int:    return ptr_int    ? std::to_string(*ptr_int)    : "?";
            case HyperparamType::Int64:  return ptr_int64  ? std::to_string(*ptr_int64)  : "?";
            case HyperparamType::Float:  {
                if (!ptr_float) return "?";
                char buf[32];
                snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(*ptr_float));
                return buf;
            }
            case HyperparamType::String: return ptr_string ? *ptr_string : "?";
            case HyperparamType::SizeT:  return ptr_sizet  ? std::to_string(*ptr_sizet) : "?";
        }
        return "?";
    }
};

// ─────────────────────────────────────────────────────────
//  HyperparameterRegistry
//
//  Populate once from a live TrainingHyperparameters&.
//  Then query entries(), categories(), or filtered().
// ─────────────────────────────────────────────────────────
class HyperparameterRegistry {
public:
    // Populate the registry by binding pointers into `params`.
    // params MUST outlive the registry.
    void populate(TrainingHyperparameters& params) {
        entries_.clear();
        categories_.clear();

        // ── Core ──
        addFloat (params.learning_rate,       "learning_rate",       "Learning Rate",       "Core");
        addInt   (params.epochs,              "epochs",              "Epochs",              "Core");
        addInt64 (params.seed,                "seed",                "Seed",                "Core");
        addInt   (params.batch_size,          "batch_size",          "Batch Size",          "Core");
        addInt   (params.gradient_accumulation_steps, "gradient_accumulation_steps", "Gradient Accum Steps", "Core");
        addBool  (params.single_batch_overfit_enabled, "single_batch_overfit_enabled", "Single Batch Overfit", "Core");
        addInt   (params.single_batch_overfit_max_steps, "single_batch_overfit_max_steps", "Overfit Max Steps", "Core");
        addString(params.batch_strategy,      "batch_strategy",      "Batch Strategy",      "Core");
        addInt   (params.architecture.max_seq_len,         "max_seq_len",         "Max Seq Length",      "Core");
        addInt   (params.min_seq_valid_tokens,"min_seq_valid_tokens","Min Valid Tokens",    "Core");
        addInt   (params.log_interval,        "log_interval",        "Log Interval",        "Core");
        addInt   (params.atom_stats_interval, "atom_stats_interval", "Atom Stats Interval", "Core");
        addInt   (params.atom_stats_max_seqs, "atom_stats_max_seqs", "Atom Stats Max Seqs", "Core");
        addInt   (params.validation_interval, "validation_interval", "Validation Interval", "Core");
        addInt   (params.checkpoint_interval, "checkpoint_interval", "Checkpoint Interval", "Core");
        addBool  (params.architecture.use_gpu,             "use_gpu",             "Use GPU",             "Core");

        // ── Optimizer ──
        addFloat (params.weight_decay,        "weight_decay",        "Weight Decay",        "Optimizer");
        addFloat (params.grad_clip_norm,      "grad_clip_norm",      "Grad Clip Norm",      "Optimizer");
        addBool  (params.per_token_grad_scale,"per_token_grad_scale","Per-Token Grad Scale","Optimizer");
        addFloat (params.warmup_fraction,    "warmup_fraction",    "Warmup Fraction",    "Optimizer");
        addBool  (params.cosine_decay_enabled,"cosine_decay_enabled","Cosine Decay",        "Optimizer");
        addBool  (params.cosine_warm_restarts,"cosine_warm_restarts","Warm Restarts",       "Optimizer");
        addFloat (params.cosine_decay_min_lr, "cosine_decay_min_lr", "Cosine Decay Min LR", "Optimizer");

        // ── Soft Restart ──
        addBool  (params.soft_restart_enabled, "soft_restart_enabled", "Enabled", "Soft Restart");
        addFloat (params.soft_restart_loss_increase_threshold, "soft_restart_loss_increase_threshold", "Loss Increase Threshold", "Soft Restart");
        addInt   (params.soft_restart_max_step_window, "soft_restart_max_step_window", "Max Step Window", "Soft Restart");
        addInt   (params.soft_restart_cooldown_steps,  "soft_restart_cooldown_steps",  "Cooldown Steps",  "Soft Restart");

        // ── Auto Stop ──
        addBool  (params.auto_stop_enabled,             "auto_stop_enabled",             "Enabled",             "Auto Stop");
        addInt   (params.auto_stop_plateau_patience,    "auto_stop_plateau_patience",    "Plateau Patience",    "Auto Stop");
        addFloat (params.auto_stop_plateau_min_delta,   "auto_stop_plateau_min_delta",   "Plateau Min Delta",   "Auto Stop");
        addFloat (params.auto_stop_high_loss_threshold, "auto_stop_high_loss_threshold", "High Loss Threshold", "Auto Stop");
        addInt   (params.auto_stop_high_loss_patience,  "auto_stop_high_loss_patience",  "High Loss Patience",  "Auto Stop");

        // ── Guess Aux ──
        addBool  (params.guess_aux_enabled,        "guess_aux_enabled",        "Enabled",        "Guess Aux");
        addFloat (params.guess_aux_lambda,         "guess_aux_lambda",         "Lambda",         "Guess Aux");
        addFloat (params.guess_aux_min_confidence, "guess_aux_min_confidence", "Min Confidence", "Guess Aux");

        // ── Shuffle ──
        addBool  (params.shuffle_train_enabled, "shuffle_train_enabled", "Enabled",     "Shuffle");
        addInt   (params.shuffle_train_epochs,  "shuffle_train_epochs",  "Train Epochs","Shuffle");

        // ── Telemetry ──
        addBool  (params.telemetry_control_enabled,                "telemetry_control_enabled",                "Enabled",                      "Telemetry");
        addFloat (params.telemetry_spike_mild_threshold,           "telemetry_spike_mild_threshold",           "Spike Mild Threshold",         "Telemetry");
        addFloat (params.telemetry_spike_moderate_threshold,       "telemetry_spike_moderate_threshold",       "Spike Moderate Threshold",     "Telemetry");
        addFloat (params.telemetry_spike_severe_threshold,         "telemetry_spike_severe_threshold",         "Spike Severe Threshold",       "Telemetry");
        addFloat (params.telemetry_moderate_grad_scale,            "telemetry_moderate_grad_scale",            "Moderate Grad Scale",          "Telemetry");
        addInt   (params.telemetry_moderate_cooldown_extension,    "telemetry_moderate_cooldown_extension",    "Moderate Cooldown Extension",  "Telemetry");
        addFloat (params.telemetry_min_grad_for_nonzero_loss,      "telemetry_min_grad_for_nonzero_loss",      "Min Grad Nonzero Loss",       "Telemetry");
        addFloat (params.telemetry_loss_threshold_for_grad_check,  "telemetry_loss_threshold_for_grad_check",  "Loss Threshold Grad Check",   "Telemetry");
        addInt   (params.telemetry_max_consecutive_zero_grad_steps,"telemetry_max_consecutive_zero_grad_steps","Max Zero Grad Steps",         "Telemetry");
        addFloat (params.telemetry_seq_len_regime_change_threshold,"telemetry_seq_len_regime_change_threshold","Seq Regime Change Threshold","Telemetry");
        addInt   (params.telemetry_regime_change_suppression_steps,"telemetry_regime_change_suppression_steps","Regime Suppress Steps",       "Telemetry");
        addFloat (params.telemetry_volatility_damping_threshold,   "telemetry_volatility_damping_threshold",   "Volatility Damp Threshold",   "Telemetry");
        addFloat (params.telemetry_max_volatility_damping,         "telemetry_max_volatility_damping",         "Max Volatility Damping",      "Telemetry");
        addFloat (params.telemetry_gradient_decay_threshold,       "telemetry_gradient_decay_threshold",       "Gradient Decay Threshold",    "Telemetry");
        addFloat (params.telemetry_max_decay_boost,                "telemetry_max_decay_boost",                "Max Decay Boost",             "Telemetry");
        addFloat (params.telemetry_progress_boost_threshold,       "telemetry_progress_boost_threshold",       "Progress Boost Threshold",    "Telemetry");
        addFloat (params.telemetry_max_progress_boost,             "telemetry_max_progress_boost",             "Max Progress Boost",          "Telemetry");
        addFloat (params.telemetry_outlier_frequency_trigger,      "telemetry_outlier_frequency_trigger",      "Outlier Freq Trigger",        "Telemetry");
        addFloat (params.telemetry_outlier_persistence_trigger,    "telemetry_outlier_persistence_trigger",    "Outlier Persist Trigger",     "Telemetry");
        addFloat (params.telemetry_anchor_drift_sigma_multiplier,  "telemetry_anchor_drift_sigma_multiplier",  "Anchor Drift Sigma",          "Telemetry");
        addInt   (params.telemetry_soft_restart_cooldown_steps,    "telemetry_soft_restart_cooldown_steps",    "Soft Restart Cooldown",       "Telemetry");
        addInt   (params.telemetry_warmup_steps,                   "telemetry_warmup_steps",                   "Warmup Steps",                "Telemetry");
        addInt   (params.telemetry_baseline_stabilization_steps,   "telemetry_baseline_stabilization_steps",   "Baseline Stabilization",      "Telemetry");
        addBool  (params.telemetry_verbose_logging,                "telemetry_verbose_logging",                "Verbose Logging",             "Telemetry");
        addBool  (params.telemetry_fail_loud_on_accumulation_bug,  "telemetry_fail_loud_on_accumulation_bug",  "Fail-Loud Accum Bug",         "Telemetry");
        addBool  (params.telemetry_plateau_noise_enabled,          "telemetry_plateau_noise_enabled",          "Plateau Noise Enabled",       "Telemetry");
        addInt   (params.telemetry_plateau_noise_patience,         "telemetry_plateau_noise_patience",         "Plateau Noise Patience",      "Telemetry");
        addFloat (params.telemetry_plateau_noise_variance_threshold,"telemetry_plateau_noise_variance_threshold","Plateau Noise Var Threshold","Telemetry");
        addFloat (params.telemetry_plateau_noise_std,              "telemetry_plateau_noise_std",              "Plateau Noise Std",           "Telemetry");
        addBool  (params.telemetry_plateau_noise_proportional,     "telemetry_plateau_noise_proportional",     "Plateau Noise Proportional",  "Telemetry");
        addInt   (params.telemetry_plateau_noise_cooldown,         "telemetry_plateau_noise_cooldown",         "Plateau Noise Cooldown",      "Telemetry");
        addInt   (params.telemetry_plateau_noise_max_per_epoch,    "telemetry_plateau_noise_max_per_epoch",    "Plateau Noise Max/Epoch",     "Telemetry");

        // ── Telemetry Lattice (TelemetryLattice construction params) ──
        addInt   (params.telemetry_lattice_num_levels,  "telemetry_lattice_num_levels",  "Lattice Num Levels",  "Telemetry Lattice");
        addInt   (params.telemetry_lattice_num_streams, "telemetry_lattice_num_streams", "Lattice Num Streams", "Telemetry Lattice");
        addFloat (params.telemetry_lattice_beta_mu,     "telemetry_lattice_beta_mu",     "Lattice beta_mu",     "Telemetry Lattice");
        addFloat (params.telemetry_lattice_beta_a,      "telemetry_lattice_beta_a",      "Lattice beta_a",      "Telemetry Lattice");
        addFloat (params.telemetry_lattice_beta_delta,  "telemetry_lattice_beta_delta",  "Lattice beta_delta",  "Telemetry Lattice");
        addFloat (params.telemetry_lattice_beta_r,      "telemetry_lattice_beta_r",      "Lattice beta_r",      "Telemetry Lattice");
        addFloat (params.telemetry_lattice_beta_run,    "telemetry_lattice_beta_run",    "Lattice beta_run",    "Telemetry Lattice");
        addFloat (params.telemetry_lattice_beta_v,      "telemetry_lattice_beta_v",      "Lattice beta_v",      "Telemetry Lattice");
        addFloat (params.telemetry_lattice_k_out0,      "telemetry_lattice_k_out0",      "Lattice k_out0",      "Telemetry Lattice");
        addFloat (params.telemetry_lattice_alpha_v,     "telemetry_lattice_alpha_v",     "Lattice alpha_v",     "Telemetry Lattice");
        addFloat (params.telemetry_lattice_epsilon,     "telemetry_lattice_epsilon",     "Lattice epsilon",     "Telemetry Lattice");
        addBool  (params.telemetry_lattice_strict_mode, "telemetry_lattice_strict_mode", "Lattice Strict Mode", "Telemetry Lattice");

        // ── Loss ──
        addBool  (params.loss_label_smoothing_enabled, "loss_label_smoothing_enabled", "Label Smoothing",     "Loss");
        addFloat (params.loss_label_smoothing_epsilon, "loss_label_smoothing_epsilon", "Smoothing Epsilon",   "Loss");
        addBool  (params.loss_focal_enabled,           "loss_focal_enabled",           "Focal Loss",          "Loss");
        addFloat (params.loss_focal_gamma,             "loss_focal_gamma",             "Focal Gamma",         "Loss");
        addFloat (params.loss_focal_alpha,             "loss_focal_alpha",             "Focal Alpha",         "Loss");
        addBool  (params.loss_preference_enabled,      "loss_preference_enabled",      "Preference Loss",     "Loss");
        addFloat (params.loss_preference_beta,         "loss_preference_beta",         "Preference Beta",     "Loss");
        addBool  (params.loss_distillation_enabled,    "loss_distillation_enabled",    "Distillation",        "Loss");
        addFloat (params.loss_distillation_temperature,"loss_distillation_temperature","Distill Temperature", "Loss");
        addFloat (params.loss_distillation_lambda,     "loss_distillation_lambda",     "Distill Lambda",      "Loss");
        addBool  (params.loss_masking_enabled,         "loss_masking_enabled",         "Loss Masking",        "Loss");
        addString(params.loss_masking_tag,             "loss_masking_tag",             "Masking Tag",         "Loss");
        addBool  (params.loss_entropy_reg_enabled,     "loss_entropy_reg_enabled",     "Entropy Reg",         "Loss");
        addFloat (params.loss_entropy_reg_lambda,      "loss_entropy_reg_lambda",      "Entropy Reg Lambda",  "Loss");
        addBool  (params.loss_class_balanced_enabled,  "loss_class_balanced_enabled",  "Class Balanced",      "Loss");
        addFloat (params.loss_class_balanced_beta,     "loss_class_balanced_beta",     "Class Balanced Beta", "Loss");

        // ── LM Head ──
        addBool  (params.lm_head_centering_enabled,                "lm_head_centering_enabled",       "Centering Enabled",     "LM Head");
        addBool  (params.architecture.lm_head_center_hidden_states,    "lm_head_center_hidden_states",    "Center Hidden States",  "LM Head");
        addBool  (params.architecture.lm_head_freeze_final_rms_gamma,  "lm_head_freeze_final_rms_gamma",  "Freeze γ_final",        "LM Head");
        addBool  (params.architecture.center_logits,                   "center_logits",                   "Center Logits",         "LM Head");
        addBool  (params.architecture.center_encoder_residuals,      "center_encoder_residuals",      "Center Encoder Resids", "LM Head");
        addBool  (params.architecture.project_out_pc1,               "project_out_pc1",               "Project Out PC1",       "LM Head");
        addInt   (params.architecture.pc1_power_iters,               "pc1_power_iters",               "PC1 Power Iters",       "LM Head");

        // ── Attention ──
        addBool  (params.architecture.use_flash_attention,      "use_flash_attention",      "Flash Attention",      "Attention");
        addInt   (params.architecture.min_seq_len_for_flash,    "min_seq_len_for_flash",    "Min Seq For Flash",    "Attention");
        addBool  (params.architecture.qk_norm_enabled,          "qk_norm_enabled",          "QK-Norm",              "Attention");
        addBool  (params.attention_diag_enabled,   "attention_diag_enabled",   "Attention Diag",       "Attention");
        addInt   (params.attention_diag_layer,     "attention_diag_layer",     "Diag Layer",           "Attention");
        addInt   (params.attention_diag_head,      "attention_diag_head",      "Diag Head",            "Attention");

        // ── Layer Scale ──
        addBool  (params.architecture.use_layer_scale,  "use_layer_scale",  "Enabled",    "Layer Scale");
        addFloat (params.architecture.layer_scale_init, "layer_scale_init", "Init Value", "Layer Scale");

        // ── Scratch Block ──
        addBool  (params.scratch_blocks_enabled,               "scratch_blocks_enabled",               "Enabled",                "Scratch Block");
        addSizeT (params.scratch_max_tokens_per_block,         "scratch_max_tokens_per_block",         "Max Tokens/Block",       "Scratch Block");
        addSizeT (params.scratch_num_blocks,                   "scratch_num_blocks",                   "Num Blocks",             "Scratch Block");
        addBool  (params.scratch_write_combined,               "scratch_write_combined",               "Write Combined",         "Scratch Block");
        addBool  (params.architecture.use_scratch_block,                    "use_scratch_block",                    "Reasoning Enabled",      "Scratch Block");
        addInt   (params.architecture.scratch_block_atom_embedding_dim,     "scratch_block_atom_embedding_dim",     "Atom Embed Dim",         "Scratch Block");
        addInt   (params.architecture.scratch_block_max_atoms,              "scratch_block_max_atoms",              "Max Atoms",              "Scratch Block");
        addFloat (params.architecture.scratch_block_atom_scale,             "scratch_block_atom_scale",             "Atom Scale",             "Scratch Block");

        // ── Execution Block ──
        addBool  (params.architecture.execution_block_enabled,                    "execution_block_enabled",                    "Enabled",                  "Execution Block");
        addBool  (params.architecture.scratch_block_execution_first_type_only,    "scratch_block_execution_first_type_only",    "Type-Only Scratch",        "Execution Block");
        addInt   (params.architecture.execution_block_layer,                      "execution_block_layer",                      "Layer",                    "Execution Block");
        addInt   (params.architecture.execution_block_num_ops,                    "execution_block_num_ops",                    "Num Ops",                  "Execution Block");
        addInt   (params.architecture.execution_block_num_slots,                  "execution_block_num_slots",                  "Num Slots",                "Execution Block");
        addInt   (params.architecture.execution_block_num_steps,                  "execution_block_num_steps",                  "Num Steps",                "Execution Block");
        addInt   (params.architecture.execution_block_d_key,                      "execution_block_d_key",                      "d_key",                    "Execution Block");
        addInt   (params.architecture.execution_block_d_type,                     "execution_block_d_type",                     "d_type",                   "Execution Block");
        addInt   (params.architecture.execution_block_cross_attn_head_dim,        "execution_block_cross_attn_head_dim",        "Cross-Attn Head Dim",      "Execution Block");
        addInt   (params.architecture.execution_block_cross_attn_topk,            "execution_block_cross_attn_topk",            "Cross-Attn Top-K",         "Execution Block");
        addFloat (params.architecture.execution_block_usage_decay,                "execution_block_usage_decay",                "Usage Decay",              "Execution Block");
        addFloat (params.architecture.execution_block_diversity_kappa,            "execution_block_diversity_kappa",            "Diversity Kappa",          "Execution Block");
        addFloat (params.architecture.execution_block_temp_start,                 "execution_block_temp_start",                 "Temp Start",               "Execution Block");
        addFloat (params.architecture.execution_block_temp_end,                   "execution_block_temp_end",                   "Temp End",                 "Execution Block");
        addInt   (params.architecture.execution_block_temp_schedule,              "execution_block_temp_schedule",              "Temp Schedule",            "Execution Block");
        addFloat (params.architecture.execution_block_entropy_weight,             "execution_block_entropy_weight",             "Entropy Weight",           "Execution Block");
        addFloat (params.architecture.step_x_multiplier,                          "step_x_multiplier",                          "Step X Multiplier",        "Execution Block");
        addFloat (params.architecture.step_y_multiplier,                          "step_y_multiplier",                          "Step Y Multiplier",        "Execution Block");
        addBool  (params.architecture.step_y_overrides_x,                         "step_y_overrides_x",                         "Y Overrides X",            "Execution Block");
        addFloat (params.architecture.entropy_aux_weight,                         "entropy_aux_weight",                         "Entropy Aux Weight",       "Execution Block");
        addFloat (params.architecture.value_match_epsilon,                        "value_match_epsilon",                        "Value Match Epsilon",      "Execution Block");
        addFloat (params.architecture.final_slot_consistency_weight,              "final_slot_consistency_weight",              "Slot Consistency Weight",  "Execution Block");
        addFloat (params.architecture.execution_block_transition_hard_threshold,  "execution_block_transition_hard_threshold",  "Transition Hard Threshold","Execution Block");
        addInt   (params.architecture.execution_block_gate_warmup_steps,          "execution_block_gate_warmup_steps",          "Gate Warmup Steps",        "Execution Block");
        addFloat (params.architecture.execution_block_causal_w1_transition,       "execution_block_causal_w1_transition",       "Causal W1 Transition",    "Execution Block");
        addFloat (params.architecture.div_magnitude_penalty_weight,               "div_magnitude_penalty_weight",               "Div Magnitude Penalty",   "Execution Block");
        addFloat (params.architecture.arg_reinforce_weight,                       "arg_reinforce_weight",                       "Arg REINFORCE Weight",    "Execution Block");
        addFloat (params.architecture.arg_reinforce_baseline_decay,               "arg_reinforce_baseline_decay",               "REINFORCE Baseline Decay","Execution Block");
        addBool  (params.architecture.structured_ce_enabled,                       "structured_ce_enabled",                       "Structured CE Enabled",   "Execution Block");
        addFloat (params.architecture.structured_ce_weight,                        "structured_ce_weight",                        "Structured CE Weight",    "Execution Block");
        addBool  (params.architecture.selector_enabled,                           "selector_enabled",                           "Selector Enabled",         "Execution Block");
        addInt   (params.architecture.selector_d_selector,                        "selector_d_selector",                        "Selector d_selector",      "Execution Block");
        addFloat (params.architecture.selector_selection_margin,                  "selector_selection_margin",                  "Selection Margin",         "Execution Block");
        addFloat (params.architecture.selector_supervision_weight,                "selector_supervision_weight",                "Supervision Weight",       "Execution Block");

        // ── CUDA ──
        addBool  (params.single_stream_mode,         "single_stream_mode",         "Single Stream Mode",     "CUDA");
        addBool  (params.disable_async_frees,        "disable_async_frees",        "Disable Async Frees",    "CUDA");
        addBool  (params.synchronize_after_kernels,  "synchronize_after_kernels",  "Sync After Kernels",     "CUDA");

        // ── MTP ──
        addBool  (params.architecture.mtp_enabled,               "mtp_enabled",               "Enabled",            "MTP");
        addInt   (params.architecture.mtp_k,                     "mtp_k",                     "K (Lookahead)",      "MTP");
        addFloat (params.architecture.mtp_alpha,                 "mtp_alpha",                 "Alpha",              "MTP");
        addInt   (params.architecture.mtp_alpha_warmup_steps,    "mtp_alpha_warmup_steps",    "Alpha Warmup Steps", "MTP");
        addBool  (params.mtp_log_ratio_monitor,     "mtp_log_ratio_monitor",     "Log Ratio Monitor",  "MTP");

        // ── Embedding ──
        addBool  (params.embedding_freeze_enabled,      "embedding_freeze_enabled",      "Freeze Enabled",    "Embedding");
        addInt   (params.embedding_freeze_after_step,    "embedding_freeze_after_step",    "Freeze After Step", "Embedding");

        // ── Stability ──
        addBool  (params.stability_overrides_enabled,        "stability_overrides_enabled",        "Enabled",           "Stability");
        addInt   (params.stability_override_batch_size,      "stability_override_batch_size",      "Override Batch",    "Stability");
        addInt   (params.stability_override_max_seq_len,     "stability_override_max_seq_len",     "Override Max Seq",  "Stability");
        addFloat (params.stability_override_clip_per_token,  "stability_override_clip_per_token",  "Override Clip/Tok", "Stability");
        addFloat (params.stability_override_lr_min,          "stability_override_lr_min",          "Override LR Min",   "Stability");

        // ── Diagnostics ──
        addInt   (params.architecture.hardcoded_log_every_n_batches,      "hardcoded_log_every_n_batches",      "Log Every N Batches",    "Diagnostics");
        addBool  (params.prediction_comparison_enabled,      "prediction_comparison_enabled",      "Pred Compare Enabled",   "Diagnostics");
        addInt   (params.prediction_comparison_interval,     "prediction_comparison_interval",     "Pred Compare Interval",  "Diagnostics");
        addInt   (params.prediction_comparison_top_k,        "prediction_comparison_top_k",        "Pred Compare Top-K",     "Diagnostics");
        addInt   (params.prediction_comparison_max_positions,"prediction_comparison_max_positions","Pred Compare Max Pos",   "Diagnostics");
        addString(params.prediction_comparison_log_path,     "prediction_comparison_log_path",     "Pred Compare Log Path",  "Diagnostics");
        addBool  (params.logit_update_trace_enabled,         "logit_update_trace_enabled",         "Logit Trace Enabled",    "Diagnostics");
        addInt   (params.logit_update_trace_interval,        "logit_update_trace_interval",        "Logit Trace Interval",   "Diagnostics");

        // ── Tokenizer-Linked ──
        addBool  (params.tokenizer_enable_scratch_block_reasoning, "tokenizer_enable_scratch_block_reasoning", "Tokenizer Scratch Reason", "Tokenizer");
        addBool  (params.tokenizer_detect_numbers,                 "tokenizer_detect_numbers",                 "Detect Numbers",           "Tokenizer");

        // Build sorted category list
        std::set<std::string> cats;
        for (const auto& e : entries_) cats.insert(e.category);
        categories_.assign(cats.begin(), cats.end());
    }

    // All registered entries
    const std::vector<HyperparamEntry>& entries() const { return entries_; }

    // Sorted unique category names
    const std::vector<std::string>& categories() const { return categories_; }

    // Filter entries by category (empty = all)
    std::vector<const HyperparamEntry*> filtered(const std::string& category) const {
        std::vector<const HyperparamEntry*> result;
        for (const auto& e : entries_) {
            if (category.empty() || e.category == category) {
                result.push_back(&e);
            }
        }
        return result;
    }

    bool empty() const { return entries_.empty(); }

private:
    std::vector<HyperparamEntry> entries_;
    std::vector<std::string> categories_;

    void addBool(bool& field, const char* key, const char* display, const char* cat) {
        HyperparamEntry e;
        e.key = key; e.display_name = display; e.category = cat;
        e.type = HyperparamType::Bool; e.ptr_bool = &field;
        entries_.push_back(std::move(e));
    }
    void addInt(int& field, const char* key, const char* display, const char* cat) {
        HyperparamEntry e;
        e.key = key; e.display_name = display; e.category = cat;
        e.type = HyperparamType::Int; e.ptr_int = &field;
        entries_.push_back(std::move(e));
    }
    void addInt64(int64_t& field, const char* key, const char* display, const char* cat) {
        HyperparamEntry e;
        e.key = key; e.display_name = display; e.category = cat;
        e.type = HyperparamType::Int64; e.ptr_int64 = &field;
        entries_.push_back(std::move(e));
    }
    void addFloat(float& field, const char* key, const char* display, const char* cat) {
        HyperparamEntry e;
        e.key = key; e.display_name = display; e.category = cat;
        e.type = HyperparamType::Float; e.ptr_float = &field;
        entries_.push_back(std::move(e));
    }
    void addString(std::string& field, const char* key, const char* display, const char* cat) {
        HyperparamEntry e;
        e.key = key; e.display_name = display; e.category = cat;
        e.type = HyperparamType::String; e.ptr_string = &field;
        entries_.push_back(std::move(e));
    }
    void addSizeT(size_t& field, const char* key, const char* display, const char* cat) {
        HyperparamEntry e;
        e.key = key; e.display_name = display; e.category = cat;
        e.type = HyperparamType::SizeT; e.ptr_sizet = &field;
        entries_.push_back(std::move(e));
    }
};

} // namespace Config
} // namespace GRIM
