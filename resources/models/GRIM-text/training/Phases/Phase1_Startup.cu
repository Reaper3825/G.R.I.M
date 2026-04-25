//======================================================//
//  Phase1_Startup.cu
//  Configuration loading, model initialization, data loading
//======================================================//
//
//  IMPLEMENTATION
//  ==============
//  This file implements all startup operations:
//  1. Configuration loading from ai_config.json
//  2. Path validation and directory creation
//  3. Logging infrastructure initialization
//  4. Tokenizer loading
//  5. Training data loading with sliding windows
//  6. Model initialization and weight seeding
//  7. Optimizer and gradient controller setup
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#include "Phase1_Startup.hpp"
#include "../OptimizerCheckpoint.hpp"
#include "ConfigDump.hpp"
// MUST be first - defines GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED
#include "../../../../../control/ai_config_paths.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
// curand.h moved to Startup/Rng.cu
// Xavier.hpp removed - weights initialized via Tensor::xavier_uniform_() with Philox PRNG (Issue #107)
#endif

using json = nlohmann::json;

// Module logging aliases
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {

//======================================================//
//  String Utilities
//======================================================//

namespace {

std::string trimCopy(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(start, end - start + 1);
}

std::string toLowerCopy(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

void registerDefaultLoggingProfiles() {
    // Module log profile registration removed: API never implemented.
    // EmitModule*() functions remain available via LogRecorder.hpp.
}

} // anonymous namespace

//======================================================//
//  PathConfig Implementation
//======================================================//

bool PathConfig::validate() const {
    return !data_path.empty() &&
           !vocab_path.empty() &&
           !output_model_path.empty() &&
           !checkpoint_dir.empty() &&
           !log_dir.empty();
}

//======================================================//
//  Internal Helper Implementations
//======================================================//

namespace Internal {

StartupConfig loadConfiguration(int argc, char** argv) {
    StartupConfig config;
    
    // Resolve config path: --config overrides default ai_config.json discovery
    std::string config_path = "ai_config.json";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            break;
        }
    }
    
    // Load ai_config.json snapshot (or path from --config, e.g. model_config.json)
    auto snapshot = GRIM::Config::loadAiConfigSnapshot(config_path);
    if (!snapshot) {
        throw std::runtime_error("FATAL: ai_config.json not found or unreadable");
    }
    
    config.paths.config_path = snapshot->config_path;
    const json& doc = snapshot->document;
    
    // Extract GRIM-text paths
    if (!doc.contains("paths") || !doc["paths"].contains("grim_text")) {
        throw std::runtime_error("FATAL: ai_config.json missing 'paths.grim_text' section");
    }
    
    const auto& grim_text = doc["paths"]["grim_text"];
    
    // Required paths
    if (!grim_text.contains("training_data") || grim_text["training_data"].get<std::string>().empty()) {
        throw std::runtime_error("FATAL: ai_config.json missing or empty 'grim_text.training_data'");
    }
    if (!grim_text.contains("vocab") || grim_text["vocab"].get<std::string>().empty()) {
        throw std::runtime_error("FATAL: ai_config.json missing or empty 'grim_text.vocab'");
    }
    
    config.paths.data_path = grim_text["training_data"].get<std::string>();
    config.paths.vocab_path = grim_text["vocab"].get<std::string>();
    config.paths.output_model_path = grim_text.value("model", "");
    config.paths.checkpoint_dir = grim_text.value("checkpoints", "checkpoints");
    config.paths.log_dir = grim_text.value("logs", "logs");
    config.paths.status_path = grim_text.value("training_status", "temp_status.txt");
    
    // Load hyperparameters
    if (snapshot->has_training) {
        config.hyperparameters = snapshot->hyperparameters;
    }
    GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);

    // Configure LogRecorder
    if (config.hyperparameters.log_recorder.enabled) {
        // Initialize log recorder system - MUST pass path (Rule 20: no hardcoded fallback)
        GRIM::Logging::InitLogRecorder(config.paths.log_dir);
        
        // Configure layer logging enables
        const auto& layers = config.hyperparameters.log_recorder.layers;
        GRIM::Logging::ConfigureLayerLogging(
            config.hyperparameters.log_recorder.enabled,
            layers.embedding,
            layers.rms_norm,
            layers.attention,
            layers.feed_forward,
            layers.residual,
            layers.encoding,
            layers.serialization,
            layers.execution_block);
    } else {
        // Disable all layer logging if master disabled
        GRIM::Logging::ConfigureLayerLogging(false, false, false, false, false, false, false, false, false);
    }
    
    // Map loss options
    config.loss_options.label_smoothing_enabled = config.hyperparameters.loss_label_smoothing_enabled;
    config.loss_options.label_smoothing_epsilon = config.hyperparameters.loss_label_smoothing_epsilon;
    config.loss_options.focal_enabled = config.hyperparameters.loss_focal_enabled;
    config.loss_options.focal_gamma = config.hyperparameters.loss_focal_gamma;
    config.loss_options.focal_alpha = config.hyperparameters.loss_focal_alpha;
    config.loss_options.preference_enabled = config.hyperparameters.loss_preference_enabled;
    config.loss_options.preference_beta = config.hyperparameters.loss_preference_beta;
    config.loss_options.distillation_enabled = config.hyperparameters.loss_distillation_enabled;
    config.loss_options.distillation_temperature = config.hyperparameters.loss_distillation_temperature;
    config.loss_options.distillation_lambda = config.hyperparameters.loss_distillation_lambda;
    config.loss_options.masking_enabled = config.hyperparameters.loss_masking_enabled;
    config.loss_options.masking_tag = config.hyperparameters.loss_masking_tag;
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    config.loss_options.entropy_reg_enabled = config.hyperparameters.loss_entropy_reg_enabled;
    config.loss_options.entropy_reg_lambda = config.hyperparameters.loss_entropy_reg_lambda;

    // Class-balanced loss: reweights per-token loss by 1/freq^β
    config.loss_options.class_balanced_enabled = config.hyperparameters.loss_class_balanced_enabled;
    config.loss_options.class_balanced_beta = config.hyperparameters.loss_class_balanced_beta;

    // Map stability overrides
    config.stability.enabled = config.hyperparameters.stability_overrides_enabled;
    config.stability.batch_size = config.hyperparameters.stability_override_batch_size;
    config.stability.max_seq_len = config.hyperparameters.stability_override_max_seq_len;
    config.stability.clip_per_token = config.hyperparameters.stability_override_clip_per_token;
    config.stability.lr_min = config.hyperparameters.stability_override_lr_min;
    
    // Map scratch block configuration
    config.scratch.enabled = config.hyperparameters.scratch_blocks_enabled;
    // max_tokens_per_block is computed from max_cached_tokens in InitTrainingState — not from config
    config.scratch.num_blocks = config.hyperparameters.scratch_num_blocks;
    config.scratch.write_combined = config.hyperparameters.scratch_write_combined;
    
    // Map CUDA execution mode
    config.cuda_exec.single_stream_mode = config.hyperparameters.single_stream_mode;
    config.cuda_exec.disable_async_frees = config.hyperparameters.disable_async_frees;
    config.cuda_exec.synchronize_after_kernels = config.hyperparameters.synchronize_after_kernels;
    
    // Map prediction comparison
    config.pred_comparison.enabled = config.hyperparameters.prediction_comparison_enabled;
    config.pred_comparison.interval = config.hyperparameters.prediction_comparison_interval;
    config.pred_comparison.top_k = config.hyperparameters.prediction_comparison_top_k;
    config.pred_comparison.max_positions = config.hyperparameters.prediction_comparison_max_positions;
    config.pred_comparison.log_path = config.hyperparameters.prediction_comparison_log_path;
    
    // Load architecture
    if (doc.contains("training") && doc["training"].contains("config")) {
        const auto& cfg = doc["training"]["config"];
        if (cfg.contains("d_model") && cfg["d_model"].is_number())
            config.architecture.d_model = cfg["d_model"].get<int>();
        if (cfg.contains("num_layers") && cfg["num_layers"].is_number())
            config.architecture.num_layers = cfg["num_layers"].get<int>();
        if (cfg.contains("num_heads") && cfg["num_heads"].is_number())
            config.architecture.num_heads = cfg["num_heads"].get<int>();
        if (cfg.contains("num_kv_heads") && cfg["num_kv_heads"].is_number())
            config.architecture.num_kv_heads = cfg["num_kv_heads"].get<int>();
        // d_ff: always derived as d_model * DEFAULT_D_FF_MULTIPLIER (never read from JSON)
        config.architecture.d_ff = config.architecture.d_model * GRIM::HyperParameters::DEFAULT_D_FF_MULTIPLIER;
        if (cfg.contains("dropout_rate") && cfg["dropout_rate"].is_number())
            config.architecture.dropout_rate = cfg["dropout_rate"].get<float>();
        // attention_dropout: always derived from dropout_rate
        config.architecture.attention_dropout = config.architecture.dropout_rate;
        
        // Load tie_embeddings config (affects memory layout and parameter count)
        if (cfg.contains("tie_embeddings") && cfg["tie_embeddings"].is_boolean())
            config.architecture.tie_embeddings = cfg["tie_embeddings"].get<bool>();

        // Parse positional_encoding from JSON config (object or string form).
        if (cfg.contains("positional_encoding")) {
            const auto& pe = cfg["positional_encoding"];
            if (pe.is_object()) {
                const bool use_rope = pe.value("use_rope", false);
                const bool use_alibi = pe.value("use_alibi", false);
                if (use_rope && use_alibi) {
                    config.architecture.positional_encoding = GRIM::HyperParameters::PositionalEncodingType::ALIBI_ROPE;
                } else if (use_rope) {
                    config.architecture.positional_encoding = GRIM::HyperParameters::PositionalEncodingType::ROPE;
                } else if (use_alibi) {
                    config.architecture.positional_encoding = GRIM::HyperParameters::PositionalEncodingType::ALIBI;
                } else {
                    // Rule 20: learned position embeddings were removed; require ALiBi and/or RoPE.
                    throw std::runtime_error(
                        "Phase1_Startup::loadConfiguration: positional_encoding requires use_rope and/or use_alibi "
                        "(learned position embeddings have been removed)");
                }
            } else if (pe.is_string()) {
                config.architecture.positional_encoding =
                    GRIM::HyperParameters::parsePositionalEncodingType(pe.get<std::string>());
                if (config.architecture.positional_encoding == GRIM::HyperParameters::PositionalEncodingType::NONE) {
                    throw std::runtime_error(
                        "Phase1_Startup::loadConfiguration: positional_encoding=NONE is no longer supported "
                        "(learned position embeddings have been removed). Use ALIBI, ROPE, or ALIBI_ROPE.");
                }
            } else {
                throw std::runtime_error(
                    "Phase1_Startup::loadConfiguration: training.config.positional_encoding must be object or string");
            }
        }
            
        config.force_rebuild_vocab = cfg.value("force_rebuild_vocab", false);
        
        // Load generation config for inference samples during training
        if (cfg.contains("generation") && cfg["generation"].is_object()) {
            const auto& gen = cfg["generation"];
            if (gen.contains("strategy") && gen["strategy"].is_string()) {
                const std::string strat = gen["strategy"].get<std::string>();
                if (strat == "greedy") { config.generation.strategy = GRIM::SamplingStrategy::GREEDY; config.generation.do_sample = false; }
                else if (strat == "top_k") config.generation.strategy = GRIM::SamplingStrategy::TOP_K;
                else if (strat == "top_p") config.generation.strategy = GRIM::SamplingStrategy::TOP_P;
                else if (strat == "min_p") config.generation.strategy = GRIM::SamplingStrategy::MIN_P;
                else if (strat == "typical") config.generation.strategy = GRIM::SamplingStrategy::TYPICAL;
                else if (strat == "top_k_top_p") config.generation.strategy = GRIM::SamplingStrategy::TOP_K_TOP_P;
                else throw std::runtime_error("Phase1_Startup: unknown generation.strategy: " + strat);
            }
            if (gen.contains("max_new_tokens") && gen["max_new_tokens"].is_number())
                config.generation.max_new_tokens = gen["max_new_tokens"].get<int>();
            if (gen.contains("min_new_tokens") && gen["min_new_tokens"].is_number())
                config.generation.min_new_tokens = gen["min_new_tokens"].get<int>();
            if (gen.contains("temperature") && gen["temperature"].is_number())
                config.generation.temperature = gen["temperature"].get<float>();
            if (gen.contains("top_k") && gen["top_k"].is_number())
                config.generation.top_k = gen["top_k"].get<int>();
            if (gen.contains("top_p") && gen["top_p"].is_number())
                config.generation.top_p = gen["top_p"].get<float>();
            if (gen.contains("min_p") && gen["min_p"].is_number())
                config.generation.min_p = gen["min_p"].get<float>();
            if (gen.contains("typical_p") && gen["typical_p"].is_number())
                config.generation.typical_p = gen["typical_p"].get<float>();
            if (gen.contains("repetition_penalty") && gen["repetition_penalty"].is_number())
                config.generation.repetition_penalty = gen["repetition_penalty"].get<float>();
            if (gen.contains("repetition_penalty_window") && gen["repetition_penalty_window"].is_number())
                config.generation.repetition_penalty_window = gen["repetition_penalty_window"].get<int>();
            if (gen.contains("frequency_penalty") && gen["frequency_penalty"].is_number())
                config.generation.frequency_penalty = gen["frequency_penalty"].get<float>();
            if (gen.contains("presence_penalty") && gen["presence_penalty"].is_number())
                config.generation.presence_penalty = gen["presence_penalty"].get<float>();
            if (gen.contains("no_repeat_ngram_size") && gen["no_repeat_ngram_size"].is_number())
                config.generation.no_repeat_ngram_size = gen["no_repeat_ngram_size"].get<int>();
            if (gen.contains("do_sample") && gen["do_sample"].is_boolean())
                config.generation.do_sample = gen["do_sample"].get<bool>();
            if (gen.contains("enable_scratchblock_reasoning") && gen["enable_scratchblock_reasoning"].is_boolean())
                config.generation.enable_scratchblock_reasoning = gen["enable_scratchblock_reasoning"].get<bool>();
        }
    }
    config.architecture.max_seq_len = config.hyperparameters.max_seq_len;
    config.architecture.validate();
    
    // Compute derived values
    // Rule 20: No fallback - max_seq_len must be configured (used for cache allocation)
    // ONLY use stability override if stability mode is actually enabled
    if (config.stability.enabled && config.stability.max_seq_len > 0) {
        config.max_seq_len = config.stability.max_seq_len;
    } else if (config.hyperparameters.max_seq_len > 0) {
        config.max_seq_len = config.hyperparameters.max_seq_len;
    } else {
        throw std::runtime_error("FATAL: max_seq_len not configured in ai_config.json (stability or hyperparameters)");
    }
    // stride = 7/8 of max_seq_len → 12.5% overlap (128 context tokens at seq_len=1024)
    // Previous: max_seq_len/2 = 50% overlap wasted ~30% of total GPU compute on masked tokens.
    // 12.5% overlap is standard practice — enough context for attention continuity
    // without burning half the token budget on zero-gradient positions.
    config.sliding_window_stride = std::max(1, config.max_seq_len * 7 / 8);
    
    // Apply stability overrides to batch size and LR (only if stability mode enabled)
    if (config.stability.enabled) {
        if (config.stability.batch_size <= 0) {
            throw std::runtime_error("FATAL: stability_overrides enabled but stability_override_batch_size=" + 
                                     std::to_string(config.stability.batch_size) + " (must be > 0)");
        }
        config.hyperparameters.batch_size = config.stability.batch_size;
        if (config.stability.clip_per_token > 0.0f) {
            config.hyperparameters.grad_clip_norm = config.stability.clip_per_token;
        }
    }
 
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) config.paths.data_path = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) config.paths.vocab_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) config.paths.output_model_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) config.hyperparameters.epochs = std::atoi(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc) config.hyperparameters.batch_size = std::atoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) config.hyperparameters.learning_rate = std::atof(argv[++i]);
        else if (arg == "--save-test") config.save_test_mode = true;
        else if (arg == "--config" && i + 1 < argc) { ++i; }  // consumed above for loadAiConfigSnapshot
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --config <path>    Config file (ai_config.json or model_config.json)\n";
            std::cout << "  --data <path>      Training data path\n";
            std::cout << "  --vocab <path>     Vocabulary path\n";
            std::cout << "  --output <path>    Output model path\n";
            std::cout << "  --epochs <n>       Number of epochs\n";
            std::cout << "  --batch-size <n>   Batch size\n";
            std::cout << "  --lr <rate>        Learning rate\n";
            std::cout << "  --save-test        Test serialization save and exit\n";
            std::exit(0);
        }
    }
    
    return config;
}

void validatePaths(const PathConfig& paths) {
    if (!fs::exists(paths.vocab_path)) {
        throw std::runtime_error("Vocabulary file does not exist: " + paths.vocab_path);
    }
    if (!fs::exists(paths.data_path)) {
        throw std::runtime_error("Training data file does not exist: " + paths.data_path);
    }
    if (paths.output_model_path.empty()) {
        throw std::runtime_error("Output model path not configured");
    }
    if (paths.checkpoint_dir.empty()) {
        throw std::runtime_error("Checkpoint directory not configured");
    }
    if (paths.log_dir.empty()) {
        throw std::runtime_error("Log directory not configured");
    }
    
    // Create directories
    auto model_parent = fs::path(paths.output_model_path).parent_path();
    if (!model_parent.empty())
        fs::create_directories(model_parent);
    fs::create_directories(paths.checkpoint_dir);
    fs::create_directories(paths.log_dir);
}

LoggingContext initializeLogging(const PathConfig& paths) {
    LoggingContext ctx;
    
    ctx.session_id = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    ctx.raw_log_path = paths.log_dir + "/training_" + ctx.session_id + ".log";
    
    // Bootstrap log file
    {
        std::ofstream bootstrap(ctx.raw_log_path, std::ios::app);
        if (bootstrap.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            bootstrap << "[BOOT] Phase1 logging bootstrap at "
                      << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
                      << std::endl;
        }
    }
    
    ctx.logger = std::make_unique<TrainingLogger>(paths.log_dir, ctx.session_id);
    ctx.metrics_collector = std::make_unique<MetricsCollector>();
    ctx.status_writer = std::make_unique<StatusFileWriter>(paths.status_path);
    
    return ctx;
}

GRIM::Tokenizer::UniByte initializeTokenizer(
    const std::string& vocab_path,
    const GRIM::Config::TokenizerConfig& tok_config,
    const GRIM::Config::TrainingHyperparameters& hyperparameters,
    TrainingLogger& logger) {
    
    logger.log("Loading tokenizer configuration...");
    
    GRIM::Tokenizer::UniByteConfig cfg;
    // Rule 20: No fallback - vocab_size must be in config. Actual vocab comes from .grmt file later.
    if (tok_config.vocab_size <= 0) {
        throw std::runtime_error("FATAL: tokenizer.vocab_size not configured in ai_config.json");
    }
    // Use all tokenizer config values from hyperparameters (single source of truth)
    cfg.target_vocab_size = tok_config.vocab_size;
    cfg.character_coverage = GRIM::HyperParameters::TOKENIZER_CHARACTER_COVERAGE;
    cfg.enable_scratch_block_reasoning = hyperparameters.tokenizer_enable_scratch_block_reasoning;
    cfg.detect_numbers = hyperparameters.tokenizer_detect_numbers;
    cfg.enable_byte_fallback = tok_config.enable_byte_fallback;
    cfg.prefer_gpu = GRIM::HyperParameters::TOKENIZER_PREFER_GPU;
    
    GRIM::Tokenizer::UniByte tokenizer(cfg);
    if (!tokenizer.load(vocab_path)) {
        throw std::runtime_error("Failed to load vocabulary: " + vocab_path);
    }
    
    // The GRMT is the ground truth — it was written with totalVocabSize() at encode time.
    // Phase1 must match that exactly.  Post-load capping is wrong: it reduces totalVocabSize()
    // below what the GRMT recorded, guaranteeing a mismatch.  If you want fewer tokens,
    // lower vocab_size in ai_config.json and let the DataLoader retrain the tokenizer.
    logger.log("✓ Loaded " + std::to_string(tokenizer.totalVocabSize()) + " tokens (" +
               std::to_string(tokenizer.vocabSize()) + " pieces)");
    
    return tokenizer;
}

SequenceData loadTrainingData(
    const std::string& data_path,
    int max_seq_len,
    int min_seq_valid_tokens,
    int sliding_window_stride,
    bool add_bos_token,
    bool add_eos_token,
    const GRIM::Tokenizer::UniByte& tokenizer,
    TrainingLogger& logger) {
    
    SequenceData data;
    
    logger.log("Loading training data...");
    GRMTDataLoader loader;
    if (!loader.load(data_path)) {
        throw std::runtime_error("Failed to load training data");
    }
    logger.log("✓ Loaded " + std::to_string(loader.size()) + " sequences");
    
    // Store vocab size from training data for later validation
    data.vocab_size = loader.vocabSize();
    
    auto all_sequences = loader.getSequences();

    const int bos_id = tokenizer.bosId();
    const int eos_id = tokenizer.eosId();

    size_t added_bos = 0;
    size_t added_eos = 0;

    for (auto& seq : all_sequences) {
        if (seq.token_ids.empty()) continue;

        // Add BOS if missing at start (controlled by config flag add_bos_token)
        if (add_bos_token && bos_id >= 0 && seq.token_ids.front() != bos_id) {
            seq.token_ids.insert(seq.token_ids.begin(), bos_id);
            seq.token_numeric_values.insert(seq.token_numeric_values.begin(), 0.0f);
            seq.token_atom_mask.insert(seq.token_atom_mask.begin(), 0);
            seq.atom_entry_ids.insert(seq.atom_entry_ids.begin(), GRIM::Tokenizer::kAtomEntryNone);
            seq.token_atom_flags.insert(seq.token_atom_flags.begin(), 0);
            for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) {
                seq.token_text_features.insert(seq.token_text_features.begin(), 0);
            }
            seq.targets.insert(seq.targets.begin(), -1);
            if (!seq.token_exec_slots.empty())
                seq.token_exec_slots.insert(seq.token_exec_slots.begin(), static_cast<int32_t>(-1));
            if (!seq.slot_selection_targets.empty())
                seq.slot_selection_targets.insert(seq.slot_selection_targets.begin(),
                    GRIM::Execution::SlotSelectionTarget{GRIM::Execution::SlotSelectionTargetKind::Ignore, -1});
            // BOS insertion shifted all existing token positions right by 1.
            // Remap compiled_bootstrap_bindings token_pos to match.
            for (auto& b : seq.compiled_bootstrap_bindings)
                b.token_pos += 1;
            added_bos++;
        }

        // Add EOS if missing at end (controlled by config flag add_eos_token)
        if (add_eos_token && eos_id >= 0 && seq.token_ids.back() != eos_id) {
            seq.token_ids.push_back(eos_id);
            seq.token_numeric_values.push_back(0.0f);
            seq.token_atom_mask.push_back(0);
            seq.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
            seq.token_atom_flags.push_back(0);
            for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) {
                seq.token_text_features.push_back(0);
            }
            // Fix shift: the PREVIOUS position's target was -1 (no next token existed
            // when DataLoader ran). Now EOS follows it, so set target = eos_id.
            if (!seq.targets.empty()) {
                seq.targets.back() = eos_id;  // position before EOS → predict EOS
            }
            seq.targets.push_back(-1);  // EOS position itself: nothing follows
            if (!seq.token_exec_slots.empty())
                seq.token_exec_slots.push_back(static_cast<int32_t>(-1));
            if (!seq.slot_selection_targets.empty())
                seq.slot_selection_targets.push_back(
                    GRIM::Execution::SlotSelectionTarget{GRIM::Execution::SlotSelectionTargetKind::Ignore, -1});
            added_eos++;
        }
    }

    if (added_bos > 0 || added_eos > 0) {
        logger.log("[Data] Boundary tokens: added_bos=" + std::to_string(added_bos) +
                   " added_eos=" + std::to_string(added_eos) + " (controlled by config flags)");
    } else {
        logger.log("[Data] Boundary token insertion disabled by config flags (add_bos=" + 
                   std::string(add_bos_token ? "true" : "false") + ", add_eos=" + 
                   std::string(add_eos_token ? "true" : "false") + ")");
    }

    // Split train/val
    size_t val_size = all_sequences.size() / 10;
    data.train_seqs.assign(all_sequences.begin() + val_size, all_sequences.end());
    data.val_seqs.assign(all_sequences.begin(), all_sequences.begin() + val_size);
    
    // Apply sliding windows.
    //
    // Padding ownership: BatchPayload is the SINGLE owner of padded
    // [batch_size * max_seq_len] layout. Phase1 produces variable-length
    // TrainingSequences (length ∈ [min_seq_valid_tokens, max_seq_len]) and
    // does NOT pre-pad. BatchPayload's per-row memcpy fills [0, seq_len) and
    // leaves [seq_len, max_seq_len) at its assign() defaults (PAD / -1 / 0).
    // Pre-padding here would destroy the logical length, polluting
    // seq_lengths / actual_tokens / token_stats / MTP shift math.
    //
    // Final-target contract: BatchPayload asserts targets.back() == -1
    // (no autoregressive supervision at the sequence boundary). Phase1 must
    // always mask the final target before handing the sequence off, for both
    // train and val — there is no "keep the val final target" path because
    // BatchPayload would silently drop it anyway.
    auto applySlidingWindows = [&](std::vector<TrainingSequence>& sequences,
                                   const std::string& split_name) {
        std::vector<TrainingSequence> windowed;
        windowed.reserve(sequences.size());

        size_t long_seq_count = 0;
        size_t generated_windows = 0;
        size_t bos_prepended = 0;

        // Final-position autoregressive boundary mask. Required by
        // BatchPayload's Rule 20 invariant.
        auto MaskFinalTarget = [](TrainingSequence& seq) {
            if (!seq.targets.empty()) {
                seq.targets.back() = -1;
            }
        };

        for (const auto& seq : sequences) {
            if (static_cast<int>(seq.token_ids.size()) <= max_seq_len) {
                // Short sequence or exactly max_seq_len — no windowing.
                TrainingSequence copy = seq;
                MaskFinalTarget(copy);
                windowed.push_back(std::move(copy));
                continue;
            }
            
            // Execution-active rows MUST NOT be fragmented — compiled_bootstrap_bindings
            // and teacher_steps are whole-sequence structures with no windowing semantics.
            if (seq.execution_active) {
                throw std::runtime_error(
                    "Execution-active sequence exceeds max_seq_len (" +
                    std::to_string(seq.token_ids.size()) + " > " +
                    std::to_string(max_seq_len) +
                    "). Execution rows cannot be split by sliding window. "
                    "Increase max_seq_len or shorten the source data.");
            }
            
            long_seq_count++;
            const size_t seq_len = seq.token_ids.size();
            size_t start = 0;
            const size_t stride = static_cast<size_t>(sliding_window_stride);
            bool is_first_window = true;
            size_t prev_source_end = 0;  // Track previous window's source end for overlap masking
            
            while (start < seq_len) {
                // Reserve 1 token for BOS if this is not the first window and BOS is enabled
                const bool prepend_bos = !is_first_window && add_bos_token && bos_id >= 0;
                const size_t effective_max = (is_first_window || !prepend_bos)
                    ? static_cast<size_t>(max_seq_len) 
                    : static_cast<size_t>(max_seq_len - 1);
                size_t end = std::min(seq_len, start + effective_max);
                
                TrainingSequence window;
                
                // For non-first windows, prepend BOS token (gated on add_bos_token config)
                if (prepend_bos) {
                    window.token_ids.push_back(bos_id);
                    window.targets.push_back(-1);  // BOS position masked
                    window.token_numeric_values.push_back(0.0f);
                    window.token_atom_mask.push_back(0);
                    window.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
                    window.token_atom_flags.push_back(0);
                    for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) {
                        window.token_text_features.push_back(0);
                    }
                    if (!seq.token_exec_slots.empty())
                        window.token_exec_slots.push_back(static_cast<int32_t>(-1));
                    if (!seq.slot_selection_targets.empty())
                        window.slot_selection_targets.push_back(
                            GRIM::Execution::SlotSelectionTarget{GRIM::Execution::SlotSelectionTargetKind::Ignore, -1});
                    bos_prepended++;
                }
                
                // Copy window content
                window.token_ids.insert(window.token_ids.end(),
                    seq.token_ids.begin() + start, seq.token_ids.begin() + end);
                window.targets.insert(window.targets.end(),
                    seq.targets.begin() + start, seq.targets.begin() + end);
                window.token_numeric_values.insert(window.token_numeric_values.end(),
                    seq.token_numeric_values.begin() + start, seq.token_numeric_values.begin() + end);
                window.token_atom_mask.insert(window.token_atom_mask.end(),
                    seq.token_atom_mask.begin() + start, seq.token_atom_mask.begin() + end);
                // Atom side channel — share parent sequence's AtomTable
                window.atom_table = seq.atom_table;
                window.atom_entry_ids.insert(window.atom_entry_ids.end(),
                    seq.atom_entry_ids.begin() + start, seq.atom_entry_ids.begin() + end);
                window.token_atom_flags.insert(window.token_atom_flags.end(),
                    seq.token_atom_flags.begin() + start, seq.token_atom_flags.begin() + end);
                // GRMT v4: slice text features (16 values per token)
                window.token_text_features.insert(window.token_text_features.end(),
                    seq.token_text_features.begin() + start * GRIM::Tokenizer::kTextFeatureDim,
                    seq.token_text_features.begin() + end * GRIM::Tokenizer::kTextFeatureDim);

                if (!seq.token_exec_slots.empty()) {
                    window.token_exec_slots.insert(window.token_exec_slots.end(),
                        seq.token_exec_slots.begin() + static_cast<ptrdiff_t>(start),
                        seq.token_exec_slots.begin() + static_cast<ptrdiff_t>(end));
                }
                if (!seq.slot_selection_targets.empty()) {
                    window.slot_selection_targets.insert(window.slot_selection_targets.end(),
                        seq.slot_selection_targets.begin() + static_cast<ptrdiff_t>(start),
                        seq.slot_selection_targets.begin() + static_cast<ptrdiff_t>(end));
                }

                
                // Mask first position if it's the first window (BOS already there)
                // For non-first windows, BOS was prepended above with target=-1
                if (is_first_window && !window.targets.empty()) {
                    window.targets[0] = -1;  // Mask BOS position
                }
                
                // Issue #143: Mask overlap prefix targets in non-first windows.
                // With stride < max_seq_len, the first (prev_source_end - start)
                // tokens were already trained on in the previous window. Mask them
                // to prevent double-training on the same targets.
                //
                // Issue #147: Subtract 1 from overlap_len. The position at
                // (prev_source_end - 1) was the LAST position in the previous
                // window, which was already  masked there (last-position mask).
                // Its target was NEVER trained. If we mask it here too, we create
                // a one-token training gap at every window boundary. By reducing
                // overlap by 1, this window trains that target instead.
                if (!is_first_window && prev_source_end > start) {
                    const size_t raw_overlap = prev_source_end - start;
                    const size_t overlap_len = (raw_overlap > 0) ? (raw_overlap - 1) : 0;
                    const size_t bos_offset = prepend_bos ? 1 : 0;  // Skip BOS (already masked)
                    for (size_t i = bos_offset; i < bos_offset + overlap_len && i < window.targets.size(); ++i) {
                        window.targets[i] = -1;
                    }
                }
                
                // Mask last position for window boundary.
                // BatchPayload requires targets.back() == -1 unconditionally;
                // val no longer gets a special-cased "keep the final target"
                // path because BatchPayload would have masked it anyway.
                if (!window.targets.empty()) {
                    window.targets.back() = -1;
                }
                
                // Issue #146: Inject EOS at end of non-final windows.
                // Without this, ~55% of training windows have NO EOS token,
                // so the model never learns when to stop generating.
                // The last position is already target-masked above, so replacing
                // its token_id with EOS costs nothing — the model sees EOS as
                // input and the second-to-last position learns to predict EOS.
                const bool is_final_window = (end == seq_len);
                if (!is_final_window && eos_id >= 0 && !window.token_ids.empty()) {
                    window.token_ids.back() = eos_id;
                    window.token_numeric_values.back() = 0.0f;
                    window.token_atom_mask.back() = 0;
                    if (!window.token_exec_slots.empty())
                        window.token_exec_slots.back() = static_cast<int32_t>(-1);
                    if (!window.slot_selection_targets.empty())
                        window.slot_selection_targets.back() =
                            GRIM::Execution::SlotSelectionTarget{GRIM::Execution::SlotSelectionTargetKind::Ignore, -1};
                    // Clear text features for the replaced position
                    const size_t last_tf_start = (window.token_ids.size() - 1) * GRIM::Tokenizer::kTextFeatureDim;
                    for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) {
                        window.token_text_features[last_tf_start + i] = 0;
                    }
                    // Second-to-last position learns to predict EOS
                    if (window.targets.size() >= 2) {
                        window.targets[window.targets.size() - 2] = eos_id;
                    }
                }
                
                // Variable-length window — BatchPayload owns padding.
                windowed.push_back(std::move(window));
                generated_windows++;
                
                prev_source_end = end;  // Track for overlap masking in next window
                if (end == seq_len) break;
                start += stride;
                is_first_window = false;
            }
        }
        
        sequences = std::move(windowed);
        if (long_seq_count > 0) {
            logger.log("Sliding window (" + split_name + "): " +
                       std::to_string(long_seq_count) + " long sequences expanded into " +
                       std::to_string(generated_windows) + " windows" +
                       " (BOS prepended to " + std::to_string(bos_prepended) + " mid-sequence windows)");
        }
    };

    applySlidingWindows(data.train_seqs, "train");
    applySlidingWindows(data.val_seqs, "val");
    
    // HARD FILTER: Remove any sequences still exceeding max_seq_len after sliding window
    // This catches cached .grmt files with old sequence lengths
    auto filterOverlong = [&](std::vector<TrainingSequence>& sequences, const std::string& split_name) {
        size_t before = sequences.size();
        sequences.erase(
            std::remove_if(sequences.begin(), sequences.end(),
                [max_seq_len](const TrainingSequence& seq) {
                    return static_cast<int>(seq.token_ids.size()) > max_seq_len;
                }),
            sequences.end());
        size_t removed = before - sequences.size();
        if (removed > 0) {
            logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) + 
                       " sequences exceeding max_seq_len=" + std::to_string(max_seq_len));
        }
    };
    filterOverlong(data.train_seqs, "train");
    filterOverlong(data.val_seqs, "val");

    // HARD FILTER: Remove sequences with too few valid tokens after masking
    // This prevents "valid_tokens=0" errors during loss computation
    // (position 0 is always masked as BOS, final position is masked as boundary)
    auto filterShortSequences = [&](std::vector<TrainingSequence>& sequences, const std::string& split_name) {
        if (min_seq_valid_tokens <= 0) return;  // Disabled if <= 0
        size_t before = sequences.size();
        sequences.erase(
            std::remove_if(sequences.begin(), sequences.end(),
                [min_seq_valid_tokens](const TrainingSequence& seq) {
                    // Count valid targets: excludes position 0 (BOS) and final position (boundary)
                    // Mirrors the masking logic in BatchPayload.cu::buildBatchPayload()
                    int valid = 0;
                    for (size_t i = 1; i + 1 < seq.targets.size(); ++i) {
                        if (seq.targets[i] >= 0) valid++;
                    }
                    return valid < min_seq_valid_tokens;
                }),
            sequences.end());
        size_t removed = before - sequences.size();
        if (removed > 0) {
            logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) + 
                       " sequences with < " + std::to_string(min_seq_valid_tokens) + " valid tokens");
        }
    };
    filterShortSequences(data.train_seqs, "train");
    filterShortSequences(data.val_seqs, "val");
    
    // Build views and catalogs
    data.train_views.reserve(data.train_seqs.size());
    for (uint32_t i = 0; i < data.train_seqs.size(); ++i) {
        data.train_views.push_back(&data.train_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.train_seqs[i].token_ids.size());
        data.train_catalog.add(len, len, 0, 0, 0);
    }
    
    data.val_views.reserve(data.val_seqs.size());
    for (uint32_t i = 0; i < data.val_seqs.size(); ++i) {
        data.val_views.push_back(&data.val_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.val_seqs[i].token_ids.size());
        data.val_catalog.add(len, len, 0, 0, 0);
    }
    
    logger.log("Train sequences: " + std::to_string(data.train_seqs.size()));
    logger.log("Val sequences: " + std::to_string(data.val_seqs.size()));
    
    return data;
}

std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const StartupConfig& config,
    uint32_t vocab_size,
    uint64_t xavier_seed,
    TrainingLogger& logger,
    std::string& loaded_checkpoint_path) {
    loaded_checkpoint_path.clear();
    
    logger.log("Initializing model with xavier_seed=" + std::to_string(xavier_seed) + "...");
    
    const auto& arch = config.architecture;
    const auto& hp = config.hyperparameters;
    
    GRIM::LanguageModelConfig model_config;
    // Copy all architecture fields from validated ModelArchitecture (Single Source of Truth)
    static_cast<GRIM::HyperParameters::ModelArchitecture&>(model_config) = arch;
    model_config.max_seq_len = config.max_seq_len;  // Override: uses StartupConfig's derived value
    model_config.vocab_size = vocab_size;
    model_config.vocab_path = config.paths.vocab_path;
    model_config.infer_vocab_from_file = true;
    model_config.causal_mask = true;
    model_config.use_pre_norm = true;
    model_config.fuse_qkv = true;
    model_config.use_bias = true;
    model_config.use_gpu = true;
    model_config.use_flash_attention = hp.use_flash_attention;
    model_config.min_seq_len_for_flash = hp.min_seq_len_for_flash;
    logger.log("Flash attention: enabled=" + std::string(model_config.use_flash_attention ? "true" : "false") +
               ", min_seq_len=" + std::to_string(model_config.min_seq_len_for_flash));
    
    // ScratchBlock reasoning config (loaded from ai_config.json via config.hyperparameters)
    model_config.use_scratch_block = hp.scratch_block_reasoning_enabled;
    model_config.scratch_block_atom_embedding_dim = hp.scratch_block_reasoning_atom_embedding_dim;
    model_config.scratch_block_max_atoms = hp.scratch_block_reasoning_max_atoms;
    model_config.scratch_block_atom_scale = hp.scratch_block_reasoning_atom_scale;
    
    // Compute derived values (head_dim = d_model / num_heads)
    model_config.computeDerivedValues();
    
    logger.log("ScratchBlock reasoning: enabled=" + std::string(model_config.use_scratch_block ? "true" : "false") +
              ", atom_embedding_dim=" + std::to_string(model_config.scratch_block_atom_embedding_dim) +
              ", max_atoms=" + std::to_string(model_config.scratch_block_max_atoms) +
              ", atom_scale=" + std::to_string(model_config.scratch_block_atom_scale));

    model_config.execution_block_enabled = hp.execution_block_enabled;
    model_config.scratch_block_execution_first_type_only = hp.scratch_block_execution_first_type_only;
    model_config.execution_block_layer = hp.execution_block_layer;
    model_config.execution_block_num_ops = hp.execution_block_num_ops;
    model_config.execution_block_num_slots = hp.execution_block_num_slots;
    model_config.execution_block_num_steps = hp.execution_block_num_steps;
    model_config.execution_block_d_key = hp.execution_block_d_key;
    model_config.execution_block_d_type = hp.execution_block_d_type;
    model_config.execution_block_cross_attn_head_dim = hp.execution_block_cross_attn_head_dim;
    model_config.execution_block_cross_attn_topk = hp.execution_block_cross_attn_topk;
    model_config.execution_block_usage_decay = hp.execution_block_usage_decay;
    model_config.execution_block_diversity_kappa = hp.execution_block_diversity_kappa;
    model_config.execution_block_temp_start = hp.execution_block_temp_start;
    model_config.execution_block_temp_end = hp.execution_block_temp_end;
    model_config.execution_block_temp_schedule = hp.execution_block_temp_schedule;
    model_config.execution_block_entropy_weight = hp.execution_block_entropy_weight;
    model_config.step_x_multiplier = hp.execution_step_x_multiplier;
    model_config.step_y_multiplier = hp.execution_step_y_multiplier;
    model_config.step_y_overrides_x = hp.execution_step_y_overrides_x;
    model_config.entropy_aux_weight = hp.execution_entropy_aux_weight;
    model_config.value_match_epsilon = hp.execution_value_match_epsilon;
    model_config.final_slot_consistency_weight = hp.execution_final_slot_consistency_weight;
    model_config.execution_block_transition_hard_threshold = hp.execution_block_transition_hard_threshold;
    model_config.execution_block_gate_warmup_steps = hp.execution_block_gate_warmup_steps;
    model_config.execution_block_causal_w1_transition = hp.execution_block_causal_w1_transition;
    model_config.div_invalid_penalty_weight = hp.execution_div_invalid_penalty_weight;
    model_config.div_magnitude_penalty_weight = hp.execution_div_magnitude_penalty_weight;
    model_config.arg_reinforce_weight = hp.execution_arg_reinforce_weight;
    model_config.arg_reinforce_baseline_decay = hp.execution_arg_reinforce_baseline_decay;
    model_config.structured_ce_enabled = hp.structured_ce_enabled;
    model_config.structured_ce_weight  = hp.structured_ce_weight;
    if (model_config.structured_ce_enabled && model_config.structured_ce_weight <= 0.0f) {
        throw std::runtime_error("structured_ce_enabled=true but structured_ce_weight=" +
                                 std::to_string(model_config.structured_ce_weight) + " (must be > 0)");
    }

    // Decode-time slot selector config
    model_config.selector_enabled = hp.selector_enabled;
    model_config.selector_d_selector = hp.selector_d_selector;
    model_config.selector_selection_margin = hp.selector_selection_margin;
    model_config.selector_supervision_weight = hp.selector_supervision_weight;

    logger.log("ExecutionBlock: enabled=" + std::string(model_config.execution_block_enabled ? "true" : "false") +
              ", V=" + std::to_string(model_config.execution_block_num_slots) +
              ", K=" + std::to_string(model_config.execution_block_num_steps) +
              ", ops=" + std::to_string(model_config.execution_block_num_ops) +
              ", layer=" + std::to_string(model_config.execution_block_layer) +
              ", execution_first_type_only=" +
              std::string(model_config.scratch_block_execution_first_type_only ? "true" : "false"));

    // LM Head centering configuration (Issue #37 / #40)
    model_config.lm_head_center_hidden_states = hp.lm_head_center_hidden_states;
    model_config.project_out_pc1 = hp.project_out_pc1;
    model_config.pc1_power_iters = hp.pc1_power_iters;
    model_config.center_logits = hp.center_logits;
    model_config.center_encoder_residuals = hp.center_encoder_residuals;
    model_config.lm_head_freeze_final_rms_gamma = hp.lm_head_freeze_final_rms_gamma;

    logger.log("LM Head centering: center_hidden_states=" + std::string(model_config.lm_head_center_hidden_states ? "true" : "false") +
              ", project_out_pc1=" + std::string(model_config.project_out_pc1 ? "true" : "false") +
              ", pc1_power_iters=" + std::to_string(model_config.pc1_power_iters) +
              ", center_logits=" + std::string(model_config.center_logits ? "true" : "false") +
              ", center_encoder_residuals=" + std::string(model_config.center_encoder_residuals ? "true" : "false") +
              ", freeze_final_rms_gamma=" + std::string(model_config.lm_head_freeze_final_rms_gamma ? "true" : "false"));
    
    // Issue #109: LayerScale configuration (learnable residual scaling from CaiT paper)
    model_config.use_layer_scale = hp.use_layer_scale;
    model_config.layer_scale_init = hp.layer_scale_init;
    
    // QK-norm: per-head RMSNorm on Q and K (Gemma-2 style)
    model_config.qk_norm_enabled = hp.qk_norm_enabled;
    
    logger.log("LayerScale: enabled=" + std::string(model_config.use_layer_scale ? "true" : "false") +
              ", init=" + std::to_string(model_config.layer_scale_init) +
              " | QK-norm: " + std::string(model_config.qk_norm_enabled ? "ENABLED" : "disabled"));
    
    // Multi-token prediction (MTP) - auxiliary heads
    model_config.mtp_enabled = hp.mtp_enabled;
    model_config.mtp_k = hp.mtp_k;
    model_config.mtp_alpha = hp.mtp_alpha;
    model_config.mtp_alpha_warmup_steps = hp.mtp_alpha_warmup_steps;
    if (model_config.mtp_enabled && (model_config.mtp_k <= 0 || model_config.mtp_alpha <= 0.0f)) {
        throw std::runtime_error("multi_token_prediction: when enabled, k and alpha must be > 0 (k=" +
            std::to_string(model_config.mtp_k) + " alpha=" + std::to_string(model_config.mtp_alpha) + ")");
    }

    // Hardcoded Hidden States Diagnostic (Issue #42)
    model_config.hardcoded_hidden_pattern = static_cast<GRIM::LanguageModelConfig::HardcodedPattern>(hp.hardcoded_hidden_pattern);
    model_config.hardcoded_log_every_n_batches = hp.hardcoded_log_every_n_batches;
    
    if (model_config.hardcoded_hidden_pattern != GRIM::LanguageModelConfig::HardcodedPattern::DISABLED) {
        logger.log("⚠️  HARDCODED HIDDEN STATES DIAGNOSTIC ENABLED: pattern=" + std::to_string(static_cast<int>(model_config.hardcoded_hidden_pattern)) +
                  ", log_every_n=" + std::to_string(model_config.hardcoded_log_every_n_batches));
        logger.log("⚠️  Encoder output will be REPLACED with synthetic patterns - this is a DIAGNOSTIC MODE ONLY!");
    }
    
    // Cache sizing - Use configured batch_size directly (stability override already applied in loadConfiguration)
    const int actual_batch_size = config.hyperparameters.batch_size;
    
    // Cache sequence length derives directly from configured max_seq_len
    const uint32_t seq_cap = static_cast<uint32_t>(model_config.max_seq_len);
    
    // Cache allocation: allocate for the ACTUAL batch size, not some derived "max"
    // No arbitrary margins - if you need more, increase batch_size in config
    model_config.max_cached_batch = actual_batch_size;
    model_config.max_cached_seq_len = seq_cap;
    
    // Token budget is just batch * seq_len (used for logits allocation limit)
    const uint32_t token_budget = static_cast<uint32_t>(actual_batch_size) * seq_cap;
    model_config.max_tokens_per_batch = static_cast<int>(token_budget);
    
    logger.log("Cache allocation: batch=" + std::to_string(actual_batch_size) + 
               ", seq_len=" + std::to_string(seq_cap) + 
               ", tokens=" + std::to_string(token_budget));
    
    // Activation quantization
    model_config.activation_quantization.enabled = hp.activation_quantization_enabled;
    model_config.activation_quantization.apply_to_embeddings = hp.activation_quantization_apply_to_embeddings;
    model_config.activation_quantization.apply_to_encoder_outputs = hp.activation_quantization_apply_to_encoder_outputs;
    model_config.activation_quantization.apply_to_layer_caches = hp.activation_quantization_apply_to_layer_caches;
    model_config.activation_quantization.apply_to_qkv_cache = hp.activation_quantization_apply_to_qkv_cache;
    model_config.activation_quantization.apply_to_logits = hp.activation_quantization_apply_to_logits;
    model_config.activation_quantization.scale = hp.activation_quantization_scale;
    model_config.activation_quantization.clip_min = hp.activation_quantization_clip_min;
    model_config.activation_quantization.clip_max = hp.activation_quantization_clip_max;
    model_config.activation_quantization.zero_point = hp.activation_quantization_zero_point;
    model_config.activation_quantization.symmetric = hp.activation_quantization_symmetric;
    
    logger.log("Model architecture: d_model=" + std::to_string(arch.d_model) + 
               ", num_layers=" + std::to_string(arch.num_layers) + 
               ", num_heads=" + std::to_string(arch.num_heads) + 
               ", d_ff=" + std::to_string(arch.d_ff) +
               ", max_seq_len=" + std::to_string(model_config.max_seq_len));
    logger.log("Derived hyperparameters: d_ff=" + std::to_string(arch.d_ff) + " (d_model*4)" +
               ", attention_dropout=" + std::to_string(arch.attention_dropout) + " (=dropout_rate)" +
               ", min_seq_valid_tokens=" + std::to_string(hp.min_seq_valid_tokens) +
               ", min_seq_len_for_flash=" + std::to_string(hp.min_seq_len_for_flash) +
               ", cosine_decay_min_lr=" + std::to_string(hp.cosine_decay_min_lr) +
               ", cosine_warm_restarts=" + std::string(hp.cosine_warm_restarts ? "true" : "false") +
               ", scratch_max_tokens_per_block=" + std::to_string(hp.scratch_max_tokens_per_block) +
               ", warmup_fraction=" + std::to_string(hp.warmup_fraction) +
               " (warmup_steps derived in Phase2 from fraction * total_steps)");
    logger.log("Derived hyperparameters (cont): d_key=" + std::to_string(hp.execution_block_d_key) +
               ", cross_attn_head_dim=" + std::to_string(hp.execution_block_cross_attn_head_dim) +
               ", atom_embedding_dim=" + std::to_string(hp.scratch_block_reasoning_atom_embedding_dim) +
               ", stability_batch=" + std::to_string(hp.stability_override_batch_size) +
               ", stability_max_seq=" + std::to_string(hp.stability_override_max_seq_len) +
               ", stability_clip_per_token=" + std::to_string(hp.stability_override_clip_per_token) +
               ", stability_lr_min=" + std::to_string(hp.stability_override_lr_min) + ")");
    
    // STEP 0: Initialize CUDA device context FIRST (REQUIRED before any stream/allocation operations)
    // This ensures CUDA driver is loaded and device context exists before StreamController creates streams.
    // MUST be done exactly once at the start of training, not in StreamController or initGPU.
    {
        logger.log("Initializing CUDA device context...");
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaSetDevice(0) failed: ") + cudaGetErrorString(err));
        }
        
        // Force context creation by triggering a no-op CUDA operation
        err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidValue) {
            throw std::runtime_error(std::string("FATAL: CUDA context creation failed: ") + cudaGetErrorString(err));
        }
        
        // Verify device is accessible
        int device;
        err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDevice() failed: ") + cudaGetErrorString(err));
        }
        
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDeviceProperties() failed: ") + cudaGetErrorString(err));
        }
        
        logger.log("✓ CUDA device initialized: " + std::string(props.name));
    }
    
    // StreamController is owned by TrainingState 
    // Initialize stream_ctrl after CUDA context exists
    
    auto model = std::make_unique<GRIM::LanguageModel>(model_config);
    
    // ═══════════════════════════════════════════════════════════════
    // CORRUPTION DIAGNOSTIC: Check batch_prep vector integrity after each step
    // STEP 1: Initialize just the stream controller part of TrainingState
    // This creates CUDA streams (CUDA context must exist from STEP 0)
    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;
        
        if (!model->getTrainingState().stream_ctrl.initialize(stream_config)) {
            throw std::runtime_error("FATAL: Failed to initialize StreamController");
        }
        logger.log("✓ StreamController initialized");
    }
    
    // STEP 2: Initialize cuBLAS handle (needed by encoder layers in initGPU)
    logger.log("Initializing cuBLAS handle...");
    model->initCuBLASHandle();
    logger.log("✓ cuBLAS handle initialized with Tensor Core acceleration");
    
    // STEP 2.5: Initialize RoPE BEFORE encoder construction (CRITICAL!)
    logger.log("Initializing RoPE (required before encoder construction)...");
    model->initPBM();
    logger.log("✓ RoPE initialized");
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 2.75: Store Autograd Seed (Session 7: TrainingTensors deleted)
    // Weight init seed is stored on TrainingState.weight_init_seed.
    // Pattern B layers use it for layer-specific weight seeding.
    // layer-specific weight seeding. All weight allocation is now done by
    // Pattern B layers (EmbeddingLayer, EncodingLayer, LMHeadLayer, ScratchBlockLayer).
    // ═══════════════════════════════════════════════════════════════
    {
        logger.log("Storing autograd weight init seed...");
        
        // CRITICAL: Use xavier_seed (derived init_seed from RNG context), not config.hyperparameters.seed
        // The xavier_seed is the computed hierarchical seed from initializeRNG()
        model->getTrainingState().initializeAutogradSeed(xavier_seed);
        logger.log("✓ Autograd seed stored (weight_init_seed=" + std::to_string(xavier_seed) + ")");
    }
    
    // STEP 3: Initialize GPU encoder (uses cuBLAS handle from step 2, seed from step 2.75)
    // NOTE: initGPU() creates Pattern B layers which self-allocate their weights
    logger.log("Initializing GPU encoder...");
    model->initGPU();
    logger.log("✓ GPU encoder fully initialized");
    
    // STEP 4: Finish TrainingState initialization (grad buffers, activation caches)
    // NOTE: Layers already set up in step 3, this just does the rest
    logger.log("Initializing TrainingState (grad buffers, activation caches)...");
    model->initTrainingState();
    logger.log("✓ TrainingState fully initialized");

    // Build parameter groups for optimizer and grad-norm (required even when logit_update_trace is off).
    // Otherwise Phase2 allocateGradNormScratch gets max_groups=0 and fails.
    model->buildParameterGroups();

    // ═══════════════════════════════════════════════════════════════
    // tie_embeddings pointer verification (Step C: non-negotiable truth)
    // Log raw pointers, ownership, and optimizer group count
    // ═══════════════════════════════════════════════════════════════
    {
        const float* emb_w_ptr = model->getEmbeddingLayer()->tokenWeights().data;
        const float* lm_w_ptr  = model->getLmHeadLayer()->weights().data;
        const float* emb_g_ptr = model->getEmbeddingLayer()->tokenWeights().grad_data();
        const float* lm_g_ptr  = model->getLmHeadLayer()->weights().grad_data();
        const bool cfg_tied = model->getConfig().tie_embeddings;
        const bool lm_owns = model->getLmHeadLayer()->ownsWeights();
        const bool ptrs_same = (emb_w_ptr == lm_w_ptr);
        const bool grads_same = (emb_g_ptr == lm_g_ptr);

        // Count optimizer groups referencing each buffer
        int emb_groups = 0, lm_groups = 0;
        for (const auto& g : model->parameterGroups()) {
            if (g.tensor && g.tensor->data == emb_w_ptr) ++emb_groups;
            if (g.tensor && g.tensor->data == lm_w_ptr)  ++lm_groups;
        }

        std::ostringstream tie_msg;
        tie_msg << "[TieEmbeddingsVerify] config.tie_embeddings=" << (cfg_tied ? "true" : "false")
                << " lm_owns_weights=" << (lm_owns ? "true" : "false")
                << "\n  emb_weight_ptr=" << (const void*)emb_w_ptr
                << " lm_weight_ptr=" << (const void*)lm_w_ptr
                << " SAME=" << (ptrs_same ? "YES" : "NO")
                << "\n  emb_grad_ptr=" << (const void*)emb_g_ptr
                << " lm_grad_ptr=" << (const void*)lm_g_ptr
                << " SAME=" << (grads_same ? "YES" : "NO")
                << "\n  optimizer_groups: emb_buffer=" << emb_groups
                << " lm_buffer=" << lm_groups
                << " total=" << model->parameterGroups().size();
        if (cfg_tied && !ptrs_same) {
            tie_msg << "\n  [BUG] tie_embeddings=true but pointers differ!";
        }
        if (!cfg_tied && ptrs_same) {
            tie_msg << "\n  [BUG] tie_embeddings=false but pointers are SAME!";
        }
        if (cfg_tied && (emb_groups + lm_groups) > lm_groups) {
            // When tied, only lm_groups should reference the buffer (not emb separately)
            tie_msg << "\n  [WARNING] tied buffer has " << emb_groups << " emb refs + " << lm_groups << " lm refs (expect lm only)";
        }
        logger.log(tie_msg.str());
    }

    // UpdateProbe subsystem deleted (Rule 26): probe was instrumentation that
    // was never wired to actually populate the weights/grads sample buffers,
    // so the consumer in Phase2 never fired. logit_update_trace_enabled still
    // controls the PostLoss cached_logits trace below.

    // Configure scratch blocks
    fprintf(stderr, "[initializeModel] DIAG: About to configure scratch blocks (pool_init=%d)\n", (int)model->isScratchPoolInitialized()); fflush(stderr);
    if (model->isScratchPoolInitialized()) {
        model->configureScratchPool(config.scratch.enabled);
        if (config.scratch.enabled) {
            logger.log("✓ Scratch blocks enabled (" +
                      std::to_string(config.scratch.num_blocks) + " blocks, sized from max_cached_tokens)");
        }
    }
    
    model->setLossOptions(config.loss_options);
    
    // Log loss configuration
    {
        std::ostringstream loss_msg;
        loss_msg << "[LossConfig] startup: "
                 << "label_smoothing=" << (config.loss_options.label_smoothing_enabled ? "ON" : "off")
                 << " focal=" << (config.loss_options.focal_enabled ? "ON" : "off")
                 << " distill=" << (config.loss_options.distillation_enabled ? "ON" : "off")
                 << " pref=" << (config.loss_options.preference_enabled ? "ON" : "off")
                 << " entropy_reg=" << (config.loss_options.entropy_reg_enabled ? "ON" : "off")
                 << " ent_lambda=" << config.loss_options.entropy_reg_lambda
                 << " class_balanced=" << (config.loss_options.class_balanced_enabled ? "ON" : "off")
                 << " cb_beta=" << config.loss_options.class_balanced_beta;

        logger.log(loss_msg.str());
    }
    
#ifdef USE_CUDA
    // Verify encoder weights are initialized (Pattern B layers handle Xavier init via Tensor::xavier_uniform_)
    {
        auto* gpu_encoder = &model->getGpuEncoder();
        const auto& cfg = model->getConfig();
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            auto* enc = gpu_encoder->getLayer(layer);
            if (!enc) {
                throw std::runtime_error("Encoder layer " + std::to_string(layer) + " is NULL after initGPU");
            }
        }
        logger.log("✓ All " + std::to_string(cfg.num_layers) + " encoder layers verified");
    }
#endif
    
    // Try to load checkpoint - scan newest-to-oldest and stop at first valid file.
    std::vector<std::pair<int, std::string>> checkpoint_candidates;
    if (fs::exists(config.paths.checkpoint_dir) && fs::is_directory(config.paths.checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(config.paths.checkpoint_dir)) {
            const auto& p = entry.path();
            if (p.extension() == ".bin" && p.stem().string().rfind("checkpoint_epoch_", 0) == 0) {
                // Extract epoch number from "checkpoint_epoch_N"
                std::string stem = p.stem().string();
                std::string epoch_str = stem.substr(std::string("checkpoint_epoch_").size());
                try {
                    int epoch = std::stoi(epoch_str);
                    checkpoint_candidates.emplace_back(epoch, p.string());
                } catch (const std::exception& e) {
                    logger.log("[WARNING] Skipping malformed checkpoint filename: " + stem + " (" + e.what() + ")");
                }
            }
        }
    }

    std::sort(checkpoint_candidates.begin(), checkpoint_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    bool loaded_checkpoint = false;
    for (const auto& [epoch, checkpoint_path] : checkpoint_candidates) {
        logger.log("Found checkpoint candidate: " + checkpoint_path + " (epoch " + std::to_string(epoch) + ")");
        if (model->load(checkpoint_path)) {
            logger.log("✓ Loaded weights from checkpoint: " + checkpoint_path);
            loaded_checkpoint = true;
            loaded_checkpoint_path = checkpoint_path;
            break;
        }
        logger.log("⚠ Failed to load checkpoint candidate, trying older checkpoint");
    }

    if (!loaded_checkpoint) {
        if (!checkpoint_candidates.empty()) {
            logger.log("⚠ No loadable checkpoint found, starting fresh");
        } else {
            logger.log("No checkpoint found, starting fresh");
        }
    }
    
    logger.log("✓ Model initialized");
    return model;
}

OptimizerContext initializeOptimizer(
    GRIM::LanguageModel& model,
    const StartupConfig& config,
    TrainingLogger& logger) {
    
    OptimizerContext ctx;
    const auto& hp = config.hyperparameters;
    
    logger.log("Initializing optimizer state...");
    
    // Optimizer state (AdamW hyperparameters are defined as constants in AdamW_Kernal_GPU.cu)
    ctx.optimizer_state.step = 0;
    
    // Soft restart controller — loss-detection thresholds (loss_increase_threshold,
    // max_step_window) now live in GRIM::Loss::LossSignalConfig (validation_delta_threshold)
    // and are wired in Phase2 when the LossSignalBus is constructed. SoftRestartConfig
    // owns ONLY cooldown_steps now.
    GRIM::SoftRestart::SoftRestartConfig sr_cfg;
    sr_cfg.cooldown_steps = hp.soft_restart_cooldown_steps;
    ctx.soft_restart_controller = GRIM::SoftRestart::SoftRestartController(sr_cfg);
    
    // Gradient accumulation now tracked directly via hyperparameters
    // ctx.optimizer.current_micro_step is reset at start of each accumulation window
    const int accum_steps = std::max(1, hp.gradient_accumulation_steps);
    logger.log("✓ Gradient accumulation configured: " + std::to_string(accum_steps) + 
               " steps (effective batch = " + std::to_string(hp.batch_size * accum_steps) + ")");
    ctx.current_micro_step = 0;
    
    // Verify ALL encoder layers initialized (prevents lazy init during forward pass)
    auto* gpu_encoder = &model.getGpuEncoder();
    for (int layer = 0; layer < model.getConfig().num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("Encoder layer " + std::to_string(layer) + " not initialized - "
                                     "ensure model.initGPU() completes all layers before training");
        }
    }
    
    logger.log("✓ Optimizer state initialized");
    return ctx;
}

// initializeRNG moved to Startup/Rng.cu

} // namespace Internal

//======================================================//
//  Phase1 Main Entry Point
//======================================================//

std::unique_ptr<TrainingContext> executePhase1(int argc, char** argv) {
    auto ctx = std::make_unique<TrainingContext>();
    
    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    EmitModuleInfo(ModuleId::Training, "  Phase 1: Startup & Initialization", 0);
    EmitModuleInfo(ModuleId::Training, "  GRIM-text GPU Training v3.0.0", 0);
    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    
    registerDefaultLoggingProfiles();
    
    // 1. Load configuration
    EmitModuleInfo(ModuleId::Training, "[Phase1] Loading configuration...", 0);
    ctx->config = Internal::loadConfiguration(argc, argv);
    EmitModuleInfo(ModuleId::Training, 
        std::string("[Phase1] ✓ Configuration loaded from: ") + ctx->config.paths.config_path.string(), 0);

    // 2. Initialize logging as early as possible (before cache prep)
    if (!ctx->config.paths.log_dir.empty()) {
        fs::create_directories(ctx->config.paths.log_dir);
    }
    if (!ctx->config.paths.status_path.empty()) {
        fs::path status_parent = fs::path(ctx->config.paths.status_path).parent_path();
        if (!status_parent.empty()) {
            fs::create_directories(status_parent);
        }
    }
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing logging...", 0);
        ctx->logging = Internal::initializeLogging(ctx->config.paths);
    
    // ================================================================
    //  Initialize unified BatchLogTape system
    // ================================================================
    {
        const auto& tape_cfg = ctx->config.hyperparameters.tape_logging;
        
        auto tc = GRIM::Logging::parseTapeConfig(
            tape_cfg.default_level.c_str(),
            tape_cfg.equation_csv_enabled,
            tape_cfg.stderr_enabled,
            tape_cfg.initial_capacity);
        
        // Apply per-group overrides from ai_config.json
        for (const auto& [group_name, level_name] : tape_cfg.group_overrides) {
            GRIM::Logging::applyGroupOverride(tc, group_name.c_str(), level_name.c_str());
        }
        
        ctx->logging.tape = std::make_unique<GRIM::Logging::BatchLogTape>(tc);
        
        // Create sinks
        std::string text_log_path = ctx->config.paths.log_dir + "/training_" + ctx->logging.session_id + "_tape.log";
        ctx->logging.text_sink = std::make_unique<GRIM::Logging::TextLogSink>(
            text_log_path.c_str(), /*also_stdout=*/false);
        ctx->logging.tape->addSink(ctx->logging.text_sink.get());
        
        if (tape_cfg.equation_csv_enabled) {
            std::string eq_csv_path = ctx->config.paths.log_dir + "/equation_log.csv";
            ctx->logging.equation_sink = std::make_unique<GRIM::Logging::CsvEquationSink>(
                eq_csv_path.c_str());
            ctx->logging.tape->addSink(ctx->logging.equation_sink.get());
        }
        
        if (tape_cfg.stderr_enabled) {
            ctx->logging.stderr_sink = std::make_unique<GRIM::Logging::StderrSink>();
            ctx->logging.tape->addSink(ctx->logging.stderr_sink.get());
        }
        
        // Set global tape pointer for layer-level code
        GRIM::Logging::setGlobalTape(ctx->logging.tape.get());
        
        ctx->logging.logger->log("BatchLogTape initialized: " + GRIM::Logging::dumpTapeConfig(tc));
    }
    
    // 2b. Auto-prepare training data/vocab from merged cache when needed.
    {
        const fs::path vocab_path = ctx->config.paths.vocab_path;
        const bool data_missing = !fs::exists(ctx->config.paths.data_path);
        const bool vocab_missing = vocab_path.empty() || !fs::exists(vocab_path);

        if (ctx->config.force_rebuild_vocab || data_missing || vocab_missing) {
            if (ctx->logging.logger) {
                ctx->logging.logger->log("[Phase1] Auto-preparing training data/vocab...");
            }
            GRIM::Config::GrimTextPaths grim_paths{};
            grim_paths.training_data = ctx->config.paths.data_path;
            grim_paths.vocab = ctx->config.paths.vocab_path;
            std::string training_path = ctx->config.paths.data_path;
            std::string vocab_path_str = ctx->config.paths.vocab_path;
            bool prepared = false;

            try {
                prepared = GRIM::PrepareTrainingDataFromCache(
                    grim_paths,
                    training_path,
                    vocab_path_str,
                    ctx->config.force_rebuild_vocab);
            } catch (const std::exception& e) {
                if (ctx->logging.logger) {
                    ctx->logging.logger->log(std::string("[Phase1] Auto-prepare exception: ") + e.what());
                }
                throw;
            }

            if (prepared) {
                ctx->config.paths.data_path = training_path;
                ctx->config.paths.vocab_path = vocab_path_str;
                if (ctx->logging.logger) {
                    ctx->logging.logger->log("[Phase1] Auto-prepare completed.");
                }
            } else {
                if (ctx->logging.logger) {
                    ctx->logging.logger->log("[Phase1] Auto-prepare failed.");
                }
                throw std::runtime_error(
                    "FATAL: training data/vocab missing and cache preparation failed");
            }
        }
    }

    // 3. Validate paths
    EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
    Internal::validatePaths(ctx->config.paths);
    EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

    EmitModuleInfo(ModuleId::Training, "Configuration:", 0);
    // Data paths + vocab + sequence counts are logged together with the
    // hyperparameter dump (post-harmonize) via DataStatsSnapshot below,
    // so derived ratios from HyperParameters_GPU.hpp are reflected and
    // the visual block is unified.
    
    // 4. Initialize tokenizer
    GRIM::Config::TokenizerConfig tok_cfg;
    if (auto snapshot = GRIM::Config::loadAiConfigSnapshot()) {
        if (snapshot->has_tokenizer) {
            tok_cfg = snapshot->tokenizer_config;
        }
    }
    ctx->tokenizer = Internal::initializeTokenizer(ctx->config.paths.vocab_path, tok_cfg, ctx->config.hyperparameters, *ctx->logging.logger);
    
    ctx->logging.logger->log("[DEBUG] Starting training data load...");
    // 5. Load training data
    ctx->data = Internal::loadTrainingData(
        ctx->config.paths.data_path,
        ctx->config.max_seq_len,
        ctx->config.hyperparameters.min_seq_valid_tokens,
        ctx->config.sliding_window_stride,
        tok_cfg.add_bos,
        tok_cfg.add_eos,
        ctx->tokenizer,
        *ctx->logging.logger);
    
    // Use vocab size extracted during data loading (no second load needed)
    ctx->config.actual_vocab_size = ctx->data.vocab_size;
    if (ctx->config.actual_vocab_size == 0) {
        throw std::runtime_error("FATAL: training data missing vocab_size; regenerate GRMT with tokenizer.totalVocabSize()");
    }
    if (ctx->config.actual_vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error("FATAL: training data vocab_size must include special+byte+atom ranges (>= " + 
            std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) + ")");
    }
    
    // CRITICAL: Validate vocab size matches tokenizer (detects stale .grmt files after atom encoding changes)
    const uint32_t tokenizer_vocab_size = ctx->tokenizer.totalVocabSize();
    if (ctx->config.actual_vocab_size != tokenizer_vocab_size) {
        std::ostringstream err;
        err << "FATAL: Vocab size mismatch!\n"
            << "  Training data (.grmt): " << ctx->config.actual_vocab_size << " tokens\n"
            << "  Current tokenizer:     " << tokenizer_vocab_size << " tokens\n"
            << "Delete .grmt files to force regeneration.";
        throw std::runtime_error(err.str());
    }
    // (Successful vocab match is logged as part of the unified data-stats
    // block emitted by dumpAllHyperparameters below.)
    
    // 6. Harmonize hyperparameters
    GRIM::HyperParameters::DerivationContext hp_ctx;
    hp_ctx.train_sequence_count = static_cast<int>(ctx->data.train_seqs.size());
    hp_ctx.validation_interval = ctx->config.hyperparameters.validation_interval;
    
    auto hp_logger = [&](const std::string& msg) { ctx->logging.logger->log(msg); };

    // Three primitives, in order:
    //   (A) validate inputs (fail loud)
    //   (B) compute derived schedule from (seq_count, batch_size, epochs) ratios
    //   (C) apply cross-field policy / clamps (mutates hp, logs adjustments)
    GRIM::HyperParameters::validateTrainingHyperparameters(ctx->config.hyperparameters);
    ctx->derived_schedule = GRIM::HyperParameters::computeDerivedSchedule(
        ctx->config.hyperparameters, hp_ctx);
    GRIM::HyperParameters::applyTrainingHyperparameterPolicy(
        ctx->config.hyperparameters, ctx->derived_schedule, hp_ctx, hp_logger);

    // Single-source hyperparameter dump (post-policy, includes derived
    // schedule values computed in HyperParameters_GPU.hpp). The
    // DataStatsSnapshot folds vocab + sequence-count info into the same
    // visual block.
    GRIMText::Training::DataStatsSnapshot data_stats;
    data_stats.data_path            = ctx->config.paths.data_path;
    data_stats.vocab_path           = ctx->config.paths.vocab_path;
    data_stats.tokenizer_vocab_size = tokenizer_vocab_size;
    data_stats.actual_vocab_size    = ctx->config.actual_vocab_size;
    data_stats.train_sequence_count = ctx->data.train_seqs.size();
    data_stats.val_sequence_count   = ctx->data.val_seqs.size();

    dumpAllHyperparameters(
        ctx->config.hyperparameters,
        &ctx->derived_schedule,
        &data_stats,
        [](const std::string& msg) { EmitModuleInfo(ModuleId::Training, msg, 0); });
    
    // 8. Initialize production-grade RNG system (BEFORE model init for Xavier seeding)
    ctx->rng = Internal::initializeRNG(ctx->config, *ctx->logging.logger);
    
    // 9. Initialize model (uses RNG init_seed for Xavier)
    //    loaded_checkpoint_path receives the path that ACTUALLY loaded
    //    so the optimizer sidecar can be matched exactly (no independent scan).
    try {
        ctx->model = Internal::initializeModel(
            ctx->config,
            ctx->config.actual_vocab_size,
            ctx->rng.init_seed,
            *ctx->logging.logger,
            ctx->loaded_checkpoint_path);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }
    
    // Handle save test mode
    if (ctx->config.save_test_mode) {
        ctx->logging.logger->log("========================================");
        ctx->logging.logger->log("  SAVE TEST MODE");
        ctx->logging.logger->log("========================================");
        std::string test_save_path = ctx->config.paths.checkpoint_dir + "/save_test.bin";
        ctx->logging.logger->log("Testing model->save() to: " + test_save_path);
        bool save_ok = ctx->model->save(test_save_path);
        if (save_ok) {
            EmitModuleInfo(ModuleId::Checkpoint, "✓ Save test PASSED", 0);
            if (fs::exists(test_save_path)) {
                auto file_size = fs::file_size(test_save_path);
                EmitModuleInfo(ModuleId::Checkpoint, 
                    std::string("  File size: ") + std::to_string(file_size) + " bytes", 0);
            }
        } else {
            EmitModuleError(ModuleId::Checkpoint, "✗ Save test FAILED", 0);
        }
        std::exit(save_ok ? 0 : 1);
    }
    
    // 10. Initialize optimizer
    ctx->optimizer = Internal::initializeOptimizer(*ctx->model, ctx->config, *ctx->logging.logger);
    
    // 10a. Restore optimizer state from the EXACT checkpoint that loaded weights.
    // CRITICAL: Do NOT rescan the checkpoint dir independently — that can pick a
    // different epoch's .opt sidecar if a newer .bin failed to load.
    if (!ctx->loaded_checkpoint_path.empty()) {
        std::string opt_path = optimizerSidecarPath(ctx->loaded_checkpoint_path);
        if (fs::exists(opt_path)) {
            ctx->logging.logger->log("Found optimizer sidecar for loaded checkpoint: " + opt_path);
            try {
                loadOptimizerState(*ctx, opt_path);
                ctx->logging.logger->log("✓ Optimizer state restored (step=" + 
                    std::to_string(ctx->optimizer.optimizer_state.step) +
                    ", global_step=" + std::to_string(ctx->global_step) +
                    ", best_val_loss=" + std::to_string(ctx->best_val_loss) +
                    ", epochs_completed=" + std::to_string(ctx->epochs_completed) + ")");
            } catch (const std::exception& e) {
                ctx->logging.logger->log(std::string("⚠ Optimizer sidecar load failed: ") + e.what());
                ctx->logging.logger->log("  Continuing with fresh optimizer state for checkpoint: " + ctx->loaded_checkpoint_path);
            }
        } else {
            ctx->logging.logger->log("⚠ No optimizer sidecar for loaded checkpoint: " + ctx->loaded_checkpoint_path);
            ctx->logging.logger->log("  Continuing with fresh optimizer state (moments reset, LR from step 0)");
        }
    }
    
    // 10b. Compute class-balanced loss weights if enabled
    // w_v = 1/freq(v)^β where freq(v) = count(v) / total_targets
    // Weights indexed by vocab token ID, uploaded to GPU once at startup.
    if (ctx->config.loss_options.class_balanced_enabled) {
        auto& ts = ctx->model->getTrainingState();
        const uint32_t vocab_size = ctx->config.actual_vocab_size;
        const float beta = ctx->config.loss_options.class_balanced_beta;
        cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
        
        // Count target frequencies across ALL training sequences
        std::vector<int64_t> target_counts(vocab_size, 0);
        int64_t total_targets = 0;
        for (const auto& seq : ctx->data.train_seqs) {
            for (int tgt : seq.targets) {
                if (tgt >= 0 && tgt < static_cast<int>(vocab_size)) {
                    target_counts[tgt]++;
                    total_targets++;
                }
            }
        }
        
        if (total_targets <= 0) {
            throw std::runtime_error("[class_balanced] total_targets=0 — no valid targets in training data");
        }
        
        // Compute weights: w_v = 1/freq(v)^β, clamped for unseen tokens
        // Unseen tokens get weight = max_weight (same as freq=1 token)
        std::vector<float> h_class_weights(vocab_size);
        float max_weight = 0.0f;
        int seen_count = 0;
        int unseen_count = 0;
        
        for (uint32_t v = 0; v < vocab_size; v++) {
            if (target_counts[v] > 0) {
                const float freq = static_cast<float>(target_counts[v]) / static_cast<float>(total_targets);
                h_class_weights[v] = 1.0f / std::pow(freq, beta);
                max_weight = std::max(max_weight, h_class_weights[v]);
                seen_count++;
            } else {
                h_class_weights[v] = 0.0f;  // placeholder, filled after loop
                unseen_count++;
            }
        }
        
        // Unseen tokens: clamp to max_weight (the rarest seen token's weight)
        // This prevents infinite weights while still giving rare tokens maximum upweight
        if (max_weight <= 0.0f) max_weight = 1.0f;  // Safety: shouldn't happen
        for (uint32_t v = 0; v < vocab_size; v++) {
            if (target_counts[v] == 0) {
                h_class_weights[v] = max_weight;
            }
        }
        
        // Upload to GPU
        const size_t weights_bytes = vocab_size * sizeof(float);
        cudaMallocOrThrow(reinterpret_cast<void**>(&ts.d_class_weights), weights_bytes, "phase1_class_weights");
        cudaMemcpyAsync(ts.d_class_weights, h_class_weights.data(), weights_bytes,
                         cudaMemcpyHostToDevice, stream);
        ts.class_weights_vocab_size = static_cast<int>(vocab_size);
        cudaStreamSynchronize(stream);
        
        // Log top-10 highest and lowest weight tokens for verification
        {
            std::vector<std::pair<float, uint32_t>> weight_pairs;
            for (uint32_t v = 0; v < vocab_size; v++) {
                if (target_counts[v] > 0) {
                    weight_pairs.push_back({h_class_weights[v], v});
                }
            }
            std::sort(weight_pairs.begin(), weight_pairs.end());
            
            std::ostringstream cb_msg;
            cb_msg << "[CLASS_BALANCED] β=" << beta
                   << " total_targets=" << total_targets
                   << " seen_tokens=" << seen_count
                   << " unseen_tokens=" << unseen_count
                   << " max_weight=" << max_weight;
            ctx->logging.logger->log(cb_msg.str());
            
            // Lowest weights = most frequent tokens
            cb_msg.str("");
            cb_msg << "[CLASS_BALANCED] Lowest weights (most frequent): ";
            for (int i = 0; i < std::min(10, (int)weight_pairs.size()); i++) {
                cb_msg << "tok" << weight_pairs[i].second 
                       << "(w=" << std::fixed << std::setprecision(2) << weight_pairs[i].first
                       << ",cnt=" << target_counts[weight_pairs[i].second] << ") ";
            }
            ctx->logging.logger->log(cb_msg.str());
            
            // Highest weights = rarest seen tokens
            cb_msg.str("");
            cb_msg << "[CLASS_BALANCED] Highest weights (rarest seen): ";
            for (int i = std::max(0, (int)weight_pairs.size() - 10); i < (int)weight_pairs.size(); i++) {
                cb_msg << "tok" << weight_pairs[i].second
                       << "(w=" << std::fixed << std::setprecision(2) << weight_pairs[i].first
                       << ",cnt=" << target_counts[weight_pairs[i].second] << ") ";
            }
            ctx->logging.logger->log(cb_msg.str());
            
            // Log ratio: max_weight / min_weight shows the dynamic range
            if (!weight_pairs.empty()) {
                float ratio = weight_pairs.back().first / weight_pairs.front().first;
                cb_msg.str("");
                cb_msg << "[CLASS_BALANCED] Dynamic range: max/min weight ratio = " 
                       << std::fixed << std::setprecision(1) << ratio << "x";
                ctx->logging.logger->log(cb_msg.str());
            }
        }
    }
    
    // 10d. Issue #60 DEBUG: Allocate gradient attribution buffers if enabled
    // This lets us separately track LM head vs embedding backward contributions
    // to debug the positive feedback loop causing mode collapse to the tracked token
    {
        auto& ts = ctx->model->getTrainingState();
        const auto& model_cfg = ctx->model->getConfig();
        
        // PRODUCTION: Disabled (causes GPU sync + D2H copies every backward pass)
        // Set to true only when debugging tied embedding gradient issues
        ts.debug_gradient_attribution = false;
        
        if (ts.debug_gradient_attribution) {
            const int vocab_size = static_cast<int>(model_cfg.vocab_size);
            const int d_model = model_cfg.d_model;
            cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
            
            ts.allocateDebugGradBuffers(vocab_size, d_model, stream);
            
            // Set global hooks for gradient capture (use Tensor.data for raw pointer)
            g_debug_lm_head_only_grad = ts.debug_lm_head_only_grad.data;
            g_debug_embedding_only_grad = ts.debug_embedding_only_grad.data;
            g_debug_grad_buffer_size = static_cast<size_t>(ts.debug_lm_head_only_grad.numel());
            g_debug_capture_enabled = true;
            
            ctx->logging.logger->log("✓ Issue #60 DEBUG: Gradient attribution buffers allocated");
            ctx->logging.logger->log("  LM_HEAD_ONLY buffer: " + std::to_string(ts.debug_lm_head_only_grad.numel() * sizeof(float) / (1024*1024)) + " MB");
            ctx->logging.logger->log("  Will log tracked collapse token gradient sources after each backward pass");
        }

        if (model_cfg.tie_embeddings) {
            ctx->logging.logger->log("✓ Tied weight gradients use PyTorch-style direct accumulation");
        }
    }
    
    // 11. Initialize telemetry lattice
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry lattice...", 0);
    ctx->telemetry.config.num_levels = 8;  // k ∈ [0,7]: strides [1,2,4,8,16,32,64,128]
    ctx->telemetry.config.num_streams = 48; // 0-4: core, 5-8: rho, 9-13: adam, 14-20: exec block, 21-26: EB/SB injection, 27-30: PBM, 31-34: rho raw, 35-37: RMS gamma, 38: rho rms-spread, 39-44: h↔W alignment, 45-46: unigram-dir cosine, 47: lm_head_w_rms_rms
    ctx->telemetry.config.hyperparams.beta_mu = 0.95f;
    ctx->telemetry.config.hyperparams.beta_a = 0.995f;
    ctx->telemetry.config.hyperparams.beta_delta = 0.90f;
    ctx->telemetry.config.hyperparams.beta_r = 0.85f;
    ctx->telemetry.config.hyperparams.beta_run = 0.80f;
    ctx->telemetry.config.hyperparams.beta_v = 0.90f;
    ctx->telemetry.config.hyperparams.k_out0 = 2.5f;
    ctx->telemetry.config.hyperparams.alpha_v = 1.5f;
    ctx->telemetry.config.hyperparams.epsilon = 1e-7f;
    ctx->telemetry.config.hyperparams.strict_mode = true; // Fail loud (Rule 20: no compatibility shims)
    ctx->telemetry.config.stream = ctx->model->getTrainingState().stream_ctrl.getPrimaryStream(); // Use same stream as training
    
    ctx->telemetry.lattice = std::make_unique<GRIM::Telemetry::TelemetryLattice>(ctx->telemetry.config);
    ctx->logging.logger->log("✓ Telemetry lattice: 8 levels, " + std::to_string(ctx->telemetry.config.num_streams) + " streams, GPU-resident");

    // 11a. Initialize telemetry CSV logger (per-step measured data export)
    {
        const std::string csv_path = ctx->config.paths.log_dir + "/telemetry_" + ctx->logging.session_id + ".csv";
        ctx->telemetry.csv_logger = std::make_unique<GRIM::Telemetry::TelemetryCsvLogger>(
            csv_path, *ctx->telemetry.lattice);
        ctx->logging.logger->log("✓ Telemetry CSV logger: " + csv_path);
    }

    // 11b. Initialize telemetry control (GPU-native kernel-based control)
    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry control...", 0);
    // Rule 20: Use model's actual token budget (already computed during cache allocation)
    const uint32_t token_budget = ctx->model->getConfig().max_tokens_per_batch;
    if (token_budget == 0) {
        throw std::runtime_error("FATAL: model token_budget is 0 - cache allocation failed");
    }
    ctx->telemetry.control_config.reference_tokens = static_cast<float>(token_budget);
    ctx->telemetry.control_config.reference_seq_len = static_cast<float>(ctx->config.max_seq_len);
    ctx->logging.logger->log("[DEBUG] ctx->config.max_seq_len = " + std::to_string(ctx->config.max_seq_len));
    
    // TelemetryControl configuration - load from global config
    const auto& hp = ctx->config.hyperparameters;
    
    // Spike thresholds (diagnostic only — no interventions)
    ctx->telemetry.control_config.spike_mild_threshold = hp.telemetry_spike_mild_threshold;
    ctx->telemetry.control_config.spike_moderate_threshold = hp.telemetry_spike_moderate_threshold;
    ctx->telemetry.control_config.spike_severe_threshold = hp.telemetry_spike_severe_threshold;

    // Accumulation bug detection (Rule 20: crash on zero gradients with non-zero loss)
    ctx->telemetry.control_config.min_grad_for_nonzero_loss = hp.telemetry_min_grad_for_nonzero_loss;
    ctx->telemetry.control_config.loss_threshold_for_grad_check = hp.telemetry_loss_threshold_for_grad_check;
    ctx->telemetry.control_config.max_consecutive_zero_grad_steps = hp.telemetry_max_consecutive_zero_grad_steps;

    // Monitoring config
    ctx->telemetry.control_config.warmup_steps = hp.telemetry_warmup_steps;
    ctx->telemetry.control_config.baseline_stabilization_steps = hp.telemetry_baseline_stabilization_steps;
    ctx->telemetry.control_config.verbose_logging = hp.telemetry_verbose_logging;
    ctx->telemetry.control_config.fail_loud_on_accumulation_bug = hp.telemetry_fail_loud_on_accumulation_bug;
    
    ctx->logging.logger->log("Telemetry control: MONITORING ONLY (all interventions removed)");
    
    // 12. Check GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    EmitModuleInfo(ModuleId::Training, 
        std::string("GPU Memory: ") + std::to_string(free_mem / (1024*1024)) + " MB free / " + 
        std::to_string(total_mem / (1024*1024)) + " MB total", 0);
    
    // Write initial status
    ctx->logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Training,
        0, ctx->config.hyperparameters.epochs, 0, 0,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        "Phase 1 complete - ready for training"
    );
    
    // Initialize timing
    ctx->start_time = std::chrono::steady_clock::now();
    

    return ctx;  // Heap-allocated, no move
}

} // namespace GRIMText::Training
