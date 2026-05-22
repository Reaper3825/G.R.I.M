#include "TrainingData.hpp"

#include "DataInfo.hpp"
#include "../SlidingWindow.hpp"
#include "../../ConfigDump.hpp"
#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/UnigramByte/UniByte.hpp"
#include "../../../../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace Internal {

void validateStartupPaths(
    const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    const PathConfig& paths)
{
    if (!fs::exists(tokenizer_hp.vocab_path)) {
        throw std::runtime_error("Vocabulary file does not exist: " + tokenizer_hp.vocab_path);
    }
    if (!fs::exists(tokenizer_hp.data_path)) {
        throw std::runtime_error("Training data file does not exist: " + tokenizer_hp.data_path);
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

    auto model_parent = fs::path(paths.output_model_path).parent_path();
    if (!model_parent.empty()) {
        fs::create_directories(model_parent);
    }
    fs::create_directories(paths.checkpoint_dir);
    fs::create_directories(paths.log_dir);
}

std::unique_ptr<GRIM::Tokenizer::UniByte> initializeTokenizer(
    const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    TrainingLogger& logger)
{
    logger.log("Loading tokenizer artifact bundle...");

    auto tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>(tokenizer_hp);
    (void)GRIM::TokenizerArtifacts::loadTokenizerArtifactBundle(tokenizer_hp, *tokenizer);
    logger.log("Initializing tokenizer CUDA Viterbi runtime...");
    if (!tokenizer->initGPU()) {
        throw std::runtime_error("initializeTokenizer: UniByte::initGPU() returned false after artifact load");
    }

    return tokenizer;
}

SequenceData buildPhase1SequenceData(
    const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    const GRIM::HyperParameters::DataLoadingHP& data_hp,
    int max_seq_len,
    TrainingLogger& logger)
{
    if (max_seq_len <= 0) {
        throw std::runtime_error(
            "buildPhase1SequenceData: max_seq_len must be configured before sliding-window data loading (got " +
            std::to_string(max_seq_len) + ")");
    }

    SequenceData data;

    GRMTDataLoader loader;
    logger.log("[Data] Loading GRMT corpus from " + tokenizer_hp.data_path + "...");
    auto progress_logger = [&logger](const std::string& message) {
        logger.log(message);
    };
    if (!loader.load(tokenizer_hp.data_path, progress_logger)) {
        throw std::runtime_error("Failed to load training data");
    }

    data.vocab_size = loader.vocabSize();
    const auto& all_sequences = loader.getSequences();

    logger.log("[Data] Loaded raw GRMT sequences: count=" +
               std::to_string(all_sequences.size()) +
               " vocab_size=" + std::to_string(data.vocab_size));

    std::size_t val_size = all_sequences.size() / 10;
    data.train_seqs.assign(all_sequences.begin() + val_size, all_sequences.end());
    data.val_seqs.assign(all_sequences.begin(), all_sequences.begin() + val_size);
    logger.log("[Data] Train/val split ready: train_sequences=" +
               std::to_string(data.train_seqs.size()) +
               " val_sequences=" + std::to_string(data.val_seqs.size()) +
               " holdout_ratio=10%");

    logger.log("[Data] Applying sliding windows to train split...");
    applySlidingWindows(data.train_seqs, "train",
                        max_seq_len, data_hp.sliding_window_stride, data_hp.min_seq_valid_tokens,
                        tokenizer_hp.add_bos, tokenizer_hp.add_eos, logger);
    logger.log("[Data] Train split post-window sequence count=" +
               std::to_string(data.train_seqs.size()));

    logger.log("[Data] Applying sliding windows to validation split...");
    applySlidingWindows(data.val_seqs, "val",
                        max_seq_len, data_hp.sliding_window_stride, data_hp.min_seq_valid_tokens,
                        tokenizer_hp.add_bos, tokenizer_hp.add_eos, logger);
    logger.log("[Data] Validation split post-window sequence count=" +
               std::to_string(data.val_seqs.size()));

    logger.log("[Data] Materializing train/val sequence views for batching...");
    data.train_views.reserve(data.train_seqs.size());
    data.train_seq_lengths.reserve(data.train_seqs.size());
    for (std::size_t i = 0; i < data.train_seqs.size(); ++i) {
        data.train_views.push_back(&data.train_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.train_seqs[i].token_ids.size());
        data.train_seq_lengths.push_back(len);
    }

    data.val_views.reserve(data.val_seqs.size());
    data.val_seq_lengths.reserve(data.val_seqs.size());
    for (std::size_t i = 0; i < data.val_seqs.size(); ++i) {
        data.val_views.push_back(&data.val_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.val_seqs[i].token_ids.size());
        data.val_seq_lengths.push_back(len);
    }

    logger.log("[Data] Sequence views ready: train_views=" +
               std::to_string(data.train_views.size()) +
               " val_views=" + std::to_string(data.val_views.size()));

    return data;
}

} // namespace Internal

namespace {

DataInfo summarizeDataInfoOrThrow(const SequenceData& data)
{
    if (data.vocab_size == 0) {
        throw std::runtime_error("FATAL: training data missing vocab_size; regenerate GRMT with tokenizer.vocabSize()");
    }
    if (data.train_views.size() != data.train_seqs.size()) {
        throw std::runtime_error("FATAL: train view count does not match train sequence count (views=" +
                                 std::to_string(data.train_views.size()) +
                                 " seqs=" + std::to_string(data.train_seqs.size()) + ")");
    }
    if (data.val_views.size() != data.val_seqs.size()) {
        throw std::runtime_error("FATAL: val view count does not match val sequence count (views=" +
                                 std::to_string(data.val_views.size()) +
                                 " seqs=" + std::to_string(data.val_seqs.size()) + ")");
    }

    DataInfo info;
    info.actual_vocab_size = data.vocab_size;
    info.train_sequence_count = data.train_seqs.size();
    info.val_sequence_count = data.val_seqs.size();
    return info;
}

} // namespace

void LoadTrainingData(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(ctx.config);
    const auto data_hp = GRIM::HyperParameters::dataLoadingHP(ctx.config);
    const int max_seq_len = ctx.config.max_seq_len;

    EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
    Internal::validateStartupPaths(tokenizer_hp, ctx.config.paths);
    EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

    ctx.tokenizer = Internal::initializeTokenizer(
        tokenizer_hp, *ctx.logging.logger);

    ctx.data = Internal::buildPhase1SequenceData(
        tokenizer_hp,
        data_hp,
        max_seq_len,
        *ctx.logging.logger);

    ctx.data_info = summarizeDataInfoOrThrow(ctx.data);
    if (ctx.data_info.actual_vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error("FATAL: training data vocab_size must include special+byte+atom ranges (>= " +
            std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) + ")");
    }

    GRIM::HyperParameters::DerivationContext hp_ctx;
    hp_ctx.train_sequence_count = static_cast<int>(ctx.data.train_seqs.size());
    hp_ctx.validation_interval = ctx.config.hyperparameters.validation_interval;
    ctx.derived_schedule = GRIM::HyperParameters::computeDerivedSchedule(
        ctx.config.hyperparameters, hp_ctx);
    auto hp_logger = [&](const std::string& msg) { ctx.logging.logger->log(msg); };
    GRIM::HyperParameters::applyTrainingHyperparameterPolicy(
        ctx.config.hyperparameters, ctx.derived_schedule, hp_ctx, hp_logger);

    GRIMText::Training::DataStatsSnapshot data_stats;
    data_stats.data_path = tokenizer_hp.data_path;
    data_stats.vocab_path = tokenizer_hp.vocab_path;
    data_stats.actual_vocab_size = ctx.data_info.actual_vocab_size;
    data_stats.train_sequence_count = ctx.data_info.train_sequence_count;
    data_stats.val_sequence_count = ctx.data_info.val_sequence_count;
    data_stats.memory_device = ctx.memory_snapshot.device;
    data_stats.memory_device_name = ctx.memory_snapshot.device_name;
    data_stats.memory_total_bytes = ctx.memory_snapshot.total_bytes;
    data_stats.memory_free_bytes = ctx.memory_snapshot.free_bytes;

    dumpAllHyperparameters(
        ctx.config.hyperparameters,
        &ctx.derived_schedule,
        &data_stats,
        [](const std::string& msg) { GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Training, msg, 0); });
}

} // namespace GRIMText::Training